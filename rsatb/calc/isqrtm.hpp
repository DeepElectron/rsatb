#ifndef RSATB_CALC_ISQRTM_HPP
#define RSATB_CALC_ISQRTM_HPP

// S^{-1/2} of a sparse LCAO overlap, distributed, at controlled sparsity.
//
// The algorithm has two stages:
//
//   1. SEED (graph_inverse_sqrt). Each block row of S^{-1/2} is taken from
//      the dense inverse square root of that row's neighbourhood subgraph --
//      one thin "action" per row through vbcsr's generic graph_function_apply
//      driver, dense kernels in isqrt_action.hpp. The subgraph is S's own row
//      pattern, so the seed's cost per row and its accuracy are set by the
//      overlap's interaction radius, not by system size.
//
//   2. NEWTON-SCHULZ (inverse_sqrt). The seed Z is refined by the cubic
//      purification
//
//          X  = Z^H S Z                          (congruence residual)
//          P  = 1.875 I - 1.25 X + 0.375 X^2     (cubic update polynomial)
//          Z <- Z P
//
//      which drives X -> I with cubic order while Z -> S^{-1/2}. Every
//      product is block-sparse under a Frobenius drop threshold (sptol), so
//      the achievable residual is floored by sptol; convergence is judged on
//      the max block Frobenius norm of X - I.
//
//      X can be advanced two ways -- reformed exactly as Z^H S Z, or
//      propagated as P X P -- whose costs move in opposite directions across
//      the run. A per-step cost model chooses; see the route section below.
//
// Layering: the generic graph traversal and all sparse kernels live in
// vbcsr; everything specific to the inverse square root -- the dense action
// kernel, the routing, the purification -- is rsatb's.
//
// The design constraint throughout is PEAK MEMORY at fixed accuracy: fused
// upper-triangle kernels, in-place completions, consuming products, and
// operands released the moment they are dead. Measurements, history, and the
// full rationale live in doc/design_isqrtm_memory.md; comments here state
// the invariants the code relies on.

#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "vbcsr/core/detail/ops/spmf/graph_function.hpp"
#include "isqrt_action.hpp"

// Thread policy: see vbcsr's dense_kernels.hpp. The batched graph loop holds a
// ScopedSerialBLAS, so the dense kernels here run single-threaded while the
// outer OpenMP loop owns the budget.

namespace rsatb {
namespace calc {

using vbcsr::BlockSpMat;
using vbcsr::DistGraph;

// |z|^2, for the residual metrics below. Defined locally rather than
// borrowing vbcsr::detail's equivalent, which lives in an SpGEMM-internal
// header this file has no business including.
template <typename T>
inline double squared_magnitude(const T& v) {
    if constexpr (std::is_same<T, std::complex<double>>::value ||
                  std::is_same<T, std::complex<float>>::value) {
        return std::norm(v);
    } else {
        return static_cast<double>(v) * static_cast<double>(v);
    }
}

// Where the per-subgraph work went, aggregated over one graph_inverse_sqrt call.
struct GraphInverseSqrtReport {
    long long krylov = 0;      // problems solved by the inverse block Lanczos route
    long long band = 0;        // ... by the dense-to-band + Zolotarev route
    long long eigen = 0;       // ... by the clamped eigendecomposition
    long long failed = 0;
    double mean_krylov_blocks = 0.0;
};

namespace detail {

// Thread-safe accumulator behind GraphInverseSqrtReport.
struct IsqrtTally {
    std::atomic<long long> krylov{0};
    std::atomic<long long> band{0};
    std::atomic<long long> eigen{0};
    std::atomic<long long> failed{0};
    std::atomic<long long> krylov_blocks{0};

    void record(const isqrt::Stats& s) {
        if (s.status != isqrt::Status::Success) {
            failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        switch (s.used_method) {
            case isqrt::Method::InverseBlockLanczos:
                krylov.fetch_add(1, std::memory_order_relaxed);
                krylov_blocks.fetch_add(s.krylov_blocks, std::memory_order_relaxed);
                break;
            case isqrt::Method::BandZolotarev:
                band.fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                eigen.fetch_add(1, std::memory_order_relaxed);
                break;
        }
    }
};

// RSATB_ISQRT_INIT=auto|krylov|band|eigen overrides the requested route
// without a rebuild; "eigen" reproduces the plain per-subgraph diagonalization.
inline isqrt::Method resolve_isqrt_method(isqrt::Method requested) {
    const char* env = std::getenv("RSATB_ISQRT_INIT");
    if (env == nullptr) return requested;
    const std::string value(env);
    if (value == "auto") return isqrt::Method::Auto;
    if (value == "krylov") return isqrt::Method::InverseBlockLanczos;
    if (value == "band") return isqrt::Method::BandZolotarev;
    if (value == "eigen") return isqrt::Method::Eigen;
    return requested;
}

} // namespace detail

// Out <- S^{-1/2}, one block row per block row of S, each from the dense
// subgraph of that row's neighbourhood.
//
// Only one block row of every subgraph's inverse square root is used, so the
// dense problems go to the thin action kernel (isqrt_action.hpp) rather than to
// a full diagonalization: Cholesky + inverse block Lanczos where its convergence
// certificate holds, the dense-to-band Zolotarev action otherwise, and the clamped
// eigendecomposition for tiny or numerically singular subgraphs. The kernel
// consumes the dense matrix as its working buffer, so a problem needs no second
// n x n allocation - which is what lets several run per rank concurrently.
//
// Throws if any subgraph action failed (collectively agreed, after the loop).
template <typename T>
void graph_inverse_sqrt(BlockSpMat<T>& S, BlockSpMat<T>* out,
                        const isqrt::Options& kernel = {},
                        bool verbose = false,
                        GraphInverseSqrtReport* report = nullptr) {
    // opt.method is the route *requested*, not the route taken: the
    // per-problem Router still falls back when the fast path's convergence
    // certificate fails, a subgraph turns out numerically singular, or the
    // problem is tiny. One call therefore spreads over several routes, and
    // the tally below is the only record of where the work actually went.
    // (RSATB_ISQRT_INIT and benchmark_isqrt_diamond's --route force a single
    // route for isolated timing.)
    isqrt::Options opt = kernel;
    opt.method = detail::resolve_isqrt_method(kernel.method);
    // The dense view handed to the action is row-major, so its column-major
    // upper triangle is the one the generic diagonalization reads too.
    opt.uplo = 'U';

    if (opt.method == isqrt::Method::Eigen) {
        // Plain diagonalization: the generic driver already does exactly this.
        auto func = [](double x) { return (x > 1e-12) ? std::pow(x, -0.5) : 0.0; };
        vbcsr::graph_matrix_function(S, out, std::function<T(double)>(func), verbose);
        if (report) *report = GraphInverseSqrtReport{};
        return;
    }

    isqrt::Router router;
    detail::IsqrtTally tally;

    vbcsr::graph_function_apply(
        S, out,
        [&](int total_dim, std::vector<T>& M, int k_cols, int col_start) {
            isqrt::Stats stats;
            isqrt::inverse_sqrt_columns_inplace<T>(total_dim, M, k_cols, col_start, opt, &router,
                                                   &stats);
            tally.record(stats);
        },
        verbose);

    // Unconditionally collective: every rank must agree on the failure count
    // before anyone decides to report or throw.
    const long long krylov = tally.krylov.load();
    long long counts[4] = {krylov, tally.band.load(), tally.eigen.load(),
                           tally.failed.load()};
    long long totals[4] = {0, 0, 0, 0};
    MPI_Comm comm = S.graph->comm;
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF) {
        MPI_Allreduce(counts, totals, 4, MPI_LONG_LONG, MPI_SUM, comm);
    } else {
        std::copy(counts, counts + 4, totals);
    }

    GraphInverseSqrtReport summary;
    summary.krylov = totals[0];
    summary.band = totals[1];
    summary.eigen = totals[2];
    summary.failed = totals[3];
    // Global over global, like every other field in the report: a local
    // numerator over a local denominator would describe one rank while being
    // labelled as if it described the run.
    long long krylov_blocks_local = tally.krylov_blocks.load();
    long long krylov_blocks_total = krylov_blocks_local;
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF) {
        MPI_Allreduce(&krylov_blocks_local, &krylov_blocks_total, 1, MPI_LONG_LONG,
                      MPI_SUM, comm);
    }
    if (summary.krylov > 0) {
        summary.mean_krylov_blocks = static_cast<double>(krylov_blocks_total) /
                                     static_cast<double>(summary.krylov);
    }
    if (report) *report = summary;

    if (S.graph->rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
        std::cout << "graph_inverse_sqrt: subgraph actions - krylov " << summary.krylov
                  << ", band " << summary.band
                  << ", eigen " << summary.eigen << ", failed " << summary.failed;
        if (krylov > 0) std::cout << " (mean krylov blocks " << summary.mean_krylov_blocks << ")";
        std::cout << "\n";
    }
    if (summary.failed > 0) {
        throw std::runtime_error("graph_inverse_sqrt: " + std::to_string(summary.failed) +
                                 " subgraph inverse-square-root actions failed");
    }
}

struct InverseSqrtOptions {
    double tol = 1e-7;        // target for max block ||(Z^H S Z - I)[i,j]||_F
    double sptol = 1e-6;      // block-sparsity filter of the SpGEMM products
    int maxit = 20;           // 0 returns the seed itself
    isqrt::Options kernel{};  // dense per-subgraph kernel of the seed
};

// What the iteration did, for a caller (or a test) that needs to act on it
// rather than read it off a log line. Same optional-out-param shape as
// GraphInverseSqrtReport.
struct InverseSqrtReport {
    int steps = 0;             // Newton-Schulz steps taken
    bool converged = false;    // residual reached tol
    bool hit_maxiter = false;  // stopped on the step count
    // Whether the last X came from P X P rather than a reform, which is what
    // says whether `residual` below is exact or propagated.
    bool last_advance_propagated = false;
    double residual = 0.0;          // loop metric: max block Frobenius of X - I
    double applied_residual = 0.0;  // ||(Z^H S Z - I) v|| on a unit random probe
};

// out <- S^{-1/2}: a seed (by default the subgraph matrix function above)
// refined by Newton-Schulz iterations on the sparse matrix.
template <typename T>
void inverse_sqrt(BlockSpMat<T>& S_mat, BlockSpMat<T>* S_inv_sqrt,
                  const InverseSqrtOptions& options = {},
                  bool verbose = false,
                  InverseSqrtReport* report = nullptr) {
    DistGraph* graph = S_mat.graph;
    const int rank = graph->rank;
    const double tol = options.tol;
    const double sptol = options.sptol;
    const int maxit = options.maxit;

    // The subgraph matrix function is the only seed.
    const double t_seed0 = MPI_Wtime();
    {
        RSATB_PHASE("setup/isqrt_seed");
        isqrt::Options kernel = options.kernel;
        // The seed only has to be as accurate as the iteration's target.
        kernel.relative_tolerance = (tol > 0.0)
                                        ? std::min(kernel.relative_tolerance, tol)
                                        : kernel.relative_tolerance;
        graph_inverse_sqrt(S_mat, S_inv_sqrt, kernel, verbose);
    }

    const double t_seed = MPI_Wtime() - t_seed0;
    if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
        // std::endl, not "\n": cluster stdout is block-buffered, and this
        // line is the only signal between the seed and the first (possibly
        // hour-scale) Newton-Schulz product.
        std::cout << "inverse_sqrt: seed " << t_seed << "s" << std::endl;
    }

    if (maxit == 0) return;

    // Newton-Schulz on X = Z^H S Z: Z <- Z (1.875 I - 1.25 X + 0.375 X^2),
    // the order-3 iteration for X -> I. Three commitments, each the measured
    // winner over its alternative (doc/design_isqrtm_memory.md §1):
    //
    //  - CUBIC, not quadratic. The seed's spectral spread is wider than its
    //    max-block residual suggests, and the quadratic map contracts slowly
    //    where X's lowest eigenvalue sits; the cubic's wider contraction
    //    basin wins at roughly equal flops.
    //
    //  - BOTH ROUTES for advancing X, chosen per step by predicted cost.
    //    Which route is cheaper INVERTS across the run: reform (Z^H S Z)
    //    scales with Z, which fills in and stays filled; propagation (P X P)
    //    scales with P and X, which collapse toward the identity until the
    //    threshold drops nearly all of their blocks. Propagation also drifts
    //    (see the truncation note at the end of the loop), which the
    //    periodic exact reformation bounds -- that period is a stability
    //    bound and cost never overrides it. Known limitation: the model
    //    ranks TIME, not memory, and a reform peaks higher than the fused
    //    propagation.
    //
    //  - The congruence must stay the CONJUGATE transpose. Z^T S Z is
    //    identical at Gamma but diverges instantly at complex k -- the seed
    //    approximates the Hermitian S^-1/2, which violates the unconjugated
    //    congruence by O(1) -- and Z S Z (no transpose) diverges from any
    //    truncated seed. Measurements: calc/test/experiment_ns_variants.cpp.
    const double t_ns0 = MPI_Wtime();
    RSATB_PHASE("setup/isqrt_newton_schulz");
    // tmp1 holds P, tmp2 holds X, and both are declared EMPTY, not
    // graph-shaped: the first product assigns each wholesale, so shaping them
    // here would hold the loop's two largest matrices before either is
    // written.
    BlockSpMat<T> tmp1, tmp2;

    // How good Z is where it is actually used: applied to a vector.
    //
    //     ||(Z^H S Z - I) v||_2,   v a fixed random unit vector
    //
    // Three sparse matrix-vector products, WITHOUT forming Z^H S Z --
    // reforming the product purely to report would re-run the largest phase
    // of the solve and its peak (measured in the doc, §1).
    //
    // A probe, not a norm, and it does not claim to be one. The
    // preconditioner is applied to vectors, so the drift of Z^H S Z v from v
    // IS the error a caller sees; one probe can miss the worst direction,
    // which is the honest price of three products. Cheap enough to run
    // unconditionally, so every run reports the same quantity whichever
    // route the last step took.
    auto applied_residual = [&]() -> double {
        vbcsr::DistMultiVector<T> v(graph, 1), w(graph, 1), u(graph, 1), y(graph, 1);
        // vbcsr's sampler is decomposition-invariant, so the reported number
        // does not move with the rank count or the partition; it normalises
        // too, so ||v|| == 1 and the quotient is free. No zero_padding()
        // calls are needed: every writer here is lane-bounded, and vbcsr
        // asserts the pad invariant on both sides of each apply in a debug
        // build (test_isqrtm runs one).
        v.set_random(/*seed=*/1);
        S_inv_sqrt->mult_dense(v, w);          // w = Z v
        S_mat.mult_dense(w, u);                // u = S w
        S_inv_sqrt->mult_dense_adjoint(u, y);  // y = Z^H S Z v
        y.axpy(T(-1.0), v);                    // y -= v
        return std::sqrt(std::abs(y.bdot(y)[0]));
    };

    // The loop metric: max over blocks of ||(X - I)[i,j]||_F, without
    // mutating X.
    //
    // Two properties are load-bearing. Every local row must contribute
    // whether or not it stores a diagonal block -- a row whose diagonal was
    // dropped by the threshold is a block of X - I equal to exactly -I, and
    // skipping it turns an invalid X into a false "converged". And the
    // metric is the max block FROBENIUS norm, not max element magnitude: an
    // upper bound on the largest element, so the convergence test is
    // conservative (stops later, never earlier). It does NOT bound
    // ||I - X||_2, so it is not a convergence-basin certificate, and it is
    // block-size dependent -- which is why it is not called "maxabs".
    auto residual_norm = [&](const BlockSpMat<T>& X) {
        RSATB_PHASE("setup/isqrt_ns_residual");
        const DistGraph& gx = *X.graph;
        const int n_rows = static_cast<int>(gx.adj_ptr.size()) - 1;
        double local = 0.0;
        #pragma omp parallel for reduction(max : local) schedule(static)
        for (int i = 0; i < n_rows; ++i) {
            const int g_row = gx.get_global_index(i);
            const int r_dim = gx.block_sizes[i];
            bool saw_diagonal = false;
            for (int k = gx.adj_ptr[i]; k < gx.adj_ptr[i + 1]; ++k) {
                const int c_dim = gx.block_sizes[gx.adj_ind[k]];
                const bool is_diagonal = (gx.get_global_index(gx.adj_ind[k]) == g_row);
                const T* block = X.block_data(k);
                const size_t count = X.block_size_elements(k);
                double sq = 0.0;
                for (size_t e = 0; e < count; ++e) {
                    // I is subtracted here rather than in the matrix, so X is
                    // untouched and the scan stays thread safe.
                    T value = block[e];
                    if (is_diagonal) {
                        const size_t row_in_block = e / static_cast<size_t>(c_dim);
                        const size_t col_in_block = e % static_cast<size_t>(c_dim);
                        if (row_in_block == col_in_block) value -= T(1.0);
                    }
                    sq += squared_magnitude<T>(value);
                }
                if (is_diagonal) saw_diagonal = true;
                local = std::max(local, std::sqrt(sq));
            }
            // No stored diagonal: that block of X - I is exactly -I.
            if (!saw_diagonal) {
                local = std::max(local, std::sqrt(static_cast<double>(r_dim)));
            }
        }
        double global = local;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        return global;
    };

    // Mean block-neighbours per row: density * N. This is the unit the cost
    // model is linear in, and the unit worth logging -- "118 nbr/row" says
    // what a density of 6.6e-4 at 2e5 rows does not.
    //
    // get_block_density() is a collective: called on every rank, never inside
    // a rank-0 log guard.
    auto nbr_per_row = [&](const BlockSpMat<T>& m) {
        const double density = m.get_block_density();
        const long long n_blocks = graph->block_displs.empty()
                                       ? 0
                                       : graph->block_displs.back();
        return density * static_cast<double>(n_blocks);
    };

    // ---- route cost model -------------------------------------------------
    //
    // Which route updates X is a COST decision, ranked in the unit the SpGEMM
    // is linear in: block-pair multiplies per row, which for A*B is exactly
    // nbr(A) * nbr(B). The only quantity not measured is the intermediate's
    // neighbour count, and the SAME estimator serves both routes, so its
    // error largely cancels in the comparison rather than biasing one side.
    // (A residual-keyed guard cannot do this job: with a good seed the
    // residual is below any threshold from step 1, exactly when the routes'
    // costs differ most.)
    //
    // S's own count is fixed for the whole iteration. Collective, so it is
    // taken here on every rank rather than under a log guard.
    const double s_nbr = nbr_per_row(S_mat);

    // Block-sparsity radii add under a product, and a radius-r row holds ~r^d
    // neighbours, so nbr(AB)^(1/d) = nbr(A)^(1/d) + nbr(B)^(1/d). d is NOT
    // hardcoded to the lattice dimension -- it is solved for from the one
    // product whose operands and result are all measured for free, the fused
    // reform Z^H S Z (radii add twice there: nbr(X)^(1/d) = s^(1/d) +
    // 2 z^(1/d)), so a 2D moire and a 3D crystal each calibrate themselves.
    // The measured counts are post-threshold, so the model carries sptol's
    // bias consistently on both sides of the comparison.
    double fill_dim = 3.0;  // until the first reform measures one
    auto calibrate_fill_dim = [](double s, double z, double x, double fallback) {
        // f(d) = s^(1/d) + 2 z^(1/d) - x^(1/d) tends to +2 as d grows (every
        // term tends to 1) and is negative at small d whenever the product is
        // wide enough, so a root is bracketed in every non-degenerate case.
        // Degenerate ones keep the previous calibration.
        if (!(s > 1.0) || !(z > 1.0) || !(x > std::max(s, z))) return fallback;
        auto f = [&](double d) {
            return std::pow(s, 1.0 / d) + 2.0 * std::pow(z, 1.0 / d) -
                   std::pow(x, 1.0 / d);
        };
        double lo = 0.5, hi = 8.0;
        if (f(lo) > 0.0 || f(hi) < 0.0) return fallback;
        for (int i = 0; i < 40; ++i) {
            const double mid = 0.5 * (lo + hi);
            if (f(mid) < 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    };
    // A row this full is at the structural limit, not at a measured value:
    // nothing can be learned from how a saturated product filled in, and
    // nothing can be gained by keeping a saturated operand sparse. Used by the
    // fill-dimension calibration (which skips) and by the route choice (which
    // stops paying to contract).
    constexpr double kSaturatedFraction = 0.98;
    // How much more than the reform's cost a contracting propagation is worth.
    // One step that halves X more than repays a single step at twice the price;
    // the factor bounds how far that argument is allowed to run before cost
    // wins outright.
    constexpr double kTrajectoryPremium = 2.0;
    // Capped at the row length: radii add without bound, neighbour counts do
    // not, and a small cell saturates. Uncapped, a dense system was predicted
    // to fill in past full -- which biased the two routes unequally, since
    // reform's intermediate has the sparse S in it and propagation's has two
    // saturated factors.
    const double row_length = graph->block_displs.empty()
                                  ? 1.0
                                  : static_cast<double>(graph->block_displs.back());
    auto nbr_product = [&](double a, double b) {
        const double e = 1.0 / fill_dim;
        return std::min(row_length, std::pow(std::pow(a, e) + std::pow(b, e), fill_dim));
    };

    // Both triple products form only the upper triangle -- the mirror is a copy,
    // not a product -- hence the halves on the second product of each.
    auto reform_units = [&](double z) { // ZH S Z
        return s_nbr * z + 0.5 * z * nbr_product(s_nbr, z);
    };
    auto propagate_units = [&](double p, double x) { // P X P
        return p * x + 0.5 * nbr_product(p, x) * p;
    };

    // Block pairs are the unit, but the two routes do NOT cost the same per
    // pair -- the constants are properties of the machine and the block size,
    // not of the algorithm -- so each route carries its own measured
    // seconds-per-unit and the comparison happens in SECONDS.
    //
    // Only the LARGEST measurement per route is kept, not the latest and not
    // an average: a millisecond-scale step is dominated by fixed setup
    // (result graph, collectives) rather than by its block pairs, and
    // fitting a constant to it would describe the overhead instead of the
    // work. The large steps are also the only ones where the decision
    // changes anything. Until a route has been measured it borrows the
    // other's constant -- a pure count comparison, self-correcting after one
    // step.
    double reform_units_seen = 0.0, reform_seconds_seen = 0.0;
    double propagate_units_seen = 0.0, propagate_seconds_seen = 0.0;
    // Below this a step is setup, not work: four orders below any
    // decision-relevant step and comfortably above setup noise. Learning
    // from a setup-dominated step would bias routing away from whichever
    // route happened to run it.
    const double kCalibrationFloorSeconds = 0.1;
    // COLLECTIVE, called at the same point on every rank, and the duration is
    // MAX-reduced first: a route is a collective sequence, so its real cost
    // is the slowest rank's -- and identical constants everywhere are what
    // keep the per-rank predictions from drifting apart. (The broadcast at
    // the decision point is the hard guarantee; this is what keeps it from
    // ever being needed.)
    auto note_route_cost = [&](bool propagated, double units, double local_seconds) {
        double seconds = local_seconds;
        if (graph->size > 1) {
            MPI_Allreduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        }
        if (!(units > 0.0) || !(seconds > kCalibrationFloorSeconds)) return;
        if (propagated) {
            if (units > propagate_units_seen) {
                propagate_units_seen = units;
                propagate_seconds_seen = seconds;
            }
        } else if (units > reform_units_seen) {
            reform_units_seen = units;
            reform_seconds_seen = seconds;
        }
    };
    auto seconds_per_unit = [&](bool propagated) {
        const double units = propagated ? propagate_units_seen : reform_units_seen;
        const double seconds = propagated ? propagate_seconds_seen : reform_seconds_seen;
        if (units > 0.0 && seconds > 0.0) return seconds / units;
        // Fall back to whatever the other route measured, so the comparison
        // stays a pure count ratio rather than mixing a calibrated side with an
        // uncalibrated one.
        const double other_units = propagated ? reform_units_seen : propagate_units_seen;
        const double other_secs = propagated ? reform_seconds_seen : propagate_seconds_seen;
        return (other_units > 0.0 && other_secs > 0.0) ? other_secs / other_units : 1.0;
    };

    // Stage heartbeats: a distributed product can run for hours with low CPU
    // during its fetch/symbolic phases, and a log that goes quiet there is
    // indistinguishable from a hang. One flushed line per stage bounds the
    // search to a single product. Each product also opens its own profile
    // phase -- the heartbeat says where a stalled run IS, the phase table
    // says where a finished run WENT, and the products' cost shapes differ
    // enough that neither can stand in for the other.
    auto stage = [&](const char* what) {
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            std::cout << "inverse_sqrt:   " << what << std::endl;
        }
    };

    // X = Z^H S Z, one fused kernel (rhar_upper). X comes out exactly
    // Hermitian by construction -- Z^H S Z is Hermitian for ANY Z, which is
    // what everything downstream relies on and why the reform conjugates Z
    // rather than trusting its approximate hermiticity.
    //
    // rhar_upper computes the triple product without materialising either
    // Z-sized intermediate (S Z, or a transposed copy of Z^H): local Z blocks
    // are read in place through a transpose index, only partition-boundary
    // blocks travel (pushed, already conjugated), one answer row lives per
    // thread, and the drop threshold applies once, to finished values. Z is
    // the densest matrix in the loop, so this is what keeps the reform's
    // peak near the S+Z+X floor.
    //
    // Takes Z's neighbour count rather than measuring it: every caller has
    // just measured it, and get_block_density() is a collective.
    auto reform_x = [&](double z_nbr) {
        RSATB_PHASE("setup/isqrt_ns_ZH_S_Z");
        stage("Z^H S Z (one fused kernel)");
        // X is dead: the kernel does not read it. Released before the result
        // allocates rather than at the assignment afterwards.
        tmp2.reset();
        BlockSpMat<T> upper = S_inv_sqrt->rhar_upper(S_mat, sptol);
        upper.complete_hermitian_inplace();
        tmp2 = std::move(upper);
        // Free calibration point for fill_dim: the reform's own result obeys
        // nbr(Z^H S Z)^(1/d) = s^(1/d) + 2 z^(1/d) under the additive-radius
        // model, and X has just been measured. Skipped when the product
        // saturated: a count clipped at the full row is a FLOOR, not a
        // measurement, and solving with it biases d low.
        const double x_nbr_reformed = nbr_per_row(tmp2);
        if (x_nbr_reformed < kSaturatedFraction * row_length) {
            fill_dim = calibrate_fill_dim(s_nbr, z_nbr, x_nbr_reformed, fill_dim);
        }
    };

    // The seed's reform is a free calibration point for the reform constant:
    // it runs unconditionally, and it is the same fused kernel a scheduled
    // reform would run.
    double z_nbr = nbr_per_row(*S_inv_sqrt);
    const double t0_seed_reform = util::wall_time();
    reform_x(z_nbr);
    note_route_cost(/*propagated=*/false, reform_units(z_nbr),
                    util::wall_time() - t0_seed_reform);
    double norm = residual_norm(tmp2);
    double norm_pre = 0.0;
    int count = 0;
    // Whether the last X in hand came from P X P rather than a reform.
    bool last_advance_propagated = false;

    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        util::throw_if_interrupted(graph->comm);
        const double step_t0 = util::wall_time();
        // Collective (Allreduce): must run on EVERY rank, outside the rank-0
        // log guard, or the ranks' collective sequences diverge.
        const double x_nbr = nbr_per_row(tmp2);
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            // Memory on the step line, not only in the end-of-setup table: a
            // run killed inside this loop never reaches the table, and then
            // the one thing worth knowing -- how close it came -- is lost.
            const double rss_gb = util::read_rss_kb() / 1048576.0;
            const double peak_gb =
                util::MemorySampler::instance().global_peak_kb() / 1048576.0;
            std::cout << "inverse_sqrt: step " << count + 1 << ": max block residual " << norm
                      << ", Z " << z_nbr << " nbr/row, X " << x_nbr << " nbr/row"
                      << ", rss " << rss_gb << " GB (peak " << peak_gb << " GB)" << std::endl;
        }
        if (norm > 100.0) {
            if (rank == 0 && util::fault()) {
                std::cerr << "inverse_sqrt: diverged, residual " << norm << "\n";
            }
            break;
        }
        norm_pre = norm;

        // P = 1.875 I - 1.25 X + 0.375 X^2, then Z <- Z P. X is exactly
        // Hermitian (every route that forms it mirrors), X^2 of a Hermitian
        // matrix is Hermitian, and the polynomial's coefficients are real --
        // so P is exactly Hermitian too, which the propagation below uses.
        const double t0_x2 = util::wall_time();
        {
            RSATB_PHASE("setup/isqrt_ns_polynomial");
            stage("P = 1.875 I - 1.25 X + 0.375 X^2 (one fused kernel)");
            // One fused kernel, not a square followed by an axpby: X^2 is
            // DENSER than the P it becomes (squaring doubles the sparsity
            // radius while the polynomial collapses toward the identity), so
            // materialising it was the loop's largest transient.
            // square_polynomial_upper holds one row of the square per thread
            // and drops once, on the finished P. Same accuracy ORDER as the
            // two-call route, not an improvement -- the two drop different
            // blocks; see square_polynomial.hpp.
            //
            // The previous step's P is dead; X is the operand, so only the
            // destination can be released here.
            tmp1.reset();
            BlockSpMat<T> upper =
                tmp2.square_polynomial_upper(T(0.375), T(-1.25), T(1.875), sptol);
            upper.complete_hermitian_inplace();
            tmp1 = std::move(upper);
        }
        const double t_x2 = util::wall_time() - t0_x2;

        // P's neighbour count decides the Z*P cost together with Z's, decides
        // how much of P every rank has to fetch, and is one of the two inputs
        // to the route decision below. Collective, so outside the log guard.
        const double p_nbr = nbr_per_row(tmp1);

        const double t0_zp = util::wall_time();
        {
            RSATB_PHASE("setup/isqrt_ns_Z_times_P");
            if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
                std::cout << "inverse_sqrt:   Z * P (P " << p_nbr << " nbr/row)"
                          << std::endl;
            } else {
                stage("Z * P");
            }
            // In place: the ordinary product would hold the old Z and the new
            // one whole at the same moment, for a matrix that only ever needs
            // to exist once. spmm_inplace consumes Z row block by row block.
            // Nothing else references it: S_inv_sqrt is the caller's output
            // and P is a separate matrix.
            S_inv_sqrt->spmm_inplace(tmp1, sptol);
        }
        const double t_zp = util::wall_time() - t0_zp;

        // Z has just changed, and it is the NEW Z a reform would multiply, so
        // that is the count the model gets. Also carried to the next step's
        // heartbeat, which is why it outlives this scope.
        z_nbr = nbr_per_row(*S_inv_sqrt);
        const double units_reform = reform_units(z_nbr);
        const double units_propagate = propagate_units(p_nbr, x_nbr);
        // Predicted SECONDS, not block pairs: each route's own measured
        // constant is applied. The constants and fill_dim are CAPTURED here,
        // not re-read at print time -- note_route_cost and reform_x update
        // them below, and the log line must multiply out from the values the
        // decision actually used.
        const double spu_reform = seconds_per_unit(false);
        const double spu_propagate = seconds_per_unit(true);
        const double fill_dim_used = fill_dim;
        const double cost_reform = units_reform * spu_reform;
        const double cost_propagate = units_propagate * spu_propagate;
        // The every-fifth-step exact reformation is a DRIFT bound, not a cost
        // choice: P X P accumulates truncation that only Z^H S Z clears, and
        // the period was fixed by measurement. Cost never overrides it.
        const bool drift_refresh = ((count + 1) % 5 == 0);
        // The residual guard handles stability, the one job the cost model
        // cannot: a seed poor enough to leave the residual above 1e-1 is
        // where propagation has the least margin. It can only ever force MORE
        // reforms, never fewer, and on a healthy seed it never binds.
        //
        // Cost alone is not the objective, because the two routes do not
        // leave the same X behind. Propagation CONTRACTS X (P collapses
        // toward the identity under sptol); reformation re-densifies it to
        // Z's width, which every later step then pays for -- a reform can be
        // the cheapest step of the run and still the most expensive choice.
        // Whether that trade exists at all depends on whether X has room to
        // contract: saturated (a full block row), P X P is dense whatever
        // the polynomial does and cost decides alone; with headroom,
        // contraction is worth buying at up to kTrajectoryPremium times the
        // reform's price, which one contracting step repays by shrinking
        // every step after it.
        const bool x_saturated = !(x_nbr < kSaturatedFraction * row_length);
        const bool cost_favours_propagate =
            x_saturated ? (cost_propagate < cost_reform)
                        : (cost_propagate < kTrajectoryPremium * cost_reform);
        // COLLECTIVE. The two branches issue DIFFERENT collective sequences
        // (rarh_upper against reform_x's rhar_upper), so a split decision is
        // not a slightly worse route, it is a hang or a mismatched exchange.
        // Rank 0's choice is broadcast and is the only one that runs.
        const bool propagate_local =
            !drift_refresh && norm < 1e-1 && cost_favours_propagate;
        int propagate_flag = propagate_local ? 1 : 0;
        if (graph->size > 1) {
            MPI_Bcast(&propagate_flag, 1, MPI_INT, 0, graph->comm);
        }
        const bool propagate = propagate_flag != 0;

        const double t0_update = util::wall_time();
        if (propagate) {
            RSATB_PHASE("setup/isqrt_ns_P_X_P");
            stage("P X P (one fused kernel)");
            // One fused kernel, not two products: the P X intermediate is
            // denser than the answer (radius r_P + r_X against a result that
            // collapses under sptol), so rarh_upper holds one of its rows per
            // thread and drops once, on the finished result. P^H = P exactly
            // -- established at the polynomial above -- so neither half needs
            // a transpose; rarh REQUIRES that hermiticity.
            BlockSpMat<T> upper = tmp1.rarh_upper(tmp2, sptol);
            // BOTH operands die with the product: X is being replaced, and
            // P's last read was the kernel call above -- the next step
            // rebuilds it from the new X. Released before the mirror, which
            // is the phase this step peaks in.
            tmp1.reset();
            tmp2.reset();
            upper.complete_hermitian_inplace();
            tmp2 = std::move(upper);
        } else {
            // P is dead on this route -- the reform rebuilds X from Z and S.
            // Reform is chosen precisely at the early steps where P is
            // densest, so it must not be held across the fused reform.
            tmp1.reset();
            reform_x(z_nbr);
        }
        const double t_update = util::wall_time() - t0_update;
        // Feed the route that actually ran back into its own constant; there
        // is no counterfactual timing for the one that did not.
        note_route_cost(propagate, propagate ? units_propagate : units_reform, t_update);

        norm = residual_norm(tmp2);

        // Per-step times, because the profile table cannot show them: these
        // products vary by three orders of magnitude between the first step
        // and the last, and a total divided by a call count describes
        // neither end. The model's inputs and its two predictions are
        // printed next to the times they predicted, so the routing can be
        // checked against reality after a run rather than trusted.
        const double t_step = util::wall_time() - step_t0;
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            std::cout << "inverse_sqrt: step " << count + 1 << " took " << t_step
                      << "s: X^2+P " << t_x2 << "s, Z*P " << t_zp
                      << "s, " << (propagate ? "PXP " : "reform ") << t_update << "s"
                      << std::endl;
            std::cout << "inverse_sqrt:   route " << (propagate ? "propagate" : "reform")
                      << (drift_refresh ? " (periodic drift refresh)"
                                        : (x_saturated ? " (X saturated: cost only)"
                                                       : " (X has headroom: contraction priced in)"))
                      << ": S " << s_nbr << ", Z " << z_nbr << ", X " << x_nbr << ", P "
                      << p_nbr << " nbr/row; predicted reform " << cost_reform
                      << "s vs propagate " << cost_propagate << "s (units "
                      << units_reform << " / " << units_propagate << ", s per unit "
                      << spu_reform << " / " << spu_propagate
                      << ", fill dim " << fill_dim_used << ")" << std::endl;
        }
        ++count;

        last_advance_propagated = propagate;
    }

    // The honest number, once, AFTER the loop.
    //
    // A propagated X is not the residual of the Z being returned: Z is
    // filter(Z P), and whatever the threshold dropped from that product never
    // enters P X P, so the true operand is (ZP - D)^H S (ZP - D). The gap is
    // truncation, not rounding, and it is a MULTIPLE (measured 5-7x; doc §1).
    // Hence the applied-residual probe here: it measures the returned Z
    // itself, without reforming anything.
    //
    // Two boundaries are deliberate and load-bearing:
    //
    //  - The probe NEVER feeds the termination test. The exact residual is
    //    bounded below by sptol, so "still above tol" is permanently true for
    //    any tol under sptol; and the plateau detector assumes one metric, so
    //    mixing an exact norm into a propagated norm_pre disarms it. Both
    //    failure modes were hit when this was tried (doc §1): the run grinds
    //    at the sptol floor, its exact residual worsening, and never exits.
    //
    //  - The loop verdict below (hit_tol, the "max block residual" on the
    //    summary line) is still the propagated `norm`, printed beside the
    //    probe. Changing that means deciding what the loop should terminate
    //    ON -- a bigger question than reporting, deliberately left open.
    //
    // Unconditional: three SpMVs is cheap enough that gating on the route
    // would only make two runs report incomparable quantities.
    const double applied = applied_residual();
    if (report) report->applied_residual = applied;
    if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
        std::cout << "inverse_sqrt:   ||(Z^H S Z - I) v|| = " << applied
                  << " on a unit random probe (loop metric, max block Frobenius of "
                  << (last_advance_propagated ? "the PROPAGATED X" : "X") << ", "
                  << norm << ")" << std::endl;
    }

    // Say what actually happened: "converged", "stopped at maxiter" and
    // "plateaued" are three different outcomes, and dressing a plateau
    // against the sptol floor as success misleads exactly the caller who set
    // tol below sptol.
    const bool hit_tol = norm <= tol;
    const bool hit_maxiter = count >= maxit;
    if (report) {
        report->steps = count;
        report->converged = hit_tol;
        report->hit_maxiter = hit_maxiter;
        report->last_advance_propagated = last_advance_propagated;
        report->residual = norm;
    }
    // Collective: every rank, before the rank-0 guard (see the note above).
    const double final_density = S_inv_sqrt->get_block_density();

    // What the iteration cost in memory, against what the caller KEEPS: the
    // question a bigger run asks is "how much headroom over the state I end
    // up holding", and the answer is a multiple, not a number of bytes. The
    // denominator is measured rss after the temporaries are released -- the
    // steady state also holds S, every adjacency, and whatever the allocator
    // has not returned, and quoting anything tighter makes the ratio look
    // worse than the machine ever sees. X and P die HERE rather than at the
    // closing brace, so the reading is of the state that actually persists.
    tmp1.reset();
    tmp2.reset();
    const long long steady_kb_local = util::read_rss_kb();
    long long peak_kb_local = util::MemorySampler::instance().global_peak_kb();
    // MAXLOC, not two independent MAXes: the peak and the steady value must
    // come from the SAME rank -- the one that actually peaked -- or the
    // ratio describes no rank at all.
    struct { double value; int rank; } peak_in{static_cast<double>(peak_kb_local), rank},
                                       peak_out{0.0, 0};
    MPI_Allreduce(&peak_in, &peak_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, graph->comm);
    long long merged[2] = {static_cast<long long>(peak_out.value), steady_kb_local};
    MPI_Bcast(&merged[1], 1, MPI_LONG_LONG, peak_out.rank, graph->comm);

    if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
        std::cout << "inverse_sqrt: " << (hit_tol ? "converged" : (hit_maxiter ? "stopped at maxiter" : "plateaued"))
                  << " at max block residual " << norm << ", density "
                  << final_density << " (" << count << " Newton-Schulz steps, "
                  << (MPI_Wtime() - t_ns0) << "s)" << std::endl;
        const double peak_gb = static_cast<double>(merged[0]) / 1048576.0;
        const double steady_gb = static_cast<double>(merged[1]) / 1048576.0;
        std::cout << "inverse_sqrt: peak rss " << peak_gb << " GB, steady "
                  << steady_gb << " GB after release";
        if (steady_gb > 0.0) std::cout << " = " << (peak_gb / steady_gb) << "x";
        std::cout << " (worst rank)" << std::endl;
    }
    if (!hit_tol && rank == 0 && util::Log::enabled(util::LogLevel::kSummary)) {
        std::cout << "inverse_sqrt: NOTE tol " << tol << " was not reached (stopped at "
                  << norm << "). The iteration cannot converge much below sptol ("
                  << sptol << "); either raise tol to match or lower sptol."
                  << std::endl;
    }
}

} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_ISQRTM_HPP
