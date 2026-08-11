#ifndef RSATB_CALC_ISQRTM_HPP
#define RSATB_CALC_ISQRTM_HPP

// S^{-1/2} for the ISQRTM preconditioner: the graph seed (one thin dense
// inverse-square-root action per block row, through vbcsr's generic
// graph_function_apply driver) refined by Newton-Schulz purification on the
// sparse matrix. The generic graph traversal lives in vbcsr; everything
// specific to the inverse square root - the dense kernel (isqrt_action.hpp),
// the route bookkeeping and the purification - is rsatb's.

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
#include <set>
#include <stdexcept>
#include <string>
#include "vbcsr/core/detail/ops/spmf/graph_function.hpp"
#include "isqrt_action.hpp"

// Thread policy: see vbcsr's dense_kernels.hpp. The batched graph loop holds a
// ScopedSerialBLAS, so the dense kernels here run single-threaded while the
// outer OpenMP loop owns the budget.

namespace rsatb {
namespace calc {

using vbcsr::BlockSpMat;
using vbcsr::DistGraph;


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
    // opt.method is the route *requested*, not the route taken. Its default is
    // Auto, and even when it is not, the per-problem Router still falls back:
    // the fast path carries a convergence certificate and hands off when the
    // certificate fails, a subgraph can turn out numerically singular, and a
    // tiny one goes straight to the eigendecomposition. So one call typically
    // spreads over several routes and the tally below is the only record of
    // where the work actually went -- it is measurement, not a restatement of
    // this line. (Forcing a single route for isolated timing is what the
    // RSATB_ISQRT_INIT override and benchmark_isqrt_diamond's --route are for;
    // there the tally does collapse to one bucket, plus whatever still fell
    // back, which is exactly the number that benchmark wants to see.)
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
    // Global over global. Both halves used to be this rank's own -- a local
    // numerator over a local denominator -- while every other field in the
    // report was collective, so the mean described one rank and was labelled as
    // if it described the run.
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

// out <- S^{-1/2}: a seed (by default the subgraph matrix function above)
// refined by Newton-Schulz iterations on the sparse matrix.
template <typename T>
void inverse_sqrt(BlockSpMat<T>& S_mat, BlockSpMat<T>* S_inv_sqrt,
                  const InverseSqrtOptions& options = {},
                  bool verbose = false) {
    DistGraph* graph = S_mat.graph;
    const int rank = graph->rank;
    const double tol = options.tol;
    const double sptol = options.sptol;
    const int maxit = options.maxit;

    // The subgraph matrix function is the only seed.
    //
    // Two others used to be selectable, a scaled identity and a first-order
    // Taylor step. Neither is viable on a real overlap: Newton-Schulz from
    // c*I converges only while ||I - c^2 S|| < 1, so the step count runs with
    // kappa(S), and an LCAO overlap with a diffuse shell reaches kappa ~ 4e3
    // (measured on diamond). The seed here is instead accurate from the start
    // and independent of kappa. The Taylor branch was also sign-wrong -- it
    // computed 1.5 I + 0.5 S where the expansion of S^{-1/2} about I is
    // 1.5 I - 0.5 S -- which stood undetected, itself good evidence that
    // nothing production ever selected it.
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
    // the order-3 iteration for X -> I.
    //
    // Every design choice below was challenged by measurement on the real
    // problem (diamond sc888, seed residual 1.9e-2, 3 steps to 4.4e-9) and
    // the measured winner kept:
    //
    //  - Cubic, not quadratic. The seed's max-block residual understates
    //    its spectral spread: X's lowest eigenvalue sits far from 1, where
    //    the quadratic map contracts only ~2x per step and needs four steps
    //    against the cubic's three. Roughly a wash in flops; the wider
    //    contraction basin keeps it.
    //
    //  - Both routes for X, chosen per step by predicted cost. Neither wins
    //    outright, and which one is cheaper INVERTS across the iteration:
    //    reform (Z^H (S Z)) scales with Z, which fills in and stays filled,
    //    while propagation (P X P) scales with P and X, which collapse
    //    toward the identity until the threshold drops nearly all of their
    //    blocks. Measured on 4096-atom diamond at sptol 1e-4, the one cell
    //    here big enough that the fill-in stops before its own periodic
    //    image: Z runs 293 -> 2255 nbr/row and STAYS there while P runs
    //    1628 -> 552 -> 1, and the fused product that costs 1983s at step 1
    //    costs 6.4s by step 3. A 178k-atom moire cell shows the same shape
    //    (Z 118 -> 727, P 540 -> 87 -> 1.02). An earlier fixed choice of
    //    propagation, guarded only by the residual, took the wrong route at
    //    exactly the step where it was most expensive; see the cost model
    //    below. A/B on diamond sc888: Newton-Schulz 955.8s under the old
    //    always-propagate rule against 223.2s routed, 4.3x. The propagation
    //    drifts, which the periodic exact reformation bounds -- that period
    //    is a stability bound and cost never overrides it.
    //
    //    Note that sc888 (1024 atoms) SATURATES: the fill-in reaches the
    //    periodic image and every operand sits at the full row width, so the
    //    routing there is a degenerate choice and its neighbour counts teach
    //    nothing. Benchmark route behaviour on 4096 atoms or larger.
    //
    //    One caveat the model does NOT know: it ranks TIME, not memory, and
    //    on sc888 the faster routing also peaked higher, 15.9 GB against
    //    13.3 GB, because a reform materialises S Z and the completion where
    //    the fused kernel holds only a row per thread. On a system where
    //    memory is the binding constraint that trade may be the wrong one.
    //    (The per-block-pair constants the model once got wrong are now
    //    measured per route rather than assumed equal -- see below.)
    //
    //  - The congruence must stay the CONJUGATE transpose. Z^T S Z is
    //    identical at Gamma but diverges instantly at complex k -- the seed
    //    approximates the Hermitian S^-1/2, which violates the unconjugated
    //    congruence by O(1) -- and Z S Z (no transpose) diverges from any
    //    truncated seed. Measurements: calc/test/experiment_ns_variants.cpp.
    const double t_ns0 = MPI_Wtime();
    RSATB_PHASE("setup/isqrt_newton_schulz");
    // Declared empty, not graph-shaped: the first product assigns each of
    // these wholesale, so allocating the local sparsity pattern here would
    // hold the two largest matrices in the loop before either is written.
    // Measured on a 1024-atom cell: 1.5 GB, freed unread.
    BlockSpMat<T> tmp1, tmp2;

    // ||Z^H S Z - I||_2, estimated by power iteration, WITHOUT forming Z^H S Z.
    //
    // This replaces a full reform whose only purpose was to report an honest
    // number. Measured on 4096 atoms that reform cost 1389 s and took the peak
    // from 85.6 GB to 107.0 GB -- it re-ran Z^H (S Z), the largest phase in the
    // solve, and undid the memory saving the whole design exists for.
    //
    // The operator M = Z^H S Z - I is Hermitian, so power iteration on it
    // converges to |lambda|_max = ||M||_2. Each iteration is three sparse
    // matrix-vector products and costs O(nnz) time and O(N) memory: a handful
    // of vectors against tens of gigabytes of matrix.
    //
    // NOTE this is a different norm from the loop's max-block-Frobenius metric,
    // and deliberately so: the spectral norm is what a preconditioner's quality
    // actually depends on, and the max block norm does not bound it.
    auto residual_spectral_estimate = [&](int iterations) -> double {
        vbcsr::DistMultiVector<T> v(graph, 1), w(graph, 1), u(graph, 1), y(graph, 1);
        // Deterministic start, so a run is reproducible; any vector with a
        // component along the dominant eigenvector will do.
        const int rank_offset = graph->rank * 7919;
        for (int r = 0; r < v.local_rows; ++r) {
            const unsigned k = static_cast<unsigned>(r + rank_offset) * 2654435761u;
            v(r, 0) = T(static_cast<double>(k % 1000u) / 500.0 - 1.0);
        }
        auto norm_of = [&](vbcsr::DistMultiVector<T>& x) {
            double local = 0.0;
            for (int r = 0; r < x.local_rows; ++r) {
                local += vbcsr::detail::block_squared_magnitude<T>(x(r, 0));
            }
            double global = local;
            MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, graph->comm);
            return std::sqrt(global);
        };
        double scale = norm_of(v);
        if (!(scale > 0.0)) return 0.0;
        for (int r = 0; r < v.local_rows; ++r) v(r, 0) /= T(scale);

        double lambda = 0.0;
        for (int it = 0; it < iterations; ++it) {
            v.zero_padding();
            S_inv_sqrt->mult_dense(v, w);        // w  = Z v
            w.zero_padding();
            S_mat.mult_dense(w, u);              // u  = S w
            u.zero_padding();
            S_inv_sqrt->mult_dense_adjoint(u, y);  // y = Z^H u
            for (int r = 0; r < y.local_rows; ++r) y(r, 0) -= v(r, 0);
            const double nrm = norm_of(y);
            lambda = nrm;                        // ||v|| == 1
            if (!(nrm > 0.0)) break;
            for (int r = 0; r < y.local_rows; ++r) v(r, 0) = y(r, 0) / T(nrm);
        }
        return lambda;
    };

    // max over blocks of ||(X - I)[i,j]||_F, without mutating X.
    //
    // The previous form was shift(-1) / scan get_block_norms() / shift(+1), and
    // it could report ZERO for a completely wrong X. shift() only touches a
    // diagonal block that is already STORED; on a row whose diagonal was
    // dropped by the threshold -- or on an empty X -- it does nothing, so that
    // row contributed 0 instead of ||-I||_F. With a loose enough sptol that is
    // an immediate false "converged" on an invalid result. Every local row is
    // now accounted for, present diagonal or not.
    //
    // The metric is the max block FROBENIUS norm, not max element magnitude,
    // and is named accordingly. It is an upper bound on the largest element
    // (||B||_F >= max|b_ij|), so using it as the convergence test is
    // conservative -- it stops later, never earlier. It does NOT bound
    // ||I - X||_2, so it is not a convergence-basin certificate; see the design
    // note. It is also block-size dependent, which is why it is not called
    // "maxabs".
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
                    sq += vbcsr::detail::block_squared_magnitude<T>(value);
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

    // Mean block-neighbours per row: density * N. At 2e5 block rows a density
    // reads 6.6e-4 and means nothing; "118 nbr/row" is the same number in the
    // units the cost is actually linear in. Both operands of every product
    // below are reported, because the product's cost is their PRODUCT of
    // neighbour counts, and a run where X and P have filled in behaves nothing
    // like the one where they have not -- which used to be invisible, since
    // only Z's density was logged.
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
    // Which route updates X is a COST decision, and the two routes' costs move
    // in opposite directions across the iteration (see the note above). The old
    // guard keyed the choice on the residual, which cannot see that: with a
    // good seed the residual is already below the threshold at step 1, so the
    // guard never fired and the loop propagated at exactly the step where P and
    // X were densest and Z sparsest -- the most expensive propagation and the
    // cheapest reform of the whole run, and the run's peak-memory phase.
    //
    // So rank the routes by predicted cost instead, in the unit the SpGEMM is
    // linear in: block-pair multiplies per row, which for A*B is exactly
    // nbr(A) * nbr(B). The only quantity not measured is the intermediate's
    // neighbour count, and the SAME estimator serves both routes, so its error
    // largely cancels in the comparison rather than biasing one side.
    //
    // S's own count is fixed for the whole iteration. Collective, so it is
    // taken here on every rank rather than under a log guard.
    const double s_nbr = nbr_per_row(S_mat);

    // Block-sparsity radii add under a product, and a radius-r row holds ~r^d
    // neighbours, so nbr(AB)^(1/d) = nbr(A)^(1/d) + nbr(B)^(1/d). d is NOT
    // hardcoded to the lattice dimension -- it is solved for from the one
    // product whose operands and result are all measured for free, S Z inside
    // the reform, so a 2D moire and a 3D crystal each calibrate themselves.
    double fill_dim = 3.0;  // until the first reform measures one
    auto calibrate_fill_dim = [](double a, double b, double ab, double fallback) {
        // f(d) = a^(1/d) + b^(1/d) - ab^(1/d) tends to +1 as d grows (every
        // term tends to 1) and is negative at small d whenever the product is
        // wider than either operand, so a root is bracketed in every
        // non-degenerate case. Degenerate ones keep the previous calibration.
        if (!(a > 1.0) || !(b > 1.0) || !(ab > std::max(a, b))) return fallback;
        auto f = [&](double d) {
            return std::pow(a, 1.0 / d) + std::pow(b, 1.0 / d) - std::pow(ab, 1.0 / d);
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
    auto reform_units = [&](double z) {
        return s_nbr * z + 0.5 * z * nbr_product(s_nbr, z);
    };
    auto propagate_units = [&](double p, double x) {
        return p * x + 0.5 * nbr_product(p, x) * p;
    };

    // Block pairs are the unit, but the two routes do NOT cost the same per
    // pair, and comparing raw counts silently assumes they do. Measured on
    // diamond sc888 the model predicted a 1.9x edge for the reform where the
    // truth was 9x. So each route carries its own measured seconds-per-unit and
    // the comparison happens in SECONDS.
    //
    // Worth keeping even now that the two constants are close (the fused kernel
    // reaching for the same block kernels spmm uses brought propagation to
    // 4.6e-5 s/unit against the reform's 5.2e-5, i.e. slightly CHEAPER per
    // pair): the constants are properties of the machine and the block size,
    // not of the algorithm, and the whole reason this exists is that assuming
    // them equal was wrong once already.
    //
    // Only the LARGEST measurement per route is kept, not the latest and not an
    // average: a step where a route cost 4 ms is dominated by fixed setup
    // (result graph, collectives) rather than by its block pairs, and fitting a
    // constant to it would describe the overhead instead of the work. The large
    // steps are also the only ones where the decision changes anything.
    //
    // Until a route has been measured it borrows the other's constant, which is
    // exactly the equal-cost assumption this replaces -- no worse than before,
    // and self-correcting after one step.
    double reform_units_seen = 0.0, reform_seconds_seen = 0.0;
    double propagate_units_seen = 0.0, propagate_seconds_seen = 0.0;
    // Below this a step is setup, not work. Step 3 of a converged run measured
    // 4 ms of propagation over 5 predicted units, which fits a constant 18x too
    // expensive -- entirely result-graph construction and collectives. Learning
    // from it would bias the next run's routing AWAY from propagation, and so
    // away from the route that costs less memory. Four orders below any
    // decision-relevant step and comfortably above setup noise.
    const double kCalibrationFloorSeconds = 0.1;
    // COLLECTIVE, and called at the same point on every rank. The duration is
    // reduced with MAX first: a route is a collective sequence, so its real
    // cost is the slowest rank's, and leaving each rank to learn its own local
    // time gave every rank a slightly different constant -- which is what let
    // the route choice itself diverge. Reducing here keeps the constants (and
    // so the predictions) identical everywhere; the broadcast at the decision
    // is the guarantee, this is what stops them drifting apart in the first
    // place.
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

    // Stage heartbeats: each distributed product below can run for hours at
    // large scale with low CPU during its fetch/symbolic phases, and a log
    // that goes quiet there is indistinguishable from a hang. One flushed
    // line per stage bounds the search to a single product.
    //
    // Each product also opens its own profile phase. The heartbeat says where
    // a stalled run is now; the phase table says where a finished run spent
    // its time, accumulated over steps with a call count -- and, on more than
    // one rank, the max/mean gap per product, which separates a genuinely
    // expensive product from one rank straggling on it. A single
    // setup/isqrt_newton_schulz total could not answer either question: the
    // six products have completely different cost shapes (S is the sparsest
    // matrix in the loop, Z the densest and permanently so, P and X collapse
    // toward the identity as the iteration converges), so which one dominates
    // changes with the system and with the step.
    auto stage = [&](const char* what) {
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            std::cout << "inverse_sqrt:   " << what << std::endl;
        }
    };

    // X = Z^H (S Z): grouped so the first product is the sparse one, and the
    // second computed as a known-Hermitian product -- half the block products,
    // and X comes out exactly Hermitian, which everything below relies on.
    //
    // Takes Z's neighbour count rather than measuring it: every caller has just
    // measured it, and get_block_density() is a collective, so re-taking it here
    // would add a round of Allreduces for a number already in hand.
    auto reform_x = [&](double z_nbr) {
        {
            RSATB_PHASE("setup/isqrt_ns_S_times_Z");
            stage("S * Z (sparse product)");
            // X is dead: this product does not read it. Released before the
            // result allocates rather than at the assignment afterwards.
            tmp2.reset();
            tmp2 = S_mat.spmm(*S_inv_sqrt, sptol);
            // The one product in the loop whose operands and result are all
            // measured. The every-fifth-step drift refresh guarantees this runs
            // often enough to keep the calibration current as Z fills in.
            //
            // Unless the product saturated: a count at the full row is a FLOOR,
            // not a measurement, and solving with it biases d low -- diamond
            // sc888 calibrated to d = 1.8 from a clipped S Z where the crystal
            // is 3D. Harmless there (the cap in nbr_product bounds every
            // saturated prediction anyway) but wrong to learn from.
            const double sz_nbr = nbr_per_row(tmp2);
            if (sz_nbr < kSaturatedFraction * row_length) {
                fill_dim = calibrate_fill_dim(s_nbr, z_nbr, sz_nbr, fill_dim);
            }
        }
        {
            RSATB_PHASE("setup/isqrt_ns_ZH_S_Z");
            stage("Z^H (S Z)");
            // Split so S Z can be freed between the two halves: it dies with
            // the upper triangle, and the completion is the larger allocation.
            //
            // complete_hermitian_inplace, not the copying form: the completed
            // matrix is about twice the triangle it is built from, and copying
            // leaves the triangle standing beside it for the whole of the
            // mirror -- 5.72 GB of a 21.50 GB step peak, measured at 4096
            // atoms. The in-place form hands each row back as it is mirrored.
            // All three completions in this file are written this way.
            BlockSpMat<T> upper =
                S_inv_sqrt->spmm_hermitian_upper(tmp2, sptol, /*transA=*/true);
            tmp2.reset();
            upper.complete_hermitian_inplace();
            tmp2 = std::move(upper);
        }
    };

    // The seed's reform is a free calibration point for the reform constant:
    // it runs unconditionally, and it is the same two products a scheduled
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
        // get_block_density() is a collective (Allreduce on the group comm):
        // it must run on EVERY rank, outside the rank-0 log guard. Calling it
        // inside the guard paired rank 0's Allreduce with whatever collective
        // the other ranks issued next -- MPI_ERR_TRUNCATE on them and a
        // garbage density on rank 0 were exactly that.
        const double x_nbr = nbr_per_row(tmp2);
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            // Memory on the step line, not only in the end-of-setup table: a
            // run that is killed inside this loop never reaches the table, and
            // then the one thing worth knowing -- how close it came -- is lost.
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
            // One kernel, not a square followed by an axpby. X^2 is DENSER than
            // the P it becomes -- squaring doubles the sparsity radius while the
            // polynomial collapses toward the identity under sptol -- so forming
            // it was the largest transient in the loop: measured at 39 GB on a
            // 4096-atom step, filtered away to produce 17 GB of P.
            // square_polynomial_upper holds one row of the square per thread
            // instead, and drops once on the finished P. That single drop
            // removes one truncation, but it does NOT make this route pointwise
            // more accurate -- the two drop different blocks (the two-call route
            // decides on |X^2|, this one on the finished polynomial), and on a
            // scattered pattern it measures ~16% worse at matched nnz. Expect
            // the same order, not an improvement; see square_polynomial.hpp.
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
            // In place: Z is dead the moment it has been multiplied, and this
            // was the loop's peak-memory phase precisely because the ordinary
            // product holds the old Z and the new one at full size at once --
            // 46 of the 104 GB peak on a 4096-atom cell, for a matrix that only
            // ever needs to exist once. spmm_inplace consumes Z row block by
            // row block instead. Nothing else references it: S_inv_sqrt is the
            // caller's output and P is a separate matrix.
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
        // constant is applied, so a route that runs at a quarter of the other's
        // flop rate is charged for it.
        // Captured, not re-read at print time: note_route_cost below updates
        // these, and a log line that reported the post-update constants next to
        // the pre-update prediction would not multiply out -- which defeats the
        // whole point of printing the model's inputs beside its output.
        const double spu_reform = seconds_per_unit(false);
        const double spu_propagate = seconds_per_unit(true);
        // Captured for the same reason as the two constants above, and missed
        // there originally: reform_x recalibrates fill_dim, so a reform step
        // printed the POST-reform dimension beside the units the PRE-reform one
        // produced. The line then did not multiply out -- on a 4096-atom step it
        // showed units that require d = 2.15 next to "fill dim 1.05741", which
        // reads as a calibration collapse that never happened.
        const double fill_dim_used = fill_dim;
        const double cost_reform = units_reform * spu_reform;
        const double cost_propagate = units_propagate * spu_propagate;
        // The every-fifth-step exact reformation is a DRIFT bound, not a cost
        // choice: P X P accumulates rounding that only Z^H (S Z) clears, and
        // the period was fixed by measurement. Cost never overrides it.
        const bool drift_refresh = ((count + 1) % 5 == 0);
        // The residual guard stays as well, now doing only the job it can do.
        // The model ranks cost, not stability, and a seed poor enough to leave
        // the residual above 1e-1 is where propagation has the least margin. It
        // can only ever force MORE reforms, never fewer, and on a healthy seed
        // it never binds -- a residual that large comes with a Z still sparse
        // enough that the model picks reform on cost anyway.
        // Cost alone is not the objective, because the two routes do not leave
        // the same X behind.
        //
        // Propagation sets X <- P X P, and P collapses toward the identity under
        // sptol as the iteration converges, so X CONTRACTS: 2080 -> 1514 -> 933
        // neighbours per row on a 4096-atom run. Reformation sets X <- Z^H S Z,
        // which inherits Z's density, and Z only ever fills in: the same run,
        // routed on step cost alone, took X from 2080 to 3092 in one step.
        // Reform is the cheaper STEP there and the model said so correctly
        // (predicted 1206s, measured 1168s) -- but it re-densifies the operand
        // every later step pays for, and the run went from 3279s/101GB over
        // three propagating steps to >3500s in step 2 alone, past 122GB.
        //
        // What decides whether that trade is real is whether X has room to
        // contract at all. Saturated -- a full block row, as on sc888 where X
        // sits at 1024 of 1024 -- P X P is dense whatever the polynomial does,
        // reform costs nothing in trajectory terms, and routing on cost is the
        // 4.3x win it was measured to be. With headroom, contraction is worth
        // buying: below, up to kTrajectoryPremium times the reform's cost, which
        // one contracting step repays by shrinking every step after it.
        const bool x_saturated = !(x_nbr < kSaturatedFraction * row_length);
        const bool cost_favours_propagate =
            x_saturated ? (cost_propagate < cost_reform)
                        : (cost_propagate < kTrajectoryPremium * cost_reform);
        //
        // COLLECTIVE. Every other input to this decision already agrees across
        // ranks -- drift_refresh is a step counter, norm and the neighbour
        // counts are Allreduced -- but the seconds-per-unit constants are timed
        // locally, so cost_reform and cost_propagate can differ between ranks.
        // The two branches issue DIFFERENT collective sequences (rarh_upper
        // against reform_x's two products), so a split decision is not a
        // slightly worse route, it is a hang or a mismatched exchange. Rank 0's
        // choice is broadcast and is the only one that runs.
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
            // One kernel, not two products. P X is denser than P X P -- its
            // radius is r_P + r_X while the answer collapses toward the
            // identity under sptol -- so forming it was both the largest
            // allocation in the loop and the place the threshold did the most
            // damage. rarh_upper holds one of its rows per thread instead, and
            // drops once on the finished result. P^H = P exactly, so neither
            // half needs a transpose.
            BlockSpMat<T> upper = tmp1.rarh_upper(tmp2, sptol);
            // BOTH operands die with the product: X is being replaced, and P's
            // last read was the kernel call above -- the next step rebuilds it
            // from the new X. Held past this point, P would stand beside the
            // mirroring X_new for exactly the phase this step peaks in.
            tmp1.reset();
            tmp2.reset();
            upper.complete_hermitian_inplace();
            tmp2 = std::move(upper);
        } else {
            // P is dead on this route -- the reform rebuilds X from Z and S,
            // and the next step's polynomial starts from that X. Reform is
            // chosen precisely at the early steps where P is densest, so
            // holding it across the two products below was the single largest
            // avoidable term in the reform-step peak.
            tmp1.reset();
            reform_x(z_nbr);
        }
        const double t_update = util::wall_time() - t0_update;
        // Feed the route that actually ran back into its own constant. Only the
        // route taken is learned from -- there is no counterfactual timing for
        // the one that did not run, which is exactly why the earlier
        // reform-vs-propagate comparison off a single call was biased.
        note_route_cost(propagate, propagate ? units_propagate : units_reform, t_update);

        norm = residual_norm(tmp2);

        // Per-step times, because the profile table cannot show them: it
        // accumulates a total and a call count per phase, and over a run these
        // products vary by three orders of magnitude between the first step and
        // the last. A total divided by a count describes neither end, and
        // reading one as if it did is what hid the misroute above. The model's
        // inputs and its two predictions are printed next to the times they
        // predicted, so the routing can be checked against reality after a run
        // rather than trusted.
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

    // Never REPORT a propagated residual.
    //
    // X <- P X P is not the residual of the Z being returned: Z is
    // filter(Z P), and whatever the threshold dropped from that product never
    // enters P X P, so the true operand is (ZP - D)^H S (ZP - D). The gap is
    // truncation, not rounding -- measured here at 5x (1.98e-4 reported against
    // 9.95e-4 exact on 4096 atoms) -- so the number handed back was telling the
    // caller the iteration was five times better than it was.
    //
    // Reform once, AFTER the loop, purely to report. It deliberately does NOT
    // feed the termination test.
    //
    // Feeding it back was tried and is a trap. The exact residual is bounded
    // below by sptol, so "still above tol" is permanently true on any run whose
    // tol is under sptol; and re-entering compares an exact norm against a
    // propagated norm_pre, so the plateau detector -- which assumes one metric
    // -- never fires either. A 4096-atom run that had been exiting cleanly in
    // three steps instead ground on for eight hours, its exact residual getting
    // WORSE with each extra step (9.95e-4, then 1.49e-3) because at the sptol
    // floor the extra propagation only adds truncation. Report honestly, stop
    // where the iteration actually stopped improving.
    if (last_advance_propagated) {
        const double spectral = residual_spectral_estimate(12);
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            std::cout << "inverse_sqrt:   ||Z^H S Z - I||_2 estimate " << spectral
                      << " (loop metric, max block Frobenius of the PROPAGATED X, "
                      << norm << ")" << std::endl;
        }
    }

    // Say what actually happened. A run that plateaus against the sptol
    // floor used to report "converged" -- on a 145k-orbital moire system that
    // dressed a 2e-5 stall (target 1e-7) as success at the end of 25 minutes
    // of iteration, one of which the plateau exit spent confirming the stall.
    const bool hit_tol = norm <= tol;
    const bool hit_maxiter = count >= maxit;
    // Collective: every rank, before the rank-0 guard (see the note above).
    const double final_density = S_inv_sqrt->get_block_density();

    // What the iteration cost in memory, against what the caller KEEPS. The
    // absolute peak alone does not size a machine; the question a bigger run
    // asks is "how much do I need in order to end up holding this", and the
    // answer is a multiple, not a number of bytes.
    //
    // The denominator is measured rss after the iteration's temporaries are
    // released, not Z's block bytes: the steady state also holds S, every
    // matrix's adjacency, and whatever the allocator has not returned -- on a
    // 4096-atom cell Z's values are 23.2 GB where the state that survives is
    // 26+ GB, and quoting the tighter one makes the ratio look worse than the
    // machine ever sees. X and P die HERE rather than at the closing brace, so
    // the reading is of the state that actually persists.
    tmp1.reset();
    tmp2.reset();
    const long long steady_kb_local = util::read_rss_kb();
    long long peak_kb_local = util::MemorySampler::instance().global_peak_kb();
    // MAXLOC, not two independent MAXes. The comment here used to claim the
    // steady value was "its own steady state ... for its own peak", but taking
    // the max of each separately picks whichever rank happened to be highest on
    // each, which need not be the same rank -- so the ratio could pair one
    // rank's peak with another's steady and describe no rank at all. MAXLOC
    // finds the rank that actually peaked; its steady value is then fetched
    // from that rank, which is what makes the ratio mean something.
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
