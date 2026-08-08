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
    if (krylov > 0) {
        summary.mean_krylov_blocks =
            static_cast<double>(tally.krylov_blocks.load()) / static_cast<double>(krylov);
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
    double tol = 1e-7;        // target for ||Z^H S Z - I||_maxabs
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
    //  - Cubic, not quadratic. The seed's maxabs-block residual understates
    //    its spectral spread: X's lowest eigenvalue sits far from 1, where
    //    the quadratic map contracts only ~2x per step and needs four steps
    //    against the cubic's three. Roughly a wash in flops; the wider
    //    contraction basin keeps it.
    //
    //  - Propagated X (X <- P X P), not per-step exact reformation. Static
    //    flop counting says the exact Z^H (S Z) refresh should win (S at
    //    0.286 block density is the sparsest matrix in the loop; P and X are
    //    dense) -- and it measures SLOWER, 345s against 302s: P and X
    //    converge to the identity, their off-diagonal blocks fall under
    //    sptol, and the threshold-aware SpGEMM makes the late P X P products
    //    nearly free, while an exact refresh multiplies the permanently
    //    dense Z at full price every step. The propagation drifts, which the
    //    periodic exact reformation bounds.
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

    auto residual_norm = [&](BlockSpMat<T>& X) {
        RSATB_PHASE("setup/isqrt_ns_residual");
        X.shift(T(-1.0));
        double norm = 0.0;
        for (double n : X.get_block_norms()) norm = std::max(norm, n);
        double global = norm;
        MPI_Allreduce(&norm, &global, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        X.shift(T(1.0));
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
    auto reform_x = [&]() {
        {
            RSATB_PHASE("setup/isqrt_ns_S_times_Z");
            stage("S * Z (sparse product)");
            // X is dead: this product does not read it. Released before the
            // result allocates rather than at the assignment afterwards.
            tmp2.reset();
            tmp2 = S_mat.spmm(*S_inv_sqrt, sptol);
        }
        {
            RSATB_PHASE("setup/isqrt_ns_congruence");
            stage("Z^H (S Z) (hermitian congruence)");
            // Split so S Z can be freed between the two halves: it dies with
            // the upper triangle, and the completion is the larger allocation.
            BlockSpMat<T> upper =
                S_inv_sqrt->spmm_hermitian_upper(tmp2, sptol, /*transA=*/true);
            tmp2.reset();
            tmp2 = upper.complete_hermitian();
        }
    };

    reform_x();
    double norm = residual_norm(tmp2);
    double norm_pre = 0.0;
    int count = 0;

    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        util::throw_if_interrupted(graph->comm);
        // get_block_density() is a collective (Allreduce on the group comm):
        // it must run on EVERY rank, outside the rank-0 log guard. Calling it
        // inside the guard paired rank 0's Allreduce with whatever collective
        // the other ranks issued next -- MPI_ERR_TRUNCATE on them and a
        // garbage density on rank 0 were exactly that.
        const double z_nbr = nbr_per_row(*S_inv_sqrt);
        const double x_nbr = nbr_per_row(tmp2);
        if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
            // Memory on the step line, not only in the end-of-setup table: a
            // run that is killed inside this loop never reaches the table, and
            // then the one thing worth knowing -- how close it came -- is lost.
            const double rss_gb = util::read_rss_kb() / 1048576.0;
            const double peak_gb =
                util::MemorySampler::instance().global_peak_kb() / 1048576.0;
            std::cout << "inverse_sqrt: step " << count + 1 << ": maxabs residual " << norm
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
        {
            RSATB_PHASE("setup/isqrt_ns_X_squared");
            stage("X^2 (hermitian square)");
            // The previous step's P is dead; X is both operands, so only the
            // destination can be released here.
            tmp1.reset();
            tmp1 = tmp2.spmm_hermitian(tmp2, sptol);
        }
        {
            // Assembling P from X^2: three passes over P's blocks, no product.
            // Separated from the square so a surprise here (a filter that
            // drops nothing, an axpby across mismatched graphs) is visible
            // rather than folded into the SpGEMM above.
            RSATB_PHASE("setup/isqrt_ns_polynomial");
            tmp1.axpby(T(-1.25), tmp2, T(0.375));
            tmp1.shift(T(1.875));
            tmp1.filter_blocks(sptol);
        }

        {
            RSATB_PHASE("setup/isqrt_ns_Z_times_P");
            // P's neighbour count decides this product's cost together with
            // Z's, and it decides how much of P every rank has to fetch. Both
            // are on the line so a stall here names its own cause.
            const double p_nbr = nbr_per_row(tmp1);
            if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
                std::cout << "inverse_sqrt:   Z * P (P " << p_nbr << " nbr/row)"
                          << std::endl;
            } else {
                stage("Z * P");
            }
            *S_inv_sqrt = S_inv_sqrt->spmm(tmp1, sptol);
        }

        // X <- P X P while converging -- cheap, because P and X collapse
        // toward the identity and the threshold drops their blocks -- with
        // an exact reformation every fifth step and while still far out.
        // P^H = P exactly, so the first product needs no transpose at all,
        // and the second is the known-Hermitian congruence.
        if (norm < 1e-1 && ((count + 1) % 5 != 0)) {
            {
                RSATB_PHASE("setup/isqrt_ns_P_times_X");
                stage("P X (propagate)");
                tmp2 = tmp1.spmm(tmp2, sptol);
            }
            {
                RSATB_PHASE("setup/isqrt_ns_PX_times_P");
                stage("(P X) P (hermitian)");
                // The peak of the whole iteration sits here. P X is the left
                // operand and dies with the upper triangle, so splitting the
                // product frees it before the completion allocates.
                BlockSpMat<T> upper = tmp2.spmm_hermitian_upper(tmp1, sptol);
                tmp2.reset();
                tmp2 = upper.complete_hermitian();
            }
        } else {
            reform_x();
        }
        norm = residual_norm(tmp2);
        ++count;
    }

    // Say what actually happened. A run that plateaus against the sptol
    // floor used to report "converged" -- on a 145k-orbital moire system that
    // dressed a 2e-5 stall (target 1e-7) as success at the end of 25 minutes
    // of iteration, one of which the plateau exit spent confirming the stall.
    const bool hit_tol = norm <= tol;
    const bool hit_maxiter = count >= maxit;
    // Collective: every rank, before the rank-0 guard (see the note above).
    const double final_density = S_inv_sqrt->get_block_density();
    if (rank == 0 && util::Log::enabled(util::LogLevel::kProgress)) {
        std::cout << "inverse_sqrt: " << (hit_tol ? "converged" : (hit_maxiter ? "stopped at maxiter" : "plateaued"))
                  << " at maxabs residual " << norm << ", density "
                  << final_density << " (" << count << " Newton-Schulz steps, "
                  << (MPI_Wtime() - t_ns0) << "s)" << std::endl;
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
