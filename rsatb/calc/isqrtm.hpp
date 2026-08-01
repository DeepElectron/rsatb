#ifndef RSATB_CALC_ISQRTM_HPP
#define RSATB_CALC_ISQRTM_HPP

// S^{-1/2} for the ISQRTM preconditioner: the graph seed (one thin dense
// inverse-square-root action per block row, through vbcsr's generic
// graph_function_apply driver) refined by Newton-Schulz purification on the
// sparse matrix. The generic graph traversal lives in vbcsr; everything
// specific to the inverse square root - the dense kernel (isqrt_action.hpp),
// the route bookkeeping and the purification - is rsatb's.

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
    long long rational = 0;    // ... by the tridiagonal sinc-rational route
    long long eigen = 0;       // ... by the clamped eigendecomposition
    long long failed = 0;
    double mean_krylov_blocks = 0.0;
};

namespace detail {

// Thread-safe accumulator behind GraphInverseSqrtReport.
struct IsqrtTally {
    std::atomic<long long> krylov{0};
    std::atomic<long long> band{0};
    std::atomic<long long> rational{0};
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
            case isqrt::Method::TridiagonalRational:
                rational.fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                eigen.fetch_add(1, std::memory_order_relaxed);
                break;
        }
    }
};

// RSATB_ISQRT_INIT=auto|krylov|band|rational|eigen overrides the requested route
// without a rebuild; "eigen" reproduces the plain per-subgraph diagonalization.
inline isqrt::Method resolve_isqrt_method(isqrt::Method requested) {
    const char* env = std::getenv("RSATB_ISQRT_INIT");
    if (env == nullptr) return requested;
    const std::string value(env);
    if (value == "auto") return isqrt::Method::Auto;
    if (value == "krylov") return isqrt::Method::InverseBlockLanczos;
    if (value == "band") return isqrt::Method::BandZolotarev;
    if (value == "rational") return isqrt::Method::TridiagonalRational;
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
// certificate holds, tridiagonal + sinc rational otherwise, and the clamped
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
    long long counts[5] = {krylov, tally.band.load(), tally.rational.load(),
                           tally.eigen.load(), tally.failed.load()};
    long long totals[5] = {0, 0, 0, 0, 0};
    MPI_Comm comm = S.graph->comm;
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF) {
        MPI_Allreduce(counts, totals, 5, MPI_LONG_LONG, MPI_SUM, comm);
    } else {
        std::copy(counts, counts + 5, totals);
    }

    GraphInverseSqrtReport summary;
    summary.krylov = totals[0];
    summary.band = totals[1];
    summary.rational = totals[2];
    summary.eigen = totals[3];
    summary.failed = totals[4];
    if (krylov > 0) {
        summary.mean_krylov_blocks =
            static_cast<double>(tally.krylov_blocks.load()) / static_cast<double>(krylov);
    }
    if (report) *report = summary;

    if (verbose && S.graph->rank == 0) {
        std::cout << "graph_inverse_sqrt: subgraph actions - krylov " << summary.krylov
                  << ", band " << summary.band << ", rational " << summary.rational
                  << ", eigen " << summary.eigen << ", failed " << summary.failed;
        if (krylov > 0) std::cout << " (mean krylov blocks " << summary.mean_krylov_blocks << ")";
        std::cout << std::endl;
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
        isqrt::Options kernel = options.kernel;
        // The seed only has to be as accurate as the iteration's target.
        kernel.relative_tolerance = (tol > 0.0)
                                        ? std::min(kernel.relative_tolerance, tol)
                                        : kernel.relative_tolerance;
        graph_inverse_sqrt(S_mat, S_inv_sqrt, kernel, verbose);
    }

    const double t_seed = MPI_Wtime() - t_seed0;
    if (rank == 0 && verbose) {
        std::cout << "inverse_sqrt: seed " << t_seed << "s" << std::endl;
    }

    if (maxit == 0) return;

    // Newton-Schulz on X = Z^H S Z: Z <- Z (1.875 I - 1.25 X + 0.375 X^2),
    // which is the order-3 iteration for X -> I.
    const double t_ns0 = MPI_Wtime();
    BlockSpMat<T> tmp1(graph), tmp2(graph);

    auto residual_norm = [&](BlockSpMat<T>& X) {
        X.shift(T(-1.0));
        double norm = 0.0;
        for (double n : X.get_block_norms()) norm = std::max(norm, n);
        double global = norm;
        MPI_Allreduce(&norm, &global, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        X.shift(T(1.0));
        return global;
    };

    tmp2 = S_inv_sqrt->spmm(S_mat, sptol, true, false);
    tmp2 = tmp2.spmm(*S_inv_sqrt, sptol, false, false);
    double norm = residual_norm(tmp2);
    double norm_pre = 0.0;
    int count = 0;

    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        if (rank == 0 && verbose) {
            std::cout << "inverse_sqrt: maxabs residual " << norm << ", density "
                      << S_inv_sqrt->get_block_density() << std::endl;
        }
        if (norm > 100.0) {
            if (rank == 0) {
                std::cout << "inverse_sqrt: diverged, residual " << norm << std::endl;
            }
            break;
        }
        norm_pre = norm;

        tmp1 = tmp2.spmm(tmp2, sptol);
        tmp1.axpby(T(-1.25), tmp2, T(0.375));
        tmp1.shift(T(1.875));
        tmp1.filter_blocks(sptol);

        *S_inv_sqrt = S_inv_sqrt->spmm(tmp1, sptol);

        // Propagating X through the same polynomial is cheaper than reforming
        // Z^H S Z, but it drifts; refresh it exactly every fifth step and while
        // still far from convergence.
        if (norm < 1e-1 && ((count + 1) % 5 != 0)) {
            tmp2 = tmp1.spmm(tmp2, sptol, true, false);
            tmp2 = tmp2.spmm(tmp1, sptol, false, false);
        } else {
            tmp2 = S_inv_sqrt->spmm(S_mat, sptol, true, false);
            tmp2 = tmp2.spmm(*S_inv_sqrt, sptol, false, false);
        }
        norm = residual_norm(tmp2);
        ++count;
    }

    if (rank == 0 && verbose) {
        std::cout << "inverse_sqrt: converged at maxabs residual " << norm << ", density "
                  << S_inv_sqrt->get_block_density() << " (" << count << " Newton-Schulz steps, "
                  << (MPI_Wtime() - t_ns0) << "s)" << std::endl;
    }
}

} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_ISQRTM_HPP
