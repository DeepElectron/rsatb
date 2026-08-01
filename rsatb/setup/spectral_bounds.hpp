#ifndef RSATB_SETUP_SPECTRAL_BOUNDS_HPP
#define RSATB_SETUP_SPECTRAL_BOUNDS_HPP

// The spectral window [emin, emax] the Chebyshev expansions are rescaled onto.
//
// One question -- the extreme eigenvalues of H, or of the pencil (H, S) --
// answered at two levels:
//
//   ExtremalEigenvalues()    one operator, one k-point.
//   EstimateSpectralBounds() every k-point this rank owns, reduced to a window.
//
// Which numerical route answers the first is deliberately not part of the
// interface: an iterative distributed solve where that is sound, an exact
// dense diagonalization where it is not, a norm bound where neither can run.
// Callers state the problem, not the algorithm.
//
// This is the most expensive part of setup -- two eigensolves per k-point --
// and it used to be buried inside OperatorSet::initialize alongside the
// k-point partitioning and the operator sampling. As free functions over
// already-built operators these can be tested, replaced, or skipped on their
// own.

#include "kpoint_partition.hpp"
#include "../calc/eigen.hpp"
#include "../calc/preconditioner.hpp"

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_graph.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "vbcsr/core/detail/kernels/lapack_api.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

extern "C" {
// Generalized Hermitian-definite eigenproblem, for the non-orthogonal case.
// zheev_ comes from vbcsr's lapack_api.hpp; zhegv_ is not declared there. The
// integer width must be vbcsr's typedef, not int -- an ILP64 BLAS is detected
// automatically at configure time, and a mismatch here would corrupt silently.
void zhegv_(const vbcsr::vbcsr_lapack_int* itype, const char* jobz, const char* uplo,
            const vbcsr::vbcsr_lapack_int* n, std::complex<double>* a,
            const vbcsr::vbcsr_lapack_int* lda, std::complex<double>* b,
            const vbcsr::vbcsr_lapack_int* ldb, double* w, std::complex<double>* work,
            const vbcsr::vbcsr_lapack_int* lwork, double* rwork,
            vbcsr::vbcsr_lapack_int* info);
}

namespace rsatb {
namespace setup {

/// The energy window, plus whether it was actually established.
struct SpectralBounds {
    double emin = -1.0;
    double emax = 1.0;
    bool ready = false;
};

struct BoundsOptions {
    /// Skips estimation entirely when false.
    bool compute = true;
    /// Below this, a Hamiltonian is treated as identically zero.
    double zero_tol = 1e-12;
    /// Cap on iterative matrix-vector products per solve.
    int max_matvecs = 10000;
    /// Padding added outside the extreme eigenvalues.
    double padding = 0.5;
};

/// Communicators the estimate is reduced over.
struct BoundsComms {
    /// Leaders of each sub-communicator; the extremes are reduced across it.
    MPI_Comm master = MPI_COMM_NULL;
    /// This rank's sub-communicator; the result is broadcast within it.
    MPI_Comm sub = MPI_COMM_NULL;
};

/// A single k-point's operators, as far as bounds estimation is concerned.
///
/// Deliberately a view rather than the owning state: the estimator does not
/// need to know how the operators were built or who keeps them alive.
struct BoundsKPointView {
    using Scalar = std::complex<double>;
    using Matrix = vbcsr::BlockSpMat<Scalar>;
    using Preconditioner = rsatb::calc::Preconditioner<Scalar>;

    KPoint kpoint;
    // Non-const because PrimmeEigenSolver::solve takes a mutable reference;
    // the estimate itself does not modify the operator.
    Matrix* hamiltonian = nullptr;
    Matrix* overlap = nullptr;                 // null for an orthogonal basis
    Preconditioner* preconditioner = nullptr;  // null when none was built
};

inline std::string KPointToString(const KPoint& kpoint) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < kpoint.size(); ++i) {
        if (i > 0) out << ", ";
        out << kpoint[i];
    }
    out << "]";
    return out.str();
}

namespace detail {

/// Scalar rows a rank must own for the distributed iterative solve to
/// terminate. PRIMME deadlocks at one row per rank: the ranks stop agreeing
/// on how many collective reductions to perform and the solve never returns.
/// The boundary is sharp -- two rows per rank is fine, one is not.
constexpr int kMinLocalRowsForPrimme = 2;

/// Largest global dimension the dense route will assemble on one rank.
///
/// The dense matrix is 16 n^2 bytes; at this cap that is 64 MB, held only by
/// rank 0 and only for the duration of the solve. A distribution that defeats
/// PRIMME implies n < 2 * n_ranks, so reaching this cap needs thousands of
/// ranks on a system of a few thousand orbitals -- possible, hence the
/// explicit error rather than an allocation failure.
constexpr int kMaxDenseBoundsDimension = 2048;

/// Scalar rows owned by this rank.
inline int LocalRowCount(const vbcsr::DistGraph& graph) {
    int rows = 0;
    for (size_t i = 0; i < graph.owned_global_indices.size(); ++i) {
        rows += graph.block_sizes[i];
    }
    return rows;
}

/// Whether any rank owns too few rows for the iterative solve to be used.
///
/// Collective over the graph's communicator: every rank must reach the same
/// verdict, or the ranks that call PRIMME will wait forever on the ones that
/// do not.
inline bool DistributionDefeatsPrimme(const vbcsr::DistGraph& graph) {
    int local_rows = LocalRowCount(graph);
    if (graph.comm == MPI_COMM_NULL || graph.size <= 1) {
        return false;
    }
    int min_rows = local_rows;
    MPI_Allreduce(&local_rows, &min_rows, 1, MPI_INT, MPI_MIN, graph.comm);
    return min_rows < kMinLocalRowsForPrimme;
}

/// Exact extreme eigenvalues by gathering the operator and calling LAPACK.
///
/// The matrix is materialized by multiplying through a globally-identity block
/// of vectors, gathered onto rank 0, diagonalized there, and the two numbers
/// broadcast back. Row and column order agree by construction -- both follow
/// the rank-major layout established by the MPI_Exscan below -- so the result
/// is a symmetric permutation of the true matrix and its eigenvalues are exact.
inline void DenseExtremalEigenvalues(
    vbcsr::BlockSpMat<std::complex<double>>& hamiltonian,
    vbcsr::BlockSpMat<std::complex<double>>* overlap,
    double* emin,
    double* emax
) {
    using Scalar = std::complex<double>;
    vbcsr::DistGraph* graph = hamiltonian.graph;
    const bool distributed = (graph->comm != MPI_COMM_NULL && graph->size > 1);

    const int n_local = LocalRowCount(*graph);
    int n_global = n_local;
    int row_offset = 0;
    if (distributed) {
        MPI_Allreduce(&n_local, &n_global, 1, MPI_INT, MPI_SUM, graph->comm);
        MPI_Exscan(&n_local, &row_offset, 1, MPI_INT, MPI_SUM, graph->comm);
        if (graph->rank == 0) row_offset = 0;
    }

    if (n_global > kMaxDenseBoundsDimension) {
        throw std::runtime_error(
            "Spectral bounds fell back to a dense solve because a rank owns fewer than " +
            std::to_string(kMinLocalRowsForPrimme) + " rows, but the global dimension (" +
            std::to_string(n_global) + ") exceeds the dense limit of " +
            std::to_string(kMaxDenseBoundsDimension) +
            ". Use fewer ranks per k-point group, or supply explicit energy bounds.");
    }
    if (n_global == 0) {
        throw std::runtime_error("Spectral bounds requested for an empty operator.");
    }

    // Local rows of the operator, obtained as op * I.
    const auto local_rows_of = [&](vbcsr::BlockSpMat<Scalar>& op) {
        vbcsr::DistMultiVector<Scalar> identity(graph, n_global);
        identity.set_constant(Scalar(0.0));
        for (int i = 0; i < n_local; ++i) {
            identity.row_data(i)[row_offset + i] = Scalar(1.0);
        }
        vbcsr::DistMultiVector<Scalar> result(graph, n_global);
        op.mult_dense(identity, result);

        std::vector<Scalar> rows(static_cast<size_t>(n_local) * n_global);
        for (int i = 0; i < n_local; ++i) {
            const Scalar* src = result.row_data(i);
            std::copy(src, src + n_global, rows.begin() + static_cast<size_t>(i) * n_global);
        }
        return rows;
    };

    const auto gather_to_root = [&](const std::vector<Scalar>& rows) {
        std::vector<Scalar> full;
        if (!distributed) {
            full = rows;
            return full;
        }
        std::vector<int> counts(graph->size, 0);
        const int my_count = n_local * n_global;
        MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, graph->comm);
        std::vector<int> displs(graph->size, 0);
        if (graph->rank == 0) {
            for (int r = 1; r < graph->size; ++r) displs[r] = displs[r - 1] + counts[r - 1];
            full.resize(static_cast<size_t>(n_global) * n_global);
        }
        MPI_Gatherv(rows.data(), my_count, MPI_C_DOUBLE_COMPLEX, full.data(), counts.data(),
                    displs.data(), MPI_C_DOUBLE_COMPLEX, 0, graph->comm);
        return full;
    };

    std::vector<Scalar> h_dense = gather_to_root(local_rows_of(hamiltonian));
    std::vector<Scalar> s_dense;
    if (overlap != nullptr) {
        s_dense = gather_to_root(local_rows_of(*overlap));
    }

    double bounds[2] = {0.0, 0.0};
    int failed = 0;
    if (!distributed || graph->rank == 0) {
        // h_dense is row-major; LAPACK reads column-major, so it sees the
        // transpose. For a Hermitian matrix that is the conjugate, which has
        // the same (real) eigenvalues -- and the same holds for the pencil,
        // since both operands transpose together.
        using LapackInt = vbcsr::vbcsr_lapack_int;
        const LapackInt n = n_global;
        std::vector<double> evals(n_global);
        std::vector<double> rwork(std::max(1, 3 * n_global - 2));
        const LapackInt lwork = std::max(1, 2 * n_global - 1) * 8;
        std::vector<Scalar> work(lwork);
        LapackInt info = 0;
        const char jobz = 'N';
        const char uplo = 'U';

        if (overlap != nullptr) {
            const LapackInt itype = 1;
            zhegv_(&itype, &jobz, &uplo, &n, h_dense.data(), &n, s_dense.data(), &n,
                   evals.data(), work.data(), &lwork, rwork.data(), &info);
        } else {
            vbcsr::zheev_(&jobz, &uplo, &n, h_dense.data(), &n, evals.data(), work.data(), &lwork,
                          rwork.data(), &info);
        }

        if (info != 0) {
            failed = static_cast<int>(info);
        } else {
            bounds[0] = evals.front();
            bounds[1] = evals.back();
        }
    }

    if (distributed) {
        MPI_Bcast(&failed, 1, MPI_INT, 0, graph->comm);
        MPI_Bcast(bounds, 2, MPI_DOUBLE, 0, graph->comm);
    }
    if (failed != 0) {
        throw std::runtime_error(
            "Dense spectral-bounds solve failed: LAPACK returned info=" + std::to_string(failed) +
            (overlap != nullptr ? " from zhegv (is the overlap positive definite?)."
                                : " from zheev."));
    }

    *emin = bounds[0];
    *emax = bounds[1];
}

}  // namespace detail

/// Extreme eigenvalues of H, or of the pencil (H, S), for one k-point.
///
/// Routes internally, in this order:
///   - a Hamiltonian below `zero_tol` in magnitude contributes the point 0;
///   - a distribution that would deadlock the iterative solver (a rank owning
///     one scalar row) takes an exact dense diagonalization, which such a
///     matrix is small enough to afford by construction;
///   - otherwise two iterative solves, smallest and largest;
///   - if those fail with neither overlap nor preconditioner present, the
///     Frobenius norm, which bounds the spectrum validly if loosely.
///
/// Args:
///   view: The operators and the k-point they belong to (for diagnostics).
///   options: Tolerances and the matrix-vector cap.
///   emin, emax: Set to the extreme eigenvalues on every rank.
///   verbose: Whether to report a fallback on stderr.
/// Throws:
///   std::runtime_error if the operator holds non-finite entries, or if the
///   solve fails and no conservative fallback applies.
inline void ExtremalEigenvalues(const BoundsKPointView& view,
                                const BoundsOptions& options,
                                double* emin,
                                double* emax,
                                bool verbose = false) {
    using Scalar = BoundsKPointView::Scalar;

    const double h_max_abs = view.hamiltonian->maxabs();
    if (!std::isfinite(h_max_abs)) {
        throw std::runtime_error(
            "Automatic bounds estimation found a non-finite Hamiltonian entry at k-point " +
            KPointToString(view.kpoint) + ".");
    }
    if (h_max_abs <= options.zero_tol) {
        *emin = 0.0;
        *emax = 0.0;
        return;
    }

    if (detail::DistributionDefeatsPrimme(*view.hamiltonian->graph)) {
        detail::DenseExtremalEigenvalues(*view.hamiltonian, view.overlap, emin, emax);
        return;
    }

    rsatb::calc::PrimmeEigenSolver<Scalar> esolver;
    std::vector<double> evals_min;
    std::vector<double> evals_max;
    std::vector<vbcsr::DistVector<Scalar>> evecs;
    const auto pc_type = view.preconditioner ? view.preconditioner->get_type()
                                             : rsatb::calc::PreconditionerType::Left;
    try {
        esolver.solve(*view.hamiltonian, view.overlap, view.preconditioner, 1, primme_smallest,
                      1e-3, evals_min, evecs, verbose, pc_type, options.max_matvecs);
        esolver.solve(*view.hamiltonian, view.overlap, view.preconditioner, 1, primme_largest,
                      1e-3, evals_max, evecs, verbose, pc_type, options.max_matvecs);
    } catch (const std::exception& error) {
        // Without an overlap or a preconditioner the Frobenius norm is a
        // valid, if loose, bound on the spectrum. With either one present
        // there is no such cheap fallback, so the failure is reported.
        if (view.overlap != nullptr || view.preconditioner != nullptr) {
            throw std::runtime_error(
                "Automatic bounds estimation failed at k-point " + KPointToString(view.kpoint) +
                ". Provide explicit energy bounds or increase max_matvecs. Underlying error: " +
                error.what());
        }
        const double fallback = view.hamiltonian->frobenius_norm();
        if (!std::isfinite(fallback)) {
            throw std::runtime_error(
                "Automatic bounds fallback found a non-finite Hamiltonian norm at k-point " +
                KPointToString(view.kpoint) + ".");
        }
        if (verbose && view.hamiltonian->graph != nullptr &&
            view.hamiltonian->graph->rank == 0) {
            std::cerr << "Warning: iterative bounds solve failed at k-point "
                      << KPointToString(view.kpoint)
                      << "; using conservative Frobenius norm bound. Underlying error: "
                      << error.what() << std::endl;
        }
        *emin = -fallback;
        *emax = fallback;
        return;
    }

    *emin = evals_min.empty() ? std::numeric_limits<double>::max() : evals_min[0];
    *emax = evals_max.empty() ? std::numeric_limits<double>::lowest() : evals_max[0];
}

/// Validates and adopts a caller-supplied window.
inline SpectralBounds MakeExplicitBounds(double emin, double emax) {
    if (!std::isfinite(emin) || !std::isfinite(emax) || !(emin < emax)) {
        throw std::runtime_error("Explicit energy bounds must be finite and satisfy emin < emax.");
    }
    return SpectralBounds{emin, emax, true};
}

/// Estimates the spectral window from the extreme eigenvalues at each k-point.
///
/// Each rank takes the extremes of its own k-points via ExtremalEigenvalues,
/// they are reduced across sub-communicator leaders, and the result is
/// broadcast back down.
///
/// Returns:
///   The padded window, with `ready` set.
/// Throws:
///   std::runtime_error if a solve fails and no conservative fallback applies,
///   or if the reduction does not produce a finite window.
inline SpectralBounds EstimateSpectralBounds(
    const std::vector<BoundsKPointView>& views,
    const BoundsOptions& options,
    const BoundsComms& comms,
    bool verbose = false
) {
    if (options.zero_tol < 0.0) {
        throw std::runtime_error("BoundsOptions.zero_tol must be non-negative.");
    }
    if (options.max_matvecs < 0) {
        throw std::runtime_error("BoundsOptions.max_matvecs must be non-negative.");
    }

    double local_emin = std::numeric_limits<double>::max();
    double local_emax = std::numeric_limits<double>::lowest();

    for (const auto& view : views) {
        if (view.hamiltonian == nullptr) {
            continue;
        }
        double k_emin = 0.0;
        double k_emax = 0.0;
        ExtremalEigenvalues(view, options, &k_emin, &k_emax, verbose);
        local_emin = std::min(local_emin, k_emin);
        local_emax = std::max(local_emax, k_emax);
    }

    double global_emin = local_emin;
    double global_emax = local_emax;

    if (comms.master != MPI_COMM_NULL) {
        MPI_Allreduce(&local_emin, &global_emin, 1, MPI_DOUBLE, MPI_MIN, comms.master);
        MPI_Allreduce(&local_emax, &global_emax, 1, MPI_DOUBLE, MPI_MAX, comms.master);
    }
    if (comms.sub != MPI_COMM_NULL) {
        MPI_Bcast(&global_emin, 1, MPI_DOUBLE, 0, comms.sub);
        MPI_Bcast(&global_emax, 1, MPI_DOUBLE, 0, comms.sub);
    }

    if (!std::isfinite(global_emin) || !std::isfinite(global_emax) || global_emin > global_emax) {
        throw std::runtime_error("Automatic bounds estimation did not produce finite energy bounds.");
    }

    return SpectralBounds{global_emin - options.padding, global_emax + options.padding, true};
}

}  // namespace setup
}  // namespace rsatb

#endif  // RSATB_SETUP_SPECTRAL_BOUNDS_HPP
