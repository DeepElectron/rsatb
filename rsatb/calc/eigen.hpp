#ifndef RSATB_CALC_EIGEN_HPP
#define RSATB_CALC_EIGEN_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "vbcsr/core/scalar_traits.hpp"
#include "solver.hpp" // For LinearSolver
#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <cstring>

// Include PRIMME header
#include <primme.h>

// Declared so this header compiles on its own; the using-directive below
// previously relied on some earlier include having opened the namespace.
namespace rsatb { namespace setup {} }

namespace rsatb {
namespace calc {

// The operator vocabulary this header names is the setup layer's.
using namespace rsatb::setup;

template <typename T>
class PrimmeEigenSolver {
public:
    using Real = typename ScalarTraits<T>::real_type;

    /// What the last solve cost, so a caller can put the price next to the
    /// answer. A bounds estimate that silently swallows a third of setup is
    /// exactly the failure mode this exists to expose.
    struct SolveStats {
        long long matvecs = 0;
        long long preconds = 0;
        double seconds = 0.0;
    };
    SolveStats last_stats;

    // Context to pass to PRIMME callbacks
    struct Context {
        BlockSpMat<T>* H = nullptr;
        BlockSpMat<T>* S = nullptr;
        Preconditioner<T>* M = nullptr;
        LinearSolver<T>* solver = nullptr; // For S inversion
        PreconditionerType pc_type = PreconditionerType::Left; // For composed operator
        
        // Multivector buffers for block operations
        std::unique_ptr<DistMultiVector<T>> tmp_mv_in;
        std::unique_ptr<DistMultiVector<T>> tmp_mv_out;
        std::unique_ptr<DistMultiVector<T>> tmp_mv_aux; // Extra buffer for composed op
        
        DistGraph* graph = nullptr;
    };

    // Main Solve Function
    // Supports:
    // 1. Standard: H x = lambda x (S=nullptr, M=nullptr)
    // 2. Generalized: H x = lambda S x (S!=nullptr)
    //    - If S is provided, we use PRIMME's generalized interface.
    //    - We also provide a preconditioner that solves S y = x using CG (InverseS_Wrapper).
    //    - If M is provided, it is used as a preconditioner for the inner CG solve.
    // 3. Composed Standard: Solve for eigenvalues of transformed operator (S=nullptr, M!=nullptr)
    //    - Left: M H x = lambda x
    //    - Right: H M x = lambda x
    //    - Symmetric: M H M x = lambda x
    void solve(BlockSpMat<T>& H, 
               BlockSpMat<T>* S = nullptr, 
               Preconditioner<T>* M = nullptr, 
               int num_eigenvalues = 1, 
               primme_target target = primme_largest, 
               double tol = 1e-6, 
               std::vector<Real>& evalues = {}, 
               std::vector<DistVector<T>>& evectors = {},
               bool verbose = false,
               PreconditionerType pc_type = PreconditionerType::Left,
               int max_matvecs = 0,
               const DistVector<T>* initial = nullptr) {
        
        DistGraph* graph = H.graph;
        const bool mpi_ready = comm_is_usable(graph->comm);
        int n_local = 0;
        for(int i=0; i<graph->owned_global_indices.size(); ++i) n_local += graph->block_sizes[i];
        
        long long n_global_long = 0;
        long long n_local_long = n_local;
        if (mpi_ready && graph->size > 1) {
            MPI_Allreduce(&n_local_long, &n_global_long, 1, MPI_LONG_LONG, MPI_SUM, graph->comm);
        } else {
            n_global_long = n_local_long;
        }
        int n_global = static_cast<int>(n_global_long);
        
        primme_params primme;
        primme_initialize(&primme);

        primme.n = n_global;
        primme.nLocal = n_local;
        primme.numEvals = num_eigenvalues;
        primme.target = target;
        primme.eps = tol;
        primme.printLevel = verbose ? 3 : 0;
        if (max_matvecs > 0) {
            primme.maxMatvecs = max_matvecs;
        }
        
        int numProcs = 1;
        int procID = 0;
        if (mpi_ready && graph->size > 1) {
            MPI_Comm_size(graph->comm, &numProcs);
            MPI_Comm_rank(graph->comm, &procID);
        }
        primme.numProcs = numProcs;
        primme.procID = procID;
        
        // Setup Context
        Context ctx;
        ctx.H = &H;
        ctx.S = S;
        ctx.M = M;
        ctx.graph = graph;
        ctx.pc_type = pc_type;
        
        // Instantiate LinearSolver for S inversion if needed
        LinearSolver<T> linear_solver(tol * 0.1, 1000, false); 
        if (S) {
            ctx.solver = &linear_solver;
        }
        
        primme.matrix = &ctx;
        
        if (S) {
            // Generalized Problem
            primme.matrixMatvec = matvec_wrapper;
            primme.massMatrix = &ctx;
            primme.massMatrixMatvec = mass_matvec_wrapper;
            
            // Use InverseS_Wrapper as preconditioner for the generalized problem
            primme.preconditioner = &ctx;
            primme.applyPreconditioner = inverse_s_wrapper;
        } else if (M) {
            // Composed Standard Problem (MHM, MH, HM)
            primme.matrixMatvec = composed_matvec_wrapper;
            // No mass matrix (Identity)
            // No preconditioner (M is part of the operator)
        } else {
            // Standard Problem
            primme.matrixMatvec = matvec_wrapper;
        }
        
        // Set global sum callback
        if (mpi_ready && graph->size > 1) {
            primme.globalSumReal = global_sum_real_wrapper;
            primme.commInfo = &graph->comm;
        }

        // Pin the method. Left unset, PRIMME runs DYNAMIC, which picks
        // between GD+k and JDQMR from wall-clock measurements taken as it
        // goes (main_iter.c, primme_CostModel) -- the one input that differs
        // between two otherwise bit-identical runs. On a production moire
        // step the same 6-rank bounds solve took 142s in one run and 1633s in
        // the next, with the operators, partition and seed all identical;
        // timing noise flipped the switch, nothing else had changed.
        //
        // MIN_MATVECS (GD+k with Olsen) rather than pinning JDQMR: our matvec
        // is a distributed SpMV and, on the generalized path, every
        // preconditioner application is itself a CG solve of S y = x. JDQMR
        // applies the preconditioner once per INNER step, so it multiplies
        // precisely the expensive thing; GD+k applies it once per outer
        // iteration.
        //
        // AFTER the operator wiring above, not before: set_method reads
        // applyPreconditioner to decide correctionParams.precondition, and
        // calling it early silently disabled preconditioning -- visible as
        // preconds=0 in the stats where the generalized path was in use.
        primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);

        // Allocate result arrays
        std::vector<T> evecs_buf(n_local * num_eigenvalues);
        std::vector<Real> evals_buf(num_eigenvalues);
        std::vector<Real> res_norms_buf(num_eigenvalues);

        // A caller-supplied start. Without one PRIMME draws its own from a
        // rank-seeded generator over rank-local rows, which bakes the
        // decomposition into the answer twice over.
        if (initial != nullptr) {
            std::memcpy(evecs_buf.data(), initial->local_data(),
                        static_cast<size_t>(n_local) * sizeof(T));
            primme.initSize = 1;
        }

        int ret = 0;
        if constexpr (std::is_same_v<T, double>) {
            ret = dprimme(evals_buf.data(), evecs_buf.data(), res_norms_buf.data(), &primme);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            ret = zprimme(evals_buf.data(), evecs_buf.data(), res_norms_buf.data(), &primme);
        } else {
            throw std::runtime_error("PrimmeEigenSolver only supports double and complex<double>");
        }

        last_stats.matvecs = static_cast<long long>(primme.stats.numMatvecs);
        last_stats.preconds = static_cast<long long>(primme.stats.numPreconds);
        last_stats.seconds = primme.stats.elapsedTime;

        if (ret != 0) {
            primme_free(&primme);
            throw std::runtime_error("PRIMME failed with error code: " + std::to_string(ret));
        }

        // Copy results
        evalues = evals_buf;
        evectors.clear();
        evectors.reserve(num_eigenvalues);
        for(int i=0; i<num_eigenvalues; ++i) {
            DistVector<T> vec(graph);
            T* local_data = vec.local_data();
            for(int j=0; j<n_local; ++j) {
                local_data[j] = evecs_buf[i * n_local + j];
            }
            evectors.push_back(std::move(vec));
        }
        
        primme_free(&primme);
    }

    // Helper to find largest eigenvalues
    void solve_largest(BlockSpMat<T>& H, 
                       BlockSpMat<T>* S, 
                       Preconditioner<T>* M,
                       int num, 
                       double tol, 
                       std::vector<Real>& evals, 
                       std::vector<DistVector<T>>& evecs, 
                       bool verbose=false,
                       PreconditionerType pc_type = PreconditionerType::Left,
                       int max_matvecs = 0) {
        solve(H, S, M, num, primme_largest, tol, evals, evecs, verbose, pc_type, max_matvecs);
    }

    void solve_smallest(BlockSpMat<T>& H, 
                        BlockSpMat<T>* S, 
                        Preconditioner<T>* M,
                        int num, 
                        double tol, 
                        std::vector<Real>& evals, 
                        std::vector<DistVector<T>>& evecs, 
                       bool verbose=false,
                       PreconditionerType pc_type = PreconditionerType::Left,
                       int max_matvecs = 0) {
        solve(H, S, M, num, primme_smallest, tol, evals, evecs, verbose, pc_type, max_matvecs);
    }

    // Helper to find extremal eigenvalues (min and max) of H
    void solve_extremal(BlockSpMat<T>& H, 
                        BlockSpMat<T>* S, 
                        Preconditioner<T>* M,
                        double tol, 
                        Real& emin, 
                        Real& emax, 
                        bool verbose=false,
                        PreconditionerType pc_type = PreconditionerType::Left,
                        int max_matvecs = 0) {
        
        std::vector<Real> evals;
        std::vector<DistVector<T>> evecs;
        
        // Smallest
        solve(H, S, M, 1, primme_smallest, tol, evals, evecs, verbose, pc_type, max_matvecs);
        emin = evals[0];
        
        // Largest
        solve(H, S, M, 1, primme_largest, tol, evals, evecs, verbose, pc_type, max_matvecs);
        emax = evals[0];
    }

private:
    static bool comm_is_usable(MPI_Comm comm) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized && comm != MPI_COMM_NULL;
    }

    // Helper to copy data between PRIMME buffer and DistMultiVector
    static void copy_in(Context* ctx, void* buffer, int blockSize, std::unique_ptr<DistMultiVector<T>>& mv) {
        if (!mv || mv->num_vectors != blockSize) {
            mv = std::make_unique<DistMultiVector<T>>(ctx->graph, blockSize);
        }
        
        T* buf = static_cast<T*>(buffer);
        int n_local = mv->local_rows; // Assuming owned rows match nLocal
        
        // PRIMME's buffer is column-major (n_local x blockSize); DistMultiVector is
        // row-major, so this is a transposing copy rather than a per-column memcpy.
        #pragma omp parallel for
        for (int i = 0; i < n_local; ++i) {
            T* dst = mv->row_data(i);
            for (int c = 0; c < blockSize; ++c) {
                dst[c] = buf[static_cast<size_t>(c) * n_local + i];
            }
        }
    }

    static void copy_out(Context* ctx, void* buffer, int blockSize, std::unique_ptr<DistMultiVector<T>>& mv) {
        T* buf = static_cast<T*>(buffer);
        int n_local = mv->local_rows;
        
        #pragma omp parallel for
        for (int i = 0; i < n_local; ++i) {
            const T* src = mv->row_data(i);
            for (int c = 0; c < blockSize; ++c) {
                buf[static_cast<size_t>(c) * n_local + i] = src[c];
            }
        }
    }

    // Callbacks
    static void matvec_wrapper(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
        Context* ctx = static_cast<Context*>(primme->matrix);
        int bs = *blockSize;
        
        copy_in(ctx, x, bs, ctx->tmp_mv_in);
        if (!ctx->tmp_mv_out || ctx->tmp_mv_out->num_vectors != bs) {
            ctx->tmp_mv_out = std::make_unique<DistMultiVector<T>>(ctx->graph, bs);
        }
        
        ctx->H->mult_dense(*ctx->tmp_mv_in, *ctx->tmp_mv_out);
        
        copy_out(ctx, y, bs, ctx->tmp_mv_out);
        *ierr = 0;
    }
    
    // Wrapper for composed operator (MHM, MH, HM)
    static void composed_matvec_wrapper(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
        Context* ctx = static_cast<Context*>(primme->matrix);
        int bs = *blockSize;
        
        copy_in(ctx, x, bs, ctx->tmp_mv_in);
        if (!ctx->tmp_mv_out || ctx->tmp_mv_out->num_vectors != bs) {
            ctx->tmp_mv_out = std::make_unique<DistMultiVector<T>>(ctx->graph, bs);
        }
        if (!ctx->tmp_mv_aux || ctx->tmp_mv_aux->num_vectors != bs) {
            ctx->tmp_mv_aux = std::make_unique<DistMultiVector<T>>(ctx->graph, bs);
        }
        
        if (ctx->pc_type == PreconditionerType::Left) {
            // y = M * H * x
            ctx->H->mult_dense(*ctx->tmp_mv_in, *ctx->tmp_mv_aux); // aux = H * x
            ctx->M->apply(*ctx->tmp_mv_aux, *ctx->tmp_mv_out);     // out = M * aux
        } else if (ctx->pc_type == PreconditionerType::Right) {
            // y = H * M * x
            ctx->M->apply(*ctx->tmp_mv_in, *ctx->tmp_mv_aux);      // aux = M * x
            ctx->H->mult_dense(*ctx->tmp_mv_aux, *ctx->tmp_mv_out); // out = H * aux
        } else if (ctx->pc_type == PreconditionerType::Symmetric) {
            // y = M * H * M * x
            ctx->M->apply(*ctx->tmp_mv_in, *ctx->tmp_mv_out);      // out = M * x (reuse out as temp)
            ctx->H->mult_dense(*ctx->tmp_mv_out, *ctx->tmp_mv_aux); // aux = H * out
            ctx->M->apply_adjoint(*ctx->tmp_mv_aux, *ctx->tmp_mv_out);     // out = M * aux
        }
        
        copy_out(ctx, y, bs, ctx->tmp_mv_out);
        *ierr = 0;
    }

    static void mass_matvec_wrapper(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
        Context* ctx = static_cast<Context*>(primme->massMatrix);
        int bs = *blockSize;
        
        copy_in(ctx, x, bs, ctx->tmp_mv_in);
        if (!ctx->tmp_mv_out || ctx->tmp_mv_out->num_vectors != bs) {
            ctx->tmp_mv_out = std::make_unique<DistMultiVector<T>>(ctx->graph, bs);
        }
        
        ctx->S->mult_dense(*ctx->tmp_mv_in, *ctx->tmp_mv_out);
        
        copy_out(ctx, y, bs, ctx->tmp_mv_out);
        *ierr = 0;
    }

    static void preconditioner_wrapper(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
        Context* ctx = static_cast<Context*>(primme->preconditioner);
        int bs = *blockSize;
        
        copy_in(ctx, x, bs, ctx->tmp_mv_in);
        if (!ctx->tmp_mv_out || ctx->tmp_mv_out->num_vectors != bs) {
            ctx->tmp_mv_out = std::make_unique<DistMultiVector<T>>(ctx->graph, bs);
        }
        
        ctx->M->apply(*ctx->tmp_mv_in, *ctx->tmp_mv_out);
        
        copy_out(ctx, y, bs, ctx->tmp_mv_out);
        *ierr = 0;
    }

    static void inverse_s_wrapper(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
        Context* ctx = static_cast<Context*>(primme->preconditioner);
        int bs = *blockSize;
        
        copy_in(ctx, x, bs, ctx->tmp_mv_in);
        if (!ctx->tmp_mv_out || ctx->tmp_mv_out->num_vectors != bs) {
            ctx->tmp_mv_out = std::make_unique<DistMultiVector<T>>(ctx->graph, bs);
        }
        
        // Solve S * y = x
        // Use tmp_mv_out as initial guess (0)
        ctx->tmp_mv_out->set_constant(0.0);
        
        if (ctx->M) {
            ctx->solver->solve(*ctx->S, *ctx->tmp_mv_in, *ctx->tmp_mv_out, *ctx->M, ctx->M->get_type());
        } else {
            ctx->solver->solve(*ctx->S, *ctx->tmp_mv_in, *ctx->tmp_mv_out);
        }
        
        copy_out(ctx, y, bs, ctx->tmp_mv_out);
        *ierr = 0;
    }

    static void global_sum_real_wrapper(void *sendBuf, void *recvBuf, int *count, primme_params *primme, int *ierr) {
        MPI_Comm comm = *(static_cast<MPI_Comm*>(primme->commInfo));
        // PRIMME uses Real type for global sums (e.g. dot products)
        MPI_Datatype type = (sizeof(Real) == 8) ? MPI_DOUBLE : MPI_FLOAT;
        if (!comm_is_usable(comm)) {
            if (sendBuf != recvBuf) {
                std::memcpy(recvBuf, sendBuf, static_cast<size_t>(*count) * sizeof(Real));
            }
        } else if (sendBuf == recvBuf) {
            MPI_Allreduce(MPI_IN_PLACE, recvBuf, *count, type, MPI_SUM, comm);
        } else {
            MPI_Allreduce(sendBuf, recvBuf, *count, type, MPI_SUM, comm);
        }
        *ierr = 0;
    }
};

}; // namespace calc
}; // namespace rsatb

#endif
