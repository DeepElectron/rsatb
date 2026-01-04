#ifndef RSATB_CALC_EIGEN_HPP
#define RSATB_CALC_EIGEN_HPP

#include "rsatb/core/backend/block_csr.hpp"
#include "rsatb/core/backend/dist_vector.hpp"
#include "rsatb/core/backend/dist_multivector.hpp"
#include "rsatb/core/backend/scalar_traits.hpp"
#include "solver.hpp" // For LinearSolver
#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <cstring>

// Include PRIMME header
#include <primme.h>

namespace rsatb {
namespace calc {

using namespace rsatb::backend;

template <typename T, typename Kernel = DefaultKernel<T>>
class PrimmeEigenSolver {
public:
    using Real = typename ScalarTraits<T>::real_type;

    // Context to pass to PRIMME callbacks
    struct Context {
        BlockSpMat<T, Kernel>* H = nullptr;
        BlockSpMat<T, Kernel>* S = nullptr;
        Preconditioner<T, Kernel>* M = nullptr;
        LinearSolver<T, Kernel>* solver = nullptr; // For S inversion
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
    void solve(BlockSpMat<T, Kernel>& H, 
               BlockSpMat<T, Kernel>* S = nullptr, 
               Preconditioner<T, Kernel>* M = nullptr, 
               int num_eigenvalues = 1, 
               primme_target target = primme_largest, 
               double tol = 1e-6, 
               std::vector<Real>& evalues = {}, 
               std::vector<DistVector<T>>& evectors = {},
               bool verbose = false,
               PreconditionerType pc_type = PreconditionerType::Left) {
        
        DistGraph* graph = H.graph;
        int n_local = 0;
        for(int dim : H.row_dims) n_local += dim;
        
        long long n_global_long = 0;
        long long n_local_long = n_local;
        MPI_Allreduce(&n_local_long, &n_global_long, 1, MPI_LONG_LONG, MPI_SUM, graph->comm);
        int n_global = static_cast<int>(n_global_long);
        
        primme_params primme;
        primme_initialize(&primme);

        primme.n = n_global;
        primme.nLocal = n_local;
        primme.numEvals = num_eigenvalues;
        primme.target = target;
        primme.eps = tol;
        primme.printLevel = verbose ? 3 : 0;
        
        int numProcs, procID;
        MPI_Comm_size(graph->comm, &numProcs);
        MPI_Comm_rank(graph->comm, &procID);
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
        LinearSolver<T, Kernel> linear_solver(tol * 0.1, 1000, false); 
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
        primme.globalSumReal = global_sum_real_wrapper;
        primme.commInfo = &graph->comm;

        // Allocate result arrays
        std::vector<T> evecs_buf(n_local * num_eigenvalues);
        std::vector<Real> evals_buf(num_eigenvalues);
        std::vector<Real> res_norms_buf(num_eigenvalues);

        int ret = 0;
        if constexpr (std::is_same_v<T, double>) {
            ret = dprimme(evals_buf.data(), evecs_buf.data(), res_norms_buf.data(), &primme);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            ret = zprimme(evals_buf.data(), (void*)evecs_buf.data(), res_norms_buf.data(), &primme);
        } else {
            throw std::runtime_error("PrimmeEigenSolver only supports double and complex<double>");
        }

        if (ret != 0) {
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
    void solve_largest(BlockSpMat<T, Kernel>& H, 
                       BlockSpMat<T, Kernel>* S, 
                       Preconditioner<T, Kernel>* M,
                       int num, 
                       double tol, 
                       std::vector<Real>& evals, 
                       std::vector<DistVector<T>>& evecs, 
                       bool verbose=false,
                       PreconditionerType pc_type = PreconditionerType::Left) {
        solve(H, S, M, num, primme_largest, tol, evals, evecs, verbose, pc_type);
    }

    // Helper to find extremal eigenvalues (min and max) of H
    void solve_extremal(BlockSpMat<T, Kernel>& H, 
                        BlockSpMat<T, Kernel>* S, 
                        Preconditioner<T, Kernel>* M,
                        double tol, 
                        Real& emin, 
                        Real& emax, 
                        bool verbose=false,
                        PreconditionerType pc_type = PreconditionerType::Left) {
        
        std::vector<Real> evals;
        std::vector<DistVector<T>> evecs;
        
        // Smallest
        solve(H, S, M, 1, primme_smallest, tol, evals, evecs, verbose, pc_type);
        emin = evals[0];
        
        // Largest
        solve(H, S, M, 1, primme_largest, tol, evals, evecs, verbose, pc_type);
        emax = evals[0];
    }

private:
    // Helper to copy data between PRIMME buffer and DistMultiVector
    static void copy_in(Context* ctx, void* buffer, int blockSize, std::unique_ptr<DistMultiVector<T>>& mv) {
        if (!mv || mv->num_vectors != blockSize) {
            mv = std::make_unique<DistMultiVector<T>>(ctx->graph, blockSize);
        }
        
        T* buf = static_cast<T*>(buffer);
        int n_local = mv->local_rows; // Assuming owned rows match nLocal
        
        #pragma omp parallel for
        for (int c = 0; c < blockSize; ++c) {
            T* dst = mv->col_data(c);
            const T* src = buf + c * n_local;
            std::memcpy(dst, src, n_local * sizeof(T));
        }
    }

    static void copy_out(Context* ctx, void* buffer, int blockSize, std::unique_ptr<DistMultiVector<T>>& mv) {
        T* buf = static_cast<T*>(buffer);
        int n_local = mv->local_rows;
        
        #pragma omp parallel for
        for (int c = 0; c < blockSize; ++c) {
            T* dst = buf + c * n_local;
            const T* src = mv->col_data(c);
            std::memcpy(dst, src, n_local * sizeof(T));
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
            ctx->M->apply(*ctx->tmp_mv_aux, *ctx->tmp_mv_out);     // out = M * aux
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
            ctx->solver->solve(*ctx->S, *ctx->tmp_mv_in, *ctx->tmp_mv_out, *ctx->M, PreconditionerType::Left);
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
        if (sendBuf == recvBuf) {
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
