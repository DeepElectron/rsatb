#ifndef RSATB_CALC_PRECONDITIONER_HPP
#define RSATB_CALC_PRECONDITIONER_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "isqrtm_vbcsr.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

// LAPACK declarations for block inversion
// LAPACK declarations for block inversion
#if !defined(RSATB_USE_MKL)
extern "C" {
    void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
    void dgetri_(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info);
    void zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info);
    void zgetri_(const int* n, std::complex<double>* a, const int* lda, const int* ipiv, std::complex<double>* work, const int* lwork, int* info);
}
#endif

namespace rsatb {
namespace calc {

using namespace rsatb::backend;
using namespace vbcsr;

enum class PreconditionerType { Left, Right, Symmetric };

// Base class for Preconditioners
template <typename T, typename Kernel = DefaultKernel<T>>
class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void apply(const DistVector<T>& src, DistVector<T>& dst) = 0;
    virtual void apply(const DistMultiVector<T>& src, DistMultiVector<T>& dst) = 0;
    virtual void apply_adjoint(const DistVector<T>& src, DistVector<T>& dst) = 0;
    virtual void apply_adjoint(const DistMultiVector<T>& src, DistMultiVector<T>& dst) = 0;
    virtual PreconditionerType get_type() const = 0;
};

// Block Jacobian Preconditioner
// Approximates A^-1 by inverting diagonal blocks.
// Stores the inverse diagonal blocks in a BlockSpMat P.
template <typename T, typename Kernel = DefaultKernel<T>>
class BlockJacobianPreconditioner : public Preconditioner<T, Kernel> {
public:
    BlockSpMat<T, Kernel> P; // Stores block diagonal inverse

    BlockJacobianPreconditioner(BlockSpMat<T, Kernel>& A) : P(A.graph) {
        compute(A);
    }

    PreconditionerType get_type() const override { return PreconditionerType::Left; }

    void compute(BlockSpMat<T, Kernel>& A) {
        // Iterate local blocks of A
        int n_rows = A.row_ptr.size() - 1;
        
        for (int i = 0; i < n_rows; ++i) {
            int start = A.row_ptr[i];
            int end = A.row_ptr[i+1];
            int global_row = A.graph->get_global_index(i);
            
            for (int k = start; k < end; ++k) {
                int local_col = A.col_ind[k];
                int global_col = A.graph->get_global_index(local_col);
                
                if (global_row == global_col) {
                    // Found diagonal block
                    size_t offset = A.blk_ptr[k];
                    int rows = A.graph->block_sizes[i];
                    int cols = A.graph->block_sizes[local_col];
                    
                    if (rows != cols) {
                        throw std::runtime_error("BlockJacobianPreconditioner: Diagonal block is not square.");
                    }
                    
                    // Copy block data
                    std::vector<T> block(rows * cols);
                    std::copy(A.val.begin() + offset, A.val.begin() + offset + rows*cols, block.begin());
                    
                    // Invert block
                    invert_block(block.data(), rows);
                    
                    // Add to P
                    P.add_block(global_row, global_col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::ColMajor);
                }
            }
        }
        P.assemble();
    }

    void apply(const DistVector<T>& src, DistVector<T>& dst) override {
        P.mult_optimized(const_cast<DistVector<T>&>(src), dst);
    }
    
    void apply(const DistMultiVector<T>& src, DistMultiVector<T>& dst) override {
        P.mult_dense(const_cast<DistMultiVector<T>&>(src), dst);
    }

    void apply_adjoint(const DistVector<T>& src, DistVector<T>& dst) override {
        P.mult_adjoint(const_cast<DistVector<T>&>(src), dst);
    }

    void apply_adjoint(const DistMultiVector<T>& src, DistMultiVector<T>& dst) override {
        P.mult_dense_adjoint(const_cast<DistMultiVector<T>&>(src), dst);
    }

private:
    void invert_block(double* a, int n) {
        if (n == 1) {
            a[0] = 1.0 / a[0];
            return;
        }
        int info;
        std::vector<int> ipiv(n);
        dgetrf_(&n, &n, a, &n, ipiv.data(), &info);
        if (info != 0) throw std::runtime_error("Block inversion failed (dgetrf).");
        
        int lwork = n * n;
        std::vector<double> work(lwork);
        dgetri_(&n, a, &n, ipiv.data(), work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("Block inversion failed (dgetri).");
    }

    void invert_block(std::complex<double>* a, int n) {
        if (n == 1) {
            a[0] = 1.0 / a[0];
            return;
        }
        int info;
        std::vector<int> ipiv(n);
        
#ifdef RSATB_USE_MKL
        zgetrf_(&n, &n, reinterpret_cast<MKL_Complex16*>(a), &n, ipiv.data(), &info);
#else
        zgetrf_(&n, &n, a, &n, ipiv.data(), &info);
#endif

        if (info != 0) throw std::runtime_error("Block inversion failed (zgetrf).");
        
        int lwork = n * n;
        std::vector<std::complex<double>> work(lwork);
        
#ifdef RSATB_USE_MKL
        zgetri_(&n, reinterpret_cast<MKL_Complex16*>(a), &n, ipiv.data(), reinterpret_cast<MKL_Complex16*>(work.data()), &lwork, &info);
#else
        zgetri_(&n, a, &n, ipiv.data(), work.data(), &lwork, &info);
#endif

        if (info != 0) throw std::runtime_error("Block inversion failed (zgetri).");
    }
};

// ISQRTM Preconditioner
// Approximates A^-1/2 using Newton-Schulz iteration.
// Useful for Symmetric Preconditioning.
template <typename T, typename Kernel = DefaultKernel<T>>
class ISQRTMPreconditioner : public Preconditioner<T, Kernel> {
public:
    BlockSpMat<T, Kernel> P; // Stores S^-1/2

    ISQRTMPreconditioner(BlockSpMat<T, Kernel>& S, int max_iter = 10, double tol = 1e-5, bool verbose = false) : P(S.graph) {
        // Compute P = S^-1/2
        isqrtm(S, &P, tol, 1e-5, max_iter, verbose);
    }

    PreconditionerType get_type() const override { return PreconditionerType::Symmetric; }

    ISQRTMPreconditioner(BlockSpMat<T, Kernel>& P) : P(P) {}

    void apply(const DistVector<T>& src, DistVector<T>& dst) override {
        P.mult_optimized(const_cast<DistVector<T>&>(src), dst);
    }
    
    void apply(const DistMultiVector<T>& src, DistMultiVector<T>& dst) override {
        P.mult_dense(const_cast<DistMultiVector<T>&>(src), dst);
    }

    void apply_adjoint(const DistVector<T>& src, DistVector<T>& dst) override {
        P.mult_adjoint(const_cast<DistVector<T>&>(src), dst);
    }

    void apply_adjoint(const DistMultiVector<T>& src, DistMultiVector<T>& dst) override {
        P.mult_dense_adjoint(const_cast<DistMultiVector<T>&>(src), dst);
    }
};

// Wrapper for Symmetric Preconditioning: P * A * P
// This allows treating the preconditioned system as a single operator.
template <typename MatrixType, typename PrecondType, typename T>
class SymPrecondMat {
public:
    MatrixType& A;
    PrecondType& P;
    DistGraph* graph;

    SymPrecondMat(MatrixType& mat, PrecondType& pc) 
        : A(mat), P(pc), graph(mat.graph), 
          v_tmp1(nullptr), v_tmp2(nullptr),
          mv_tmp1(nullptr), mv_tmp2(nullptr) {}

    void mult_optimized(const DistVector<T>& x, DistVector<T>& y) {
        if (!v_tmp1) v_tmp1 = std::make_unique<DistVector<T>>(graph);
        if (!v_tmp2) v_tmp2 = std::make_unique<DistVector<T>>(graph);
        
        // y = P * A * P * x
        P.apply(x, *v_tmp1);
        A.mult_optimized(*v_tmp1, *v_tmp2);
        P.apply(*v_tmp2, y);
    }

    void mult_dense(const DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        if (!mv_tmp1 || mv_tmp1->num_vectors != X.num_vectors) {
            mv_tmp1 = std::make_unique<DistMultiVector<T>>(graph, X.num_vectors);
        }
        if (!mv_tmp2 || mv_tmp2->num_vectors != X.num_vectors) {
            mv_tmp2 = std::make_unique<DistMultiVector<T>>(graph, X.num_vectors);
        }
        
        // Y = P * A * P * X
        P.apply(X, *mv_tmp1);
        A.mult_dense(*mv_tmp1, *mv_tmp2);
        P.apply(*mv_tmp2, Y);
    }
    

private:
    std::unique_ptr<DistVector<T>> v_tmp1, v_tmp2;
    std::unique_ptr<DistMultiVector<T>> mv_tmp1, mv_tmp2;
};

} // namespace calc
} // namespace rsatb

#endif
