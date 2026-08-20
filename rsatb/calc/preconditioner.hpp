#ifndef RSATB_CALC_PRECONDITIONER_HPP
#define RSATB_CALC_PRECONDITIONER_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "vbcsr/core/detail/kernels/lapack_api.hpp"
#include "isqrtm.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

// LAPACK declarations for block inversion
// LAPACK declarations for block inversion
// LAPACK declarations are now in lapack_types.hpp

namespace rsatb {
namespace calc {

using namespace vbcsr;

enum class PreconditionerType { Left, Right, Symmetric };

/// Tag selecting the borrowing constructor of ISQRTMPreconditioner, so that
/// "wrap this factor" cannot be confused with "compute the factor of this".
struct BorrowFactor {};
inline constexpr BorrowFactor kBorrowFactor{};

// Base class for Preconditioners
template <typename T>
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
template <typename T>
class BlockJacobianPreconditioner : public Preconditioner<T> {
public:
    BlockSpMat<T> P; // Stores block diagonal inverse

    BlockJacobianPreconditioner(BlockSpMat<T>& A) : P(A.graph) {
        compute(A);
    }

    PreconditionerType get_type() const override { return PreconditionerType::Left; }

    void compute(BlockSpMat<T>& A) {
        // Iterate local blocks of A
        const auto& row_ptr = A.row_ptr();
        const auto& col_ind = A.col_ind();
        int n_rows = static_cast<int>(row_ptr.size()) - 1;
        
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            int global_row = A.graph->get_global_index(i);
            
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
                int global_col = A.graph->get_global_index(local_col);
                
                if (global_row == global_col) {
                    // Found diagonal block
                    const T* data = A.block_data(k);
                    int rows = A.graph->block_sizes[i];
                    int cols = A.graph->block_sizes[local_col];
                    
                    if (rows != cols) {
                        throw std::runtime_error("BlockJacobianPreconditioner: Diagonal block is not square.");
                    }
                    
                    // Copy block data
                    std::vector<T> block(rows * cols);
                    std::copy(data, data + rows*cols, block.begin());
                    
                    // Invert block
                    invert_block(block.data(), rows);
                    
                    // Add to P. invert_block() (LAPACK getrf/getri) reads and writes the
                    // same convention, so the inverse comes back in the convention it was
                    // given -- i.e. the canonical row-major of block_data() above.
                    P.add_block(global_row, global_col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
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
    void invert_block(double* a, int n_in) {
        vbcsr_lapack_int n = n_in;
        if (n == 1) {
            a[0] = 1.0 / a[0];
            return;
        }
        vbcsr_lapack_int info;
        std::vector<vbcsr_lapack_int> ipiv(n);
        dgetrf_(&n, &n, a, &n, ipiv.data(), &info);
        if (info != 0) throw std::runtime_error("Block inversion failed (dgetrf).");
        
        vbcsr_lapack_int lwork = n * n;
        std::vector<double> work(lwork);
        dgetri_(&n, a, &n, ipiv.data(), work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("Block inversion failed (dgetri).");
    }

    void invert_block(std::complex<double>* a, int n_in) {
        vbcsr_lapack_int n = n_in;
        if (n == 1) {
            a[0] = 1.0 / a[0];
            return;
        }
        vbcsr_lapack_int info;
        std::vector<vbcsr_lapack_int> ipiv(n);
        
        vbcsr_complex_double* a_ptr = reinterpret_cast<vbcsr_complex_double*>(a);
        zgetrf_(&n, &n, a_ptr, &n, ipiv.data(), &info);

        if (info != 0) throw std::runtime_error("Block inversion failed (zgetrf).");
        
        vbcsr_lapack_int lwork = n * n;
        std::vector<vbcsr_complex_double> work(lwork);
        
        zgetri_(&n, a_ptr, &n, ipiv.data(), work.data(), &lwork, &info);

        if (info != 0) throw std::runtime_error("Block inversion failed (zgetri).");
    }
};

// Symmetric preconditioning by a sparse factor Z: applies Z and Z^H.
//
// It has to be a class rather than a plain BlockSpMat& because Chebyshev and
// SymPrecondMat take a Preconditioner<T>&, and what they need from it is the
// four-way apply/apply_adjoint interface over both DistVector and
// DistMultiVector -- a matrix reference alone supplies none of that. Storage
// is not the reason: the borrowing form below holds exactly a reference.
//
// Two ways in, because both cases are real. Given an overlap it computes
// S^{-1/2} and owns the result; given a factor somebody else already has --
// a shared Z reused across samplers, a purified one from elsewhere -- it
// borrows without copying, which matters because Z is large and BlockSpMat is
// deliberately move-only.
template <typename T>
class ISQRTMPreconditioner : public Preconditioner<T> {
public:
    /// Computes and owns S^{-1/2}, and keeps what the iteration reported.
    ///
    /// The report is not optional bookkeeping. inverse_sqrt can return a
    /// factor that did not converge -- silicon at 4x4x4 starts outside the
    /// Newton-Schulz convergence radius, trips the divergence guard on step
    /// one, and comes back with ||Z^H S Z - I|| = 3.6e3 against a target of
    /// 1e-7. Passing nullptr here
    /// threw that verdict away, and every consumer then used the factor as if
    /// it were S^{-1/2}. What that costs depends on the caller, so the
    /// judgement belongs to them (see OperatorSet::build_preconditioner):
    /// preconditioning a CG solve, a bad factor is merely slow; defining the
    /// Z^H H Z sandwich, it silently replaces the spectrum -- the same silicon
    /// cell reported spectral bounds of (-737, 9230) eV instead of (-6.3, 208).
    ISQRTMPreconditioner(BlockSpMat<T>& S, int max_iter = 10, double tol = 1e-5,
                         double sptol = 1e-5, bool verbose = false)
        : owned_(std::make_unique<BlockSpMat<T>>(S.graph)), factor_(owned_.get()) {
        InverseSqrtOptions options;
        options.tol = tol;
        options.sptol = sptol;
        options.maxit = max_iter;
        inverse_sqrt(S, owned_.get(), options, verbose, &report_);
    }

    /// Borrows a factor owned elsewhere, which must outlive this.
    explicit ISQRTMPreconditioner(BlockSpMat<T>& factor, BorrowFactor)
        : owned_(nullptr), factor_(&factor) {}

    /// The factor being applied, owned or borrowed.
    BlockSpMat<T>& factor() { return *factor_; }
    const BlockSpMat<T>& factor() const { return *factor_; }

    /// What the Newton-Schulz iteration reported, or a default-constructed
    /// report (converged == false, residuals 0) for a borrowed factor, whose
    /// quality this object never measured.
    const InverseSqrtReport& report() const { return report_; }

    /// True when this object computed the factor itself, so `report()` is a
    /// measurement rather than a placeholder.
    bool owns_factor() const { return owned_ != nullptr; }

    PreconditionerType get_type() const override { return PreconditionerType::Symmetric; }

    void apply(const DistVector<T>& src, DistVector<T>& dst) override {
        factor_->mult_optimized(const_cast<DistVector<T>&>(src), dst);
    }

    void apply(const DistMultiVector<T>& src, DistMultiVector<T>& dst) override {
        factor_->mult_dense(const_cast<DistMultiVector<T>&>(src), dst);
    }

    void apply_adjoint(const DistVector<T>& src, DistVector<T>& dst) override {
        factor_->mult_adjoint(const_cast<DistVector<T>&>(src), dst);
    }

    void apply_adjoint(const DistMultiVector<T>& src, DistMultiVector<T>& dst) override {
        factor_->mult_dense_adjoint(const_cast<DistMultiVector<T>&>(src), dst);
    }

private:
    // Null when borrowing. A BlockSpMat member cannot express "no matrix":
    // it has no default constructor, and constructing one from a graph is not
    // a cheap placeholder -- it allocates and first-touches value pages for the
    // graph's whole sparsity pattern, which on a large overlap is hundreds of
    // megabytes of resident memory that the borrowing case would never read.
    // Indirection also keeps the factor address stable if this object is moved,
    // which a member held by value would not.
    std::unique_ptr<BlockSpMat<T>> owned_;
    BlockSpMat<T>* factor_;  // -> *owned_, or the caller's
    InverseSqrtReport report_{};
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
        P.apply_adjoint(*v_tmp2, y);
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
        P.apply_adjoint(*mv_tmp2, Y);
    }
    

private:
    std::unique_ptr<DistVector<T>> v_tmp1, v_tmp2;
    std::unique_ptr<DistMultiVector<T>> mv_tmp1, mv_tmp2;
};

} // namespace calc
} // namespace rsatb

#endif
