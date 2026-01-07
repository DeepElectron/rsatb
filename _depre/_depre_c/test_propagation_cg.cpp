// test_driver.cpp -----------------------------------------------------------
#include <HPDDM.hpp>
#include <mkl_spblas.h>
#include <mkl_cblas.h>
#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

using cd = std::complex<double>;
// ---------------------------------------------------------------------------
// Prototype of the routine from propagation.o

struct OverlapOperator:
        public HPDDM::CustomOperator<HPDDM::MatrixCSR<cd>, cd>
{
    using base = HPDDM::CustomOperator<HPDDM::MatrixCSR<cd>, cd>;
    // store the MKL descriptor
    matrix_descr _descr;
    sparse_matrix_t _mklHandle;

    explicit OverlapOperator(const HPDDM::MatrixCSR<cd>* A, sparse_matrix_t Amkl, const matrix_descr& descr) : base(A), _descr(descr), _mklHandle(Amkl) {}

    /*  y = S · x  -------------------------------------------------
        HPDDM calls this with    mu   right-hand sides packed
        column-major.  We reuse MKL sparse BLAS.
    ----------------------------------------------------------------*/
    template<bool excluded>
    int apply(const cd* in, cd* out,
              const unsigned short& mu = 1,
              cd* /*work*/ = nullptr,
              const unsigned short& /*advance*/ = 0) const
    {
        const auto* A = this->getMatrix();                // CSR owned by HPDDM
        
        MKL_INT n     = static_cast<MKL_INT>(A->n_);
        MKL_INT ld    = n;
        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta  = {0.0, 0.0};

        mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        /* reinterpret_cast because HPDDM::MatrixCSR<cd>
                           stores the MKL handle in A->matrix_             */
                        _mklHandle,
                        /* dummy descr — it is stored inside MKL handle   */
                        _descr,
                        SPARSE_LAYOUT_COLUMN_MAJOR,
                        reinterpret_cast<const MKL_Complex16*>(in),
                        static_cast<MKL_INT>(mu),
                        ld,
                        beta,
                        reinterpret_cast<MKL_Complex16*>(out),
                        ld);
        
        return 0;
    }
};

void propagate_cg(const sparse_matrix_t&,
                  const matrix_descr&,
                  const std::vector<std::complex<double>>&,
                  const std::complex<double>*,
                  std::complex<double>*,
                  std::size_t n,
                  std::size_t nrhs,
                  OverlapOperator& S,
                  MPI_Comm comm,
                  bool        forward = true);
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::cout << "[DEBUG] First line of test script " << std::endl;

    std::cout << sizeof(MKL_INT) << std::endl;
    int provided;
    MPI_Init(&argc, &argv);
    HPDDM::Option& opt = *HPDDM::Option::get();
    opt["verbosity"]       =      3;
    opt["max_it"]       =      100;
    opt["tol"]           = 1.0e-10;

    const std::size_t n    = 2;     // matrix dimension
    const std::size_t nrhs = 2;     // two RHS columns
    const double      dt   = 0.3;   // physical Δt   (already scaled)

    // ----- 1) build H as CSR ----------------------------------------------
    MKL_INT rows[3]    = {0, 1, 2};         // row pointers
    MKL_INT cols[2]    = {1, 0};      // column indices
    cd      vals[2]    = {1.0, 1.0};

    
    MKL_INT rows_r[3]    = {0, 1, 2};         // row pointers
    MKL_INT cols_r[2]    = {0, 1};      // column indices
    cd      vals_r[2]    = {2.0, 2.0};

    std::cout << "[DEBUG] construction of CSR MKL " << std::endl;

    sparse_matrix_t H_csr = nullptr;
    mkl_sparse_z_create_csr(&H_csr, SPARSE_INDEX_BASE_ZERO,
                            n, n, rows, rows + 1, cols,
                            reinterpret_cast<MKL_Complex16*>(vals));
    
    sparse_status_t status;
    sparse_matrix_t S_csr = nullptr;
    status = mkl_sparse_z_create_csr(&S_csr, SPARSE_INDEX_BASE_ZERO,
                            n, n, rows_r, rows_r + 1, cols_r,
                            reinterpret_cast<MKL_Complex16*>(vals_r));
    if (status != SPARSE_STATUS_SUCCESS || !S_csr)
        throw std::runtime_error("[ERROR] MKL failed to create S_csr matrix!");
    MKL_INT n_, m_;
    sparse_index_base_t indexBase;
    MKL_INT *ia_, *ja_;
    MKL_INT *iae_;
    MKL_Complex16 *val_;

    std::cout << "[DEBUG] n = " << n << std::endl;
    std::cout << "rows_r = {";
    for (int i = 0; i <= n; ++i) std::cout << rows_r[i] << " ";
    std::cout << "}, cols_r = {";
    for (int i = 0; i < rows_r[n]; ++i) std::cout << cols_r[i] << " ";
    std::cout << "}, vals_r = {";
    for (int i = 0; i < rows_r[n]; ++i) std::cout << vals_r[i] << " ";
    std::cout << "}" << std::endl;


    std::cout << "[DEBUG] extract data from S_csr MKL " << std::endl;
    status = mkl_sparse_z_export_csr(S_csr, &indexBase,
        &n_, &m_,
        &ia_, /*row_start*/
        &iae_, /*row_end ignored*/
        &ja_, &val_);

    if (status != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("[ERROR] MKL failed to export S_csr matrix!");

    std::cout << "[DEBUG] construction of overlap CSR HPDDM " << std::endl;

    std::cout << "n_ = " << n_ << ", m_ = " << m_ << ", indexBase = " << indexBase << std::endl;
    std::cout << "ia_ = " << ia_ << ", ja_ = " << ja_ << ", val_ = " << val_ << std::endl;
    std::cout << "ia_[n_] = " << ia_[n_] << std::endl;

    if (!ia_)  throw std::runtime_error("ia_ is null!");
    if (!ja_)  throw std::runtime_error("ja_ is null!");
    if (!val_) throw std::runtime_error("val_ is null!");

    HPDDM::MatrixCSR<cd> S_;
    S_ = HPDDM::MatrixCSR<cd>(static_cast<int>(n_), static_cast<int>(m_), static_cast<int>(ia_[n_]-indexBase),
                        reinterpret_cast<cd*>(val_), reinterpret_cast<int*>(ia_), reinterpret_cast<int*>(ja_),
                        /*sym=*/false,
                        /*takeOwnership=*/false);
    
    std::cout << "[DEBUG] get matrix description " << std::endl;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    matrix_descr ovp_descr;
    ovp_descr.diag = SPARSE_DIAG_NON_UNIT;
    ovp_descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    ovp_descr.mode = SPARSE_FILL_MODE_FULL;

    std::cout << "[DEBUG] construction of overlap Operator " << std::endl;

    OverlapOperator op(&S_, S_csr, ovp_descr);

    // ----- 2) initial block of states --------------------------------------
    std::vector<cd> Psi_in  = { 1.0, 0.0,   // ψ¹
                                0.0, 1.0 }; // ψ²
    std::vector<cd> Psi_out(n * nrhs);

    // ----- 3) Bessel table --------------------------------------------------
    const std::size_t m_max = 5;
    std::vector<cd> B(m_max + 1);
    for(std::size_t k = 0; k <= m_max; ++k){
        B[k] = std::cyl_bessel_j(k, dt);   // J_k(dt)
        std::cout << B[k] << ' ';
    }
    std::cout << "\n\n";

    // ----- 4) propagate -----------------------------------------------------
    propagate_cg(H_csr, descr, B,
                 Psi_in.data(), Psi_out.data(),
                 n, nrhs,
                 op, MPI_COMM_SELF, /*forward=*/true);
    

    // ----- 5) dump results --------------------------------------------------
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial Ψ:\n";
    for(std::size_t j = 0; j < nrhs; ++j) {
        std::cout << "  column " << j << ": ";
        for(std::size_t i = 0; i < n; ++i)
            std::cout << Psi_in[i + j*n] << ' ';
        std::cout << '\n';
    }

    std::cout << "Propagated Ψ:\n";
    for(std::size_t j = 0; j < nrhs; ++j) {
        std::cout << "  column " << j << ": ";
        for(std::size_t i = 0; i < n; ++i)
            std::cout << Psi_out[i + j*n] << ' ';
        std::cout << '\n';
    }

    mkl_sparse_destroy(H_csr);
    mkl_sparse_destroy(S_csr);

    MPI_Finalize();

    return 0;
}
