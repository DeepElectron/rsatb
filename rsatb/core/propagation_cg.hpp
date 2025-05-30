// chebyshev_propagation.cpp
//
// Chebyshev time-propagation (serial CPU version).
// but calls Intel oneMKL sparse BLAS for every H·x product.
// And HPDDM for solving S^-1·x where S is the overlap matrix.
// * H must already be scaled so that its spectrum lies in [-1,1].
// * B[k] = J_k(dt) with dt already multiplied by the scaling factor
// * wf_in and wf_out may alias (in-place update is supported).
//
// Compile (ILP64 example):
//   g++ -O3 -std=c++17 chebyshev_propagation.cpp -I${MKLROOT}/include \
//       -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
//       -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
//
// Author: <zhanghao yinzhanghao.zhou@mail.mcgill.ca>
// -----------------------------------------------------------------------------

// #include <pybind11/pybind11.h>
#ifndef PROP_CG
#define PROP_CG

#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#include <mkl_spblas.h>
#include <mkl_cblas.h>
#include <HPDDM.hpp>

#include <optional>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include "functools.hpp"

// namespace py = pybind11;

using cd = std::complex<double>;


/**
 * Chebyshev time-propagation |ψ(t+Δt)⟩ = exp(∓iHΔt) |ψ(t)⟩.
 *
 * @param H      MKL handle created with mkl_sparse_z_create_csr (Hermitian or general)
 * @param descr  matrix_descr describing H (must match the build step)
 * @param B      Bessel–J coefficients, B.size() ≥ 3
 * @param Psi_in  input vector (length n)
 * @param Psi_out output vector (length n) – may alias wf_in
 * @param n      dimension of the Hilbert space (rows of H)
 * @param nrhs   number of RHS vector in the block
 * @param dt     physical time-step already multiplied by the spectral radius
 * @param S      overlap matrix or handler
 * @param forward true  ⇒  propagates e^{-iHΔt};  false ⇒ e^{+iHΔt}
 */

/* ---------- minimal HPDDM operator for MKL CSR --------------- */
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
    int apply(const cd* in, cd* out, const unsigned short& mu = 1, cd* /*work*/ = nullptr, const unsigned short& /*advance*/ = 0) const
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

    int GMV(const cd* in, cd* out, const unsigned short& mu = 1) const {
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

// TODO: add the cholmod solver in mkl.
inline void propagate_cg(const sparse_matrix_t&                        H,
                  const matrix_descr&                           descr,
                  const std::vector<cd>&      B,
                  const std::vector<cd>&                   Psi_in,  // n × nrhs  (ld=n)
                  std::vector<cd>&                         Psi_out,  // n × nrhs  (ld=n)
                  std::size_t                                   n,
                  std::size_t                                   nrhs,
                  OverlapOperator&                              S,
                  MPI_Comm                                      comm,
                  bool                                          forward = true)
{
    HPDDM::Option& opt = *HPDDM::Option::get();
    opt["krylov_method"]    = HPDDM_KRYLOV_METHOD_BCG;
    opt["verbosity"]       =      1;
    opt["max_it"]       =      100;
    opt["tol"]           = 1.0e-9;

    if(n == 0 || nrhs == 0) return;
    if(B.size() < 3)
        throw std::invalid_argument("chebyshev_propagate: need at least three Bessel coefficients");

    const cd  I(0.0, 1.0);
    const cd  img_dt = forward ? I : -I;   //  ±i Δt
    const cd  alpha1 = -img_dt;                   //  -i Δt
    const cd  alpha2 = -2.0 * img_dt;             //  -2i Δt
    const cd  beta0  = cd(0.0, 0.0);

    auto to_mkl = [](const cd& z) { return MKL_Complex16{ z.real(), z.imag() }; };
    const MKL_INT ld = static_cast<MKL_INT>(n);   // leading dimension (row stride)
    const sparse_layout_t layout = SPARSE_LAYOUT_COLUMN_MAJOR;

    MKL_Complex16 alpha1_mkl{alpha1.real(), alpha1.imag()};
    MKL_Complex16 alpha2_mkl{ alpha2.real(), alpha2.imag() };
    MKL_Complex16 beta0_mkl{ beta0.real(), beta0.imag() };

    // — scratch buffers —
    std::vector<cd> T_prev(n * nrhs), T_curr(n * nrhs), T_next(n * nrhs), T_tmp(n * nrhs);
    // T_prev =  -iΔt * H * ψ

    mkl_check(mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                            alpha1_mkl, H, descr,
                            layout,
                            reinterpret_cast<const MKL_Complex16*>(Psi_in.data()),
                            static_cast<MKL_INT>(nrhs), ld,
                            beta0_mkl,
                            reinterpret_cast<MKL_Complex16*>(T_tmp.data()), ld),
            "H·ψ (T_prev)");

    // std::cout << "[DEBUG] print next -iS^-1Hv. " << std::endl;
    HPDDM::IterativeMethod::solve(S, T_tmp.data(), T_prev.data(), static_cast<int>(nrhs), comm);

    // T_curr = -2i * H * T_prev + ψ

    mkl_check(mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                            alpha2_mkl, H, descr,
                            layout,
                            reinterpret_cast<const MKL_Complex16*>(T_prev.data()),
                            static_cast<MKL_INT>(nrhs), ld,
                            beta0_mkl,
                            reinterpret_cast<MKL_Complex16*>(T_tmp.data()), ld),
            "H·T_prev (T_curr)");
            
    HPDDM::IterativeMethod::solve(S, T_tmp.data(), T_curr.data(), static_cast<int>(nrhs), comm);

    const MKL_INT len = static_cast<MKL_INT>(n * nrhs);
    const MKL_Complex16 one = { 1.0, 0.0 };

    cblas_zaxpy(len,                               // vector length
                &one,                              // α = 1
                reinterpret_cast<const MKL_Complex16*>(Psi_in.data()), 1,
                reinterpret_cast<MKL_Complex16*>(T_curr.data()), 1);
    // cblas_zaxpy(len,                               // vector length
    //             &one,                              // α = 1
    //             Psi_in, 1,
    //             T_curr.data(), 1);

    // ---- 3)  Psi_out  =  B0*Psi + 2B1*T_prev + 2B2*T_curr ------------------
    const MKL_Complex16 B0    = { B[0].real(),          B[0].imag()          };
    const MKL_Complex16 twoB1 = { (2.0*B[1]).real(),    (2.0*B[1]).imag()    };
    const MKL_Complex16 twoB2 = { (2.0*B[2]).real(),    (2.0*B[2]).imag()    };

    cblas_zcopy(len,
                reinterpret_cast<const MKL_Complex16*>(Psi_in.data()), 1,
                reinterpret_cast<MKL_Complex16*>(Psi_out.data()),      1);
    cblas_zscal(len, &B0,
                reinterpret_cast<MKL_Complex16*>(Psi_out.data()), 1);
    //  c) Psi_out += 2 B1 T_prev
    cblas_zaxpy(len, &twoB1,
                reinterpret_cast<const MKL_Complex16*>(T_prev.data()), 1,
                reinterpret_cast<MKL_Complex16*>(Psi_out.data()),            1);
    //  d) Psi_out += 2 B2 T_curr
    cblas_zaxpy(len, &twoB2,
                reinterpret_cast<const MKL_Complex16*>(T_curr.data()), 1,
                reinterpret_cast<MKL_Complex16*>(Psi_out.data()),            1);

    // Higher-order Chebyshev recurrence
    
    for(std::size_t m = 3; m < B.size(); ++m)
    {
        // T_next = -2iΔt * H * T_curr + T_prev

        mkl_check(mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                alpha2_mkl, H, descr,
                                layout,
                                reinterpret_cast<const MKL_Complex16*>(T_curr.data()),
                                static_cast<MKL_INT>(nrhs), ld,
                                beta0_mkl,
                                reinterpret_cast<MKL_Complex16*>(T_tmp.data()), ld),
                "iteration mm");
        // std::cout << "[DEBUG] T_tmp: " << T_tmp.data()[0] << std::endl;
        HPDDM::IterativeMethod::solve(S, T_tmp.data(), T_next.data(), static_cast<int>(nrhs), comm);

        const MKL_Complex16 coeff = { (2.0*B[m]).real(), (2.0*B[m]).imag() };

        //  a)  T_next += T_prev   (α = 1)
        cblas_zaxpy(len, &one,
                    reinterpret_cast<const MKL_Complex16*>(T_prev.data()), 1,
                    reinterpret_cast<MKL_Complex16*>(T_next.data()),       1);

        //  b)  Psi_out += coeff · T_next
        cblas_zaxpy(len, &coeff,
                    reinterpret_cast<const MKL_Complex16*>(T_next.data()), 1,
                    reinterpret_cast<MKL_Complex16*>(Psi_out.data()),             1);

        // roll buffers:  prev ← curr, curr ← next
        T_prev.swap(T_curr);
        T_curr.swap(T_next);
    }
}

#endif