#ifndef PROP
#define PROP

#include <mkl.h>
#include <optional>
#include <complex>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include "functools.hpp"

// namespace py = pybind11;

using cd = std::complex<double>;

// This is the method that does not include overlap, used to be a prototype for benchmarking.
inline void propagate(const sparse_matrix_t&                           H,
                  const matrix_descr&                           descr,
                  const std::vector<std::complex<double>>&      B,
                  const std::vector<cd>&                        Psi_in,  // n × nrhs  (ld=n)
                  std::vector<cd>&                              Psi_out,  // n × nrhs  (ld=n)
                  std::size_t                                   n,
                  std::size_t                                   nrhs,
                  bool                                          forward = true)
{

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
    std::vector<cd> T_prev(n * nrhs), T_curr(n * nrhs), T_next(n * nrhs);

    // T_prev =  -iΔt * H * ψ
    mkl_check(mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                              alpha1_mkl, H, descr,
                              layout,
                              reinterpret_cast<const MKL_Complex16*>(Psi_in.data()),
                              static_cast<MKL_INT>(nrhs), ld,
                              beta0_mkl,
                              reinterpret_cast<MKL_Complex16*>(T_prev.data()), ld),
               "H·ψ (T_prev)");

    // T_curr = -2iΔt * H * T_prev + ψ
    mkl_check(mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                              alpha2_mkl, H, descr,
                              layout,
                              reinterpret_cast<const MKL_Complex16*>(T_prev.data()),
                              static_cast<MKL_INT>(nrhs), ld,
                              beta0_mkl,
                              reinterpret_cast<MKL_Complex16*>(T_curr.data()), ld),
               "H·T_prev (T_curr)");

    const MKL_INT len = static_cast<MKL_INT>(n * nrhs);
    const MKL_Complex16 one = { 1.0, 0.0 };

    cblas_zaxpy(len,                               // vector length
                &one,                              // α = 1
                reinterpret_cast<const MKL_Complex16*>(Psi_in.data()), 1,
                reinterpret_cast<MKL_Complex16*>(T_curr.data()), 1);

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
                                  reinterpret_cast<MKL_Complex16*>(T_next.data()), ld),
                   "iteration mm");

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