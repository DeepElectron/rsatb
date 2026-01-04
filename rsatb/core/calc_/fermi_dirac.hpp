#ifndef RSATB_CALC_FERMI_DIRAC_HPP
#define RSATB_CALC_FERMI_DIRAC_HPP

#include "cheby.hpp"
#include "kpm.hpp"
#include "fft.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>

namespace rsatb {
namespace calc {

inline void fermi_dirac_func(double E, double beta, double mu, double *out){
    double tmp = 0.0;
    if (E >= mu) {
        tmp = std::exp(beta * (mu - E));
        *out = tmp / (1.0 + tmp);
    } else {
        tmp = std::exp(beta * (E - mu));
        *out = 1.0 / (1.0 + tmp);
    }
}

// Reimplementation of fermi_dirac_kpm using new backend
template<typename T, typename Kernel, typename VectorType>
void fermi_dirac_kpm(Chebyshev<T, Kernel>& cheby, 
                     const VectorType& Psi_in, 
                     VectorType& Psi_out, 
                     double temp, 
                     double mu, 
                     int n_kpm, 
                     double eps = 1e-14, 
                     ChebyMode mode = ChebyMode::NORMAL) {
    
    // Scale parameters
    // H_scaled = (H - center) / half_width
    // We need to map physical energy E to x in [-1, 1]
    // x = (E - center) / half_width
    // E = x * half_width + center
    
    // Original code parameters:
    // double mu_re = (mu - center) / scale;
    // double beta_re = scale / (temp * 8.617333262e-5);
    // double shift_re = center / (temp * 8.617333262e-5);
    // Here scale corresponds to half_width, and center corresponds to shift.
    
    double scale = cheby.half_width;
    double center = cheby.center;
    
    // Adjust mu and beta for scaled Hamiltonian
    // f(E) = 1 / (exp(beta*(E-mu)) + 1)
    // E = x * scale + center
    // f(x) = 1 / (exp(beta*(x*scale + center - mu)) + 1)
    //      = 1 / (exp( (beta*scale) * (x - (mu-center)/scale) ) + 1)
    // So effective beta' = beta * scale
    // Effective mu' = (mu - center) / scale
    
    double kb = 8.617333262e-5; // eV/K
    double beta = 1.0 / (temp * kb);
    
    double beta_re = beta * scale;
    double mu_re = (mu - center) / scale;
    
    // Compute coefficients using FFT
    double dE = M_PI / n_kpm;
    std::vector<std::complex<double>> tmp_coeff(2 * n_kpm);
    double fermi_tmp;
    
    for (int i = 0; i < 2 * n_kpm; ++i) {
        // x_k = cos(theta_k)
        // We sample at Chebyshev nodes?
        // Original code: fermi_dirac(std::cos(dE*i), beta_re, mu_re, &fermi_tmp);
        // This samples f(cos(theta)) at uniform theta.
        fermi_dirac_func(std::cos(dE * i), beta_re, mu_re, &fermi_tmp);
        tmp_coeff[i] = fermi_tmp;
    }

    std::vector<double> kernel(n_kpm);
    jackson_kernel(n_kpm, kernel.data());

    fft_1d(2 * n_kpm, tmp_coeff.data(), tmp_coeff.data()); 
    
    // Keep first n_kpm moments and apply kernel
    for (int i = 0; i < n_kpm; ++i) {
        tmp_coeff[i] = tmp_coeff[i].real() * kernel[i] / n_kpm;
    }

    tmp_coeff[0] *= 0.5;

    // Truncate
    int n_kpm_ = n_kpm;
    for (int i = 0; i < n_kpm - 1; ++i) {
        if ((std::abs(tmp_coeff[i]) <= eps) && (std::abs(tmp_coeff[i + 1]) <= eps)) {
            n_kpm_ = i;
            break; 
        }
    }
    
    // Ensure n_kpm_ is at least 1
    if (n_kpm_ < 1) n_kpm_ = 1;

    // std::cout << "fermi-dirac calculation: #KPM moments - " << n_kpm_ << "\n";

    // Propagate
    // y = sum c_n T_n(H) x
    // cheby.propagate expects coefficients for T_n.
    // tmp_coeff holds exactly that.
    
    // Resize tmp_coeff to actual used size
    tmp_coeff.resize(n_kpm_);
    
    cheby.propagate(Psi_in, Psi_out, tmp_coeff, mode);
}

} // namespace calc
} // namespace rsatb

#endif
