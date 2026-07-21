#ifndef RSATB_CALC_FERMI_DIRAC_HPP
#define RSATB_CALC_FERMI_DIRAC_HPP

#include "cheby.hpp"
#include "kpm.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>

using namespace vbcsr;

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

inline double occupation_projector_func(double E, double beta, double mu) {
    if (!std::isfinite(beta) || beta <= 0.0) {
        return E <= mu ? 1.0 : 0.0;
    }

    double out = 0.0;
    fermi_dirac_func(E, beta, mu, &out);
    return out;
}

// Reimplementation of fermi_dirac_kpm using new backend
template<typename T, typename VectorType>
void fermi_dirac_kpm(Chebyshev<T>& cheby, 
                     const VectorType& Psi_in, 
                     VectorType& Psi_out, 
                     double temp, 
                     double mu, 
                     int n_kpm, 
                     double eps = 1e-14, 
                     ChebyMode mode = ChebyMode::NORMAL,
                     bool verbose=false) {
    
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
    int rank = Psi_in.graph->rank;
    
    // Adjust mu and beta for scaled Hamiltonian
    // f(E) = 1 / (exp(beta*(E-mu)) + 1)
    // E = x * scale + center
    // f(x) = 1 / (exp(beta*(x*scale + center - mu)) + 1)
    //      = 1 / (exp( (beta*scale) * (x - (mu-center)/scale) ) + 1)
    // So effective beta' = beta * scale
    // Effective mu' = (mu - center) / scale
    
    double kb = 8.617333262e-5; // eV/K
    double beta = (temp > 0.0) ? (1.0 / (temp * kb)) : 0.0;
    
    double beta_re = beta * scale;
    double mu_re = (mu - center) / scale;
    
    const auto tmp_coeff = chebyshev_coefficients_from_function(
        n_kpm,
        [&](double x) -> double {
            return occupation_projector_func(x, beta_re, mu_re);
        },
        true,
        eps
    );

    if (verbose){
        if (rank == 0){
            std::cout << "fermi-dirac calculation: #KPM moments - " << tmp_coeff.size() << "\n";
        }
    }

    cheby.propagate(Psi_in, Psi_out, tmp_coeff, mode);
}

} // namespace calc
} // namespace rsatb

#endif
