#ifndef RSATB_CALC_KPM_HPP
#define RSATB_CALC_KPM_HPP

#include "fft.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace rsatb {
namespace calc {

// Jackson Kernel for KPM smoothing
// g_n = ((N - n + 1) * cos(pi * n / (N + 1)) + sin(pi * n / (N + 1)) * cot(pi / (N + 1))) / (N + 1)
inline void jackson_kernel(int n, double *out) {
    auto denom = static_cast<double>(n + 1);
    double theta = M_PI / denom;
    double cot_theta = std::cos(theta) / std::sin(theta);
    out[0] = 1.0;
    for (int i = 1; i < n; ++i) {
        out[i] = ((denom - i) * std::cos(theta * i) + cot_theta * std::sin(theta * i)) / denom;
    }
}

// Vector version
inline std::vector<double> jackson_kernel(int n) {
    std::vector<double> out(n);
    jackson_kernel(n, out.data());
    return out;
}

inline std::vector<double> chebyshev_node_grid(int n, bool ascending = false) {
    std::vector<double> nodes(n, 0.0);
    for (int k = 0; k < n; ++k) {
        const int src = ascending ? (n - 1 - k) : k;
        const double theta = M_PI * (static_cast<double>(src) + 0.5) / static_cast<double>(n);
        nodes[k] = std::cos(theta);
    }
    return nodes;
}

inline std::vector<double> chebyshev_energy_grid(double center, double half_width, int n, bool ascending = false) {
    std::vector<double> energy(n, 0.0);
    const std::vector<double> nodes = chebyshev_node_grid(n, ascending);
    for (int k = 0; k < n; ++k) {
        energy[k] = center + half_width * nodes[k];
    }
    return energy;
}

inline void truncate_chebyshev_coefficients(std::vector<cd>& coeffs, double eps = 1e-14) {
    if (coeffs.empty()) {
        return;
    }

    size_t keep = coeffs.size();
    for (size_t i = 0; i + 1 < coeffs.size(); ++i) {
        if (std::abs(coeffs[i]) <= eps && std::abs(coeffs[i + 1]) <= eps) {
            keep = i + 1;
            break;
        }
    }
    keep = std::max<size_t>(1, keep);
    coeffs.resize(keep);
}

template <typename Func>
std::vector<cd> chebyshev_coefficients_from_function(
    int n_moments,
    Func&& func,
    bool apply_jackson_kernel = true,
    double trunc_eps = 1e-14
) {
    if (n_moments <= 0) {
        throw std::runtime_error("n_moments must be positive");
    }

    const double dtheta = M_PI / static_cast<double>(n_moments);
    std::vector<cd> coeffs(2 * static_cast<size_t>(n_moments), cd(0.0, 0.0));
    for (int i = 0; i < 2 * n_moments; ++i) {
        const double x = std::cos(dtheta * static_cast<double>(i));
        coeffs[static_cast<size_t>(i)] = static_cast<cd>(func(x));
    }

    fft_1d(2 * n_moments, coeffs.data(), coeffs.data());

    std::vector<double> kernel;
    if (apply_jackson_kernel) {
        kernel = jackson_kernel(n_moments);
    }

    coeffs.resize(static_cast<size_t>(n_moments));
    for (int i = 0; i < n_moments; ++i) {
        const double kernel_value = apply_jackson_kernel ? kernel[static_cast<size_t>(i)] : 1.0;
        coeffs[static_cast<size_t>(i)] = (coeffs[static_cast<size_t>(i)].real() * kernel_value) /
                                         static_cast<double>(n_moments);
    }
    coeffs[0] *= 0.5;
    truncate_chebyshev_coefficients(coeffs, trunc_eps);
    return coeffs;
}

template <typename ScalarT>
void reconstruct_spectrum_from_moments(
    const std::vector<ScalarT>& moments,
    std::vector<cd>& spectrum_out,
    std::vector<double>* eps_out = nullptr,
    bool ascending = true
) {
    const int n_moments = static_cast<int>(moments.size());
    spectrum_out.assign(static_cast<size_t>(n_moments), cd(0.0, 0.0));
    if (n_moments == 0) {
        if (eps_out) {
            eps_out->clear();
        }
        return;
    }

    const std::vector<double> eps = chebyshev_node_grid(n_moments, ascending);
    const std::vector<double> kernel = jackson_kernel(n_moments);
    if (eps_out) {
        *eps_out = eps;
    }

    for (int k = 0; k < n_moments; ++k) {
        const double theta = std::acos(std::max(-1.0, std::min(1.0, eps[static_cast<size_t>(k)])));
        const double root = std::max(1e-14, std::sin(theta));

        cd accum = kernel[0] * static_cast<cd>(moments[0]);
        for (int m = 1; m < n_moments; ++m) {
            accum += 2.0 * kernel[static_cast<size_t>(m)] * static_cast<cd>(moments[static_cast<size_t>(m)]) *
                     std::cos(static_cast<double>(m) * theta);
        }
        spectrum_out[static_cast<size_t>(k)] = accum / (M_PI * root);
    }
}

} // namespace calc
} // namespace rsatb

#endif
