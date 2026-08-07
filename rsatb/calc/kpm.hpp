#ifndef RSATB_CALC_KPM_HPP
#define RSATB_CALC_KPM_HPP

#include "fft.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
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
// TODO: style here is not unified, the jackson kernel is left outside the switched kernel function?

// Summation rules ("kernels") for truncated Chebyshev expansions.
//
// Jackson turns the truncated series into a positive smearing of width
// sigma ~ pi*half_width/N. That is the right tool for reconstructing a
// spectral function pointwise (no negative DOS, no Gibbs ringing), but for
// spectrally *integrated* observables its bias is one-signed and decays only
// like 1/N^2. The tapers leave the first half of the series untouched (they
// reproduce polynomials up to degree N/2 exactly) and roll off only the
// tail, which preserves the self-cancellation of the oscillatory truncation
// residual against a smooth (local) density of states; on such spectra the
// degree needed for a target observable accuracy is set by the spectral
// regularity near mu, not by beta*W. Background, benchmarks, and the
// validity conditions live in benchmarks/fermi_dirac/METHOD.md.
enum class KpmKernel { None, Jackson, CosTaper, VpTaper };

// Weights g_n, n = 0..n-1, for summing n Chebyshev coefficients.
inline std::vector<double> kpm_kernel_weights(KpmKernel kernel, int n) {
    if (n <= 0) {
        throw std::runtime_error("kpm_kernel_weights: n must be positive");
    }
    std::vector<double> g(static_cast<size_t>(n), 1.0);
    switch (kernel) {
    case KpmKernel::None:
        break;
    case KpmKernel::Jackson:
        jackson_kernel(n, g.data());
        break;
    case KpmKernel::CosTaper: {
        // Raised-cosine (Hann) roll-off over the trailing half: C^1 in n, so
        // the far-from-mu ringing of the effective kernel decays two orders
        // faster than for a bare truncation.
        const int n0 = n / 2;
        for (int i = n0; i < n; ++i) {
            const double s = static_cast<double>(i - n0) / static_cast<double>(n - n0);
            g[static_cast<size_t>(i)] = 0.5 * (1.0 + std::cos(M_PI * s));
        }
        break;
    }
    case KpmKernel::VpTaper: {
        // de la Vallee Poussin style linear roll-off; nearly identical to
        // CosTaper in practice, kept for cross-checks.
        const int n0 = n / 2;
        for (int i = n0; i < n; ++i) {
            g[static_cast<size_t>(i)] = static_cast<double>(n - i) / static_cast<double>(n - n0);
        }
        break;
    }
    }
    return g;
}

// q_g(N) = sum_{n<N} g_n(N) c_n mu_n: the expansion summed in observable
// space against precomputed moments mu_n = <psi|T_n(Op)|psi>, rather than
// applied to a state. The adaptive drivers compare this at several N to
// decide a degree without touching the operator again.
inline double kpm_weighted_sum(const std::vector<cd>& coeffs,
                               const std::vector<double>& moments,
                               int n, KpmKernel kernel) {
    const std::vector<double> g = kpm_kernel_weights(kernel, n);
    double q = 0.0;
    for (int i = 0; i < n; ++i) {
        q += g[static_cast<size_t>(i)] * coeffs[static_cast<size_t>(i)].real() *
             moments[static_cast<size_t>(i)];
    }
    return q;
}

// Maps the user-facing kernel names onto the enum (API boundaries).
inline KpmKernel parse_kpm_kernel(const std::string& name) {
    if (name == "jackson") return KpmKernel::Jackson;
    if (name == "costaper") return KpmKernel::CosTaper;
    if (name == "vptaper") return KpmKernel::VpTaper;
    if (name == "none") return KpmKernel::None;
    throw std::runtime_error(
        "Unknown KPM kernel '" + name + "' (expected jackson, costaper, vptaper, or none)");
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

// Chebyshev coefficients of `func` on [-1, 1] under a summation rule:
//   f(x) ~ sum_{n<n_moments} g_n c_n T_n(x)
// with c_n the exact coefficients and g_n the kernel weights (use
// KpmKernel::None for a bare truncation).
//
// The angular grid is oversampled. A bare 2n-point grid is exact only once
// the coefficients beyond n have decayed -- true for Jackson-length Fermi
// expansions, false for the short tapered ones whose tail still carries
// O(1/n) weight, where aliasing folds that tail back onto the kept
// coefficients. The grid is a power of two >= oversample*n with a 4096 floor;
// the FFT cost is negligible next to a single distributed matvec.
template <typename Func>
std::vector<cd> chebyshev_coefficients_from_function(
    int n_moments,
    Func&& func,
    KpmKernel kernel,
    double trunc_eps = 1e-14,
    int oversample = 8
) {
    if (n_moments <= 0) {
        throw std::runtime_error("n_moments must be positive");
    }

    int m = 4096;
    while (m < n_moments * std::max(2, oversample)) {
        m <<= 1;
    }

    const double dtheta = M_PI / static_cast<double>(m);
    std::vector<cd> samples(2 * static_cast<size_t>(m), cd(0.0, 0.0));
    for (int i = 0; i < 2 * m; ++i) {
        const double x = std::cos(dtheta * static_cast<double>(i));
        samples[static_cast<size_t>(i)] = static_cast<cd>(func(x));
    }
    fft_1d(2 * m, samples.data(), samples.data());

    const std::vector<double> g = kpm_kernel_weights(kernel, n_moments);
    std::vector<cd> coeffs(static_cast<size_t>(n_moments));
    for (int i = 0; i < n_moments; ++i) {
        coeffs[static_cast<size_t>(i)] =
            samples[static_cast<size_t>(i)].real() * g[static_cast<size_t>(i)] / static_cast<double>(m);
    }
    coeffs[0] *= 0.5;
    if (trunc_eps > 0.0) {
        truncate_chebyshev_coefficients(coeffs, trunc_eps);
    }
    return coeffs;
}

// Kernel-smoothed spectral density at arbitrary points x in (-1, 1) from
// precomputed trace moments mu_n = <psi|T_n(Op)|psi>:
//   rho(x) = (g_0 mu_0 + 2 sum_{n>=1} g_n mu_n T_n(x)) / (pi sqrt(1 - x^2))
// normalised so that the integral over [-1, 1] equals mu_0. Imaginary parts
// of the moments are dropped: for a Hermitian operator they are pure
// stochastic-estimator noise. The Chebyshev weight diverges at the interval
// ends, so points are pulled into the open interval by `edge_inset`; a
// uniform grid that touches -1 (the DOS grid does) gets the half-bin-inset
// value there instead of a 1/sqrt spike.
template <typename ScalarT>
std::vector<double> evaluate_kpm_density(
    const std::vector<ScalarT>& moments,
    const std::vector<double>& x_values,
    double edge_inset,
    KpmKernel kernel = KpmKernel::Jackson
) {
    const int n_moments = static_cast<int>(moments.size());
    std::vector<double> rho(x_values.size(), 0.0);
    if (n_moments == 0) {
        return rho;
    }
    const std::vector<double> g = kpm_kernel_weights(kernel, n_moments);

    for (size_t j = 0; j < x_values.size(); ++j) {
        const double x = std::max(-1.0 + edge_inset,
                                  std::min(1.0 - edge_inset, x_values[j]));
        // T_n(x) by the recurrence rather than cos(n acos x): one multiply
        // per term and exact at the clamped ends.
        double t_prev = 1.0;
        double t_curr = x;
        double accum = g[0] * static_cast<cd>(moments[0]).real();
        if (n_moments > 1) {
            accum += 2.0 * g[1] * static_cast<cd>(moments[1]).real() * t_curr;
        }
        for (int n = 2; n < n_moments; ++n) {
            const double t_next = 2.0 * x * t_curr - t_prev;
            accum += 2.0 * g[static_cast<size_t>(n)] *
                     static_cast<cd>(moments[static_cast<size_t>(n)]).real() * t_next;
            t_prev = t_curr;
            t_curr = t_next;
        }
        rho[j] = accum / (M_PI * std::sqrt(std::max(1e-14, 1.0 - x * x)));
    }
    return rho;
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
