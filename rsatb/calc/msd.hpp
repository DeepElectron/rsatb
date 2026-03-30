#ifndef RSATB_CALC_MSD_HPP
#define RSATB_CALC_MSD_HPP

#include "ac.hpp"
#include "kpm.hpp"

#include <array>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace rsatb {
namespace calc {

namespace detail {

template <typename T>
std::vector<cd> average_moment_columns(const std::vector<std::vector<T>>& moments) {
    std::vector<cd> averaged(moments.size(), cd(0.0, 0.0));
    if (moments.empty()) {
        return averaged;
    }

    const size_t width = moments.front().size();
    if (width == 0) {
        return averaged;
    }

    for (size_t m = 0; m < moments.size(); ++m) {
        cd accum = 0.0;
        for (const auto& value : moments[m]) {
            accum += static_cast<cd>(value);
        }
        averaged[m] = accum / static_cast<double>(width);
    }
    return averaged;
}

template <typename T>
std::vector<cd> lift_moments(const std::vector<T>& moments) {
    std::vector<cd> out(moments.size(), cd(0.0, 0.0));
    for (size_t i = 0; i < moments.size(); ++i) {
        out[i] = static_cast<cd>(moments[i]);
    }
    return out;
}

} // namespace detail

template <typename T, typename Kernel, typename VectorType>
void corr_msd(
    Chebyshev<T, Kernel>& cheby,
    BlockSpMat<T, Kernel>& X,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out,
    bool verbose = false
) {
    if (n_steps < 0) {
        throw std::runtime_error("n_steps must be non-negative");
    }
    if (n_kpm <= 0) {
        throw std::runtime_error("n_kpm must be positive");
    }
    if (dt_fs < 0.0) {
        throw std::runtime_error("dt_fs must be non-negative");
    }

    const double hbar_ev_fs = 0.6582119569;
    const double tau = cheby.half_width * dt_fs / hbar_ev_fs;
    std::vector<cd> coeffs;
    timeprop_coeff(tau, 1e-14, true, coeffs);

    std::vector<cd> density_moments;
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        density_moments = detail::average_moment_columns(cheby.compute_moments(psi_in, n_kpm));
    } else {
        density_moments = detail::lift_moments(cheby.compute_moments(psi_in, n_kpm));
    }

    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm), 0.0);

    auto make_like = [&](const VectorType& ref) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            return VectorType(ref.graph, ref.num_vectors);
        } else {
            return VectorType(ref.graph);
        }
    };

    VectorType psi_curr = make_like(psi_in);
    VectorType psi_next = make_like(psi_in);
    VectorType xpsi_curr = make_like(psi_in);
    VectorType xpsi0 = make_like(psi_in);
    VectorType xpsi0_prop = make_like(psi_in);
    VectorType xpsi0_prop_next = make_like(psi_in);
    VectorType comm_state = make_like(psi_in);

    psi_curr.copy_from(psi_in);
    apply_operator_dual(cheby, X, psi_in, xpsi0, ChebyMode::NORMAL, 1.0);
    xpsi0_prop.copy_from(xpsi0);

    for (int step = 0; step <= n_steps; ++step) {
        if (step > 0) {
            cheby.propagate(psi_curr, psi_next, coeffs, ChebyMode::NORMAL);
            psi_curr.copy_from(psi_next);

            cheby.propagate(xpsi0_prop, xpsi0_prop_next, coeffs, ChebyMode::NORMAL);
            xpsi0_prop.copy_from(xpsi0_prop_next);
        }

        if (verbose && psi_in.graph->rank == 0) {
            std::cout << "corr_msd calculation: " << step + 1 << " / " << (n_steps + 1) << "\n";
        }

        apply_operator_dual(cheby, X, psi_curr, xpsi_curr, ChebyMode::NORMAL, 1.0);
        comm_state.copy_from(xpsi_curr);
        comm_state.axpy(-1.0, xpsi0_prop);

        std::vector<cd> msd_moments;
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            msd_moments = detail::average_moment_columns(cheby.compute_moments(comm_state, n_kpm));
        } else {
            msd_moments = detail::lift_moments(cheby.compute_moments(comm_state, n_kpm));
        }

        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);
        for (int k = 0; k < n_kpm; ++k) {
            const double rho = density_spectrum[static_cast<size_t>(k)].real();
            if (std::abs(rho) <= 1e-12) {
                continue;
            }
            msd_out[static_cast<size_t>(step) * static_cast<size_t>(n_kpm) + static_cast<size_t>(k)] =
                msd_spectrum[static_cast<size_t>(k)].real() / rho;
        }
    }
}

} // namespace calc
} // namespace rsatb

#endif
