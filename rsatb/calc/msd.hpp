#ifndef RSATB_CALC_MSD_HPP
#define RSATB_CALC_MSD_HPP

#include <algorithm>
#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include "ac.hpp"
#include "kpm.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rsatb {
namespace calc {

namespace detail {

// ZeroLike(...) and RealDot(...) come from calc/vector_utils.hpp.

template <typename VectorType>
struct GammaAffineState {
    // Translation-covariant states along one periodic axis are represented as
    // u_R = a + R b, where both a and b are cell-periodic vectors. `affine`
    // stores the periodic remainder a, and `periodic` stores the coefficient b
    // of the cell-linear term.
    VectorType affine;
    VectorType periodic;
};

template <typename T, typename VectorType>
std::vector<cd> compute_density_moments(
    Chebyshev<T>& cheby,
    const VectorType& psi_in,
    int n_kpm
) {
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        const auto moments = cheby.compute_moments(psi_in, n_kpm);
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
    } else {
        const auto moments = cheby.compute_moments(psi_in, n_kpm);
        std::vector<cd> lifted(moments.size(), cd(0.0, 0.0));
        for (size_t i = 0; i < moments.size(); ++i) {
            lifted[i] = static_cast<cd>(moments[i]);
        }
        return lifted;
    }
}

inline void write_msd_row(
    const std::vector<cd>& density_spectrum,
    const std::vector<cd>& msd_spectrum,
    int step,
    int n_kpm,
    std::vector<double>& msd_out // [nt, n_kpm]
) {
    for (int k = 0; k < n_kpm; ++k) {
        const double rho = density_spectrum[static_cast<size_t>(k)].real();
        if (std::abs(rho) <= 1e-12) {
            continue;
        }
        msd_out[static_cast<size_t>(step) * static_cast<size_t>(n_kpm) + static_cast<size_t>(k)] =
            msd_spectrum[static_cast<size_t>(k)].real() / rho;
    }
}

template <typename T, typename VectorType>
void propagate_affine(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& H1,
    const GammaAffineState<VectorType>& in,
    GammaAffineState<VectorType>& out,
    const std::vector<std::complex<double>>& coeffs
) {
    const int n_coeff = static_cast<int>(coeffs.size());
    if (n_coeff == 0) {
        return;
    }

    GammaAffineState<VectorType> t0{ZeroLike(in.affine), ZeroLike(in.affine)};
    GammaAffineState<VectorType> t1{ZeroLike(in.affine), ZeroLike(in.affine)};
    GammaAffineState<VectorType> t2{ZeroLike(in.affine), ZeroLike(in.affine)};
    VectorType h1_periodic = ZeroLike(in.affine);

    auto apply_scaled_op = [&](const GammaAffineState<VectorType>& state_in, GammaAffineState<VectorType>& state_out) {
        // The affine Gamma-point generator is block-upper-triangular:
        //   [A0  B1]
        //   [0   A0]
        // where A0 is the scaled Hamiltonian and B1 is the scaled first-moment coupling.
        //
        // For a translation-covariant state u_R = a + R b, the unscaled Hamiltonian acts as
        //   H u_R = (H0 a + H1 b) + R (H0 b),
        // so in the affine (a, b) representation
        //   H_aff = [H0  H1; 0  H0].
        // After Chebyshev centering/scaling, H_scaled = (H - center I) / half_width, hence
        //   H_aff,scaled = [(H0 - center I)/half_width   H1/half_width;
        //                   0                            (H0 - center I)/half_width].
        // The center shift is an identity term, so it remains on the diagonal blocks and
        // does not modify the upper-right H1 coupling.
        cheby.apply_op(state_in.affine, state_out.affine, ChebyMode::NORMAL);
        cheby.apply_op(state_in.periodic, state_out.periodic, ChebyMode::NORMAL);

        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            H1.mult_dense(const_cast<VectorType&>(state_in.periodic), h1_periodic);
        } else {
            H1.mult(const_cast<VectorType&>(state_in.periodic), h1_periodic);
        }
        h1_periodic.scale(static_cast<T>(1.0 / cheby.half_width));
        state_out.affine.axpy(static_cast<T>(1.0), h1_periodic);
    };

    t0.affine.copy_from(in.affine);
    t0.periodic.copy_from(in.periodic);
    out.affine.copy_from(t0.affine);
    out.periodic.copy_from(t0.periodic);
    out.affine.scale(static_cast<T>(coeffs[0]));
    out.periodic.scale(static_cast<T>(coeffs[0]));

    if (n_coeff > 1) {
        apply_scaled_op(t0, t1);
        out.affine.axpy(static_cast<T>(coeffs[1]), t1.affine);
        out.periodic.axpy(static_cast<T>(coeffs[1]), t1.periodic);
    }

    for (int n = 2; n < n_coeff; ++n) {
        apply_scaled_op(t1, t2);
        t2.affine.axpby(static_cast<T>(-1.0), t0.affine, static_cast<T>(2.0));
        t2.periodic.axpby(static_cast<T>(-1.0), t0.periodic, static_cast<T>(2.0));
        out.affine.axpy(static_cast<T>(coeffs[n]), t2.affine);
        out.periodic.axpy(static_cast<T>(coeffs[n]), t2.periodic);
        std::swap(t0, t1);
        std::swap(t1, t2);
    }
}

template <typename T, typename PositionOp, typename VectorType>
void run_explicit_position_msd(
    Chebyshev<T>& cheby,
    PositionOp& position,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
) {
    // This helper backs both explicit-position entry points below. The route is
    // already decided by overload selection; the only difference here is whether
    // X is applied as a full matrix or as an orthogonal diagonal.
    const double hbar_ev_fs = 0.6582119569;
    const double tau = cheby.half_width * dt_fs / hbar_ev_fs;
    std::vector<cd> coeffs;
    timeprop_coeff(tau, 1e-14, true, coeffs);

    std::vector<cd> density_moments = compute_density_moments(cheby, psi_in, n_kpm);
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm), 0.0);

    auto apply_position = [&](const VectorType& psi_src, VectorType& xpsi_out) {
        if constexpr (std::is_same_v<PositionOp, DistVector<T>>) {
            xpsi_out.copy_from(psi_src);
            xpsi_out.pointwise_mult(position);
        } else {
            apply_operator_dual(cheby, position, psi_src, xpsi_out, ChebyMode::NORMAL, 1.0);
        }
    };

    VectorType psi_curr = ZeroLike(psi_in);
    VectorType psi_next = ZeroLike(psi_in);
    VectorType xpsi_curr = ZeroLike(psi_in);
    VectorType xpsi0 = ZeroLike(psi_in);
    VectorType xpsi0_prop = ZeroLike(psi_in);
    VectorType xpsi0_prop_next = ZeroLike(psi_in);
    VectorType comm_state = ZeroLike(psi_in);

    psi_curr.copy_from(psi_in);
    apply_position(psi_in, xpsi0);
    xpsi0_prop.copy_from(xpsi0);

    for (int step = 0; step <= n_steps; ++step) {
        ::rsatb::util::throw_if_interrupted(psi_in.graph->comm);
        if (step > 0) {
            cheby.propagate(psi_curr, psi_next, coeffs, ChebyMode::NORMAL);
            psi_curr.copy_from(psi_next);

            cheby.propagate(xpsi0_prop, xpsi0_prop_next, coeffs, ChebyMode::NORMAL);
            xpsi0_prop.copy_from(xpsi0_prop_next);
        }

        if (psi_in.graph->rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
            if (::rsatb::util::every_nth(step, std::max(1, (n_steps + 1) / 20)))
                std::cout << "msd: step " << step + 1 << " / " << (n_steps + 1) << std::endl;
        }

        apply_position(psi_curr, xpsi_curr);
        comm_state.copy_from(xpsi_curr);
        comm_state.axpy(static_cast<T>(-1.0), xpsi0_prop);

        std::vector<cd> msd_moments = compute_density_moments(cheby, comm_state, n_kpm);
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);
        write_msd_row(density_spectrum, msd_spectrum, step, n_kpm, msd_out);
    }
}

template <typename T, typename VectorType>
void run_gamma_affine_msd(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& H1,
    DistVector<T>& diagonal_position,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
) {
    // This is the periodic/Gamma route. The caller supplies the first-moment
    // coupling H1 and the cell-periodic diagonal part of X, and we propagate the
    // translation-covariant affine representation instead of an explicit X matrix.
    const double hbar_ev_fs = 0.6582119569;
    const double tau = cheby.half_width * dt_fs / hbar_ev_fs;
    std::vector<cd> coeffs;
    timeprop_coeff(tau, 1e-14, true, coeffs);

    std::vector<cd> density_moments = compute_density_moments(cheby, psi_in, n_kpm);
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm), 0.0);

    GammaAffineState<VectorType> state_curr{ZeroLike(psi_in), ZeroLike(psi_in)};
    GammaAffineState<VectorType> state_next{ZeroLike(psi_in), ZeroLike(psi_in)};
    VectorType xpsi_curr = ZeroLike(psi_in);
    VectorType comm_state = ZeroLike(psi_in);

    state_curr.periodic.copy_from(psi_in);
    state_curr.affine.copy_from(psi_in);
    state_curr.affine.pointwise_mult(diagonal_position);

    for (int step = 0; step <= n_steps; ++step) {
        ::rsatb::util::throw_if_interrupted(psi_in.graph->comm);
        if (step > 0) {
            propagate_affine(cheby, H1, state_curr, state_next, coeffs);
            state_curr.affine.copy_from(state_next.affine);
            state_curr.periodic.copy_from(state_next.periodic);
        }

        if (psi_in.graph->rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
            if (::rsatb::util::every_nth(step, std::max(1, (n_steps + 1) / 20)))
                std::cout << "msd: step " << step + 1 << " / " << (n_steps + 1) << std::endl;
        }

        // The affine branch stores U(t)X|psi> as a(t) + R b(t). In the commutator
        // XU(t)|psi> - U(t)X|psi>, the cell-linear R b(t) part cancels identically,
        // leaving only the periodic remainder X0 b(t) - a(t).
        xpsi_curr.copy_from(state_curr.periodic);
        xpsi_curr.pointwise_mult(diagonal_position);
        comm_state.copy_from(xpsi_curr);
        comm_state.axpy(static_cast<T>(-1.0), state_curr.affine);

        std::vector<cd> msd_moments = compute_density_moments(cheby, comm_state, n_kpm);
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);
        write_msd_row(density_spectrum, msd_spectrum, step, n_kpm, msd_out);
    }
}

} // namespace detail

// Route selection is done by overload, not by a runtime mode switch in this file:
//   corr_msd(cheby, X_matrix, ...)         -> explicit position matrix route
//   corr_msd(cheby, x_diagonal, ...)       -> explicit orthogonal diagonal route
//   corr_msd(cheby, H1, x_diagonal, ...)   -> periodic Gamma affine route
// The caller decides once which data is available and then invokes the matching
// overload directly.


// open-boundary condition MSD with an explicit position operator, for general systems (orthogonal or non-orthogonal).
template <typename T, typename VectorType>
void corr_msd(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& X,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
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

    detail::run_explicit_position_msd(
        cheby,
        X,
        psi_in,
        n_steps,
        dt_fs,
        n_kpm,
        msd_out,
        energy_out);
}

// open-boundary condition MSD with an explicit position operator, for orthogonal systems where only the diagonal of X is needed.
template <typename T, typename VectorType>
void corr_msd(
    Chebyshev<T>& cheby,
    DistVector<T>& diagonal_position,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
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

    detail::run_explicit_position_msd(
        cheby,
        diagonal_position,
        psi_in,
        n_steps,
        dt_fs,
        n_kpm,
        msd_out,
        energy_out);
}

// periodic boundary condition MSD for Gamma-point calculations on orthogonal systems, using the affine translation-covariant representation.
template <typename T, typename VectorType>
void corr_msd(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& H1,
    DistVector<T>& diagonal_position,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
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
    if (cheby.S != nullptr) {
        throw std::runtime_error("Gamma-affine MSD currently supports orthogonal systems only");
    }

    detail::run_gamma_affine_msd(
        cheby,
        H1,
        diagonal_position,
        psi_in,
        n_steps,
        dt_fs,
        n_kpm,
        msd_out,
        energy_out);
}

} // namespace calc
} // namespace rsatb

#endif
