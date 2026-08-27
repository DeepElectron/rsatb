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
#include <memory>
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

// KPM moments of delta(E - A) on |v>, weighted by the metric the form
// implies (doc/position_commutator.md section 5): the density operator is
// self-adjoint in the S inner product, so
//
//   Solve       mu_n = <v| S T_n(A~) |v>     (A~ = scaled S^-1 H)
//   Sandwich    mu_n = <v| T_n(A~) |v>       (eta-space dots ARE the S metric)
//   Orthogonal  mu_n = <v| T_n(A~) |v>
//
// One doubled recurrence serves all three. Write W for the metric (S under
// Solve, the identity otherwise): A~ is self-adjoint in W, so W T_n(A~) =
// T_n(A~)^H W and the product identity 2 T_m T_n = T_{m+n} + T_{|m-n|}
// folds onto the single chain t_n = T_n(A~) v, exactly as in
// kpm_dos_moments (dos.hpp):
//
//   mu_{2n}   = 2 <t_n| W |t_n>     - mu_0
//   mu_{2n-1} = 2 <t_n| W |t_{n-1}> - mu_1
//
// -- two moments per apply_op instead of one. This is not the doubling the
// op_is_hermitian() gate in cheby.hpp forbids: that gate concerns PLAIN
// moments under the Solve form, whose bra would need a second Chebyshev
// chain seeded from S^-1 v. The S-weighted moments that comment sets aside
// as "the sandwich estimator resampled" are, here, the defined quantity --
// this file's spectra are S-weighted from the start. Per pair of moments
// the Solve form pays one S SpMV for W t_n, noise next to the CG solve
// inside its apply_op.
//
// This recurrence is the whole cost of the diffusion spectrum: n_steps *
// n_kpm applies against n_steps * O(tau) for the time evolution (tau =
// half_width * dt / hbar). The concurrent shared-state scheme that took the
// AC correlator to ~9x fewer propagation matvecs (corr_ac_concurrent) does
// NOT transfer to this term, for two reasons that survive every regrouping:
// its gain is the Bessel tail of e^{-i A dt} amortised over a block, and
// the moment recurrence has no tail -- all n_kpm terms are signal; and the
// moments are BILINEAR in the evolved state, so expanding phi(t_k) over
// block-shared Chebyshev states (b*tau + tail of them per block of b steps,
// tau ~ 100-200 at production dt_fs = 10) needs a T_n chain per stored
// state where AC needs one dot per sample. Whenever tau > 1 -- every
// physical MSD run -- stepping in time IS the compressed representation,
// and the halving above is what remains.
template <typename T, typename VectorType>
std::vector<cd> metric_density_moments(
    Chebyshev<T>& cheby,
    const VectorType& v,
    int n_kpm
) {
    const bool weighted = (cheby.form == ChebyOpForm::Solve);

    auto dot_avg = [&](const VectorType& a, const VectorType& b) -> cd {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            std::vector<T> per_column(static_cast<size_t>(a.num_vectors));
            a.dot(b, per_column);
            cd accum = 0.0;
            for (const auto& value : per_column) accum += static_cast<cd>(value);
            return accum / static_cast<double>(a.num_vectors);
        } else {
            return static_cast<cd>(a.dot(b));
        }
    };

    VectorType t0 = ZeroLike(v);
    VectorType t1 = ZeroLike(v);
    VectorType t2 = ZeroLike(v);
    std::unique_ptr<VectorType> wbuf;
    if (weighted) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            wbuf = std::make_unique<VectorType>(v.graph, v.num_vectors);
        } else {
            wbuf = std::make_unique<VectorType>(v.graph);
        }
    }
    // W t, or t itself when the metric is the identity. S is Hermitian, so
    // <W t_n| t_m> = <t_n| W |t_m> and the one weighted vector per
    // iteration serves both moment formulas.
    auto weight = [&](VectorType& t) -> VectorType& {
        if (!weighted) return t;
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            cheby.S->mult_dense(t, *wbuf);
        } else {
            cheby.S->mult(t, *wbuf);
        }
        return *wbuf;
    };

    std::vector<cd> moments(static_cast<size_t>(std::max(n_kpm, 0)), cd(0.0, 0.0));
    if (n_kpm <= 0) return moments;

    t0.copy_from(v);
    moments[0] = dot_avg(weight(t0), t0);
    if (n_kpm == 1) return moments;

    cheby.apply_op(t0, t1, ChebyMode::NORMAL);
    moments[1] = dot_avg(weight(t0), t1);

    for (int n = 1;; ++n) {
        ::rsatb::util::throw_if_interrupted(v.graph->comm);
        // t1 = T_n v, t0 = T_{n-1} v
        VectorType& wt = weight(t1);
        if (2 * n - 1 < n_kpm && 2 * n - 1 > 1) {
            moments[static_cast<size_t>(2 * n - 1)] = 2.0 * dot_avg(wt, t0) - moments[1];
        }
        if (2 * n < n_kpm) {
            moments[static_cast<size_t>(2 * n)] = 2.0 * dot_avg(wt, t1) - moments[0];
        }
        if (2 * n + 1 >= n_kpm) break;
        cheby.apply_op(t1, t2, ChebyMode::NORMAL);
        t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
        t0.swap(t1);
        t1.swap(t2);
    }
    return moments;
}

// The per-step line prints at kSummary, not kProgress: a production MSD task
// spends minutes to hours in this loop (n_kpm moment applies per step), and
// the level contract (profile.hpp) says silence over that span is unhelpful,
// not quiet. The every_nth throttle keeps it to ~20 lines per task --
// phase-boundary cadence, not the per-iteration reporting kProgress adds.
inline void log_msd_step(int rank, int step, int n_steps) {
    if (rank != 0 || !::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kSummary)) return;
    if (::rsatb::util::every_nth(step, std::max(1, (n_steps + 1) / 20)))
        std::cout << "msd: step " << step + 1 << " / " << (n_steps + 1) << std::endl;
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

    std::vector<cd> density_moments = metric_density_moments(cheby, psi_in, n_kpm);
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm), 0.0);

    auto apply_position = [&](const VectorType& psi_src, VectorType& xpsi_out) {
        if constexpr (std::is_same_v<PositionOp, DistVector<T>>) {
            xpsi_out.copy_from(psi_src);
            xpsi_out.pointwise_mult(position);
        } else {
            cheby.apply_dual(position, psi_src, xpsi_out, ChebyMode::NORMAL, 1.0);
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

        log_msd_step(psi_in.graph->rank, step, n_steps);

        apply_position(psi_curr, xpsi_curr);
        comm_state.copy_from(xpsi_curr);
        comm_state.axpy(static_cast<T>(-1.0), xpsi0_prop);

        std::vector<cd> msd_moments = metric_density_moments(cheby, comm_state, n_kpm);
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

    std::vector<cd> density_moments = metric_density_moments(cheby, psi_in, n_kpm);
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

        log_msd_step(psi_in.graph->rank, step, n_steps);

        // The affine branch stores U(t)X|psi> as a(t) + R b(t). In the commutator
        // XU(t)|psi> - U(t)X|psi>, the cell-linear R b(t) part cancels identically,
        // leaving only the periodic remainder X0 b(t) - a(t).
        xpsi_curr.copy_from(state_curr.periodic);
        xpsi_curr.pointwise_mult(diagonal_position);
        comm_state.copy_from(xpsi_curr);
        comm_state.axpy(static_cast<T>(-1.0), state_curr.affine);

        std::vector<cd> msd_moments = metric_density_moments(cheby, comm_state, n_kpm);
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);
        write_msd_row(density_spectrum, msd_spectrum, step, n_kpm, msd_out);
    }
}


// The periodic MSD for ANY basis and form: the section-2a derivation
// recursion, driven by the commutator velocity.
//
// The pair (phi, psi) with phi_m = [X, T_m(A~)] psi is propagated under the
// block generator [[A~, C~],[0, A~]] with C~ = [X, A~] = (i / half_width) *
// (form-dressed V) -- exactly what apply_dual computes for the operator set's
// velocity, bare or composite. phi(0) = 0, and phi(t) IS the commutator state
// [X, U(t)] |psi>: no position operator, no diagonal split, no gauge choice.
// In the orthogonal limit this is the same physics as run_gamma_affine_msd in
// the other gauge (driver -M[H] against coupling H1 plus the diagonal split);
// both are kept because the affine split is cheaper per step there.
template <typename T, typename VectorType>
void run_commutator_msd(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
) {
    const double hbar_ev_fs = 0.6582119569;
    const double tau = cheby.half_width * dt_fs / hbar_ev_fs;
    std::vector<cd> coeffs;
    timeprop_coeff(tau, 1e-14, true, coeffs);
    const int n_coeff = static_cast<int>(coeffs.size());

    std::vector<cd> density_moments = metric_density_moments(cheby, psi_in, n_kpm);
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm), 0.0);

    // driver(x) = C~ x = (i / half_width) * apply_dual(V, x)
    VectorType driver_tmp = ZeroLike(psi_in);
    auto add_driver = [&](const VectorType& x, VectorType& into) {
        cheby.apply_dual(velocity, x, driver_tmp, ChebyMode::NORMAL, 1.0 / cheby.half_width);
        into.axpy(static_cast<T>(cd(0.0, 1.0)), driver_tmp);
    };

    GammaAffineState<VectorType> curr{ZeroLike(psi_in), ZeroLike(psi_in)};
    GammaAffineState<VectorType> next{ZeroLike(psi_in), ZeroLike(psi_in)};
    GammaAffineState<VectorType> t0{ZeroLike(psi_in), ZeroLike(psi_in)};
    GammaAffineState<VectorType> t1{ZeroLike(psi_in), ZeroLike(psi_in)};
    GammaAffineState<VectorType> t2{ZeroLike(psi_in), ZeroLike(psi_in)};

    curr.periodic.copy_from(psi_in);   // psi;  curr.affine = phi = 0

    auto apply_pair = [&](const GammaAffineState<VectorType>& in,
                          GammaAffineState<VectorType>& out) {
        cheby.apply_op(in.affine, out.affine, ChebyMode::NORMAL);
        cheby.apply_op(in.periodic, out.periodic, ChebyMode::NORMAL);
        add_driver(in.periodic, out.affine);
    };

    for (int step = 0; step <= n_steps; ++step) {
        ::rsatb::util::throw_if_interrupted(psi_in.graph->comm);
        if (step > 0) {
            // One Chebyshev time step of the pair, the same recurrence
            // propagate_affine runs, written against apply_pair.
            t0.affine.copy_from(curr.affine);
            t0.periodic.copy_from(curr.periodic);
            next.affine.copy_from(t0.affine);
            next.periodic.copy_from(t0.periodic);
            next.affine.scale(static_cast<T>(coeffs[0]));
            next.periodic.scale(static_cast<T>(coeffs[0]));
            if (n_coeff > 1) {
                apply_pair(t0, t1);
                next.affine.axpy(static_cast<T>(coeffs[1]), t1.affine);
                next.periodic.axpy(static_cast<T>(coeffs[1]), t1.periodic);
            }
            for (int n = 2; n < n_coeff; ++n) {
                apply_pair(t1, t2);
                t2.affine.axpby(static_cast<T>(-1.0), t0.affine, static_cast<T>(2.0));
                t2.periodic.axpby(static_cast<T>(-1.0), t0.periodic, static_cast<T>(2.0));
                next.affine.axpy(static_cast<T>(coeffs[n]), t2.affine);
                next.periodic.axpy(static_cast<T>(coeffs[n]), t2.periodic);
                std::swap(t0, t1);
                std::swap(t1, t2);
            }
            curr.affine.copy_from(next.affine);
            curr.periodic.copy_from(next.periodic);
        }

        log_msd_step(psi_in.graph->rank, step, n_steps);

        std::vector<cd> msd_moments = metric_density_moments(cheby, curr.affine, n_kpm);
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);
        write_msd_row(density_spectrum, msd_spectrum, step, n_kpm, msd_out);
    }
}

} // namespace detail

// Route selection is done by overload, not by a runtime mode switch in this file:
//   corr_msd(cheby, X_matrix, ...)         -> explicit position matrix route (OBC)
//   corr_msd(cheby, x_diagonal, ...)       -> explicit orthogonal diagonal route (OBC)
//   corr_msd(cheby, H1, x_diagonal, ...)   -> periodic Gamma affine route (orthogonal)
//   corr_msd(cheby, velocity, ...)         -> periodic commutator route (any basis)
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
    if (cheby.form != ChebyOpForm::Orthogonal) {
        throw std::runtime_error(
            "Diagonal-position MSD needs an orthogonal basis. This overload applies X as a "
            "plain elementwise product, which is the position operator only when the basis "
            "is orthonormal; with S^-1 H or Z^H H Z the recurrence would iterate a dressed "
            "operator against an undressed X. Pass the explicit position MATRIX instead -- "
            "that overload routes X through apply_dual, which dresses it by the form.");
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

// periodic boundary condition MSD driven by the commutator velocity: valid for
// every basis and every operator form, because the driver is apply_dual's
// form-dressed velocity and the spectral readout is metric-weighted. This is
// the non-orthogonal periodic route doc/position_commutator.md section 8
// staged; on an orthogonal basis it computes the same physics as the affine
// overload above in the other gauge.
template <typename T, typename VectorType>
void corr_msd(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
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
    if constexpr (std::is_floating_point_v<T>) {
        throw std::runtime_error(
            "The commutator MSD carries an explicit i in its driver and needs a "
            "complex scalar type.");
    } else {
    detail::run_commutator_msd(
        cheby,
        velocity,
        psi_in,
        n_steps,
        dt_fs,
        n_kpm,
        msd_out,
        energy_out);
    }
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
    if (cheby.form != ChebyOpForm::Orthogonal) {
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
