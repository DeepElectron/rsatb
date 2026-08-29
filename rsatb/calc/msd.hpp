#ifndef RSATB_CALC_MSD_HPP
#define RSATB_CALC_MSD_HPP

#include <algorithm>
#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include "ac.hpp"
#include "kpm.hpp"
#include "moment_extrapolation.hpp"

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

// The state pair the commutator recursion propagates: psi (the evolving
// sample) and phi = [X, U(t)] psi (built by the velocity driver, never by a
// position operator).
template <typename VectorType>
struct CommutatorPair {
    VectorType phi;
    VectorType psi;
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
// `n_compute` (0 = off) runs the recurrence only to that order and extends
// the sequence to n_kpm by Burg linear prediction
// (moment_extrapolation.hpp): the DMRG community's resolution multiplier for
// Chebyshev spectral functions, here cutting the dominant per-step apply
// count by n_kpm / n_compute. An explicit opt-in knob, never inferred.
template <typename T, typename VectorType>
std::vector<cd> metric_density_moments(
    Chebyshev<T>& cheby,
    const VectorType& v,
    int n_kpm,
    int n_compute = 0
) {
    const bool weighted = (cheby.form == ChebyOpForm::Solve);
    const int n_run = (n_compute > 0 && n_compute < n_kpm) ? n_compute : n_kpm;

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

    std::vector<cd> moments(static_cast<size_t>(std::max(n_run, 0)), cd(0.0, 0.0));
    if (n_run <= 0) return moments;

    t0.copy_from(v);
    moments[0] = dot_avg(weight(t0), t0);
    if (n_run > 1) {
        cheby.apply_op(t0, t1, ChebyMode::NORMAL);
        moments[1] = dot_avg(weight(t0), t1);

        for (int n = 1;; ++n) {
            ::rsatb::util::throw_if_interrupted(v.graph->comm);
            // t1 = T_n v, t0 = T_{n-1} v
            VectorType& wt = weight(t1);
            if (2 * n - 1 < n_run && 2 * n - 1 > 1) {
                moments[static_cast<size_t>(2 * n - 1)] = 2.0 * dot_avg(wt, t0) - moments[1];
            }
            if (2 * n < n_run) {
                moments[static_cast<size_t>(2 * n)] = 2.0 * dot_avg(wt, t1) - moments[0];
            }
            if (2 * n + 1 >= n_run) break;
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
            t0.swap(t1);
            t1.swap(t2);
        }
    }
    if (n_run < n_kpm) return extrapolate_moments_lp(moments, n_kpm);
    return moments;
}

// The doubled recurrence over a (time-tile x sample) multivector: the same
// per-column arithmetic as metric_density_moments, with the dots read per
// column (bdot -- ONE Allreduce for the whole tile per dot) and averaged
// inside each consecutive `group` of columns, one output time per group.
// This is what makes the stepwise readout time-batched: a tile of commutator
// states runs one recurrence at SpMM width instead of one recurrence per
// output time, dividing both the per-apply overhead (a width-1 recurrence
// measured ~4x overhead-bound) and the reduction count by the tile depth,
// while every column's values -- and therefore the positivity of every
// reconstructed row -- are those of the untiled run.
template <typename T>
std::vector<std::vector<cd>> metric_density_moments_grouped(
    Chebyshev<T>& cheby,
    const DistMultiVector<T>& v,
    int n_kpm,
    int group,
    int n_compute = 0
) {
    const bool weighted = (cheby.form == ChebyOpForm::Solve);
    const int n_run = (n_compute > 0 && n_compute < n_kpm) ? n_compute : n_kpm;
    const int n_groups = v.num_vectors / group;

    std::vector<std::vector<cd>> moments(
        static_cast<size_t>(n_groups),
        std::vector<cd>(static_cast<size_t>(std::max(n_run, 0)), cd(0.0, 0.0)));
    if (n_run <= 0 || n_groups == 0) return moments;

    std::vector<T> per_col(static_cast<size_t>(v.num_vectors));
    auto dots_into = [&](const DistMultiVector<T>& a, const DistMultiVector<T>& b,
                         auto&& sink) {
        a.dot(b, per_col);
        for (int g = 0; g < n_groups; ++g) {
            cd acc(0.0, 0.0);
            for (int c = 0; c < group; ++c) {
                acc += static_cast<cd>(per_col[static_cast<size_t>(g) * group + c]);
            }
            sink(g, acc / static_cast<double>(group));
        }
    };

    DistMultiVector<T> t0(v.graph, v.num_vectors), t1(v.graph, v.num_vectors),
        t2(v.graph, v.num_vectors);
    std::unique_ptr<DistMultiVector<T>> wbuf;
    if (weighted) wbuf = std::make_unique<DistMultiVector<T>>(v.graph, v.num_vectors);
    auto weight = [&](DistMultiVector<T>& t) -> DistMultiVector<T>& {
        if (!weighted) return t;
        cheby.S->mult_dense(t, *wbuf);
        return *wbuf;
    };

    t0.copy_from(v);
    dots_into(weight(t0), t0, [&](int g, cd val) { moments[static_cast<size_t>(g)][0] = val; });
    if (n_run > 1) {
        cheby.apply_op(t0, t1, ChebyMode::NORMAL);
        dots_into(weight(t0), t1, [&](int g, cd val) { moments[static_cast<size_t>(g)][1] = val; });

        for (int n = 1;; ++n) {
            ::rsatb::util::throw_if_interrupted(v.graph->comm);
            DistMultiVector<T>& wt = weight(t1);
            if (2 * n - 1 < n_run && 2 * n - 1 > 1) {
                dots_into(wt, t0, [&](int g, cd val) {
                    moments[static_cast<size_t>(g)][static_cast<size_t>(2 * n - 1)] =
                        2.0 * val - moments[static_cast<size_t>(g)][1];
                });
            }
            if (2 * n < n_run) {
                dots_into(wt, t1, [&](int g, cd val) {
                    moments[static_cast<size_t>(g)][static_cast<size_t>(2 * n)] =
                        2.0 * val - moments[static_cast<size_t>(g)][0];
                });
            }
            if (2 * n + 1 >= n_run) break;
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
            t0.swap(t1);
            t1.swap(t2);
        }
    }
    if (n_run < n_kpm) {
        for (auto& m : moments) m = extrapolate_moments_lp(m, n_kpm);
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

// THE stepwise MSD kernel, for every basis, form, and boundary condition:
// the section-2a derivation recursion, driven by the commutator velocity.
//
// The pair (phi, psi) with phi_m = [X, T_m(A~)] psi is propagated under the
// block generator [[A~, C~],[0, A~]] with C~ = [X, A~] = (i / half_width) *
// (form-dressed V) -- exactly what apply_dual computes for the operator set's
// velocity, bare or composite. phi(0) = 0, and phi(t) IS the commutator state
// [X, U(t)] |psi>: no position operator, no diagonal split, no gauge choice.
// Boundary conditions enter only through V itself: the periodic bond moment
// carries (R + r_j - r_i), and with a single R = 0 image that weight IS the
// open-boundary [H, X] -- so the position-representation routes this file
// used to carry (explicit X, diagonal X, the Gamma-affine split) were the
// same estimator built three more ways, and were removed.
template <typename T, typename VectorType>
void run_commutator_msd(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out,
    int n_kpm_compute,
    int time_tile
) {
    const double hbar_ev_fs = 0.6582119569;
    const double tau = cheby.half_width * dt_fs / hbar_ev_fs;
    std::vector<cd> coeffs;
    timeprop_coeff(tau, 1e-14, true, coeffs);
    const int n_coeff = static_cast<int>(coeffs.size());

    std::vector<cd> density_moments = metric_density_moments(cheby, psi_in, n_kpm, n_kpm_compute);
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm), 0.0);

    // Time-batched readout: commutator states are stashed for `t_tile`
    // output times and read out through ONE grouped moment recurrence at
    // combined width t_tile * s_width. The auto tile fills the combined
    // width to 16 columns -- the measured SpMM efficiency plateau, the same
    // cap the spectral chunking uses -- so a few-sample production run
    // stops paying width-1 recurrences. Memory: ~6 tile-width columns of
    // recurrence state.
    int s_width = 1;
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        s_width = psi_in.num_vectors;
    }
    int t_tile = time_tile;
    if (t_tile <= 0) t_tile = std::max(1, 16 / s_width);

    DistGraph* graph = psi_in.graph;
    DistMultiVector<T> stash(graph, t_tile * s_width);
    std::vector<int> pending;
    pending.reserve(static_cast<size_t>(t_tile));

    auto copy_cols = [](DistMultiVector<T>& dst, int dcol0,
                        const DistMultiVector<T>& src, int scol0, int n_cols) {
        for (int row = 0; row < src.local_rows; ++row) {
            for (int c = 0; c < n_cols; ++c) {
                dst.data[static_cast<size_t>(row) * dst.ld + dcol0 + c] =
                    src.data[static_cast<size_t>(row) * src.ld + scol0 + c];
            }
        }
    };

    auto flush_tile = [&]() {
        if (pending.empty()) return;
        const int n_groups = static_cast<int>(pending.size());
        DistMultiVector<T> work(graph, n_groups * s_width);
        copy_cols(work, 0, stash, 0, n_groups * s_width);
        const auto grouped =
            metric_density_moments_grouped(cheby, work, n_kpm, s_width, n_kpm_compute);
        for (int g = 0; g < n_groups; ++g) {
            std::vector<cd> msd_spectrum;
            reconstruct_spectrum_from_moments(grouped[static_cast<size_t>(g)],
                                              msd_spectrum, nullptr, true);
            write_msd_row(density_spectrum, msd_spectrum, pending[static_cast<size_t>(g)],
                          n_kpm, msd_out);
        }
        pending.clear();
    };

    auto stash_phi = [&](const VectorType& phi, int step) {
        const int slot = static_cast<int>(pending.size());
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            copy_cols(stash, slot * s_width, phi, 0, s_width);
        } else {
            stash.set_col(slot, phi);
        }
        pending.push_back(step);
        if (static_cast<int>(pending.size()) == t_tile) flush_tile();
    };

    // driver(x) = C~ x = (i / half_width) * apply_dual(V, x)
    VectorType driver_tmp = ZeroLike(psi_in);
    auto add_driver = [&](const VectorType& x, VectorType& into) {
        cheby.apply_dual(velocity, x, driver_tmp, ChebyMode::NORMAL, 1.0 / cheby.half_width);
        into.axpy(static_cast<T>(cd(0.0, 1.0)), driver_tmp);
    };

    CommutatorPair<VectorType> curr{ZeroLike(psi_in), ZeroLike(psi_in)};
    CommutatorPair<VectorType> next{ZeroLike(psi_in), ZeroLike(psi_in)};
    CommutatorPair<VectorType> t0{ZeroLike(psi_in), ZeroLike(psi_in)};
    CommutatorPair<VectorType> t1{ZeroLike(psi_in), ZeroLike(psi_in)};
    CommutatorPair<VectorType> t2{ZeroLike(psi_in), ZeroLike(psi_in)};

    curr.psi.copy_from(psi_in);   // curr.phi = [X, U(0)] psi = 0

    auto apply_pair = [&](const CommutatorPair<VectorType>& in,
                          CommutatorPair<VectorType>& out) {
        cheby.apply_op(in.phi, out.phi, ChebyMode::NORMAL);
        cheby.apply_op(in.psi, out.psi, ChebyMode::NORMAL);
        add_driver(in.psi, out.phi);
    };

    for (int step = 0; step <= n_steps; ++step) {
        ::rsatb::util::throw_if_interrupted(psi_in.graph->comm);
        if (step > 0) {
            // One Chebyshev time step of the pair, written against apply_pair.
            t0.phi.copy_from(curr.phi);
            t0.psi.copy_from(curr.psi);
            next.phi.copy_from(t0.phi);
            next.psi.copy_from(t0.psi);
            next.phi.scale(static_cast<T>(coeffs[0]));
            next.psi.scale(static_cast<T>(coeffs[0]));
            if (n_coeff > 1) {
                apply_pair(t0, t1);
                next.phi.axpy(static_cast<T>(coeffs[1]), t1.phi);
                next.psi.axpy(static_cast<T>(coeffs[1]), t1.psi);
            }
            for (int n = 2; n < n_coeff; ++n) {
                apply_pair(t1, t2);
                t2.phi.axpby(static_cast<T>(-1.0), t0.phi, static_cast<T>(2.0));
                t2.psi.axpby(static_cast<T>(-1.0), t0.psi, static_cast<T>(2.0));
                next.phi.axpy(static_cast<T>(coeffs[n]), t2.phi);
                next.psi.axpy(static_cast<T>(coeffs[n]), t2.psi);
                std::swap(t0, t1);
                std::swap(t1, t2);
            }
            curr.phi.copy_from(next.phi);
            curr.psi.copy_from(next.psi);
        }

        log_msd_step(psi_in.graph->rank, step, n_steps);

        stash_phi(curr.phi, step);
    }
    flush_tile();
}

} // namespace detail

// THE stepwise MSD, driven by the commutator velocity: valid for every
// basis, every operator form, and either boundary condition, because the
// driver is apply_dual's form-dressed velocity and the spectral readout is
// metric-weighted (doc/position_commutator.md section 8). This is the one
// stepwise entry point -- the position-representation overloads it replaced
// computed the same estimator from redundant data. It is also the
// positivity oracle: at full moments every row is a diagonal Jackson
// density of the commutator state, non-negative bin by bin.
// `time_tile` (0 = auto) batches that many output times into one grouped
// moment recurrence; the auto tile fills the combined SpMM width to 16
// columns. Any tiling reproduces the untiled readout column for column.
template <typename T, typename VectorType>
void corr_msd(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const VectorType& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out,
    int n_kpm_compute = 0,
    int time_tile = 0
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
        energy_out,
        n_kpm_compute,
        time_tile);
    }
}
} // namespace calc
} // namespace rsatb

#endif
