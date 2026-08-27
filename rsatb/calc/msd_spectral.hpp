#ifndef RSATB_CALC_MSD_SPECTRAL_HPP
#define RSATB_CALC_MSD_SPECTRAL_HPP

// Spectral (single-chain) diffusion spectrum: MSD(E, t) without a KPM
// recurrence per time step. doc/design_msd_spectral.md carries the full
// derivation, cost model, and references; the short version:
//
// The stepwise route (msd.hpp) pays n_steps * n_kpm operator applies because
// delta(E - A~) sits between two time-evolved states and is re-expanded at
// every output time. The identity
//
//   <phi(t)| T_p |phi(t)> = 2 Re int_0^t (t - u) C_p(u) du,
//   C_p(u) = Tr[ T_p(A~) v(u) v ],   v(u) = U(u)^-1 v U(u)
//
// moves all time dependence into one Heisenberg factor; expanding
// U(u)^-1 = sum_j c_j(u)^* T_j(A~) and folding T_p T_j = (T_{p+j} +
// T_{|p-j|})/2 onto the SAME chain leaves
//
//   C_p(u) = sum_j c_j(u)^* [ d_{p+j}(u) + d_{|p-j|}(u) ] / 2,
//   d_q(u) = < W T_q(A~) r | D U(u) D r >,
//
// with D the form-dressed velocity (apply_dual) and W the form's metric (S
// under Solve; W T_q(A~) = T_q(A~)^H W is the same self-adjointness the
// doubled moments lean on). One chain of n_kpm + K applies (K = the
// Chebyshev content of the whole time window, T * half_width / hbar plus the
// Bessel tail), one block-concurrent evolution of ~K applies
// (arXiv:2510.17432), one ZGEMM dot table -- and every output time is scalar
// post-processing. Cost is resolution + evolution instead of resolution x
// n_steps; the win is ~ n_kpm / (4 tau_step) at long windows, aimed at the
// expensive-apply forms (Solve: one CG per apply; Sandwich: three SpMVs).
//
// The time integral is done with Kaiser-windowed-sinc (cardinal) quadrature
// on a 2x-Nyquist grid: C_p(u) is band-limited to |omega| <= 2 in scaled
// units (eigenvalue differences), so cardinal reconstruction is spectrally
// accurate where a trapezoid at the same spacing errs at the percent level.
// Re C_p(u) is even in u, so the grid mirrors exactly through u = 0; the
// right edge is covered by evolving one kernel half-width past the window.
//
// Estimator note: stepwise and spectral estimators share the physical trace
// as expectation but differ per sample (the identity uses trace cyclicity).
// Summed over a complete metric basis both are the exact trace, which is
// what the dense unit tests compare.

#include "msd.hpp"
#include "moment_extrapolation.hpp"
#include "lapack_isqrt.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rsatb {
namespace calc {

namespace detail {

// Modified Bessel I0 by its power series: the argument here is a Kaiser
// shape parameter (~20), far inside the series' comfortable range.
inline double bessel_i0(double x) {
    const double half = 0.5 * x;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k < 200; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

// Cardinal (Kaiser-windowed sinc) quadrature of int_0^{t_k} (t_k - u) C(u) du
// from uniform samples C[c], c = 0..n_cols-1, spacing h, with the exact even
// mirror through u = 0. w[k][c] are the folded weights; they depend only on
// the grid, never on the integrand, so one build serves every moment index
// and every sample.
struct SincQuadWeights {
    int n_steps = 0;
    int m_sub = 1;
    int half_width_samples = 16;
    double h = 0.0;
    std::vector<std::vector<double>> w;  // [n_steps + 1][n_cols]

    SincQuadWeights(int n_steps_, int m_sub_, double h_, int n_cols, int ws = 16)
        : n_steps(n_steps_), m_sub(m_sub_), half_width_samples(ws), h(h_) {
        const double beta = 0.45 * M_PI * static_cast<double>(ws);
        const double i0_beta = bessel_i0(beta);
        auto kernel = [&](double s) {
            const double a = std::abs(s);
            if (a >= static_cast<double>(ws)) return 0.0;
            const double core = (a < 1e-12) ? 1.0 : std::sin(M_PI * s) / (M_PI * s);
            const double arg = 1.0 - (s / ws) * (s / ws);
            return core * bessel_i0(beta * std::sqrt(std::max(0.0, arg))) / i0_beta;
        };

        w.assign(static_cast<size_t>(n_steps + 1),
                 std::vector<double>(static_cast<size_t>(n_cols), 0.0));
        const int points_per_sample = 12;
        for (int k = 0; k <= n_steps; ++k) {
            const double t = static_cast<double>(k) * m_sub * h;
            auto& wk = w[static_cast<size_t>(k)];
            for (int e = -ws; e <= k * m_sub + ws; ++e) {
                const double lo = std::max(0.0, (e - ws) * h);
                const double hi = std::min(t, (e + ws) * h);
                if (hi <= lo) continue;
                // Simpson over the support; the kernel oscillates with
                // period 2h, so 12 points per sample resolves it well.
                const int n_iv = std::max(4, static_cast<int>(
                    std::ceil((hi - lo) / h * points_per_sample / 2.0)) * 2);
                const double du = (hi - lo) / n_iv;
                double acc = 0.0;
                for (int i = 0; i <= n_iv; ++i) {
                    const double u = lo + du * i;
                    const double f = (t - u) * kernel(u / h - e);
                    const double simpson =
                        (i == 0 || i == n_iv) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
                    acc += simpson * f;
                }
                acc *= du / 3.0;
                const int c = std::abs(e);
                if (c < n_cols) wk[static_cast<size_t>(c)] += acc;
            }
        }
    }
};

// One stochastic sample's contribution to the spectral-route numerator
// moments num_p(t_k) and to the density moments. Complex double only: the
// evolution and the dot table are intrinsically complex, and the tile GEMM
// goes through zgemm.
template <typename T>
void msd_spectral_accumulate(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const DistVector<T>& r,
    int n_steps,
    double dt_fs,
    int n_kpm,
    const SincQuadWeights& quad,
    int block_substeps,
    std::vector<double>& num_accum,  // [(n_steps + 1) * n_kpm], += contribution
    std::vector<cd>& rho_accum       // [n_kpm], += contribution
) {
    static_assert(std::is_same_v<T, std::complex<double>>,
                  "corr_msd_spectral runs in complex double");
    DistGraph* graph = r.graph;
    const int rank = graph->rank;
    const double h = quad.h;
    const int m_sub = quad.m_sub;
    const int n_cols = static_cast<int>(quad.w.front().size());
    const double hw = cheby.half_width;

    // Backward coefficients per grid point: U(u)^-1 = sum_j bwd[u][j] T_j(A~).
    std::vector<std::vector<cd>> bwd(static_cast<size_t>(n_cols));
    bwd[0] = {cd(1.0, 0.0)};
    for (int u = 1; u < n_cols; ++u) {
        timeprop_coeff(u * h, 1e-14, false, bwd[u]);
    }
    const int k_j = static_cast<int>(bwd.back().size());
    const int n_q = n_kpm + k_j - 1;  // chain states T_q, q = 0..n_q-1

    auto apply_velocity = [&](const DistVector<T>& in, DistVector<T>& out) {
        cheby.apply_dual(velocity, in, out, ChebyMode::NORMAL, 1.0);
    };

    // ---- evolution: y(u) = D U(u) D r on the grid, block-concurrent -------
    DistMultiVector<T> Y(graph, n_cols);
    DistVector<T> w(graph), g(graph), tmp(graph), yv(graph);
    apply_velocity(r, w);       // w = D r, the evolving seed
    apply_velocity(w, yv);      // y(0) = D w
    Y.set_col(0, yv);

    int b_sub = block_substeps;
    if (b_sub <= 0) b_sub = std::max(1, static_cast<int>(std::ceil(192.0 / h)));
    b_sub = std::min(b_sub, n_cols - 1);

    std::vector<std::vector<cd>> ctab(static_cast<size_t>(b_sub) + 1);
    for (int k = 1; k <= b_sub; ++k) {
        timeprop_coeff(k * h, 1e-14, true, ctab[static_cast<size_t>(k)]);
    }
    const int n_states = static_cast<int>(ctab[static_cast<size_t>(b_sub)].size());
    std::vector<DistVector<T>> X;
    X.reserve(static_cast<size_t>(n_states));
    for (int m = 0; m < n_states; ++m) X.emplace_back(graph);

    g.copy_from(w);
    int done = 1;  // next grid index to fill
    while (done < n_cols) {
        ::rsatb::util::throw_if_interrupted(graph->comm);
        const int b = std::min(b_sub, n_cols - done);
        const int nk = static_cast<int>(ctab[static_cast<size_t>(b)].size());
        X[0].copy_from(g);
        if (nk > 1) cheby.apply_op(X[0], X[1], ChebyMode::NORMAL);
        for (int m = 2; m < nk; ++m) {
            cheby.apply_op(X[static_cast<size_t>(m - 1)], X[static_cast<size_t>(m)],
                           ChebyMode::NORMAL);
            X[static_cast<size_t>(m)].axpby(static_cast<T>(-1.0),
                                            X[static_cast<size_t>(m - 2)],
                                            static_cast<T>(2.0));
        }
        for (int k = 1; k <= b; ++k) {
            const auto& c = ctab[static_cast<size_t>(k)];
            tmp.copy_from(X[0]);
            tmp.scale(CoeffCast<T>::apply(c[0]));
            for (size_t m = 1; m < c.size(); ++m) {
                tmp.axpy(CoeffCast<T>::apply(c[m]), X[m]);
            }
            apply_velocity(tmp, yv);
            Y.set_col(done + k - 1, yv);
            if (k == b) g.copy_from(tmp);  // block-end state seeds the next block
        }
        done += b;
        if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kSummary)) {
            std::cout << "msd-spectral: evolve " << (done - 1) << " / " << (n_cols - 1)
                      << std::endl;
        }
    }

    // ---- left chain + dot table d_q(u) = <W T_q r | y(u)> ------------------
    const bool weighted = (cheby.form == ChebyOpForm::Solve);
    const int tile_w = 32;
    DistMultiVector<T> ltile(graph, tile_w);
    DistVector<T> t0(graph), t1(graph), t2(graph), wq(graph);
    std::vector<cd> dtab(static_cast<size_t>(n_q) * static_cast<size_t>(n_cols),
                         cd(0.0, 0.0));
    std::vector<cd> tile_buf(static_cast<size_t>(tile_w) * static_cast<size_t>(n_cols));

    auto flush_tile = [&](int q_start, int width) {
        using isqrt::vbcsr_complex_double;
        const vbcsr_lapack_int m_ = width;
        const vbcsr_lapack_int n_ = n_cols;
        const vbcsr_lapack_int k_ = ltile.local_rows;
        const vbcsr_lapack_int lda_ = ltile.ld;
        const vbcsr_lapack_int ldb_ = Y.ld;
        const vbcsr_lapack_int ldc_ = width;
        const cd one(1.0, 0.0), zero(0.0, 0.0);
        // Row-major storage read as column-major is the transpose, so this
        // computes  C(q, u) = sum_rows L[row, q] * conj(Y[row, u]);
        // the wanted d = sum_rows conj(L) * Y is its conjugate.
        isqrt::zgemm_("N", "C", &m_, &n_, &k_,
                      reinterpret_cast<const vbcsr_complex_double*>(&one),
                      reinterpret_cast<const vbcsr_complex_double*>(ltile.data.data()), &lda_,
                      reinterpret_cast<const vbcsr_complex_double*>(Y.data.data()), &ldb_,
                      reinterpret_cast<const vbcsr_complex_double*>(&zero),
                      reinterpret_cast<vbcsr_complex_double*>(tile_buf.data()), &ldc_);
        MPI_Allreduce(MPI_IN_PLACE, tile_buf.data(), 2 * width * n_cols, MPI_DOUBLE,
                      MPI_SUM, graph->comm);
        for (int q = 0; q < width; ++q) {
            cd* row = dtab.data() + static_cast<size_t>(q_start + q) * n_cols;
            for (int u = 0; u < n_cols; ++u) {
                row[u] = std::conj(tile_buf[static_cast<size_t>(u) * width + q]);
            }
        }
    };

    auto stash = [&](int q, const DistVector<T>& state) {
        if (weighted) {
            cheby.S->mult(const_cast<DistVector<T>&>(state), wq);
            ltile.set_col(q % tile_w, wq);
        } else {
            ltile.set_col(q % tile_w, state);
        }
        if ((q % tile_w) == tile_w - 1 || q == n_q - 1) {
            flush_tile(q - (q % tile_w), (q % tile_w) + 1);
        }
    };

    t0.copy_from(r);
    stash(0, t0);
    if (n_q > 1) {
        cheby.apply_op(t0, t1, ChebyMode::NORMAL);
        stash(1, t1);
    }
    const int stride = std::max(1, n_q / 10);
    for (int q = 2; q < n_q; ++q) {
        ::rsatb::util::throw_if_interrupted(graph->comm);
        cheby.apply_op(t1, t2, ChebyMode::NORMAL);
        t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
        stash(q, t2);
        t0.swap(t1);
        t1.swap(t2);
        if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kSummary) &&
            ::rsatb::util::every_nth(q - 2, stride)) {
            std::cout << "msd-spectral: chain " << q << " / " << n_q << std::endl;
        }
    }

    // ---- scalar assembly ---------------------------------------------------
    // C_p(u) = Re sum_j bwd[u][j] (d_{p+j}(u) + d_{|p-j|}(u)) / 2
    std::vector<double> cre(static_cast<size_t>(n_kpm) * static_cast<size_t>(n_cols));
#pragma omp parallel for schedule(static)
    for (int p = 0; p < n_kpm; ++p) {
        for (int u = 0; u < n_cols; ++u) {
            const auto& b = bwd[static_cast<size_t>(u)];
            cd acc(0.0, 0.0);
            for (size_t j = 0; j < b.size(); ++j) {
                const int jp = p + static_cast<int>(j);
                const int jm = std::abs(p - static_cast<int>(j));
                acc += b[j] * (0.5 * (dtab[static_cast<size_t>(jp) * n_cols + u] +
                                      dtab[static_cast<size_t>(jm) * n_cols + u]));
            }
            cre[static_cast<size_t>(p) * n_cols + u] = acc.real();
        }
    }

    // num_p(t_k) = (2 / hw^2) int_0^{t_k} (t_k - u) C_p(u) du
    const double scale = 2.0 / (hw * hw);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < n_kpm; ++p) {
        const double* cp = cre.data() + static_cast<size_t>(p) * n_cols;
        for (int k = 0; k <= n_steps; ++k) {
            const auto& wk = quad.w[static_cast<size_t>(k)];
            double acc = 0.0;
            for (int u = 0; u < n_cols; ++u) acc += wk[static_cast<size_t>(u)] * cp[u];
            num_accum[static_cast<size_t>(k) * n_kpm + p] += scale * acc;
        }
    }

    // Density moments of the same sample, for the shared normalisation.
    const std::vector<cd> rho = metric_density_moments(cheby, r, n_kpm);
    for (int p = 0; p < n_kpm; ++p) rho_accum[static_cast<size_t>(p)] += rho[static_cast<size_t>(p)];
}

// Grid setup shared by the public overloads: substeps per output step so that
// the sample spacing is at most (pi/2) / time_oversample in scaled time
// (Nyquist for the |omega| <= 2 band, times the safety factor), plus one
// quadrature kernel half-width of padding past the window.
struct SpectralGrid {
    int m_sub = 1;
    double h = 0.0;
    int n_cols = 0;
    static constexpr int kKernelHalfWidth = 16;

    SpectralGrid(double half_width, double dt_fs, int n_steps, int time_oversample) {
        const double hbar_ev_fs = 0.6582119569;
        const double tau_step = half_width * dt_fs / hbar_ev_fs;
        const double target = 0.5 * M_PI / std::max(1, time_oversample);
        m_sub = std::max(1, static_cast<int>(std::ceil(tau_step / target)));
        h = tau_step / m_sub;
        n_cols = n_steps * m_sub + kKernelHalfWidth + 1;
    }
};

// `n_acc` is the accumulated moment count; when it is below `n_kpm_out` the
// rows are extended by Burg linear prediction (moment_extrapolation.hpp)
// before the shared Jackson reconstruction, exactly as in the stepwise
// route's n_compute knob.
template <typename T>
void msd_spectral_finalize(
    Chebyshev<T>& cheby,
    int n_steps,
    int n_acc,
    int n_kpm_out,
    double inv_samples,
    const std::vector<double>& num_accum,
    const std::vector<cd>& rho_accum,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out
) {
    auto extend = [&](std::vector<cd>& m) {
        if (n_acc < n_kpm_out) m = extrapolate_moments_lp(m, n_kpm_out);
    };

    std::vector<cd> rho(static_cast<size_t>(n_acc));
    for (int p = 0; p < n_acc; ++p) rho[static_cast<size_t>(p)] = rho_accum[static_cast<size_t>(p)] * inv_samples;
    extend(rho);
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(rho, density_spectrum, nullptr, true);
    energy_out = chebyshev_energy_grid(cheby.center, cheby.half_width, n_kpm_out, true);

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm_out), 0.0);
    std::vector<cd> num(static_cast<size_t>(n_acc));
    for (int k = 0; k <= n_steps; ++k) {
        num.assign(static_cast<size_t>(n_acc), cd(0.0, 0.0));
        for (int p = 0; p < n_acc; ++p) {
            num[static_cast<size_t>(p)] =
                cd(num_accum[static_cast<size_t>(k) * n_acc + p] * inv_samples, 0.0);
        }
        extend(num);
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(num, msd_spectrum, nullptr, true);
        write_msd_row(density_spectrum, msd_spectrum, k, n_kpm_out, msd_out);
    }
}

}  // namespace detail

// Spectral-route periodic/any-basis MSD: same contract as the commutator
// overload of corr_msd ([n_steps + 1, n_kpm] density-normalised rows on the
// ascending Chebyshev grid), computed at resolution-plus-evolution cost. See
// the header note and doc/design_msd_spectral.md; intended for the
// expensive-apply forms and fine output grids -- the stepwise route remains
// the right tool for cheap orthogonal matvecs with very long windows.
template <typename T>
void corr_msd_spectral(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const DistVector<T>& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out,
    int time_oversample = 2,
    int block_substeps = 0,
    int n_kpm_compute = 0
) {
    if (n_steps < 0) throw std::runtime_error("n_steps must be non-negative");
    if (n_kpm <= 0) throw std::runtime_error("n_kpm must be positive");
    if (dt_fs <= 0.0) throw std::runtime_error("dt_fs must be positive");
    if constexpr (!std::is_same_v<T, std::complex<double>>) {
        throw std::runtime_error("corr_msd_spectral needs complex double scalars");
    } else {
        const int nc = (n_kpm_compute > 0 && n_kpm_compute < n_kpm) ? n_kpm_compute : n_kpm;
        detail::SpectralGrid grid(cheby.half_width, dt_fs, n_steps, time_oversample);
        detail::SincQuadWeights quad(n_steps, grid.m_sub, grid.h, grid.n_cols,
                                     detail::SpectralGrid::kKernelHalfWidth);
        std::vector<double> num(static_cast<size_t>(n_steps + 1) * nc, 0.0);
        std::vector<cd> rho(static_cast<size_t>(nc), cd(0.0, 0.0));
        detail::msd_spectral_accumulate(cheby, velocity, psi_in, n_steps, dt_fs, nc,
                                        quad, block_substeps, num, rho);
        detail::msd_spectral_finalize(cheby, n_steps, nc, n_kpm, 1.0, num, rho, msd_out,
                                      energy_out);
    }
}

// Sample-block overload: columns are independent stochastic samples; their
// numerator and density moments are averaged BEFORE the spectrum ratio,
// matching the multivector semantics of the stepwise route. The columns run
// through the vector path one at a time (matvec batching across columns is a
// follow-up; the spectral route's economy is in the apply count itself).
template <typename T>
void corr_msd_spectral(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const DistMultiVector<T>& psi_in,
    int n_steps,
    double dt_fs,
    int n_kpm,
    std::vector<double>& msd_out,
    std::vector<double>& energy_out,
    int time_oversample = 2,
    int block_substeps = 0,
    int n_kpm_compute = 0
) {
    if (n_steps < 0) throw std::runtime_error("n_steps must be non-negative");
    if (n_kpm <= 0) throw std::runtime_error("n_kpm must be positive");
    if (dt_fs <= 0.0) throw std::runtime_error("dt_fs must be positive");
    if constexpr (!std::is_same_v<T, std::complex<double>>) {
        throw std::runtime_error("corr_msd_spectral needs complex double scalars");
    } else {
        const int nc = (n_kpm_compute > 0 && n_kpm_compute < n_kpm) ? n_kpm_compute : n_kpm;
        detail::SpectralGrid grid(cheby.half_width, dt_fs, n_steps, time_oversample);
        detail::SincQuadWeights quad(n_steps, grid.m_sub, grid.h, grid.n_cols,
                                     detail::SpectralGrid::kKernelHalfWidth);
        std::vector<double> num(static_cast<size_t>(n_steps + 1) * nc, 0.0);
        std::vector<cd> rho(static_cast<size_t>(nc), cd(0.0, 0.0));
        DistMultiVector<T>& samples = const_cast<DistMultiVector<T>&>(psi_in);
        for (int col = 0; col < psi_in.num_vectors; ++col) {
            DistVector<T> r = samples.get_col(col);
            detail::msd_spectral_accumulate(cheby, velocity, r, n_steps, dt_fs, nc,
                                            quad, block_substeps, num, rho);
        }
        detail::msd_spectral_finalize(cheby, n_steps, nc, n_kpm,
                                      1.0 / static_cast<double>(psi_in.num_vectors),
                                      num, rho, msd_out, energy_out);
    }
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_MSD_SPECTRAL_HPP
