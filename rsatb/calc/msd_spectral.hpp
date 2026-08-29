#ifndef RSATB_CALC_MSD_SPECTRAL_HPP
#define RSATB_CALC_MSD_SPECTRAL_HPP

// Spectral (segmented single-chain) diffusion spectrum: MSD(E, t) without a
// KPM recurrence per time step, at cost LINEAR in the evolution window.
// doc/design_msd_spectral.md carries the derivation, cost model, and
// references; the short version:
//
// The stepwise route (msd.hpp) pays n_steps * n_kpm operator applies because
// delta(E - A~) sits between two time-evolved states and is re-expanded at
// every output time. The identity
//
//   <phi(t)| T_p |phi(t)> = 2 Re int_0^t (t - u) C_p(u) du,
//   C_p(u) = Tr[ T_p(A~) v(u) v ],   v(u) = U(u)^-1 v U(u)
//
// moves all time dependence into one Heisenberg factor, and expanding
// U(u)^-1 in Chebyshev folds T_p T_j = (T_{p+j} + T_{|p-j|})/2 onto a single
// chain. Anchoring that expansion at the time ORIGIN spreads it over ~u
// coefficients and makes the dot table quadratic in the window; instead the
// window is cut into segments of scaled length ~L, and for u = u_s + delta
// inside segment s metric-unitarity moves U^-1(u_s) onto the bra, where --
// because U commutes with polynomials of A~ -- the evolved chain family IS
// the chain of the evolved seed, U(u_s) T_q(A~) r = T_q(A~) r_s with
// r_s = U(u_s) r:
//
//   C_p(u_s + delta) = sum_j c_j(delta)^* [ d^s_{p+j}(delta) + d^s_{|p-j|}(delta) ] / 2,
//   d^s_q(delta)     = < W T_q(A~) r_s | D g(u_s + delta) >,
//
// with delta <= L: coefficients never spread beyond a segment. Every
// rearrangement is an operator identity on the same quadratic form, so the
// segmented estimator equals the single-segment one per sample, to rounding
// (unit-tested), and a window short enough for one segment reproduces the
// unsegmented route exactly. D is the form-dressed velocity and W the
// form's metric; for the Solve form with a non-composite velocity the
// dressing cancels into the metric, <S T_q r_s | S^-1 V g> = <T_q r_s | V g>,
// so both sides run bare.
//
// The computation is organised so that every axis is either one sequential
// recurrence or a scalar contraction:
//
//   trajectories  g(u) = U(u) D r and the chain anchor r_s = U(u_s) r both
//                 advance through Chebyshev blocks (arXiv:2510.17432); only
//                 the velocity-applied block states VX_m = D_v T_m(A~) g_b
//                 are stored (~0.8 columns per grid point, SEGMENT-local;
//                 raw states are never kept -- block-end seeds accumulate on
//                 the fly). ~2K applies total, K = window * half_width /
//                 hbar.
//   chains        per segment, T_q(A~) r_s for q < n_kpm + M(L), dotted
//                 against the segment's VX columns in tiled ZGEMMs --
//                 K (n_kpm/L + 1) applies amortised over the window. The dot
//                 table G_{q,m} is trapezoidal (a block's states are read
//                 only up to the chain order its last grid point reaches),
//                 so high-q tiles restrict to their late-block suffix.
//   grid points   TRACE-FIRST: the dot table is summed over the sample
//                 chunk as it forms (everything downstream is linear in it),
//                 so the scalar fold runs once per chunk, not per sample.
//                 d_q(u) = sum_m c_m(k) G_{q,m} is one ZGEMM per block, and
//                 the Bessel fold to C_p(u) is a Toeplitz-plus-Hankel
//                 product done as one FFT convolution per grid point
//                 (O(L log L) instead of O(n_kpm M)). Blocks' points are
//                 dealt contiguously across MPI ranks (each rank's D comes
//                 from its own GEMM slice), OpenMP inside. No per-grid-point
//                 vector work exists at all.
//   time          num_p(t) = 2/hw^2 int_0^t (t - u) C_p(u) du by cardinal
//                 (Kaiser-windowed sinc) quadrature in closed form: an
//                 interior sample's kernel integral is exactly
//                 h^2 mu0 (t/h - c), so only O(W) edge samples carry
//                 shift-invariant correction tables, and with prefix sums
//                 every output time costs O(W). Output grids of any density
//                 are free. Re C_p(u) is even, so the grid mirrors exactly
//                 through u = 0; the evolution runs one kernel half-width
//                 past the window for the right edge.
//
// Cost per unit scaled time: ~(2 + n_kpm/L) applies plus ~10 n_kpm dot-flops
// per matrix row-element, linear in the window, independent of the output
// grid. Memory per rank: ~1.3 min(L, window) N_local x 16 B of segment-local
// columns plus scalar tables. `segment_scaled` sets L (0 = auto:
// max(192, min(768, n_kpm))), trading chain amortisation against dots and
// memory.
//
// Estimator note: stepwise and spectral estimators share the physical trace
// as expectation but differ per sample (the derivation uses trace
// cyclicity). Summed over a complete metric basis both are the exact trace,
// which is what the dense unit tests compare.

#include "msd.hpp"
#include "moment_extrapolation.hpp"
#include "lapack_isqrt.hpp"
#include "fft.hpp"

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

// Grid setup shared by the public overloads: substeps per output step so
// that the sample spacing is at most (pi/2) / time_oversample in scaled
// time (Nyquist for the |omega| <= 2 band, times the safety factor), plus
// one quadrature kernel half-width of padding past the window.
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

// int_0^{t_k} (t_k - u) C(u) du from uniform samples C[c] (spacing h, even
// mirror through u = 0), through the cardinal interpolant
// C(u) = sum_e C[|e|] kappa(u/h - e), kappa a Kaiser-windowed sinc of
// half-width W samples. See the header note: interior weights are closed
// form, edges are three shift-invariant tables, output times shorter than
// the two-edge overlap use directly integrated dense weights.
struct TriangleQuad {
    int W = 0;
    int m_sub = 1;
    int n_steps = 0;
    int n_cols = 0;
    double h = 0.0;
    double mu0 = 0.0;
    std::vector<double> abar, bbar;  // near-zero folded tables, c = 0..W-1
    std::vector<double> r_edge;      // near-t table, m = -(W-1)..(W-1)
    int small_rows = 0;
    std::vector<std::vector<double>> small_w;

    TriangleQuad(int n_steps_, int m_sub_, double h_, int n_cols_, int w)
        : W(w), m_sub(m_sub_), n_steps(n_steps_), n_cols(n_cols_), h(h_) {
        const double beta = 0.45 * M_PI * static_cast<double>(W);
        const double i0_beta = bessel_i0(beta);
        auto kernel = [&](double s) {
            const double a = std::abs(s);
            if (a >= static_cast<double>(W)) return 0.0;
            const double core = (a < 1e-12) ? 1.0 : std::sin(M_PI * s) / (M_PI * s);
            const double arg = 1.0 - (s / W) * (s / W);
            return core * bessel_i0(beta * std::sqrt(std::max(0.0, arg))) / i0_beta;
        };
        auto integrate = [&](double lo, double hi, auto f) {
            if (hi <= lo) return 0.0;
            const int n_iv = 2 * std::max(64, static_cast<int>(std::ceil((hi - lo) * 24)));
            const double ds = (hi - lo) / n_iv;
            double acc = 0.0;
            for (int i = 0; i <= n_iv; ++i) {
                const double s = lo + ds * i;
                const double simpson = (i == 0 || i == n_iv) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
                acc += simpson * f(s) * kernel(s);
            }
            return acc * ds / 3.0;
        };

        mu0 = integrate(-W, W, [](double) { return 1.0; });
        abar.assign(static_cast<size_t>(W), 0.0);
        bbar.assign(static_cast<size_t>(W), 0.0);
        for (int c = 0; c < W; ++c) {
            abar[static_cast<size_t>(c)] = integrate(-c, W, [](double) { return 1.0; });
            bbar[static_cast<size_t>(c)] = integrate(-c, W, [&](double s) { return c + s; });
            if (c > 0) {  // the mirrored sample at e = -c
                abar[static_cast<size_t>(c)] += integrate(c, W, [](double) { return 1.0; });
                bbar[static_cast<size_t>(c)] += integrate(c, W, [&](double s) { return s - c; });
            }
        }
        r_edge.assign(static_cast<size_t>(2 * W - 1), 0.0);
        for (int m = -(W - 1); m <= W - 1; ++m) {
            r_edge[static_cast<size_t>(m + W - 1)] =
                integrate(-W, m, [&](double s) { return m - s; });
        }

        small_rows = std::min(n_steps + 1, (2 * W + 2 + m_sub - 1) / m_sub + 1);
        small_w.resize(static_cast<size_t>(small_rows));
        for (int k = 0; k < small_rows; ++k) {
            const int kt = k * m_sub;
            const double t = kt * h;
            auto& wk = small_w[static_cast<size_t>(k)];
            wk.assign(static_cast<size_t>(std::min(n_cols, kt + W)), 0.0);
            for (int e = -(W - 1); e < kt + W; ++e) {
                const int c = std::abs(e);
                if (c >= static_cast<int>(wk.size())) continue;
                const double lo = std::max(static_cast<double>(-W), static_cast<double>(-e));
                const double hi = std::min(static_cast<double>(W), static_cast<double>(kt - e));
                wk[static_cast<size_t>(c)] +=
                    h * integrate(lo, hi, [&](double s) { return t - (e + s) * h; });
            }
        }
    }

    // s0[i] = sum of the first i samples, s1 likewise weighted by the index.
    double apply(const double* c_vals, const double* s0, const double* s1, int k) const {
        if (k < small_rows) {
            const auto& wk = small_w[static_cast<size_t>(k)];
            double acc = 0.0;
            for (size_t c = 0; c < wk.size(); ++c) acc += wk[c] * c_vals[c];
            return acc;
        }
        const int kt = k * m_sub;
        const double t = kt * h;
        double acc = 0.0;
        for (int c = 0; c < W; ++c) {
            acc += c_vals[c] *
                   h * (t * abar[static_cast<size_t>(c)] - h * bbar[static_cast<size_t>(c)]);
        }
        acc += h * h * mu0 *
               (kt * (s0[kt - W + 1] - s0[W]) - (s1[kt - W + 1] - s1[W]));
        for (int m = -(W - 1); m <= W - 1; ++m) {
            acc += h * h * r_edge[static_cast<size_t>(m + W - 1)] * c_vals[kt - m];
        }
        return acc;
    }
};

// One out-of-place complex FFT plan pair of fixed length, executed on
// caller-supplied buffers. fftw_execute_dft IS thread-safe; plan creation is
// not, so a FoldPlans is built once per segment in serial code and shared by
// the OpenMP fold threads. FFTW_UNALIGNED keeps new-array execution valid
// for plain std::vector storage.
struct FoldPlans {
    fftw_plan fwd = nullptr;
    fftw_plan bwd = nullptr;
    explicit FoldPlans(int len) {
        std::vector<cd> a(static_cast<size_t>(len)), b(static_cast<size_t>(len));
        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        fwd = fftw_plan_dft_1d(len, reinterpret_cast<fftw_complex*>(a.data()),
                               reinterpret_cast<fftw_complex*>(b.data()),
                               FFTW_FORWARD, flags);
        bwd = fftw_plan_dft_1d(len, reinterpret_cast<fftw_complex*>(a.data()),
                               reinterpret_cast<fftw_complex*>(b.data()),
                               FFTW_BACKWARD, flags);
    }
    ~FoldPlans() {
        if (fwd) fftw_destroy_plan(fwd);
        if (bwd) fftw_destroy_plan(bwd);
    }
    FoldPlans(const FoldPlans&) = delete;
    FoldPlans& operator=(const FoldPlans&) = delete;
    void forward(cd* in, cd* out) const {
        fftw_execute_dft(fwd, reinterpret_cast<fftw_complex*>(in),
                         reinterpret_cast<fftw_complex*>(out));
    }
    void backward(cd* in, cd* out) const {
        fftw_execute_dft(bwd, reinterpret_cast<fftw_complex*>(in),
                         reinterpret_cast<fftw_complex*>(out));
    }
};

// One CHUNK of stochastic samples' contributions to the correlator samples
// C_p(u) (accumulated u-major into cre_accum) and to the density moments.
// Complex double only: the trajectories and the dot table are intrinsically
// complex, and the tile GEMM goes through zgemm.
//
// BLOCKED over samples on the APPLY axis: every recursion (the g
// trajectory, the chain, the anchor advance) runs the whole chunk through
// one SpMM per term -- that is where the width-1 route lost 5-10x on
// per-apply efficiency. The dot structure is SAMPLE-DIAGONAL
// (G_s[q, c] = <W T_q r_s | VX_c> pairs a sample's chain with the SAME
// sample's trajectory columns), so the stores are laid out sample-major
// (per-sample contiguous column slabs) and the tile GEMMs run one slice per
// sample against its own vx slab. TRACE-FIRST, the slices accumulate into
// ONE aggregate table (everything downstream is linear in it, with
// geometry-only coefficients), so the table, its Allreduce, and the whole
// scalar fold are chunk-width-independent. Memory: segment store
// chunk * ~1.3 L N_local x 16 B; dot table q_seg * seg_cols x 16 B, once.
template <typename T>
void msd_spectral_accumulate(
    Chebyshev<T>& cheby,
    VelocityOp<T> velocity,
    const DistMultiVector<T>& rblk,
    int n_kpm,
    const SpectralGrid& grid,
    int block_substeps,
    double segment_scaled,
    std::vector<double>& cre_accum,  // [n_cols * n_kpm], u-major, +=
    std::vector<cd>& rho_accum       // [n_kpm], += (sum over samples)
) {
    static_assert(std::is_same_v<T, std::complex<double>>,
                  "corr_msd_spectral runs in complex double");
    DistGraph* graph = rblk.graph;
    const int rank = graph->rank;
    const int nb = rblk.num_vectors;
    const double h = grid.h;
    const int n_cols = grid.n_cols;

    // Column copy between multivectors, ld-aware (DMV rows are stride ld,
    // padded -- never copy across rows).
    auto copy_col = [](DistMultiVector<T>& dst, int dcol,
                       const DistMultiVector<T>& src, int scol) {
        for (int row = 0; row < src.local_rows; ++row) {
            dst.data[static_cast<size_t>(row) * dst.ld + dcol] =
                src.data[static_cast<size_t>(row) * src.ld + scol];
        }
    };

    // Velocity placement: see the header note. The seed w = D r is physical
    // and stays dressed in every form; the paired velocity and the chain run
    // bare when the Solve metric cancels the dressing.
    const bool bare_velocity =
        !velocity.composite() && cheby.form != ChebyOpForm::Sandwich;
    const bool weighted = (cheby.form == ChebyOpForm::Solve) && !bare_velocity;
    auto apply_velocity_paired = [&](DistMultiVector<T>& in, DistMultiVector<T>& out) {
        if (bare_velocity) {
            velocity.matrix->mult_dense(in, out);
        } else {
            cheby.apply_dual(velocity, in, out, ChebyMode::NORMAL, 1.0);
        }
    };

    DistMultiVector<T> w(graph, nb);
    cheby.apply_dual(velocity, const_cast<DistMultiVector<T>&>(rblk), w,
                     ChebyMode::NORMAL, 1.0);

    // Block layout in grid points: block b's m = 0 state sits at seed_point
    // and its offsets k = 1..len cover the next len points; the first
    // block's m = 0 state IS grid point 0.
    int b_sub = block_substeps;
    if (b_sub <= 0) b_sub = std::max(1, static_cast<int>(std::ceil(256.0 / h)));

    struct Block {
        int seed_point = 0;
        int len = 0;
        int n_states = 0;
        int col0 = 0;     // first column in its segment's VX store
        int q_need = 0;   // chain length its last grid point requires
    };
    std::vector<Block> blocks;
    {
        std::vector<cd> probe;
        int covered = 0;
        while (covered < n_cols - 1) {
            Block b;
            b.seed_point = covered;
            b.len = std::min(b_sub, n_cols - 1 - b.seed_point);
            timeprop_coeff(b.len * h, 1e-14, true, probe);
            b.n_states = static_cast<int>(probe.size());
            blocks.push_back(b);
            covered = b.seed_point + b.len;
        }
    }

    // Forward Bessel tables, hoisted: the offset-k expansion depends only on
    // k h, which every block (and every chunk) shares.
    std::vector<std::vector<cd>> cfwd_tab(static_cast<size_t>(b_sub) + 1);
    cfwd_tab[0] = {cd(1.0, 0.0)};
    for (int k = 1; k <= b_sub; ++k) {
        timeprop_coeff(k * h, 1e-14, true, cfwd_tab[static_cast<size_t>(k)]);
    }

    // Segments: contiguous groups of whole blocks covering ~segment_scaled
    // in scaled time. The expansion anchor re-seeds at every segment start.
    double seg_target = segment_scaled;
    if (seg_target <= 0.0) {
        seg_target = std::max(192.0, std::min(768.0, static_cast<double>(n_kpm)));
    }
    const int seg_pts = std::max(1, static_cast<int>(std::ceil(seg_target / h)));
    std::vector<std::pair<int, int>> segments;  // block index ranges [b0, b1)
    {
        int b0 = 0;
        while (b0 < static_cast<int>(blocks.size())) {
            int b1 = b0, pts = 0;
            while (b1 < static_cast<int>(blocks.size()) && pts < seg_pts) {
                pts += blocks[static_cast<size_t>(b1)].len;
                ++b1;
            }
            segments.emplace_back(b0, b1);
            b0 = b1;
        }
    }

    DistMultiVector<T> t0(graph, nb), t1(graph, nb), t2(graph, nb),
        acc_end(graph, nb), vx_tmp(graph, nb), wq(graph, nb);
    DistMultiVector<T> g_curr(graph, nb), r_seg(graph, nb);
    g_curr.copy_from(w);
    r_seg.copy_from(rblk);

    const int tile_w = 32;
    DistMultiVector<T> ltile(graph, tile_w * nb);  // sample-major tile slabs

    for (size_t iseg = 0; iseg < segments.size(); ++iseg) {
        const int s0 = segments[iseg].first;
        const int s1 = segments[iseg].second;
        const int seg_start = blocks[static_cast<size_t>(s0)].seed_point;
        const int seg_end = blocks[static_cast<size_t>(s1 - 1)].seed_point +
                            blocks[static_cast<size_t>(s1 - 1)].len;
        const int seg_len = seg_end - seg_start;

        // Segment-relative expansion lengths M(delta), and the backward
        // coefficient table alongside (forward and backward truncate
        // identically -- same Bessel magnitudes -- and the backward
        // coefficients are the forward ones' conjugates).
        std::vector<int> mrel(static_cast<size_t>(seg_len) + 1, 1);
        std::vector<std::vector<cd>> bwd_tab(static_cast<size_t>(seg_len) + 1);
        bwd_tab[0] = {cd(1.0, 0.0)};
        {
            std::vector<cd> probe;
            for (int d = 1; d <= seg_len; ++d) {
                timeprop_coeff(d * h, 1e-14, true, probe);
                mrel[static_cast<size_t>(d)] = static_cast<int>(probe.size());
                auto& bw = bwd_tab[static_cast<size_t>(d)];
                bw.resize(probe.size());
                for (size_t j = 0; j < probe.size(); ++j) bw[j] = std::conj(probe[j]);
            }
        }
        int seg_cols = 0;
        for (int b = s0; b < s1; ++b) {
            Block& blk = blocks[static_cast<size_t>(b)];
            blk.col0 = seg_cols;
            seg_cols += blk.n_states;
            blk.q_need = n_kpm + mrel[static_cast<size_t>(blk.seed_point + blk.len - seg_start)] - 1;
        }
        const int q_seg = blocks[static_cast<size_t>(s1 - 1)].q_need;
        const int n_q = q_seg;

        // ---- g trajectory through this segment's blocks, storing VX --------
        // Sample-major store: sample s's columns are the contiguous slab
        // [s * seg_cols, (s + 1) * seg_cols) -- each tile GEMM below slices
        // its own sample's slab.
        DistMultiVector<T> vx(graph, seg_cols * nb);
        auto store_vx = [&](int c, const DistMultiVector<T>& block_cols) {
            for (int s = 0; s < nb; ++s) {
                copy_col(vx, s * seg_cols + c, block_cols, s);
            }
        };
        for (int b = s0; b < s1; ++b) {
            const Block& blk = blocks[static_cast<size_t>(b)];
            std::vector<cd> cend;
            timeprop_coeff(blk.len * h, 1e-14, true, cend);
            t0.copy_from(g_curr);
            apply_velocity_paired(t0, vx_tmp);
            store_vx(blk.col0, vx_tmp);
            acc_end.copy_from(t0);
            acc_end.scale(CoeffCast<T>::apply(cend[0]));
            if (blk.n_states > 1) {
                cheby.apply_op(t0, t1, ChebyMode::NORMAL);
                apply_velocity_paired(t1, vx_tmp);
                store_vx(blk.col0 + 1, vx_tmp);
                acc_end.axpy(CoeffCast<T>::apply(cend[1]), t1);
            }
            for (int m = 2; m < blk.n_states; ++m) {
                ::rsatb::util::throw_if_interrupted(graph->comm);
                cheby.apply_op(t1, t2, ChebyMode::NORMAL);
                t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
                apply_velocity_paired(t2, vx_tmp);
                store_vx(blk.col0 + m, vx_tmp);
                acc_end.axpy(CoeffCast<T>::apply(cend[static_cast<size_t>(m)]), t2);
                t0.swap(t1);
                t1.swap(t2);
            }
            g_curr.copy_from(acc_end);
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kSummary)) {
                std::cout << "msd-spectral: evolve " << (blk.seed_point + blk.len)
                          << " / " << (n_cols - 1) << std::endl;
            }
        }

        // ---- chain on the segment anchor r_s, trapezoidal tile GEMMs --------
        // TRACE-FIRST: everything downstream of the dot table is linear in
        // it, and its coefficients depend only on grid geometry -- so the
        // per-sample tables G_s are summed into ONE aggregate table the
        // moment they are formed. The dot structure stays SAMPLE-DIAGONAL
        // (one GEMM slice per sample against its own vx slab, same total
        // dot flops), but the slices accumulate (beta = 1) into a single
        // tile buffer: table memory, the Allreduce volume, and the whole
        // scalar fold below shrink by the chunk width. Exactly the same
        // estimator, to summation order.
        std::vector<cd> gtab(static_cast<size_t>(q_seg) * seg_cols, cd(0.0, 0.0));
        std::vector<cd> tile_buf(static_cast<size_t>(tile_w) * seg_cols);

        auto flush_tile = [&](int q_start, int width) {
            using isqrt::vbcsr_complex_double;
            int col_from = seg_cols;
            for (int b = s0; b < s1; ++b) {
                if (blocks[static_cast<size_t>(b)].q_need > q_start) {
                    col_from = blocks[static_cast<size_t>(b)].col0;
                    break;
                }
            }
            if (col_from >= seg_cols) return;
            const vbcsr_lapack_int m_ = width;
            const vbcsr_lapack_int n_ = seg_cols - col_from;
            const vbcsr_lapack_int k_ = ltile.local_rows;
            const vbcsr_lapack_int lda_ = ltile.ld;
            const vbcsr_lapack_int ldb_ = vx.ld;
            const vbcsr_lapack_int ldc_ = width;
            const cd one(1.0, 0.0), zero(0.0, 0.0);
            const size_t slab = static_cast<size_t>(width) * n_;
            // Row-major storage read as column-major is the transpose, so
            // this computes C(q, c) = sum_rows L[row, q] * conj(VX[row, c]);
            // the wanted G = sum_rows conj(L) * VX is its conjugate. Column
            // offsets select sample s's tile slab and vx slab; beta chains
            // the samples into one aggregate tile.
            for (int s = 0; s < nb; ++s) {
                isqrt::zgemm_("N", "C", &m_, &n_, &k_,
                              reinterpret_cast<const vbcsr_complex_double*>(&one),
                              reinterpret_cast<const vbcsr_complex_double*>(
                                  ltile.data.data()) + s * tile_w,
                              &lda_,
                              reinterpret_cast<const vbcsr_complex_double*>(
                                  vx.data.data()) + s * seg_cols + col_from,
                              &ldb_,
                              reinterpret_cast<const vbcsr_complex_double*>(
                                  s == 0 ? &zero : &one),
                              reinterpret_cast<vbcsr_complex_double*>(
                                  tile_buf.data()), &ldc_);
            }
            MPI_Allreduce(MPI_IN_PLACE, tile_buf.data(),
                          2 * static_cast<int>(slab), MPI_DOUBLE, MPI_SUM,
                          graph->comm);
            for (int q = 0; q < width; ++q) {
                cd* row = gtab.data() + static_cast<size_t>(q_start + q) * seg_cols;
                for (int c = col_from; c < seg_cols; ++c) {
                    row[c] = std::conj(
                        tile_buf[static_cast<size_t>(c - col_from) * width + q]);
                }
            }
        };

        auto stash = [&](int q, const DistMultiVector<T>& state) {
            const DistMultiVector<T>* src = &state;
            if (weighted) {
                cheby.S->mult_dense(const_cast<DistMultiVector<T>&>(state), wq);
                src = &wq;
            }
            for (int s = 0; s < nb; ++s) {
                copy_col(ltile, s * tile_w + (q % tile_w), *src, s);
            }
            if ((q % tile_w) == tile_w - 1 || q == q_seg - 1) {
                flush_tile(q - (q % tile_w), (q % tile_w) + 1);
            }
        };

        t0.copy_from(r_seg);
        stash(0, t0);
        if (q_seg > 1) {
            cheby.apply_op(t0, t1, ChebyMode::NORMAL);
            stash(1, t1);
        }
        const int stride = std::max(1, q_seg / 10);
        for (int q = 2; q < q_seg; ++q) {
            ::rsatb::util::throw_if_interrupted(graph->comm);
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
            stash(q, t2);
            t0.swap(t1);
            t1.swap(t2);
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kSummary) &&
                ::rsatb::util::every_nth(q - 2, stride)) {
                std::cout << "msd-spectral: segment " << (iseg + 1) << " / "
                          << segments.size() << " chain " << q << " / " << q_seg
                          << std::endl;
            }
        }

        // ---- scalar fold on the aggregate table: D by GEMM, C_p by FFT ------
        // d_q(u) = sum_m c_m(k) G[q, col0 + m] is one ZGEMM per block over
        // its grid points, and the Bessel fold
        //   C_p = sum_j b_j (d_{p+j} + d_{|p-j|}) / 2
        // is one Toeplitz-plus-Hankel product: with the even extension
        // Dt_t = d_{|t|} and the symmetric kernel kap_0 = b_0,
        // kap_{+-j} = b_j / 2, it is the single linear convolution
        // C_p = (kap * Dt)_p, evaluated by zero-padded FFTs -- O(L log L)
        // per grid point instead of O(n_kpm M(delta)). Each rank takes a
        // contiguous share of every block's points (its D columns come from
        // its own GEMM slice), so the fold scales with ranks and cre_accum
        // stays partial until reduce_correlator.
        {
            const int max_nj = mrel[static_cast<size_t>(seg_len)];
            int lfft = 1;
            while (lfft < n_q + 3 * max_nj) lfft <<= 1;
            const FoldPlans plans(lfft);
            const double inv_l = 1.0 / static_cast<double>(lfft);

            for (int b = s0; b < s1; ++b) {
                const Block& blk = blocks[static_cast<size_t>(b)];
                const int k_lo = (blk.seed_point == 0) ? 0 : 1;
                const int n_pts = blk.len - k_lo + 1;
                const int per = (n_pts + graph->size - 1) / graph->size;
                const int p_from = std::min(n_pts, rank * per);
                const int p_to = std::min(n_pts, p_from + per);
                if (p_from >= p_to) continue;
                const int q_u = blk.q_need;
                const int my_pts = p_to - p_from;

                // This rank's forward-coefficient columns, zero-padded to the
                // block's state count.
                std::vector<cd> cmat(static_cast<size_t>(blk.n_states) * my_pts,
                                     cd(0.0, 0.0));
                for (int ik = 0; ik < my_pts; ++ik) {
                    const int k = k_lo + p_from + ik;
                    const std::vector<cd>& cf = cfwd_tab[static_cast<size_t>(k)];
                    const int n_m = std::min<int>(static_cast<int>(cf.size()),
                                                  blk.n_states);
                    std::copy(cf.begin(), cf.begin() + n_m,
                              cmat.begin() + static_cast<size_t>(ik) * blk.n_states);
                }
                std::vector<cd> dtab(static_cast<size_t>(q_u) * my_pts);
                {
                    using isqrt::vbcsr_complex_double;
                    const vbcsr_lapack_int m_ = q_u;
                    const vbcsr_lapack_int n_ = my_pts;
                    const vbcsr_lapack_int k_ = blk.n_states;
                    const vbcsr_lapack_int lda_ = seg_cols;
                    const vbcsr_lapack_int ldb_ = blk.n_states;
                    const vbcsr_lapack_int ldc_ = q_u;
                    const cd one(1.0, 0.0), zero(0.0, 0.0);
                    // gtab is q-major (row-major over [q_seg, seg_cols]);
                    // read as column-major it is the [seg_cols, q_seg]
                    // transpose, so "T" recovers D[q, k].
                    isqrt::zgemm_("T", "N", &m_, &n_, &k_,
                                  reinterpret_cast<const vbcsr_complex_double*>(&one),
                                  reinterpret_cast<const vbcsr_complex_double*>(
                                      gtab.data()) + blk.col0,
                                  &lda_,
                                  reinterpret_cast<const vbcsr_complex_double*>(
                                      cmat.data()),
                                  &ldb_,
                                  reinterpret_cast<const vbcsr_complex_double*>(&zero),
                                  reinterpret_cast<vbcsr_complex_double*>(dtab.data()),
                                  &ldc_);
                }

#pragma omp parallel
                {
                    std::vector<cd> fa(static_cast<size_t>(lfft));
                    std::vector<cd> fb(static_cast<size_t>(lfft));
                    std::vector<cd> fc(static_cast<size_t>(lfft));
#pragma omp for schedule(dynamic, 4)
                    for (int ik = 0; ik < my_pts; ++ik) {
                        const int k = k_lo + p_from + ik;
                        const int u = blk.seed_point + k;
                        const int delta = u - seg_start;
                        const int n_j = mrel[static_cast<size_t>(delta)];
                        const int q_pt = n_kpm + n_j - 1;
                        const cd* dcol = dtab.data() + static_cast<size_t>(ik) * q_u;
                        // Even extension Dt(a) = d_{|a - (n_j - 1)|}; the
                        // convolution index p + 2 (n_j - 1) is C_p.
                        std::fill(fa.begin(), fa.end(), cd(0.0, 0.0));
                        for (int a = 0; a < q_pt + n_j - 1; ++a) {
                            fa[static_cast<size_t>(a)] =
                                dcol[std::abs(a - (n_j - 1))];
                        }
                        plans.forward(fa.data(), fc.data());
                        const std::vector<cd>& bw =
                            bwd_tab[static_cast<size_t>(delta)];
                        std::fill(fb.begin(), fb.end(), cd(0.0, 0.0));
                        fb[static_cast<size_t>(n_j - 1)] = bw[0];
                        for (int j = 1; j < n_j; ++j) {
                            const cd half = 0.5 * bw[static_cast<size_t>(j)];
                            fb[static_cast<size_t>(n_j - 1 + j)] = half;
                            fb[static_cast<size_t>(n_j - 1 - j)] = half;
                        }
                        plans.forward(fb.data(), fa.data());
                        for (int i = 0; i < lfft; ++i) {
                            fc[static_cast<size_t>(i)] *= fa[static_cast<size_t>(i)];
                        }
                        plans.backward(fc.data(), fa.data());
                        double* cre = cre_accum.data() + static_cast<size_t>(u) * n_kpm;
                        const int off = 2 * (n_j - 1);
                        for (int p = 0; p < n_kpm; ++p) {
                            cre[p] += fa[static_cast<size_t>(p + off)].real() * inv_l;
                        }
                    }
                }
            }
        }

        // ---- advance the anchor to the next segment start -------------------
        if (iseg + 1 < segments.size()) {
            std::vector<cd> cend;
            timeprop_coeff(seg_len * h, 1e-14, true, cend);
            t0.copy_from(r_seg);
            acc_end.copy_from(t0);
            acc_end.scale(CoeffCast<T>::apply(cend[0]));
            if (cend.size() > 1) {
                cheby.apply_op(t0, t1, ChebyMode::NORMAL);
                acc_end.axpy(CoeffCast<T>::apply(cend[1]), t1);
            }
            for (size_t m = 2; m < cend.size(); ++m) {
                ::rsatb::util::throw_if_interrupted(graph->comm);
                cheby.apply_op(t1, t2, ChebyMode::NORMAL);
                t2.axpby(static_cast<T>(-1.0), t0, static_cast<T>(2.0));
                acc_end.axpy(CoeffCast<T>::apply(cend[m]), t2);
                t0.swap(t1);
                t1.swap(t2);
            }
            r_seg.copy_from(acc_end);
        }
    }

    // Density moments of the same samples, for the shared normalisation.
    // metric_density_moments averages over the block's columns; rho_accum
    // carries the SUM over samples, so scale back up by the chunk width.
    const std::vector<cd> rho = metric_density_moments(cheby, rblk, n_kpm);
    for (int p = 0; p < n_kpm; ++p) {
        rho_accum[static_cast<size_t>(p)] += rho[static_cast<size_t>(p)] *
                                             static_cast<double>(nb);
    }
}

// The assembly above leaves each rank its round-robin share of the
// correlator samples; one reduction after the last sample makes them whole.
// (The density moments are already global: metric_density_moments dots
// reduce internally.)
inline void reduce_correlator(std::vector<double>& cre_accum, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, cre_accum.data(),
                  static_cast<int>(cre_accum.size()), MPI_DOUBLE, MPI_SUM, comm);
}

// `n_acc` is the accumulated moment count; when it is below `n_kpm_out` the
// rows are extended by linear prediction (moment_extrapolation.hpp) before
// the shared Jackson reconstruction, exactly as in the stepwise route's
// n_compute knob.
template <typename T>
void msd_spectral_finalize(
    Chebyshev<T>& cheby,
    int n_steps,
    int n_acc,
    int n_kpm_out,
    double inv_samples,
    const SpectralGrid& grid,
    const TriangleQuad& quad,
    const std::vector<double>& cre_accum,
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

    // num_p(t_k) = (2 / hw^2) int_0^{t_k} (t_k - u) C_p(u) du; prefix sums
    // make every output time O(kernel width) per moment.
    const double hw = cheby.half_width;
    const double scale = 2.0 / (hw * hw) * inv_samples;
    const int n_cols = grid.n_cols;
    std::vector<double> num(static_cast<size_t>(n_steps + 1) * n_acc, 0.0);
#pragma omp parallel
    {
        std::vector<double> c_vals(static_cast<size_t>(n_cols));
        std::vector<double> s0(static_cast<size_t>(n_cols) + 1, 0.0);
        std::vector<double> s1(static_cast<size_t>(n_cols) + 1, 0.0);
#pragma omp for schedule(static)
        for (int p = 0; p < n_acc; ++p) {
            for (int u = 0; u < n_cols; ++u) {
                c_vals[static_cast<size_t>(u)] = cre_accum[static_cast<size_t>(u) * n_acc + p];
                s0[static_cast<size_t>(u) + 1] = s0[static_cast<size_t>(u)] + c_vals[static_cast<size_t>(u)];
                s1[static_cast<size_t>(u) + 1] =
                    s1[static_cast<size_t>(u)] + static_cast<double>(u) * c_vals[static_cast<size_t>(u)];
            }
            for (int k = 0; k <= n_steps; ++k) {
                num[static_cast<size_t>(k) * n_acc + p] =
                    scale * quad.apply(c_vals.data(), s0.data(), s1.data(), k);
            }
        }
    }

    msd_out.assign(static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm_out), 0.0);
    std::vector<cd> row(static_cast<size_t>(n_acc));
    for (int k = 0; k <= n_steps; ++k) {
        for (int p = 0; p < n_acc; ++p) {
            row[static_cast<size_t>(p)] = cd(num[static_cast<size_t>(k) * n_acc + p], 0.0);
        }
        std::vector<cd> extended(row);
        extend(extended);
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(extended, msd_spectrum, nullptr, true);
        write_msd_row(density_spectrum, msd_spectrum, k, n_kpm_out, msd_out);
    }
}

}  // namespace detail

// Apply-count comparison for the auto route dispatch on expensive-apply
// forms. Both routes' dominant cost there is applies of the SAME operator,
// so the counts compare machine-independently -- no flop rates, no
// bandwidth model. Dots and S-multiplies are left out on purpose: they
// decide the route only when applies are cheap, which is the orthogonal
// case the caller dispatches structurally and never sends here.
//
//   stepwise  two propagated states (the commutator pair) per output step
//             plus one doubled moment run per output row;
//   spectral  the g and anchor trajectories plus the chain amortised over
//             segments, ~(3 + n_kpm / L) applies per unit scaled time.
//
// The model reproduces the measured sweep in doc/design_msd_spectral.md
// section 6 (predicting 0.9x and 2.7x where 1.10x and 2.78x were measured).
// The regime it exists for is sparse output sampling of very long windows
// -- dt_step beyond ~2 L / lambda in scaled units -- where the per-output
// moment runs amortise away while the chain term does not.
//
// `sample_width` is the stochastic block's column count. The two routes run
// their applies at different SpMM widths -- stepwise sweeps the whole block,
// the spectral chunks cap at kSpectralSampleChunk = 16 -- and per-apply
// cost is NOT width-independent (a width-1 SpMV is overhead-bound; measured
// 4x end-to-end on a 1664-orbital sandwich block of 64). The counts are
// therefore derated by the coarse saturating efficiency eff(w) = w/(w + 3):
// machine-independence survives as a RATIO of widths, not a flop model, and
// at equal widths the factor is exactly 1. The misroute this prevents was
// measured 2026-08-27 (silicon 64-sample block, pre-blocking: auto chose
// spectral at 4x the stepwise wall).
template <typename T>
bool spectral_msd_cheaper(
    const Chebyshev<T>& cheby,
    int n_steps,
    double dt_fs,
    int n_kpm,
    int n_kpm_compute = 0,
    int time_oversample = 2,
    double segment_scaled = 0.0,
    int sample_width = 1
) {
    if (n_steps <= 0 || dt_fs <= 0.0 || n_kpm <= 0) return false;
    const int nc = (n_kpm_compute > 0 && n_kpm_compute < n_kpm) ? n_kpm_compute : n_kpm;
    const double hbar_ev_fs = 0.6582119569;
    const double tau_step = cheby.half_width * dt_fs / hbar_ev_fs;
    std::vector<cd> probe;
    timeprop_coeff(tau_step, 1e-14, true, probe);
    const double stepwise = 2.0 * n_steps * static_cast<double>(probe.size()) +
                            (n_steps + 2) * 0.5 * nc;
    double seg = segment_scaled;
    if (seg <= 0.0) seg = std::max(192.0, std::min(768.0, static_cast<double>(nc)));
    const detail::SpectralGrid grid(cheby.half_width, dt_fs, n_steps, time_oversample);
    const double window = grid.n_cols * grid.h;  // includes the kernel pad
    auto eff = [](int w) { return static_cast<double>(w) / (w + 3.0); };
    // The stepwise readout time-tiles its recurrences to a combined width of
    // at least 16 columns (msd.hpp), so its effective width is never below
    // the spectral chunk cap.
    const int w_step = std::max(std::max(1, sample_width), 16);
    const int w_spec = std::min(std::max(1, sample_width), 16);  // kSpectralSampleChunk
    const double width_penalty = eff(w_step) / eff(w_spec);
    const double spectral =
        window * (3.0 + static_cast<double>(nc) / seg) * width_penalty;
    return spectral < stepwise;
}

// Spectral-route periodic/any-basis MSD: same contract as the commutator
// overload of corr_msd ([n_steps + 1, n_kpm] density-normalised rows on the
// ascending Chebyshev grid), computed at resolution-plus-evolution cost,
// linear in the window through segmented anchoring. See the header note and
// doc/design_msd_spectral.md. `n_kpm_compute` (0 = off) computes that many
// moments and extends to n_kpm by linear prediction; `segment_scaled` sets
// the anchor segment length in scaled time (0 = auto), which bounds memory
// and trades chain amortisation against dot cost.
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
    int n_kpm_compute = 0,
    double segment_scaled = 0.0
) {
    if (n_steps < 0) throw std::runtime_error("n_steps must be non-negative");
    if (n_kpm <= 0) throw std::runtime_error("n_kpm must be positive");
    if (dt_fs <= 0.0) throw std::runtime_error("dt_fs must be positive");
    if constexpr (!std::is_same_v<T, std::complex<double>>) {
        throw std::runtime_error("corr_msd_spectral needs complex double scalars");
    } else {
        const int nc = (n_kpm_compute > 0 && n_kpm_compute < n_kpm) ? n_kpm_compute : n_kpm;
        detail::SpectralGrid grid(cheby.half_width, dt_fs, n_steps, time_oversample);
        detail::TriangleQuad quad(n_steps, grid.m_sub, grid.h, grid.n_cols,
                                  detail::SpectralGrid::kKernelHalfWidth);
        std::vector<double> cre(static_cast<size_t>(grid.n_cols) * nc, 0.0);
        std::vector<cd> rho(static_cast<size_t>(nc), cd(0.0, 0.0));
        DistMultiVector<T> one_col(psi_in.graph, 1);
        one_col.set_col(0, psi_in);
        detail::msd_spectral_accumulate(cheby, velocity, one_col, nc, grid,
                                        block_substeps, segment_scaled, cre, rho);
        detail::reduce_correlator(cre, psi_in.graph->comm);
        detail::msd_spectral_finalize(cheby, n_steps, nc, n_kpm, 1.0, grid, quad, cre,
                                      rho, msd_out, energy_out);
    }
}

// Sample-block overload: columns are independent stochastic samples; their
// correlator samples and density moments are averaged BEFORE the spectrum
// ratio, matching the multivector semantics of the stepwise route. Samples
// run through the blocked accumulate in chunks of kSpectralSampleChunk:
// every recursion then advances the whole chunk per SpMM (the width-1 route
// measured 4x slower end to end from per-apply overhead alone), the dot
// table is trace-first aggregated over each chunk, and only the segment
// store scales with the CHUNK -- the chunk is the memory knob (segment
// store chunk * ~1.3 L N_local x 16 B), and 16 already sits on the SpMM
// efficiency plateau.
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
    int n_kpm_compute = 0,
    double segment_scaled = 0.0,
    int sample_chunk = 0
) {
    if (n_steps < 0) throw std::runtime_error("n_steps must be non-negative");
    if (n_kpm <= 0) throw std::runtime_error("n_kpm must be positive");
    if (dt_fs <= 0.0) throw std::runtime_error("dt_fs must be positive");
    if constexpr (!std::is_same_v<T, std::complex<double>>) {
        throw std::runtime_error("corr_msd_spectral needs complex double scalars");
    } else {
        constexpr int kSpectralSampleChunk = 16;
        const int chunk_cap =
            sample_chunk > 0 ? sample_chunk : kSpectralSampleChunk;
        const int nc = (n_kpm_compute > 0 && n_kpm_compute < n_kpm) ? n_kpm_compute : n_kpm;
        detail::SpectralGrid grid(cheby.half_width, dt_fs, n_steps, time_oversample);
        detail::TriangleQuad quad(n_steps, grid.m_sub, grid.h, grid.n_cols,
                                  detail::SpectralGrid::kKernelHalfWidth);
        std::vector<double> cre(static_cast<size_t>(grid.n_cols) * nc, 0.0);
        std::vector<cd> rho(static_cast<size_t>(nc), cd(0.0, 0.0));
        DistMultiVector<T>& samples = const_cast<DistMultiVector<T>&>(psi_in);
        for (int col0 = 0; col0 < psi_in.num_vectors; col0 += chunk_cap) {
            const int nb = std::min(chunk_cap, psi_in.num_vectors - col0);
            DistMultiVector<T> chunk(psi_in.graph, nb);
            for (int s = 0; s < nb; ++s) {
                chunk.set_col(s, samples.get_col(col0 + s));
            }
            detail::msd_spectral_accumulate(cheby, velocity, chunk, nc, grid,
                                            block_substeps, segment_scaled, cre, rho);
        }
        detail::reduce_correlator(cre, psi_in.graph->comm);
        detail::msd_spectral_finalize(cheby, n_steps, nc, n_kpm,
                                      1.0 / static_cast<double>(psi_in.num_vectors),
                                      grid, quad, cre, rho, msd_out, energy_out);
    }
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_MSD_SPECTRAL_HPP
