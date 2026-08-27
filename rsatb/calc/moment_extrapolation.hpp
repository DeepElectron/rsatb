#ifndef RSATB_CALC_MOMENT_EXTRAPOLATION_HPP
#define RSATB_CALC_MOMENT_EXTRAPOLATION_HPP

// Linear-prediction extrapolation of Chebyshev moment sequences.
//
// A Chebyshev moment sequence mu_p = sum_a w_a T_p(x_a) = sum_a w_a cos(p
// theta_a) is a mixture of sampled (near-)undamped sinusoids -- exactly the
// signal class autoregressive models represent. Fitting an AR(L) model to
// the computed part of the sequence and iterating it forward extends the
// sequence at zero operator cost, which the DMRG community established as a
// resolution multiplier for Chebyshev spectral functions: Ganahl,
// Thunstroem, Verstraete, Held, Evertz, PRB 90, 045144 (2014) and Wolf,
// Justiniano, McCulloch, Schollwoeck, PRB 91, 115144 (2015), building on
// White & Affleck's linear prediction for time series, PRB 77, 134437
// (2008).
//
// Following those references, the model is fit by covariance least squares
// (Tikhonov-regularised normal equations) and iterated through the
// companion-matrix eigendecomposition with poles outside the unit circle
// renormalised onto it. NOT Burg's lattice method: Burg's greedy stagewise
// fit carries a frequency bias for undamped sinusoids (poles land at
// ~0.999 instead of 1), which decays and dephases long extrapolations --
// measured at O(1) error over a 5x extension where the least-squares fit is
// exact to 1e-8 over 50x.
//
// Contract notes:
// - The sequences this codebase feeds here (metric-weighted density and MSD
//   moments) are real in exact arithmetic; the fit runs on the real part and
//   the extrapolated entries carry zero imaginary part. Computed entries are
//   returned unchanged.
// - Extrapolated moments pass through the SAME Jackson reconstruction as
//   computed ones (kpm.hpp), at the extended length: the kernel damps
//   whatever the model got wrong at high order, so the failure mode is
//   losing the resolution gain, not spraying artifacts.
// - This is an explicit opt-in everywhere it is wired (a knob, never an
//   inference): extrapolation trades a provable computation for a model
//   fit, and the caller owns that trade.

#include "lapack_isqrt.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace rsatb {
namespace calc {

namespace lp_detail {

using vbcsr::vbcsr_complex_double;
using vbcsr::vbcsr_lapack_int;

extern "C" {
void zgeev_(const char* jobvl, const char* jobvr, const vbcsr_lapack_int* n,
            vbcsr_complex_double* a, const vbcsr_lapack_int* lda,
            vbcsr_complex_double* w, vbcsr_complex_double* vl,
            const vbcsr_lapack_int* ldvl, vbcsr_complex_double* vr,
            const vbcsr_lapack_int* ldvr, vbcsr_complex_double* work,
            const vbcsr_lapack_int* lwork, double* rwork, vbcsr_lapack_int* info);
void zgesv_(const vbcsr_lapack_int* n, const vbcsr_lapack_int* nrhs,
            vbcsr_complex_double* a, const vbcsr_lapack_int* lda,
            vbcsr_lapack_int* ipiv, vbcsr_complex_double* b,
            const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
void dposv_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* nrhs,
            double* a, const vbcsr_lapack_int* lda, double* b,
            const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
}

// AR coefficients a[0..order-1] of x[t] ~ sum_j a[j] x[t-1-j], covariance
// least squares over the whole handed sequence with Tikhonov shift
// delta * tr(R)/order. Returns empty on solver failure.
inline std::vector<double> ar_least_squares(const std::vector<double>& x, int order,
                                            double delta) {
    const int n = static_cast<int>(x.size());
    std::vector<double> R(static_cast<size_t>(order) * order, 0.0);
    std::vector<double> rhs(static_cast<size_t>(order), 0.0);
    for (int t = order; t < n; ++t) {
        for (int i = 0; i < order; ++i) {
            const double xi = x[static_cast<size_t>(t - 1 - i)];
            rhs[static_cast<size_t>(i)] += xi * x[static_cast<size_t>(t)];
            for (int j = i; j < order; ++j) {
                R[static_cast<size_t>(i) * order + j] += xi * x[static_cast<size_t>(t - 1 - j)];
            }
        }
    }
    double trace = 0.0;
    for (int i = 0; i < order; ++i) trace += R[static_cast<size_t>(i) * order + i];
    for (int i = 0; i < order; ++i) {
        R[static_cast<size_t>(i) * order + i] += delta * trace / order;
        for (int j = 0; j < i; ++j) {
            R[static_cast<size_t>(i) * order + j] = R[static_cast<size_t>(j) * order + i];
        }
    }
    const vbcsr_lapack_int n_ = order, one = 1;
    vbcsr_lapack_int info = 0;
    // Column-major vs row-major is moot: R is symmetric.
    dposv_("U", &n_, &one, R.data(), &n_, rhs.data(), &n_, &info);
    if (info != 0) return {};
    return rhs;
}

}  // namespace lp_detail

// Extend a Chebyshev moment sequence from its computed length to n_target by
// linear prediction. `order` 0 picks the default min(max(8, n/4), 200);
// sequences too short to train, or a failed fit, return the zero-padded
// sequence (the honest fallback: no model, no invented content).
inline std::vector<std::complex<double>> extrapolate_moments_lp(
    const std::vector<std::complex<double>>& moments,
    int n_target,
    int order = 0,
    double delta = 1e-10
) {
    using cd = std::complex<double>;
    using lp_detail::vbcsr_complex_double;
    using lp_detail::vbcsr_lapack_int;

    const int n = static_cast<int>(moments.size());
    std::vector<cd> out(moments);
    if (n_target <= n) {
        out.resize(static_cast<size_t>(std::max(n_target, 0)));
        return out;
    }
    out.resize(static_cast<size_t>(n_target), cd(0.0, 0.0));

    if (order <= 0) order = std::min(std::max(8, n / 4), 200);
    if (n < 2 * order || order < 2) return out;

    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = moments[static_cast<size_t>(i)].real();
    const std::vector<double> a = lp_detail::ar_least_squares(x, order, delta);
    if (a.empty()) return out;

    // Companion matrix (column-major), complex so zgeev needs no unpacking.
    const int L = order;
    std::vector<cd> comp(static_cast<size_t>(L) * L, cd(0.0, 0.0));
    for (int j = 0; j < L; ++j) comp[static_cast<size_t>(j) * L] = cd(a[static_cast<size_t>(j)], 0.0);
    for (int i = 1; i < L; ++i) comp[static_cast<size_t>(i - 1) * L + i] = cd(1.0, 0.0);

    std::vector<cd> w(static_cast<size_t>(L)), vr(static_cast<size_t>(L) * L);
    {
        const vbcsr_lapack_int n_ = L;
        vbcsr_lapack_int info = 0, lwork = 4 * L;
        std::vector<cd> work(static_cast<size_t>(lwork));
        std::vector<double> rwork(static_cast<size_t>(2 * L));
        lp_detail::zgeev_("N", "V", &n_,
                          reinterpret_cast<vbcsr_complex_double*>(comp.data()), &n_,
                          reinterpret_cast<vbcsr_complex_double*>(w.data()), nullptr, &n_,
                          reinterpret_cast<vbcsr_complex_double*>(vr.data()), &n_,
                          reinterpret_cast<vbcsr_complex_double*>(work.data()), &lwork,
                          rwork.data(), &info);
        if (info != 0) return out;
    }
    // Poles outside the unit circle come from noise or spectral weight
    // outside the window; renormalise them onto it (Ganahl et al.).
    for (int i = 0; i < L; ++i) {
        const double mag = std::abs(w[static_cast<size_t>(i)]);
        if (mag > 1.0) w[static_cast<size_t>(i)] /= mag;
    }

    // Modal amplitudes z from the companion state (x_{n-1}, ..., x_{n-L}).
    std::vector<cd> z(static_cast<size_t>(L));
    for (int i = 0; i < L; ++i) z[static_cast<size_t>(i)] = cd(x[static_cast<size_t>(n - 1 - i)], 0.0);
    {
        std::vector<cd> vr_copy(vr);
        std::vector<vbcsr_lapack_int> ipiv(static_cast<size_t>(L));
        const vbcsr_lapack_int n_ = L, one = 1;
        vbcsr_lapack_int info = 0;
        lp_detail::zgesv_(&n_, &one,
                          reinterpret_cast<vbcsr_complex_double*>(vr_copy.data()), &n_,
                          ipiv.data(),
                          reinterpret_cast<vbcsr_complex_double*>(z.data()), &n_, &info);
        if (info != 0) return out;
    }

    // x_{n-1+k} = Re sum_i VR[0, i] w_i^k z_i, iterated.
    std::vector<cd> lam(static_cast<size_t>(L), cd(1.0, 0.0));
    for (int p = n; p < n_target; ++p) {
        cd val(0.0, 0.0);
        for (int i = 0; i < L; ++i) {
            lam[static_cast<size_t>(i)] *= w[static_cast<size_t>(i)];
            val += vr[static_cast<size_t>(i) * L] * lam[static_cast<size_t>(i)] * z[static_cast<size_t>(i)];
        }
        out[static_cast<size_t>(p)] = cd(val.real(), 0.0);
    }
    return out;
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_MOMENT_EXTRAPOLATION_HPP
