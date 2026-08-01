#ifndef RSATB_CALC_FERMI_DIRAC_HPP
#define RSATB_CALC_FERMI_DIRAC_HPP

#include "cheby.hpp"
#include "kpm.hpp"
#include "vector_utils.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

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

namespace detail {

// Error indicator at truncation N: the maximum over three structurally
// different comparisons. A nested |q(N)-q(N/2)| alone is unsafe — the
// difference polynomial has roots, and spectral weight sitting on a root
// yields false convergence — so distinct degrees and a distinct kernel
// family (Richardson-extrapolated Jackson) are compared as well. This is an
// indicator, not a bound; the caller additionally requires it to fire at two
// consecutive doubling levels (see AceDegreeSchedule).
inline double ace_indicator(const std::vector<cd>& coeffs,
                            const std::vector<double>& moments, int n) {
    const auto q_at = [&](int m, KpmKernel k) {
        return kpm_weighted_sum(coeffs, moments, m, k);
    };
    const double q = q_at(n, KpmKernel::CosTaper);
    const double r1 = (4.0 * q_at(n, KpmKernel::Jackson) -
                       q_at(n / 2, KpmKernel::Jackson)) / 3.0;
    return std::max({std::abs(q - q_at(n / 2, KpmKernel::CosTaper)),
                     std::abs(q - q_at(3 * n / 4, KpmKernel::CosTaper)),
                     std::abs(q - r1)});
}

} // namespace detail

// Chebyshev coefficients of the Fermi function on the scaled interval, with
// a summation rule applied.
//
// Grid-size rule: Gauss-Chebyshev aliasing folds the coefficient tail beyond
// the sampling grid back onto the kept coefficients. The Fermi tail only
// decays past the thermal index n_c ~ beta_scaled/pi, so a short tapered
// expansion at low temperature must still be sampled on >= ~2*beta_scaled
// points (a fixed floor of 4096 silently corrupts the 1 K regime at the
// 1e-4 level). Beyond the grid cap the *exact* T=0 step coefficients
//   a_0 = 1 - theta_mu/pi,  a_n = -2 sin(n*theta_mu)/(pi*n)
// are used instead; at those temperatures (kT_scaled < ~4e-6) the
// step-vs-Fermi difference enters smooth-DOS observables at
// O((kT/half_width)^2) < 1e-10, so the swap is rigorous, FFT-free, and
// valid at T = 0 itself.
inline std::vector<cd> fermi_chebyshev_coefficients(int n, double beta_scaled,
                                                    double mu_scaled,
                                                    KpmKernel kernel) {
    if (n <= 0) {
        throw std::runtime_error("fermi_chebyshev_coefficients: n must be positive");
    }
    constexpr double kGridCap = static_cast<double>(1 << 19);
    const bool step_limit = !std::isfinite(beta_scaled) || beta_scaled <= 0.0 ||
                            2.0 * beta_scaled > kGridCap;
    if (!step_limit) {
        const double min_grid = std::max(4096.0, 2.0 * beta_scaled);
        const int oversample =
            std::max(8, static_cast<int>(min_grid / static_cast<double>(n)) + 1);
        return chebyshev_coefficients_from_function(
            n,
            [&](double x) { return occupation_projector_func(x, beta_scaled, mu_scaled); },
            kernel, 0.0, oversample);
    }

    const double x_mu = std::max(-1.0, std::min(1.0, mu_scaled));
    const double theta_mu = std::acos(x_mu);
    const std::vector<double> g = kpm_kernel_weights(kernel, n);
    std::vector<cd> coeffs(static_cast<size_t>(n));
    coeffs[0] = g[0] * (1.0 - theta_mu / M_PI);
    for (int i = 1; i < n; ++i) {
        coeffs[static_cast<size_t>(i)] =
            g[static_cast<size_t>(i)] *
            (-2.0 * std::sin(static_cast<double>(i) * theta_mu) / (M_PI * static_cast<double>(i)));
    }
    return coeffs;
}

// Coefficients of (half_width*x + center) * f(x) derived from the plain
// Fermi coefficients via the exact identity
//   x T_0 = T_1,   x T_m = (T_{m+1} + T_{m-1}) / 2  (m >= 1),
// avoiding a second (large) FFT for band-energy evaluations.
inline std::vector<cd> fermi_energy_weighted_coefficients(
    int n, double beta_scaled, double mu_scaled, double half_width, double center,
    KpmKernel kernel) {
    const std::vector<cd> base =
        fermi_chebyshev_coefficients(n + 1, beta_scaled, mu_scaled, KpmKernel::None);
    const std::vector<double> g = kpm_kernel_weights(kernel, n);
    std::vector<cd> out(static_cast<size_t>(n));
    for (int m = 0; m < n; ++m) {
        cd b;
        if (m == 0) {
            b = 0.5 * base[1];
        } else if (m == 1) {
            b = base[0] + 0.5 * base[2];
        } else {
            b = 0.5 * (base[static_cast<size_t>(m - 1)] + base[static_cast<size_t>(m + 1)]);
        }
        out[static_cast<size_t>(m)] =
            g[static_cast<size_t>(m)] *
            (half_width * b + center * base[static_cast<size_t>(m)]);
    }
    return out;
}

// Forward declaration; definition below the option structs.
struct FermiExpansionOptions;
struct FermiExpansionReport;

namespace detail {

// Where a Fermi expansion stops -- the whole degree policy, for the vector
// driver below and the trace-moment driver in occupation.hpp alike.
//
// Two policies behind one interface, so the drivers have one loop rather than
// a branch each. A fixed degree stops there and says so. An adaptive one runs
// the persistent three-check indicator at the rungs of a doubling ladder over
// a growing moment stream, with the degree capped near the pointwise Jackson
// degree ~ 2*pi*beta*half_width -- there the tapered sum resolves the Fermi
// edge in the strong sense, so a single sub-tolerance check already certifies
// convergence, while a weak-form early stop additionally requires the
// indicator to fire at two consecutive rungs.
template <typename OptionsT, typename ReportT>
class AceDegreeSchedule {
public:
    AceDegreeSchedule(double beta_scaled, double mu_scaled, const OptionsT& opt)
        : tol_(opt.indicator_threshold),
          n_cap_(std::max(opt.n_min, opt.n_max)),
          next_check_(std::max(16, opt.n_min)) {
        if (opt.n_fixed > 0) {
            // The degree is given, so there is nothing to probe: the first and
            // only stop is at n_fixed. No indicator, and no moments needed.
            adaptive_ = false;
            n_cap_ = next_check_ = opt.n_fixed;
            coeffs_ = fermi_chebyshev_coefficients(n_cap_, beta_scaled, mu_scaled,
                                                   KpmKernel::None);
            return;
        }
        // The pointwise cap: 2 * pi * beta * half_width. Twice Eq. (7) of
        // METHOD.md because the taper's effective degree is N/2 (Sec. 5.1), so
        // this is the degree at which the tapered sum is Jackson-grade by
        // construction and further probing cannot help.
        if (std::isfinite(beta_scaled) && beta_scaled > 0.0) {
            const double n_pointwise = 2.0 * M_PI * beta_scaled;
            if (n_pointwise < static_cast<double>(n_cap_)) {
                // The cap wins over n_min. Where the pointwise degree is the
                // cheaper of the two there is nothing for probing to find --
                // expanding straight to it is both fewer matvecs and
                // Jackson-grade. Clamping up to n_min instead would spend more
                // to get no more, and would put the cost above the parity bound
                // Sec. 5.2 claims. The floor of 4 is the indicator's, which
                // compares against degrees n/2 and 3n/4.
                n_cap_ = std::max(4, static_cast<int>(n_pointwise) + 1);
                cap_is_pointwise_ = true;
            }
        }
        // Hang the check ladder off the cap -- n_cap, n_cap/2, n_cap/4, ...
        // down to n_min -- rather than building it up from n_min.
        //
        // Two promises have to hold at once. fermi_dirac_kpm needs consecutive
        // stopping degrees to differ by exactly a factor of two, since that is
        // the span over which the tapers stay flat and is what lets it weight
        // terms as it generates them. And the cap must be hit exactly, because
        // METHOD.md Sec. 5.2 rests the whole worst case on it: at the pointwise
        // degree the result is Jackson-grade, so cost never exceeds current
        // practice. Rounding the cap up to a rung would break that bound by up
        // to 2x. Anchoring at the cap and halving keeps both -- the only thing
        // that moves is where probing starts, which lands in [n_min, 2*n_min)
        // instead of exactly n_min, and n_min is a heuristic anyway.
        const int n_first = std::max(16, opt.n_min);
        level_ = 0;
        while ((n_cap_ >> (level_ + 1)) >= n_first) ++level_;
        next_check_ = n_cap_ >> level_;
        // One coefficient set for the whole run, at the largest degree the
        // schedule can reach. The exact a_n do not depend on where we stop,
        // so every check level reads a prefix of this array -- previously
        // each level paid its own FFT, and the driver below can now share it.
        coeffs_ = fermi_chebyshev_coefficients(n_cap_, beta_scaled, mu_scaled,
                                               KpmKernel::None);
    }

    int n_cap() const { return n_cap_; }

    // False when the degree was given up front, in which case no moments are
    // needed and the stopping degree never moves.
    bool adaptive() const { return adaptive_; }

    // Degree the next check will evaluate at: the next rung of the ladder,
    // and so the only degree the run can stop at before the one after it.
    // fermi_dirac_kpm sums against this as a standing hypothesis.
    int next_check() const { return next_check_; }

    // Exact (unweighted) coefficients a_n, n < n_cap().
    const std::vector<cd>& coefficients() const { return coeffs_; }

    bool due(int n_now) const { return n_now >= next_check_ || n_now >= n_cap_; }

    // Whether the run should stop, with the report updated either way.
    // `moments` is unread when the degree is fixed. Coefficient FFTs here are
    // negligible next to one distributed matvec.
    bool evaluate(const std::vector<double>& moments, ReportT& report) {
        if (!adaptive_) {
            report.n_used = n_cap_;
            report.indicator_fired = true;
            return true;
        }
        const int n_now = static_cast<int>(moments.size());
        const int n_eval = std::min(n_now, n_cap_);
        const double norm2 = std::max(moments[0], std::numeric_limits<double>::min());
        const double est = ace_indicator(coeffs_, moments, n_eval) / norm2;
        const bool at_pointwise_cap = cap_is_pointwise_ && n_now >= n_cap_;
        const bool fired = est < tol_ && (prev_est_ < tol_ || at_pointwise_cap);
        report.n_used = n_eval;
        report.indicator = est;
        report.indicator_fired = fired;
        if (fired || n_now >= n_cap_) {
            return true;
        }
        prev_est_ = est;
        // Down one level of the ladder, so the next rung is exactly twice this
        // one (integer shift, not a multiply, so the last rung is the cap).
        --level_;
        next_check_ = n_cap_ >> level_;
        return false;
    }

private:
    std::vector<cd> coeffs_;
    double tol_;
    int n_cap_;
    int next_check_;
    int level_ = 0;  // rungs left below the cap; next_check_ == n_cap_ >> level_
    bool adaptive_ = true;
    bool cap_is_pointwise_ = false;
    double prev_est_ = std::numeric_limits<double>::infinity();
};

} // namespace detail

// How a Fermi-Dirac operator expansion is built and summed.
//
// The default is the adaptive tapered ("ACE-FD") expansion: exact Fermi
// coefficients, CosTaper summation, and a degree chosen a posteriori from
// the input vector's own spectral measure. On extended systems whose local
// DOS is smooth near mu, the chosen degree is set by that smoothness and is
// essentially temperature independent — orders of magnitude below the
// pointwise sigma <= kT Jackson degree at low temperature. The result is
// accurate for quantities *linear* in the expansion (traces, densities via
// correlation estimators, smooth-contraction correlation functions); it is
// NOT a pointwise-accurate f(H)|psi> and not a projector. Scope, failure
// modes, and validation: benchmarks/fermi_dirac/METHOD.md and RESULTS.md.
// The two modes take disjoint settings, and which one is in force is decided
// by n_fixed alone. Nothing from the other group is read.
struct FermiExpansionOptions {
    // Summation rule; applies to both modes.
    KpmKernel summation = KpmKernel::CosTaper;

    // --- fixed degree ----------------------------------------------------
    // > 0 selects this mode, and none of the adaptive settings below are read.
    // This is where Jackson belongs: its resolution sigma ~ pi*half_width/N is
    // itself the degree choice, so asking for a resolution has already picked
    // N (n ~ pi*W/kT for pointwise accuracy at temperature T).
    int n_fixed = 0;

    // --- adaptive degree, ACE-FD (read only when n_fixed == 0) -----------
    //
    // NOT an error tolerance, which is why it is not called one. It is the
    // level the a-posteriori error *indicator* of METHOD.md Eq. (15) must fall
    // below, and Sec. 5.2 is explicit that no inequality relating that
    // indicator to the true error is proven. It can and does certify degrees
    // whose true error is larger: on diamond sc444 with mu inside the valence
    // band at 3 K, a threshold of 1e-4 reports a fired indicator at a degree
    // whose actual weak-form error is 1.6e-2 (benchmarks/fermi_dirac_diamond).
    //
    // What it is good for is the regime the method is stated over -- many
    // levels within the resolution window, METHOD.md Sec. 3.2 -- where it
    // tracks the error well and buys orders of magnitude. Outside that regime
    // it is a heuristic, and spot-validation against a fixed-degree run at the
    // pointwise degree is the recommended check.
    double indicator_threshold = 1e-4;
    // First level probed; the ladder then doubles up to the cap.
    int n_min = 256;
    // Hard degree cap. The effective cap is the smaller of this and the
    // pointwise degree ~ 2*pi*beta*half_width, where a tapered sum is
    // Jackson-grade by construction and further probing cannot help.
    int n_max = 1 << 16;

    bool verbose = false;
};

struct FermiExpansionReport {
    int n_used = 0;
    // Final indicator value, relative to <psi|psi>.
    double indicator = -1.0;
    // The indicator fell below its threshold -- a stopping signal, not a
    // statement about the true error. False means the degree cap was reached
    // instead, where the result is Jackson-grade at that degree. See
    // indicator_threshold above for what a fired indicator does and does not
    // promise; it is named for what it observes rather than "converged"
    // precisely because the two are not the same thing.
    bool indicator_fired = false;
    // Matvecs actually spent: always n_used - 1. Finding the degree and
    // summing the series are the same recurrence, so discovering N costs no
    // more than being told it.
    int matvecs = 0;
};

// Fermi-Dirac expansion applied to a vector:
//   psi_out = sum_{n<N} g_n^(N) a_n T_n(Op) psi_in
//
// One routine, one recurrence, N - 1 matvecs -- whether N was given (n_fixed)
// or discovered. Nothing is probed and then redone.
//
// What makes the adaptive case non-obvious is that the taper weight g_n^(N)
// depends on the degree the run stops at, which is not known while the terms
// are being generated. Two facts remove the difficulty. The tapers are flat
// over the leading half (g_n^(N) = 1 for n < N/2), and the schedule can only
// stop on a rung of a ladder hung off the cap, n_cap >> k, whose steps are
// exact doublings. So at any moment there are two futures, and one accumulator
// each covers them:
//
//   psi_out  =  sum_{n<now} g_n^(Nc) a_n T_n psi    if the run stops at Nc
//   plain    =  sum_{n<now}          a_n T_n psi    if it carries on
//
// with Nc the next rung. Fire at Nc and psi_out is already the answer. Miss,
// and the new hypothesis is 2 Nc, whose taper is flat below Nc: every term so
// far takes weight 1, so psi_out <- plain restarts the hypothesis exactly.
// Either way the recurrence advances, never rewinds.
//
// Jackson is the one rule that cannot work this way -- it reweights every term
// at every degree, so no partial sum survives a change of hypothesis. It also
// never needs to. In ACE-FD (METHOD.md) Jackson is the fixed-degree baseline,
// where its resolution sigma ~ pi*half_width/N chooses N up front, and a
// participant in the moment-space cross-checks of Secs. 5.2-5.3 -- both of
// which are scalar work on moments, not a vector expansion. Nothing in the
// method wants an adaptive Jackson f(H)|psi>, so it is refused rather than
// served by a second recurrence. The adaptive fallback is a *tapered* run at
// the pointwise degree, which is Jackson-grade by construction (Sec. 5.2).
template <typename T, typename VectorType>
void fermi_dirac_kpm(Chebyshev<T>& cheby,
                     const VectorType& psi_in,
                     VectorType& psi_out,
                     double temp,
                     double mu,
                     const FermiExpansionOptions& opt,
                     ChebyMode mode = ChebyMode::NORMAL,
                     FermiExpansionReport* report = nullptr) {
    if (opt.n_fixed <= 0 && opt.summation == KpmKernel::Jackson) {
        throw std::runtime_error(
            "fermi_dirac_kpm: Jackson cannot be combined with an adaptive degree. "
            "It reweights every term at every degree, so the series cannot be summed "
            "until the degree is known. A Jackson run has one already -- its "
            "resolution sigma ~ pi*half_width/N fixes it -- so set n_fixed, or pick "
            "a taper.");
    }

    const double kb = 8.617333262e-5; // eV/K
    const double beta = (temp > 0.0) ? (1.0 / (temp * kb))
                                     : std::numeric_limits<double>::infinity();
    const double beta_scaled = beta * cheby.half_width;
    const double mu_scaled = (mu - cheby.center) / cheby.half_width;

    detail::AceDegreeSchedule<FermiExpansionOptions, FermiExpansionReport>
        schedule(beta_scaled, mu_scaled, opt);
    const std::vector<cd>& a = schedule.coefficients();

    // The second accumulator earns its memory only when the hypothesis can
    // still move and the rule actually tapers; otherwise psi_out is the answer
    // from the first term on.
    const bool hypothesis_moves =
        schedule.adaptive() && opt.summation != KpmKernel::None;

    FermiExpansionReport rep;
    VectorType t0 = ZeroLike(psi_in);
    VectorType t1 = ZeroLike(psi_in);
    VectorType t2 = ZeroLike(psi_in);
    VectorType plain = ZeroLike(psi_in);

    // Moments feed the indicator, so a fixed-degree run skips the dots -- each
    // is a collective, and at scale that latency is not free.
    const bool needs_moments = schedule.adaptive();
    std::vector<double> moments;
    moments.reserve(static_cast<size_t>(std::min(schedule.n_cap(), 1 << 14)));

    // Weights of the standing hypothesis, "the run stops at this rung".
    std::vector<double> g = kpm_kernel_weights(opt.summation, schedule.next_check());
    int n_now = 0;

    const auto accumulate = [&](int n, const VectorType& t_n) {
        psi_out.axpy(CoeffCast<T>::apply(g[static_cast<size_t>(n)] *
                                         a[static_cast<size_t>(n)]), t_n);
        if (hypothesis_moves) plain.axpy(CoeffCast<T>::apply(a[static_cast<size_t>(n)]), t_n);
    };

    // T_0, seeding the accumulators. g_0 = 1 for every summation rule.
    t0.copy_from(psi_in);
    if (needs_moments) moments.push_back(RealDot(psi_in, t0));
    psi_out.copy_from(t0);
    psi_out.scale(CoeffCast<T>::apply(a[0]));
    if (hypothesis_moves) {
        plain.copy_from(t0);
        plain.scale(CoeffCast<T>::apply(a[0]));
    }
    n_now = 1;

    // T_1.
    cheby.apply_op(t0, t1, mode);
    ++rep.matvecs;
    if (needs_moments) moments.push_back(RealDot(psi_in, t1));
    accumulate(1, t1);
    n_now = 2;

    while (true) {
        if (schedule.due(n_now)) {
            if (schedule.evaluate(moments, rep)) break;
            // Missed: the hypothesis moves out one rung, and its taper is flat
            // over everything summed so far.
            psi_out.copy_from(plain);
            g = kpm_kernel_weights(opt.summation, schedule.next_check());
        }
        cheby.apply_op(t1, t2, mode);
        ++rep.matvecs;
        t2.axpby(-1.0, t0, 2.0);  // t2 = T_{n_now}
        if (needs_moments) moments.push_back(RealDot(psi_in, t2));
        accumulate(n_now, t2);
        t0.swap(t1);
        t1.swap(t2);
        ++n_now;
    }

    if (opt.verbose && psi_in.graph->rank == 0) {
        std::cout << "fermi-dirac: N = " << rep.n_used
                  << (schedule.adaptive() ? " (adaptive, indicator = " : " (fixed")
                  << (schedule.adaptive() ? std::to_string(rep.indicator) + ", " : ", ")
                  << (rep.indicator_fired ? "indicator fired" : "degree cap")
                  << "), matvecs = " << rep.matvecs << "\n";
    }
    if (report) *report = rep;
}

} // namespace calc
} // namespace rsatb

#endif
