#ifndef RSATB_CALC_OCCUPATION_HPP
#define RSATB_CALC_OCCUPATION_HPP

// Trace-mode ACE-FD: electron count, band energy, and Fermi-level search
// from Chebyshev trace moments.
//
// The moments mu_n = Re<v|T_n(Op)|v> (sample-averaged) are the only
// quantity that touches the Hamiltonian. Chemical potential and temperature
// enter through Chebyshev coefficients alone, so every (mu, T) evaluation —
// including a full chemical-potential bisection and temperature scans —
// reuses one moment set at zero additional matvecs. Degree selection uses
// the same persistent three-check schedule as the vector-mode expansion
// (fermi_dirac.hpp); scope and validity conditions are documented in
// benchmarks/fermi_dirac/METHOD.md.

#include "cheby.hpp"
#include "fermi_dirac.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace rsatb {
namespace calc {

// Trace moments of one Hamiltonian plus the metadata needed to evaluate
// observables from them.
struct TraceMoments {
    std::vector<double> moments;
    double half_width = 1.0;
    double center = 0.0;
    int n_used = 0;
    double indicator = -1.0;
    // The stopping indicator fired -- see FermiExpansionOptions for what that
    // does and does not promise about the true error.
    bool indicator_fired = false;
};

// Streams mu_n = Re<v|T_n(Op)|v> adaptively, anchored at (temp, mu_anchor).
//
// When the scaled operator is Hermitian in the Euclidean metric (orthogonal
// basis; symmetric preconditioning) the Chebyshev product identities
//   mu_{2j}   = 2<t_j , t_j>     - mu_0
//   mu_{2j+1} = 2<t_{j+1}, t_j>  - mu_1
// deliver two moments per matvec. With an overlap matrix or one-sided
// preconditioning the effective operator is only S-metric self-adjoint, the
// identities do not hold in the plain dot product, and the driver falls back
// to one moment per matvec.
template <typename T, typename VectorType>
TraceMoments adaptive_trace_moments(Chebyshev<T>& cheby,
                                    const VectorType& samples,
                                    double temp,
                                    double mu_anchor,
                                    const FermiExpansionOptions& opt) {
    const double kb = 8.617333262e-5; // eV/K
    const double beta = (temp > 0.0) ? (1.0 / (temp * kb))
                                     : std::numeric_limits<double>::infinity();
    const double beta_scaled = beta * cheby.half_width;
    const double mu_scaled = (mu_anchor - cheby.center) / cheby.half_width;

    detail::AceDegreeSchedule<FermiExpansionOptions, FermiExpansionReport>
        schedule(beta_scaled, mu_scaled, opt);

    TraceMoments tm;
    tm.half_width = cheby.half_width;
    tm.center = cheby.center;
    std::vector<double>& m = tm.moments;
    m.reserve(static_cast<size_t>(std::min(schedule.n_cap() + 2, 1 << 14)));

    const bool hermitian =
        (cheby.S == nullptr) &&
        (cheby.M == nullptr || cheby.ptype == PreconditionerType::Symmetric);

    VectorType t0 = ZeroLike(samples);
    VectorType t1 = ZeroLike(samples);
    VectorType t2 = ZeroLike(samples);
    FermiExpansionReport rep;

    t0.copy_from(samples);
    m.push_back(RealDot(samples, t0));
    cheby.apply_op(t0, t1, ChebyMode::NORMAL);
    m.push_back(RealDot(samples, t1));

    if (hermitian) {
        const double m0 = m[0];
        const double m1 = m[1];
        bool done = false;
        while (!done) {
            // Entering with t0 = T_{j-1}, t1 = T_j and 2j moments stored.
            m.push_back(2.0 * RealDot(t1, t1) - m0);
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(-1.0, t0, 2.0);
            t0.swap(t1);
            t1.swap(t2);
            m.push_back(2.0 * RealDot(t1, t0) - m1);
            done = schedule.due(static_cast<int>(m.size())) &&
                   schedule.evaluate(m, rep);
        }
    } else {
        bool done = schedule.due(static_cast<int>(m.size())) &&
                    schedule.evaluate(m, rep);
        while (!done) {
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(-1.0, t0, 2.0);
            m.push_back(RealDot(samples, t2));
            t0.swap(t1);
            t1.swap(t2);
            done = schedule.due(static_cast<int>(m.size())) &&
                   schedule.evaluate(m, rep);
        }
    }

    tm.n_used = rep.n_used;
    tm.indicator = rep.indicator;
    tm.indicator_fired = rep.indicator_fired;
    m.resize(static_cast<size_t>(tm.n_used));
    return tm;
}

// Weak-form evaluation sum_n g_n a_n mu_n at any (mu, T) from stored trace
// moments — coefficient space only, zero matvecs. With energy_weighted the
// expanded function is E*f(E) (band energy); with KpmKernel::Jackson the
// result is guaranteed monotone in mu (used for bracketing).
inline double fermi_trace_sum(const TraceMoments& tm,
                              double temp,
                              double mu,
                              KpmKernel kernel = KpmKernel::CosTaper,
                              bool energy_weighted = false,
                              int n_override = 0) {
    const int n = (n_override > 0) ? std::min(n_override, tm.n_used) : tm.n_used;
    if (n <= 0) {
        return 0.0;
    }
    const double kb = 8.617333262e-5;
    const double beta = (temp > 0.0) ? (1.0 / (temp * kb))
                                     : std::numeric_limits<double>::infinity();
    const double beta_scaled = beta * tm.half_width;
    const double mu_scaled = (mu - tm.center) / tm.half_width;

    const std::vector<cd> coeffs =
        energy_weighted
            ? fermi_energy_weighted_coefficients(n, beta_scaled, mu_scaled,
                                                 tm.half_width, tm.center,
                                                 KpmKernel::None)
            : fermi_chebyshev_coefficients(n, beta_scaled, mu_scaled, KpmKernel::None);
    return kpm_weighted_sum(coeffs, tm.moments, n, kernel);
}

// Re-checks the convergence indicator of an existing moment set at a
// (mu, T) different from the adaptivity anchor (free; used to keep reports
// honest after a chemical-potential search moved mu).
inline double fermi_trace_indicator(const TraceMoments& tm, double temp, double mu) {
    if (tm.n_used <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    const double kb = 8.617333262e-5;
    const double beta = (temp > 0.0) ? (1.0 / (temp * kb))
                                     : std::numeric_limits<double>::infinity();
    const std::vector<cd> coeffs = fermi_chebyshev_coefficients(
        tm.n_used, beta * tm.half_width, (mu - tm.center) / tm.half_width,
        KpmKernel::None);
    const double norm2 = std::max(tm.moments[0], std::numeric_limits<double>::min());
    return detail::ace_indicator(coeffs, tm.moments, tm.n_used) / norm2;
}

// Bisection for the chemical potential on a non-decreasing ne(mu) callable
// (Jackson-summed occupation is monotone by kernel positivity). `ne` may be
// collective — every rank must then call bisect_fermi_level in lockstep with
// identical brackets, which holds when ne() ends in a broadcast.
template <typename F>
double bisect_fermi_level(F&& ne, double lo, double hi, double target, int iters = 80) {
    double flo = ne(lo) - target;
    if (flo >= 0.0) {
        return lo;
    }
    if (ne(hi) - target <= 0.0) {
        return hi;
    }
    for (int i = 0; i < iters; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (ne(mid) - target < 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

} // namespace calc
} // namespace rsatb

#endif
