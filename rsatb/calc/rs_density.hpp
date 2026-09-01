#ifndef RSATB_CALC_RS_DENSITY_HPP
#define RSATB_CALC_RS_DENSITY_HPP

// Real-space electron density by square-root-filtered stochastic states --
// the LDOS estimator (rs_ldos.hpp, route 5.3b) with the energy filter
// replaced by the square root of the occupation:
//
//     n(r) = sum_n f_FD(eps_n) |psi_n(r)|^2 = g_s E[ |phi(r)^T chi|^2 ],
//     chi  = F(Op) seed,   F = sqrt(f_FD((E - mu) / kT)),
//
// with g_s the spin degeneracy. The identity is the rs_ldos one with F^2 the
// occupation instead of a delta: in the solve form F(A) S^-1 F(A)^H =
// C f_FD C^H = D, and in the sandwich form Z F^2(Z^H H Z) Z^H = D, the
// density matrix either way. Everything the LDOS map established carries
// over unchanged -- one Chebyshev sweep, the demand-built orbital set, the
// square-on-grid contraction, non-negativity of every sample -- and the
// density is CHEAPER than a map: one filter table and one accumulator where
// the LDOS holds one per energy.
//
// Scope: a post-processing observable of a FIXED Hamiltonian. Inside an SCF
// loop the stochastic noise would propagate nonlinearly through the density
// -- doc/design_realspace_density_ldos.md section 4 excludes exactly that,
// and its deterministic probing (section 5) is the SCF answer. Do not wire
// this into one.
//
// Cost: the Chebyshev degree is set by the Matsubara branch point of
// sqrt(f_FD) at mu +- i pi kT, giving m ~ ln(1/eps) / ln(rho) with
// rho = t + sqrt(1 + t^2), t = pi kT / half-width -- about
// (half / (pi kT)) ln(1/eps) when t is small. Room temperature over a 20 eV
// spectrum is a few thousand moments; it drops linearly with the smearing.

#include "rs_ldos.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace rsatb {
namespace calc {

struct RealSpaceDensityOptions {
    double mu = 0.0;             ///< Chemical potential, eV, absolute.
    double kT = 0.0;             ///< Smearing temperature, eV. Required > 0.
    double spin_degeneracy = 2.0;  ///< States per orbital; 2 for spin-paired.
    double emin = 0.0;           ///< Lower bound of the (H, S) pencil, eV.
    double emax = 0.0;           ///< Upper bound, eV. Required: emax > emin.
    double bound_pad = 0.01;     ///< Fractional pad on the fitted interval.
    double spacing = 0.4;        ///< Grid spacing, Angstrom.
    int box_edge = 0;            ///< Grid box edge; 0 = physical default.
    int n_samples = 64;          ///< Random states in the estimator.
    int block = 16;              ///< States per Chebyshev sweep (SpMM width).
    double coeff_eps = 1e-8;     ///< Filter truncation.
    unsigned long seed = 12345;  ///< RNG seed; per-rank streams derive from it.
    bool complete_basis = false;   ///< Exact validation sweep; see rs_ldos.
    bool certify_sampler = true;   ///< Solve-form sampler certificate.
    /// Fragment control variate; see RealSpaceLdosOptions. The buffer is the
    /// systematic accuracy knob; 0 disables.
    double fragment_core = 4.5;    ///< Core bin edge, Angstrom.
    double fragment_buffer = 0.0;  ///< Dressing radius, Angstrom; 0 = off.
    /// False gives the coefficient-one estimator, exactly unbiased for every
    /// fixed fragment. True retains the fitted legacy robustness heuristic.
    bool fit_fragment_coefficient = true;
    /// Energy-window dilution (ew-sDFT: Chen & Baer, JCP 2019; hybrid with
    /// fragments = ew-efsDFT, Chen et al. 2021). The occupation is split into
    /// n windows bounded by Fermi steps AT THE SAME kT, each window filtered
    /// and squared separately from the SAME sweep and the SAME seeds, and the
    /// maps summed. Removes the cross-window part of the sampling variance at
    /// no extra Chebyshev degree; by Cauchy-Schwarz it can never increase it.
    /// This is the one mechanism that keeps working where fragments fail
    /// (delocalized/metallic occupations); 1 disables.
    int energy_windows = 1;
    MetricSamplerOptions sampler;  ///< Options of the S^{-1}-covariance sampler.
};

struct RealSpaceDensityResult {
    RealSpaceGrid grid;
    GridDomain domain;              ///< This rank's boxes and points.
    std::vector<double> density;    ///< [owned point], electrons / A^3.
    int filter_degree = 0;
    int n_states = 0;
    double sampler_residual = 0.0;
    int n_fragments = 0;          ///< Fragment count; 0 = plain estimator.
    /// RMS of the stochastic correction over the grid, electrons/A^3 -- the
    /// measured residual the samples still carry after the fragment baseline.
    double correction_rms = 0.0;
    /// Regression coefficient of the control variate; 1 = surrogate tracks
    /// the estimator sample for sample, 0 = fell back to the plain mean.
    /// With energy windows: the minimum over windows.
    double control_beta = 1.0;
    int n_windows = 1;            ///< Energy windows the density summed over.
    /// Integral of the density over this rank's grid points, Allreduced: the
    /// electron count as the GRID sees it. Against the exact
    /// sum_n g_s f_FD(eps_n) it folds in sampling noise, filter truncation,
    /// Zt quality AND collocation quadrature -- which is what makes it an
    /// end-to-end check rather than a tautology.
    double electrons_on_grid = 0.0;
    RealSpaceLdosProfile profile;
};

namespace detail {

/// Chebyshev table of sqrt(f_FD) on the scaled interval. Dimensionless -- no
/// physical normalization is folded in, unlike the sqrt-Gaussian table whose
/// (2 pi eta^2)^{-1/4} carries the 1/eV of a spectral density.
inline std::vector<std::complex<double>> SqrtFermiTable(double mu, double kT,
                                                        double center, double half,
                                                        double eps) {
    const double mu_bar = (mu - center) / half;
    const double kT_bar = kT / half;
    // Bernstein-ellipse degree. The nearest singularities of sqrt(f_FD) are
    // the branch points at mu +- i pi kT (the first Matsubara poles of f_FD);
    // the worst placement over mu in the interval is the center, where the
    // ellipse parameter is rho = t + sqrt(1 + t^2), t = pi kT_bar.
    const double t = M_PI * kT_bar;
    const double rho = t + std::sqrt(1.0 + t * t);
    const int m =
        static_cast<int>(std::ceil(std::log(1.0 / eps) / std::log(rho))) + 8;
    auto table = chebyshev_coefficients_from_function(
        m,
        [mu_bar, kT_bar](double x) {
            const double u = (x - mu_bar) / kT_bar;
            // sqrt(1 / (1 + e^u)), evaluated without overflowing either tail.
            return u > 0.0 ? std::exp(-0.5 * u) / std::sqrt(1.0 + std::exp(-u))
                           : 1.0 / std::sqrt(1.0 + std::exp(u));
        },
        KpmKernel::None, /*trunc_eps=*/0.0);
    return table;
}

/// Chebyshev table of sqrt(f_FD(:, m_hi) - f_FD(:, m_lo)) -- one energy
/// window's share of the occupation, edges smeared at the SAME kT as the full
/// filter, so the degree formula (Matsubara branch points at the edges) is
/// unchanged. m_lo far below the spectrum recovers the plain filter's window.
inline std::vector<std::complex<double>> SqrtFermiWindowTable(
    double m_lo, double m_hi, double kT, double center, double half, double eps) {
    const double lo_bar = (m_lo - center) / half;
    const double hi_bar = (m_hi - center) / half;
    const double kT_bar = kT / half;
    const double t = M_PI * kT_bar;
    const double rho = t + std::sqrt(1.0 + t * t);
    const int m =
        static_cast<int>(std::ceil(std::log(1.0 / eps) / std::log(rho))) + 8;
    auto fermi = [kT_bar](double x, double mu_bar) {
        const double u = (x - mu_bar) / kT_bar;
        return u > 0.0 ? std::exp(-u) / (1.0 + std::exp(-u)) : 1.0 / (1.0 + std::exp(u));
    };
    return chebyshev_coefficients_from_function(
        m,
        [&](double x) {
            return std::sqrt(std::max(0.0, fermi(x, hi_bar) - fermi(x, lo_bar)));
        },
        KpmKernel::None, /*trunc_eps=*/0.0);
}

}  // namespace detail

/// Computes the electron density on this rank's share of the grid. Same
/// argument contract as ComputeRealSpaceLdos; opt.mu, opt.kT and the pencil
/// bounds are required.
inline RealSpaceDensityResult ComputeRealSpaceDensity(
    vbcsr::atomic::AtomicData& atomic_data, vbcsr::BlockSpMat<double>& H,
    vbcsr::BlockSpMat<double>* S, vbcsr::BlockSpMat<double>& Zt, ChebyOpForm form,
    const OrbitalCollocation& collocation, const RealSpaceDensityOptions& opt,
    MPI_Comm comm, detail::FragmentBasisCache* fragment_cache = nullptr) {
    if (!(opt.kT > 0.0)) {
        throw std::runtime_error(
            "Real-space density needs a positive smearing kT; the zero-temperature "
            "step has no convergent Chebyshev expansion.");
    }
    if (!(opt.spin_degeneracy > 0.0)) {
        throw std::runtime_error("Real-space density needs a positive spin degeneracy.");
    }

    // The shared knobs travel through the LDOS options struct the core reads.
    // eta is set to kT so the profile's "broadening" line reports the
    // smearing; the core itself never reads eta or energies.
    RealSpaceLdosOptions base;
    base.eta = opt.kT;
    base.emin = opt.emin;
    base.emax = opt.emax;
    base.bound_pad = opt.bound_pad;
    base.spacing = opt.spacing;
    base.box_edge = opt.box_edge;
    base.n_samples = opt.n_samples;
    base.block = opt.block;
    base.coeff_eps = opt.coeff_eps;
    base.seed = opt.seed;
    base.complete_basis = opt.complete_basis;
    base.certify_sampler = opt.certify_sampler;
    base.sampler = opt.sampler;
    base.fragment_core = opt.fragment_core;
    base.fragment_buffer = opt.fragment_buffer;
    base.fit_fragment_coefficient = opt.fit_fragment_coefficient;
    // The Ritz-subspace control is the sharp-filter lever; the occupied
    // manifold's rank is extensive, so no O(n_samples) sketch of it can
    // correlate. Fragments and energy windows are the density's levers.
    base.subspace_cv = false;

    const int n_win = std::max(1, opt.energy_windows);
    RealSpaceLdosResult raw = detail::FilteredSquareMap(
        atomic_data, H, S, Zt, form, collocation, base,
        [&opt, n_win](double center, double half) {
            std::vector<std::vector<std::complex<double>>> tables;
            if (n_win == 1) {
                tables.push_back(
                    detail::SqrtFermiTable(opt.mu, opt.kT, center, half, opt.coeff_eps));
                return tables;
            }
            // Boundaries uniform over [emin, mu]; the lowest edge sits far
            // below the spectrum so the first window opens from nothing, and
            // the top window closes at mu -- the squared filters then
            // telescope back to f_FD(:, mu) up to f evaluated AT that lowest
            // edge, which is the entire telescoping error: at 10 kT below the
            // spectrum it is e^-10 ~ 4e-5 (measured, exactly); 30 kT puts it
            // at e^-30, below every other error in the pipeline.
            const double lo0 = opt.emin - 30.0 * opt.kT - 1.0;
            for (int w = 0; w < n_win; ++w) {
                const double m_lo =
                    (w == 0) ? lo0 : opt.emin + (opt.mu - opt.emin) * w / n_win;
                const double m_hi =
                    (w == n_win - 1) ? opt.mu
                                     : opt.emin + (opt.mu - opt.emin) * (w + 1) / n_win;
                tables.push_back(detail::SqrtFermiWindowTable(m_lo, m_hi, opt.kT, center,
                                                              half, opt.coeff_eps));
            }
            return tables;
        },
        comm, fragment_cache);

    RealSpaceDensityResult out;
    out.grid = raw.grid;
    out.domain = std::move(raw.domain);
    out.filter_degree = raw.filter_degree;
    out.n_states = raw.n_states;
    out.sampler_residual = raw.sampler_residual;
    out.n_fragments = raw.n_fragments;
    out.n_windows = n_win;
    if (!raw.correction_rms.empty()) {
        // Report the worst window: the largest residual and the weakest
        // surrogate, scaled like the map they correct.
        out.correction_rms =
            *std::max_element(raw.correction_rms.begin(), raw.correction_rms.end()) *
            opt.spin_degeneracy;
        out.control_beta =
            *std::min_element(raw.control_beta.begin(), raw.control_beta.end());
    }
    out.profile = raw.profile;
    // The density is the sum of the window maps -- with one window this is
    // the move it always was.
    out.density = std::move(raw.ldos.front());
    for (size_t w = 1; w < raw.ldos.size(); ++w) {
        for (size_t p = 0; p < out.density.size(); ++p) {
            out.density[p] += raw.ldos[w][p];
        }
    }
    for (double& v : out.density) v *= opt.spin_degeneracy;

    // Voxel volume from the cell determinant and the GLOBAL point count; the
    // local sum runs over owned points only, so the Allreduce composes the
    // integral without double counting.
    const std::vector<double>& c = atomic_data.cell;
    const double det =
        c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
        c[2] * (c[3] * c[7] - c[4] * c[6]);
    const double voxel = std::abs(det) / static_cast<double>(out.grid.n_point());
    double local = 0.0;
    for (const double v : out.density) local += v;
    local *= voxel;
    out.electrons_on_grid = local;
    if (comm != MPI_COMM_NULL) {
        MPI_Allreduce(&local, &out.electrons_on_grid, 1, MPI_DOUBLE, MPI_SUM, comm);
    }
    return out;
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_RS_DENSITY_HPP
