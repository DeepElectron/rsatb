#ifndef RSATB_CALC_RS_LDOS_HPP
#define RSATB_CALC_RS_LDOS_HPP

// Full-cell LDOS map on the real-space grid, by square-root-filtered
// stochastic states (doc/design_stochastic_ldos.md; route 5.3b).
//
//     rho(r, E) = sum_n |psi_n(r)|^2 delta_eta(E - eps_n)
//               = E[ |phi(r)^T chi_E|^2 ],      chi_E = F_E(S^-1 H) c0,
//
// with F_E = delta_eta^{1/2}(E - .) -- a Gaussian of width sqrt(2) eta -- and
// c0 drawn with covariance E[c0 c0^H] = S^{-1}. The identity behind it: for
// A = S^-1 H with S-orthonormal eigenvectors C, F(A) S^-1 F(A)^H = C F^2 C^H
// = A(E), the energy-resolved density matrix.
// Estimating a NORM rather than a bilinear form makes every sample map
// non-negative pointwise -- no Jackson kernel needed for positivity, so the
// full sqrt(2) of resolution is kept (design 3.4).
//
// Both operator forms produce this chi; they differ only in where the metric
// is carried, and the estimator is otherwise the same code:
//
//   Solve     chi = F(S^-1 H) c0,   c0 = MetricSampler(xi)   [design 5.5]
//   Sandwich  chi = Zt F(Z^H H Z) xi,  xi Rademacher         [design 5.4]
//
// The sandwich line is the same quantity: with c = Zt eta the covariance
// Zt^-1 S^-1 Zt^-H is already the identity, so the sampler collapses to the
// raw draw, and the Zt that the solve form hides inside S^-1 reappears as one
// SpMM per energy at the end. It is strictly the cheaper of the two -- no
// sampler construction, no CG solve inside the M-step recursion.
//
// The cost structure is what makes full-cell maps affordable:
//   - ONE Chebyshev sweep serves every energy: the energy dependence lives in
//     scalar coefficients, so an extra energy point is one AXPY per step
//     (design 5.1). M ~ a/eta matvecs per sample block, total independent of
//     the energy count.
//   - The grid stage squares each filtered block onto the boxes this rank
//     owns (SquareVectorsOntoBoxes): no n_orb x n_orb object is ever formed,
//     and the output is the compact per-rank buffer of the grid partition.
//   - Everything distributed is O(N/P): the demand-built orbital set, its
//     one-alltoallv coefficient halo (GatherSetRows), the recursion vectors,
//     and the map itself.
//
// Squaring a random state carries a chi^2 noise floor of sqrt(2/n) per
// point, so the LDOS map does not square the probes at all: it is the
// survey-sampling estimator of doc/design_survey_sampling_ldos.md. Each
// probe's sweep also yields a flat-top sketch of the energy window and the
// sketch's exact filtered images, the map is the captured part
// |Pi p(H) phi_r|^2 of the window divided by the capture fraction c measured
// on the probes, and its relative error is sqrt(2 (1 - c) / k) with k the
// sketch rank -- bounded at every system size, exact once the sketch spans
// the window. The density (rs_density.hpp) keeps the plain squares behind a
// fragment surrogate control variate; both run through the one engine
// below. For a noise-free answer on a small system (validation),
// complete_basis sweeps the whole identity instead of random draws and the
// estimator becomes exact up to the filter truncation and the sampler
// certificate.

#include "../utils/profile.hpp"

#include "cheby.hpp"
#include "collocation.hpp"
#include "kpm.hpp"
#include "metric_sampler.hpp"
#include "rs_fragment.hpp"
#include "preconditioner.hpp"
#include "real_space_grid.hpp"


#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

namespace rsatb {
namespace calc {

/// Options of the LDOS map: the survey-sampling estimator of
/// doc/design_survey_sampling_ldos.md. Every probe's Chebyshev sweep yields
/// the estimator filter p(H) xi (a sqrt-Gaussian of width eta), a flat-top
/// sketch of the window q(H) xi with `sketch_families` Chebyshev modulations,
/// and the sketch's exact images p(H) q(H) xi -- all tables on one recursion.
/// The map is the captured part |Pi p(H) phi_r|^2 of the window's sketch,
/// divided by the global capture fraction c. Unbiased for an isotropic sketch
/// (the flat top makes it one); relative error sqrt(2 (1 - c) / k) with
/// k = sketch_families x n_samples the sketch rank, bounded by sqrt(2 / k)
/// at every system size, exact once k reaches the number of states in the
/// window. Cost O(n_samples x degree x N) plus O(k^2 N + k^3).
struct RealSpaceLdosOptions {
    std::vector<double> energies;  ///< Evaluation energies, eV, absolute.
    double eta = 0.1;              ///< Gaussian broadening (std of the window), eV.
    double emin = 0.0;             ///< Lower bound of the (H, S) pencil, eV.
    double emax = 0.0;             ///< Upper bound, eV. Required: emax > emin.
    double bound_pad = 0.01;       ///< Fractional pad on the fitted interval.
    double spacing = 0.4;          ///< Grid spacing, Angstrom.
    int box_edge = 0;              ///< Grid box edge; 0 = physical default.
    int n_samples = 64;            ///< Probes n; the sketch rank is sketch_families x n.
    int block = 16;                ///< Probes per Chebyshev sweep (SpMM width).
    /// Chebyshev modulation families per probe, J. Each family lengthens the
    /// sweep by degree/10 (the comb stride) and adds n columns to the sketch;
    /// families of one probe are correlated draws (measured design effect
    /// 1.0 at J = 6, 2.5 at J = 12), so J ~ 6 with probes supplying the rest
    /// of the rank is the efficient setting.
    int sketch_families = 6;
    double coeff_eps = 1e-8;       ///< Filter truncation eps_M (design 3.3).
    unsigned long seed = 12345;    ///< RNG seed; per-rank streams derive from it.
    bool complete_basis = false;   ///< Sweep the whole basis instead of
                                   ///< sampling: the estimator becomes exact.
                                   ///< Validation mode -- it holds an O(N_atom)
                                   ///< index on every rank and costs N states.
    bool certify_sampler = true;   ///< Spend one extra propagation pair on the
                                   ///< solve-form sampler certificate.
    MetricSamplerOptions sampler;  ///< Options of the S^{-1}-covariance sampler.
};

/// What the shared sweep-and-square engine reads. Two estimators use it:
/// the LDOS map (survey = true: sketch, images, captured part) and the
/// electron density (survey = false: plain squares plus the fragment
/// surrogate control variate of rs_fragment.hpp, enabled by
/// fragment_buffer > 0). The two never combine.
struct MapEngineOptions {
    std::vector<double> energies;
    double eta = 0.1;
    double emin = 0.0;
    double emax = 0.0;
    double bound_pad = 0.01;
    double spacing = 0.4;
    int box_edge = 0;
    int n_samples = 64;
    int block = 16;
    double coeff_eps = 1e-8;
    unsigned long seed = 12345;
    bool complete_basis = false;
    bool certify_sampler = true;
    MetricSamplerOptions sampler;
    bool survey = false;           ///< LDOS: flat-top sketch and captured map.
    int sketch_families = 6;       ///< J under survey.
    /// Density: fragment surrogate control variate. The buffer is the
    /// systematic knob (the residual falls exponentially with it); the core
    /// edge amortizes the dense fragment solves; the workspace cap bounds the
    /// fragments in flight (0 = one per thread); the fitted coefficient is
    /// the per-box regression of the plain map on the surrogate (false keeps
    /// the exactly unbiased coefficient one).
    double fragment_core = 4.5;
    double fragment_buffer = 0.0;
    int fragment_workspace_mb = 0;
    bool fit_fragment_coefficient = true;
};


/// The sizes a scaling study needs, measured rather than assumed.
///
/// Timings alone cannot separate "slow" from "not scaling": a per-rank cost
/// drifting toward O(N) and one that is merely large look the same on a clock.
/// These are the quantities the linear-scaling argument actually rests on, so
/// a run at two sizes shows directly which of them moved.
///
/// The ones to watch, and what they should do as atoms and ranks grow together
/// at fixed density:
///   owned_atoms, owned_boxes, owned_points   flat  (that is what O(N/P) means)
///   set_atoms / owned_atoms                  flat-ish; this ratio IS the halo,
///                                            and it grows only as surface area
///   max_box_orbitals                         FLAT. Set by cutoff and density,
///                                            not by cell size. If it climbs,
///                                            the box loop is not linear.
///   sweep seconds / matvecs                  flat per matvec
/// Anything here that tracks the total atom count instead is the bug.
struct RealSpaceLdosProfile {
    // Per rank, as measured on this one.
    int64_t owned_atoms = 0;
    int64_t owned_boxes = 0;
    int64_t owned_points = 0;
    int64_t set_atoms = 0;        ///< Atoms the orbital set carries: owned + halo.
    int64_t set_orbitals = 0;
    int64_t demand_records = 0;   ///< (rank, atom) pairs this rank shipped out.
    int64_t coeff_halo_doubles = 0;  ///< Coefficients gathered per sweep, all energies.
    GridWorkStats grid_work;      ///< Per-box sizes; see GridWorkStats.

    // Identical on every rank.
    int64_t global_points = 0;
    int shape[3] = {0, 0, 0};
    int box_edge[3] = {0, 0, 0};
    double spacing = 0.0;
    int n_energy = 0;
    int n_states = 0;
    int block = 0;
    int filter_degree = 0;
    int64_t matvecs = 0;          ///< Sweep length (filter and sketch tables) x blocks.
    double span_ev = 0.0;
    double eta_ev = 0.0;
};

struct RealSpaceLdosResult {
    RealSpaceGrid grid;
    GridDomain domain;                       ///< This rank's boxes.
    std::vector<std::vector<double>> ldos;   ///< [energy][owned point], states/eV/A^3.
    int filter_degree = 0;                   ///< Chebyshev terms per sweep.
    int n_states = 0;                        ///< Probes actually swept.
    double sampler_residual = -1.0;          ///< ||W q(W)^2 xi - xi||/||xi||, or -1.
    /// Survey estimator, per energy: the global capture fraction c (1 = the
    /// sketch spans the window and the map is exact), the sketch rank k the
    /// map was extrapolated from, and the predicted relative rms error
    /// sqrt(2 (1 - c) / k). Empty for the density engine.
    std::vector<double> capture;
    std::vector<int> sketch_rank;
    std::vector<double> predicted_rel_error;
    /// Density engine (fragment surrogate), per window: fragment count, rms of
    /// the stochastic correction the samples still carry, and the fitted
    /// control coefficient. Empty for the survey estimator.
    int n_fragments = 0;
    std::vector<double> correction_rms;
    std::vector<double> control_beta;
    RealSpaceLdosProfile profile;            ///< Sizes; see RealSpaceLdosProfile.
};


namespace detail {

/// Chebyshev tables of the sqrt-Gaussian filter, one per energy, all of one
/// degree M (design eqs. 3.3-3.5). The physical normalization
/// (2 pi eta^2)^{-1/4} is folded in, so accumulating with these coefficients
/// yields chi in eV^{-1/2} and |chi|^2 in the 1/eV the LDOS needs.
/// Eigendecomposition of a symmetric matrix in place (ascending): a
/// wrapper over dsyevd's two-call workspace dance, shared by every Gram
/// eigensolve in this file. On exit `a` holds the eigenvectors
/// column-major, `w` the eigenvalues. Throws on LAPACK failure.
inline void SymEig(int n, double* a, double* w) {
    int info = 0, lwork = -1, liwork = -1, iwq = 0;
    double wq = 0.0;
    dsyevd_("V", "U", &n, a, &n, w, &wq, &lwork, &iwq, &liwork, &info);
    lwork = static_cast<int>(wq);
    liwork = iwq;
    std::vector<double> work(static_cast<size_t>(lwork));
    std::vector<int> iwork(static_cast<size_t>(liwork));
    dsyevd_("V", "U", &n, a, &n, w, work.data(), &lwork, iwork.data(), &liwork,
            &info);
    if (info != 0) {
        throw std::runtime_error("Symmetric eigensolve (dsyevd) failed.");
    }
}

/// Counter-based Rademacher draw: the sign of (seed, global row, state) is a
/// pure function of its key, so every rank draws the identical value and the
/// sampled estimate is invariant to the rank count and the partitioning --
/// the property stateful per-rank streams cannot have. SplitMix64 finalizer
/// chain; same construction class as the counter-based generators of Salmon
/// et al. (SC'11, "Parallel random numbers: as easy as 1, 2, 3").
inline double RademacherSign(uint64_t seed, uint64_t global_row, uint64_t state) {
    auto mix = [](uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    const uint64_t h = mix(mix(mix(seed) ^ global_row) ^ state);
    return (h & 1ull) ? 1.0 : -1.0;
}

/// Drop the trailing coefficients of a filter table that fall below `floor`
/// RELATIVE to its largest one (two consecutive ones, as the KPM truncation
/// does): the degree the sweep runs to is the longest table actually needed.
/// Callers pass a floor four orders below the coefficient tolerance their
/// degree formula targets: the dropped tail then perturbs the filter at the
/// level the untruncated table already had (measured: a floor equal to the
/// tolerance itself moved the full-fragment density identity from 1e-11 to
/// 1e-8 of the peak).
inline void TruncateTableRelative(std::vector<std::complex<double>>& table, double floor_rel) {
    if (table.size() < 2 || floor_rel <= 0.0) return;
    double cmax = 0.0;
    for (const auto& c : table) cmax = std::max(cmax, std::abs(c));
    const double floor = floor_rel * cmax;
    size_t keep = table.size();
    for (size_t i = 0; i + 1 < table.size(); ++i) {
        if (std::abs(table[i]) <= floor && std::abs(table[i + 1]) <= floor) {
            keep = i + 1;
            break;
        }
    }
    table.resize(std::max<size_t>(1, keep));
}

inline std::vector<std::vector<std::complex<double>>> SqrtGaussianTables(
    const std::vector<double>& energies, double eta, double center, double half,
    double eps) {
    // m is the centre-of-spectrum worst case: a window at x0 has angular width
    // eta_bar / sqrt(1 - x0^2) and its coefficients reach eps at a degree
    // smaller by that factor -- 0.5-0.65x for the band energies of a 200 eV
    // LCAO spectrum. Each table is therefore TRUNCATED where its own
    // coefficients fall below eps (the same tolerance m was sized for), and
    // the shared sweep runs to the longest table actually needed: 1.6x fewer
    // matvecs on silicon, identical results to eps.
    const int m = static_cast<int>(std::ceil((half / eta) * std::sqrt(std::log(1.0 / eps)))) + 8;
    const double norm = std::pow(2.0 * M_PI * eta * eta, -0.25);
    std::vector<std::vector<std::complex<double>>> tables;
    tables.reserve(energies.size());
    for (const double energy : energies) {
        const double e_bar = (energy - center) / half;
        const double eta_bar = eta / half;
        auto table = chebyshev_coefficients_from_function(
            m,
            [e_bar, eta_bar](double x) {
                const double d = (e_bar - x) / (2.0 * eta_bar);
                return std::exp(-d * d);
            },
            KpmKernel::None, /*trunc_eps=*/0.0);
        TruncateTableRelative(table, 1e-4 * eps);
        for (auto& c : table) c *= norm;
        tables.push_back(std::move(table));
    }
    return tables;
}

/// Chebyshev tables of a flat-top window: 1 over |E - E_k| < 4 eta with erf
/// edges of width eta (10-90 % over ~1.8 eta), 0 outside -- the sketch filter
/// of the survey-sampling estimator. Same
/// degree formula and truncation as the sqrt-Gaussian: the edge sets the
/// degree, and it is the Gaussian's own width.
inline std::vector<std::vector<std::complex<double>>> FlatTopTables(
    const std::vector<double>& energies, double eta, double center, double half,
    double eps) {
    const int m = static_cast<int>(std::ceil((half / eta) * std::sqrt(std::log(1.0 / eps)))) + 8;
    std::vector<std::vector<std::complex<double>>> tables;
    tables.reserve(energies.size());
    for (const double energy : energies) {
        const double e_bar = (energy - center) / half;
        const double eta_bar = eta / half;
        auto table = chebyshev_coefficients_from_function(
            m,
            [e_bar, eta_bar](double x) {
                const double lo = (x - (e_bar - 4.0 * eta_bar)) / eta_bar;
                const double hi = (x - (e_bar + 4.0 * eta_bar)) / eta_bar;
                return 0.5 * (std::erf(lo) - std::erf(hi));
            },
            KpmKernel::None, /*trunc_eps=*/0.0);
        TruncateTableRelative(table, 1e-4 * eps);
        tables.push_back(std::move(table));
    }
    return tables;
}

/// Chebyshev series of the product of two tabulated filters (real
/// coefficients), from T_i T_j = (T_{i+j} + T_{|i-j|}) / 2, truncated where
/// its tail falls below eps relative: the image table p q of the sketch.
inline std::vector<std::complex<double>> ProductTable(
    const std::vector<std::complex<double>>& ta,
    const std::vector<std::complex<double>>& tb, double eps) {
    const int ma = static_cast<int>(ta.size()), mb = static_cast<int>(tb.size());
    std::vector<double> q(static_cast<size_t>(ma + mb - 1), 0.0);
    for (int i = 0; i < ma; ++i) {
        for (int j = 0; j < mb; ++j) {
            const double v = 0.5 * ta[i].real() * tb[j].real();
            q[i + j] += v;
            q[std::abs(i - j)] += v;
        }
    }
    double amax = 0.0;
    for (double v : q) amax = std::max(amax, std::abs(v));
    int last = 0;
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        if (std::abs(q[i]) > eps * amax) last = i;
    }
    std::vector<std::complex<double>> out(last + 1);
    for (int i = 0; i <= last; ++i) out[i] = q[i];
    return out;
}

/// Chebyshev series of T_j times a tabulated filter, from the product
/// identity T_j T_m = (T_{m+j} + T_{|m-j|}) / 2 -- a shift-fold of the
/// coefficients, degree grows by j only. The sketch families of one probe
/// are its sketch under T_0, T_s, T_2s, ... (s the comb stride), so J
/// families cost J - 1 shift-folds and no extra matvec.
inline std::vector<std::complex<double>> ChebyshevTimesTable(
    int j, const std::vector<std::complex<double>>& t) {
    if (j == 0) return t;
    const int m = static_cast<int>(t.size());
    std::vector<std::complex<double>> out(m + j, 0.0);
    for (int i = 0; i < m; ++i) {
        out[i + j] += 0.5 * t[i];
        out[std::abs(i - j)] += 0.5 * t[i];
    }
    return out;
}

}  // namespace detail

/// Prints the profile, reduced across ranks, on the logging rank.
///
/// Every per-rank row is min / mean / max with the imbalance max/mean beside
/// it, because for a distributed run the spread is the interesting part: a
/// mean that halves when ranks double proves nothing if one rank kept all the
/// work. Collective -- every rank must call it.
inline void ReportRealSpaceLdosProfile(const RealSpaceLdosProfile& p, MPI_Comm comm) {
    int size = 1;
    if (comm != MPI_COMM_NULL) MPI_Comm_size(comm, &size);

    // Order fixed here and mirrored below; nine per-rank quantities.
    double local[9] = {
        static_cast<double>(p.owned_atoms),
        static_cast<double>(p.owned_boxes),
        static_cast<double>(p.owned_points),
        static_cast<double>(p.set_atoms),
        static_cast<double>(p.set_orbitals),
        static_cast<double>(p.demand_records),
        static_cast<double>(p.coeff_halo_doubles),
        static_cast<double>(p.grid_work.boxes_contracted),
        static_cast<double>(p.grid_work.max_box_orbitals),
    };
    double lo[9], hi[9], sum[9];
    double box_orb_sum = static_cast<double>(p.grid_work.sum_box_orbitals);
    double kept = static_cast<double>(p.grid_work.points_kept);
    double total_rows = static_cast<double>(p.grid_work.points_total);
    if (comm != MPI_COMM_NULL && size > 1) {
        MPI_Allreduce(local, lo, 9, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(local, hi, 9, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(local, sum, 9, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &box_orb_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &kept, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &total_rows, 1, MPI_DOUBLE, MPI_SUM, comm);
    } else {
        for (int i = 0; i < 9; ++i) { lo[i] = hi[i] = sum[i] = local[i]; }
    }
    if (!util::Log::enabled(util::LogLevel::kSummary)) return;

    const double inv = 1.0 / size;
    const char* names[9] = {"atoms", "boxes", "points", "set atoms", "set orbitals",
                            "demands out", "halo doubles", "boxes done", "max box orb"};
    std::printf("\n  real-space LDOS profile\n");
    std::printf("    grid          %d x %d x %d = %lld points, spacing %.3f A, "
                "box edge %d/%d/%d\n",
                p.shape[0], p.shape[1], p.shape[2],
                static_cast<long long>(p.global_points), p.spacing,
                p.box_edge[0], p.box_edge[1], p.box_edge[2]);
    std::printf("    spectrum      span %.3f eV, eta %.4f eV -> filter degree %d\n",
                p.span_ev, p.eta_ev, p.filter_degree);
    std::printf("    work          %d states x %d energies, block %d, %lld matvecs\n",
                p.n_states, p.n_energy, p.block, static_cast<long long>(p.matvecs));
    std::printf("    per rank (%d)   %12s %12s %12s   imbalance\n", size, "min", "mean", "max");
    for (int i = 0; i < 9; ++i) {
        const double mean = sum[i] * inv;
        const double imb = (mean > 0.0) ? hi[i] / mean : 1.0;
        std::printf("      %-13s %12.0f %12.1f %12.0f   %6.2f\n",
                    names[i], lo[i], mean, hi[i], imb);
    }
    const double halo_ratio = (sum[0] > 0.0) ? sum[3] / sum[0] : 0.0;
    const double mean_box_orb =
        (sum[7] > 0.0) ? box_orb_sum / sum[7] : 0.0;
    const double trim = (total_rows > 0.0) ? kept / total_rows : 1.0;
    std::printf("    halo ratio    %.3f set atoms per owned atom  (surface/volume; "
                "must not track N)\n", halo_ratio);
    std::printf("    box orbitals  %.1f mean, %.0f max  (cutoff x density; must be "
                "flat in N)\n", mean_box_orb, hi[8]);
    std::printf("    trim kept     %.3f of box rows\n", trim);
    std::fflush(stdout);
}

namespace detail {

/// The engine shared by the LDOS map and the electron density: sweep a
/// stochastic (or complete) basis through Chebyshev tables, one output
/// channel per table. Under opt.survey the sweep also carries the sketch
/// and image tables and the channel is the captured map; otherwise the
/// channel is the mean square onto the grid, with the fragment surrogate
/// control variate when opt.fragment_buffer > 0. The tables are built by
/// the caller THROUGH the callback because the scaled interval (center,
/// half) they must be fitted on is computed here, from emin/emax/bound_pad.
///
/// opt.energies and opt.eta are read only by the survey estimator (its
/// sketch window); each wrapper validates and encodes its own filter physics
/// into the tables.
inline RealSpaceLdosResult FilteredSquareMap(
    vbcsr::atomic::AtomicData& atomic_data,
    vbcsr::BlockSpMat<double>& H,
    vbcsr::BlockSpMat<double>* S,
    vbcsr::BlockSpMat<double>& Zt,
    ChebyOpForm form,
    const OrbitalCollocation& collocation,
    const MapEngineOptions& opt,
    const std::function<std::vector<std::vector<std::complex<double>>>(
        double center, double half)>& table_builder,
    MPI_Comm comm, FragmentBasisCache* fragment_cache = nullptr) {
    using vbcsr::DistGraph;
    using vbcsr::DistMultiVector;
    using vbcsr::DistVector;

    if (!(opt.emax > opt.emin)) {
        throw std::runtime_error(
            "Real-space LDOS needs spectral bounds emin < emax of the (H, S) pencil.");
    }
    if (opt.block < 1 || (!opt.complete_basis && opt.n_samples < 1)) {
        throw std::runtime_error("Real-space LDOS needs at least one sample and block >= 1.");
    }
    if (form == ChebyOpForm::Orthogonal) {
        throw std::runtime_error(
            "The real-space LDOS needs a non-orthogonal pencil; there is no S^{-1/2} to "
            "carry the metric in an orthogonal basis.");
    }
    if (form == ChebyOpForm::Solve && S == nullptr) {
        throw std::runtime_error("The solve-form real-space LDOS needs the overlap S.");
    }
    if (opt.survey && opt.fragment_buffer > 0.0) {
        throw std::runtime_error(
            "The survey LDOS estimator and the fragment surrogate never combine.");
    }

    int rank = 0, size = 1;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    std::vector<double> type_rcut(collocation.n_type());
    for (int t = 0; t < collocation.n_type(); ++t) type_rcut[t] = collocation.rcut_angstrom(t);

    RealSpaceLdosResult result;
    result.grid = MakeRealSpaceGrid(atomic_data.cell, opt.spacing, opt.box_edge);

    // Box ownership comes from the atom partition itself -- whatever produced
    // it -- so nothing here needs to know how the atoms were distributed, and
    // no rank ever walks a box outside its own share.
    GridOwnership ownership;
    {
        RSATB_PHASE("rs_ldos/ownership");
        ownership = ResolveGridOwnership(atomic_data, result.grid, type_rcut, comm);
    }
    result.domain = MakeGridDomain(result.grid, ownership.boxes);

    OrbitalSetExchangePlan plan;
    OrbitalSet set;
    {
        RSATB_PHASE("rs_ldos/orbital_set");
        set = BuildOrbitalSetFromDemands(atomic_data, result.grid, ownership,
                                         type_rcut, rank, comm, &plan);
    }

    RealSpaceLdosProfile& prof = result.profile;
    prof.owned_atoms = atomic_data.n_atom;
    prof.owned_boxes = result.domain.n_box();
    prof.owned_points = result.domain.n_point();
    prof.set_atoms = set.n_atom();
    prof.set_orbitals = set.n_orb();
    for (int c : plan.send_counts) prof.demand_records += c;
    prof.global_points = result.grid.n_point();
    for (int d = 0; d < 3; ++d) {
        prof.shape[d] = result.grid.shape[d];
        prof.box_edge[d] = result.grid.box_edge[d];
    }
    prof.spacing = opt.spacing;

    // Scaled interval of the pencil, padded so the filter never samples the
    // polynomial outside its fitted range.
    const double pad = opt.bound_pad * (opt.emax - opt.emin);
    const double lo = opt.emin - pad, hi = opt.emax + pad;
    const double center = 0.5 * (hi + lo), half = 0.5 * (hi - lo);
    const auto tables = table_builder(center, half);
    if (tables.empty()) {
        throw std::runtime_error("FilteredSquareMap: the table builder returned no filter.");
    }
    const int degree = static_cast<int>(tables.front().size());
    result.filter_degree = degree;

    ISQRTMPreconditioner<double> precond(Zt, kBorrowFactor);

    // The sandwich form needs neither the sampler (its seed covariance is
    // already the identity) nor S inside the operator. Built before the
    // fragment surrogate: the solve chain compresses the sampler's own
    // coloring polynomial.
    std::unique_ptr<MetricSampler<double>> sampler;
    std::unique_ptr<Chebyshev<double>> cheb;
    if (form == ChebyOpForm::Solve) {
        sampler = std::make_unique<MetricSampler<double>>(*S, Zt, opt.sampler);
        cheb = std::make_unique<Chebyshev<double>>(H, *S, precond,
                                                   PreconditionerType::Symmetric, lo, hi);
    } else {
        cheb = std::make_unique<Chebyshev<double>>(H, precond,
                                                   PreconditionerType::Symmetric, lo, hi);
    }

    // Fragment control variate: deterministic local baseline + corrected
    // remainder (rs_fragment.hpp), built from the SAME operator chain the
    // estimator applies to its raw white seeds so the surrogate matches the
    // operator's gauge: (Zt, H) for the sandwich form, and additionally
    // (S, q) for the solve form, whose full chain is p(S^-1 H) Zt q(W)
    // (design_subspace_deflation.md section 9.3).
    FragmentCv frag_cv;
    FragmentBasis local_basis;  // owns the basis when no cache is supplied
    if (opt.fragment_buffer > 0.0) {
        if (opt.complete_basis) {
            throw std::runtime_error(
                "complete_basis is already exact; fragment_buffer does not apply.");
        }
        SolveChainSpec chain;
        if (form == ChebyOpForm::Solve) {
            chain.S = S;
            chain.q = sampler->q_coeffs();
            chain.lo = sampler->q_lo();
            chain.hi = sampler->q_hi();
        }
        RSATB_PHASE("rs_ldos/fragment_build");
        // The basis carries the per-table rows for THIS task's tables (the
        // eigenvectors are not retained -- FragmentBasis::P); with a cache
        // from the caller a later task with the same geometry, chain and
        // tables reuses it, any other rebuilds.
        const FragmentBasis* fb = nullptr;
        if (fragment_cache != nullptr && fragment_cache->valid &&
            fragment_cache->core_edge == opt.fragment_core &&
            fragment_cache->buffer == opt.fragment_buffer &&
            fragment_cache->solve_chain == chain.active() &&
            fragment_cache->basis.tables == tables &&
            fragment_cache->basis.center == center &&
            fragment_cache->basis.half == half) {
            fb = &fragment_cache->basis;
        } else if (fragment_cache != nullptr) {
            fragment_cache->invalidate();
            fragment_cache->basis = BuildFragmentBasis(
                atomic_data, H, Zt, opt.fragment_core, opt.fragment_buffer, tables,
                center, half, comm, chain, opt.fragment_workspace_mb);
            fragment_cache->valid = true;
            fragment_cache->core_edge = opt.fragment_core;
            fragment_cache->buffer = opt.fragment_buffer;
            fragment_cache->solve_chain = chain.active();
            fb = &fragment_cache->basis;
        } else {
            local_basis = BuildFragmentBasis(
                atomic_data, H, Zt, opt.fragment_core, opt.fragment_buffer, tables,
                center, half, comm, chain, opt.fragment_workspace_mb);
            fb = &local_basis;
        }
        frag_cv = TablesToCv(*fb, tables, center, half);
        result.n_fragments = frag_cv.n_frag();
    }

    DistGraph* graph = H.graph;
    const int n_energy = static_cast<int>(tables.size());
    const int n_total = opt.complete_basis ? graph->global_scalar_rows() : opt.n_samples;
    result.n_states = n_total;
    result.ldos.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));

    int64_t n_blocks = 0;
    std::mt19937_64 gen(opt.seed + 977 * static_cast<unsigned long>(rank));
    std::bernoulli_distribution coin(0.5);

    // The global basis enumeration, gid-major: basis vector s lives on the
    // atom g whose orbital range covers s. The enumeration is this function's
    // own, not the library's row ordering. It used to serve only the
    // complete-basis sweep; the counter-based seeding keys every random draw
    // on the GLOBAL row it lands on, so the enumeration is now unconditional
    // -- one O(N_atom) allgather, negligible against any sweep.
    std::vector<std::pair<int64_t, int>> basis_of_owned;  // (offset, local atom)
    std::vector<int64_t> global_row_of_local;  // local scalar row -> global
    int64_t n_global_rows = 0;
    std::vector<int> local_row_offset(atomic_data.n_atom + 1, 0);
    for (int a = 0; a < atomic_data.n_atom; ++a) {
        local_row_offset[a + 1] =
            local_row_offset[a] + atomic_data.type_norb[atomic_data.atom_type[a]];
    }
    // Sign keys need a rank-count-INVARIANT row numbering. The graph's global
    // index is the atom's position in the post-partition ownership order
    // (DistGraph iota), so it permutes with the rank count: keying random
    // draws on it gave a different realization -- a one-noise-quantum shift
    // of every stochastic map -- at every np (measured: 3796.49 vs 3786.08
    // electrons on 216-atom Al at np 1 vs 2, deterministic paths identical
    // to 13 digits). The ORIGINAL file id (atomic_data.atom_index) is the
    // stable name of an atom, so the draws below are keyed by the row
    // numbering the original order induces, while every enumeration and
    // matrix structure keeps the graph numbering it is built on.
    std::vector<int64_t> seed_row_of_local;    // owned scalar row -> invariant row
    std::vector<int64_t> seed_row0_of_atom;    // graph atom gid -> invariant row0
    std::vector<int64_t> atom_prefix;          // graph atom gid -> global row0
    {
        std::vector<int> mine;
        mine.reserve(3 * static_cast<size_t>(atomic_data.n_atom));
        for (int a = 0; a < atomic_data.n_atom; ++a) {
            mine.push_back(atomic_data.get_global_index(a));
            mine.push_back(local_row_offset[a + 1] - local_row_offset[a]);
            mine.push_back(atomic_data.atom_index[a]);
        }
        std::vector<int> all = mine;
        if (comm != MPI_COMM_NULL && size > 1) {
            const int local_count = static_cast<int>(mine.size());
            std::vector<int> counts(size, 0);
            MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
            std::vector<int> displs(size + 1, 0);
            for (int r = 0; r < size; ++r) displs[r + 1] = displs[r] + counts[r];
            all.assign(displs[size], 0);
            MPI_Allgatherv(mine.data(), local_count, MPI_INT, all.data(), counts.data(),
                           displs.data(), MPI_INT, comm);
        }
        const size_t n_atom_global = all.size() / 3;
        atom_prefix.assign(n_atom_global + 1, 0);
        std::vector<int64_t>& prefix = atom_prefix;
        {
            std::vector<int> norb_of(n_atom_global, 0);
            for (size_t i = 0; i < n_atom_global; ++i) {
                norb_of[static_cast<size_t>(all[3 * i])] = all[3 * i + 1];
            }
            for (size_t g = 0; g < n_atom_global; ++g) prefix[g + 1] = prefix[g] + norb_of[g];
        }
        if (prefix.back() != graph->global_scalar_rows()) {
            throw std::runtime_error(
                "complete_basis enumeration disagrees with the matrix size; the atom global "
                "indices are not the dense 0..N_atom-1 this mode assumes.");
        }
        {
            // The invariant numbering: row offsets in ORIGINAL atom order.
            std::vector<int> norb_by_orig(n_atom_global, 0);
            for (size_t i = 0; i < n_atom_global; ++i) {
                const size_t orig = static_cast<size_t>(all[3 * i + 2]);
                if (orig >= n_atom_global) {
                    throw std::runtime_error(
                        "Original atom ids are not the dense 0..N_atom-1 the "
                        "invariant seed numbering assumes.");
                }
                norb_by_orig[orig] = all[3 * i + 1];
            }
            std::vector<int64_t> orig_prefix(n_atom_global + 1, 0);
            for (size_t g = 0; g < n_atom_global; ++g) {
                orig_prefix[g + 1] = orig_prefix[g] + norb_by_orig[g];
            }
            seed_row0_of_atom.assign(n_atom_global, 0);
            for (size_t i = 0; i < n_atom_global; ++i) {
                seed_row0_of_atom[static_cast<size_t>(all[3 * i])] =
                    orig_prefix[static_cast<size_t>(all[3 * i + 2])];
            }
        }
        basis_of_owned.reserve(atomic_data.n_atom);
        for (int a = 0; a < atomic_data.n_atom; ++a) {
            basis_of_owned.emplace_back(prefix[atomic_data.get_global_index(a)], a);
        }
        std::sort(basis_of_owned.begin(), basis_of_owned.end());
        global_row_of_local.assign(local_row_offset.back(), 0);
        seed_row_of_local.assign(local_row_offset.back(), 0);
        n_global_rows = prefix.back();
        for (int a = 0; a < atomic_data.n_atom; ++a) {
            const int64_t g0 = prefix[atomic_data.get_global_index(a)];
            const int64_t s0 = seed_row0_of_atom[atomic_data.get_global_index(a)];
            for (int o = 0; o < local_row_offset[a + 1] - local_row_offset[a]; ++o) {
                global_row_of_local[local_row_offset[a] + o] = g0 + o;
                seed_row_of_local[local_row_offset[a] + o] = s0 + o;
            }
        }
    }

    if (opt.certify_sampler && sampler) {
        DistVector<double> xi(graph);
        for (int i = 0; i < xi.local_size; ++i) xi[i] = coin(gen) ? 1.0 : -1.0;
        result.sampler_residual = sampler->Residual(xi);
    }

    std::unique_ptr<DistMultiVector<double>> xi, c0, v0, v1, v2, chi, chi0;
    // Flat accumulator for ALL filter tables of the current block, laid out
    // column-major (L x T_all) with L = local_rows * blk: the sweep folds
    // buffered recursion iterates into every table with one GEMM per flush
    // (blocked-moment accumulation) instead of T per-step axpys -- an
    // order-of-magnitude less accumulator traffic at many tables (deep
    // combs, multi-energy scans), and the same memory the per-table
    // DistMultiVector fleet used to hold.
    std::vector<double> acc_all;
    std::vector<double> coeffs, values, values0;
    // The fragment Apply reads the seed block over GLOBAL rows. For identity
    // blocks it is built analytically; for random blocks every rank evaluates
    // the counter-based draw for all rows itself -- no communication either
    // way, and bitwise the same at any rank count.
    std::vector<double> xi_global;

    // Columns start .. start+blk of the identity, each set by the one rank
    // owning the atom the basis vector lives on. Shared by the complete-basis
    // sweep and the fragment baseline, which streams the columns of T0
    // through the same contraction.
    auto fill_identity = [&](DistMultiVector<double>& mv, int start, int blk) {
        mv.scale(0.0);
        for (int v = 0; v < blk; ++v) {
            const int64_t s = start + v;
            auto it = std::upper_bound(basis_of_owned.begin(), basis_of_owned.end(),
                                       std::make_pair(s, std::numeric_limits<int>::max()));
            if (it == basis_of_owned.begin()) continue;
            const int64_t off = (it - 1)->first;
            const int a = (it - 1)->second;
            const int norb = local_row_offset[a + 1] - local_row_offset[a];
            if (s < off + norb) {
                mv(local_row_offset[a] + static_cast<int>(s - off), v) = 1.0;
            }
        }
    };

    // Fragment baseline: the exact map of T0, as the column sum
    // n0(r) = sum_s |Phi T0 e_s|^2(r), streamed in blocks through the same
    // halo + squaring path the samples use so every convention is shared.
    // Kept separate from the output: the final assembly scales it by the
    // regression coefficient estimated below.
    std::vector<std::vector<double>> base;
    if (frag_cv.enabled()) {
        base.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));
    }
    if (frag_cv.enabled()) {
        RSATB_PHASE("rs_ldos/fragment_baseline");
        const int64_t n_basis = graph->global_scalar_rows();
        // T0's columns against identity seeds are stored numbers, not
        // products (ApplyIdentity), and they are spatially local. The GEMM
        // work per pass was always bounded by the overlap geometry; what
        // made the STREAM quadratic was everything around it touching the
        // full domain once per block -- the fragment scan, the chi0 blanket
        // zero, the dense set gather, the all-box screening loop, and the
        // full-grid accumulate. Each is now bounded by the block's actual
        // support: the inverted column index finds the touched fragments,
        // the sparse gather moves only their core rows, and the reachable
        // owned boxes are enumerated from geometry -- so per-block work is
        // O(touched) and the whole stream is O(N), which the per-pass GEMM
        // already was. The identical products are formed in the identical
        // order; untouched boxes contribute the exact zeros they always did.
        const int base_blk = std::max(opt.block, 512);
        std::vector<int> chi_rows, coeff_slots, active_slots, touched_atoms;
        std::vector<char> atom_flag(atomic_data.n_atom, 0);
        std::vector<char> slot_flag(result.domain.n_box(), 0);
        std::unordered_map<int, int> slot_of_box;
        slot_of_box.reserve(result.domain.boxes.size());
        for (int s = 0; s < result.domain.n_box(); ++s) {
            slot_of_box.emplace(result.domain.boxes[s], s);
        }
        double max_box_radius = 0.0;
        for (const int b : result.domain.boxes) {
            max_box_radius = std::max(max_box_radius, result.grid.BoxAt(b).radius);
        }
        const auto cinv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());
        // Fractional half-extent of a Cartesian reach R along each axis.
        double frac_per_ang[3];
        for (int d = 0; d < 3; ++d) {
            frac_per_ang[d] = std::sqrt(cinv[0 + d] * cinv[0 + d] +
                                        cinv[3 + d] * cinv[3 + d] +
                                        cinv[6 + d] * cinv[6 + d]);
        }
        for (int64_t start = 0; start < n_basis; start += base_blk) {
            const int blk = static_cast<int>(std::min<int64_t>(base_blk, n_basis - start));
            if (!chi0 || chi0->num_vectors != blk) {
                chi0 = std::make_unique<DistMultiVector<double>>(graph, blk);
                chi_rows.clear();
            }
            for (int k = 0; k < n_energy; ++k) {
                frag_cv.ApplyIdentity(k, start, blk, chi0.get(), &chi_rows);
                if (k == 0) {
                    // The touched support is the same for every table of the
                    // block (same fragments, different values): flag the
                    // owned atoms once, and find the reachable owned boxes
                    // once, after the gather names the set atoms.
                    for (const int a : touched_atoms) atom_flag[a] = 0;
                    touched_atoms.clear();
                    int a = 0;
                    for (const int row : chi_rows) {
                        while (local_row_offset[a + 1] <= row) ++a;
                        if (!atom_flag[a]) {
                            atom_flag[a] = 1;
                            touched_atoms.push_back(a);
                        }
                    }
                }
                GatherSetRowsSparse(atomic_data, *chi0, set, plan, comm, atom_flag,
                                    &coeffs, &coeff_slots);
                if (k == 0) {
                    for (const int slot : active_slots) slot_flag[slot] = 0;
                    active_slots.clear();
                    for (const int sa : coeff_slots) {
                        const double* p = set.positions.data() + 3 * static_cast<size_t>(sa);
                        const double reach =
                            type_rcut[set.atom_type[sa]] + max_box_radius;
                        double frac[3];
                        for (int d = 0; d < 3; ++d) {
                            frac[d] = p[0] * cinv[0 + d] + p[1] * cinv[3 + d] +
                                      p[2] * cinv[6 + d];
                        }
                        int lo[3], hi[3];
                        for (int d = 0; d < 3; ++d) {
                            const double per_box = static_cast<double>(
                                result.grid.box_edge[d]) / result.grid.shape[d];
                            const double ext = reach * frac_per_ang[d];
                            lo[d] = static_cast<int>(
                                std::floor((frac[d] - ext) / per_box));
                            hi[d] = static_cast<int>(
                                std::floor((frac[d] + ext) / per_box));
                            if (!atomic_data.pbc[d]) {
                                lo[d] = std::max(lo[d], 0);
                                hi[d] = std::min(hi[d], result.grid.box_count[d] - 1);
                            }
                        }
                        for (int bx = lo[0]; bx <= hi[0]; ++bx) {
                            const int wx = ((bx % result.grid.box_count[0]) +
                                            result.grid.box_count[0]) %
                                           result.grid.box_count[0];
                            for (int by = lo[1]; by <= hi[1]; ++by) {
                                const int wy = ((by % result.grid.box_count[1]) +
                                                result.grid.box_count[1]) %
                                               result.grid.box_count[1];
                                for (int bz = lo[2]; bz <= hi[2]; ++bz) {
                                    const int wz =
                                        ((bz % result.grid.box_count[2]) +
                                         result.grid.box_count[2]) %
                                        result.grid.box_count[2];
                                    const int id =
                                        (wx * result.grid.box_count[1] + wy) *
                                            result.grid.box_count[2] +
                                        wz;
                                    const auto it = slot_of_box.find(id);
                                    if (it == slot_of_box.end()) continue;
                                    if (slot_flag[it->second]) continue;
                                    slot_flag[it->second] = 1;
                                    active_slots.push_back(it->second);
                                }
                            }
                        }
                    }
                }
                collocation.SquareVectorsOntoBoxes(atomic_data, set, coeffs, blk,
                                                   result.grid, result.domain, &values,
                                                   &prof.grid_work, &active_slots);
                double* out = base[k].data();
                for (const int slot : active_slots) {
                    for (int64_t pnt = result.domain.offset[slot];
                         pnt < result.domain.offset[slot + 1]; ++pnt) {
                        out[pnt] += static_cast<double>(blk) * values[pnt];
                    }
                }
            }
        }
    }
    // Regression control variate: the plain estimator X and the sampled
    // surrogate Y accumulate separately, along with the per-block moments the
    // coefficient beta = Cov(X_b, Y_b) / Var(Y_b) is estimated from. The map
    // assembles as X - beta (Y - n0): beta = 1 is the raw control variate,
    // beta = 0 the plain estimator, and the estimated optimum is never worse
    // than either (standard regression CV; the moments aggregate over all
    // points, so at a handful of blocks beta is still sharply determined).
    std::vector<std::vector<double>> cv_X, cv_Y, cv_PXY, cv_PYY;
    std::vector<double> mom_xy, mom_yy;
    int64_t cv_blocks = 0;

    // The survey estimator stores every probe's filtered state and its
    // sketch and image columns across the sweep and assembles the captured
    // map afterwards; the density (and the complete-basis sweep, which is
    // exact by itself) squares each block onto the grid as it comes.
    const bool survey = opt.survey && !opt.complete_basis;
    // Solve form: the seed covariance is S^-1, so the sketch is whitened in
    // the S metric -- Grams Y^T (S Y) -- and the grid stage consumes the
    // states directly (no Zt): the sandwich construction conjugated by
    // S^{1/2}.
    const bool survey_metric = survey && form == ChebyOpForm::Solve;
    std::vector<std::unique_ptr<DistMultiVector<double>>> y_store, z_store;
    std::vector<std::pair<int, int>> blk_info;                    // (start, width)
    // The survey's table family, all riding the same recursion: per energy
    // k, the J sketch tables T_{js} q_k and their exact images T_{js} p_k q_k.
    // Positions in xtab (and z_store):
    //   sketch(k, j) = k J + j,          j = 0..J-1
    //   image(k, j)  = n_E J + k J + j
    const int J = survey ? std::max(1, opt.sketch_families) : 1;
    std::vector<std::vector<std::complex<double>>> xtab;  // sketch + image tables
    int degree_all = degree;
    auto idx_sketch = [&](int k, int j) { return k * J + j; };
    auto idx_image = [&](int k, int j) { return n_energy * J + k * J + j; };
    std::vector<std::vector<std::complex<double>>> sketch_tables;
    if (survey) {
        // Comb stride: consecutive T_1, T_2, ... are constants across a
        // +-5 eta window (they vary on the spectrum scale, the window is
        // ~10 eta / h of it), so they would add no independent shape --
        // measured as a sketch rank pinned at n on a 485-state metallic
        // window. T_j and T_{j'} decorrelate over the window once
        // |j - j'| dtheta >~ pi, dtheta = (10 eta / h) / sqrt(1 - x0^2) the
        // window's angular width, so the family is the frequency comb
        // j = 0, s, 2s, ... with s = pi sqrt(1 - x0^2) h / (10 eta) ~
        // degree / 10. The image tables then run the recursion to
        // degree + (J - 1) s ~ degree (1 + J / 10).
        std::vector<int> stride(n_energy, 1);
        for (int k = 0; k < n_energy; ++k) {
            const double x0 =
                std::max(-1.0, std::min(1.0, (opt.energies[k] - center) / half));
            const double win = 10.0 * opt.eta / half;
            stride[k] = std::max(
                1, static_cast<int>(std::lround(
                       M_PI * std::sqrt(std::max(1.0 - x0 * x0, 1e-4)) / win)));
        }
        sketch_tables = FlatTopTables(opt.energies, opt.eta, center, half, opt.coeff_eps);
        for (int k = 0; k < n_energy; ++k) {
            for (int j = 0; j < J; ++j) {
                xtab.push_back(ChebyshevTimesTable(j * stride[k], sketch_tables[k]));
            }
        }
        for (int k = 0; k < n_energy; ++k) {
            const auto pq = ProductTable(tables[k], sketch_tables[k], opt.coeff_eps);
            for (int j = 0; j < J; ++j) {
                xtab.push_back(ChebyshevTimesTable(j * stride[k], pq));
            }
        }
        for (const auto& t : xtab) {
            degree_all = std::max(degree_all, static_cast<int>(t.size()));
        }
        y_store.resize(n_energy);
        z_store.resize(static_cast<size_t>(xtab.size()));
    }

    if (frag_cv.enabled()) {
        cv_X.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));
        cv_Y.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));
        cv_PXY.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));
        cv_PYY.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));
        mom_xy.assign(n_energy, 0.0);
        mom_yy.assign(n_energy, 0.0);
    }

    for (int start = 0; start < n_total; start += opt.block) {
        const int blk = std::min(opt.block, n_total - start);
        auto need = [&](std::unique_ptr<DistMultiVector<double>>& mv) {
            if (!mv || mv->num_vectors != blk) {
                mv = std::make_unique<DistMultiVector<double>>(graph, blk);
            }
        };
        need(xi); need(v0); need(v1); need(v2);
        if (sampler) need(c0); else need(chi);
        if (frag_cv.enabled()) need(chi0);

        xi->scale(0.0);
        if (opt.complete_basis) {
            fill_identity(*xi, start, blk);
        } else {
            // Counter-based draw keyed on (seed, INVARIANT row, state): the
            // invariant numbering is the one the original atom order induces,
            // so the same physical basis row draws the same sign on every
            // rank AND at every rank count. The graph row would not do: it is
            // partition-order dependent (DistGraph iota), and keying on it
            // made the whole stochastic map shift by one noise quantum with
            // the rank count.
            for (int row = 0; row < xi->local_rows; ++row) {
                double* r = xi->row_data(row);
                const uint64_t grow = static_cast<uint64_t>(seed_row_of_local[row]);
                for (int v = 0; v < blk; ++v) {
                    r[v] = RademacherSign(opt.seed, grow,
                                          static_cast<uint64_t>(start + v));
                }
            }
        }
        // The seed the recursion starts from, in the representation the
        // operator iterates: c0 with covariance S^-1 for the solve form, and
        // xi itself for the sandwich form, whose covariance is already I.
        if (sampler) {
            sampler->Sample(*xi, *c0);
            v0->copy_from(*c0);
        } else {
            v0->copy_from(*xi);
        }

        // One sweep, all energies: X_k = sum_m N c_m(E_k) T_m(Op) seed.
        {
            RSATB_PHASE("rs_ldos/sweep");
            const int T_all = n_energy + static_cast<int>(xtab.size());
            const size_t L = static_cast<size_t>(v0->local_rows) * blk;
            const int B = 8;  // ring depth: Vbuf is B/T_all of the store
            acc_all.assign(L * static_cast<size_t>(T_all), 0.0);
            std::vector<double> Vbuf(L * B), Cbuf(static_cast<size_t>(B) * T_all);
            int nb_buf = 0, m_first = 0;
            auto flush = [&]() {
                if (nb_buf == 0) return;
                for (int t = 0; t < T_all; ++t) {
                    const auto& tab = (t < n_energy) ? tables[t] : xtab[t - n_energy];
                    for (int b = 0; b < nb_buf; ++b) {
                        const int m = m_first + b;
                        Cbuf[static_cast<size_t>(t) * B + b] =
                            (m < static_cast<int>(tab.size()))
                                ? CoeffCast<double>::apply(tab[m])
                                : 0.0;
                    }
                }
                const double one = 1.0;
                const int Li = static_cast<int>(L), Bi = B;
                dgemm_("N", "N", &Li, &T_all, &nb_buf, &one, Vbuf.data(), &Li,
                       Cbuf.data(), &Bi, &one, acc_all.data(), &Li);
                m_first += nb_buf;
                nb_buf = 0;
            };
            auto push = [&](const DistMultiVector<double>& v) {
                // Per-row copy: DMV rows are stride v.ld, which the 64-byte
                // alignment PADS whenever blk*8 is not a multiple of 64 --
                // a contiguous L-copy scrambles every row by the pad lanes
                // (found as a 17% complete-basis error at blk = 20).
                double* dst = Vbuf.data() + static_cast<size_t>(nb_buf) * L;
                for (int row = 0; row < v.local_rows; ++row) {
                    std::copy(v.row_data(row), v.row_data(row) + blk,
                              dst + static_cast<size_t>(row) * blk);
                }
                if (++nb_buf == B) flush();
            };
            push(*v0);                                  // T_0
            if (degree_all > 1) {
                cheb->apply_op(*v0, *v1);
                push(*v1);                              // T_1
            }
            for (int m = 2; m < degree_all; ++m) {
                cheb->apply_op(*v1, *v2);
                v2->axpby(-1.0, *v0, 2.0);
                push(*v2);                              // T_m
                v0->swap(*v1);
                v1->swap(*v2);
            }
            flush();
        }

        const int lr = v0->local_rows;  // this rank's scalar rows

        // Grid stage: halo the filtered block onto the set's numbering, square
        // it onto the owned boxes, fold into the running estimate. The mean
        // SquareVectorsOntoBoxes takes over this block is reweighted so the
        // final map is the mean over all states (or the exact sum for the
        // complete basis, whose estimator has no 1/n).
        const double weight =
            opt.complete_basis ? static_cast<double>(blk)
                               : static_cast<double>(blk) / static_cast<double>(n_total);
        if (survey) {
            blk_info.emplace_back(start, blk);
        }
        for (int k = 0; k < n_energy; ++k) {
            if (survey) {
                // Store the filtered probes and, once per block, every sketch
                // and image column; the captured map is assembled from them
                // after the sweep, and no probe is ever squared.
                if (!y_store[k]) {
                    y_store[k] =
                        std::make_unique<DistMultiVector<double>>(graph, n_total);
                }
                {
                    const double* col = acc_all.data() +
                                        static_cast<size_t>(k) *
                                            static_cast<size_t>(lr) * blk;
                    for (int row = 0; row < lr; ++row) {
                        std::copy(col + static_cast<size_t>(row) * blk,
                                  col + static_cast<size_t>(row) * blk + blk,
                                  y_store[k]->row_data(row) + start);
                    }
                }
                if (k == 0) {
                    for (size_t t = 0; t < xtab.size(); ++t) {
                        if (!z_store[t]) {
                            z_store[t] = std::make_unique<DistMultiVector<double>>(
                                graph, n_total);
                        }
                        const double* col =
                            acc_all.data() + (static_cast<size_t>(n_energy) + t) *
                                                 static_cast<size_t>(lr) * blk;
                        for (int row = 0; row < lr; ++row) {
                            std::copy(col + static_cast<size_t>(row) * blk,
                                      col + static_cast<size_t>(row) * blk + blk,
                                      z_store[t]->row_data(row) + start);
                        }
                    }
                }
                continue;
            }
            // chi = X_k in the solve form; Zt X_k in the sandwich form, where
            // the metric the solve form carried inside S^-1 is applied once,
            // here, instead of at every step of the recursion.
            DistMultiVector<double> stage(graph, blk);
            {
                const double* col = acc_all.data() +
                                    static_cast<size_t>(k) *
                                        static_cast<size_t>(lr) * blk;
                for (int row = 0; row < lr; ++row) {
                    std::copy(col + static_cast<size_t>(row) * blk,
                              col + static_cast<size_t>(row) * blk + blk,
                              stage.row_data(row));
                }
            }
            DistMultiVector<double>* chi_k = &stage;
            if (form == ChebyOpForm::Sandwich) {
                RSATB_PHASE("rs_ldos/metric_apply");
                precond.apply(stage, *chi);
                chi_k = chi.get();
            }
            {
                RSATB_PHASE("rs_ldos/coeff_halo");
                GatherSetRows(atomic_data, *chi_k, set, plan, comm, &coeffs);
            }
            prof.coeff_halo_doubles += static_cast<int64_t>(coeffs.size());
            {
                RSATB_PHASE("rs_ldos/grid_contract");
                collocation.SquareVectorsOntoBoxes(atomic_data, set, coeffs, blk,
                                                   result.grid, result.domain, &values,
                                                   &prof.grid_work);
            }
            if (frag_cv.enabled()) {
                // Control variate: the same seed through the fragment
                // operator, squared through the same path. X and Y accumulate
                // toward their means; the per-block products feed the
                // regression coefficient.
                if (k == 0) {
                    xi_global.assign(
                        static_cast<size_t>(frag_cv.n_global_rows()) * blk, 0.0);
                    for (size_t ag = 0; ag + 1 < atom_prefix.size(); ++ag) {
                        for (int64_t g = atom_prefix[ag]; g < atom_prefix[ag + 1]; ++g) {
                            double* r = xi_global.data() + static_cast<size_t>(g) * blk;
                            const uint64_t srow = static_cast<uint64_t>(
                                seed_row0_of_atom[ag] + (g - atom_prefix[ag]));
                            for (int v = 0; v < blk; ++v) {
                                r[v] = RademacherSign(opt.seed, srow,
                                                      static_cast<uint64_t>(start + v));
                            }
                        }
                    }
                }
                frag_cv.Apply(k, xi_global, blk, chi0.get());
                GatherSetRows(atomic_data, *chi0, set, plan, comm, &coeffs);
                collocation.SquareVectorsOntoBoxes(atomic_data, set, coeffs, blk,
                                                   result.grid, result.domain, &values0,
                                                   &prof.grid_work);
                double* xk = cv_X[k].data();
                double* yk = cv_Y[k].data();
                double* pxy = cv_PXY[k].data();
                double* pyy = cv_PYY[k].data();
                double sxy = 0.0, syy = 0.0;
                for (int64_t p = 0; p < result.domain.n_point(); ++p) {
                    xk[p] += weight * values[p];
                    yk[p] += weight * values0[p];
                    pxy[p] += values[p] * values0[p];
                    pyy[p] += values0[p] * values0[p];
                    sxy += values[p] * values0[p];
                    syy += values0[p] * values0[p];
                }
                mom_xy[k] += sxy;
                mom_yy[k] += syy;
            } else {
                double* out = result.ldos[k].data();
                for (int64_t p = 0; p < result.domain.n_point(); ++p) {
                    out[p] += weight * values[p];
                }
            }
        }
        if (frag_cv.enabled()) ++cv_blocks;
        ++n_blocks;
    }

    if (frag_cv.enabled()) {
        result.correction_rms.assign(n_energy, 0.0);
        result.control_beta.assign(n_energy, 1.0);
        // Moments aggregate over ranks as well as points: beta must be one
        // number per energy, or the pieces of one map would mix estimators.
        if (comm != MPI_COMM_NULL) {
            MPI_Allreduce(MPI_IN_PLACE, mom_xy.data(), n_energy, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(MPI_IN_PLACE, mom_yy.data(), n_energy, MPI_DOUBLE, MPI_SUM, comm);
        }
        for (int k = 0; k < n_energy; ++k) {
            const double nb = static_cast<double>(cv_blocks);
            // The mean-product subtraction is PER POINT: Cov = sum_p [ sum_b
            // x_b y_b - nb <x>_p <y>_p ]. Subtracting the product of GLOBAL
            // sums instead computes the covariance of the spatially integrated
            // series -- which always co-fluctuates, reads beta ~ 1, and lets a
            // locally useless surrogate import its noise unchecked (measured:
            // the band-edge LDOS at beta = 1 was 4x WORSE than plain while the
            // integrated-series estimate still said 1). cv_X and cv_Y hold the
            // weighted block means, so nb * X * Y is the per-point term.
            double dot_xy = 0.0, dot_yy = 0.0;
            for (int64_t p = 0; p < result.domain.n_point(); ++p) {
                dot_xy += cv_X[k][p] * cv_Y[k][p];
                dot_yy += cv_Y[k][p] * cv_Y[k][p];
            }
            if (comm != MPI_COMM_NULL) {
                double both[2] = {dot_xy, dot_yy};
                MPI_Allreduce(MPI_IN_PLACE, both, 2, MPI_DOUBLE, MPI_SUM, comm);
                dot_xy = both[0];
                dot_yy = both[1];
            }
            const double cov = mom_xy[k] - nb * dot_xy;
            const double var = mom_yy[k] - nb * dot_yy;
            // A vanishing Var(Y) means the surrogate reproduced the estimator
            // sample for sample (the full-fragment limit): beta = 1 by
            // continuity. Otherwise clamp to [0, 1]: below 0 the surrogate
            // anticorrelates (pure noise -- take the plain estimator), above 1
            // the estimate is extrapolating its own sampling error.
            double beta = 1.0;
            if (opt.fit_fragment_coefficient && var > 1e-300 && cv_blocks > 1) {
                beta = std::max(0.0, std::min(1.0, cov / var));
            }
            result.control_beta[k] = beta;

            // The residual diagnostic stays beta-independent: rms of the raw
            // sampled difference X - Y, which is what the samples still had
            // to carry. beta only decides how much of the surrogate's own
            // sampling noise the assembly imports back.
            // Spatially resolved coefficient: the same regression per grid
            // box, shrunk toward the global beta by one global-box-worth of
            // prior evidence. A strict generalization -- homogeneous systems
            // recover the global value box by box -- whose gain appears where
            // surrogate quality varies in space (defects, interfaces): one
            // global beta under-serves every region of a heterogeneous cell.
            // The shrinkage keeps the few-block local estimates from chasing
            // their own noise, and the [0, 1] clamp bounds the damage of any
            // remaining misfit by the plain estimator's own variance.
            // The prior divisor is the GLOBAL box count -- cov and var are
            // global moments, and dividing by the rank-LOCAL box count made
            // the shrinkage strength (and so the assembled map) depend on
            // the rank count.
            const int n_box = result.domain.n_box();
            const double prior_cov = cov / std::max(1, result.grid.n_box());
            const double prior_var = var / std::max(1, result.grid.n_box());
            double sumsq = 0.0;
            double* out = result.ldos[k].data();
            for (int b = 0; b < n_box; ++b) {
                const int64_t p0 = result.domain.offset[b];
                const int64_t p1 = result.domain.offset[b + 1];
                double bxy = 0.0, byy = 0.0, bx_y = 0.0, by_y = 0.0;
                for (int64_t p = p0; p < p1; ++p) {
                    bxy += cv_PXY[k][p];
                    byy += cv_PYY[k][p];
                    bx_y += cv_X[k][p] * cv_Y[k][p];
                    by_y += cv_Y[k][p] * cv_Y[k][p];
                }
                const double cov_b = (bxy - nb * bx_y) + prior_cov;
                const double var_b = (byy - nb * by_y) + prior_var;
                double beta_b = beta;
                if (opt.fit_fragment_coefficient && var_b > 1e-300 && cv_blocks > 1) {
                    beta_b = std::max(0.0, std::min(1.0, cov_b / var_b));
                }
                for (int64_t p = p0; p < p1; ++p) {
                    out[p] += cv_X[k][p] + beta_b * (base[k][p] - cv_Y[k][p]);
                    const double resid = cv_X[k][p] - cv_Y[k][p];
                    sumsq += resid * resid;
                }
            }
            double n_pts = static_cast<double>(result.domain.n_point());
            if (comm != MPI_COMM_NULL) {
                MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_DOUBLE, MPI_SUM, comm);
                MPI_Allreduce(MPI_IN_PLACE, &n_pts, 1, MPI_DOUBLE, MPI_SUM, comm);
            }
            result.correction_rms[k] = std::sqrt(sumsq / std::max(1.0, n_pts));
        }
    }

    if (survey) {
        RSATB_PHASE("rs_ldos/survey");
        // The one grid pipeline every square pass uses: metric-apply (Zt,
        // sandwich form only -- the solve form's states already carry AO
        // coefficients), coefficient halo, per-box square. `values` returns
        // the MEAN of the squared columns on this rank's points.
        auto grid_square = [&](DistMultiVector<double>& x, int width) {
            if (form == ChebyOpForm::Sandwich) {
                DistMultiVector<double> c(graph, width);
                precond.apply(x, c);
                GatherSetRows(atomic_data, c, set, plan, comm, &coeffs);
            } else {
                GatherSetRows(atomic_data, x, set, plan, comm, &coeffs);
            }
            collocation.SquareVectorsOntoBoxes(atomic_data, set, coeffs, width,
                                               result.grid, result.domain,
                                               &values, &prof.grid_work);
        };
        const int64_t npt = result.domain.n_point();
        const int lr = y_store[0]->local_rows;
        const int m_all = J * n_total;

        // The sketch families and their exact images as dense row-major
        // blocks, column (j, i) = j n_total + i on both.
        std::vector<double> Ybig(static_cast<size_t>(lr) * m_all, 0.0);
        std::vector<double> Zbig(static_cast<size_t>(lr) * m_all, 0.0);
        auto fill_big = [&](int k) {
            for (int row = 0; row < lr; ++row) {
                double* yd = Ybig.data() + static_cast<size_t>(row) * m_all;
                double* zd = Zbig.data() + static_cast<size_t>(row) * m_all;
                for (int jj = 0; jj < J; ++jj) {
                    const double* src = z_store[idx_sketch(k, jj)]->row_data(row);
                    std::copy(src, src + n_total,
                              yd + static_cast<size_t>(jj) * n_total);
                }
                for (int jj = 0; jj < J; ++jj) {
                    const double* src = z_store[idx_image(k, jj)]->row_data(row);
                    std::copy(src, src + n_total,
                              zd + static_cast<size_t>(jj) * n_total);
                }
            }
        };

        result.capture.assign(n_energy, 0.0);
        result.sketch_rank.assign(n_energy, 0);
        result.predicted_rel_error.assign(n_energy, 0.0);
        std::vector<double> G_all(static_cast<size_t>(m_all) * m_all);

        // Solve form: the sketch family weighted by the metric, SY = S Ybig,
        // refreshed per energy, so the Gram Y^T (S Y) whitens in the S metric.
        std::vector<double> SYbig;
        std::unique_ptr<DistMultiVector<double>> sy_in, sy_out;
        auto fill_sy = [&]() {
            if (!survey_metric) return;
            SYbig.resize(static_cast<size_t>(lr) * m_all);
            if (!sy_in) {
                sy_in = std::make_unique<DistMultiVector<double>>(graph, m_all);
                sy_out = std::make_unique<DistMultiVector<double>>(graph, m_all);
            }
            for (int row = 0; row < lr; ++row) {
                std::copy(Ybig.data() + static_cast<size_t>(row) * m_all,
                          Ybig.data() + static_cast<size_t>(row + 1) * m_all,
                          sy_in->row_data(row));
            }
            S->mult_dense(*sy_in, *sy_out);
            for (int row = 0; row < lr; ++row) {
                std::copy(sy_out->row_data(row), sy_out->row_data(row) + m_all,
                          SYbig.data() + static_cast<size_t>(row) * m_all);
            }
        };

        for (int k = 0; k < n_energy; ++k) {
            fill_big(k);
            fill_sy();
            {
                const double one = 1.0, zero = 0.0;
                const double* right = survey_metric ? SYbig.data() : Ybig.data();
                dgemm_("N", "T", &m_all, &m_all, &lr, &one, Ybig.data(), &m_all,
                       right, &m_all, &zero, G_all.data(), &m_all);
            }
            if (comm != MPI_COMM_NULL) {
                MPI_Allreduce(MPI_IN_PLACE, G_all.data(), m_all * m_all, MPI_DOUBLE,
                              MPI_SUM, comm);
            }
            if (survey_metric) {
                // Y^T S Y is symmetric exactly; the SpMM rounds it only.
                for (int i2 = 0; i2 < m_all; ++i2) {
                    for (int j2 = 0; j2 < i2; ++j2) {
                        const double v =
                            0.5 * (G_all[static_cast<size_t>(i2) * m_all + j2] +
                                   G_all[static_cast<size_t>(j2) * m_all + i2]);
                        G_all[static_cast<size_t>(i2) * m_all + j2] = v;
                        G_all[static_cast<size_t>(j2) * m_all + i2] = v;
                    }
                }
            }

            // Absolute eigenvalue floor for the rank filter. A relative
            // cutoff alone cannot tell a nearly EMPTY energy window apart
            // from a captured one: with no states inside the window the
            // sketch Gram holds only far-tail responses and sweep roundoff,
            // its spectrum is nearly flat, s_max is itself negligible, and
            // every direction passes the relative test -- then B's
            // 1/sqrt(sv) whitening amplifies the STORED image roundoff into
            // an O(1) garbage map (first seen on Al: rms 7.5 where the
            // truth is 0). The floor is set against the filter's own scale:
            // sum|c| bounds p_max on the scaled interval, so a direction
            // with sv below 1e-10 * (sum|c|)^2 responds at under 1e-5 of
            // the filter peak -- it contributes nothing the plain estimator
            // cannot absorb, while whitening it divides image roundoff by
            // sqrt(sv). Dropping it is free; keeping it is the only risk.
            // Calibration is pinched from both sides: Al's empty-window
            // noise directions sit at ~4e-12 * (sum|c|)^2 (must drop), and
            // silicon's in-gap map is carried by band-edge tails whose sv
            // reaches down toward 1e-8 * (sum|c|)^2 (must keep).
            double tab_l1 = 0.0;
            for (const auto& c : sketch_tables[k]) tab_l1 += std::abs(c);
            const double s_floor = 1e-10 * tab_l1 * tab_l1;

            // The sketch basis: whiten the Gram, keep every direction above
            // the floor. With a flat-top sketch the span is an isotropic
            // random subspace of the window, so this rank IS the survey's
            // sample size k (design_survey_sampling_ldos.md, Section 5).
            // Eigen-decompose a COPY: G_all itself feeds the leave-one-block-
            // out principal submatrices of the normalization below.
            std::vector<double> Gfull(G_all);
            std::vector<double> sv(m_all, 0.0);
            SymEig(m_all, Gfull.data(), sv.data());
            const double s_max = sv.empty() ? 0.0 : sv[m_all - 1];
            int first = m_all;
            for (int i2 = 0; i2 < m_all; ++i2) {
                if (sv[i2] > 1e-12 * s_max && sv[i2] > s_floor) { first = i2; break; }
            }
            const int rk = m_all - first;
            result.sketch_rank[k] = rk;
            double* out = result.ldos[k].data();
            std::fill(out, out + npt, 0.0);
            double captured = 0.0, total = 1.0;   // probe energy, captured / all
            if (rk > 0) {
                std::vector<double> B(static_cast<size_t>(rk) * m_all);
                // Captured part |Pi p phi_r|^2 = sum_j |phi_r^T u_j|^2 with
                // u_j = p(H) Q e_j the exact images of the orthonormal sketch
                // basis Q = Y B: images = Zbig B, one grid square.
                for (int j2 = 0; j2 < rk; ++j2) {
                    const double inv_s = 1.0 / std::sqrt(sv[first + j2]);
                    for (int i2 = 0; i2 < m_all; ++i2) {
                        B[static_cast<size_t>(j2) * m_all + i2] =
                            Gfull[static_cast<size_t>(first + j2) * m_all + i2] * inv_s;
                    }
                }
                std::vector<double> U(static_cast<size_t>(lr) * rk);
                {
                    const double one = 1.0, zero = 0.0;
                    dgemm_("T", "N", &rk, &lr, &m_all, &one, B.data(), &m_all,
                           Zbig.data(), &m_all, &zero, U.data(), &rk);
                }
                DistMultiVector<double> Ud(graph, rk);
                for (int row = 0; row < lr; ++row) {
                    std::copy(U.data() + static_cast<size_t>(row) * rk,
                              U.data() + static_cast<size_t>(row + 1) * rk,
                              Ud.row_data(row));
                }
                grid_square(Ud, rk);
                for (int64_t pnt = 0; pnt < npt; ++pnt) {
                    out[pnt] = values[pnt] * static_cast<double>(rk);
                }
            }
            // The normalization c, measured in STATE space on the probes
            // themselves: c = sum_l ||Pi p xi_l||^2 / sum_l ||p xi_l||^2 with
            // ||Pi y||^2 = ||B^T (Y^T y)||^2 (Q = Y B orthonormal; S-metric in
            // the solve form). Numerator and denominator share the probes,
            // so the ratio carries none of the phase noise a grid integral
            // would on a small window, and it is exactly 1 once the sketch
            // spans the window (then Pi y = y for every probe).
            if (rk > 0) {
                std::vector<double> YtY(static_cast<size_t>(m_all) * n_total, 0.0);
                std::vector<double> ystore(static_cast<size_t>(lr) * n_total);
                for (int row = 0; row < lr; ++row) {
                    std::copy(y_store[k]->row_data(row), y_store[k]->row_data(row) + n_total,
                              ystore.data() + static_cast<size_t>(row) * n_total);
                }
                std::vector<double> systore;
                const double* yl = ystore.data();
                if (survey_metric) {
                    // S y for the metric norms: one SpMM of n_total columns.
                    DistMultiVector<double> sy_in2(graph, n_total), sy_out2(graph, n_total);
                    for (int row = 0; row < lr; ++row) {
                        std::copy(ystore.data() + static_cast<size_t>(row) * n_total,
                                  ystore.data() + static_cast<size_t>(row + 1) * n_total,
                                  sy_in2.row_data(row));
                    }
                    S->mult_dense(sy_in2, sy_out2);
                    systore.resize(ystore.size());
                    for (int row = 0; row < lr; ++row) {
                        std::copy(sy_out2.row_data(row), sy_out2.row_data(row) + n_total,
                                  systore.data() + static_cast<size_t>(row) * n_total);
                    }
                    yl = systore.data();
                }
                {
                    // Y^T (S) y: rows of Ybig (m_all) against columns of y (n_total).
                    const double one = 1.0, zero = 0.0;
                    dgemm_("N", "T", &n_total, &m_all, &lr, &one, yl, &n_total,
                           Ybig.data(), &m_all, &zero, YtY.data(), &n_total);
                }
                if (comm != MPI_COMM_NULL) {
                    MPI_Allreduce(MPI_IN_PLACE, YtY.data(), m_all * n_total, MPI_DOUBLE,
                                  MPI_SUM, comm);
                }
                // CROSS-FITTED: the capture of block b's probes is measured
                // with the sketch of the OTHER blocks (a principal submatrix
                // of the Gram), so no probe is projected onto columns it
                // built -- the in-sample projection over-captures a probe's
                // own state. Under isotropy the left-out capture is
                // rank_{-b} / m_S; rescaled by rank / rank_{-b} it is the
                // full sketch's c = rank / m_S, exactly 1 once the window is
                // spanned (both ranks saturate at m_S).
                std::vector<double> ynorm(n_total, 0.0);
                for (int row = 0; row < lr; ++row) {
                    for (int l = 0; l < n_total; ++l) {
                        const size_t t = static_cast<size_t>(row) * n_total + l;
                        ynorm[l] += ystore[t] * yl[t];
                    }
                }
                if (comm != MPI_COMM_NULL) {
                    MPI_Allreduce(MPI_IN_PLACE, ynorm.data(), n_total, MPI_DOUBLE, MPI_SUM,
                                  comm);
                }
                // The folds of the cross-fit are the sweep blocks when there
                // are at least two of them. A run whose probes all fit in ONE
                // block has nothing to leave out at block granularity: the
                // left-out sketch is empty, every fold would be skipped, and
                // c would come out 0 -- which once silently wrote maps scaled
                // by c(E) with capture reported as 0 (n_samples = block = 16
                // in production). So a single block is split into two halves
                // and cross-fitted between them; a single probe cannot be
                // cross-fitted at all and is refused.
                std::vector<std::pair<int, int>> folds(blk_info);
                if (folds.size() == 1) {
                    const int start = folds[0].first, blk = folds[0].second;
                    if (blk < 2) {
                        throw std::runtime_error(
                            "rs_ldos survey: the capture fraction is cross-fitted between "
                            "probe folds, which needs at least 2 probes (n_samples = " +
                            std::to_string(n_total) + ").");
                    }
                    folds = {{start, blk / 2}, {start + blk / 2, blk - blk / 2}};
                }
                double cap_w = 0.0, tot_w = 0.0;
                int folds_used = 0;
                for (const auto& fold : folds) {
                    const int start = fold.first;
                    const int blk = fold.second;
                    std::vector<int> cols;
                    cols.reserve(m_all - J * blk);
                    for (int jj = 0; jj < J; ++jj) {
                        for (int i2 = 0; i2 < n_total; ++i2) {
                            if (i2 < start || i2 >= start + blk) cols.push_back(jj * n_total + i2);
                        }
                    }
                    const int m = static_cast<int>(cols.size());
                    double blk_tot = 0.0;
                    for (int l = start; l < start + blk; ++l) blk_tot += ynorm[l];
                    tot_w += blk_tot;
                    if (m == 0) continue;
                    std::vector<double> Gs(static_cast<size_t>(m) * m);
                    for (int a2 = 0; a2 < m; ++a2) {
                        for (int b2 = 0; b2 < m; ++b2) {
                            Gs[static_cast<size_t>(a2) * m + b2] =
                                G_all[static_cast<size_t>(cols[a2]) * m_all + cols[b2]];
                        }
                    }
                    std::vector<double> svs(m, 0.0);
                    SymEig(m, Gs.data(), svs.data());
                    const double smax = svs[m - 1];
                    int f2 = m;
                    for (int i2 = 0; i2 < m; ++i2) {
                        if (svs[i2] > 1e-12 * smax && svs[i2] > s_floor) { f2 = i2; break; }
                    }
                    const int rs = m - f2;
                    if (rs == 0) continue;
                    // (B_{-b}^T) (Y_{-b}^T y_l) for l in block b: the rows of YtY at
                    // the left-out columns, whitened by B_{-b}.
                    double blk_cap = 0.0;
                    for (int j2 = 0; j2 < rs; ++j2) {
                        const double inv_s = 1.0 / std::sqrt(svs[f2 + j2]);
                        const double* ev = Gs.data() + static_cast<size_t>(f2 + j2) * m;
                        for (int l = start; l < start + blk; ++l) {
                            double acc = 0.0;
                            for (int a2 = 0; a2 < m; ++a2) {
                                acc += ev[a2] * YtY[static_cast<size_t>(cols[a2]) * n_total + l];
                            }
                            acc *= inv_s;
                            blk_cap += acc * acc;
                        }
                    }
                    // Rescale the left-out capture to the full sketch's rank.
                    cap_w += blk_cap * static_cast<double>(rk) / static_cast<double>(rs);
                    ++folds_used;
                }
                if (folds_used == 0) {
                    // The sketch has rank, so the window holds states, yet no
                    // fold's left-out sketch did: the capture is unmeasured and
                    // the captured map cannot be normalized. Say so rather than
                    // write a map scaled by an unknown c(E).
                    throw std::runtime_error(
                        "rs_ldos survey: the capture fraction could not be cross-fitted at "
                        "energy " + std::to_string(opt.energies[k]) + " eV (sketch rank " +
                        std::to_string(rk) + ", " + std::to_string(folds.size()) +
                        " folds, none with a left-out sketch of rank > 0).");
                }
                captured = std::min(cap_w, tot_w);
                total = tot_w;
            }
            const double c = (total > 1e-300) ? captured / total : 0.0;
            result.capture[k] = c;
            if (rk > 0 && !(c > 1e-300)) {
                throw std::runtime_error(
                    "rs_ldos survey: capture fraction is 0 at energy " +
                    std::to_string(opt.energies[k]) + " eV although the sketch has rank " +
                    std::to_string(rk) + "; refusing to write an unnormalized map.");
            }
            if (c > 1e-300) {
                for (int64_t pnt = 0; pnt < npt; ++pnt) out[pnt] /= c;
            }
            result.predicted_rel_error[k] =
                std::sqrt(2.0 * std::max(0.0, 1.0 - std::min(1.0, c)) / std::max(1, rk));
        }
    }

    // Density engine: predicted relative rms error of the reported map from
    // its own measurements -- the plain scale sqrt(2/n) reduced by the
    // fragment control's fitted coefficient to sqrt(1 - beta^2) (measured:
    // the unregressed X - Y scale under-predicted a beta ~ 0 window by 2x).
    if (!survey && !opt.complete_basis && n_total > 0) {
        result.predicted_rel_error.assign(n_energy, 0.0);
        const bool have_beta = static_cast<int>(result.control_beta.size()) == n_energy;
        for (int k = 0; k < n_energy; ++k) {
            double resid = 1.0;
            if (have_beta) {
                const double beta = std::max(0.0, std::min(1.0, result.control_beta[k]));
                resid = std::sqrt(std::max(0.0, 1.0 - beta * beta));
            }
            result.predicted_rel_error[k] = resid * std::sqrt(2.0 / n_total);
        }
    }

    prof.n_energy = n_energy;
    prof.n_states = n_total;
    prof.block = opt.block;
    prof.filter_degree = degree;
    prof.matvecs = static_cast<int64_t>(degree_all) * n_blocks;
    prof.span_ev = opt.emax - opt.emin;
    prof.eta_ev = opt.eta;
    // The grid stats accumulate over every (block, energy) pass; report the
    // per-pass sizes, which are what the scaling claim is about.
    const int64_t passes = static_cast<int64_t>(n_blocks) * std::max(1, n_energy);
    if (passes > 1) {
        prof.grid_work.boxes_contracted /= passes;
        prof.grid_work.boxes_empty /= passes;
        prof.grid_work.sum_box_orbitals /= passes;
        prof.grid_work.points_kept /= passes;
        prof.grid_work.points_total /= passes;
    }
    prof.coeff_halo_doubles /= std::max<int64_t>(1, n_blocks);
    ReportRealSpaceLdosProfile(prof, comm);
    return result;
}

}  // namespace detail

/// Computes the LDOS map on this rank's share of the grid.
///
/// Args:
///   atomic_data: Distributed geometry; also fixes cell and boundary conditions.
///   H: The Hamiltonian, in the matrix distribution of `atomic_data`'s graph.
///   S: The overlap. Required by (and only by) ChebyOpForm::Solve; pass null
///     for ChebyOpForm::Sandwich, which never references it.
///   Zt: Approximate S^{-1/2} factor (isqrtm). In the solve form its quality
///     sets iteration counts -- the sampler degree and the CG preconditioning
///     -- never the answer. In the sandwich form it defines the operator, so
///     it must be purified (design 2.4).
///   form: Which operator the recursion iterates; see the header note.
///   collocation: Orbital evaluators, one per atom type, same files as H/S.
///   opt: See RealSpaceLdosOptions. energies, eta, emin/emax are required.
///   comm: The communicator H, S and atomic_data live on.
inline RealSpaceLdosResult ComputeRealSpaceLdos(vbcsr::atomic::AtomicData& atomic_data,
                                                vbcsr::BlockSpMat<double>& H,
                                                vbcsr::BlockSpMat<double>* S,
                                                vbcsr::BlockSpMat<double>& Zt,
                                                ChebyOpForm form,
                                                const OrbitalCollocation& collocation,
                                                const RealSpaceLdosOptions& opt,
                                                MPI_Comm comm) {
    if (opt.energies.empty()) {
        throw std::runtime_error("Real-space LDOS needs at least one energy.");
    }
    if (!(opt.eta > 0.0)) {
        throw std::runtime_error("Real-space LDOS needs a positive broadening eta.");
    }
    if (opt.sketch_families < 1) {
        throw std::runtime_error("Real-space LDOS needs sketch_families >= 1.");
    }
    MapEngineOptions eng;
    eng.energies = opt.energies;
    eng.eta = opt.eta;
    eng.emin = opt.emin;
    eng.emax = opt.emax;
    eng.bound_pad = opt.bound_pad;
    eng.spacing = opt.spacing;
    eng.box_edge = opt.box_edge;
    eng.n_samples = opt.n_samples;
    eng.block = opt.block;
    eng.coeff_eps = opt.coeff_eps;
    eng.seed = opt.seed;
    eng.complete_basis = opt.complete_basis;
    eng.certify_sampler = opt.certify_sampler;
    eng.sampler = opt.sampler;
    eng.survey = true;
    eng.sketch_families = opt.sketch_families;
    eng.fragment_buffer = 0.0;
    return detail::FilteredSquareMap(
        atomic_data, H, S, Zt, form, collocation, eng,
        [&opt](double center, double half) {
            return detail::SqrtGaussianTables(opt.energies, opt.eta, center, half,
                                              opt.coeff_eps);
        },
        comm);
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_RS_LDOS_HPP
