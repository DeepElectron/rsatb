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
// Sampling error is the price of the full-cell scope, and it is the honest
// trade: relative error ~ 1/sqrt(n_samples per resolution volume), it
// averages down with samples the user controls, and the estimator is
// positive so the noise can never produce a negative LDOS. For a noise-free
// answer on a small system (validation), complete_basis sweeps the whole
// identity instead of random draws and the estimator becomes exact up to the
// filter truncation and the sampler certificate.

#include "../utils/profile.hpp"

#include "cheby.hpp"
#include "collocation.hpp"
#include "kpm.hpp"
#include "metric_sampler.hpp"
#include "rs_fragment.hpp"
#include "preconditioner.hpp"
#include "real_space_grid.hpp"

#include "vbcsr/core/atomic/inertial_partition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>

namespace rsatb {
namespace calc {

struct RealSpaceLdosOptions {
    std::vector<double> energies;  ///< Evaluation energies, eV, absolute.
    double eta = 0.1;              ///< Gaussian broadening, eV.
    double emin = 0.0;             ///< Lower bound of the (H, S) pencil, eV.
    double emax = 0.0;             ///< Upper bound, eV. Required: emax > emin.
    double bound_pad = 0.01;       ///< Fractional pad on the fitted interval.
    double spacing = 0.4;          ///< Grid spacing, Angstrom.
    int box_edge = 0;              ///< Grid box edge; 0 = physical default.
    int n_samples = 64;            ///< Random states in the estimator.
    int block = 16;                ///< States per Chebyshev sweep (SpMM width).
    /// Fragment control variate (rs_fragment.hpp). `fragment_buffer` > 0
    /// enables it: a deterministic local baseline map plus a stochastically
    /// corrected remainder, cutting the pointwise sampling error by
    /// sqrt(2 x residual fraction) at unchanged sweep cost. The buffer is the
    /// systematic knob (error falls exponentially with it); the core edge
    /// only amortizes the dense fragment solves. Sandwich form; fragments are
    /// geometric objects over the global atom list, so the surrogate -- and
    /// with the counter-based seeding, the whole estimate -- does not depend
    /// on the rank count.
    double fragment_core = 4.5;    ///< Core bin edge, Angstrom.
    double fragment_buffer = 0.0;  ///< Dressing radius, Angstrom; 0 = off.
    /// Subspace deflation, the sharp-filter complement of the fragments.
    /// The filtered sample states ARE a spectrum-slicing sketch of the
    /// energy window (Chebyshev-filtered subspace iteration with the
    /// production filter itself, for zero extra sweeps), and because the
    /// SQUARED filter rides the same recursion (SquaredTable; the
    /// sqrt-Gaussian squares to the Gaussian, so no extra matvecs), the true
    /// filtered image of the sketch basis is an EXACT linear combination of
    /// stored states. That upgrades a regression control to an exact
    /// orthogonal split: with Pi the projector on one half's filtered span,
    ///
    ///   n = sum_j |Phi Zt (p(M) q_j)|^2  +  E |Phi Zt p(M)(I - Pi) xi|^2,
    ///
    /// first term deterministic, second sampled with the OTHER half's seeds
    /// (cross-fitted, so unbiased for any Pi) and row norms that can only
    /// SHRINK -- never worse than plain pointwise, no fitted coefficient
    /// needed. A first attempt that kept only a regression control measured
    /// the trap this construction removes: a partial sketch shares only its
    /// within-sketch interference with the estimator (variance factor
    /// 1 - phi^2 at captured mass phi), while the split also removes the
    /// sketch-to-rest cross term (factor (1 - phi)^2) -- at phi = 1/2 that
    /// is 4x against 1.3x, and at the sparse windows where sharp-eta maps
    /// are actually read (gaps, band edges, defect levels) phi -> 1 and the
    /// chi^2 floor collapses to filter leakage. This is the Hutch++/XTrace
    /// exact-sketch-plus-residual construction [Meyer et al. 2021;
    /// Epperly-Tropp-Webber 2024] carried from traces to point maps, with
    /// Krylov-aware reuse [Chen-Hallman 2023] of the sweep's own states.
    /// Both operator forms: the construction is representation-independent
    /// (design_subspace_deflation.md section 9). In the sandwich form the
    /// seed covariance is I and the split is Euclidean; in the solve form
    /// the seed covariance is S^-1, the sketch is whitened in the S metric
    /// (Gram Y^T S Y), and the deflation contracts Q^T S c0 -- the same
    /// algebra conjugated by S^{1/2}, with identical statistics. Inactive
    /// for the complete-basis sweep and runs of fewer than two sample
    /// blocks. The density task turns it off: the occupied manifold's rank
    /// is extensive, so no O(n_samples) sketch captures it (fragments and
    /// energy windows are the density's levers).
    bool subspace_cv = true;
    /// Sketch-rank multiplier of the deflation: each sample contributes its
    /// filtered state AND its first `subspace_filters - 1` Chebyshev
    /// modulations T_j(M) p_k(M) xi -- together the block-Krylov space of
    /// the filtered states -- so the captured window rank grows J-fold at
    /// ZERO extra matvecs (the modulated tables are coefficient shift-folds
    /// riding the same recursion; costs are J extra accumulator axpys and
    /// J-fold state storage). 1 = the bare filtered states.
    int subspace_filters = 4;

    /// Block-local capture: the through-the-wall mechanism for DENSE energy
    /// windows. Global deflation's capacity J(n-w) is fixed while the
    /// window rank m_eff grows with N; but m_eff's LOCAL rank is intensive
    /// (participation law: ~ m_eff dim_F / N states are distinguishable on
    /// a fragment of dim_F rows), so a capture basis built PER FRAGMENT
    /// from the sketch's local SVD has total capacity sum_F r_F, unbounded
    /// by the sketch width. Sandbox measurement (729-atom Al metal, 15309
    /// orbitals, m_eff = 1772, sketch of 256 columns): plain 16.97%,
    /// global deflation 14.23% (rank wall), block-local r = 40/64/96 per
    /// fragment -> 4.91/0.71/0.08%. The price is information-theoretically
    /// forced: a RESTRICTED basis vector's exact filtered image is not
    /// spanned by the stored tables, so this mode runs dedicated image
    /// sweeps -- local_pack fragment groups share sweep columns (far-apart
    /// fragments, crosstalk decays with distance), giving image cost
    /// local_capture x local_pack column-sweeps, independent of N.
    /// Unbiasedness by sample splitting: the first half of the sample
    /// blocks builds the bases, only the second half is contracted into
    /// the estimate. 0 disables (default); typical dense-window setting
    /// 64-96. Enabling it does NOT commit the cost: the global deflation
    /// runs first for every energy, and only the energies whose measured
    /// capture falls below the routing threshold pay for local bases and
    /// image sweeps -- built from, and priced by, those routed energies
    /// alone. result.local_route says which path served each energy.
    int local_capture = 0;

    /// Region groups sharing image-sweep columns under local_capture.
    /// Column count = local_capture x local_pack x local_pack_replicas.
    /// Regions are compact orbital-weighted RIB subdivisions of each rank's
    /// already-owned atoms; their centers are assigned to a global spatial
    /// coloring so nearby regions preferentially use different columns. The
    /// color count is bounded by the GLOBAL region count, so many ranks with
    /// one region each still spread across local_pack columns instead of
    /// collapsing onto one.
    int local_pack = 8;
    /// Independent random-sign realizations of the packed image columns.
    /// The reported map averages the replicas (packing crosstalk falls as
    /// 1/replicas), and their spread is the packing error certificate:
    /// result.local_packing_se is the between-replica standard error of the
    /// captured part, an internal measurement one realization cannot give.
    /// Image-sweep cost scales linearly. 1 = legacy single realization, no
    /// certificate (local_packing_se stays 0).
    int local_pack_replicas = 2;
    double coeff_eps = 1e-8;       ///< Filter truncation eps_M (design 3.3).
    unsigned long seed = 12345;    ///< RNG seed; per-rank streams derive from it.
    bool complete_basis = false;   ///< Sweep the whole basis instead of
                                   ///< sampling: the estimator becomes exact.
                                   ///< Validation mode -- it holds an O(N_atom)
                                   ///< index on every rank and costs N states.
    bool certify_sampler = true;   ///< Spend one extra propagation pair on the
                                   ///< deterministic sampler bias certificate.
    MetricSamplerOptions sampler;  ///< Options of the S^{-1}-covariance sampler.
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
    int64_t matvecs = 0;          ///< filter_degree * number of blocks.
    double span_ev = 0.0;
    double eta_ev = 0.0;
};

struct RealSpaceLdosResult {
    RealSpaceGrid grid;
    GridDomain domain;                       ///< This rank's boxes.
    std::vector<std::vector<double>> ldos;   ///< [energy][owned point], states/eV/A^3.
    int filter_degree = 0;                   ///< Chebyshev terms per sweep.
    int n_states = 0;                        ///< States actually swept.
    double sampler_residual = -1.0;          ///< ||W q(W)^2 xi - xi||/||xi||, or -1.
    int n_fragments = 0;                     ///< Fragment count; 0 = plain estimator.
    /// Per energy: rms over owned points of the STOCHASTIC CORRECTION -- the
    /// part of the map the samples still carry after the fragment baseline.
    /// Against the map's own rms this is the measured residual fraction, and
    /// (x sqrt(2/n_samples)) the honest pointwise error scale.
    std::vector<double> correction_rms;
    /// Per energy: the regression coefficient of the control variate,
    /// estimated from the samples. 1 means the fragment surrogate tracks the
    /// estimator sample for sample (density-like filters); near 0 it is
    /// uninformative at that energy (sharp filters at band edges, where the
    /// fragment spectrum is boundary-modified) and the estimator falls back
    /// toward the plain mean instead of importing the surrogate's variance.
    std::vector<double> control_beta;
    /// Per energy: the fraction of the integrated map carried by the
    /// deterministic deflated part (0 when inactive; -> 1 as the sketch
    /// captures the window) and the captured rank (min over the two
    /// cross-fitted halves).
    std::vector<double> subspace_capture;
    std::vector<int> subspace_rank;
    /// Per energy: 1 when the block-local capture replaced the global
    /// deflation for this energy, 0 when the global path met the routing
    /// test (or local capture is disabled). The route is decided from the
    /// measured global capture BEFORE any local image sweep is paid.
    std::vector<int> local_route;
    /// Per energy: rms over owned points of the between-replica standard
    /// error of the packed captured part (local_pack_replicas >= 2 and a
    /// locally routed energy; otherwise 0). This is a measured packing-noise
    /// bar; correction_rms remains the residual-sampling scale and neither
    /// substitutes for the other.
    std::vector<double> local_packing_se;
    /// Local-capture geometry of the run (0 / -1 when it never activated):
    /// the global region count, the packing color count actually used, and
    /// the audited minimum gap between same-color region bounding spheres
    /// (minimum-image convention; negative = overlapping spheres shared a
    /// color, the crosstalk signature to watch). The audit is skipped above
    /// 16384 regions to keep setup metadata bounded; the distance then
    /// reports -1 (unaudited), not a claim.
    int local_regions = 0;
    int local_colors = 0;
    double min_same_color_distance = -1.0;
    RealSpaceLdosProfile profile;            ///< Sizes; see RealSpaceLdosProfile.
};

namespace detail {

/// Chebyshev tables of the sqrt-Gaussian filter, one per energy, all of one
/// degree M (design eqs. 3.3-3.5). The physical normalization
/// (2 pi eta^2)^{-1/4} is folded in, so accumulating with these coefficients
/// yields chi in eV^{-1/2} and |chi|^2 in the 1/eV the LDOS needs.
/// Counter-based Rademacher draw: the sign of (seed, global row, state) is a
/// pure function of its key, so every rank draws the identical value and the
/// sampled estimate is invariant to the rank count and the partitioning --
/// the property stateful per-rank streams cannot have. SplitMix64 finalizer
/// chain; same construction class as the counter-based generators of Salmon
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

inline std::vector<std::vector<std::complex<double>>> SqrtGaussianTables(
    const std::vector<double>& energies, double eta, double center, double half,
    double eps) {
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
        for (auto& c : table) c *= norm;
        tables.push_back(std::move(table));
    }
    return tables;
}

/// Chebyshev series of T_j times a tabulated filter, from the product
/// identity T_j T_m = (T_{m+j} + T_{|m-j|}) / 2 -- a shift-fold of the
/// coefficients, degree grows by j only. These are the sketch-enrichment
/// tables of the subspace deflation: {T_j(M) p(M) xi} spans the
/// block-Krylov space of the filtered states, multiplying the capture rank
/// at zero extra matvecs.
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

/// Chebyshev series of the SQUARE of a tabulated filter, from the product
/// identity T_i T_j = (T_{i+j} + T_{|i-j|}) / 2 -- exact at degree 2m - 1
/// (the unconditional cap), then truncated at the tail. The truncated
/// length is GUARANTEED, not hoped for (design section 4): for the
/// Gaussian pair it is sqrt(2(1 - x0^2)) x the center-worst-case base
/// degree -- at most sqrt(2), independent of eta/span/tolerance (all
/// cancel in the ratio), and BELOW the base degree for probes at
/// |x0| > 1/sqrt(2), where physical LCAO energies usually sit; for
/// analyticity-limited filters (sqrt-Fermi) the ratio is ~1 (same
/// Bernstein ellipse bounds both).
inline std::vector<std::complex<double>> SquaredTable(
    const std::vector<std::complex<double>>& t, double eps) {
    const int m = static_cast<int>(t.size());
    std::vector<double> a(m);
    for (int i = 0; i < m; ++i) a[i] = t[i].real();
    std::vector<double> q(2 * m - 1, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            const double v = 0.5 * a[i] * a[j];
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

/// One compact region used by the block-local LDOS projector.
///
/// Atom ownership is already fixed before the LDOS operator is constructed.
/// These regions therefore subdivide only the atoms present on this rank;
/// no atom migration or cross-rank projector reduction is introduced.
struct LocalCaptureRegion {
    std::vector<int> rows;            ///< Rank-local scalar orbital rows.
    std::array<double, 3> center{};    ///< Orbital-weighted Cartesian center.
    int orbital_rows = 0;
    int rib_part = 0;                 ///< Stable leaf number within this rank.
    /// Minimum original global atom id among the member atoms. Atoms are
    /// uniquely owned, so this id is unique across ALL regions of the run and
    /// independent of the rank count -- it keys the packing signs, keeping the
    /// packed estimate invariant under repartitioning (the same property the
    /// counter-based sample seeds have; design_adaptive_local_ldos.md 12.2).
    uint64_t stable_id = 0;
    /// Largest member-atom distance from the center: the bounding-sphere
    /// radius the same-color distance audit measures gaps between.
    double radius = 0.0;
};

/// Divide one rank's already-owned atoms into compact, orbital-balanced
/// regions.  Recursive inertial bisection follows the shape of the final
/// RIB/ParMETIS ownership domain instead of intersecting it with an unrelated
/// global box grid.  `target_rows` controls the approximate scalar-orbital
/// dimension of a region; indivisible atoms and weighted medians make it a
/// target rather than a hard bound.  `global_atom_id` holds each atom's
/// original global id (AtomicData::atom_index), which stamps every region
/// with its rank-independent stable id.
inline std::vector<LocalCaptureRegion> BuildLocalCaptureRegions(
    const std::vector<double>& positions,
    const std::vector<int>& orbital_weight,
    const std::vector<int>& global_atom_id,
    int target_rows) {
    if (positions.size() % 3 != 0 || positions.size() / 3 != orbital_weight.size()) {
        throw std::runtime_error(
            "Local LDOS regions need one Cartesian position and orbital weight per atom.");
    }
    if (global_atom_id.size() != orbital_weight.size()) {
        throw std::runtime_error(
            "Local LDOS regions need one global atom id per atom.");
    }
    if (target_rows < 1) {
        throw std::runtime_error("Local LDOS regions need target_rows >= 1.");
    }
    const int n_atom = static_cast<int>(orbital_weight.size());
    if (n_atom == 0) return {};

    int total_rows = 0;
    std::vector<double> weights(static_cast<size_t>(n_atom), 0.0);
    std::vector<int> row_offset(static_cast<size_t>(n_atom) + 1, 0);
    for (int a = 0; a < n_atom; ++a) {
        if (orbital_weight[a] < 1) {
            throw std::runtime_error(
                "Local LDOS regions need a positive orbital weight per atom.");
        }
        total_rows += orbital_weight[a];
        weights[a] = static_cast<double>(orbital_weight[a]);
        row_offset[a + 1] = total_rows;
    }

    const int n_part = std::max(
        1, std::min(n_atom, static_cast<int>(std::lround(
                                static_cast<double>(total_rows) / target_rows))));
    const auto partition =
        vbcsr::atomic::PartitionByInertia(positions, weights, n_part);

    std::vector<LocalCaptureRegion> by_part(static_cast<size_t>(n_part));
    for (int part = 0; part < n_part; ++part) {
        by_part[part].rib_part = part;
        by_part[part].stable_id = std::numeric_limits<uint64_t>::max();
    }
    for (int a = 0; a < n_atom; ++a) {
        LocalCaptureRegion& region = by_part[partition.owner[a]];
        const int weight = orbital_weight[a];
        region.orbital_rows += weight;
        for (int d = 0; d < 3; ++d) {
            region.center[d] += weight * positions[3 * static_cast<size_t>(a) + d];
        }
        for (int row = row_offset[a]; row < row_offset[a + 1]; ++row) {
            region.rows.push_back(row);
        }
        region.stable_id = std::min(
            region.stable_id, static_cast<uint64_t>(global_atom_id[a]));
    }

    for (auto& region : by_part) {
        if (region.orbital_rows == 0) continue;
        for (double& x : region.center) x /= region.orbital_rows;
    }
    for (int a = 0; a < n_atom; ++a) {
        LocalCaptureRegion& region = by_part[partition.owner[a]];
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double dx =
                positions[3 * static_cast<size_t>(a) + d] - region.center[d];
            d2 += dx * dx;
        }
        region.radius = std::max(region.radius, std::sqrt(d2));
    }

    std::vector<LocalCaptureRegion> regions;
    regions.reserve(static_cast<size_t>(n_part));
    for (auto& region : by_part) {
        if (region.orbital_rows == 0) continue;
        regions.push_back(std::move(region));
    }
    return regions;
}

/// The estimator shared by the LDOS map and the electron density: filter a
/// stochastic (or complete) basis through Chebyshev tables, square onto the
/// grid. One output channel per table; nothing here knows what the filter
/// means. The tables are built by the caller THROUGH the callback because the
/// scaled interval (center, half) they must be fitted on is computed here,
/// from emin/emax/bound_pad.
///
/// opt.energies and opt.eta are NOT read; each wrapper validates and encodes
/// its own filter physics into the tables. Everything else in opt applies.
inline RealSpaceLdosResult FilteredSquareMap(
    vbcsr::atomic::AtomicData& atomic_data,
    vbcsr::BlockSpMat<double>& H,
    vbcsr::BlockSpMat<double>* S,
    vbcsr::BlockSpMat<double>& Zt,
    ChebyOpForm form,
    const OrbitalCollocation& collocation,
    const RealSpaceLdosOptions& opt,
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
        // The basis (geometry + eigensolves) is table-independent; with a
        // cache from the caller it is built once per setup and every task --
        // all LDOS energies, all density windows -- reuses the same
        // decomposition, paying only the per-table GEMMs of TablesToCv.
        const FragmentBasis* fb = nullptr;
        if (fragment_cache != nullptr && fragment_cache->valid &&
            fragment_cache->core_edge == opt.fragment_core &&
            fragment_cache->buffer == opt.fragment_buffer &&
            fragment_cache->solve_chain == chain.active()) {
            fb = &fragment_cache->basis;
        } else if (fragment_cache != nullptr) {
            fragment_cache->basis =
                BuildFragmentBasis(atomic_data, H, Zt, opt.fragment_core,
                                   opt.fragment_buffer, comm, chain);
            fragment_cache->valid = true;
            fragment_cache->core_edge = opt.fragment_core;
            fragment_cache->buffer = opt.fragment_buffer;
            fragment_cache->solve_chain = chain.active();
            fb = &fragment_cache->basis;
        } else {
            local_basis = BuildFragmentBasis(atomic_data, H, Zt, opt.fragment_core,
                                             opt.fragment_buffer, comm, chain);
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
    {
        std::vector<int> mine;
        mine.reserve(2 * static_cast<size_t>(atomic_data.n_atom));
        for (int a = 0; a < atomic_data.n_atom; ++a) {
            mine.push_back(atomic_data.get_global_index(a));
            mine.push_back(local_row_offset[a + 1] - local_row_offset[a]);
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
        const size_t n_atom_global = all.size() / 2;
        std::vector<int64_t> prefix(n_atom_global + 1, 0);
        {
            std::vector<int> norb_of(n_atom_global, 0);
            for (size_t i = 0; i < n_atom_global; ++i) {
                norb_of[static_cast<size_t>(all[2 * i])] = all[2 * i + 1];
            }
            for (size_t g = 0; g < n_atom_global; ++g) prefix[g + 1] = prefix[g] + norb_of[g];
        }
        if (prefix.back() != graph->global_scalar_rows()) {
            throw std::runtime_error(
                "complete_basis enumeration disagrees with the matrix size; the atom global "
                "indices are not the dense 0..N_atom-1 this mode assumes.");
        }
        basis_of_owned.reserve(atomic_data.n_atom);
        for (int a = 0; a < atomic_data.n_atom; ++a) {
            basis_of_owned.emplace_back(prefix[atomic_data.get_global_index(a)], a);
        }
        std::sort(basis_of_owned.begin(), basis_of_owned.end());
        global_row_of_local.assign(local_row_offset.back(), 0);
        n_global_rows = prefix.back();
        for (int a = 0; a < atomic_data.n_atom; ++a) {
            const int64_t g0 = prefix[atomic_data.get_global_index(a)];
            for (int o = 0; o < local_row_offset[a + 1] - local_row_offset[a]; ++o) {
                global_row_of_local[local_row_offset[a] + o] = g0 + o;
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
        // products (ApplyIdentity), and they are spatially local, so the
        // contraction skips every box a column never reaches. What remains
        // per pass is the per-box orbital evaluation, which the wide block
        // amortizes: the width costs only scratch memory here, and narrow
        // passes made the baseline the slowest phase of the run.
        const int base_blk = std::max(opt.block, 512);
        for (int64_t start = 0; start < n_basis; start += base_blk) {
            const int blk = static_cast<int>(std::min<int64_t>(base_blk, n_basis - start));
            if (!chi0 || chi0->num_vectors != blk) {
                chi0 = std::make_unique<DistMultiVector<double>>(graph, blk);
            }
            for (int k = 0; k < n_energy; ++k) {
                frag_cv.ApplyIdentity(k, start, blk, chi0.get());
                GatherSetRows(atomic_data, *chi0, set, plan, comm, &coeffs);
                collocation.SquareVectorsOntoBoxes(atomic_data, set, coeffs, blk,
                                                   result.grid, result.domain, &values,
                                                   &prof.grid_work);
                double* out = base[k].data();
                for (int64_t pnt = 0; pnt < result.domain.n_point(); ++pnt) {
                    out[pnt] += static_cast<double>(blk) * values[pnt];
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

    // Ritz-subspace control (see the option comment): active only where its
    // representation is defined and at least two sample blocks exist to
    // cross-fit. When active, the per-block grids are STORED and the whole
    // assembly (fragment control included) runs jointly after the subspace
    // pass; when inactive, every path below is untouched.
    const bool use_ritz =
        opt.subspace_cv && !opt.complete_basis && n_total >= 2 * opt.block;
    // Solve form: the seed covariance is S^-1, so the deflation whitens in
    // the S metric -- Grams Y^T (S Y), contractions Q^T (S c0) -- and the
    // grid stage consumes the states directly (no Zt). Same construction,
    // conjugated (design section 9.1).
    const bool metric_ritz = use_ritz && form == ChebyOpForm::Solve;
    std::vector<std::unique_ptr<DistMultiVector<double>>> y_store, z_store;
    // Solve-form deflation: the colored seeds c0 = Zt q(W) xi, stored so the
    // assembly can contract Q^T (S c0) per left-out block. The sandwich form
    // regenerates its raw seeds from the counters instead.
    std::unique_ptr<DistMultiVector<double>> c0_store;
    std::vector<std::vector<std::vector<double>>> x_blk, c1_blk;  // [block][energy][pt]
    std::vector<std::pair<int, int>> blk_info;                    // (start, width)
    // The deflation's table family, all riding the same recursion: per
    // energy, the sketch tables T_j p_k (j = 0 is the estimator's own
    // filter) and their exact p_k-images T_j p_k^2. Index layout:
    //   main(k)        = k                                  (j = 0 sketch)
    //   sketch(k, j)   = n_E + k (J-1) + (j-1),  j = 1..J-1
    //   image(k, j)    = n_E J + k J + j,        j = 0..J-1
    const int n_filt = use_ritz ? std::max(1, opt.subspace_filters) : 1;
    std::vector<std::vector<std::complex<double>>> xtab;  // extra tables
    int degree_all = degree;
    auto idx_sketch = [&](int k, int j) { return k * (n_filt - 1) + (j - 1); };
    auto idx_image = [&](int k, int j) {
        return n_energy * (n_filt - 1) + k * n_filt + j;
    };
    if (use_ritz) {
        // Modulation stride: consecutive T_1, T_2, ... are CONSTANTS across
        // a ±5η window (they vary on the spectrum scale, the window is
        // ~10η/h of it), so they enrich nothing where enrichment matters --
        // measured as capture rank pinned at exactly n-w on a 485-state
        // metallic window. T_j and T_{j'} decorrelate over the window once
        // |j-j'| dθ ≳ π with dθ = (10η/h)/sqrt(1-x0²) the window's angular
        // width, so the family is a frequency COMB j = 0, s, 2s, ... with
        // s = π sqrt(1-x0²) h/(10η) ≈ degree/10. Each family is then a
        // genuinely independent shape on the window and capacity truly
        // multiplies by J. Cost: the image tables run the recursion to
        // degree + (J-1)s ≈ degree (1 + J/10) -- capacity grows J-fold for
        // a (1 + J/10)x sweep, the only superlinear knob in the stack
        // (samples and the η-ladder both trade 2x cost for 2x capacity).
        std::vector<int> stride(n_energy, 1);
        for (int k = 0; k < n_energy; ++k) {
            const double x0 =
                std::max(-1.0, std::min(1.0, (opt.energies[k] - center) / half));
            const double win = 10.0 * opt.eta / half;
            stride[k] = std::max(
                1, static_cast<int>(std::lround(
                       M_PI * std::sqrt(std::max(1.0 - x0 * x0, 1e-4)) / win)));
        }
        for (int k = 0; k < n_energy; ++k) {
            for (int j = 1; j < n_filt; ++j) {
                xtab.push_back(ChebyshevTimesTable(j * stride[k], tables[k]));
            }
        }
        for (int k = 0; k < n_energy; ++k) {
            const auto p2 = SquaredTable(tables[k], opt.coeff_eps);
            for (int j = 0; j < n_filt; ++j) {
                xtab.push_back(ChebyshevTimesTable(j * stride[k], p2));
            }
        }
        for (const auto& t : xtab) {
            degree_all = std::max(degree_all, static_cast<int>(t.size()));
        }
        y_store.resize(n_energy);
        z_store.resize(static_cast<size_t>(xtab.size()));
    }

    if (frag_cv.enabled() && !use_ritz) {
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
            // Counter-based draw keyed on (seed, global row, state): the same
            // sign on every rank for a given basis row, so the estimate does
            // not depend on how the atoms were partitioned.
            for (int row = 0; row < xi->local_rows; ++row) {
                double* r = xi->row_data(row);
                const uint64_t grow = static_cast<uint64_t>(global_row_of_local[row]);
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
            if (metric_ritz) {
                if (!c0_store) {
                    c0_store =
                        std::make_unique<DistMultiVector<double>>(graph, n_total);
                }
                for (int row = 0; row < c0->local_rows; ++row) {
                    std::copy(c0->row_data(row), c0->row_data(row) + blk,
                              c0_store->row_data(row) + start);
                }
            }
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
            std::vector<double> dbg;
            if (std::getenv("RSATB_SWEEP_CHECK") != nullptr) dbg.assign(L, 0.0);
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
                if (!dbg.empty()) {
                    const int m = m_first + nb_buf;
                    if (m < static_cast<int>(tables[0].size())) {
                        const double c = CoeffCast<double>::apply(tables[0][m]);
                        for (size_t i2 = 0; i2 < L; ++i2) dbg[i2] += c * dst[i2];
                    }
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
            if (std::getenv("RSATB_SWEEP_CHECK") != nullptr && start == 0) {
                // Redundant single-column re-sweep of table 0, old-style.
                DistMultiVector<double> w0(graph, blk), w1(graph, blk),
                    w2(graph, blk), ref(graph, blk);
                w0.scale(0.0);
                if (opt.complete_basis) {
                    fill_identity(w0, start, blk);
                } else {
                    for (int row = 0; row < w0.local_rows; ++row) {
                        double* r = w0.row_data(row);
                        const uint64_t grow =
                            static_cast<uint64_t>(global_row_of_local[row]);
                        for (int v = 0; v < blk; ++v) {
                            r[v] = RademacherSign(opt.seed, grow,
                                                  static_cast<uint64_t>(v));
                        }
                    }
                }
                ref.copy_from(w0);
                ref.scale(CoeffCast<double>::apply(tables[0][0]));
                if (degree_all > 1) {
                    cheb->apply_op(w0, w1);
                    if (1 < static_cast<int>(tables[0].size())) {
                        ref.axpy(CoeffCast<double>::apply(tables[0][1]), w1);
                    }
                }
                for (int m = 2; m < degree_all; ++m) {
                    cheb->apply_op(w1, w2);
                    w2.axpby(-1.0, w0, 2.0);
                    if (m < static_cast<int>(tables[0].size())) {
                        ref.axpy(CoeffCast<double>::apply(tables[0][m]), w2);
                    }
                    w0.swap(w1);
                    w1.swap(w2);
                }
                double dmax = 0.0, rmax = 0.0;
                const double* col = acc_all.data();
                for (int row = 0; row < v0->local_rows; ++row) {
                    for (int v = 0; v < blk; ++v) {
                        const double a = col[static_cast<size_t>(row) * blk + v];
                        const double b2 = ref.row_data(row)[v];
                        dmax = std::max(dmax, std::abs(a - b2));
                        rmax = std::max(rmax, std::abs(b2));
                    }
                }
                double pmax = 0.0;
                for (int row = 0; row < v0->local_rows; ++row) {
                    for (int v = 0; v < blk; ++v) {
                        pmax = std::max(
                            pmax,
                            std::abs(dbg[static_cast<size_t>(row) * blk + v] -
                                     ref.row_data(row)[v]));
                    }
                }
                int wrow = -1, wcol = -1;
                double wd = 0.0;
                for (int row = 0; row < v0->local_rows; ++row) {
                    for (int v = 0; v < blk; ++v) {
                        const double d2 =
                            std::abs(dbg[static_cast<size_t>(row) * blk + v] -
                                     ref.row_data(row)[v]);
                        if (d2 > wd) { wd = d2; wrow = row; wcol = v; }
                    }
                }
                std::fprintf(stderr,
                             "[sweep-check] max|acc-ref|=%.3e max|push-ref|=%.3e "
                             "ref_max=%.3e worst(row=%d col=%d) dbg=%.6e ref=%.6e\n",
                             dmax, pmax, rmax, wrow, wcol,
                             (wrow >= 0) ? dbg[static_cast<size_t>(wrow) * blk + wcol]
                                         : 0.0,
                             (wrow >= 0) ? ref.row_data(wrow)[wcol] : 0.0);
            }
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
        if (use_ritz) {
            if (frag_cv.enabled()) c1_blk.emplace_back(n_energy);
            blk_info.emplace_back(start, blk);
        }
        for (int k = 0; k < n_energy; ++k) {
            if (use_ritz) {
                // Keep the filtered and squared-filtered coefficient states;
                // the plain contraction is deferred to the deflation pass,
                // which contracts the DEFLATED states instead -- a plain pass
                // here would be thrown away. Only the fragment control's
                // pass stays, its seeds being the raw Rademacher block.
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
                if (frag_cv.enabled()) {
                    if (k == 0) {
                        xi_global.assign(
                            static_cast<size_t>(frag_cv.n_global_rows()) * blk, 0.0);
                        for (int64_t g = 0; g < frag_cv.n_global_rows(); ++g) {
                            double* r = xi_global.data() + static_cast<size_t>(g) * blk;
                            for (int v = 0; v < blk; ++v) {
                                r[v] = RademacherSign(opt.seed, static_cast<uint64_t>(g),
                                                      static_cast<uint64_t>(start + v));
                            }
                        }
                    }
                    frag_cv.Apply(k, xi_global, blk, chi0.get());
                    GatherSetRows(atomic_data, *chi0, set, plan, comm, &coeffs);
                    collocation.SquareVectorsOntoBoxes(atomic_data, set, coeffs, blk,
                                                       result.grid, result.domain,
                                                       &values0, &prof.grid_work);
                    c1_blk.back()[k] = values0;
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
                // regression coefficient. Under the Ritz control everything
                // is stored instead and assembled jointly below.
                if (k == 0) {
                    xi_global.assign(
                        static_cast<size_t>(frag_cv.n_global_rows()) * blk, 0.0);
                    for (int64_t g = 0; g < frag_cv.n_global_rows(); ++g) {
                        double* r = xi_global.data() + static_cast<size_t>(g) * blk;
                        for (int v = 0; v < blk; ++v) {
                            r[v] = RademacherSign(opt.seed, static_cast<uint64_t>(g),
                                                  static_cast<uint64_t>(start + v));
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

    if (frag_cv.enabled() && !use_ritz) {
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
            if (var > 1e-300 && cv_blocks > 1) {
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
            const int n_box = result.domain.n_box();
            const double prior_cov = cov / std::max(1, n_box);
            const double prior_var = var / std::max(1, n_box);
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
                if (var_b > 1e-300 && cv_blocks > 1) {
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

    if (use_ritz) {
        RSATB_PHASE("rs_ldos/subspace_cv");
        // The one grid pipeline every exact-part and remainder pass uses:
        // metric-apply (Zt, sandwich form only -- the solve form's states
        // already carry AO coefficients), coefficient halo, per-box square.
        // `values` returns the MEAN of the squared columns on this rank's
        // points.
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
        const int nb_total = static_cast<int>(blk_info.size());
        const int lr = y_store[0]->local_rows;
        const int J = n_filt;
        const int m_all = J * n_total;

        // LEAVE-ONE-BLOCK-OUT cross-fitting (the XTrace discipline): block
        // b's seeds are deflated against the span of ALL OTHER samples'
        // sketch families, so the capture rank reaches J (n - w) instead of
        // the J n/2 a half-split allows -- the exactness cliff moves 2x at
        // the same sweeps. Unbiasedness is untouched: the left-out basis
        // never saw block b's seeds. The full sketch Gram is built ONCE;
        // each left-out basis is a principal-submatrix eigensolve of it.
        std::vector<double> Ybig(static_cast<size_t>(lr) * m_all, 0.0);
        std::vector<double> Zbig(static_cast<size_t>(lr) * m_all, 0.0);
        // Column (j, i) = j n_total + i; the image family uses the SAME map.
        auto fill_big = [&](int k) {
            for (int row = 0; row < lr; ++row) {
                double* yd = Ybig.data() + static_cast<size_t>(row) * m_all;
                double* zd = Zbig.data() + static_cast<size_t>(row) * m_all;
                std::copy(y_store[k]->row_data(row),
                          y_store[k]->row_data(row) + n_total, yd);
                for (int jj = 1; jj < J; ++jj) {
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

        result.correction_rms.assign(n_energy, 0.0);
        result.control_beta.assign(n_energy, 0.0);
        result.subspace_capture.assign(n_energy, 0.0);
        result.subspace_rank.assign(n_energy, 0);
        result.local_route.assign(n_energy, 0);
        result.local_packing_se.assign(n_energy, 0.0);

        // Block-local capture, shared stage: ONE basis per fragment for the
        // whole scan (per-fragment Gram accumulated over the ROUTED energies)
        // and one image sweep per sign replica whose recursion accumulates
        // every routed energy's squared-filter table -- the image cost is per
        // scan, not per energy. Deferred into a lambda: the global pass below
        // measures every energy's capture FIRST, and only if some energy
        // actually routes local is any of this -- including the sweep, the
        // expensive part -- paid at all, and then only for the tables of the
        // routed energies (design_adaptive_local_ldos.md section 4).
        std::vector<std::vector<double>> lc_U;      // per fragment, fl x r
        std::vector<int> lc_rk, lc_grp;             // rank kept, packing group
        std::vector<std::vector<int>> lc_rows;      // local scalar rows per fragment
        // lc_W[replica][energy]; null where the energy routed global.
        std::vector<std::vector<std::unique_ptr<DistMultiVector<double>>>> lc_W;
        std::vector<std::unique_ptr<DistMultiVector<double>>> lc_X;  // per replica
        int lc_nfrag = 0, lc_G = 1, lc_wimg = 0;
        const int lc_nrep = std::max(1, opt.local_pack_replicas);
        auto local_capture_setup = [&](const std::vector<int>& routed_k) {
            RSATB_PHASE("rs_ldos/local_capture_setup");
            const int nb_half = std::max(1, nb_total / 2);
            std::vector<int> bcols;
            for (int jj = 0; jj < J; ++jj) {
                for (int b = 0; b < nb_half; ++b) {
                    for (int i2 = blk_info[b].first;
                         i2 < blk_info[b].first + blk_info[b].second; ++i2) {
                        bcols.push_back(jj * n_total + i2);
                    }
                }
            }
            const int mb = static_cast<int>(bcols.size());
            const int r_cap = opt.local_capture;
            const int target_rows = std::max(256, 6 * r_cap);

            // Atom ownership is the immutable starting point: RIB followed by
            // the optional ParMETIS refinement has already distributed the
            // operator rows. Subdivide ONLY this rank's owned atoms by a
            // second, serial weighted RIB. This follows the actual ownership
            // region and avoids the small slivers created when unrelated
            // global fractional bins cut through an MPI boundary.
            std::vector<double> owned_positions;
            std::vector<int> owned_orbitals;
            std::vector<int> owned_gids;
            owned_positions.reserve(3 * static_cast<size_t>(atomic_data.n_atom));
            owned_orbitals.reserve(static_cast<size_t>(atomic_data.n_atom));
            owned_gids.reserve(static_cast<size_t>(atomic_data.n_atom));
            for (int a = 0; a < atomic_data.n_atom; ++a) {
                owned_positions.push_back(atomic_data.x[a]);
                owned_positions.push_back(atomic_data.y[a]);
                owned_positions.push_back(atomic_data.z[a]);
                owned_orbitals.push_back(
                    atomic_data.type_norb[atomic_data.atom_type[a]]);
                owned_gids.push_back(atomic_data.atom_index[a]);
            }
            auto local_regions = BuildLocalCaptureRegions(
                owned_positions, owned_orbitals, owned_gids, target_rows);
            std::vector<std::array<double, 3>> frag_center;
            std::vector<uint64_t> frag_sid;
            std::vector<double> frag_rad;
            lc_rows.reserve(local_regions.size());
            frag_center.reserve(local_regions.size());
            frag_sid.reserve(local_regions.size());
            frag_rad.reserve(local_regions.size());
            for (auto& region : local_regions) {
                lc_rows.push_back(std::move(region.rows));
                frag_center.push_back(region.center);
                frag_sid.push_back(region.stable_id);
                frag_rad.push_back(region.radius);
            }
            lc_nfrag = static_cast<int>(lc_rows.size());
            size_t covered_rows = 0;
            for (const auto& rows : lc_rows) covered_rows += rows.size();
            if (covered_rows != static_cast<size_t>(lr)) {
                throw std::runtime_error(
                    "Local LDOS regions do not cover the owned operator rows.");
            }

            // DistMultiVector width is a communicator-wide property; every
            // rank must agree on G. Bound the color count by the GLOBAL
            // region count: the per-rank maximum used before collapsed one
            // region on each of many ranks to a single shared color -- the
            // exact G = 1 failure design_adaptive_local_ldos.md section 2
            // names -- while the global count keeps local_pack colors alive
            // whenever that many regions exist anywhere.
            int n_regions_global = lc_nfrag;
            if (comm != MPI_COMM_NULL) {
                MPI_Allreduce(MPI_IN_PLACE, &n_regions_global, 1, MPI_INT,
                              MPI_SUM, comm);
            }
            lc_G = std::max(
                1, std::min(std::max(1, opt.local_pack), n_regions_global));
            lc_wimg = lc_G * r_cap;
            result.local_regions = n_regions_global;
            result.local_colors = lc_G;
            if (std::getenv("RSATB_DEBUG_SUBSPACE") != nullptr) {
                size_t tot = 0;
                std::vector<char> seen(static_cast<size_t>(lr), 0);
                int dup = 0, oob = 0, min_rows = lr, max_rows = 0;
                for (const auto& rows : lc_rows) {
                    tot += rows.size();
                    min_rows = std::min(min_rows, static_cast<int>(rows.size()));
                    max_rows = std::max(max_rows, static_cast<int>(rows.size()));
                    for (int rr : rows) {
                        if (rr < 0 || rr >= lr) { ++oob; continue; }
                        if (seen[rr]) ++dup;
                        seen[rr] = 1;
                    }
                }
                if (lc_rows.empty()) min_rows = 0;
                std::fprintf(stderr,
                             "[lc] local_rib nfrag=%d global=%d G=%d wimg=%d "
                             "rows=%zu/%d range=%d..%d target=%d dup=%d oob=%d "
                             "n_atom=%d\n",
                             lc_nfrag, n_regions_global, lc_G, lc_wimg, tot, lr,
                             min_rows, max_rows, target_rows, dup, oob,
                             atomic_data.n_atom);
            }

            // Packing still needs a communicator-wide spatial color:
            // a distributed image column contains regions from every rank.
            // Color each local RIB region by the global fractional cell that
            // contains its center. This does NOT define the region; it only
            // keeps nearby regions out of the same packed column when the
            // virtual lattice resolves them.
            int gf[3] = {lc_G, 1, 1};
            for (int g0 = 1; g0 * g0 * g0 <= lc_G; ++g0) {
                if (lc_G % g0 != 0) continue;
                const int rem = lc_G / g0;
                for (int g1 = g0; g1 * g1 <= rem; ++g1) {
                    if (rem % g1 != 0) continue;
                    gf[0] = rem / g1;
                    gf[1] = g1;
                    gf[2] = g0;
                }
            }
            const double virtual_count = std::max(
                1.0, static_cast<double>(n_global_rows) / target_rows);
            const int color_axis = std::max(
                1, static_cast<int>(std::lround(std::cbrt(virtual_count))));
            const auto cinv =
                vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());
            lc_U.resize(lc_nfrag);
            lc_rk.assign(lc_nfrag, 0);
            lc_grp.assign(lc_nfrag, 0);
            for (int f = 0; f < lc_nfrag; ++f) {
                int cell_id[3];
                for (int d = 0; d < 3; ++d) {
                    double frac = frag_center[f][0] * cinv[0 + d] +
                                  frag_center[f][1] * cinv[3 + d] +
                                  frag_center[f][2] * cinv[6 + d];
                    frac -= std::floor(frac);
                    frac = std::round(frac * 1e9) * 1e-9;
                    if (frac >= 1.0) frac -= 1.0;
                    cell_id[d] = std::min(
                        color_axis - 1, static_cast<int>(frac * color_axis));
                }
                lc_grp[f] = (cell_id[0] % gf[0]) +
                            gf[0] * ((cell_id[1] % gf[1]) +
                                     gf[1] * (cell_id[2] % gf[2]));
                lc_grp[f] %= lc_G;
            }

            // Distance AUDIT of the periodic coloring. The virtual-cell color
            // above is fast but is NOT a distance guarantee -- regions across
            // an MPI or cell-fold boundary can be close and still share a
            // color -- so measure what it achieved instead of trusting it:
            // gather the compact region records (5 doubles each; one-time
            // setup metadata, nothing per-application) and report the
            // minimum-image gap between same-color bounding spheres.
            // Negative means overlapping spheres share a packed column --
            // crosstalk the sign replicas then have to average away, and the
            // number a certified-coloring upgrade would fix first. Audited
            // only up to 16384 regions so setup metadata stays bounded.
            if (n_regions_global <= 16384) {
                std::vector<double> rec(static_cast<size_t>(lc_nfrag) * 5);
                for (int f = 0; f < lc_nfrag; ++f) {
                    for (int d = 0; d < 3; ++d) rec[5 * f + d] = frag_center[f][d];
                    rec[5 * f + 3] = frag_rad[f];
                    rec[5 * f + 4] = static_cast<double>(lc_grp[f]);
                }
                std::vector<double> all_rec;
                if (comm != MPI_COMM_NULL) {
                    int n_ranks = 1;
                    MPI_Comm_size(comm, &n_ranks);
                    std::vector<int> counts(n_ranks, 0), displs(n_ranks + 1, 0);
                    const int mine = static_cast<int>(rec.size());
                    MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT,
                                  comm);
                    for (int p = 0; p < n_ranks; ++p) {
                        displs[p + 1] = displs[p] + counts[p];
                    }
                    all_rec.resize(static_cast<size_t>(displs[n_ranks]));
                    MPI_Allgatherv(rec.data(), mine, MPI_DOUBLE, all_rec.data(),
                                   counts.data(), displs.data(), MPI_DOUBLE,
                                   comm);
                } else {
                    all_rec = rec;
                }
                const int n_all = static_cast<int>(all_rec.size() / 5);
                double min_gap = std::numeric_limits<double>::infinity();
                for (int b = 0; b < n_all; ++b) {
                    for (int c = b + 1; c < n_all; ++c) {
                        if (all_rec[5 * b + 4] != all_rec[5 * c + 4]) continue;
                        double dr[3];
                        for (int d = 0; d < 3; ++d) {
                            dr[d] = all_rec[5 * b + d] - all_rec[5 * c + d];
                        }
                        double fr[3];
                        for (int d = 0; d < 3; ++d) {
                            fr[d] = dr[0] * cinv[0 + d] + dr[1] * cinv[3 + d] +
                                    dr[2] * cinv[6 + d];
                            if (atomic_data.pbc[d]) fr[d] -= std::round(fr[d]);
                        }
                        double d2 = 0.0;
                        for (int cdim = 0; cdim < 3; ++cdim) {
                            const double x = fr[0] * atomic_data.cell[0 + cdim] +
                                             fr[1] * atomic_data.cell[3 + cdim] +
                                             fr[2] * atomic_data.cell[6 + cdim];
                            d2 += x * x;
                        }
                        min_gap = std::min(min_gap, std::sqrt(d2) -
                                                        all_rec[5 * b + 3] -
                                                        all_rec[5 * c + 3]);
                    }
                }
                if (min_gap < std::numeric_limits<double>::infinity()) {
                    result.min_same_color_distance = min_gap;
                }
            }
            // Empty-window noise is excluded at the ENERGY level below
            // (trace floor); within the trace-normalized shared Gram a
            // relative eigenvalue cutoff is then sufficient.

            // Per-fragment shared Grams over all energies.
            std::vector<std::vector<double>> Gsh(lc_nfrag);
            for (int f = 0; f < lc_nfrag; ++f) {
                const int fl = static_cast<int>(lc_rows[f].size());
                if (fl > 0) {
                    Gsh[f].assign(static_cast<size_t>(fl) * fl, 0.0);
                }
            }
            std::vector<double> Yf;
            for (const int k : routed_k) {
                fill_big(k);
                // Only the ROUTED energies vote: a global-routed energy never
                // reads this basis, and letting it vote would spend r_cap
                // slots on windows the cheap path already served (the
                // one-difficult-energy-taxes-the-scan failure the review's
                // section 4 names).
                // Trace-balance the energies: each window gets an equal vote
                // in the shared basis, or a dense window's large-norm
                // directions crowd sparse windows out of the r_cap slots
                // (measured on silicon: the conduction energy fell from
                // exact to 16% before this). An energy whose trace sits at
                // the sweep's noise floor (empty window) is EXCLUDED -- its
                // states are roundoff and must not enter the basis.
                double tr_k = 0.0;
                for (int row = 0; row < lr; ++row) {
                    const double* src =
                        Ybig.data() + static_cast<size_t>(row) * m_all;
                    for (int a2 = 0; a2 < mb; ++a2) {
                        tr_k += src[bcols[a2]] * src[bcols[a2]];
                    }
                }
                if (comm != MPI_COMM_NULL) {
                    MPI_Allreduce(MPI_IN_PLACE, &tr_k, 1, MPI_DOUBLE, MPI_SUM,
                                  comm);
                }
                double l1_k = 0.0;
                for (const auto& cc : tables[k]) l1_k += std::abs(cc);
                if (tr_k < 1e-10 * l1_k * l1_k * mb) continue;
                const double scale_k = 1.0 / tr_k;
                for (int f = 0; f < lc_nfrag; ++f) {
                    const auto& rows = lc_rows[f];
                    const int fl = static_cast<int>(rows.size());
                    if (fl <= 0) continue;
                    Yf.assign(static_cast<size_t>(fl) * mb, 0.0);
                    for (int row = 0; row < fl; ++row) {
                        const double* src =
                            Ybig.data() + static_cast<size_t>(rows[row]) * m_all;
                        double* dst = Yf.data() + static_cast<size_t>(row) * mb;
                        for (int a2 = 0; a2 < mb; ++a2) dst[a2] = src[bcols[a2]];
                    }
                    const double one = 1.0;
                    dgemm_("T", "N", &fl, &fl, &mb, &scale_k, Yf.data(), &mb,
                           Yf.data(), &mb, &one, Gsh[f].data(), &fl);
                }
            }
            lc_X.resize(static_cast<size_t>(lc_nrep));
            for (int rep = 0; rep < lc_nrep; ++rep) {
                lc_X[rep] =
                    std::make_unique<DistMultiVector<double>>(graph, lc_wimg);
                lc_X[rep]->scale(0.0);
            }
            for (int f = 0; f < lc_nfrag; ++f) {
                const auto& rows = lc_rows[f];
                const int fl = static_cast<int>(rows.size());
                if (fl <= 0) continue;
                std::vector<double> sv(fl, 0.0);
                SymEig(fl, Gsh[f].data(), sv.data());
                const double smax = sv[fl - 1];
                int r = 0;
                for (int j2 = fl - 1; j2 >= 0 && r < r_cap; --j2) {
                    if (sv[j2] > 1e-12 * smax && sv[j2] > 0.0) ++r;
                    else break;
                }
                lc_rk[f] = r;
                if (r == 0) continue;
                auto& U = lc_U[f];
                U.assign(static_cast<size_t>(fl) * r, 0.0);
                for (int j2 = 0; j2 < r; ++j2) {
                    const double* col =
                        Gsh[f].data() + static_cast<size_t>(fl - r + j2) * fl;
                    for (int row = 0; row < fl; ++row) {
                        U[static_cast<size_t>(row) * r + j2] = col[row];
                    }
                }
                // Random-sign packing: each fragment's basis vector carries a
                // counter-hashed Rademacher sign inside its shared column.
                // Own-fragment products see sigma^2 = 1 and are untouched;
                // cross-fragment packing crosstalk becomes ZERO-MEAN and
                // averages down ~ 1/sqrt(r) instead of accumulating as bias
                // (measured on 729-atom Al: 4.9% -> the sign fix below).
                // The estimator-state deflation uses the UNSIGNED stored U,
                // which is independent of packing, so nothing else changes.
                // The sign key is the region's STABLE id -- its minimum
                // original global atom id, unique run-wide because atoms are
                // uniquely owned -- never the MPI rank, so the packed
                // estimate is invariant under repartitioning. Each replica
                // is an independent sign realization of the same columns.
                const int c0f = lc_grp[f] * r_cap;
                std::vector<double> sgn(static_cast<size_t>(lc_nrep) * r);
                for (int rep = 0; rep < lc_nrep; ++rep) {
                    for (int j2 = 0; j2 < r; ++j2) {
                        sgn[static_cast<size_t>(rep) * r + j2] = RademacherSign(
                            opt.seed ^ 0x5ca1ab1eUL, frag_sid[f],
                            (static_cast<uint64_t>(rep) << 32) |
                                static_cast<uint64_t>(j2));
                    }
                }
                for (int row = 0; row < fl; ++row) {
                    for (int rep = 0; rep < lc_nrep; ++rep) {
                        double* xr = lc_X[rep]->row_data(rows[row]);
                        const double* sg = sgn.data() + static_cast<size_t>(rep) * r;
                        for (int j2 = 0; j2 < r; ++j2) {
                            xr[c0f + j2] +=
                                sg[j2] * U[static_cast<size_t>(row) * r + j2];
                        }
                    }
                }
                Gsh[f].clear();
                Gsh[f].shrink_to_fit();
            }
            // One image sweep per replica; only the ROUTED energies' squared
            // tables ride it (a global-routed energy never reads lc_W).
            std::vector<std::vector<std::complex<double>>> p2tabs(n_energy);
            int dmax = 0;
            for (const int k : routed_k) {
                p2tabs[k] = SquaredTable(tables[k], opt.coeff_eps);
                dmax = std::max(dmax, static_cast<int>(p2tabs[k].size()));
            }
            lc_W.clear();
            lc_W.resize(static_cast<size_t>(lc_nrep));
            {
                RSATB_PHASE("rs_ldos/local_capture_sweep");
                // The image columns must represent the seed COVARIANCE, not
                // the bare squared filter: the split's deterministic
                // complement pairs X against P built from W = C_p X with
                // C_p = E[y y^T]. Sandwich: C_p = p^2(M), sweep X itself.
                // Solve: C_p = p^2(B) S^-1, so the sweep is seeded with
                // S^-1 X -- one CG per image column, the same solve the
                // recursion pays per matvec (design section 9.2).
                DistMultiVector<double> a0(graph, lc_wimg), a1(graph, lc_wimg),
                    a2v(graph, lc_wimg);
                for (int rep = 0; rep < lc_nrep; ++rep) {
                    lc_W[rep].resize(n_energy);
                    if (form == ChebyOpForm::Solve) {
                        a0.scale(0.0);
                        LinearSolver<double> s_solver(1e-8, 1000, false);
                        s_solver.solve(*S, *lc_X[rep], a0);
                    } else {
                        a0.copy_from(*lc_X[rep]);
                    }
                    for (const int k : routed_k) {
                        lc_W[rep][k] = std::make_unique<DistMultiVector<double>>(
                            graph, lc_wimg);
                        lc_W[rep][k]->copy_from(a0);
                        lc_W[rep][k]->scale(
                            CoeffCast<double>::apply(p2tabs[k][0]));
                    }
                    if (dmax > 1) {
                        cheb->apply_op(a0, a1);
                        for (const int k : routed_k) {
                            if (1 < static_cast<int>(p2tabs[k].size())) {
                                lc_W[rep][k]->axpy(
                                    CoeffCast<double>::apply(p2tabs[k][1]), a1);
                            }
                        }
                    }
                    for (int m2 = 2; m2 < dmax; ++m2) {
                        cheb->apply_op(a1, a2v);
                        a2v.axpby(-1.0, a0, 2.0);
                        for (const int k : routed_k) {
                            if (m2 < static_cast<int>(p2tabs[k].size())) {
                                lc_W[rep][k]->axpy(
                                    CoeffCast<double>::apply(p2tabs[k][m2]),
                                    a2v);
                            }
                        }
                        a0.swap(a1);
                        a1.swap(a2v);
                    }
                }
            }
        };
        const bool frag_on = frag_cv.enabled();
        std::vector<double> xbar(npt), c1bar(npt), exact_part(npt);
        std::vector<double> G_all(static_cast<size_t>(m_all) * m_all);
        x_blk.assign(nb_total, {});
        for (int b = 0; b < nb_total; ++b) x_blk[b].assign(n_energy, {});

        // Solve form: the sketch family weighted by the metric, SY = S Ybig,
        // refreshed per energy. It supplies both sides of the S-metric
        // algebra -- the Gram Y^T (SY) and the contraction rows (SY)^T c0 --
        // at one SpMM of m_all columns per energy, noise against the CG the
        // sweep itself paid per matvec.
        std::vector<double> SYbig;
        std::unique_ptr<DistMultiVector<double>> sy_in, sy_out;
        auto fill_sy = [&]() {
            if (!metric_ritz) return;
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
                const double* right = metric_ritz ? SYbig.data() : Ybig.data();
                dgemm_("N", "T", &m_all, &m_all, &lr, &one, Ybig.data(), &m_all,
                       right, &m_all, &zero, G_all.data(), &m_all);
            }
            if (comm != MPI_COMM_NULL) {
                MPI_Allreduce(MPI_IN_PLACE, G_all.data(), m_all * m_all, MPI_DOUBLE,
                              MPI_SUM, comm);
            }
            if (metric_ritz) {
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
            for (const auto& c : tables[k]) tab_l1 += std::abs(c);
            const double s_floor = 1e-10 * tab_l1 * tab_l1;

            std::fill(xbar.begin(), xbar.end(), 0.0);
            std::fill(c1bar.begin(), c1bar.end(), 0.0);
            std::fill(exact_part.begin(), exact_part.end(), 0.0);
            int rank_min = m_all;
            double s11 = 0.0, sx1 = 0.0;

            for (int b = 0; b < nb_total; ++b) {
                const int start = blk_info[b].first;
                const int blk = blk_info[b].second;
                const double wb = static_cast<double>(blk) / n_total;
                // Left-out sketch columns: all (j, i) with sample i outside
                // block b, in ascending column order.
                std::vector<int> cols;
                cols.reserve(m_all - J * blk);
                for (int jj = 0; jj < J; ++jj) {
                    for (int i2 = 0; i2 < n_total; ++i2) {
                        if (i2 < start || i2 >= start + blk) {
                            cols.push_back(jj * n_total + i2);
                        }
                    }
                }
                const int m = static_cast<int>(cols.size());

                std::vector<double> G(static_cast<size_t>(m) * m);
                for (int a2 = 0; a2 < m; ++a2) {
                    for (int b2 = 0; b2 < m; ++b2) {
                        G[static_cast<size_t>(a2) * m + b2] =
                            G_all[static_cast<size_t>(cols[a2]) * m_all + cols[b2]];
                    }
                }
                std::vector<double> sv(m, 0.0);
                SymEig(m, G.data(), sv.data());
                const double s_max = sv.empty() ? 0.0 : sv[m - 1];
                int first = m;
                for (int i2 = 0; i2 < m; ++i2) {
                    if (sv[i2] > 1e-12 * s_max && sv[i2] > s_floor) {
                        first = i2;
                        break;
                    }
                }
                const int r = m - first;
                rank_min = std::min(rank_min, r);

                DistMultiVector<double> yt(graph, blk);
                for (int row = 0; row < lr; ++row) {
                    std::copy(y_store[k]->row_data(row) + start,
                              y_store[k]->row_data(row) + start + blk,
                              yt.row_data(row));
                }
                if (r > 0) {
                    // Left-out basis and its exact filtered image, through
                    // compacted column copies of the big families.
                    std::vector<double> B(static_cast<size_t>(m) * r);
                    for (int j2 = 0; j2 < r; ++j2) {
                        const double inv_s = 1.0 / std::sqrt(sv[first + j2]);
                        for (int i2 = 0; i2 < m; ++i2) {
                            B[static_cast<size_t>(j2) * m + i2] =
                                G[static_cast<size_t>(first + j2) * m + i2] * inv_s;
                        }
                    }
                    // The contraction rows Q: against raw seeds they are the
                    // capture basis itself (sandwich, C = Q^T xi); in the S
                    // metric they carry the weight, Q := (S Y)sub B, so that
                    // C = Q^T c0 = (basis)^T S c0 -- the S-orthogonal
                    // projection coefficients of the colored seed.
                    std::vector<double> Ysub(static_cast<size_t>(lr) * m);
                    std::vector<double> Zsub(static_cast<size_t>(lr) * m);
#pragma omp parallel for
                    for (int row = 0; row < lr; ++row) {
                        const double* ys = metric_ritz
                                               ? SYbig.data() +
                                                     static_cast<size_t>(row) * m_all
                                               : Ybig.data() +
                                                     static_cast<size_t>(row) * m_all;
                        const double* zs = Zbig.data() + static_cast<size_t>(row) * m_all;
                        double* yd = Ysub.data() + static_cast<size_t>(row) * m;
                        double* zd = Zsub.data() + static_cast<size_t>(row) * m;
                        for (int a2 = 0; a2 < m; ++a2) {
                            yd[a2] = ys[cols[a2]];
                            zd[a2] = zs[cols[a2]];
                        }
                    }
                    std::vector<double> Q(static_cast<size_t>(lr) * r);
                    std::vector<double> U(static_cast<size_t>(lr) * r);
                    {
                        const double one = 1.0, zero = 0.0;
                        dgemm_("T", "N", &r, &lr, &m, &one, B.data(), &m, Ysub.data(),
                               &m, &zero, Q.data(), &r);
                        dgemm_("T", "N", &r, &lr, &m, &one, B.data(), &m, Zsub.data(),
                               &m, &zero, U.data(), &r);
                    }

                    // Exact map of this left-out capture, weighted by the
                    // block's sample share.
                    DistMultiVector<double> Ud(graph, r);
                    for (int row = 0; row < lr; ++row) {
                        std::copy(U.data() + static_cast<size_t>(row) * r,
                                  U.data() + static_cast<size_t>(row + 1) * r,
                                  Ud.row_data(row));
                    }
                    grid_square(Ud, r);
                    for (int64_t pnt = 0; pnt < npt; ++pnt) {
                        exact_part[pnt] += wb * values[pnt] * static_cast<double>(r);
                    }

                    // Deflate the block's states against the left-out span.
                    // The seed the coefficients contract: the raw counter
                    // signs (sandwich, covariance I) or the stored colored
                    // seeds (solve, covariance S^-1 -- Q already carries S).
                    DistMultiVector<double> xib(graph, blk);
                    if (metric_ritz) {
                        for (int row = 0; row < lr; ++row) {
                            std::copy(c0_store->row_data(row) + start,
                                      c0_store->row_data(row) + start + blk,
                                      xib.row_data(row));
                        }
                    } else {
                        for (int row = 0; row < lr; ++row) {
                            double* d = xib.row_data(row);
                            const uint64_t grow =
                                static_cast<uint64_t>(global_row_of_local[row]);
                            for (int v = 0; v < blk; ++v) {
                                d[v] = RademacherSign(
                                    opt.seed, grow, static_cast<uint64_t>(start + v));
                            }
                        }
                    }
                    std::vector<double> C(static_cast<size_t>(r) * blk, 0.0);
                    {
                        const double one = 1.0, zero = 0.0;
                        dgemm_("N", "T", &r, &blk, &lr, &one, Q.data(), &r,
                               xib.data.data(), &xib.ld, &zero, C.data(), &r);
                    }
                    if (comm != MPI_COMM_NULL) {
                        MPI_Allreduce(MPI_IN_PLACE, C.data(), r * blk, MPI_DOUBLE,
                                      MPI_SUM, comm);
                    }
                    {
                        const double neg = -1.0, one = 1.0;
                        dgemm_("T", "N", &blk, &lr, &r, &neg, C.data(), &r, U.data(),
                               &r, &one, yt.data.data(), &yt.ld);
                    }
                }
                grid_square(yt, blk);
                x_blk[b][k] = values;
            }

            // Assembly: per-block exact parts already accumulated; deflated
            // mean plus the fragment control regressed against it.
            result.subspace_rank[k] = rank_min;
            for (int b = 0; b < nb_total; ++b) {
                const double wb = static_cast<double>(blk_info[b].second) / n_total;
                const double* xb = x_blk[b][k].data();
                const double* c1 = frag_on ? c1_blk[b][k].data() : nullptr;
                for (int64_t pnt = 0; pnt < npt; ++pnt) {
                    xbar[pnt] += wb * xb[pnt];
                    if (frag_on) {
                        const double v1 = c1[pnt] - base[k][pnt];
                        c1bar[pnt] += wb * v1;
                        s11 += v1 * v1;
                        sx1 += v1 * xb[pnt];
                    }
                }
            }
            double beta = 0.0;
            if (frag_on && nb_total > 1) {
                double d11 = 0.0, dx1 = 0.0;
                for (int64_t pnt = 0; pnt < npt; ++pnt) {
                    d11 += c1bar[pnt] * c1bar[pnt];
                    dx1 += c1bar[pnt] * xbar[pnt];
                }
                double acc[4] = {s11, sx1, d11, dx1};
                if (comm != MPI_COMM_NULL) {
                    MPI_Allreduce(MPI_IN_PLACE, acc, 4, MPI_DOUBLE, MPI_SUM, comm);
                }
                const double nb = static_cast<double>(nb_total);
                const double var = acc[0] - nb * acc[2];
                const double cov = acc[1] - nb * acc[3];
                if (var > 1e-300) beta = std::max(0.0, std::min(1.0, cov / var));
            }
            result.control_beta[k] = beta;
            double* out = result.ldos[k].data();
            double sumsq = 0.0, sum_exact = 0.0, sum_perp = 0.0;
            for (int64_t pnt = 0; pnt < npt; ++pnt) {
                out[pnt] = exact_part[pnt] + xbar[pnt] - beta * c1bar[pnt];
                sumsq += xbar[pnt] * xbar[pnt];
                sum_exact += exact_part[pnt];
                sum_perp += xbar[pnt];
            }
            double n_pts = static_cast<double>(npt);
            double acc[4] = {sumsq, sum_exact, sum_perp, n_pts};
            if (comm != MPI_COMM_NULL) {
                MPI_Allreduce(MPI_IN_PLACE, acc, 4, MPI_DOUBLE, MPI_SUM, comm);
            }
            result.correction_rms[k] = std::sqrt(acc[0] / std::max(1.0, acc[3]));
            const double tot = acc[1] + acc[2];
            result.subspace_capture[k] = (std::abs(tot) > 1e-300) ? acc[1] / tot : 0.0;
            // Route by OUTCOME: the global stored-image deflation above is
            // ~free and exact wherever the window under-fills the sketch; a
            // measured capture below 0.9 means the window is dense beyond
            // the global rank wall -- only then is the local-capture pass
            // worth paying for, and its result replaces the global one for
            // this energy. Rank heuristics on the sketch Gram were tried and
            // are fragile (teeth collinearity thins the spectrum: a dense
            // metallic window measured r_g = 196/256 and misrouted). Only
            // the route is DECIDED here; the local pass itself -- including
            // its image sweeps, the expensive part -- runs after this loop,
            // and only if some energy actually routed local.
            if (opt.local_capture > 0 && result.subspace_capture[k] < 0.9) {
                result.local_route[k] = 1;
            }
        }

        // Deferred block-local pass: setup (basis + image sweeps, priced by
        // the routed energies alone) and the per-energy stages.
        std::vector<int> routed_k;
        for (int k = 0; k < n_energy; ++k) {
            if (result.local_route[k] != 0) routed_k.push_back(k);
        }
        if (opt.local_capture > 0 && !routed_k.empty()) {
            local_capture_setup(routed_k);
            for (const int k : routed_k) {
                // ---- Block-local capture, per-energy stage: the shared
                // basis and this energy's image columns come from the setup
                // pass above; here only small local algebra, per-replica
                // grid squares (polarization product), and the deflated
                // second-half contraction remain.
                RSATB_PHASE("rs_ldos/local_capture");
                const int nb_half = std::max(1, nb_total / 2);
                const int r_cap = opt.local_capture;
                // P = 2W - Pi W with the FULL block-diagonal projector:
                // every fragment deflates its own row-block of EVERY image
                // column. The within-fragment-only version misses the
                // cross-fragment blocks of Pi C Pi -- exactly zero for the
                // ORBITAL diagonal (disjoint rows), but the grid collocation
                // smears across fragment boundaries through orbital tails
                // and Zt's range, and in 3D the boundary shell is a large
                // volume fraction (measured: a 21% integral-preserving
                // shape error on Al before this correction).
                // Per sign replica: its own captured map from its own image
                // columns. `exact_part` carries the running replica MEAN (the
                // reported captured part; crosstalk averages down 1/L) and
                // `ex_m2` the Welford sum of squares, whose between-replica
                // spread is the packing error certificate no single
                // realization can provide.
                std::fill(exact_part.begin(), exact_part.end(), 0.0);
                std::vector<double> ex_m2(static_cast<size_t>(npt), 0.0);
                std::vector<double> ex_rep(static_cast<size_t>(npt));
                for (int rep = 0; rep < lc_nrep; ++rep) {
                    DistMultiVector<double> Pv(graph, lc_wimg);
                    Pv.copy_from(*lc_W[rep][k]);
                    Pv.scale(2.0);
                    for (int f = 0; f < lc_nfrag; ++f) {
                        const int r = lc_rk[f];
                        if (r == 0) continue;
                        const auto& rows = lc_rows[f];
                        const int fl = static_cast<int>(rows.size());
                        std::vector<double> Wf(static_cast<size_t>(fl) * lc_wimg);
                        for (int row = 0; row < fl; ++row) {
                            std::copy(
                                lc_W[rep][k]->row_data(rows[row]),
                                lc_W[rep][k]->row_data(rows[row]) + lc_wimg,
                                Wf.data() + static_cast<size_t>(row) * lc_wimg);
                        }
                        std::vector<double> S(static_cast<size_t>(r) * lc_wimg);
                        {
                            const double one = 1.0, zero = 0.0;
                            dgemm_("N", "T", &lc_wimg, &r, &fl, &one, Wf.data(),
                                   &lc_wimg, lc_U[f].data(), &r, &zero, S.data(),
                                   &lc_wimg);
                            dgemm_("N", "N", &lc_wimg, &fl, &r, &one, S.data(),
                                   &lc_wimg, lc_U[f].data(), &r, &zero, Wf.data(),
                                   &lc_wimg);
                        }
                        for (int row = 0; row < fl; ++row) {
                            double* pr = Pv.row_data(rows[row]);
                            const double* us =
                                Wf.data() + static_cast<size_t>(row) * lc_wimg;
                            for (int j2 = 0; j2 < lc_wimg; ++j2) pr[j2] -= us[j2];
                        }
                    }
                    std::fill(ex_rep.begin(), ex_rep.end(), 0.0);
                    {
                        DistMultiVector<double> sumv(graph, lc_wimg);
                        sumv.copy_from(*lc_X[rep]);
                        sumv.axpy(1.0, Pv);
                        grid_square(sumv, lc_wimg);
                        for (int64_t pnt = 0; pnt < npt; ++pnt) {
                            ex_rep[pnt] += 0.25 * lc_wimg * values[pnt];
                        }
                        sumv.copy_from(*lc_X[rep]);
                        sumv.axpy(-1.0, Pv);
                        grid_square(sumv, lc_wimg);
                        for (int64_t pnt = 0; pnt < npt; ++pnt) {
                            ex_rep[pnt] -= 0.25 * lc_wimg * values[pnt];
                        }
                    }
                    for (int64_t pnt = 0; pnt < npt; ++pnt) {
                        const double delta = ex_rep[pnt] - exact_part[pnt];
                        exact_part[pnt] += delta / (rep + 1);
                        ex_m2[pnt] += delta * (ex_rep[pnt] - exact_part[pnt]);
                    }
                }

                std::fill(xbar.begin(), xbar.end(), 0.0);
                double n_est = 0.0;
                for (int b = nb_half; b < nb_total; ++b) {
                    n_est += static_cast<double>(blk_info[b].second);
                }
                for (int b = nb_half; b < nb_total; ++b) {
                    const int start = blk_info[b].first;
                    const int blk = blk_info[b].second;
                    DistMultiVector<double> yt(graph, blk);
                    for (int row = 0; row < lr; ++row) {
                        std::copy(y_store[k]->row_data(row) + start,
                                  y_store[k]->row_data(row) + start + blk,
                                  yt.row_data(row));
                    }
                    for (int f = 0; f < lc_nfrag; ++f) {
                        const int r = lc_rk[f];
                        if (r == 0) continue;
                        const auto& rows = lc_rows[f];
                        const int fl = static_cast<int>(rows.size());
                        std::vector<double> Yb(static_cast<size_t>(fl) * blk);
                        for (int row = 0; row < fl; ++row) {
                            std::copy(yt.row_data(rows[row]),
                                      yt.row_data(rows[row]) + blk,
                                      Yb.data() + static_cast<size_t>(row) * blk);
                        }
                        std::vector<double> Cf(static_cast<size_t>(r) * blk);
                        {
                            const double one = 1.0, zero = 0.0, neg = -1.0;
                            dgemm_("N", "T", &blk, &r, &fl, &one, Yb.data(),
                                   &blk, lc_U[f].data(), &r, &zero, Cf.data(),
                                   &blk);
                            dgemm_("N", "N", &blk, &fl, &r, &neg, Cf.data(),
                                   &blk, lc_U[f].data(), &r, &one, Yb.data(),
                                   &blk);
                        }
                        for (int row = 0; row < fl; ++row) {
                            std::copy(
                                Yb.data() + static_cast<size_t>(row) * blk,
                                Yb.data() + static_cast<size_t>(row) * blk + blk,
                                yt.row_data(rows[row]));
                        }
                    }
                    grid_square(yt, blk);
                    const double wb = static_cast<double>(blk) / n_est;
                    for (int64_t pnt = 0; pnt < npt; ++pnt) {
                        xbar[pnt] += wb * values[pnt];
                    }
                }
                double* out = result.ldos[k].data();
                double sums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
                int rank_min_loc = r_cap;
                for (int f = 0; f < lc_nfrag; ++f) {
                    if (!lc_rows[f].empty()) {
                        rank_min_loc = std::min(rank_min_loc, lc_rk[f]);
                    }
                }
                for (int64_t pnt = 0; pnt < npt; ++pnt) {
                    out[pnt] = exact_part[pnt] + xbar[pnt];
                    sums[0] += xbar[pnt] * xbar[pnt];
                    sums[1] += exact_part[pnt];
                    sums[2] += exact_part[pnt] + xbar[pnt];
                    sums[3] += 1.0;
                    if (lc_nrep >= 2) {
                        // se^2 of the replica mean at this point:
                        // sample variance / L.
                        sums[4] += ex_m2[pnt] /
                                   (static_cast<double>(lc_nrep - 1) * lc_nrep);
                    }
                }
                if (comm != MPI_COMM_NULL) {
                    MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_DOUBLE, MPI_SUM,
                                  comm);
                    MPI_Allreduce(MPI_IN_PLACE, &rank_min_loc, 1, MPI_INT,
                                  MPI_MIN, comm);
                }
                result.correction_rms[k] =
                    std::sqrt(sums[0] / std::max(1.0, sums[3]));
                result.local_packing_se[k] =
                    (lc_nrep >= 2) ? std::sqrt(sums[4] / std::max(1.0, sums[3]))
                                   : 0.0;
                result.control_beta[k] = 0.0;
                result.subspace_capture[k] =
                    (std::abs(sums[2]) > 1e-300) ? sums[1] / sums[2] : 0.0;
                result.subspace_rank[k] = rank_min_loc;
            }
        }
    }

    prof.n_energy = n_energy;
    prof.n_states = n_total;
    prof.block = opt.block;
    prof.filter_degree = degree;
    prof.matvecs = static_cast<int64_t>(degree) * n_blocks;
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
                                                MPI_Comm comm,
                                                detail::FragmentBasisCache* fragment_cache =
                                                    nullptr) {
    if (opt.energies.empty()) {
        throw std::runtime_error("Real-space LDOS needs at least one energy.");
    }
    if (!(opt.eta > 0.0)) {
        throw std::runtime_error("Real-space LDOS needs a positive broadening eta.");
    }
    return detail::FilteredSquareMap(
        atomic_data, H, S, Zt, form, collocation, opt,
        [&opt](double center, double half) {
            return detail::SqrtGaussianTables(opt.energies, opt.eta, center, half,
                                              opt.coeff_eps);
        },
        comm, fragment_cache);
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_RS_LDOS_HPP
