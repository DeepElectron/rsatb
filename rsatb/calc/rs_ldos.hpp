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
#include "preconditioner.hpp"
#include "real_space_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    RealSpaceLdosProfile profile;            ///< Sizes; see RealSpaceLdosProfile.
};

namespace detail {

/// Chebyshev tables of the sqrt-Gaussian filter, one per energy, all of one
/// degree M (design eqs. 3.3-3.5). The physical normalization
/// (2 pi eta^2)^{-1/4} is folded in, so accumulating with these coefficients
/// yields chi in eV^{-1/2} and |chi|^2 in the 1/eV the LDOS needs.
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
    using vbcsr::DistGraph;
    using vbcsr::DistMultiVector;
    using vbcsr::DistVector;

    if (opt.energies.empty()) {
        throw std::runtime_error("Real-space LDOS needs at least one energy.");
    }
    if (!(opt.eta > 0.0)) {
        throw std::runtime_error("Real-space LDOS needs a positive broadening eta.");
    }
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
    const auto tables =
        detail::SqrtGaussianTables(opt.energies, opt.eta, center, half, opt.coeff_eps);
    const int degree = static_cast<int>(tables.front().size());
    result.filter_degree = degree;

    ISQRTMPreconditioner<double> precond(Zt, kBorrowFactor);

    // The sandwich form needs neither the sampler (its seed covariance is
    // already the identity) nor S inside the operator.
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

    DistGraph* graph = H.graph;
    const int n_energy = static_cast<int>(opt.energies.size());
    const int n_total = opt.complete_basis ? graph->global_scalar_rows() : opt.n_samples;
    result.n_states = n_total;
    result.ldos.assign(n_energy, std::vector<double>(result.domain.n_point(), 0.0));

    int64_t n_blocks = 0;
    std::mt19937_64 gen(opt.seed + 977 * static_cast<unsigned long>(rank));
    std::bernoulli_distribution coin(0.5);

    // The complete-basis sweep enumerates the basis gid-major: basis vector s
    // lives on the atom g whose orbital range covers s. The enumeration is
    // this function's own, not the library's row ordering, so each rank only
    // needs the offsets of the atoms it OWNS in it -- derived from one
    // allgather of (gid, norb). O(N_atom) on every rank, which is exactly why
    // this mode is documented as validation-only.
    std::vector<std::pair<int64_t, int>> basis_of_owned;  // (offset, local atom)
    std::vector<int> local_row_offset(atomic_data.n_atom + 1, 0);
    for (int a = 0; a < atomic_data.n_atom; ++a) {
        local_row_offset[a + 1] =
            local_row_offset[a] + atomic_data.type_norb[atomic_data.atom_type[a]];
    }
    if (opt.complete_basis) {
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
    }

    if (opt.certify_sampler && sampler) {
        DistVector<double> xi(graph);
        for (int i = 0; i < xi.local_size; ++i) xi[i] = coin(gen) ? 1.0 : -1.0;
        result.sampler_residual = sampler->Residual(xi);
    }

    std::unique_ptr<DistMultiVector<double>> xi, c0, v0, v1, v2, chi;
    std::vector<std::unique_ptr<DistMultiVector<double>>> accum(n_energy);
    std::vector<double> coeffs, values;

    for (int start = 0; start < n_total; start += opt.block) {
        const int blk = std::min(opt.block, n_total - start);
        auto need = [&](std::unique_ptr<DistMultiVector<double>>& mv) {
            if (!mv || mv->num_vectors != blk) {
                mv = std::make_unique<DistMultiVector<double>>(graph, blk);
            }
        };
        need(xi); need(v0); need(v1); need(v2);
        if (sampler) need(c0); else need(chi);
        for (auto& a : accum) need(a);

        xi->scale(0.0);
        if (opt.complete_basis) {
            // Columns start .. start+blk of the identity, each set by the one
            // rank owning the atom the basis vector lives on.
            for (int v = 0; v < blk; ++v) {
                const int64_t s = start + v;
                auto it = std::upper_bound(
                    basis_of_owned.begin(), basis_of_owned.end(),
                    std::make_pair(s, std::numeric_limits<int>::max()));
                if (it == basis_of_owned.begin()) continue;
                const int64_t off = (it - 1)->first;
                const int a = (it - 1)->second;
                const int norb = local_row_offset[a + 1] - local_row_offset[a];
                if (s < off + norb) {
                    (*xi)(local_row_offset[a] + static_cast<int>(s - off), v) = 1.0;
                }
            }
        } else {
            for (int row = 0; row < xi->local_rows; ++row) {
                double* r = xi->row_data(row);
                for (int v = 0; v < blk; ++v) r[v] = coin(gen) ? 1.0 : -1.0;
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
            for (int k = 0; k < n_energy; ++k) {
                accum[k]->copy_from(*v0);
                accum[k]->scale(CoeffCast<double>::apply(tables[k][0]));
            }
            if (degree > 1) {
                cheb->apply_op(*v0, *v1);
                for (int k = 0; k < n_energy; ++k) {
                    accum[k]->axpy(CoeffCast<double>::apply(tables[k][1]), *v1);
                }
            }
            for (int m = 2; m < degree; ++m) {
                cheb->apply_op(*v1, *v2);
                v2->axpby(-1.0, *v0, 2.0);
                for (int k = 0; k < n_energy; ++k) {
                    accum[k]->axpy(CoeffCast<double>::apply(tables[k][m]), *v2);
                }
                v0->swap(*v1);
                v1->swap(*v2);
            }
        }

        // Grid stage: halo the filtered block onto the set's numbering, square
        // it onto the owned boxes, fold into the running estimate. The mean
        // SquareVectorsOntoBoxes takes over this block is reweighted so the
        // final map is the mean over all states (or the exact sum for the
        // complete basis, whose estimator has no 1/n).
        const double weight =
            opt.complete_basis ? static_cast<double>(blk)
                               : static_cast<double>(blk) / static_cast<double>(n_total);
        for (int k = 0; k < n_energy; ++k) {
            // chi = X_k in the solve form; Zt X_k in the sandwich form, where
            // the metric the solve form carried inside S^-1 is applied once,
            // here, instead of at every step of the recursion.
            DistMultiVector<double>* chi_k = accum[k].get();
            if (form == ChebyOpForm::Sandwich) {
                RSATB_PHASE("rs_ldos/metric_apply");
                precond.apply(*accum[k], *chi);
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
            double* out = result.ldos[k].data();
            for (int64_t p = 0; p < result.domain.n_point(); ++p) {
                out[p] += weight * values[p];
            }
        }
        ++n_blocks;
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

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_RS_LDOS_HPP
