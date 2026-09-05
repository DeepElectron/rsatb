// The full-cell LDOS map against dense diagonalization.
//
// The fixture is a carbon chain whose overlap comes from the op2c two-center
// integrals (the real basis, SPD by construction) and whose H is a seeded
// symmetric matrix on the S pattern -- the physics is arbitrary, which is
// the point: the estimator must reproduce the dense answer for ANY pencil,
// and a synthetic H cannot hide an error behind physical smoothness.
//
// Two claims, tested separately:
//
//   1. EXACTNESS. In complete_basis mode the "sampler" sweeps the whole
//      identity, so the estimator's expectation is computed with no sampling
//      noise. The result must match, pointwise, the map obtained from the
//      other side of the algebra: dense dsygv eigenpairs assembled into
//      A(E) = sum_n delta_eta(E - eps_n) c_n c_n^T and contracted through
//      route (5.3a) ContractOntoBoxes. The two routes share no code between
//      the pencil and the grid values -- recursion vs diagonalization,
//      square-on-grid vs assemble-then-contract -- so agreement pins the
//      filter table, the operator, the sampler covariance, the halo plan and
//      both contraction kernels at once.
//
//   2. SAMPLING. With random states the map must converge toward the same
//      reference at the statistical rate, and stay non-negative exactly --
//      positivity is structural (a mean of squares), not a kernel property.

#include "../rs_ldos.hpp"
#include "../isqrtm.hpp"
#include "../../model/lcao/op2c_engine.hpp"
#include "../../test_data_path.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace vbcsr;

extern "C" {
void dsygv_(const int* itype, const char* jobz, const char* uplo, const int* n,
            double* a, const int* lda, double* b, const int* ldb, double* w,
            double* work, const int* lwork, int* info);
}

namespace {

std::string DiamondDir() { return rsatb::test::data_dir() + "diamond/"; }

// A symmetric scalar-indexed generator: h(p, q) == h(q, p) by construction,
// so the assembled H is symmetric whatever the block layout does.
double SyntheticEntry(int p, int q) {
    const double s = static_cast<double>(p + q);
    const double m = static_cast<double>(p) * static_cast<double>(q);
    double v = 0.6 * std::sin(0.37 * s) * std::cos(0.11 * std::fmod(m, 97.0));
    if (p == q) v += 3.0 + 0.02 * p;  // spread the diagonal; breaks degeneracy
    return v;
}

struct ChainSystem {
    static constexpr int kAtoms = 12;
    static constexpr int kNorb = 13;

    std::unique_ptr<vbcsr::atomic::AtomicData> ad;
    std::unique_ptr<rsatb::calc::OrbitalCollocation> collocation;
    std::unique_ptr<vbcsr::atomic::ImageContainer<double>> img;
    std::unique_ptr<BlockSpMat<std::complex<double>>> S_gamma;
    std::unique_ptr<BlockSpMat<double>> S, H, Zt;
    double rcut = 0.0;

    explicit ChainSystem(int n_atoms = kAtoms) {
        collocation = std::make_unique<rsatb::calc::OrbitalCollocation>(
            std::vector<std::string>{DiamondDir() + "C_gga_7au_100Ry_2s2p1d.orb"},
            MPI_COMM_WORLD);
        rcut = collocation->rcut_angstrom(0);

        std::vector<double> pos;
        for (int i = 0; i < n_atoms; ++i) pos.insert(pos.end(), {2.0 * i, 0.0, 0.0});
        const std::vector<int> z(n_atoms, 6);
        const double lx = 2.0 * n_atoms + 4.0 * rcut, lt = 4.0 * rcut + 6.0;
        const std::vector<double> cell = {lx, 0, 0, 0, lt, 0, 0, 0, lt};
        ad.reset(vbcsr::atomic::AtomicData::from_points(
            pos, z, cell, {false, false, false}, {rcut}, {kNorb}, MPI_COMM_WORLD));

        rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
        op.init(1, 1, false, DiamondDir(), {"C_gga_7au_100Ry_2s2p1d.orb"}, "", {},
                "rs_ldos_opconst.log");
        img = op.build_overlap(*ad);
        S_gamma.reset(img->sample_k({0.0, 0.0, 0.0},
                                    vbcsr::atomic::PhaseConvention::R_AND_POSITION));

        // Peel the (numerically zero at Gamma, open BC) imaginary part.
        S.reset(new BlockSpMat<double>(S_gamma->graph));
        H.reset(new BlockSpMat<double>(S_gamma->graph));
        const auto& row_ptr = S_gamma->row_ptr();
        const auto& col_ind = S_gamma->col_ind();
        const int n_rows = static_cast<int>(row_ptr.size()) - 1;
        std::vector<double> s_buf, h_buf;
        for (int i = 0; i < n_rows; ++i) {
            const int gr = S_gamma->graph->get_global_index(i);
            const int r = S_gamma->graph->block_sizes[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int lc = col_ind[k];
                const int gc = S_gamma->graph->get_global_index(lc);
                const int c = S_gamma->graph->block_sizes[lc];
                const std::complex<double>* blk = S_gamma->block_data(k);
                s_buf.resize(static_cast<size_t>(r) * c);
                h_buf.resize(static_cast<size_t>(r) * c);
                for (int a = 0; a < r; ++a) {
                    for (int b = 0; b < c; ++b) {
                        const size_t t = static_cast<size_t>(a) * c + b;
                        s_buf[t] = blk[t].real();
                        h_buf[t] = SyntheticEntry(gr * kNorb + a, gc * kNorb + b);
                    }
                }
                S->add_block(gr, gc, s_buf.data(), r, c, AssemblyMode::INSERT,
                             MatrixLayout::RowMajor);
                H->add_block(gr, gc, h_buf.data(), r, c, AssemblyMode::INSERT,
                             MatrixLayout::RowMajor);
            }
        }
        S->assemble();
        H->assemble();

        Zt.reset(new BlockSpMat<double>(S->duplicate(true)));
        rsatb::calc::inverse_sqrt(*S, Zt.get(), {1e-10, 1e-11, 50});
    }

    int n() const { return S->graph->global_scalar_rows(); }

    /// Dense H, S in scalar indexing p = gid * kNorb + orb (single type).
    /// Rank-independent: each rank holds disjoint rows, so summing the
    /// per-rank dense fills reproduces the full matrices everywhere.
    void Densify(std::vector<double>* h_out, std::vector<double>* s_out) const {
        const int N = n();
        h_out->assign(static_cast<size_t>(N) * N, 0.0);
        s_out->assign(static_cast<size_t>(N) * N, 0.0);
        const auto& row_ptr = S->row_ptr();
        const auto& col_ind = S->col_ind();
        const int n_rows = static_cast<int>(row_ptr.size()) - 1;
        for (int i = 0; i < n_rows; ++i) {
            const int gr = S->graph->get_global_index(i);
            const int r = S->graph->block_sizes[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int lc = col_ind[k];
                const int gc = S->graph->get_global_index(lc);
                const int c = S->graph->block_sizes[lc];
                const double* s_blk = S->block_data(k);
                const double* h_blk = H->block_data(k);
                for (int a = 0; a < r; ++a) {
                    for (int b = 0; b < c; ++b) {
                        const size_t at =
                            static_cast<size_t>(gr * kNorb + a) * N + gc * kNorb + b;
                        (*s_out)[at] = s_blk[static_cast<size_t>(a) * c + b];
                        (*h_out)[at] = h_blk[static_cast<size_t>(a) * c + b];
                    }
                }
            }
        }
        int world = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        if (world > 1) {
            MPI_Allreduce(MPI_IN_PLACE, s_out->data(), static_cast<int>(s_out->size()),
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, h_out->data(), static_cast<int>(h_out->size()),
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
    }
};

struct DenseReference {
    std::vector<double> eps;  ///< ascending eigenvalues of the pencil
    std::vector<double> z;    ///< column-major S-orthonormal eigenvectors

    explicit DenseReference(const ChainSystem& sys) {
        const int N = sys.n();
        std::vector<double> s;
        sys.Densify(&z, &s);  // z starts as dense H; dsygv overwrites in place
        eps.assign(N, 0.0);
        const int itype = 1, lwork = 8 * N;
        std::vector<double> work(lwork);
        int info = 0;
        dsygv_(&itype, "V", "U", &N, z.data(), &N, s.data(), &N, eps.data(),
               work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("dsygv failed");
    }
};

double GaussianDelta(double x, double eta) {
    return std::exp(-x * x / (2.0 * eta * eta)) / (eta * std::sqrt(2.0 * M_PI));
}

/// The reference map through route (5.3a): dense A(E) over `set`, contracted
/// by ContractOntoBoxes on the same grid and domain the driver used.
std::vector<double> ReferenceMap(ChainSystem& sys, const DenseReference& ref,
                                 const rsatb::calc::OrbitalSet& set,
                                 const rsatb::calc::RealSpaceGrid& grid,
                                 const rsatb::calc::GridDomain& domain,
                                 double energy, double eta) {
    const int N = sys.n();
    const int n_orb = set.n_orb();
    std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb, 0.0);
    // A_set[i][j] = sum_n delta_eta(E - eps_n) Z_pn Z_qn with p, q the global
    // scalar rows of the set's orbitals.
    std::vector<int> scalar(n_orb);
    for (int a = 0; a < set.n_atom(); ++a) {
        for (int o = 0; o < set.atom_norb[a]; ++o) {
            scalar[set.orb_offset[a] + o] = set.atoms[a] * ChainSystem::kNorb + o;
        }
    }
    for (int m = 0; m < N; ++m) {
        const double w = GaussianDelta(energy - ref.eps[m], eta);
        if (w < 1e-14) continue;
        const double* zm = ref.z.data() + static_cast<size_t>(m) * N;  // column m
        for (int i = 0; i < n_orb; ++i) {
            const double zi = zm[scalar[i]] * w;
            for (int j = 0; j < n_orb; ++j) {
                block[static_cast<size_t>(i) * n_orb + j] += zi * zm[scalar[j]];
            }
        }
    }
    std::vector<double> values;
    sys.collocation->ContractOntoBoxes(*sys.ad, set, block, grid, domain, &values);
    return values;
}

}  // namespace

TEST(RealSpaceLdosTest, CompleteBasisMatchesDenseDiagonalization) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = 0.05 * spread;
    opt.energies = {ref.eps[sys.n() / 5], ref.eps[sys.n() / 2], ref.eps[4 * sys.n() / 5]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.block = 32;
    opt.complete_basis = true;

    const auto result = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    ASSERT_EQ(result.n_states, sys.n());
    ASSERT_EQ(result.domain.n_point(), result.grid.n_point());  // one rank owns all
    EXPECT_LT(result.sampler_residual, 1e-6);
    printf("  filter degree %d, sampler certificate %.2e\n", result.filter_degree,
           result.sampler_residual);

    // The reference set must cover every box, like the driver's demand set
    // does at one rank -- but built through the other selector.
    std::vector<double> centers, radii;
    for (int b = 0; b < result.grid.n_box(); ++b) {
        const auto box = result.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);

    for (size_t k = 0; k < opt.energies.size(); ++k) {
        const auto reference = ReferenceMap(sys, ref, set, result.grid, result.domain,
                                            opt.energies[k], opt.eta);
        double peak = 0.0;
        for (double v : reference) peak = std::max(peak, std::abs(v));
        ASSERT_GT(peak, 0.0);
        double worst = 0.0;
        for (int64_t p = 0; p < result.domain.n_point(); ++p) {
            worst = std::max(worst, std::abs(result.ldos[k][p] - reference[p]));
        }
        printf("  E = %+.3f eV: peak %.3e, worst abs deviation %.3e (%.2e relative)\n",
               opt.energies[k], peak, worst, worst / peak);
        // Measured ~8e-13. The reference shares none of the driver's
        // approximate ingredients (filter truncation, sampler series, CG,
        // isqrtm), so this deviation IS the total method error at these
        // settings -- essentially rounding. The bound is set well above the
        // measurement but tight enough that any real regression fails it.
        EXPECT_LT(worst, 1e-10 * peak) << "energy " << opt.energies[k];
    }
}

TEST(RealSpaceLdosTest, SandwichFormMatchesSolveFormAndDenseAnswer) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    // The two operator forms are two representations of one pencil, not two
    // methods: solve iterates S^-1 H on states drawn with covariance S^-1,
    // sandwich iterates Z^H H Z on states with covariance I and applies Z once
    // at read-out. They must land on the same map.
    //
    // Checking the sandwich map against the DENSE reference, not merely
    // against the solve map, is what makes this a statement about the physics
    // rather than about one path reproducing the other's mistakes. And the
    // failure it guards is not subtle: dropping the read-out Z -- the shape of
    // the bug that let the sandwich form be mistaken for an orthogonal one --
    // is an O(1) error here, not a small one.
    ChainSystem sys;
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = 0.05 * spread;
    opt.energies = {ref.eps[sys.n() / 5], ref.eps[sys.n() / 2]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.block = 32;
    opt.complete_basis = true;

    const auto solved = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    // S is not passed at all: this form never references it, which is exactly
    // what lets setup release it.
    const auto sandwiched = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);

    ASSERT_EQ(sandwiched.n_states, sys.n());
    // No metric sampler is constructed here -- the seed covariance is already
    // the identity, so there is no certificate to report.
    EXPECT_EQ(sandwiched.sampler_residual, -1.0);

    std::vector<double> centers, radii;
    for (int b = 0; b < sandwiched.grid.n_box(); ++b) {
        const auto box = sandwiched.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);

    for (size_t k = 0; k < opt.energies.size(); ++k) {
        const auto reference = ReferenceMap(sys, ref, set, sandwiched.grid,
                                            sandwiched.domain, opt.energies[k], opt.eta);
        double peak = 0.0;
        for (double v : reference) peak = std::max(peak, std::abs(v));
        ASSERT_GT(peak, 0.0);
        double worst_ref = 0.0, worst_form = 0.0;
        for (int64_t p = 0; p < sandwiched.domain.n_point(); ++p) {
            worst_ref = std::max(worst_ref, std::abs(sandwiched.ldos[k][p] - reference[p]));
            worst_form =
                std::max(worst_form, std::abs(sandwiched.ldos[k][p] - solved.ldos[k][p]));
        }
        printf("  E = %+.3f eV: peak %.3e, vs dense %.3e (%.2e rel), vs solve form %.3e\n",
               opt.energies[k], peak, worst_ref, worst_ref / peak, worst_form);
        // Measured ~2e-13 relative against the dense answer and ~1e-13
        // between the forms -- the same rounding floor the solve-form test
        // reports, because the fixture purifies Zt and the metric error this
        // form carries (design 2.4: ~||Z^H S Z - I|| * M / 4) lands below it.
        // Same margin as the sibling test, and far under any dropped-Z
        // regression, which would be O(1).
        EXPECT_LT(worst_ref, 1e-10 * peak) << "energy " << opt.energies[k];
        EXPECT_LT(worst_form, 1e-10 * peak) << "energy " << opt.energies[k];
    }
}

TEST(RealSpaceLdosTest, StochasticConvergesAndStaysPositive) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = 0.05 * spread;
    opt.energies = {ref.eps[sys.n() / 2]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.n_samples = 96;
    opt.block = 32;
    opt.seed = 777;

    const auto result = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);

    // Positivity is exact: a mean of squares cannot go below zero, whatever
    // the noise does.
    for (int64_t p = 0; p < result.domain.n_point(); ++p) {
        ASSERT_GE(result.ldos[0][p], 0.0);
    }

    std::vector<double> centers, radii;
    for (int b = 0; b < result.grid.n_box(); ++b) {
        const auto box = result.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceMap(sys, ref, set, result.grid, result.domain,
                                        opt.energies[0], opt.eta);

    double num = 0.0, den = 0.0;
    for (int64_t p = 0; p < result.domain.n_point(); ++p) {
        const double d = result.ldos[0][p] - reference[p];
        num += d * d;
        den += reference[p] * reference[p];
    }
    const double rel_l2 = std::sqrt(num / den);
    printf("  stochastic map, %d samples: relative L2 error %.3f\n", opt.n_samples,
           rel_l2);
    // ~1/sqrt(96) per resolution volume; the map-level L2 error lands well
    // under this bound, which is loose on purpose -- it must only fail on a
    // broken estimator, not on an unlucky seed.
    EXPECT_LT(rel_l2, 0.5);
}

TEST(RealSpaceLdosTest, ResolvedOwnershipMatchesTheReferenceRule) {
    // The distributed resolution against the rule stated independently of it.
    // ResolveGridOwnership settles ownership through a claim exchange, each
    // rank seeing only its own atoms; ResolveGridOwnershipReference applies
    // the same BoxClaim order to every atom at once. They must agree box for
    // box -- and because the reference needs the true atom owners, this also
    // pins that box ownership really does follow the partition the system
    // arrived with, whatever produced it, which is the whole point of the
    // claim design.
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ChainSystem sys;
    const auto grid = rsatb::calc::MakeRealSpaceGrid(sys.ad->cell, 1.2, 4);
    const std::vector<double> type_rcut = {sys.rcut};

    const auto ownership =
        rsatb::calc::ResolveGridOwnership(*sys.ad, grid, type_rcut, MPI_COMM_WORLD);

    // The true owner of every atom, gathered so the reference can be built.
    // AtomicData redistributes, so an atom's owner is wherever it landed --
    // exactly what the claim rule is supposed to be following.
    std::vector<int> atom_owner(ChainSystem::kAtoms, 0);
    for (int a = 0; a < sys.ad->n_atom; ++a) {
        atom_owner[sys.ad->get_global_index(a)] = rank;
    }
    if (world > 1) {
        MPI_Allreduce(MPI_IN_PLACE, atom_owner.data(), ChainSystem::kAtoms, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);
    }
    // Positions in GLOBAL index order, likewise.
    std::vector<double> pos(3 * ChainSystem::kAtoms, 0.0);
    for (int a = 0; a < sys.ad->n_atom; ++a) {
        const int g = sys.ad->get_global_index(a);
        sys.ad->get_pos(a, &pos[3 * g], &pos[3 * g + 1], &pos[3 * g + 2]);
    }
    if (world > 1) {
        MPI_Allreduce(MPI_IN_PLACE, pos.data(), 3 * ChainSystem::kAtoms, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    const auto reference = rsatb::calc::ResolveGridOwnershipReference(
        grid, pos, std::vector<int>(ChainSystem::kAtoms, 0), atom_owner,
        {ChainSystem::kNorb}, type_rcut, sys.ad->cell, sys.ad->pbc, world);

    // Every box this rank owns must be a box the reference gives it...
    for (int box : ownership.boxes) {
        ASSERT_EQ(reference[box], rank)
            << "np=" << world << ": rank " << rank << " owns box " << box
            << " but the rule gives it to " << reference[box];
    }
    // ...and it must own every such box, so the two agree exactly rather than
    // one merely being a subset of the other.
    int expected = 0;
    for (int b = 0; b < grid.n_box(); ++b) {
        if (reference[b] == rank) ++expected;
    }
    EXPECT_EQ(static_cast<int>(ownership.boxes.size()), expected) << "np=" << world;

    // And the cover is exact across ranks.
    int owned_total = static_cast<int>(ownership.boxes.size());
    if (world > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &owned_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    EXPECT_EQ(owned_total, grid.n_box()) << "np=" << world;
    if (rank == 0) {
        printf("  np=%d: %d boxes, ownership matches the reference rule exactly\n",
               world, grid.n_box());
    }
}

TEST(RealSpaceLdosTest, DistributedRanksAssembleTheSerialAnswer) {
    // Runs at ANY rank count -- this is the test that executes the real MPI
    // paths when launched under mpirun: the demand-set alltoallv, the
    // coefficient-halo alltoallv, the distributed complete-basis sweep, and
    // the per-rank compact maps. The reference is rank-independent by
    // construction: dense matrices are allreduced, the tree is derived from
    // the formulaic geometry on every rank identically, and the reference
    // orbital set is written down by hand so no rank needs positions it does
    // not hold. Any overlap or gap in the rank maps double-counts or drops
    // weight in the allreduced assembly and shows up against the reference.
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ChainSystem sys;
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = 0.05 * spread;
    opt.energies = {ref.eps[sys.n() / 4], ref.eps[2 * sys.n() / 3]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.block = 32;
    opt.complete_basis = true;

    const auto result = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    ASSERT_EQ(result.n_states, sys.n());

    // The reference set, written down from the geometry every rank knows.
    rsatb::calc::OrbitalSet full_set;
    for (int i = 0; i < ChainSystem::kAtoms; ++i) {
        full_set.atoms.push_back(i);
        full_set.atom_type.push_back(0);
        full_set.atom_norb.push_back(ChainSystem::kNorb);
        full_set.positions.insert(full_set.positions.end(), {2.0 * i, 0.0, 0.0});
    }
    full_set.orb_offset.assign(ChainSystem::kAtoms + 1, 0);
    for (int i = 0; i < ChainSystem::kAtoms; ++i) {
        full_set.orb_offset[i + 1] = full_set.orb_offset[i] + ChainSystem::kNorb;
    }
    std::vector<int> all_boxes(result.grid.n_box());
    for (int b = 0; b < result.grid.n_box(); ++b) all_boxes[b] = b;
    const auto full_domain = rsatb::calc::MakeGridDomain(result.grid, all_boxes);

    for (size_t k = 0; k < opt.energies.size(); ++k) {
        // Assemble the global map from the compact per-rank buffers: the
        // boxes are a disjoint cover, so scatter-by-index plus one allreduce
        // is the whole reduction.
        std::vector<double> assembled(result.grid.n_point(), 0.0);
        for (int64_t p = 0; p < result.domain.n_point(); ++p) {
            assembled[rsatb::calc::GlobalIndexOfOwnedPoint(result.grid, result.domain, p)] =
                result.ldos[k][p];
        }
        if (world > 1) {
            MPI_Allreduce(MPI_IN_PLACE, assembled.data(),
                          static_cast<int>(assembled.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }

        const auto reference = ReferenceMap(sys, ref, full_set, result.grid, full_domain,
                                            opt.energies[k], opt.eta);
        double peak = 0.0;
        for (double v : reference) peak = std::max(peak, std::abs(v));
        ASSERT_GT(peak, 0.0);
        double worst = 0.0;
        for (int64_t p = 0; p < full_domain.n_point(); ++p) {
            const int64_t g =
                rsatb::calc::GlobalIndexOfOwnedPoint(result.grid, full_domain, p);
            worst = std::max(worst, std::abs(assembled[g] - reference[p]));
        }
        if (rank == 0) {
            printf("  np=%d, E = %+.3f eV: worst abs deviation %.3e (%.2e relative)\n",
                   world, opt.energies[k], worst, worst / peak);
        }
        EXPECT_LT(worst, 1e-10 * peak) << "energy " << opt.energies[k];
    }
}

TEST(RealSpaceLdosTest, SurveyEstimatorIsUnbiasedAndFollowsTheSurveyLaw) {
    // The survey estimator (design_survey_sampling_ldos.md): a flat-top
    // sketch of the window, the captured part |Pi p phi|^2 divided by the
    // global capture c. On a 160-atom chain (2080 states) with a window of
    // ~200 states and k = J n = 6 x 32 = 192 sketch columns the capture is
    // partial (c < 1), and the map must be unbiased against the dense
    // reference with an error consistent with sqrt(2 (1 - c) / k) -- the
    // realised error within a factor 2.5 of the prediction (a design effect
    // of 1.0 was measured at J = 6 on silicon; the chain's window is small
    // enough for finite-population fluctuations of the count itself).
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys(160);
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();
    const int i0 = 2 * sys.n() / 3;

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = std::max((ref.eps[i0 + 100] - ref.eps[i0 - 100]) / 4.0, 0.002 * spread);
    opt.energies = {ref.eps[i0]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.n_samples = 32;
    opt.block = 8;
    opt.sketch_families = 6;
    opt.seed = 424242;
    const auto res = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);

    ASSERT_EQ(res.capture.size(), 1u);
    ASSERT_GT(res.sketch_rank[0], 0);
    ASSERT_LT(res.capture[0], 0.95) << "the window must exceed the sketch for this test";

    std::vector<double> centers, radii;
    for (int b = 0; b < res.grid.n_box(); ++b) {
        const auto box = res.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceMap(sys, ref, set, res.grid, res.domain,
                                        opt.energies[0], opt.eta);
    double num = 0.0, den = 0.0, sum_est = 0.0, sum_ref = 0.0;
    for (int64_t p = 0; p < res.domain.n_point(); ++p) {
        ASSERT_TRUE(std::isfinite(res.ldos[0][p]));
        ASSERT_GE(res.ldos[0][p], 0.0);
        const double d = res.ldos[0][p] - reference[p];
        num += d * d;
        den += reference[p] * reference[p];
        sum_est += res.ldos[0][p];
        sum_ref += reference[p];
    }
    const double rel = std::sqrt(num / den);
    printf("  survey: capture %.3f rank %d, rel L2 %.4f (predicted %.4f), integral ratio %.4f\n",
           res.capture[0], res.sketch_rank[0], rel, res.predicted_rel_error[0],
           sum_est / sum_ref);
    // Unbiased up to the noise of the cross-fitted capture on this fixture
    // (a few hundred window states, 4 blocks: ~8 %; on production windows
    // of 10^3 states and 20 blocks it is below 1 %); the pointwise noise
    // follows the survey law.
    EXPECT_NEAR(sum_est / sum_ref, 1.0, 0.12);
    EXPECT_LT(rel, 2.5 * res.predicted_rel_error[0] + 0.01);
    EXPECT_GT(rel, 0.2 * res.predicted_rel_error[0]);
}

TEST(RealSpaceLdosTest, SurveyEstimatorIsExactWhenTheSketchSpansTheWindow) {
    // With more sketch columns than window states the flat-top sketch spans
    // the window with probability one (no tails, no oversampling), the
    // capture is 1 and the map equals the dense reference to the filter
    // truncation -- exactness without a residual pass.
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = std::max((ref.eps[5] - ref.eps[0]) / 4.0, 0.002 * spread);
    opt.energies = {ref.eps[2 * sys.n() / 3]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.n_samples = 32;
    opt.block = 8;
    opt.sketch_families = 6;
    opt.seed = 424242;
    const auto res = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);

    std::vector<double> centers, radii;
    for (int b = 0; b < res.grid.n_box(); ++b) {
        const auto box = res.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceMap(sys, ref, set, res.grid, res.domain,
                                        opt.energies[0], opt.eta);
    double num = 0.0, den = 0.0;
    for (int64_t p = 0; p < res.domain.n_point(); ++p) {
        const double d = res.ldos[0][p] - reference[p];
        num += d * d;
        den += reference[p] * reference[p];
    }
    const double rel = std::sqrt(num / den);
    printf("  survey exact: capture %.6f rank %d, rel L2 %.2e\n", res.capture[0],
           res.sketch_rank[0], rel);
    EXPECT_NEAR(res.capture[0], 1.0, 1e-3);
    EXPECT_LT(rel, 1e-3);
    EXPECT_LT(res.predicted_rel_error[0], 0.05);
}

TEST(RealSpaceLdosTest, SolveFormSurveyMatchesSandwich) {
    // The survey construction is representation-independent: in the solve
    // form the sketch is whitened in the S metric and the same captured map
    // results, to the accuracy of the two operator forms.
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    const double spread = ref.eps.back() - ref.eps.front();

    rsatb::calc::RealSpaceLdosOptions opt;
    opt.eta = std::max((ref.eps[5] - ref.eps[0]) / 4.0, 0.002 * spread);
    opt.energies = {ref.eps[2 * sys.n() / 3]};
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.n_samples = 32;
    opt.block = 8;
    opt.sketch_families = 6;
    opt.seed = 424242;
    const auto sandwich = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);
    const auto solve = rsatb::calc::ComputeRealSpaceLdos(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    double num = 0.0, den = 0.0;
    for (size_t p = 0; p < sandwich.ldos[0].size(); ++p) {
        const double d = sandwich.ldos[0][p] - solve.ldos[0][p];
        num += d * d;
        den += sandwich.ldos[0][p] * sandwich.ldos[0][p];
    }
    printf("  solve vs sandwich survey: capture %.4f / %.4f, rel L2 %.2e\n",
           sandwich.capture[0], solve.capture[0], std::sqrt(num / den));
    EXPECT_LT(std::sqrt(num / den), 1e-2);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
