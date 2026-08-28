// The real-space electron density (calc/rs_density.hpp) against dense
// diagonalization, both operator forms.
//
// Same fixture idea as test_rs_ldos.cpp: a carbon chain with the real op2c
// overlap and a seeded symmetric H on its pattern. Same two-sided design: the
// driver filters sqrt(f_FD) through the recursion and squares onto the grid;
// the reference assembles D = sum_n f_FD(eps_n) c_n c_n^T from dsygv
// eigenpairs and contracts through ContractOntoBoxes. Between the pencil and
// the grid values the routes share only Z.

#include "../rs_density.hpp"
#include "../isqrtm.hpp"
#include "../../model/lcao/op2c_engine.hpp"
#include "../../test_data_path.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"

#include "nao/orbital_evaluator.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <memory>
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

double SyntheticEntry(int p, int q) {
    const double s = static_cast<double>(p + q);
    const double m = static_cast<double>(p) * static_cast<double>(q);
    double v = 0.6 * std::sin(0.37 * s) * std::cos(0.11 * std::fmod(m, 97.0));
    if (p == q) v += 3.0 + 0.02 * p;
    return v;
}

double Fermi(double e, double mu, double kT) {
    const double u = (e - mu) / kT;
    return u > 0.0 ? std::exp(-u) / (1.0 + std::exp(-u)) : 1.0 / (1.0 + std::exp(u));
}

struct ChainSystem {
    static constexpr int kAtoms = 10;
    static constexpr int kNorb = 13;

    std::unique_ptr<vbcsr::atomic::AtomicData> ad;
    std::unique_ptr<rsatb::calc::OrbitalCollocation> collocation;
    std::unique_ptr<vbcsr::atomic::ImageContainer<double>> img;
    std::unique_ptr<BlockSpMat<std::complex<double>>> S_gamma;
    std::unique_ptr<BlockSpMat<double>> S, H, Zt;
    double rcut = 0.0;

    ChainSystem() {
        collocation = std::make_unique<rsatb::calc::OrbitalCollocation>(
            std::vector<std::string>{DiamondDir() + "C_gga_7au_100Ry_2s2p1d.orb"},
            MPI_COMM_WORLD);
        rcut = collocation->rcut_angstrom(0);

        // Atoms CENTERED in the open-boundary cell. The rs_ldos fixture puts
        // the chain on the y = z = 0 edge, which is harmless for pointwise
        // maps but clips most of every orbital's norm out of the cell -- an
        // s orbital on a corner keeps ~1/8 -- and this file integrates.
        const double lx = 2.0 * kAtoms + 4.0 * rcut, lt = 4.0 * rcut + 6.0;
        const double x0 = 0.5 * (lx - 2.0 * (kAtoms - 1)), yz = 0.5 * lt;
        std::vector<double> pos;
        for (int i = 0; i < kAtoms; ++i) pos.insert(pos.end(), {x0 + 2.0 * i, yz, yz});
        const std::vector<int> z(kAtoms, 6);
        const std::vector<double> cell = {lx, 0, 0, 0, lt, 0, 0, 0, lt};
        ad.reset(vbcsr::atomic::AtomicData::from_points(
            pos, z, cell, {false, false, false}, {rcut}, {kNorb}, MPI_COMM_WORLD));

        rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
        op.init(1, 1, false, DiamondDir(), {"C_gga_7au_100Ry_2s2p1d.orb"}, "", {},
                "rs_density_opconst.log");
        img = op.build_overlap(*ad);
        S_gamma.reset(img->sample_k({0.0, 0.0, 0.0},
                                    vbcsr::atomic::PhaseConvention::R_AND_POSITION));

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
    std::vector<double> eps;
    std::vector<double> z;  ///< column-major S-orthonormal eigenvectors

    explicit DenseReference(const ChainSystem& sys) {
        const int N = sys.n();
        std::vector<double> s;
        sys.Densify(&z, &s);
        eps.assign(N, 0.0);
        const int itype = 1, lwork = 8 * N;
        std::vector<double> work(lwork);
        int info = 0;
        dsygv_(&itype, "V", "U", &N, z.data(), &N, s.data(), &N, eps.data(),
               work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("dsygv failed");
    }
};

/// Reference density map: D over the set's orbitals from the dense
/// eigenpairs, contracted on the same grid and domain the driver used.
std::vector<double> ReferenceDensity(ChainSystem& sys, const DenseReference& ref,
                                     const rsatb::calc::OrbitalSet& set,
                                     const rsatb::calc::RealSpaceGrid& grid,
                                     const rsatb::calc::GridDomain& domain,
                                     double mu, double kT, double spin) {
    const int N = sys.n();
    const int n_orb = set.n_orb();
    std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb, 0.0);
    std::vector<int> scalar(n_orb);
    for (int a = 0; a < set.n_atom(); ++a) {
        for (int o = 0; o < set.atom_norb[a]; ++o) {
            scalar[set.orb_offset[a] + o] = set.atoms[a] * ChainSystem::kNorb + o;
        }
    }
    for (int m = 0; m < N; ++m) {
        const double w = spin * Fermi(ref.eps[m], mu, kT);
        if (w < 1e-15) continue;
        const double* zm = ref.z.data() + static_cast<size_t>(m) * N;
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

rsatb::calc::RealSpaceDensityOptions BaseOptions(const DenseReference& ref, int n) {
    rsatb::calc::RealSpaceDensityOptions opt;
    const double spread = ref.eps.back() - ref.eps.front();
    opt.mu = ref.eps[n / 2] + 0.01 * spread;  // mid-spectrum, off any eigenvalue
    opt.kT = 0.04 * spread;
    opt.emin = ref.eps.front();
    opt.emax = ref.eps.back();
    opt.spacing = 1.2;
    opt.block = 32;
    opt.complete_basis = true;
    return opt;
}

}  // namespace

TEST(RealSpaceDensityTest, CompleteBasisMatchesDenseDiagonalization) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    const auto opt = BaseOptions(ref, sys.n());

    const auto result = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    ASSERT_EQ(result.n_states, sys.n());
    EXPECT_LT(result.sampler_residual, 1e-6);

    // Exact occupation sum, for the quadrature-convergence check below.
    double n_exact = 0.0;
    for (const double e : ref.eps) n_exact += opt.spin_degeneracy * Fermi(e, opt.mu, opt.kT);
    printf("  filter degree %d, electrons: grid %.6f vs exact %.6f (%.2f%%)\n",
           result.filter_degree, result.electrons_on_grid, n_exact,
           100.0 * std::abs(result.electrons_on_grid - n_exact) / n_exact);

    std::vector<double> centers, radii;
    for (int b = 0; b < result.grid.n_box(); ++b) {
        const auto box = result.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceDensity(sys, ref, set, result.grid, result.domain,
                                            opt.mu, opt.kT, opt.spin_degeneracy);
    double peak = 0.0, worst = 0.0;
    for (int64_t p = 0; p < result.domain.n_point(); ++p) {
        peak = std::max(peak, std::abs(reference[p]));
        worst = std::max(worst, std::abs(result.density[p] - reference[p]));
    }
    ASSERT_GT(peak, 0.0);
    printf("  density: peak %.3e, worst abs deviation %.3e (%.2e relative)\n", peak,
           worst, worst / peak);
    // The routes share only Zt (tol 1e-10) between the pencil and the grid;
    // the deviation is the whole method error at these settings.
    EXPECT_LT(worst, 1e-8 * peak);

    // Integral against the REFERENCE's integral on the same grid: the
    // midpoint quadrature is common to both sides and cancels, so this is a
    // check of the estimator alone. The comparison against the analytic
    // occupation sum is quadrature-limited and is tested by convergence in
    // GridIntegralConvergesWithSpacing below, not by a constant here.
    double ref_integral_local = 0.0;
    for (int64_t p = 0; p < result.domain.n_point(); ++p) ref_integral_local += reference[p];
    const std::vector<double>& cc = sys.ad->cell;
    const double det = cc[0] * (cc[4] * cc[8] - cc[5] * cc[7]) -
                       cc[1] * (cc[3] * cc[8] - cc[5] * cc[6]) +
                       cc[2] * (cc[3] * cc[7] - cc[4] * cc[6]);
    const double ref_integral =
        ref_integral_local * std::abs(det) / static_cast<double>(result.grid.n_point());
    EXPECT_NEAR(result.electrons_on_grid, ref_integral,
                1e-8 * std::max(1.0, ref_integral));
    // ... and against the analytic occupation sum, quadrature-limited at this
    // 1.2 A test grid (measured 4.5%).
    EXPECT_NEAR(result.electrons_on_grid, n_exact, 0.08 * n_exact);

    // Positivity is structural: a mean of squares times a positive weight.
    for (int64_t p = 0; p < result.domain.n_point(); ++p) {
        ASSERT_GE(result.density[p], 0.0);
    }
}

TEST(RealSpaceDensityTest, SandwichFormMatchesSolveForm) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    const auto opt = BaseOptions(ref, sys.n());

    const auto solved = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    const auto sandwiched = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);

    double peak = 0.0, worst = 0.0;
    for (int64_t p = 0; p < solved.domain.n_point(); ++p) {
        peak = std::max(peak, std::abs(solved.density[p]));
        worst = std::max(worst, std::abs(solved.density[p] - sandwiched.density[p]));
    }
    printf("  solve vs sandwich: peak %.3e, worst abs deviation %.3e (%.2e relative)\n",
           peak, worst, worst / peak);
    // Both forms compute D exactly in complete_basis mode; they differ in
    // where Zt enters (sampler covariance vs congruence), so this pins the
    // two metric paths against each other.
    EXPECT_LT(worst, 1e-7 * peak);
    EXPECT_NEAR(sandwiched.electrons_on_grid, solved.electrons_on_grid,
                1e-6 * std::max(1.0, solved.electrons_on_grid));
}

TEST(RealSpaceDensityTest, GridIntegralMatchesReferenceAcrossSpacings) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());
    // mu far above the spectrum: f == 1, D = S^-1, integral = spin * N_orb --
    // an electron count known without the eigensolver.
    const double spread = ref.eps.back() - ref.eps.front();
    opt.mu = ref.eps.back() + 0.5 * spread;
    opt.kT = 0.02 * spread;
    opt.spin_degeneracy = 1.0;

    // Two claims: the driver equals the dense route on its own grid at every
    // spacing (the estimator, quadrature-free), and the count against the
    // ANALYTIC occupation sum converges with the grid (the quadrature;
    // measured 11.4% at 1.2 A, 0.55% at 0.6 A with the chain centered).
    double err[2] = {0.0, 0.0};
    const double spacings[2] = {1.2, 0.6};
    for (int i = 0; i < 2; ++i) {
        opt.spacing = spacings[i];
        const auto result = rsatb::calc::ComputeRealSpaceDensity(
            *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
            *sys.collocation, opt, MPI_COMM_WORLD);
        err[i] = std::abs(result.electrons_on_grid - sys.n()) / sys.n();

        // Reference on the same grid, to separate the estimator from the
        // quadrature: if driver == reference here, everything above the grid
        // stage is exact and the analytic gap is the grid's, not the method's.
        std::vector<double> centers, radii;
        for (int b = 0; b < result.grid.n_box(); ++b) {
            const auto box = result.grid.BoxAt(b);
            centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
            radii.push_back(box.radius);
        }
        const auto set = rsatb::calc::BuildOrbitalSetForSpheres(
            *sys.ad, centers, radii, {sys.rcut}, MPI_COMM_WORLD);
        const auto reference = ReferenceDensity(sys, ref, set, result.grid,
                                                result.domain, opt.mu, opt.kT,
                                                opt.spin_degeneracy);
        double ref_sum = 0.0, worst = 0.0, peak = 0.0;
        for (int64_t pnt = 0; pnt < result.domain.n_point(); ++pnt) {
            ref_sum += reference[pnt];
            peak = std::max(peak, std::abs(reference[pnt]));
            worst = std::max(worst, std::abs(result.density[pnt] - reference[pnt]));
        }
        const std::vector<double>& cc = sys.ad->cell;
        const double det = cc[0] * (cc[4] * cc[8] - cc[5] * cc[7]) -
                           cc[1] * (cc[3] * cc[8] - cc[5] * cc[6]) +
                           cc[2] * (cc[3] * cc[7] - cc[4] * cc[6]);
        const double ref_integral =
            ref_sum * std::abs(det) / static_cast<double>(result.grid.n_point());
        printf("  spacing %.2f A: electrons %.4f, reference-on-grid %.4f, N_orb %d "
               "(gap %.2f%%), pointwise dev %.2e of peak %.2e\n",
               spacings[i], result.electrons_on_grid, ref_integral, sys.n(),
               100.0 * err[i], worst, peak);
        // The estimator must match the dense route on ITS OWN grid at every
        // spacing; this is the method claim, quadrature-free.
        EXPECT_LT(worst, 1e-8 * peak) << "spacing " << spacings[i];
        EXPECT_NEAR(result.electrons_on_grid, ref_integral,
                    1e-8 * std::max(1.0, ref_integral))
            << "spacing " << spacings[i];
    }
}

TEST(RealSpaceDensityTest, StochasticConvergesToCompleteBasis) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());

    const auto exact = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);

    opt.complete_basis = false;
    opt.n_samples = 96;
    const auto sampled = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, nullptr, *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);

    double peak = 0.0, worst = 0.0;
    for (int64_t p = 0; p < exact.domain.n_point(); ++p) {
        peak = std::max(peak, exact.density[p]);
        worst = std::max(worst, std::abs(sampled.density[p] - exact.density[p]));
        ASSERT_GE(sampled.density[p], 0.0);  // positivity survives sampling
    }
    printf("  96 samples: worst deviation %.3e of peak %.3e (%.1f%%)\n", worst, peak,
           100.0 * worst / peak);
    // Statistical, deliberately loose: the check is that sampling converges
    // toward the exact map, not a precise rate.
    EXPECT_LT(worst, 0.35 * peak);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int code = RUN_ALL_TESTS();
    MPI_Finalize();
    return code;
}

// The overlap as the GRID sees it against op2c's two-center integrals: the
// end-to-end anchor of the absolute normalization (evaluator units, the
// Bohr->Angstrom conversions, the voxel volume), which the pointwise map
// tests cannot see because both routes share the collocation.
//
// History: this probe once appeared to expose a normalization bug -- grid
// <0|0> ~ 0.15 against op2c's 1.0, suspiciously (A/bohr)^3. The real cause
// was the FIXTURE: the chain sat on the y = z = 0 edge of the open cell, so
// ~7/8 of the on-site norm and ~3/4 of the bond overlap lay outside the
// integration volume. A direct Bohr-space probe of the evaluator
// (EvaluatorMatchesOp2cInBohr below) shows the orbital stack matches op2c
// to six digits. Geometry, not units; the fixture is now centered.
TEST(RealSpaceDensityTest, GridOverlapMatchesOp2c) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";
    ChainSystem sys;

    // Two-atom overlap block (0,1) and on-site (0,0) from op2c.
    std::vector<double> Hd, Sd;
    sys.Densify(&Hd, &Sd);
    const int N = sys.n();

    double norm_prev = 0.0;
    for (double spacing : {0.6, 0.3, 0.15}) {
        auto grid = rsatb::calc::MakeRealSpaceGrid(sys.ad->cell, spacing, 0);
        std::vector<double> centers, radii;
        for (int b = 0; b < grid.n_box(); ++b) {
            const auto box = grid.BoxAt(b);
            centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
            radii.push_back(box.radius);
        }
        auto domain = rsatb::calc::MakeGridDomain(
            grid, [&] { std::vector<int> all(grid.n_box()); for (int b = 0; b < grid.n_box(); ++b) all[b] = b; return all; }());
        const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                                {sys.rcut}, MPI_COMM_WORLD);
        const int n_orb = set.n_orb();
        std::vector<int> scalar(n_orb);
        for (int a = 0; a < set.n_atom(); ++a)
            for (int o = 0; o < set.atom_norb[a]; ++o)
                scalar[set.orb_offset[a] + o] = set.atoms[a] * ChainSystem::kNorb + o;

        // B = e_p e_p^T (norm of orbital 0 of atom 0) and the (0,0)x(1,0)
        // cross term.
        auto integral_of = [&](int p, int q) {
            std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb, 0.0);
            int ip = -1, iq = -1;
            for (int i = 0; i < n_orb; ++i) {
                if (scalar[i] == p) ip = i;
                if (scalar[i] == q) iq = i;
            }
            block[static_cast<size_t>(ip) * n_orb + iq] += 0.5;
            block[static_cast<size_t>(iq) * n_orb + ip] += 0.5;
            std::vector<double> values;
            sys.collocation->ContractOntoBoxes(*sys.ad, set, block, grid, domain, &values);
            double sum = 0.0;
            for (double v : values) sum += v;
            const std::vector<double>& cc = sys.ad->cell;
            const double det = cc[0] * (cc[4] * cc[8] - cc[5] * cc[7]) -
                               cc[1] * (cc[3] * cc[8] - cc[5] * cc[6]) +
                               cc[2] * (cc[3] * cc[7] - cc[4] * cc[6]);
            return sum * std::abs(det) / static_cast<double>(grid.n_point());
        };
        const double norm00 = integral_of(0, 0);
        const double cross = integral_of(0, ChainSystem::kNorb);  // atom0-orb0, atom1-orb0
        printf("  spacing %.2f: grid <0|0> = %.6f (op2c %.6f), grid <0|13> = %.6f (op2c %.6f)\n",
               spacing, norm00, Sd[0], cross, Sd[ChainSystem::kNorb]);
        norm_prev = norm00;
    }
    // Assert at the finest spacing only; the loop above shows the approach.
    EXPECT_NEAR(norm_prev, Sd[0], 0.02 * std::abs(Sd[0]));
    (void)Hd;
}

// The evaluator ALONE, in pure Bohr, no collocation: builds its own
// AtomicRadials exactly as OrbitalCollocation does and integrates phi_0^2 and
// the 2-Angstrom cross overlap on a fine Cartesian grid. This is the direct
// anchor of the orbital stack against op2c's two-center integrals --
// normalization AND radial shape at once, with no grid machinery in between.
TEST(RealSpaceDensityTest, EvaluatorMatchesOp2cInBohr) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    AtomicRadials radials;
    radials.build(DiamondDir() + "C_gga_7au_100Ry_2s2p1d.orb", 0, 0, 0, nullptr,
                  MPI_COMM_WORLD);
    OrbitalEvaluator ev(radials);
    printf("  nphi %d, rcut_max %.4f bohr\n", ev.nphi(), ev.rcut_max());

    const double h = 0.08;                       // bohr
    const double d = 2.0 * 1.8897259886;         // 2 Angstrom in bohr
    const double lo = -7.05, hi_a = 7.05;        // covers rcut 7
    const int nyz = static_cast<int>((hi_a - lo) / h) + 1;
    const int nx = static_cast<int>((hi_a + d - lo) / h) + 1;
    const std::vector<int> active = {0};
    const int stride = ev.nphi();

    std::vector<double> xyzA, xyzB, phiA, phiB;
    double onsite = 0.0, cross = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
        const double x = lo + ix * h;
        xyzA.clear(); xyzB.clear();
        for (int iy = 0; iy < nyz; ++iy) {
            for (int iz = 0; iz < nyz; ++iz) {
                const double y = lo + iy * h, z = lo + iz * h;
                xyzA.insert(xyzA.end(), {x, y, z});
                xyzB.insert(xyzB.end(), {x - d, y, z});
            }
        }
        const int np = static_cast<int>(xyzA.size() / 3);
        phiA.assign(static_cast<size_t>(np) * stride, 0.0);
        phiB.assign(static_cast<size_t>(np) * stride, 0.0);
        ev.evaluate_active_batch(np, xyzA.data(), stride, active, phiA.data());
        ev.evaluate_active_batch(np, xyzB.data(), stride, active, phiB.data());
        for (int p = 0; p < np; ++p) {
            const double a = phiA[static_cast<size_t>(p) * stride];
            onsite += a * a;
            cross += a * phiB[static_cast<size_t>(p) * stride];
        }
    }
    onsite *= h * h * h;
    cross *= h * h * h;
    printf("  BOHR integrals: <0|0> = %.6f  (op2c: 1.000000)\n", onsite);
    printf("                  <0A|0B> at 2 A = %.6f  (op2c: 0.188317)\n", cross);
    printf("  diagnostics: 1/(4pi) = %.6f, 4pi = %.6f, (A/bohr)^3 = %.6f\n",
           1.0 / (4.0 * M_PI), 4.0 * M_PI, std::pow(1.0 / 1.8897259886, 3));

    // Measured 1.000000 and 0.188317 at h = 0.08 bohr -- six digits against
    // op2c. The bounds leave room for the quadrature, nothing else.
    EXPECT_NEAR(onsite, 1.0, 1e-4);
    EXPECT_NEAR(cross, 0.188317, 1e-4);
}

// ---------------------------------------------------------------------------
// Fragment control variate (rs_fragment.hpp).

/// One fragment covering the whole system: T0 evaluates the SAME Chebyshev
/// table on the exact dense pencil, so the deterministic baseline carries the
/// entire map and the stochastic correction carries only the Zt-truncation
/// residual. A handful of samples must then match the dense reference at
/// near-complete-basis accuracy -- the same 4 samples WITHOUT fragments sit
/// at ~50% pointwise error.
TEST(RealSpaceDensityTest, FragmentCvExactInTheFullFragmentLimit) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());
    opt.complete_basis = false;
    opt.n_samples = 4;
    opt.fragment_core = 1e4;    // one core bin
    opt.fragment_buffer = 1e4;  // dressed with everything

    const auto result = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);
    ASSERT_EQ(result.n_fragments, 1);

    std::vector<double> centers, radii;
    for (int b = 0; b < result.grid.n_box(); ++b) {
        const auto box = result.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceDensity(sys, ref, set, result.grid, result.domain,
                                            opt.mu, opt.kT, opt.spin_degeneracy);
    double peak = 0.0, worst = 0.0;
    for (int64_t p = 0; p < result.domain.n_point(); ++p) {
        peak = std::max(peak, std::abs(reference[p]));
        worst = std::max(worst, std::abs(result.density[p] - reference[p]));
    }
    ASSERT_GT(peak, 0.0);
    printf("  full-fragment CV: worst %.3e of peak %.3e (%.2e relative), "
           "correction rms %.3e\n",
           worst, peak, worst / peak, result.correction_rms);
    // T0 is the estimator's own operator here -- the (Z, H) compression over
    // everything -- so baseline + correction reproduce it to rounding.
    // Measured 2.3e-14 relative; the bound leaves two digits of slack.
    EXPECT_LT(worst, 1e-11 * peak);
    EXPECT_LT(result.correction_rms, 1e-11 * peak);
}

/// Multi-fragment regime: cores of a few atoms, buffer covering neighbors.
/// Same seeds with and without the control variate; the CV error against the
/// dense reference must drop by well over the factor the residual predicts.
TEST(RealSpaceDensityTest, FragmentCvReducesStochasticError) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());
    opt.complete_basis = false;
    opt.n_samples = 8;

    auto run = [&](double buffer) {
        auto o = opt;
        o.fragment_core = 4.0;
        o.fragment_buffer = buffer;
        return rsatb::calc::ComputeRealSpaceDensity(
            *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
            *sys.collocation, o, MPI_COMM_WORLD);
    };
    const auto plain = run(0.0);
    const auto cv = run(8.0);
    ASSERT_GT(cv.n_fragments, 1);

    std::vector<double> centers, radii;
    for (int b = 0; b < plain.grid.n_box(); ++b) {
        const auto box = plain.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceDensity(sys, ref, set, plain.grid, plain.domain,
                                            opt.mu, opt.kT, opt.spin_degeneracy);
    double peak = 0.0, rms_plain = 0.0, rms_cv = 0.0;
    for (int64_t p = 0; p < plain.domain.n_point(); ++p) {
        peak = std::max(peak, std::abs(reference[p]));
        const double dp = plain.density[p] - reference[p];
        const double dc = cv.density[p] - reference[p];
        rms_plain += dp * dp;
        rms_cv += dc * dc;
    }
    rms_plain = std::sqrt(rms_plain / static_cast<double>(plain.domain.n_point()));
    rms_cv = std::sqrt(rms_cv / static_cast<double>(plain.domain.n_point()));
    printf("  8 samples, %d fragments: rms error plain %.3e, cv %.3e (ratio %.3f), "
           "correction rms %.3e of peak %.3e\n",
           cv.n_fragments, rms_plain, rms_cv, rms_cv / rms_plain,
           cv.correction_rms, peak);
    ASSERT_GT(rms_plain, 0.0);
    // The knob must actually buy accuracy at fixed samples: the fixture's
    // 8 A buffer captures most of the filtered operator, so a 2x margin is
    // loose against the measured reduction.
    EXPECT_LT(rms_cv, 0.5 * rms_plain);
}

/// Solve form, full-fragment limit, with a DELIBERATELY crude Zt (the raw
/// graph seed: kappa(W) ~ 2, a genuinely non-trivial sampler polynomial q).
/// The solve chain T0 = Zc V p(eps) V^T W q(W) over one full fragment IS the
/// estimator's own operator chain p(S^-1 H) Zt q(W), so baseline plus
/// correction must reproduce the dense reference down to the recursion's
/// inner CG tolerance -- this pins the generalized pencil (M[F,F], W[F,F]),
/// the K factor, and the gauge matching of the coloring in one shot. The
/// same 4 samples without fragments sit at ~50% pointwise error.
TEST(RealSpaceDensityTest, SolveFormFragmentCvExactInTheFullFragmentLimit) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());
    opt.complete_basis = false;
    opt.n_samples = 4;
    opt.fragment_core = 1e4;    // one core bin
    opt.fragment_buffer = 1e4;  // dressed with everything

    // maxit = 0 returns the graph seed itself: any invertible Zt is legal
    // for the solve form (its quality only sets iteration counts), and a
    // crude one is what makes q(W) a real polynomial here.
    BlockSpMat<double> Zt_crude(sys.S->duplicate(true));
    rsatb::calc::inverse_sqrt(*sys.S, &Zt_crude, {1e-1, 1e-6, 0});

    const auto result = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), Zt_crude, rsatb::calc::ChebyOpForm::Solve,
        *sys.collocation, opt, MPI_COMM_WORLD);
    ASSERT_EQ(result.n_fragments, 1);

    std::vector<double> centers, radii;
    for (int b = 0; b < result.grid.n_box(); ++b) {
        const auto box = result.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceDensity(sys, ref, set, result.grid, result.domain,
                                            opt.mu, opt.kT, opt.spin_degeneracy);
    double peak = 0.0, worst = 0.0;
    for (int64_t p = 0; p < result.domain.n_point(); ++p) {
        peak = std::max(peak, std::abs(reference[p]));
        worst = std::max(worst, std::abs(result.density[p] - reference[p]));
    }
    ASSERT_GT(peak, 0.0);
    printf("  solve full-fragment CV: worst %.3e of peak %.3e (%.2e relative), "
           "correction rms %.3e\n",
           worst, peak, worst / peak, result.correction_rms);
    // The residual is the inner CG tolerance accumulated over the sweep, not
    // the sandwich's pure rounding; 1e-4 is three orders below the plain
    // 4-sample noise and loose against the expected ~1e-6.
    EXPECT_LT(worst, 1e-4 * peak);
    EXPECT_LT(result.correction_rms, 1e-4 * peak);
}

/// Solve form, multi-fragment regime with the production (purified) factor:
/// same seeds with and without the control variate; the reduction must
/// match the sandwich behavior on the same fixture.
TEST(RealSpaceDensityTest, SolveFormFragmentCvReducesStochasticError) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());
    opt.complete_basis = false;
    opt.n_samples = 8;

    auto run = [&](double buffer) {
        auto o = opt;
        o.fragment_core = 4.0;
        o.fragment_buffer = buffer;
        return rsatb::calc::ComputeRealSpaceDensity(
            *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Solve,
            *sys.collocation, o, MPI_COMM_WORLD);
    };
    const auto plain = run(0.0);
    const auto cv = run(8.0);
    ASSERT_GT(cv.n_fragments, 1);

    std::vector<double> centers, radii;
    for (int b = 0; b < plain.grid.n_box(); ++b) {
        const auto box = plain.grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sys.ad, centers, radii,
                                                            {sys.rcut}, MPI_COMM_WORLD);
    const auto reference = ReferenceDensity(sys, ref, set, plain.grid, plain.domain,
                                            opt.mu, opt.kT, opt.spin_degeneracy);
    double rms_plain = 0.0, rms_cv = 0.0;
    for (int64_t p = 0; p < plain.domain.n_point(); ++p) {
        const double dp = plain.density[p] - reference[p];
        const double dc = cv.density[p] - reference[p];
        rms_plain += dp * dp;
        rms_cv += dc * dc;
    }
    rms_plain = std::sqrt(rms_plain / static_cast<double>(plain.domain.n_point()));
    rms_cv = std::sqrt(rms_cv / static_cast<double>(plain.domain.n_point()));
    printf("  solve 8 samples, %d fragments: rms error plain %.3e, cv %.3e "
           "(ratio %.3f)\n",
           cv.n_fragments, rms_plain, rms_cv, rms_cv / rms_plain);
    ASSERT_GT(rms_plain, 0.0);
    EXPECT_LT(rms_cv, 0.5 * rms_plain);
}


/// Energy windows (ew-sDFT): the windowed filters telescope back to the full
/// occupation, so the complete-basis (exact) windowed run must match the
/// unwindowed one; and at fixed seeds the windowed stochastic error against
/// the dense reference can never be meaningfully worse (Cauchy-Schwarz drops
/// the cross-window variance).
TEST(RealSpaceDensityTest, EnergyWindowsTelescopeAndNeverHurt) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    ChainSystem sys;
    DenseReference ref(sys);
    auto opt = BaseOptions(ref, sys.n());

    auto win = opt;
    win.energy_windows = 4;
    const auto exact1 = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, opt, MPI_COMM_WORLD);
    const auto exact4 = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, win, MPI_COMM_WORLD);
    double peak = 0.0, worst = 0.0;
    for (int64_t p = 0; p < exact1.domain.n_point(); ++p) {
        peak = std::max(peak, std::abs(exact1.density[p]));
        worst = std::max(worst, std::abs(exact1.density[p] - exact4.density[p]));
    }
    printf("  windows telescope: worst %.3e of peak %.3e\n", worst, peak);
    EXPECT_LT(worst, 1e-6 * peak);  // filter-table truncation, nothing else

    auto st1 = opt, st4 = opt;
    st1.complete_basis = st4.complete_basis = false;
    st1.n_samples = st4.n_samples = 8;
    st4.energy_windows = 4;
    const auto s1 = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, st1, MPI_COMM_WORLD);
    const auto s4 = rsatb::calc::ComputeRealSpaceDensity(
        *sys.ad, *sys.H, sys.S.get(), *sys.Zt, rsatb::calc::ChebyOpForm::Sandwich,
        *sys.collocation, st4, MPI_COMM_WORLD);
    double r1 = 0.0, r4 = 0.0;
    for (int64_t p = 0; p < s1.domain.n_point(); ++p) {
        const double d1 = s1.density[p] - exact1.density[p];
        const double d4 = s4.density[p] - exact1.density[p];
        r1 += d1 * d1;
        r4 += d4 * d4;
    }
    r1 = std::sqrt(r1);
    r4 = std::sqrt(r4);
    printf("  8 samples: rms error 1 window %.3e, 4 windows %.3e (ratio %.3f)\n",
           r1, r4, r4 / r1);
    // Guaranteed no harm in variance; a single-seed draw gets slack for luck.
    EXPECT_LT(r4, 1.2 * r1);
}
