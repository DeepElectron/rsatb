// Orbital collocation, checked against quantities the orbital files fix.
//
// This is the layer where a convention mistake is silent: a wrong radial
// interpolation, a mis-normalized spherical harmonic, or a Bohr/Angstrom slip
// would all still produce a plausible-looking density. The checks here pin
// each of those independently.
//
//   - Normalization. Numerical atomic orbitals are normalized, so
//     integral |phi_i(r)|^2 d3r = 1 on a quadrature grid. This catches the
//     radial spline, the Ylm normalization, and the unit conversion at once,
//     since each of those scales the integral by a different factor.
//   - Angular orthogonality. Orbitals on the same atom with different (l, m)
//     integrate to zero against each other.
//   - Compact support. The value is exactly zero beyond the tabulated cutoff,
//     which is what makes the covering-set search safe to truncate.

#include "../collocation.hpp"
#include "../../test_data_path.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <string>
#include <random>
#include <vector>

namespace {

using rsatb::calc::OrbitalCollocation;

std::string DiamondOrbitalFile() {
    return rsatb::test::data_dir() + "diamond/C_gga_7au_100Ry_2s2p1d.orb";
}

/// Cartesian quadrature over the cube enclosing an orbital's support.
///
/// A uniform mesh is enough here: the integrand is smooth away from the
/// nucleus and the radials of a numerical atomic orbital are themselves
/// tabulated on a finite grid, so refining past the tabulation buys nothing.
struct CubeGrid {
    std::vector<double> points;
    double weight = 0.0;

    CubeGrid(double half_width, int n_per_axis) {
        const double step = 2.0 * half_width / n_per_axis;
        weight = step * step * step;
        points.reserve(static_cast<size_t>(n_per_axis) * n_per_axis * n_per_axis * 3);
        for (int i = 0; i < n_per_axis; ++i) {
            for (int j = 0; j < n_per_axis; ++j) {
                for (int k = 0; k < n_per_axis; ++k) {
                    points.push_back(-half_width + (i + 0.5) * step);
                    points.push_back(-half_width + (j + 0.5) * step);
                    points.push_back(-half_width + (k + 0.5) * step);
                }
            }
        }
    }

    int n_point() const { return static_cast<int>(points.size() / 3); }
};

/// Overlap of two orbitals of one atom, by quadrature through the collocation.
///
/// Built by contracting the single-atom region against the matrix unit
/// E_ab + E_ba, so it exercises exactly the path the LDOS uses rather than a
/// private evaluation shortcut.
double QuadratureOverlap(const OrbitalCollocation& collocation,
                         vbcsr::atomic::AtomicData& atomic_data,
                         const rsatb::calc::OrbitalSet& set,
                         const CubeGrid& grid, int a, int b) {
    const int n_orb = set.n_orb();
    std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb, 0.0);
    if (a == b) {
        block[static_cast<size_t>(a) * n_orb + a] = 1.0;
    } else {
        block[static_cast<size_t>(a) * n_orb + b] = 0.5;
        block[static_cast<size_t>(b) * n_orb + a] = 0.5;
    }

    std::vector<double> values;
    collocation.ContractOntoPoints(atomic_data, set, block, grid.points, &values);

    double integral = 0.0;
    for (double value : values) integral += value;
    return integral * grid.weight;
}

/// A single carbon atom in a box large enough that no image interferes.
std::unique_ptr<vbcsr::atomic::AtomicData> BuildLoneAtom(double rcut) {
    const std::vector<double> pos = {0.0, 0.0, 0.0};
    const std::vector<int> z = {6};
    const double box = 4.0 * rcut + 10.0;
    const std::vector<double> cell = {box, 0.0, 0.0, 0.0, box, 0.0, 0.0, 0.0, box};
    const std::vector<bool> pbc = {false, false, false};
    const std::vector<double> cutoff = {rcut};
    const std::vector<int> type_norb = {13};
    return std::unique_ptr<vbcsr::atomic::AtomicData>(vbcsr::atomic::AtomicData::from_points(
        pos, z, cell, pbc, cutoff, type_norb, MPI_COMM_WORLD));
}

// A chain of carbons, spaced so their orbitals overlap. Enough atoms that an
// inertial partition has something to cut, unlike the lone-atom fixture. The
// periodic variant exercises the wrap paths; its cell exceeds twice the
// cutoff along every axis, so one image is always the whole answer.
std::unique_ptr<vbcsr::atomic::AtomicData> BuildChain(double rcut, int n,
                                                      bool periodic = false) {
    std::vector<double> pos;
    for (int i = 0; i < n; ++i) pos.insert(pos.end(), {2.0 * i, 0.0, 0.0});
    const std::vector<int> z(n, 6);
    const double lx = 2.0 * n + 4.0 * rcut, lt = 4.0 * rcut + 6.0;
    const std::vector<double> cell = {lx, 0.0, 0.0, 0.0, lt, 0.0, 0.0, 0.0, lt};
    const std::vector<double> cutoff = {rcut};
    const std::vector<int> type_norb = {13};
    return std::unique_ptr<vbcsr::atomic::AtomicData>(vbcsr::atomic::AtomicData::from_points(
        pos, z, cell, {periodic, periodic, periodic}, cutoff, type_norb, MPI_COMM_WORLD));
}

}  // namespace

TEST(CollocationTest, ReportsTheCutoffInAngstrom) {
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    // The file is a 7 Bohr orbital: 7 * 0.529177 = 3.704 Angstrom, the number
    // LcaoModel::DeriveBasis puts into the basis spec.
    EXPECT_NEAR(collocation.rcut_angstrom(0), 3.7042, 1e-3);
    EXPECT_EQ(collocation.nphi(0), 13);  // 2s 2p 1d
}

TEST(CollocationTest, OrbitalsAreNormalized) {
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);

    const std::vector<double> points = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, points, {}, {rcut},
                                                               MPI_COMM_WORLD);
    ASSERT_EQ(set.n_atom(), 1);
    ASSERT_EQ(set.n_orb(), 13);

    const CubeGrid grid(rcut * 1.05, 120);
    for (int orbital = 0; orbital < 13; ++orbital) {
        const double norm = QuadratureOverlap(collocation, *atomic_data, set, grid,
                                              orbital, orbital);
        EXPECT_NEAR(norm, 1.0, 2e-2) << "orbital " << orbital << " is not normalized; "
                                     << "check the radial units and the Ylm convention";
    }
}

TEST(CollocationTest, DistinctAngularChannelsAreOrthogonal) {
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);

    const std::vector<double> points = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, points, {}, {rcut},
                                                               MPI_COMM_WORLD);
    const CubeGrid grid(rcut * 1.05, 120);

    // Orbital 0 is the first s; orbitals 2, 3, 4 are the first p shell. An s-p
    // overlap vanishes by angular symmetry whatever the radials do, so a
    // non-zero value here means the harmonics are mislabelled.
    for (int p_orbital = 2; p_orbital <= 4; ++p_orbital) {
        const double overlap = QuadratureOverlap(collocation, *atomic_data, set, grid,
                                                 0, p_orbital);
        EXPECT_NEAR(overlap, 0.0, 2e-2) << "s-p overlap at orbital " << p_orbital;
    }
}

TEST(CollocationTest, ValuesVanishBeyondTheCutoff) {
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);

    const std::vector<double> points = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, points, {}, {rcut},
                                                               MPI_COMM_WORLD);
    const int n_orb = set.n_orb();

    // The identity block contracts to sum_i |phi_i(r)|^2, which is positive
    // wherever any orbital reaches and exactly zero past the cutoff.
    std::vector<std::complex<double>> identity(static_cast<size_t>(n_orb) * n_orb, 0.0);
    for (int i = 0; i < n_orb; ++i) identity[static_cast<size_t>(i) * n_orb + i] = 1.0;

    const std::vector<double> probes = {
        0.5, 0.0, 0.0,           // inside
        rcut * 0.9, 0.0, 0.0,    // inside, near the edge
        rcut * 1.10, 0.0, 0.0,   // outside
        0.0, rcut * 1.50, 0.0,   // outside
    };
    std::vector<double> values;
    collocation.ContractOntoPoints(*atomic_data, set, identity, probes, &values);

    EXPECT_GT(values[0], 0.0);
    EXPECT_GT(values[1], 0.0);
    EXPECT_DOUBLE_EQ(values[2], 0.0);
    EXPECT_DOUBLE_EQ(values[3], 0.0);
}

TEST(RealSpaceGridTest, TilesTheCellAndCoversEveryPointOnce) {
    const std::vector<double> cell = {10.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 6.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 0.5, 4);

    // Ceil, so the realised spacing is never coarser than asked.
    EXPECT_EQ(grid.shape[0], 20);
    EXPECT_EQ(grid.shape[1], 16);
    EXPECT_EQ(grid.shape[2], 12);
    EXPECT_EQ(grid.n_point(), 20u * 16u * 12u);

    // Boxes must tile the index space exactly: every point in one box.
    std::vector<int> seen(grid.n_point(), 0);
    for (int b = 0; b < grid.n_box(); ++b) {
        const auto box = grid.BoxAt(b);
        for (int ix = box.lo[0]; ix < box.hi[0]; ++ix) {
            for (int iy = box.lo[1]; iy < box.hi[1]; ++iy) {
                for (int iz = box.lo[2]; iz < box.hi[2]; ++iz) {
                    seen[grid.LinearIndex(ix, iy, iz)] += 1;
                }
            }
        }
    }
    for (int count : seen) EXPECT_EQ(count, 1);
}

TEST(RealSpaceGridTest, BoxSpheresContainTheirPoints) {
    // The reach test uses the bounding sphere, so a point outside its own
    // box's sphere would let an atom be skipped that actually reaches it.
    const std::vector<double> cell = {9.0, 0.0, 0.0, 3.0, 8.0, 0.0, 1.0, 2.0, 7.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 0.6, 5);

    std::vector<double> points;
    for (int b = 0; b < grid.n_box(); ++b) {
        const auto box = grid.BoxAt(b);
        grid.PointsOfBox(box, &points);
        for (size_t p = 0; p * 3 < points.size(); ++p) {
            double d2 = 0.0;
            for (int x = 0; x < 3; ++x) {
                const double d = points[3 * p + x] - box.center[x];
                d2 += d * d;
            }
            EXPECT_LE(std::sqrt(d2), box.radius + 1e-9);
        }
    }
}

TEST(RealSpaceGridTest, PointsOfBoxMatchTheGridIndexing) {
    const std::vector<double> cell = {6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 1.0, 3);

    std::vector<double> points;
    for (int b = 0; b < grid.n_box(); ++b) {
        const auto box = grid.BoxAt(b);
        grid.PointsOfBox(box, &points);
        size_t at = 0;
        for (int ix = box.lo[0]; ix < box.hi[0]; ++ix) {
            for (int iy = box.lo[1]; iy < box.hi[1]; ++iy) {
                for (int iz = box.lo[2]; iz < box.hi[2]; ++iz) {
                    double expected[3];
                    grid.PointAt(ix, iy, iz, expected);
                    for (int x = 0; x < 3; ++x) {
                        EXPECT_DOUBLE_EQ(points[3 * at + x], expected[x]);
                    }
                    ++at;
                }
            }
        }
    }
}

TEST(RealSpaceGridTest, DefaultBoxEdgeIsPhysicalNotAPointCount) {
    // The default targets ~2.4 A of box whatever the spacing. A fixed
    // 12-point edge would make 12 A boxes on a coarse 1 A property grid,
    // whose bounding spheres pull in thousands of atoms and blow up the
    // per-box buffers -- the edge must shrink with the point count.
    const std::vector<double> cell = {48.0, 0.0, 0.0, 0.0, 48.0, 0.0, 0.0, 0.0, 48.0};
    for (int d = 0; d < 3; ++d) {
        EXPECT_EQ(rsatb::calc::MakeRealSpaceGrid(cell, 0.2).box_edge[d], 12);
        EXPECT_EQ(rsatb::calc::MakeRealSpaceGrid(cell, 0.5).box_edge[d], 5);
        EXPECT_EQ(rsatb::calc::MakeRealSpaceGrid(cell, 1.0).box_edge[d], 3);
        EXPECT_EQ(rsatb::calc::MakeRealSpaceGrid(cell, 4.0).box_edge[d], 1);
        // Very fine grids stay capped so the Phi buffer stays a few thousand
        // points; an explicit edge still overrides everything.
        EXPECT_EQ(rsatb::calc::MakeRealSpaceGrid(cell, 0.05).box_edge[d], 12);
        EXPECT_EQ(rsatb::calc::MakeRealSpaceGrid(cell, 1.0, 7).box_edge[d], 7);
    }
}

TEST(CollocationTest, BoxContractionMatchesPointContraction) {
    // The box loop and the scattered-point loop must agree exactly: they use
    // the same kernel, and the box path only changes which atoms are gathered
    // and in what order the points arrive.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);

    const std::vector<double> seed_point = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, seed_point, {}, {rcut},
                                                               MPI_COMM_WORLD);
    const int n_orb = set.n_orb();

    std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb, 0.0);
    for (int i = 0; i < n_orb; ++i) {
        for (int j = 0; j < n_orb; ++j) {
            block[static_cast<size_t>(i) * n_orb + j] = 0.5 / (1.0 + std::abs(i - j));
        }
    }

    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.5, 4);
    std::vector<int> all_boxes(grid.n_box());
    std::iota(all_boxes.begin(), all_boxes.end(), 0);
    const auto domain = rsatb::calc::MakeGridDomain(grid, all_boxes);

    std::vector<double> by_box;
    collocation.ContractOntoBoxes(*atomic_data, set, block, grid, domain, &by_box);

    // Same points, fed through the scattered-point path.
    std::vector<double> points(grid.n_point() * 3);
    for (int ix = 0; ix < grid.shape[0]; ++ix) {
        for (int iy = 0; iy < grid.shape[1]; ++iy) {
            for (int iz = 0; iz < grid.shape[2]; ++iz) {
                grid.PointAt(ix, iy, iz, &points[3 * grid.LinearIndex(ix, iy, iz)]);
            }
        }
    }
    std::vector<double> by_point;
    collocation.ContractOntoPoints(*atomic_data, set, block, points, &by_point);

    // by_box is compact, in owned-box order; by_point is in grid order.
    ASSERT_EQ(static_cast<int64_t>(by_box.size()), domain.n_point());
    ASSERT_EQ(domain.n_point(), grid.n_point());
    for (int64_t k = 0; k < domain.n_point(); ++k) {
        const int64_t g = rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k);
        EXPECT_NEAR(by_box[k], by_point[g], 1e-12) << "grid point " << g;
    }
}

TEST(CollocationTest, RefusesATypeWithoutAnOrbitalFile) {
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    EXPECT_THROW(collocation.rcut_angstrom(1), std::runtime_error);
    EXPECT_THROW(OrbitalCollocation({}, MPI_COMM_WORLD), std::runtime_error);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

// --- grid ownership: bound to the atoms, and a partition ------------------
//
// These are the two properties the whole scaling argument rests on. If boxes
// were not a disjoint cover, the compact per-rank buffers would not
// concatenate into the global result and the missing allreduce would be a
// silent bug. If ownership were not derived from the atoms themselves, a
// rank's boxes would need orbitals from atoms it does not hold, and the work
// per rank would never fall no matter how many ranks were added.
//
// These drive ResolveGridOwnershipReference -- the rule stated independently
// of how it is communicated. The distributed resolution is checked against it
// separately, under mpirun, in test_rs_ldos.

namespace {

// Two well-separated clumps of carbons, and an owner per atom -- a stand-in
// for whatever partition the system arrived with. Ownership of a box is
// derived from THAT, so the test never needs a partitioner at all.
void TwoClumps(std::vector<double>* pos_out, std::vector<int>* owner_out, int n_rank) {
    std::vector<double> pos;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            pos.insert(pos.end(), {1.0 + 0.6 * i, 1.0 + 0.6 * j, 1.0});
            pos.insert(pos.end(), {8.0 + 0.6 * i, 8.0 + 0.6 * j, 8.0});
        }
    }
    *pos_out = pos;
    // Round-robin inside each clump: adversarial on purpose, since a
    // contiguous assignment would let a broken rule look right by accident.
    const int n_atom = static_cast<int>(pos.size() / 3);
    owner_out->resize(n_atom);
    for (int a = 0; a < n_atom; ++a) (*owner_out)[a] = a % n_rank;
}

std::vector<int> ReferenceOwners(const rsatb::calc::RealSpaceGrid& grid,
                                 const std::vector<double>& pos,
                                 const std::vector<int>& owners, int n_rank,
                                 double rcut, const std::vector<double>& cell) {
    const std::vector<int> types(owners.size(), 0);
    return rsatb::calc::ResolveGridOwnershipReference(
        grid, pos, types, owners, {1}, {rcut}, cell, {false, false, false}, n_rank);
}

}  // namespace

TEST(GridPartitionTest, AssignsEveryPointExactlyOnce) {
    // The property the whole no-reduction output layout rests on: the boxes a
    // rank owns, concatenated across ranks, cover every grid point exactly
    // once. With ownership settled per box by a single referee this is true by
    // construction -- the test is here to keep it true.
    const std::vector<double> cell = {10.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 8.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 0.8, 3);

    for (int n_rank : {2, 3, 4, 8, 16}) {
        std::vector<double> pos;
        std::vector<int> owners;
        TwoClumps(&pos, &owners, n_rank);
        const auto owner_of = ReferenceOwners(grid, pos, owners, n_rank, 2.0, cell);

        std::vector<int> seen(grid.n_point(), 0);
        int64_t total = 0;
        for (int r = 0; r < n_rank; ++r) {
            std::vector<int> mine;
            for (int b = 0; b < grid.n_box(); ++b) {
                if (owner_of[b] == r) mine.push_back(b);
            }
            const auto domain = rsatb::calc::MakeGridDomain(grid, mine);
            total += domain.n_point();
            for (int64_t k = 0; k < domain.n_point(); ++k) {
                seen[rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k)] += 1;
            }
        }
        EXPECT_EQ(total, grid.n_point()) << "n_rank = " << n_rank;
        for (int64_t i = 0; i < grid.n_point(); ++i) {
            ASSERT_EQ(seen[i], 1) << "point " << i << " at n_rank = " << n_rank;
        }
    }
}

TEST(GridPartitionTest, BoxesGoToARankThatOwnsAnAtomReachingThem) {
    // The locality guarantee, and the reason the halo stays small: whenever
    // any orbital reaches a box, the box goes to a rank that owns one of those
    // atoms -- never to a bystander. This is what the old cut-plane scheme
    // could only approximate, because ParMETIS moved atoms after the planes
    // were drawn.
    const std::vector<double> cell = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 0.8, 3);
    const double rcut = 2.0;

    for (int n_rank : {2, 3, 4, 8, 16}) {
        std::vector<double> pos;
        std::vector<int> owners;
        TwoClumps(&pos, &owners, n_rank);
        const auto owner_of = ReferenceOwners(grid, pos, owners, n_rank, rcut, cell);
        const int n_atom = static_cast<int>(pos.size() / 3);

        int n_reached = 0;
        for (int b = 0; b < grid.n_box(); ++b) {
            const auto box = grid.BoxAt(b);
            bool owner_reaches = false, anyone_reaches = false;
            for (int a = 0; a < n_atom; ++a) {
                double d2 = 0.0;
                for (int x = 0; x < 3; ++x) {
                    const double d = pos[3 * a + x] - box.center[x];
                    d2 += d * d;
                }
                if (std::sqrt(d2) >= rcut + box.radius) continue;
                anyone_reaches = true;
                if (owners[a] == owner_of[b]) owner_reaches = true;
            }
            if (!anyone_reaches) continue;  // vacuum: any owner will do
            ++n_reached;
            EXPECT_TRUE(owner_reaches)
                << "box " << b << " at n_rank = " << n_rank
                << " went to rank " << owner_of[b] << ", which owns no atom reaching it";
        }
        EXPECT_GT(n_reached, 0) << "no box was reached at all; the test proved nothing";
    }
}

TEST(GridPartitionTest, ResolutionIsDeterministic) {
    // Every rank resolves ownership from claims that may arrive in any order,
    // so the rule has to be a total order or the ranks disagree about who owns
    // what -- and disagreement means points written twice or not at all.
    const std::vector<double> cell = {7.0, 1.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 7.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 0.7, 4);
    std::vector<double> pos;
    std::vector<int> owners;
    TwoClumps(&pos, &owners, 4);
    EXPECT_EQ(ReferenceOwners(grid, pos, owners, 4, 2.0, cell),
              ReferenceOwners(grid, pos, owners, 4, 2.0, cell));
}

TEST(GridPartitionTest, SurvivesMoreRanksThanAtoms) {
    // 32 atoms across 64 ranks: half the ranks own nothing and can claim
    // nothing. The unclaimed boxes must still land somewhere exactly once,
    // which is what the referee fallback is for.
    const std::vector<double> cell = {6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0};
    const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, 1.0, 3);
    std::vector<double> pos;
    std::vector<int> owners;
    TwoClumps(&pos, &owners, 64);
    const auto owner_of = ReferenceOwners(grid, pos, owners, 64, 2.0, cell);

    std::vector<int> seen(grid.n_point(), 0);
    for (int r = 0; r < 64; ++r) {
        std::vector<int> mine;
        for (int b = 0; b < grid.n_box(); ++b) {
            if (owner_of[b] == r) mine.push_back(b);
        }
        const auto domain = rsatb::calc::MakeGridDomain(grid, mine);
        for (int64_t k = 0; k < domain.n_point(); ++k) {
            seen[rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k)] += 1;
        }
    }
    for (int64_t i = 0; i < grid.n_point(); ++i) ASSERT_EQ(seen[i], 1) << "point " << i;
}

TEST(GridPartitionTest, CoverIsExactWhateverTheBoxToRegionRatio) {
    // The old cut-plane scheme carried a resolution condition: a region finer
    // than a box contained no box centre and got no grid. Claiming removes the
    // condition -- an atom claims whatever it reaches, however coarse the
    // boxes -- but the cover must be exact in BOTH regimes regardless, since
    // the compact buffers are concatenated with no reduction.
    const std::vector<double> cell = {6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0};
    std::vector<double> pos;
    std::vector<int> owners;
    TwoClumps(&pos, &owners, 8);                      // ~0.9 A clump spacing

    for (double spacing : {0.5, 0.25}) {              // 2 A boxes, then 0.75 A
        const auto grid = rsatb::calc::MakeRealSpaceGrid(cell, spacing, 4);
        const auto owner_of = ReferenceOwners(grid, pos, owners, 8, 2.0, cell);
        std::vector<int> seen(grid.n_point(), 0);
        for (int r = 0; r < 8; ++r) {
            std::vector<int> mine;
            for (int b = 0; b < grid.n_box(); ++b) {
                if (owner_of[b] == r) mine.push_back(b);
            }
            const auto domain = rsatb::calc::MakeGridDomain(grid, mine);
            for (int64_t k = 0; k < domain.n_point(); ++k) {
                seen[rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k)] += 1;
            }
        }
        for (int64_t i = 0; i < grid.n_point(); ++i) {
            ASSERT_EQ(seen[i], 1) << "point " << i << " at spacing " << spacing;
        }
    }
}

namespace {

// GridOwnership for a SIMULATED rank, built from an explicit owner-per-box
// map. The distributed resolver needs real MPI ranks; this lets a serial test
// still drive the demand path for several ranks, which is where the
// interesting failures live.
rsatb::calc::GridOwnership SimulatedOwnership(const rsatb::calc::RealSpaceGrid& grid,
                                              vbcsr::atomic::AtomicData& atomic_data,
                                              const std::vector<int>& owner_of,
                                              const std::vector<double>& type_rcut,
                                              int rank) {
    rsatb::calc::GridOwnership ownership;
    for (int b = 0; b < grid.n_box(); ++b) {
        if (owner_of[b] == rank) ownership.boxes.push_back(b);
    }
    std::vector<int> reached;
    rsatb::calc::detail::ForEachBoxOfOwnedAtoms(
        atomic_data, grid, type_rcut,
        [&](int, int box, double) { reached.push_back(box); });
    std::sort(reached.begin(), reached.end());
    reached.erase(std::unique(reached.begin(), reached.end()), reached.end());
    for (int b : reached) {
        ownership.reached_box.push_back(b);
        ownership.reached_owner.push_back(owner_of[b]);
    }
    return ownership;
}

// Owner per box for a chain, from the rule, given an owner per atom.
std::vector<int> ChainOwners(const rsatb::calc::RealSpaceGrid& grid,
                             vbcsr::atomic::AtomicData& atomic_data,
                             const std::vector<int>& atom_owner, int n_rank,
                             double rcut, int norb) {
    const int n_atom = static_cast<int>(atom_owner.size());
    std::vector<double> pos(3 * n_atom);
    for (int a = 0; a < n_atom; ++a) {
        atomic_data.get_pos(a, &pos[3 * a], &pos[3 * a + 1], &pos[3 * a + 2]);
    }
    return rsatb::calc::ResolveGridOwnershipReference(
        grid, pos, std::vector<int>(n_atom, 0), atom_owner, {norb}, {rcut},
        atomic_data.cell, atomic_data.pbc, n_rank);
}

}  // namespace

TEST(CollocationTest, LocalOrbitalSetHoldsEveryAtomReachingItsBoxes) {
    // The no-communication claim: a rank's local set must contain every atom
    // the replicated set would have contributed for the same boxes. If it
    // missed one, the contraction would silently drop that orbital's weight
    // rather than fail, so this compares against the gathered answer directly.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);
    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.5, 4);

    std::vector<int> all_boxes(grid.n_box());
    std::iota(all_boxes.begin(), all_boxes.end(), 0);
    const auto domain = rsatb::calc::MakeGridDomain(grid, all_boxes);

    const auto local = rsatb::calc::BuildLocalOrbitalSet(*atomic_data, grid, domain, {rcut});

    // The replicated reference over the same box centres.
    std::vector<double> centers, radii;
    for (int b = 0; b < grid.n_box(); ++b) {
        const auto box = grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto gathered = rsatb::calc::BuildOrbitalSetForSpheres(
        *atomic_data, centers, radii, {rcut}, MPI_COMM_WORLD);

    EXPECT_EQ(local.atoms, gathered.atoms);
    EXPECT_EQ(local.n_orb(), gathered.n_orb());
}

TEST(CollocationTest, LocalOrbitalSetIsEmptyForBoxesNoOrbitalReaches) {
    // A rank whose region is vacuum owns boxes that nothing reaches. That is
    // legitimate and must not throw -- unlike the replicated builder, which
    // treats an empty result as a caller error because it means the requested
    // probe region found nothing anywhere.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);
    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.5, 4);

    // One box, and a cutoff far too short for the atom to reach it.
    const auto domain = rsatb::calc::MakeGridDomain(grid, {grid.n_box() - 1});
    const auto set = rsatb::calc::BuildLocalOrbitalSet(*atomic_data, grid, domain, {1e-6});
    EXPECT_EQ(set.n_atom(), 0);
    EXPECT_EQ(set.n_orb(), 0);
}

// --- route (5.3b): square-on-grid ----------------------------------------

TEST(CollocationTest, SquaringOneVectorMatchesItsOuterProductBlock) {
    // The identity the whole route rests on:
    //   sum_ij (c_i c_j) phi_i phi_j == ( sum_i c_i phi_i )^2
    // so for a single vector the two routes must agree to machine precision.
    // No sampling is involved, which is what makes this a correctness test
    // rather than a convergence test.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);

    const std::vector<double> seed_point = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, seed_point, {}, {rcut},
                                                            MPI_COMM_WORLD);
    const int n_orb = set.n_orb();

    std::vector<double> c(n_orb);
    for (int i = 0; i < n_orb; ++i) c[i] = std::cos(0.7 * i + 0.3) / (1.0 + 0.1 * i);

    std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb);
    for (int i = 0; i < n_orb; ++i) {
        for (int j = 0; j < n_orb; ++j) block[static_cast<size_t>(i) * n_orb + j] = c[i] * c[j];
    }

    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.5, 4);
    std::vector<int> all(grid.n_box());
    std::iota(all.begin(), all.end(), 0);
    const auto domain = rsatb::calc::MakeGridDomain(grid, all);

    std::vector<double> by_block, by_square;
    collocation.ContractOntoBoxes(*atomic_data, set, block, grid, domain, &by_block);
    collocation.SquareVectorsOntoBoxes(*atomic_data, set, c, 1, grid, domain, &by_square);

    ASSERT_EQ(by_block.size(), by_square.size());
    double peak = 0.0;
    for (double v : by_block) peak = std::max(peak, std::abs(v));
    ASSERT_GT(peak, 0.0) << "the reference is identically zero; the test proves nothing";
    for (size_t i = 0; i < by_block.size(); ++i) {
        EXPECT_NEAR(by_square[i], by_block[i], 1e-12 * peak) << "point " << i;
    }
}

TEST(CollocationTest, SquaringManyVectorsMatchesTheAveragedOuterProduct) {
    // With n_vec vectors the route returns mean_v |phi.c_v|^2, which is the
    // block route fed mean_v c_v c_v^T. Still exact -- averaging commutes with
    // both contractions -- so this checks the multi-vector bookkeeping (the
    // GEMM layout and the 1/n_vec) rather than any statistics.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);

    const std::vector<double> seed_point = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, seed_point, {}, {rcut},
                                                            MPI_COMM_WORLD);
    const int n_orb = set.n_orb();
    const int n_vec = 5;

    // Row-major (n_orb, n_vec).
    std::vector<double> c(static_cast<size_t>(n_orb) * n_vec);
    for (int i = 0; i < n_orb; ++i) {
        for (int v = 0; v < n_vec; ++v) {
            c[static_cast<size_t>(i) * n_vec + v] = std::sin(0.4 * i + 1.1 * v) / (1.0 + 0.05 * i);
        }
    }

    std::vector<std::complex<double>> block(static_cast<size_t>(n_orb) * n_orb, 0.0);
    for (int i = 0; i < n_orb; ++i) {
        for (int j = 0; j < n_orb; ++j) {
            double acc = 0.0;
            for (int v = 0; v < n_vec; ++v) {
                acc += c[static_cast<size_t>(i) * n_vec + v] * c[static_cast<size_t>(j) * n_vec + v];
            }
            block[static_cast<size_t>(i) * n_orb + j] = acc / n_vec;
        }
    }

    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.5, 4);
    std::vector<int> all(grid.n_box());
    std::iota(all.begin(), all.end(), 0);
    const auto domain = rsatb::calc::MakeGridDomain(grid, all);

    std::vector<double> by_block, by_square;
    collocation.ContractOntoBoxes(*atomic_data, set, block, grid, domain, &by_block);
    collocation.SquareVectorsOntoBoxes(*atomic_data, set, c, n_vec, grid, domain, &by_square);

    double peak = 0.0;
    for (double v : by_block) peak = std::max(peak, std::abs(v));
    ASSERT_GT(peak, 0.0);
    for (size_t i = 0; i < by_block.size(); ++i) {
        EXPECT_NEAR(by_square[i], by_block[i], 1e-12 * peak) << "point " << i;
    }
}

TEST(CollocationTest, SquaredDensityIsNonNegativeEverywhere) {
    // Route (5.3b) is a sum of squares, so unlike the block route it cannot
    // produce a negative density from any coefficient input. Worth pinning:
    // it is the one qualitative guarantee the square-on-grid form buys.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);
    const std::vector<double> seed_point = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, seed_point, {}, {rcut},
                                                            MPI_COMM_WORLD);
    const int n_orb = set.n_orb();
    const int n_vec = 3;

    std::mt19937 rng(12345);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> c(static_cast<size_t>(n_orb) * n_vec);
    for (auto& v : c) v = gauss(rng);

    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.5, 4);
    std::vector<int> all(grid.n_box());
    std::iota(all.begin(), all.end(), 0);
    const auto domain = rsatb::calc::MakeGridDomain(grid, all);

    std::vector<double> values;
    collocation.SquareVectorsOntoBoxes(*atomic_data, set, c, n_vec, grid, domain, &values);
    for (size_t i = 0; i < values.size(); ++i) EXPECT_GE(values[i], 0.0) << "point " << i;
}

// --- exact-zero row trim --------------------------------------------------

TEST(CollocationTest, TrimDropsExactlyTheAllZeroRowsAndCompacts) {
    // Exact zeros only: a row of 1e-300 is kept, because the evaluator returns
    // a true 0.0 outside a cutoff and anything else is real weight. Getting
    // this wrong would silently truncate the density instead of failing.
    std::vector<double> phi = {
        0.0, 0.0, 0.0,       // row 0: drop
        1.0, 2.0, 3.0,       // row 1: keep
        0.0, 0.0, 0.0,       // row 2: drop
        0.0, 1e-300, 0.0,    // row 3: keep -- tiny, not zero
        4.0, 5.0, 6.0,       // row 4: keep
    };
    std::vector<int> kept;
    const int n_kept = rsatb::calc::detail::TrimZeroRows(5, 3, &phi, &kept);

    ASSERT_EQ(n_kept, 3);
    EXPECT_EQ(kept, (std::vector<int>{1, 3, 4}));
    // Kept rows must be compacted to the front, in order, values intact.
    EXPECT_DOUBLE_EQ(phi[0], 1.0);
    EXPECT_DOUBLE_EQ(phi[2], 3.0);
    EXPECT_DOUBLE_EQ(phi[4], 1e-300);
    EXPECT_DOUBLE_EQ(phi[6], 4.0);
    EXPECT_DOUBLE_EQ(phi[8], 6.0);
}

TEST(CollocationTest, TrimActuallyFiresOnBoxesOrbitalsOnlyClip) {
    // The trim is only worth its scan if it removes rows in a real geometry.
    // A lone atom in a large cell is the case it targets: most boxes are
    // vacuum, and the boxes at the edge of the atom's reach are part covered.
    // Without this the trim could be a silent no-op and every other test would
    // still pass.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    auto atomic_data = BuildLoneAtom(rcut);
    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.0, 4);

    const std::vector<double> seed_point = {0.0, 0.0, 0.0};
    const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*atomic_data, seed_point, {}, {rcut},
                                                            MPI_COMM_WORLD);
    const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data->cell.data());

    int64_t total = 0, trimmed = 0;
    std::vector<double> points, phi, relative;
    std::vector<int> active, kept;
    for (int b = 0; b < grid.n_box(); ++b) {
        const auto box = grid.BoxAt(b);
        const auto local = collocation.ViewForSphereForTest(*atomic_data, set, box.center,
                                                            box.radius, cell_inv);
        const int n_local_orb = static_cast<int>(local.orbitals.size());
        if (n_local_orb == 0) continue;  // skipped before the trim is reached
        grid.PointsOfBox(box, &points);
        phi.assign(static_cast<size_t>(box.n_point()) * n_local_orb, 0.0);
        collocation.EvaluatePhiForTest(*atomic_data, set, local, points, box.n_point(),
                                       cell_inv, &relative, &active, phi.data());
        const int n_kept = rsatb::calc::detail::TrimZeroRows(box.n_point(), n_local_orb,
                                                             &phi, &kept);
        total += box.n_point();
        trimmed += box.n_point() - n_kept;
    }
    ASSERT_GT(total, 0);
    EXPECT_GT(trimmed, 0) << "the trim removed nothing; it is not earning its scan";
}

// --- end to end: the distributed map equals the serial one ----------------

TEST(CollocationTest, DistributedMapAssemblesIntoTheSerialMap) {
    // Everything from steps 1-5 acting together, against the answer it must
    // reproduce. Each simulated rank derives its own boxes from the atom cut,
    // builds its OWN local orbital set with no gather, contracts only what it
    // owns with the trim active, and writes a compact buffer. Scattering those
    // buffers back by global index has to rebuild, bit for bit, the map one
    // rank produces holding every box and the full replicated set.
    //
    // The individual properties are covered elsewhere; what is new here is
    // that they compose. A local set missing a boundary atom, an off-by-one in
    // the compact offsets, or a trim that dropped a row with real weight would
    // each leave the parts correct and the whole wrong.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    const int n_atom = 12;
    auto atomic_data = BuildChain(rcut, n_atom);
    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.2, 4);

    // Serial reference: one domain, replicated set over every box.
    std::vector<double> centers, radii;
    for (int b = 0; b < grid.n_box(); ++b) {
        const auto box = grid.BoxAt(b);
        centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
        radii.push_back(box.radius);
    }
    const auto full_set = rsatb::calc::BuildOrbitalSetForSpheres(
        *atomic_data, centers, radii, {rcut}, MPI_COMM_WORLD);

    std::vector<int> all(grid.n_box());
    std::iota(all.begin(), all.end(), 0);
    const auto full_domain = rsatb::calc::MakeGridDomain(grid, all);

    // One coefficient vector per atom, keyed by GLOBAL atom index so the
    // simulated ranks agree on the values without exchanging them.
    const int n_vec = 2;
    auto coeff_for = [&](const rsatb::calc::OrbitalSet& set) {
        std::vector<double> c(static_cast<size_t>(set.n_orb()) * n_vec, 0.0);
        for (int a = 0; a < set.n_atom(); ++a) {
            for (int o = set.orb_offset[a]; o < set.orb_offset[a + 1]; ++o) {
                const int local_o = o - set.orb_offset[a];
                for (int v = 0; v < n_vec; ++v) {
                    c[static_cast<size_t>(o) * n_vec + v] =
                        std::sin(0.31 * set.atoms[a] + 0.17 * local_o + 0.9 * v);
                }
            }
        }
        return c;
    };

    std::vector<double> serial;
    collocation.SquareVectorsOntoBoxes(*atomic_data, full_set, coeff_for(full_set), n_vec,
                                       grid, full_domain, &serial);

    double peak = 0.0;
    for (double v : serial) peak = std::max(peak, std::abs(v));
    ASSERT_GT(peak, 0.0) << "the reference map is identically zero";

    for (int n_part : {2, 3, 4, 8}) {
        std::vector<int> atom_owner(n_atom);
        for (int a = 0; a < n_atom; ++a) atom_owner[a] = a % n_part;
        const auto owner_of =
            ChainOwners(grid, *atomic_data, atom_owner, n_part, rcut, 13);

        std::vector<double> assembled(grid.n_point(), 0.0);
        std::vector<int> written(grid.n_point(), 0);
        int n_strict_subset = 0;
        for (int r = 0; r < n_part; ++r) {
            const auto mine =
                SimulatedOwnership(grid, *atomic_data, owner_of, {rcut}, r);
            const auto domain = rsatb::calc::MakeGridDomain(grid, mine.boxes);
            if (domain.n_box() == 0) continue;
            // The local set -- no gather, and generally not the full set.
            const auto local_set =
                rsatb::calc::BuildLocalOrbitalSet(*atomic_data, grid, domain, {rcut});
            if (local_set.n_atom() < full_set.n_atom()) ++n_strict_subset;
            if (local_set.n_orb() == 0) {
                for (int64_t k = 0; k < domain.n_point(); ++k) {
                    written[rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k)] += 1;
                }
                continue;
            }
            std::vector<double> values;
            collocation.SquareVectorsOntoBoxes(*atomic_data, local_set, coeff_for(local_set),
                                               n_vec, grid, domain, &values);
            for (int64_t k = 0; k < domain.n_point(); ++k) {
                const int64_t g = rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k);
                assembled[g] = values[k];
                written[g] += 1;
            }
        }

        // If every rank saw the whole set, the local-set path was never
        // exercised and the agreement above would prove nothing.
        EXPECT_GT(n_strict_subset, 0)
            << "no rank had a proper subset of the atoms at n_part = " << n_part;
        for (int64_t i = 0; i < grid.n_point(); ++i) {
            ASSERT_EQ(written[i], 1) << "point " << i << " at n_part = " << n_part;
        }
        for (int64_t k = 0; k < full_domain.n_point(); ++k) {
            const int64_t g = rsatb::calc::GlobalIndexOfOwnedPoint(grid, full_domain, k);
            EXPECT_NEAR(assembled[g], serial[k], 1e-11 * peak)
                << "point " << g << " at n_part = " << n_part;
        }
    }
}

// --- coverage: the demand side proves the local sets sufficient -----------

TEST(GridPartitionTest, DemandEnumerationMatchesTheLocalSetsExactly) {
    // BoxDemandsOfOwnedAtoms walks a WINDOW of the box lattice around each
    // atom; BuildLocalOrbitalSet scans every owned box against every held
    // atom. Same overlap predicate, opposite sides. On a serial fixture the
    // candidate pool is complete, so the atoms demanded of a rank must equal
    // its local set exactly -- a window sized too small, a wrap handled
    // wrong, or a bound that fails under the periodic images would each show
    // up as a set difference. Both boundary conditions are run because they
    // take different code paths through the window arithmetic.
    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    const int n_atom = 12;

    for (bool periodic : {false, true}) {
        auto atomic_data = BuildChain(rcut, n_atom, periodic);
        const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.2, 4);

        std::vector<double> pos(3 * n_atom);
        for (int a = 0; a < n_atom; ++a) {
            pos[3 * a + 0] = atomic_data->x[a];
            pos[3 * a + 1] = atomic_data->y[a];
            pos[3 * a + 2] = atomic_data->z[a];
        }

        for (int n_part : {2, 3, 4, 8}) {
            std::vector<int> atom_owner(n_atom);
            for (int a = 0; a < n_atom; ++a) atom_owner[a] = a % n_part;
            const auto owner_of =
                ChainOwners(grid, *atomic_data, atom_owner, n_part, rcut, 13);
            const auto ownership =
                SimulatedOwnership(grid, *atomic_data, owner_of, {rcut}, 0);
            const auto demands =
                rsatb::calc::BoxDemandsOfOwnedAtoms(*atomic_data, grid, ownership, {rcut});

            for (int r = 0; r < n_part; ++r) {
                std::vector<int> demanded;
                for (const auto& demand : demands) {
                    if (demand.first == r) demanded.push_back(demand.second);
                }
                std::sort(demanded.begin(), demanded.end());

                const auto mine =
                    SimulatedOwnership(grid, *atomic_data, owner_of, {rcut}, r);
                const auto domain = rsatb::calc::MakeGridDomain(grid, mine.boxes);
                const auto local_set =
                    rsatb::calc::BuildLocalOrbitalSet(*atomic_data, grid, domain, {rcut});
                EXPECT_EQ(demanded, local_set.atoms)
                    << "rank " << r << " at n_part = " << n_part
                    << (periodic ? " (periodic)" : " (open)");

                // The demand-BUILT set must reproduce the local one field for
                // field: on a serial fixture the candidate pool is complete,
                // so any difference is a bug in the exchange-side packing.
                const auto demand_set = rsatb::calc::BuildOrbitalSetFromDemands(
                    *atomic_data, grid, mine, {rcut}, r, MPI_COMM_NULL);
                EXPECT_EQ(demand_set.atoms, local_set.atoms);
                EXPECT_EQ(demand_set.atom_type, local_set.atom_type);
                EXPECT_EQ(demand_set.atom_norb, local_set.atom_norb);
                EXPECT_EQ(demand_set.positions, local_set.positions);
                EXPECT_EQ(demand_set.orb_offset, local_set.orb_offset);
            }
        }
    }
}

TEST(GridPartitionTest, CoverageValidatorPassesAndCatchesATruncatedSet) {
    // The MPI wrapper around the demands. Serial communicator: every demand
    // is this rank's own, so the full local set must pass, and a set with one
    // reaching atom removed must be reported as insufficient -- that is the
    // "silently dropped contribution" the validator exists to make loud.
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "fixture is serial; the wrapper is exercised at np=1";

    OrbitalCollocation collocation({DiamondOrbitalFile()}, MPI_COMM_WORLD);
    const double rcut = collocation.rcut_angstrom(0);
    const int n_atom = 12;
    auto atomic_data = BuildChain(rcut, n_atom);
    const auto grid = rsatb::calc::MakeRealSpaceGrid(atomic_data->cell, 1.2, 4);

    std::vector<double> pos(3 * n_atom);
    for (int a = 0; a < n_atom; ++a) {
        pos[3 * a + 0] = atomic_data->x[a];
        pos[3 * a + 1] = atomic_data->y[a];
        pos[3 * a + 2] = atomic_data->z[a];
    }
    const auto ownership =
        rsatb::calc::ResolveGridOwnership(*atomic_data, grid, {rcut}, MPI_COMM_WORLD);
    const auto domain = rsatb::calc::MakeGridDomain(grid, ownership.boxes);
    const auto set = rsatb::calc::BuildLocalOrbitalSet(*atomic_data, grid, domain, {rcut});
    ASSERT_EQ(set.n_atom(), n_atom);

    EXPECT_NO_THROW(rsatb::calc::ValidateLocalOrbitalCoverage(
        *atomic_data, grid, ownership, set, {rcut}, MPI_COMM_WORLD));

    auto truncated = set;
    truncated.atoms.erase(truncated.atoms.begin() + 5);
    EXPECT_THROW(rsatb::calc::ValidateLocalOrbitalCoverage(
                     *atomic_data, grid, ownership, truncated, {rcut}, MPI_COMM_WORLD),
                 std::runtime_error);
}
