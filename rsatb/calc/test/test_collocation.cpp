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
    for (const auto& box : grid.boxes) {
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
    for (const auto& box : grid.boxes) {
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
    for (const auto& box : grid.boxes) {
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
    std::vector<int> all_boxes(grid.boxes.size());
    std::iota(all_boxes.begin(), all_boxes.end(), 0);

    std::vector<double> by_box;
    collocation.ContractOntoBoxes(*atomic_data, set, block, grid, all_boxes, &by_box);

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

    ASSERT_EQ(by_box.size(), by_point.size());
    for (size_t i = 0; i < by_box.size(); ++i) {
        EXPECT_NEAR(by_box[i], by_point[i], 1e-12) << "grid point " << i;
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
