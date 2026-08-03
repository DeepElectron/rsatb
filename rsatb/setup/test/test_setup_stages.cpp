// Unit tests for the setup stages.
//
// The point of splitting these out of OperatorSet::initialize is that they can
// now be exercised on their own: the partition without building any operators,
// and the bounds estimate without building a model or an AtomicData.

#include "../kpoint_partition.hpp"
#include "../spectral_bounds.hpp"

#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

namespace setup = rsatb::setup;

namespace {

std::vector<setup::KPoint> MakeKPoints(int count) {
    std::vector<setup::KPoint> kpoints;
    for (int i = 0; i < count; ++i) {
        kpoints.push_back({0.1 * i, 0.0, 0.0});
    }
    return kpoints;
}

std::vector<int> LocalIndices(const setup::KPointPartition& partition) {
    std::vector<int> indices;
    for (const auto& entry : partition.local) indices.push_back(entry.first);
    return indices;
}

}  // namespace

TEST(KPointPartitionTest, SingleRankOwnsEverything) {
    const auto partition = setup::PartitionKPoints(MakeKPoints(5), {1, 0, 1});
    EXPECT_EQ(partition.global_nk, 5);
    EXPECT_EQ(LocalIndices(partition), (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(KPointPartitionTest, EveryKPointIsOwnedExactlyOnce) {
    // Sweep rank layouts: each k-point must land in exactly one colour, and
    // every rank of a colour must see the same set.
    for (int world_size : {1, 2, 3, 4, 8}) {
        for (int npar = 1; npar <= world_size; ++npar) {
            const auto kpoints = MakeKPoints(7);
            std::vector<std::vector<int>> per_colour;
            std::set<int> covered;

            for (int rank = 0; rank < world_size; ++rank) {
                const auto partition = setup::PartitionKPoints(kpoints, {npar, rank, world_size});
                ASSERT_EQ(partition.global_nk, 7);
                const auto indices = LocalIndices(partition);

                const int colour = (rank * std::min(npar, world_size)) / world_size;
                if (static_cast<int>(per_colour.size()) <= colour) {
                    per_colour.resize(static_cast<size_t>(colour) + 1);
                    per_colour[static_cast<size_t>(colour)] = indices;
                } else if (!per_colour[static_cast<size_t>(colour)].empty() || !indices.empty()) {
                    // Ranks sharing a colour cooperate on the same k-points.
                    EXPECT_EQ(per_colour[static_cast<size_t>(colour)], indices)
                        << "world_size=" << world_size << " npar=" << npar << " rank=" << rank;
                }
                for (int index : indices) covered.insert(index);
            }

            EXPECT_EQ(covered.size(), 7u)
                << "world_size=" << world_size << " npar=" << npar << " left a k-point unowned";
        }
    }
}

TEST(KPointPartitionTest, NparAboveWorldSizeIsClamped) {
    // Asking for more sub-communicators than ranks must not drop k-points.
    const auto kpoints = MakeKPoints(4);
    std::set<int> covered;
    for (int rank = 0; rank < 2; ++rank) {
        for (int index : LocalIndices(setup::PartitionKPoints(kpoints, {16, rank, 2}))) {
            covered.insert(index);
        }
    }
    EXPECT_EQ(covered.size(), 4u);
}

TEST(KPointPartitionTest, RejectsRankOutsideTheCommunicator) {
    EXPECT_THROW(setup::PartitionKPoints(MakeKPoints(2), {1, 4, 4}), std::runtime_error);
    EXPECT_THROW(setup::PartitionKPoints(MakeKPoints(2), {1, 0, 0}), std::runtime_error);
}

TEST(SpectralBoundsTest, ExplicitBoundsAreAdoptedAndValidated) {
    const auto bounds = setup::MakeExplicitBounds(-3.0, 4.0);
    EXPECT_TRUE(bounds.ready);
    EXPECT_DOUBLE_EQ(bounds.emin, -3.0);
    EXPECT_DOUBLE_EQ(bounds.emax, 4.0);

    EXPECT_THROW(setup::MakeExplicitBounds(4.0, -3.0), std::runtime_error);
    EXPECT_THROW(setup::MakeExplicitBounds(1.0, 1.0), std::runtime_error);
    EXPECT_THROW(
        setup::MakeExplicitBounds(std::numeric_limits<double>::quiet_NaN(), 1.0),
        std::runtime_error);
}

TEST(SpectralBoundsTest, NoKPointsWithoutCommunicatorsIsAnError) {
    // With nothing to estimate from, the reduction cannot produce a finite
    // window, and saying so beats returning a silently wrong default.
    EXPECT_THROW(
        setup::EstimateSpectralBounds({}, setup::BoundsOptions{}, setup::BoundsComms{}),
        std::runtime_error);
}

TEST(SpectralBoundsTest, RejectsNegativeTolerances) {
    setup::BoundsOptions options;
    options.zero_tol = -1.0;
    EXPECT_THROW(
        setup::EstimateSpectralBounds({}, options, setup::BoundsComms{}), std::runtime_error);

    options = setup::BoundsOptions{};
    options.max_matvecs = -1;
    EXPECT_THROW(
        setup::EstimateSpectralBounds({}, options, setup::BoundsComms{}), std::runtime_error);
}

TEST(SpectralBoundsTest, NullHamiltonianViewsContributeNothing) {
    // A view without a Hamiltonian is skipped rather than crashing; with only
    // such views there is nothing to reduce, which is reported.
    std::vector<setup::BoundsKPointView> views(2);
    views[0].kpoint = {0.0, 0.0, 0.0};
    views[1].kpoint = {0.5, 0.0, 0.0};
    EXPECT_THROW(
        setup::EstimateSpectralBounds(views, setup::BoundsOptions{}, setup::BoundsComms{}),
        std::runtime_error);
}

// The neighbour search reports shifts against the positions it was given, not
// against an internally wrapped copy. Two spellings of the same crystal must
// therefore yield the same set of bonds, with the shifts differing by exactly
// the lattice translation between the spellings.
namespace {

constexpr double kBond = 1.42;
const double kLatticeA = std::sqrt(3.0) * kBond;

std::vector<double> GrapheneCell() {
    return {kLatticeA, 0.0, 0.0,
            0.5 * kLatticeA, 1.5 * kBond, 0.0,
            0.0, 0.0, 20.0};
}

/// Bond vectors of atom 0, sorted, so two graphs can be compared directly.
std::vector<std::array<double, 3>> BondVectorsOfAtomZero(const std::vector<double>& pos) {
    const std::vector<double> cell = GrapheneCell();
    vbcsr::atomic::NeighborList list;
    list.build(pos, cell, {true, true, false}, 1.6);

    std::vector<std::array<double, 3>> bonds;
    for (const auto& neighbor : list.get_neighbors(0)) {
        std::array<double, 3> bond{};
        for (int axis = 0; axis < 3; ++axis) {
            bond[axis] = pos[3 * neighbor.index + axis] - pos[axis] +
                         neighbor.rx * cell[axis] + neighbor.ry * cell[3 + axis] +
                         neighbor.rz * cell[6 + axis];
        }
        if (std::sqrt(bond[0] * bond[0] + bond[1] * bond[1] + bond[2] * bond[2]) < 1.6) {
            bonds.push_back(bond);
        }
    }
    std::sort(bonds.begin(), bonds.end());
    return bonds;
}

}  // namespace

TEST(NeighborShiftConventionTest, OutOfCellAtomsGiveTheSameBonds) {
    // Second carbon at fractional (2/3, 2/3) -- inside the cell.
    const std::vector<double> inside = {0.0, 0.0, 10.0, kLatticeA, kBond, 10.0};
    // The same crystal with that atom written at fractional (-1/3, 2/3), and
    // again two cells further out. Every spelling must describe one carbon with
    // three neighbours at the bond length.
    const std::vector<double> outside = {0.0, 0.0, 10.0, 0.0, kBond, 10.0};
    const std::vector<double> far = {0.0, 0.0, 10.0, 3.0 * kLatticeA, kBond, 10.0};

    const auto reference = BondVectorsOfAtomZero(inside);
    ASSERT_EQ(reference.size(), 3u) << "graphene has three nearest neighbours";
    for (const auto& bond : reference) {
        EXPECT_NEAR(std::sqrt(bond[0] * bond[0] + bond[1] * bond[1] + bond[2] * bond[2]), kBond,
                    1e-12);
    }

    for (const auto* spelling : {&outside, &far}) {
        const auto bonds = BondVectorsOfAtomZero(*spelling);
        ASSERT_EQ(bonds.size(), reference.size());
        for (size_t i = 0; i < bonds.size(); ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_NEAR(bonds[i][axis], reference[i][axis], 1e-12);
            }
        }
    }
}

TEST(NeighborShiftConventionTest, AbsurdlyDistantAtomsAreRejected) {
    // Shifts are ints; a coordinate too far out to represent is an error rather
    // than an overflowed, plausible-looking image.
    const std::vector<double> pos = {0.0, 0.0, 10.0, 1e12 * kLatticeA, kBond, 10.0};
    vbcsr::atomic::NeighborList list;
    EXPECT_THROW(list.build(pos, GrapheneCell(), {true, true, false}, 1.6),
                 std::invalid_argument);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
