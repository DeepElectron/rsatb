#include "../neighbourlist.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <iostream>

using namespace rsatb::backend;

TEST(NeighborListTest, SmallCellLargeCutoff) {
    NeighborList nl;
    
    // 1. Define a small cubic cell: 1.0 x 1.0 x 1.0
    std::vector<double> cell = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    
    // 2. Define one atom at the center
    std::vector<double> positions = {0.5, 0.5, 0.5};
    
    // 3. Define PBC
    std::vector<bool> pbc = {true, true, true};
    
    // 4. Define a large cutoff: 2.5
    // This should include neighbors at distance 1.0 and 2.0
    // Images: +/- 1 (dist 1.0), +/- 2 (dist 2.0)
    double cutoff = 2.5;
    
    nl.build(positions, cell, pbc, cutoff);
    
    const auto& neighbors = nl.get_neighbors(0);
    
    // We expect to find the image at (2, 0, 0) relative shift.
    // Shift (2, 0, 0) corresponds to position (2.5, 0.5, 0.5).
    // Distance to (0.5, 0.5, 0.5) is 2.0.
    // 2.0 < 2.5, so it should be in the list.
    
    bool found_dist_2 = false;
    for (const auto& n : neighbors) {
        if (std::abs(n.rx) == 2 && n.ry == 0 && n.rz == 0) {
            found_dist_2 = true;
            break;
        }
    }
    
    if (!found_dist_2) {
        std::cout << "Failed to find neighbor at image distance 2.0 with cutoff 2.5" << std::endl;
        std::cout << "Found neighbors with shifts:" << std::endl;
        for (const auto& n : neighbors) {
            std::cout << "(" << n.rx << ", " << n.ry << ", " << n.rz << ")" << std::endl;
        }
    }
    
    EXPECT_TRUE(found_dist_2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
