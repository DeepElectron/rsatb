#include "../neighbourlist.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

using namespace rsatb::backend;

void test_simple_cubic_pbc() {
    std::cout << "Testing Simple Cubic PBC..." << std::endl;
    // 2x2x2 grid of atoms in a 2x2x2 box. Spacing 1.0.
    // Box size 2.0.
    // Cutoff 1.1 (should find nearest neighbors)
    
    std::vector<double> positions;
    for(int z=0; z<2; ++z) {
        for(int y=0; y<2; ++y) {
            for(int x=0; x<2; ++x) {
                positions.push_back(x * 1.0);
                positions.push_back(y * 1.0);
                positions.push_back(z * 1.0);
            }
        }
    }
    
    std::vector<double> cell = {2.0,0,0, 0,2.0,0, 0,0,2.0};
    std::vector<bool> pbc = {true, true, true};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // Each atom should have 6 neighbors (up, down, left, right, front, back)
    // due to PBC.
    
    for(size_t i=0; i<positions.size()/3; ++i) {
        if (nl.neighbors[i].size() != 6) {
            std::cerr << "Atom " << i << " has " << nl.neighbors[i].size() << " neighbors, expected 6." << std::endl;
            exit(1);
        }
    }
    std::cout << "Passed." << std::endl;
}

void test_non_pbc() {
    std::cout << "Testing Non-PBC..." << std::endl;
    // 3 atoms in a line: 0.0, 1.0, 2.0
    // Cutoff 1.1
    // 0-1 connected, 1-2 connected. 0-2 not connected.
    
    std::vector<double> positions = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0
    };
    
    std::vector<double> cell = {10.0,0,0, 0,10.0,0, 0,0,10.0};
    std::vector<bool> pbc = {false, false, false};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // Atom 0: neighbor 1
    // Atom 1: neighbors 0, 2
    // Atom 2: neighbor 1
    
    assert(nl.neighbors[0].size() == 1);
    assert(nl.neighbors[0][0].index == 1);
    assert(nl.neighbors[0][0].rx == 0 && nl.neighbors[0][0].ry == 0 && nl.neighbors[0][0].rz == 0);
    
    assert(nl.neighbors[1].size() == 2);
    // Order not guaranteed, check existence
    bool has0 = false, has2 = false;
    for(const auto& n : nl.neighbors[1]) {
        if (n.index == 0) has0 = true;
        if (n.index == 2) has2 = true;
    }
    assert(has0 && has2);
    
    assert(nl.neighbors[2].size() == 1);
    assert(nl.neighbors[2][0].index == 1);
    
    std::cout << "Passed." << std::endl;
}

void test_large_system() {
    std::cout << "Testing Large System (1000 atoms)..." << std::endl;
    // 10x10x10 grid
    int N = 10;
    std::vector<double> positions;
    for(int z=0; z<N; ++z) {
        for(int y=0; y<N; ++y) {
            for(int x=0; x<N; ++x) {
                positions.push_back(x * 1.0);
                positions.push_back(y * 1.0);
                positions.push_back(z * 1.0);
            }
        }
    }
    
    std::vector<double> cell = {10.0,0,0, 0,10.0,0, 0,0,10.0};
    std::vector<bool> pbc = {true, true, true};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    for(size_t i=0; i<positions.size()/3; ++i) {
        if (nl.neighbors[i].size() != 6) {
            std::cerr << "Atom " << i << " has " << nl.neighbors[i].size() << " neighbors, expected 6." << std::endl;
            exit(1);
        }
    }
    std::cout << "Passed." << std::endl;
}

void test_mixed_pbc() {
    std::cout << "Testing Mixed PBC (XY)..." << std::endl;
    // 2x2x2 grid. PBC in X and Y, not Z.
    // Atoms at z=0 and z=1.
    // Box size 2.0.
    // Cutoff 1.1.
    
    std::vector<double> positions;
    for(int z=0; z<2; ++z) {
        for(int y=0; y<2; ++y) {
            for(int x=0; x<2; ++x) {
                positions.push_back(x * 1.0);
                positions.push_back(y * 1.0);
                positions.push_back(z * 1.0);
            }
        }
    }
    
    std::vector<double> cell = {2.0,0,0, 0,2.0,0, 0,0,2.0};
    std::vector<bool> pbc = {true, true, false};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // For z=0 atoms:
    // Neighbors in X: left, right (wrapped) -> 2
    // Neighbors in Y: front, back (wrapped) -> 2
    // Neighbors in Z: up (z=1) -> 1. Down is boundary (no wrap).
    // Total: 5.
    
    // For z=1 atoms:
    // Neighbors in X: 2
    // Neighbors in Y: 2
    // Neighbors in Z: down (z=0) -> 1. Up is boundary (no wrap).
    // Total: 5.
    
    for(size_t i=0; i<positions.size()/3; ++i) {
        if (nl.neighbors[i].size() != 5) {
            std::cerr << "Atom " << i << " has " << nl.neighbors[i].size() << " neighbors, expected 5." << std::endl;
            exit(1);
        }
    }
    std::cout << "Passed." << std::endl;
}

void test_exact_indices() {
    std::cout << "Testing Exact Indices..." << std::endl;
    // 3 atoms: 0 at (0,0,0), 1 at (1,0,0), 2 at (0,1,0)
    // Box 10. PBC false. Cutoff 1.1.
    // 0-1 (dist 1), 0-2 (dist 1), 1-2 (dist sqrt(2) ~ 1.41 > 1.1)
    
    std::vector<double> positions = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };
    
    std::vector<double> cell = {10.0,0,0, 0,10.0,0, 0,0,10.0};
    std::vector<bool> pbc = {false, false, false};
    double cutoff = 1.1;
    
    NeighborList nl;
    nl.build(positions, cell, pbc, cutoff);
    
    // Atom 0: neighbors 1, 2
    assert(nl.neighbors[0].size() == 2);
    std::vector<int> n0;
    for(const auto& n : nl.neighbors[0]) n0.push_back(n.index);
    std::sort(n0.begin(), n0.end());
    assert(n0[0] == 1 && n0[1] == 2);
    
    // Atom 1: neighbor 0
    assert(nl.neighbors[1].size() == 1);
    assert(nl.neighbors[1][0].index == 0);
    
    // Atom 2: neighbor 0
    assert(nl.neighbors[2].size() == 1);
    assert(nl.neighbors[2][0].index == 0);
    
    std::cout << "Passed." << std::endl;
}

void test_robustness() {
    std::cout << "Testing Robustness..." << std::endl;
    
    std::vector<double> positions = {0,0,0};
    std::vector<double> cell = {1,0,0, 0,1,0, 0,0,1};
    std::vector<bool> pbc = {true, true, true};
    double cutoff = 1.0;
    
    NeighborList nl;
    
    // 1. Invalid positions size
    try {
        std::vector<double> bad_pos = {0,0};
        nl.build(bad_pos, cell, pbc, cutoff);
        std::cerr << "Failed to catch invalid positions size." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 2. Invalid cell size
    try {
        std::vector<double> bad_cell = {1,0,0};
        nl.build(positions, bad_cell, pbc, cutoff);
        std::cerr << "Failed to catch invalid cell size." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 3. Invalid PBC size
    try {
        std::vector<bool> bad_pbc = {true, true};
        nl.build(positions, cell, bad_pbc, cutoff);
        std::cerr << "Failed to catch invalid PBC size." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 4. Zero Volume
    try {
        std::vector<double> zero_cell = {0,0,0, 0,0,0, 0,0,0};
        nl.build(positions, zero_cell, pbc, cutoff);
        std::cerr << "Failed to catch zero volume." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}
    
    // 5. Negative Volume (Determinant is negative, but we take abs, so it might pass if valid shape)
    // Actually, we check abs(vol) < 1e-8.
    // If we give a valid cell with negative determinant, it should work (just handedness change).
    // Let's test a flattened cell (vol=0).
    try {
        std::vector<double> flat_cell = {1,0,0, 1,0,0, 0,0,1}; // Collinear a and b
        nl.build(positions, flat_cell, pbc, cutoff);
        std::cerr << "Failed to catch flat cell." << std::endl;
        exit(1);
    } catch (const std::invalid_argument&) {}

    // 6. Positions outside cell (PBC)
    // Atom at 1.5 in box of size 1.0. Should wrap to 0.5.
    {
        std::vector<double> out_pos = {1.5, 0.5, 0.5,  0.5, 0.5, 0.5}; // 0 wraps to 0.5. 1 is at 0.5.
        // They should be on top of each other -> neighbors.
        nl.build(out_pos, cell, pbc, cutoff);
        assert(nl.neighbors[0].size() == 1);
        assert(nl.neighbors[0][0].index == 1);
    }
    
    // 7. Positions outside cell (Non-PBC)
    // Atom at 100.0. Box 1.0. Cutoff 1.0.
    // Should be far away.
    {
        std::vector<double> far_pos = {0,0,0, 100,0,0};
        std::vector<bool> no_pbc = {false, false, false};
        nl.build(far_pos, cell, no_pbc, cutoff);
        assert(nl.neighbors[0].empty());
    }

    // 8. Extreme Cutoffs
    // Small cutoff
    {
        std::vector<double> pair = {0,0,0, 0.5,0,0};
        nl.build(pair, cell, pbc, 0.1); // Too small
        assert(nl.neighbors[0].empty());
    }
    // Large cutoff
    {
        std::vector<double> pair = {0,0,0, 0.5,0,0};
        std::vector<bool> no_pbc = {false, false, false};
        nl.build(pair, cell, no_pbc, 100.0); // Huge
        assert(nl.neighbors[0].size() == 1);
    }

    std::cout << "Passed." << std::endl;
}

int main() {
    test_simple_cubic_pbc();
    test_non_pbc();
    test_large_system();
    test_mixed_pbc();
    test_exact_indices();
    test_robustness();
    return 0;
}
