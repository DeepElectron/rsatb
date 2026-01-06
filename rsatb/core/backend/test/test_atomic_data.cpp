#include <gtest/gtest.h>
#include "../atomic_data.hpp"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <fstream>

using namespace rsatb::backend;

class AtomicDataTest : public ::testing::Test {
protected:
    void SetUp() override {
        // MPI is initialized in main
    }
    
    void TearDown() override {
    }
};

TEST_F(AtomicDataTest, FromPoints) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a simple 2-atom system on rank 0
    std::vector<double> pos;
    std::vector<int> z;
    if (rank == 0) {
        pos = {0.0, 0.0, 0.0,  1.5, 0.0, 0.0};
        z = {1, 1};
    }
    
    std::vector<double> cell = {10.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {2.0, 2.0}; // For types 0 and 1
    std::vector<int> type_norb = {13, 13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    int local_n = ad->n_atom;
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    int local_edges = ad->n_edge;
    int total_edges;
    MPI_Allreduce(&local_edges, &total_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        EXPECT_EQ(total_n, 2);
        
        // 0-1 and 1-0 -> 2 edges
        // Note: n_edge counts directed edges in adjacency list
        EXPECT_GT(total_edges, 0);
        EXPECT_EQ(total_edges, 2);
    }
    
    delete ad;
}

TEST_F(AtomicDataTest, Partition) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // Create a chain of atoms on rank 0
    int n_atoms = 10;
    std::vector<double> pos;
    std::vector<int> z;
    if (rank == 0) {
        pos.resize(n_atoms * 3);
        z.resize(n_atoms);
        for(int i=0; i<n_atoms; ++i) {
            pos[3*i] = i * 1.0;
            pos[3*i+1] = 0;
            pos[3*i+2] = 0;
            z[i] = 1;
        }
    }
    
    std::vector<double> cell = {20.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.5};
    std::vector<int> type_norb = {13,13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Partition is done inside from_points
    AtomicData* ad_part = ad;
    
    // Check load balance
    int local_n = ad_part->n_atom;
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    EXPECT_EQ(total_n, n_atoms);
    
    // Expect roughly equal distribution
    if (rank == 0) std::cout << "Rank 0 has " << local_n << " atoms." << std::endl;
    if (rank == 1) std::cout << "Rank 1 has " << local_n << " atoms." << std::endl;
    
    // With 10 atoms and 2 ranks, should be 5 and 5.
    EXPECT_NEAR(local_n, n_atoms/size, 1);
    
    delete ad;
}

TEST_F(AtomicDataTest, IO_POSCAR) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> type_norb = {13,13};
    
    std::string filename = "test_poscar.vasp";
    if (rank == 0) {
        std::ofstream out(filename);
        out << "Test\n1.0\n10 0 0\n0 10 0\n0 0 10\nH\n2\nDirect\n0.1 0.1 0.1\n0.2 0.2 0.2\n";
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {2.0};
    
    AtomicData* ad = AtomicData::from_file(filename, r_max, type_norb, MPI_COMM_WORLD);
    
    int local_n = ad->n_atom;
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        EXPECT_EQ(total_n, 2);
        // We can't easily check types on rank 0 if atoms are distributed.
        // But we can check if the total types are correct.
    }
    
    // Verify types locally
    for(int i=0; i<ad->n_atom; ++i) {
        EXPECT_EQ(ad->atom_type[i], 0); // H is type 0
    }
    
    delete ad;
    if (rank == 0) remove(filename.c_str());
}

TEST_F(AtomicDataTest, DeterministicTypeMapping) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Test that types are assigned deterministically based on atomic number
    // Case 1: H (1), He (2)
    // Case 1: H (1), He (2)
    {
        std::vector<double> pos = {0,0,0, 1,0,0};
        std::vector<int> z = {1, 2};
        std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
        std::vector<bool> pbc = {false, false, false};
        std::vector<double> r_max = {1.0, 1.0};
        std::vector<int> type_norb = {13,13};
        AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
        
        for(int i=0; i<ad->n_atom; ++i) {
            if (ad->z[i] == 1) EXPECT_EQ(ad->atom_type[i], 0);
            if (ad->z[i] == 2) EXPECT_EQ(ad->atom_type[i], 1);
        }
        delete ad;
    }
    
    // Case 2: He (2), H (1) -> Should still be H=0, He=1
    {
        std::vector<double> pos = {0,0,0, 1,0,0};
        std::vector<int> z = {2, 1};
        std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
        std::vector<bool> pbc = {false, false, false};
        std::vector<double> r_max = {1.0, 1.0};
        std::vector<int> type_norb = {13,13};

        AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
        
        for(int i=0; i<ad->n_atom; ++i) {
            if (ad->z[i] == 1) EXPECT_EQ(ad->atom_type[i], 0);
            if (ad->z[i] == 2) EXPECT_EQ(ad->atom_type[i], 1);
        }
        delete ad;
    }
    
    // Case 3: Gaps in Z (e.g. H(1) and O(8))
    {
        std::vector<double> pos = {0,0,0, 1,0,0};
        std::vector<int> z = {1, 8};
        std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
        std::vector<bool> pbc = {false, false, false};
        std::vector<double> r_max = {1.0, 1.0};
        std::vector<int> type_norb = {13,13};

        AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
        
        for(int i=0; i<ad->n_atom; ++i) {
            if (ad->z[i] == 1) EXPECT_EQ(ad->atom_type[i], 0);
            if (ad->z[i] == 8) EXPECT_EQ(ad->atom_type[i], 1);
        }
        delete ad;
    }
}

TEST_F(AtomicDataTest, PartitionGrid) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // 4x4x4 grid = 64 atoms
    int n_grid = 4;
    int n_atoms = n_grid * n_grid * n_grid;
    std::vector<double> pos;
    std::vector<int> z;
    
    if (rank == 0) {
        pos.resize(n_atoms * 3);
        z.resize(n_atoms);
        int idx = 0;
        for(int i=0; i<n_grid; ++i)
        for(int j=0; j<n_grid; ++j)
        for(int k=0; k<n_grid; ++k) {
            pos[3*idx] = i;
            pos[3*idx+1] = j;
            pos[3*idx+2] = k;
            z[idx] = 1;
            idx++;
        }
    }
    
    std::vector<double> cell = {4.0, 0.0, 0.0,  0.0, 4.0, 0.0,  0.0, 0.0, 4.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.1}; // Connect nearest neighbors on grid
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    AtomicData* ad_part = ad;
    
    int local_n = ad_part->n_atom;
    if (rank == 0) std::cout << "Grid Partition: Rank 0 has " << local_n << " atoms." << std::endl;
    
    // Verify total atoms
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_n, n_atoms);
    
    // Verify atom indices are preserved (global check)
    // Collect all indices
    std::vector<int> all_indices;
    std::vector<int> local_indices(local_n);
    for(int i=0; i<local_n; ++i) local_indices[i] = ad_part->atom_index[i];
    
    // Gather counts
    std::vector<int> counts(size);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::vector<int> displs(size + 1, 0);
    for(int i=0; i<size; ++i) displs[i+1] = displs[i] + counts[i];
    
    all_indices.resize(total_n);
    MPI_Allgatherv(local_indices.data(), local_n, MPI_INT, 
                   all_indices.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
                   
    if (rank == 0) {
        std::sort(all_indices.begin(), all_indices.end());
        for(int i=0; i<total_n; ++i) {
            EXPECT_EQ(all_indices[i], i);
        }
    }
    
    delete ad;
}

TEST_F(AtomicDataTest, SingleProcess) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 1) {
        if (rank == 0) std::cout << "Skipping SingleProcess test (size != 1)" << std::endl;
        return;
    }
    
    std::vector<double> pos = {0,0,0};
    std::vector<int> z = {1};
    std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
    std::vector<bool> pbc = {false, false, false};
    std::vector<int> type_norb = {13};
    
    std::vector<double> r_max = {1.0};
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    AtomicData* ad_part = ad;
    
    EXPECT_EQ(ad_part->n_atom, 1);
    EXPECT_EQ(ad_part->atom_index[0], 0);
    
    delete ad;
}

TEST_F(AtomicDataTest, NeighborListNonPeriodic) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Two atoms far apart, non-periodic
    std::vector<double> pos = {0,0,0, 1.5,0,0};
    std::vector<int> z = {1, 1};
    std::vector<double> cell = {100,0,0, 0,100,0, 0,0,100}; // Large dummy cell
    std::vector<bool> pbc = {false, false, false};
    
    std::vector<double> r_max = {2.0};
    std::vector<int> type_norb = {13};
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    int local_edges = ad->n_edge;
    int total_edges;
    MPI_Allreduce(&local_edges, &total_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        EXPECT_EQ(total_edges, 2); // 0-1 and 1-0
    }
    delete ad;
}

TEST_F(AtomicDataTest, LargeSystem) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // 8x8x8 grid = 512 atoms
    int n_grid = 8;
    int n_atoms = n_grid * n_grid * n_grid;
    std::vector<double> pos;
    std::vector<int> z;
    
    if (rank == 0) {
        pos.resize(n_atoms * 3);
        z.resize(n_atoms);
        int idx = 0;
        for(int i=0; i<n_grid; ++i)
        for(int j=0; j<n_grid; ++j)
        for(int k=0; k<n_grid; ++k) {
            pos[3*idx] = i;
            pos[3*idx+1] = j;
            pos[3*idx+2] = k;
            z[idx] = 1;
            idx++;
        }
    }
    
    std::vector<double> cell = {8.0, 0.0, 0.0,  0.0, 8.0, 0.0,  0.0, 0.0, 8.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.1}; 
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    AtomicData* ad_part = ad;
    
    int local_n = ad_part->n_atom;
    if (rank == 0) std::cout << "Large System Partition: Rank 0 has " << local_n << " atoms." << std::endl;
    
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_n, n_atoms);
    
    delete ad;
}

TEST_F(AtomicDataTest, MultiElementEdgeMapping) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // Chain: H - He - Li - Be
    // 0 - 1 - 2 - 3
    // Pos: 0, 1, 2, 3
    std::vector<double> pos = {0,0,0, 1,0,0, 2,0,0, 3,0,0};
    std::vector<int> z = {1, 2, 3, 4}; // H, He, Li, Be
    // Types should be 0, 1, 2, 3 (sorted)
    
    std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
    std::vector<bool> pbc = {false, false, false};
    std::vector<double> r_max = {0.5, 0.5, 0.5, 0.5};
    std::vector<int> type_norb = {13,13,13,13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Check local atoms and their types
    for(int i=0; i<ad->n_atom; ++i) {
        int gid = ad->atom_index[i];
        int type = ad->atom_type[i];
        EXPECT_EQ(type, gid);
    }
    
    // Check edges
    for(int i=0; i<ad->n_atom; ++i) {
        int gid = ad->atom_index[i];
        const auto& edge_indices = ad->get_atom_edges(i);
        
        for(int edge_idx : edge_indices) {
            int dst_lid = ad->get_edge_dst(edge_idx);
            int dst_orig_gid = ad->atom_index[dst_lid];
            int dst_type = ad->atom_type[dst_lid];
            
            EXPECT_EQ(dst_type, dst_orig_gid);
            EXPECT_EQ(std::abs(gid - dst_orig_gid), 1);
        }
    }
    
    delete ad;
}

TEST_F(AtomicDataTest, EdgeFiltering) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Test edge filtering with different radii
    // Atom 0 (Type 0): r=0.5
    // Atom 1 (Type 1): r=1.0
    // Distance = 1.2
    // Cutoff = 0.5 + 1.0 = 1.5 -> Connected
    
    // Atom 2 (Type 0): r=0.5
    // Atom 3 (Type 0): r=0.5
    // Distance = 1.2
    // Cutoff = 0.5 + 0.5 = 1.0 -> Not connected
    
    std::vector<double> pos = {
        0,0,0,   1.2,0,0, // Pair 1
        10,0,0,  11.2,0,0 // Pair 2 (shifted by 10)
    };
    std::vector<int> z = {1, 2, 1, 1}; // Types: 0, 1, 0, 0
    std::vector<double> cell = {20,0,0, 0,20,0, 0,0,20};
    std::vector<bool> pbc = {false, false, false};
    std::vector<int> type_norb = {13,13};
    // r_max for types 0 and 1
    std::vector<double> r_max = {0.5, 1.0}; 
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Check edges
        // Pair 1 (0-1): Should be connected
        // Pair 2 (2-3): Should NOT be connected
        
        // Count edges for each atom
        // Note: get_atom_edges uses local index. We need to find local index for global atoms 0,1,2,3.
        // Since we are on rank 0 and system is small, likely all on rank 0 or distributed.
        // But from_points redistributes.
        // We should gather edge counts globally.
    }
    
    // Gather edge counts per atom
    // Map global_id -> edge_count
    std::vector<int> my_edge_counts;
    std::vector<int> my_gids;
    
    for(int i=0; i<ad->n_atom; ++i) {
        my_gids.push_back(ad->atom_index[i]);
        my_edge_counts.push_back(ad->get_atom_edges(i).size());
    }
    
    int total_atoms_global = 4;
    std::vector<int> global_edge_counts(total_atoms_global, 0);
    
    for(int r=0; r<ad->size; ++r) {
        // Gather counts from rank r
        // Simple way: Allgather everything? No, just reduce to rank 0.
        // Since we don't know which rank has which atom, we can send (gid, count) pairs.
        // Or just use a fixed size array if we know GIDs are 0..3
        
        std::vector<int> counts_on_rank(total_atoms_global, 0);
        for(size_t k=0; k<my_gids.size(); ++k) {
            if (my_gids[k] < total_atoms_global) {
                counts_on_rank[my_gids[k]] = my_edge_counts[k];
            }
        }
        
        std::vector<int> reduced_counts(total_atoms_global);
        MPI_Reduce(counts_on_rank.data(), reduced_counts.data(), total_atoms_global, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            global_edge_counts = reduced_counts;
        }
    }
    
    if (rank == 0) {
        EXPECT_GT(global_edge_counts[0], 0); // Atom 0 connected
        EXPECT_GT(global_edge_counts[1], 0); // Atom 1 connected
        EXPECT_EQ(global_edge_counts[2], 0); // Atom 2 NOT connected
        EXPECT_EQ(global_edge_counts[3], 0); // Atom 3 NOT connected
    }
    
    delete ad;
}

TEST_F(AtomicDataTest, DistributedConstruction) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // Create a system that spans across ranks spatially
    // 0.0 ... 10.0 ... 20.0
    // Atoms at 0, 1, ..., 19
    int n_atoms = 20;
    std::vector<double> pos;
    std::vector<int> z;
    
    if (rank == 0) {
        pos.resize(n_atoms * 3);
        z.resize(n_atoms);
        for(int i=0; i<n_atoms; ++i) {
            pos[3*i] = i * 1.0;
            pos[3*i+1] = 0;
            pos[3*i+2] = 0;
            z[i] = 1;
        }
    }
    
    std::vector<double> cell = {20.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {0.6}; // Connect i to i+1
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Verify every atom has at least 1 neighbor (except ends if open, but PBC is true)
    // With PBC, 0 connects to 19 (dist 1.0? No, 20.0 box. 0 and 19 dist is 1.0 wrapped?
    // 0 -> 19 is -1.0 (wrapped +19? No. 0..20. 19 is at 19. 0 is at 0.
    // 19 - 0 = 19. Wrapped: 19 - 20 = -1. Distance 1.0.
    // So 0 and 19 should be connected.
    // So ALL atoms should have 2 neighbors.
    
    for(int i=0; i<ad->n_atom; ++i) {
        int n_edges = ad->get_atom_edges(i).size();
        EXPECT_EQ(n_edges, 2) << "Atom " << ad->atom_index[i] << " on rank " << rank << " has wrong edges";
    }
    
    // Verify total atoms
    int local_n = ad->n_atom;
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_n, n_atoms);
    
    delete ad;
}
TEST_F(AtomicDataTest, PressureTest_DenseGraph) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // 1000 atoms in a 10x10x10 box
    // Cutoff 3.5 -> Dense
    
    int n_atoms = 1000;
    std::vector<double> pos;
    std::vector<int> z;
    
    if (rank == 0) {
        pos.resize(n_atoms * 3);
        z.resize(n_atoms);
        for(int i=0; i<n_atoms; ++i) {
            pos[3*i] = (double)(i % 10) + 0.5; // Shift to center
            pos[3*i+1] = (double)((i / 10) % 10) + 0.5;
            pos[3*i+2] = (double)((i / 100) % 10) + 0.5;
            z[i] = 1;
        }
    }
    
    std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {3.5}; 
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Verify total atoms
    int local_n = ad->n_atom;
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_n, n_atoms);
    
    // Verify edges exist
    int local_edges = ad->n_edge;
    long long total_edges;
    long long local_edges_long = local_edges;
    MPI_Allreduce(&local_edges_long, &total_edges, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) std::cout << "PressureTest_DenseGraph: Total edges = " << total_edges << std::endl;
    EXPECT_GT(total_edges, n_atoms * 10); 
    
    delete ad;
}

TEST_F(AtomicDataTest, PressureTest_ExtremeAspect) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) return;
    
    // 1000 atoms in a 100x5x5 box
    // Atoms distributed along X
    
    int n_atoms = 1000;
    std::vector<double> pos;
    std::vector<int> z;
    
    if (rank == 0) {
        pos.resize(n_atoms * 3);
        z.resize(n_atoms);
        for(int i=0; i<n_atoms; ++i) {
            pos[3*i] = (double)(i) * 0.1; // 0 to 100
            pos[3*i+1] = 2.5;
            pos[3*i+2] = 2.5;
            z[i] = 1;
        }
    }
    
    std::vector<double> cell = {100,0,0, 0,5,0, 0,0,5};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.5}; 
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Verify total atoms
    int local_n = ad->n_atom;
    int total_n;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_n, n_atoms);
    
    // Verify load balance
    int min_n, max_n;
    MPI_Allreduce(&local_n, &min_n, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) std::cout << "PressureTest_ExtremeAspect: Min atoms=" << min_n << " Max atoms=" << max_n << std::endl;
    
    // Allow some imbalance but not too much (e.g., < 20% difference)
    // ParMETIS tries to balance vertices.
    EXPECT_LE(max_n, n_atoms / size * 1.2 + 10);
    
    delete ad;
}

TEST_F(AtomicDataTest, Robustness) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> cell = {10.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.5};
    std::vector<int> type_norb = {13};

    // 1. Empty Input
    std::vector<double> empty_pos;
    std::vector<int> empty_z;
    AtomicData* ad = AtomicData::from_points(empty_pos, empty_z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    EXPECT_EQ(ad->n_atom, 0);
    delete ad;

    // 2. Zero Volume Cell
    if (rank == 0) {
        std::vector<double> zero_cell = {0,0,0, 0,0,0, 0,0,0};
        std::vector<double> pos = {0,0,0};
        std::vector<int> z = {1};
        std::vector<int> type_norb = {13};
        // NeighborList throws invalid_argument on zero volume
        EXPECT_THROW({
            AtomicData::from_points(pos, z, zero_cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
        }, std::invalid_argument);
    }
}

TEST_F(AtomicDataTest, ThreeBodyGraph) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> cell = {10.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};

    // 3 atoms in a line: 0 --(0.8)--> 1 --(0.8)--> 2
    // r_max = 0.5. Cutoff = 1.0.
    // 0-1 dist 0.8 < 1.0 -> connected.
    // 1-2 dist 0.8 < 1.0 -> connected.
    
    std::vector<double> line_pos = {
        0.0, 0.0, 0.0,
        0.8, 0.0, 0.0,
        1.6, 0.0, 0.0
    };
    std::vector<int> line_z = {1, 1, 1};
    std::vector<double> r_max_3b = {0.5}; 
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(line_pos, line_z, cell, pbc, r_max_3b, type_norb, MPI_COMM_WORLD);
    
    DistGraph* g3 = ad->get_graph3b(r_max_3b, r_max_3b);
    
    // Expected edges in 3-body graph:
    // 2-body edges + self-edges + 3-body connections
    // 0: (0,0), (0,1), (1,0), (1,1) -> 4
    // 1: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2) -> 9
    // 2: (1,1), (1,2), (2,1), (2,2) -> 4
    // Total = 17
    
    int local_edges = 0;
    for(size_t i=0; i<g3->adj_list.size(); ++i) {
        local_edges += g3->adj_list[i].size();
    }
    
    int total_edges;
    MPI_Allreduce(&local_edges, &total_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Expected 9 edges:
    // 0: 0, 1, 2
    // 1: 0, 1, 2
    // 2: 0, 1, 2
    EXPECT_EQ(total_edges, 9);
    
    delete g3;
    delete ad;
}

TEST_F(AtomicDataTest, Consistency) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> cell = {10.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.5};
    
    std::vector<double> pos;
    std::vector<int> z;
    if (rank == 0) {
        pos = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0};
        z = {1, 1};
    }
    std::vector<int> type_norb = {13};

    AtomicData* ad = AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Verify ad->graph contains all edges in ad->edges
    for(size_t k=0; k<ad->edges.size(); ++k) {
        int u_lid = ad->edges[k].src;
        int v_lid = ad->edges[k].dst;
        int v_gid = ad->get_global_index(v_lid);
        
        const auto& neighbors = ad->graph->adj_list[u_lid];
        bool found = false;
        for(int n : neighbors) {
            if (n == v_gid) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Edge " << k << " missing in DistGraph";
    }
    
    // Check self-edges
    for(int i=0; i<ad->n_atom; ++i) {
        int my_gid = ad->get_global_index(i);
        const auto& neighbors = ad->graph->adj_list[i];
        bool found = false;
        for(int n : neighbors) {
            if (n == my_gid) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Self-edge missing for atom " << i;
    }
    
    delete ad;
}

TEST_F(AtomicDataTest, PressureTest_Random) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> cell = {10.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.5};

    int N = 1000;
    std::vector<double> r_pos(N * 3);
    std::vector<int> r_z(N);
    std::vector<int> type_norb = {13};
    
    if (rank == 0) {
        for(int i=0; i<N; ++i) {
            r_pos[3*i] = (double)rand() / RAND_MAX * 10.0;
            r_pos[3*i+1] = (double)rand() / RAND_MAX * 10.0;
            r_pos[3*i+2] = (double)rand() / RAND_MAX * 10.0;
            r_z[i] = rand() % 2 + 1;
        }
    }
    
    AtomicData* ad = AtomicData::from_points(r_pos, r_z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    int total_n;
    int local_n = ad->n_atom;
    MPI_Allreduce(&local_n, &total_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_n, N);
    
    for(int i=0; i<ad->n_atom; ++i) {
        bool has_self = false;
        for(int n : ad->graph->adj_list[i]) {
            if (n == i) has_self = true;
        }
        EXPECT_TRUE(has_self);
    }
    
    delete ad;
}

TEST_F(AtomicDataTest, ThreeBodyCorrectness_BruteForce) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 1. Create a random system
    int N = 50; // Small enough for O(N^3) check
    std::vector<double> r_pos(N * 3);
    std::vector<int> r_z(N);
    
    if (rank == 0) {
        for(int i=0; i<N; ++i) {
            r_pos[3*i] = (double)rand() / RAND_MAX * 10.0;
            r_pos[3*i+1] = (double)rand() / RAND_MAX * 10.0;
            r_pos[3*i+2] = (double)rand() / RAND_MAX * 10.0;
            r_z[i] = 1;
        }
    }
    
    std::vector<double> cell = {10.0, 0.0, 0.0,  0.0, 10.0, 0.0,  0.0, 0.0, 10.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {2.5}; // Large enough to have many neighbors
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(r_pos, r_z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // 2. Get 3-body graph
    DistGraph* g3 = ad->get_graph3b(r_max, r_max);
    
    // 3. Gather all edges to rank 0 for verification
    // 3.1 Gather 2-body edges (from ad->edges)
    // ad->edges stores (src_lid, dst_lid). Need to convert to global.
    std::vector<std::pair<int, int>> global_edges_2b;
    
    // Local edges
    std::vector<int> local_edges_flat;
    for(const auto& e : ad->edges) {
        local_edges_flat.push_back(ad->get_global_index(e.src));
        local_edges_flat.push_back(ad->get_global_index(e.dst));
    }
    
    // Gather
    int n_local_edges = local_edges_flat.size();
    std::vector<int> recv_counts(size);
    MPI_Gather(&n_local_edges, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> displs(size + 1, 0);
    std::vector<int> all_edges_flat;
    if (rank == 0) {
        for(int i=0; i<size; ++i) displs[i+1] = displs[i] + recv_counts[i];
        all_edges_flat.resize(displs[size]);
    }
    
    MPI_Gatherv(local_edges_flat.data(), n_local_edges, MPI_INT, 
                all_edges_flat.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
                
    if (rank == 0) {
        for(size_t i=0; i<all_edges_flat.size(); i+=2) {
            global_edges_2b.push_back({all_edges_flat[i], all_edges_flat[i+1]});
        }
    }
    
    // 3.2 Gather 3-body edges (from g3->adj_list)
    std::vector<int> local_edges_3b_flat;
    for(int i=0; i<ad->n_atom; ++i) {
        int u_gid = ad->get_global_index(i);
        for(int v_lid : g3->adj_list[i]) {
            int v_gid = g3->get_global_index(v_lid);
            local_edges_3b_flat.push_back(u_gid);
            local_edges_3b_flat.push_back(v_gid);
        }
    }
    
    int n_local_edges_3b = local_edges_3b_flat.size();
    MPI_Gather(&n_local_edges_3b, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        for(int i=0; i<size; ++i) displs[i+1] = displs[i] + recv_counts[i];
        all_edges_flat.resize(displs[size]);
    }
    
    MPI_Gatherv(local_edges_3b_flat.data(), n_local_edges_3b, MPI_INT,
                all_edges_flat.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
                
    std::vector<std::pair<int, int>> global_edges_3b;
    if (rank == 0) {
        for(size_t i=0; i<all_edges_flat.size(); i+=2) {
            global_edges_3b.push_back({all_edges_flat[i], all_edges_flat[i+1]});
        }
        std::sort(global_edges_3b.begin(), global_edges_3b.end());
        global_edges_3b.erase(std::unique(global_edges_3b.begin(), global_edges_3b.end()), global_edges_3b.end());
    }
    
    // 4. Brute Force Verification on Rank 0
    if (rank == 0) {
        // Build adjacency matrix for 2-body
        std::vector<std::vector<int>> adj(N);
        for(const auto& p : global_edges_2b) {
            adj[p.first].push_back(p.second);
        }
        
        // Compute expected 3-body edges
        // Rule: (j, k) is edge if exists i s.t. (i, j) in E and (i, k) in E
        // Plus E itself.
        std::vector<std::pair<int, int>> expected_3b;
        
        // Add 2-body edges
        for(const auto& p : global_edges_2b) expected_3b.push_back(p);
        
        // Add shared-neighbor edges
        for(int i=0; i<N; ++i) {
            const auto& neighbors = adj[i];
            for(int j : neighbors) {
                for(int k : neighbors) {
                    expected_3b.push_back({j, k});
                }
            }
        }
        
        // Add self edges (if not already covered)
        for(int i=0; i<N; ++i) expected_3b.push_back({i, i});
        
        std::sort(expected_3b.begin(), expected_3b.end());
        expected_3b.erase(std::unique(expected_3b.begin(), expected_3b.end()), expected_3b.end());
        
        // Compare
        EXPECT_EQ(global_edges_3b.size(), expected_3b.size());
        
        // Check content
        for(size_t i=0; i<expected_3b.size(); ++i) {
            EXPECT_EQ(global_edges_3b[i], expected_3b[i]) << "Mismatch at index " << i;
        }
    }
    
    delete g3;
    delete ad;
}

TEST_F(AtomicDataTest, SingleProcess_Stress) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Only run if size == 1 to strictly test single process logic
    if (size != 1) return;
    
    // 1. Create a dense system
    int N = 10; 
    std::vector<double> r_pos(N * 3);
    std::vector<int> r_z(N);
    
    for(int i=0; i<N; ++i) {
        r_pos[3*i] = (double)rand() / RAND_MAX * 5.0; // Dense
        r_pos[3*i+1] = (double)rand() / RAND_MAX * 5.0;
        r_pos[3*i+2] = (double)rand() / RAND_MAX * 5.0;
        r_z[i] = 1;
    }
    
    std::vector<double> cell = {5.0, 0.0, 0.0,  0.0, 5.0, 0.0,  0.0, 0.0, 5.0};
    std::vector<bool> pbc = {true, true, true};
    std::vector<double> r_max = {1.0}; 
    std::vector<int> type_norb = {13};
    
    AtomicData* ad = AtomicData::from_points(r_pos, r_z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD);
    
    // Verify basic properties
    EXPECT_EQ(ad->n_atom, N);
    
    // Verify 3-body graph construction with full brute-force check
    DistGraph* g3 = ad->get_graph3b(r_max, r_max);
    
    // 1. Build Expected 2-body adjacency
    std::vector<std::vector<int>> adj_2b(N);
    std::vector<std::pair<int, int>> edges_2b;
    
    // Brute force 2-body
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) { // Check all pairs, including self if distance is 0 (which it is)
            // Distance check with PBC
            double dx = r_pos[3*j] - r_pos[3*i];
            double dy = r_pos[3*j+1] - r_pos[3*i+1];
            double dz = r_pos[3*j+2] - r_pos[3*i+2];
            
            // Apply PBC (Minimum Image Convention)
            // Box is 5.0. 
            auto apply_pbc = [](double d, double L) {
                while(d > L/2) d -= L;
                while(d < -L/2) d += L;
                return d;
            };
            dx = apply_pbc(dx, 5.0);
            dy = apply_pbc(dy, 5.0);
            dz = apply_pbc(dz, 5.0);
            
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            // AtomicData uses r_cut = r_max[type_i] + r_max[type_j]
            // Here r_max = {1.0}, so r_cut = 2.0.
            if (r <= 2.0 + 1e-9) { 
                // Check if self-edge (i==j) - always included if r=0
                if (i == j) {
                    // Always add self-edge
                    adj_2b[i].push_back(j);
                    edges_2b.push_back({i, j});
                } else {
                    adj_2b[i].push_back(j);
                    edges_2b.push_back({i, j});
                }
            }
        }
    }
    
    // 2. Build Expected 3-body edges
    std::vector<std::pair<int, int>> expected_3b;
    
    // Add 2-body edges
    for(const auto& p : edges_2b) expected_3b.push_back(p);
    
    // Add shared-neighbor edges
    for(int i=0; i<N; ++i) {
        const auto& neighbors = adj_2b[i];
        for(int j : neighbors) {
            for(int k : neighbors) {
                expected_3b.push_back({j, k});
            }
        }
    }
    
    std::sort(expected_3b.begin(), expected_3b.end());
    expected_3b.erase(std::unique(expected_3b.begin(), expected_3b.end()), expected_3b.end());

    // 3. Collect Actual 3-body edges from g3 and map to original indices
    std::vector<std::pair<int, int>> actual_3b;
    for(int i=0; i<N; ++i) {
        // g3->adj_list[i] contains neighbors of atom i (local index i)
        int orig_i = ad->atom_index[i];
        for(int neighbor : g3->adj_list[i]) {
            int orig_neighbor = ad->atom_index[neighbor];
            actual_3b.push_back({orig_i, orig_neighbor});
        }
    }
    std::sort(actual_3b.begin(), actual_3b.end());
    
    // 4. Compare
    EXPECT_EQ(actual_3b.size(), expected_3b.size());
    
    // Check content
    size_t min_size = std::min(actual_3b.size(), expected_3b.size());
    for(size_t i=0; i<min_size; ++i) {
        EXPECT_EQ(actual_3b[i], expected_3b[i]) << "Mismatch at index " << i;
    }
    
    delete g3;
    delete ad;
}
    


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
