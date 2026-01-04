#include "atomic_data.hpp"
#include <mpi.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <set>

using namespace rsatb::backend;

// Helper to check if a vector contains a value
bool contains(const std::vector<int>& v, int val) {
    return std::find(v.begin(), v.end(), val) != v.end();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "This test requires exactly 2 processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 4-atom chain: 0 - 1 - 2 - 3
    // Rank 0: 0, 1
    // Rank 1: 2, 3
    // Positions: (0,0,0), (1,0,0), (2,0,0), (3,0,0)
    // Cutoff: 1.5 -> Nearest neighbors only
    
    int n_atom = 2;
    int N_atom = 4;
    int atom_offset = rank * 2;
    
    std::vector<int> atom_index;
    std::vector<int> atom_type(n_atom, 0); // All type 0
    std::vector<double> pos;
    
    if (rank == 0) {
        atom_index = {0, 1};
        pos = {0,0,0, 1,0,0};
    } else {
        atom_index = {2, 3};
        pos = {2,0,0, 3,0,0};
    }
    
    // Edges (local sources only)
    std::vector<int> edge_index;
    std::vector<int> edge_shift;
    
    // 0: (0,1)
    // 1: (1,0), (1,2)
    // 2: (2,1), (2,3)
    // 3: (3,2)
    
    if (rank == 0) {
        // From 0
        edge_index.push_back(0); edge_index.push_back(1);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
        
        // From 1
        edge_index.push_back(1); edge_index.push_back(0);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
        
        edge_index.push_back(1); edge_index.push_back(2);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
    } else {
        // From 2
        edge_index.push_back(2); edge_index.push_back(1);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
        
        edge_index.push_back(2); edge_index.push_back(3);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
        
        // From 3
        edge_index.push_back(3); edge_index.push_back(2);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
    }
    
    int n_edge = edge_index.size() / 2;
    int N_edge = 6; // Total edges
    
    std::vector<int> type_norb = {1};
    std::vector<double> cell = {10,0,0, 0,10,0, 0,0,10};
    
    try {
        AtomicData ad(n_atom, N_atom, atom_offset, n_edge, N_edge,
                      atom_index.data(), atom_type.data(), edge_index.data(),
                      type_norb.data(), edge_shift.data(), cell.data(), pos.data(),
                      MPI_COMM_WORLD);
                      
        if (rank == 0) std::cout << "AtomicData constructed." << std::endl;
        
        // Verify DistGraph
        DistGraph* g = ad.dm();
        
        // Check owned indices
        if (g->owned_global_indices.size() != 2) throw std::runtime_error("Wrong owned size");
        if (g->owned_global_indices[0] != atom_offset) throw std::runtime_error("Wrong owned index 0");
        if (g->owned_global_indices[1] != atom_offset + 1) throw std::runtime_error("Wrong owned index 1");
        
        // Check ghosts
        // Rank 0 needs 2 (connected to 1)
        // Rank 1 needs 1 (connected to 2)
        if (rank == 0) {
            if (g->ghost_global_indices.size() != 1) throw std::runtime_error("Rank 0 should have 1 ghost");
            if (g->ghost_global_indices[0] != 2) throw std::runtime_error("Rank 0 ghost should be 2");
        } else {
            if (g->ghost_global_indices.size() != 1) throw std::runtime_error("Rank 1 should have 1 ghost");
            if (g->ghost_global_indices[0] != 1) throw std::runtime_error("Rank 1 ghost should be 1");
        }
        
        if (rank == 0) std::cout << "DistGraph structure verified." << std::endl;
        
        // Verify 3-body allocation
        std::vector<double> r_max = {1.5}; // Cutoff 1.5
        DistGraph* g3 = ad.allocate_three_body_hopping_matrix(r_max, r_max);
        
        if (rank == 0) std::cout << "3-body graph constructed." << std::endl;
        
        // Expected 3-body connectivity (Global indices)
        // Row 0: 0, 2
        // Row 1: 1, 3
        // Row 2: 0, 2
        // Row 3: 1, 3
        
        // Check local rows
        // Rank 0: Rows 0, 1
        // Rank 1: Rows 2, 3
        
        // Access g3 adjacency
        // DistGraph doesn't expose adj directly in a simple way, but we can check row_ptr/col_ind if we had access.
        // Or we can use find_owner/map logic? No.
        // We can check block_sizes and structure.
        
        // But wait, DistGraph stores `col_ind` in LOCAL indices.
        // We need to map them back to global to verify.
        
        // Let's inspect g3's structure manually if possible or add a verification helper in DistGraph?
        // Or we can just check the number of non-zeros per row.
        
        // Row 0: 2 neighbors (0, 2)
        // Row 1: 2 neighbors (1, 3)
        // Row 2: 2 neighbors (0, 2)
        // Row 3: 2 neighbors (1, 3)
        
        // So each row should have 2 entries.
        // g3->row_ptr is size n_owned + 1.
        // row_ptr[i+1] - row_ptr[i] should be 2.
        
        // We need to access row_ptr. DistGraph has it public?
        // Let's check DistGraph definition.
        // It has `std::vector<int> row_ptr, col_ind;`
        
        // Wait, DistGraph in `dist_graph.hpp` has `row_ptr` and `col_ind` as public members?
        // Let's assume yes based on previous usage.
        
        // Rank 0
        if (rank == 0) {
            if (g3->adj_list.size() != 2) throw std::runtime_error("Rank 0 g3 adj_list size wrong");
            
            // Check content
            // Row 0: 0, 2.
            // Row 1: 1, 3.
            
            // Check ghosts
            std::set<int> ghosts;
            for(int x : g3->ghost_global_indices) ghosts.insert(x);
            
            if (ghosts.find(2) == ghosts.end()) throw std::runtime_error("Rank 0 missing ghost 2 in 3-body");
            if (ghosts.find(3) == ghosts.end()) throw std::runtime_error("Rank 0 missing ghost 3 in 3-body");
            
            // Check col indices for Row 0
            std::set<int> cols0;
            for(int lid : g3->adj_list[0]) {
                int gid;
                if (lid < 2) gid = g3->owned_global_indices[lid];
                else gid = g3->ghost_global_indices[lid - 2];
                cols0.insert(gid);
            }
            if (cols0.size() != 2) throw std::runtime_error("Row 0 should have 2 entries");
            if (cols0.find(0) == cols0.end()) throw std::runtime_error("Row 0 missing col 0");
            if (cols0.find(2) == cols0.end()) throw std::runtime_error("Row 0 missing col 2");
            
             // Check col indices for Row 1
            std::set<int> cols1;
            for(int lid : g3->adj_list[1]) {
                int gid;
                if (lid < 2) gid = g3->owned_global_indices[lid];
                else gid = g3->ghost_global_indices[lid - 2];
                cols1.insert(gid);
            }
            if (cols1.size() != 2) throw std::runtime_error("Row 1 should have 2 entries");
            if (cols1.find(1) == cols1.end()) throw std::runtime_error("Row 1 missing col 1");
            if (cols1.find(3) == cols1.end()) throw std::runtime_error("Row 1 missing col 3");
            
        } else {
            // Rank 1
            if (g3->adj_list.size() != 2) throw std::runtime_error("Rank 1 g3 adj_list size wrong");
            
             // Ghosts should be {0, 1}
            std::set<int> ghosts;
            for(int x : g3->ghost_global_indices) ghosts.insert(x);
            
            if (ghosts.find(0) == ghosts.end()) throw std::runtime_error("Rank 1 missing ghost 0 in 3-body");
            if (ghosts.find(1) == ghosts.end()) throw std::runtime_error("Rank 1 missing ghost 1 in 3-body");
            
             // Check col indices for Row 2
            std::set<int> cols2;
            for(int lid : g3->adj_list[0]) {
                int gid;
                if (lid < 2) gid = g3->owned_global_indices[lid];
                else gid = g3->ghost_global_indices[lid - 2];
                cols2.insert(gid);
            }
            if (cols2.size() != 2) throw std::runtime_error("Row 2 should have 2 entries");
            if (cols2.find(0) == cols2.end()) throw std::runtime_error("Row 2 missing col 0");
            if (cols2.find(2) == cols2.end()) throw std::runtime_error("Row 2 missing col 2");
            
             // Check col indices for Row 3
            std::set<int> cols3;
            for(int lid : g3->adj_list[1]) {
                int gid;
                if (lid < 2) gid = g3->owned_global_indices[lid];
                else gid = g3->ghost_global_indices[lid - 2];
                cols3.insert(gid);
            }
            if (cols3.size() != 2) throw std::runtime_error("Row 3 should have 2 entries");
            if (cols3.find(1) == cols3.end()) throw std::runtime_error("Row 3 missing col 1");
            if (cols3.find(3) == cols3.end()) throw std::runtime_error("Row 3 missing col 3");
        }
        
        if (rank == 0) std::cout << "3-body connectivity verified." << std::endl;
        
        // Verify Offsets
        // Rank 0: Owned {0, 1}, Ghost {2}. Sizes {1, 1, 1}.
        // Local offsets: 0, 1, 2.
        // Global offsets: 0, 1, 2.
        
        // Rank 1: Owned {2, 3}, Ghost {1}. Sizes {1, 1, 1}.
        // Local offsets: 0, 1, 2.
        // Global offsets: 2, 3, 1.
        
        int off_l, off_g;
        
        if (rank == 0) {
            // Atom 0 (Owned)
            ad.get_offset_local(0, &off_l);
            ad.get_offset_global(0, &off_g);
            if (off_l != 0) throw std::runtime_error("Rank 0 Atom 0 local offset wrong: " + std::to_string(off_l));
            if (off_g != 0) throw std::runtime_error("Rank 0 Atom 0 global offset wrong: " + std::to_string(off_g));
            
            // Atom 1 (Owned)
            ad.get_offset_local(1, &off_l);
            ad.get_offset_global(1, &off_g);
            if (off_l != 1) throw std::runtime_error("Rank 0 Atom 1 local offset wrong: " + std::to_string(off_l));
            if (off_g != 1) throw std::runtime_error("Rank 0 Atom 1 global offset wrong: " + std::to_string(off_g));
            
            // Atom 2 (Ghost, local index 2)
            ad.get_offset_local(2, &off_l);
            ad.get_offset_global(2, &off_g);
            if (off_l != 2) throw std::runtime_error("Rank 0 Atom 2 (ghost) local offset wrong: " + std::to_string(off_l));
            if (off_g != 2) throw std::runtime_error("Rank 0 Atom 2 (ghost) global offset wrong: " + std::to_string(off_g));
            
        } else {
            // Atom 2 (Owned, local index 0)
            ad.get_offset_local(0, &off_l);
            ad.get_offset_global(0, &off_g);
            if (off_l != 0) throw std::runtime_error("Rank 1 Atom 2 local offset wrong: " + std::to_string(off_l));
            if (off_g != 2) throw std::runtime_error("Rank 1 Atom 2 global offset wrong: " + std::to_string(off_g));
            
            // Atom 3 (Owned, local index 1)
            ad.get_offset_local(1, &off_l);
            ad.get_offset_global(1, &off_g);
            if (off_l != 1) throw std::runtime_error("Rank 1 Atom 3 local offset wrong: " + std::to_string(off_l));
            if (off_g != 3) throw std::runtime_error("Rank 1 Atom 3 global offset wrong: " + std::to_string(off_g));
            
            // Atom 1 (Ghost, local index 2)
            ad.get_offset_local(2, &off_l);
            ad.get_offset_global(2, &off_g);
            if (off_l != 2) throw std::runtime_error("Rank 1 Atom 1 (ghost) local offset wrong: " + std::to_string(off_l));
            if (off_g != 1) throw std::runtime_error("Rank 1 Atom 1 (ghost) global offset wrong: " + std::to_string(off_g));
        }
        
        if (rank == 0) std::cout << "Offsets verified." << std::endl;
        
        delete g3;
        
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
