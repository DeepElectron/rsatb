#include "../dist_graph.hpp"
#include <cassert>
#include <iostream>

using namespace rsatb::backend;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cout << "This test requires exactly 2 processes." << std::endl;
        MPI_Finalize();
        return 0;
    }

    // 4 blocks total
    // Rank 0: 0, 1
    // Rank 1: 2, 3
    // Connectivity: Ring 0-1-3-2-0 (plus diagonals 0-2, 1-3 for complexity?)
    // Let's do simple 1D chain: 0-1-2-3
    // 0: [0, 1] -> 1 is local, 0 is local. Neighbors of 0: 1. Neighbors of 1: 0, 2.
    // 1: [2, 3] -> 2 is local, 3 is local. Neighbors of 2: 1, 3. Neighbors of 3: 2.
    
    // Global Adjacency
    // 0: 1
    // 1: 0, 2
    // 2: 1, 3
    // 3: 2
    std::vector<std::vector<int>> global_adj = {
        {1},
        {0, 2},
        {1, 3},
        {2}
    };
    
    std::vector<int> block_sizes = {10, 20, 30, 40};

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(4, block_sizes, global_adj);

    if (rank == 0) {
        // Owned: 0, 1
        assert(graph.owned_global_indices.size() == 2);
        assert(graph.owned_global_indices[0] == 0);
        assert(graph.owned_global_indices[1] == 1);
        
        // Ghosts: 2 (needed by 1)
        assert(graph.ghost_global_indices.size() == 1);
        assert(graph.ghost_global_indices[0] == 2);
        
        // Block sizes
        // Local 0 (G0): 10
        // Local 1 (G1): 20
        // Local 2 (G2): 30
        assert(graph.block_sizes[0] == 10);
        assert(graph.block_sizes[1] == 20);
        assert(graph.block_sizes[2] == 30);
        
        // Check comm pattern
        // I need G2 from Rank 1.
        // Recv from Rank 1: 1 block (G2)
        assert(graph.recv_counts[1] == 1);
        
        // Rank 1 needs G1 from me.
        // Send to Rank 1: 1 block (G1)
        assert(graph.send_counts[1] == 1);
        
        // Send indices: G1 is local 1
        assert(graph.send_indices[0] == 1);
        
        // Recv indices: G2 is local 2
        assert(graph.recv_indices[0] == 2);
        
        std::cout << "Rank 0 passed." << std::endl;
    } else {
        // Owned: 2, 3
        assert(graph.owned_global_indices.size() == 2);
        assert(graph.owned_global_indices[0] == 2);
        assert(graph.owned_global_indices[1] == 3);
        
        // Ghosts: 1 (needed by 2)
        assert(graph.ghost_global_indices.size() == 1);
        assert(graph.ghost_global_indices[0] == 1);
        
        // Block sizes
        // Local 0 (G2): 30
        // Local 1 (G3): 40
        // Local 2 (G1): 20
        assert(graph.block_sizes[0] == 30);
        assert(graph.block_sizes[1] == 40);
        assert(graph.block_sizes[2] == 20);
        
        std::cout << "Rank 1 passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
