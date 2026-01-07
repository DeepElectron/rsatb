#include "../dist_vector.hpp"
#include "../dist_multivector.hpp"
#include <iostream>
#include <complex>
#include <cassert>

using namespace rsatb::backend;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing DistVector<complex>..." << std::endl;

    // 1. Setup Graph
    // 2 blocks, 1 per rank (if size=2)
    // 0: 0, 1
    // 1: 0, 1
    std::vector<std::vector<int>> global_adj = {{0, 1}, {0, 1}};
    std::vector<int> block_sizes = {2, 2};
    int n_blocks = 2;

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Test DistVector<complex>
    DistVector<std::complex<double>> v(&graph);
    
    // Fill owned data
    // Rank 0 owns block 0 (indices 0, 1) -> set to (1+i, 2+2i)
    // Rank 1 owns block 1 (indices 0, 1 local) -> set to (3+3i, 4+4i)
    
    std::complex<double>* data = v.local_data();
    for (int i = 0; i < v.local_size; ++i) {
        double val = (rank * 2 + i + 1);
        data[i] = std::complex<double>(val, val);
    }
    
    // Sync ghosts
    v.sync_ghosts();
    
    // Verify
    // Rank 0 should have ghost block 1: (3+3i, 4+4i)
    // Rank 1 should have ghost block 0: (1+i, 2+2i)
    
    // Ghost data starts at v.local_size
    // Map: ghost_global_indices[k] -> local index local_size + k
    
    bool passed = true;
    for (int i = 0; i < graph.ghost_global_indices.size(); ++i) {
        int gid = graph.ghost_global_indices[i];
        int offset = 0; // Offset within ghost part? No, DistVector is flat.
        // We need to know where this block starts in the ghost part.
        // DistVector stores ghosts in order of ghost_global_indices?
        // No, DistVector allocates local_size + ghost_size.
        // The ghosts are stored after local_size.
        // But in what order?
        // DistGraph::recv_indices tells us which LOCAL indices to receive.
        // Wait, DistGraph::recv_indices are indices in the ADJACENCY list?
        // No, let's check DistGraph.
        
        // DistGraph::recv_indices are LOCAL indices to receive into.
        // But DistVector::sync_ghosts uses them to unpack.
        // It unpacks into `data.data() + blk_offset`.
        // `blk_offset` is computed from `block_sizes`.
        // `recv_indices` in DistGraph are indices of BLOCKS in the local block array.
        // Local block array: [Owned Blocks... | Ghost Blocks...]
        // So `recv_indices` point to Ghost Blocks (indices >= n_owned).
        
        // So we can check the values directly if we know the block index.
        // For this simple case:
        // Rank 0: Owned Block 0. Ghost Block 1.
        // Rank 1: Owned Block 1. Ghost Block 0.
        
        int ghost_blk_idx = -1;
        if (rank == 0) ghost_blk_idx = 1; // Block 1 is ghost
        else ghost_blk_idx = 0; // Block 0 is ghost (which is index 1 in local block array? No)
        
        // In construct_serial:
        // owned_global_indices are [start, end).
        // ghosts are identified from adj.
        // For Rank 0: adj of 0 is {0, 1}. 0 is owned. 1 is ghost.
        // So local blocks: 0 (Global 0), 1 (Global 1).
        // For Rank 1: adj of 1 is {0, 1}. 1 is owned. 0 is ghost.
        // So local blocks: 0 (Global 1), 1 (Global 0).
        
        // So for Rank 0: data[2], data[3] should be Block 1 (3+3i, 4+4i)
        // For Rank 1: data[2], data[3] should be Block 0 (1+i, 2+2i)
        
        std::complex<double> expected1, expected2;
        if (rank == 0) {
            expected1 = {3.0, 3.0};
            expected2 = {4.0, 4.0};
        } else {
            expected1 = {1.0, 1.0};
            expected2 = {2.0, 2.0};
        }
        
        if (v[2] != expected1 || v[3] != expected2) {
            std::cout << "Rank " << rank << " Failed DistVector check." << std::endl;
            std::cout << "Got " << v[2] << ", " << v[3] << std::endl;
            passed = false;
        }
    }
    
    if (passed && rank == 0) std::cout << "DistVector<complex> PASSED" << std::endl;
    
    // 3. Test DistMultiVector<complex>
    DistMultiVector<std::complex<double>> mv(&graph, 2);
    // Fill
    for (int c = 0; c < 2; ++c) {
        std::complex<double>* col = mv.col_data(c);
        for (int i = 0; i < mv.local_rows; ++i) {
            double val = (rank * 2 + i + 1) + c * 10;
            col[i] = std::complex<double>(val, val);
        }
    }
    
    mv.sync_ghosts();
    
    // Verify
    // Rank 0, Col 0: Ghost (3+3i, 4+4i)
    // Rank 0, Col 1: Ghost (13+13i, 14+14i)
    
    if (rank == 0) {
        std::complex<double>* col0 = mv.col_data(0);
        std::complex<double>* col1 = mv.col_data(1);
        
        if (col0[2] != std::complex<double>(3,3) || col0[3] != std::complex<double>(4,4)) {
             std::cout << "Rank 0 Failed DistMultiVector Col 0 check." << std::endl;
             passed = false;
        }
        if (col1[2] != std::complex<double>(13,13) || col1[3] != std::complex<double>(14,14)) {
             std::cout << "Rank 0 Failed DistMultiVector Col 1 check." << std::endl;
             passed = false;
        }
    }
    
    if (passed && rank == 0) std::cout << "DistMultiVector<complex> PASSED" << std::endl;

    // 4. Test Conjugate and Dot
    DistVector<std::complex<double>> v2(&graph);
    std::complex<double>* d2 = v2.local_data();
    for(int i=0; i<v2.local_size; ++i) d2[i] = std::complex<double>(1, 1);
    
    // Dot with itself: sum conj(x)*x = |x|^2. Should be real.
    // Local size is 2. 2 elements of (1,1). |1+i|^2 = 2.
    // Total elements = 4 (2 ranks * 2).
    // Dot = 4 * 2 = 8.
    std::complex<double> dot_val = v2.dot(v2);
    if (rank == 0) {
        if (std::abs(dot_val - 8.0) > 1e-12) {
             std::cout << "Dot Product Failed. Expected 8.0, got " << dot_val << std::endl;
             passed = false;
        } else {
             std::cout << "Dot Product PASSED" << std::endl;
        }
    }
    
    // Conjugate
    v2.conjugate();
    // Should be (1, -1)
    for(int i=0; i<v2.local_size; ++i) {
        if (d2[i] != std::complex<double>(1, -1)) {
            std::cout << "Rank " << rank << " Conjugate Failed." << std::endl;
            passed = false;
        }
    }
    if (passed && rank == 0) std::cout << "Conjugate PASSED" << std::endl;

    if (passed && rank == 0) std::cout << "Conjugate PASSED" << std::endl;

    // 5. Test bdot
    DistMultiVector<std::complex<double>> mv2(&graph, 2);
    // Fill with (1,1)
    for (int c = 0; c < 2; ++c) {
        std::complex<double>* col = mv2.col_data(c);
        for (int i = 0; i < mv2.local_rows; ++i) {
            col[i] = std::complex<double>(1, 1);
        }
    }
    
    // bdot(mv2) -> [8, 8]
    std::vector<std::complex<double>> dots = mv2.bdot(mv2);
    if (rank == 0) {
        bool bdot_passed = true;
        if (dots.size() != 2) bdot_passed = false;
        if (std::abs(dots[0] - 8.0) > 1e-12) bdot_passed = false;
        if (std::abs(dots[1] - 8.0) > 1e-12) bdot_passed = false;
        
        if (bdot_passed) std::cout << "bdot PASSED" << std::endl;
        else std::cout << "bdot FAILED: " << dots[0] << ", " << dots[1] << std::endl;
    }

    MPI_Finalize();
    return 0;
}
