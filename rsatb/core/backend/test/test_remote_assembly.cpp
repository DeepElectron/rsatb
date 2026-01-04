#include "block_csr.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace rsatb::backend;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cout << "Test requires 2 processes" << std::endl;
        MPI_Finalize();
        return 0;
    }

    // Graph: 2 blocks. 0 connected to 1.
    // 0: 0, 1
    // 1: 0, 1
    std::vector<std::vector<int>> adj = {
        {0, 1},
        {0, 1}
    };
    std::vector<int> block_sizes = {2, 2};

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    
    // Initialize to 0
    std::fill(mat.val.begin(), mat.val.end(), 0.0);
    
    // Test Scenario:
    // 1. Rank 0 INSERTs 1.0 to (0,0) [Local]
    // 2. Rank 1 INSERTs 2.0 to (1,1) [Local]
    // 3. Rank 0 ADDs 0.5 to (1,1) [Remote]
    // 4. Rank 1 ADDs 0.5 to (0,0) [Remote]
    
    std::vector<double> block_ones(4, 1.0); 
    std::vector<double> block_twos(4, 2.0);
    std::vector<double> block_half(4, 0.5);
    
    if (rank == 0) {
        // Local INSERT
        mat.add_block(0, 0, block_ones.data(), 2, 2, AssemblyMode::INSERT);
        // Remote ADD
        mat.add_block(1, 1, block_half.data(), 2, 2, AssemblyMode::ADD);
    } else {
        // Local INSERT
        mat.add_block(1, 1, block_twos.data(), 2, 2, AssemblyMode::INSERT);
        // Remote ADD
        mat.add_block(0, 0, block_half.data(), 2, 2, AssemblyMode::ADD);
    }
    
    // Assemble
    mat.assemble();
    
    // Verify
    // Block(0,0): INSERT 1.0 + ADD 0.5 = 1.5
    // Block(1,1): INSERT 2.0 + ADD 0.5 = 2.5
    
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            int gid_c;
            if (lid_c < n_owned) gid_c = graph.owned_global_indices[lid_c];
            else gid_c = graph.ghost_global_indices[lid_c - n_owned];
            
            if (gid_r == gid_c) { 
                size_t offset = mat.blk_ptr[k];
                double expected = (gid_r == 0) ? 1.5 : 2.5;
                
                for (int n = 0; n < 4; ++n) {
                    if (std::abs(mat.val[offset + n] - expected) > 1e-9) {
                        std::cout << "Rank " << rank << " Failed at " << gid_r << "," << gid_c 
                                  << " val=" << mat.val[offset+n] << " expected=" << expected << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
            }
        }
    }
    
    if (rank == 0) std::cout << "Remote Assembly (INSERT/ADD) Passed" << std::endl;

    // ---------------------------------------------------------
    // Verify Matrix-Vector Multiplication
    // ---------------------------------------------------------
    
    DistVector<double> x(&graph), y(&graph);
    // Initialize x to 1.0
    std::fill(x.data.begin(), x.data.end(), 1.0); // Sets owned + ghosts (if sized)
    // Actually DistVector constructor only sizes owned part?
    // Let's check DistVector.
    // DistVector constructor: data.resize(local_size + ghost_size).
    // So we can fill it.
    
    // y = A * x
    // A has diagonal blocks:
    // Block(0,0) [Rank 0]: 2x2 matrix of 1.5s
    // Block(1,1) [Rank 1]: 2x2 matrix of 2.5s
    // Off-diagonals are 0.
    
    // Expected y:
    // Rank 0 (Rows 0,1): 1.5*1 + 1.5*1 = 3.0
    // Rank 1 (Rows 2,3): 2.5*1 + 2.5*1 = 5.0
    
    mat.mult(x, y);
    
    double expected_val = (rank == 0) ? 3.0 : 5.0;
    for (int i = 0; i < 2; ++i) { // 2 rows per rank
        if (std::abs(y.data[i] - expected_val) > 1e-9) {
            std::cout << "Rank " << rank << " MatVec Failed at local row " << i 
                      << " val=" << y.data[i] << " expected=" << expected_val << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    if (rank == 0) std::cout << "MatVec Verification Passed" << std::endl;

    // ---------------------------------------------------------
    // Verify Matrix-MultiVector Multiplication
    // ---------------------------------------------------------
    
    int num_vecs = 2;
    DistMultiVector<double> X(&graph, num_vecs), Y(&graph, num_vecs);
    
    // Initialize X
    // Col 0: 1.0
    // Col 1: 2.0
    for (int c = 0; c < num_vecs; ++c) {
        double val = (c == 0) ? 1.0 : 2.0;
        double* col_data = X.col_data(c);
        int total_len = X.local_rows + X.ghost_rows;
        for(int i=0; i<total_len; ++i) col_data[i] = val;
    }
    
    mat.mult_dense(X, Y);
    
    // Expected Y:
    // Col 0: Same as y (3.0 or 5.0)
    // Col 1: Double (6.0 or 10.0)
    
    for (int c = 0; c < num_vecs; ++c) {
        double base_expected = (rank == 0) ? 3.0 : 5.0;
        double expected = (c == 0) ? base_expected : base_expected * 2.0;
        
        double* col_data = Y.col_data(c);
        for (int i = 0; i < 2; ++i) {
            if (std::abs(col_data[i] - expected) > 1e-9) {
                std::cout << "Rank " << rank << " MatMat Failed at col " << c << " local row " << i 
                          << " val=" << col_data[i] << " expected=" << expected << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    if (rank == 0) std::cout << "MatMat Verification Passed" << std::endl;
}
