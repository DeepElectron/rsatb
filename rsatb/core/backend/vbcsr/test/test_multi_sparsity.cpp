#include "../block_csr.hpp"
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

    // Graph 1: 1D Stencil (Nearest Neighbor)
    // 0: 0, 1
    // 1: 0, 1, 2
    // 2: 1, 2, 3
    // 3: 2, 3
    std::vector<std::vector<int>> adj1 = {
        {0, 1},
        {0, 1, 2},
        {1, 2, 3},
        {2, 3}
    };
    std::vector<int> block_sizes = {2, 2, 2, 2};

    DistGraph graph1(MPI_COMM_WORLD);
    graph1.construct_serial(4, block_sizes, adj1);

    // Graph 2: All-to-All (Dense)
    // 0: 0, 1, 2, 3
    // ...
    std::vector<std::vector<int>> adj2(4);
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) adj2[i].push_back(j);
    }

    DistGraph graph2(MPI_COMM_WORLD);
    graph2.construct_serial(4, block_sizes, adj2);

    // Verify owned structure matches
    assert(graph1.owned_global_indices == graph2.owned_global_indices);
    
    // Matrices
    BlockSpMat<double> mat1(&graph1);
    BlockSpMat<double> mat2(&graph2);
    
    // Fill mat1 with 1.0
    for(size_t i=0; i<mat1.val.size(); ++i) mat1.val[i] = 1.0;
    
    // Fill mat2 with 2.0
    for(size_t i=0; i<mat2.val.size(); ++i) mat2.val[i] = 2.0;
    
    // Vectors
    DistVector<double> x(&graph1); // Initially bound to graph1
    DistVector<double> y(&graph1);
    
    x.set_constant(1.0);
    y.set_constant(0.0);
    
    // 1. Apply Mat1 (Graph1)
    // Expected: Same as previous test.
    // Rank 0 (Block 0): Neighbors {0, 1}. 2 blocks * 1.0 * [1,1] = [4, 4]
    // Rank 0 (Block 1): Neighbors {0, 1, 2}. 3 blocks * 1.0 * [1,1] = [6, 6]
    mat1.mult_optimized(x, y);
    
    double* y_ptr = y.local_data();
    if (rank == 0) {
        assert(std::abs(y_ptr[0] - 4.0) < 1e-9);
        assert(std::abs(y_ptr[2] - 6.0) < 1e-9);
    }
    
    // 2. Apply Mat2 (Graph2) -> Should auto-bind to graph2
    // Graph2 is dense.
    // Rank 0 (Block 0): Neighbors {0, 1, 2, 3}. 4 blocks * 2.0 * [1,1] = 8 * 2 = 16?
    // Wait, x is [1,1]. A_block is 2x2 filled with 2.0.
    // A_block * x_block = [2 2; 2 2] * [1; 1] = [4; 4].
    // 4 neighbors. Total = 4 * [4, 4] = [16, 16].
    
    mat2.mult_optimized(x, y);
    
    // Check if x is bound to graph2
    assert(x.graph == &graph2);
    assert(y.graph == &graph2);
    
    y_ptr = y.local_data();
    if (rank == 0) {
        assert(std::abs(y_ptr[0] - 16.0) < 1e-9);
        assert(std::abs(y_ptr[2] - 16.0) < 1e-9);
        std::cout << "Rank 0 Multi-Sparsity Passed" << std::endl;
    } else {
        assert(std::abs(y_ptr[0] - 16.0) < 1e-9);
        assert(std::abs(y_ptr[2] - 16.0) < 1e-9);
        std::cout << "Rank 1 Multi-Sparsity Passed" << std::endl;
    }
    
    // 3. Apply Mat1 again -> Should switch back
    mat1.mult_optimized(x, y);
    assert(x.graph == &graph1);
    
    MPI_Finalize();
    return 0;
}
