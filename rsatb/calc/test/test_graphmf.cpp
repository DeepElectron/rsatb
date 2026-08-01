#include "vbcsr/core/detail/ops/spmf/graph_function.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

// The graph matrix function is vbcsr's (detail/ops/spmf/graph_function.hpp); it declares no
// rsatb::calc entities of its own, so only the vbcsr namespace is pulled in.
using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing Graph Matrix Function with exp(-x)..." << std::endl;

    // 1. Setup Graph (1D Chain)
    int n_blocks = 20; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) {
        global_adj[i].push_back(i);
        if (i > 0) global_adj[i].push_back(i-1);
        if (i < n_blocks - 1) global_adj[i].push_back(i+1);
    }
    
    DistGraph graph_S(MPI_COMM_WORLD);
    graph_S.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix S (Diagonal for simplicity to verify exp(-x))
    // S = diag(1, 2, ..., n_blocks)
    BlockSpMat<double> S(&graph_S);
    int n_owned = graph_S.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph_S.owned_global_indices[i];
        double val = (double)(gid + 1);
        S.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    S.assemble();

    // 3. Compute exp(-S)
    BlockSpMat<double> S_exp(&graph_S);
    auto func = [](double x) { return std::exp(-x); };
    
    vbcsr::graph_matrix_function(S, &S_exp, std::function<double(double)>(func), true);
    
    // 4. Verify: S_exp should be diag(exp(-1), exp(-2), ...)
    // Since S is diagonal, the result should be diagonal (approximately, due to numerical noise/method)
    // Actually, graph_matrix_function uses subgraph method which might introduce some off-diagonal elements 
    // if the subgraph includes neighbors. But here the graph is 1D chain, so neighbors are included.
    // However, for a diagonal matrix, the eigenvectors are standard basis vectors, so any function of it is diagonal.
    // The Lanczos method should respect this.
    
    double max_err = 0.0;
    
    // Check diagonal elements
    // We need to fetch diagonal blocks of S_exp
    // Since S_exp is distributed, we iterate owned blocks
    
    // S_exp might have off-diagonal blocks due to the method (it computes on subgraphs).
    // But for a diagonal input, the result should be very close to diagonal.
    
    // Let's check the diagonal values.
    // We can use get_block if we know it exists.
    // Or we can iterate local blocks.
    
    // For simplicity, let's just check the diagonal values if they exist locally.
    // S_exp is a BlockSpMat.
    
    // Check diagonal elements locally
    // We iterate owned rows and check the diagonal block
    
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph_S.owned_global_indices[i];
        
        // Find diagonal block (gid, gid)
        // We need to look into S_exp structure
        int start = S_exp.row_ptr()[i];
        int end = S_exp.row_ptr()[i+1];
        
        bool found = false;
        for (int k = start; k < end; ++k) {
            int col_lid = S_exp.col_ind()[k];
            int col_gid = graph_S.get_global_index(col_lid);
            
            if (col_gid == gid) {
                std::vector<double> block = S_exp.get_block(i, col_lid);
                double val = block[0];
                double expected = std::exp(-(double)(gid + 1));
                max_err = std::max(max_err, std::abs(val - expected));
                found = true;
            }
        }
        if (!found) {
             // If diagonal not found, it's effectively 0
             max_err = std::max(max_err, std::exp(-(double)(gid + 1)));
        }
    }
    
    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  Verification (Diagonal exp(-x)) Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-4) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
