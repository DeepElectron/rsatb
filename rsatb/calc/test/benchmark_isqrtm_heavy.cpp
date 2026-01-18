#include "../isqrtm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>

using namespace rsatb::calc;
using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Benchmarking Matrix ISQRT (VBCSR) with Heavy Blocks..." << std::endl;

    // Parameters to achieve ~3000 dimension per subgraph
    // Let's use Block Size = 100.
    // We need ~30 neighbors per row.
    // Let's create a banded matrix with bandwidth 15 (15 left + 15 right + diagonal = 31 blocks).
    // 31 * 100 = 3100 dimension.
    
    int n_blocks = 10; // Total rows. Keep it small so total time is manageable, but each row is heavy.
    int block_dim = 100;
    int bandwidth = 15;
    
    std::vector<int> block_sizes(n_blocks, block_dim);
    std::vector<std::vector<int>> global_adj(n_blocks);
    
    for(int i=0; i<n_blocks; ++i) {
        for(int j = std::max(0, i - bandwidth); j <= std::min(n_blocks - 1, i + bandwidth); ++j) {
            global_adj[i].push_back(j);
        }
    }
    
    DistGraph graph_S(MPI_COMM_WORLD);
    graph_S.construct_serial(n_blocks, block_sizes, global_adj);

    // Setup Matrix S (Diagonally dominant to be PD)
    BlockSpMat<double> S(&graph_S);
    int n_owned = graph_S.owned_global_indices.size();
    
    // Random generator
    std::mt19937 gen(1234 + rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < n_owned; ++i) {
        int gid = graph_S.owned_global_indices[i];
        
        // Diagonal block: Make it PD and large
        std::vector<double> diag_block(block_dim * block_dim);
        for(int k=0; k<block_dim*block_dim; ++k) diag_block[k] = dis(gen);
        
        // Make symmetric
        for(int r=0; r<block_dim; ++r) {
            for(int c=0; c<r; ++c) {
                diag_block[r*block_dim + c] = diag_block[c*block_dim + r];
            }
            diag_block[r*block_dim + r] += 200.0; // Shift to ensure PD
        }
        
        S.add_block(gid, gid, diag_block.data(), block_dim, block_dim, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        // Off-diagonal blocks
        for(int j : global_adj[gid]) {
            if (j == gid) continue;
            std::vector<double> off_block(block_dim * block_dim);
            for(int k=0; k<block_dim*block_dim; ++k) off_block[k] = dis(gen) * 0.01; // Small off-diagonal
            
            // We need S to be symmetric. 
            // Since we iterate all rows, we will visit (i, j) and (j, i).
            // To ensure symmetry, we can just add random values. 
            // But for this benchmark, we just need the structure and computational load.
            // The exact values don't matter for performance, only for convergence (which we might not check fully).
            // However, isqrtm expects symmetric input.
            // Let's just add them. The result might not be perfectly symmetric if we don't coordinate, 
            // but for Z0 initialization (subgraph method), it treats the subgraph as the problem.
            // Actually, Z0 init extracts a subgraph. If S is not symmetric, the subgraph might not be symmetric?
            // The subgraph extraction pulls rows. 
            // If S is not symmetric, M might not be symmetric.
            // dense_isqrt expects symmetric/hermitian.
            // So we should try to make S symmetric.
            // But constructing distributed symmetric random matrix is tedious.
            // Let's just use 0.0 for off-diagonals? No, then subgraph is just diagonal blocks.
            // We need the neighbors to be included.
            // The neighbors are determined by sparsity pattern.
            // So even if values are 0, if they are in the graph, they are included in the subgraph.
            // Wait, `extract_submatrix` uses the graph structure.
            // Yes.
            
            S.add_block(gid, j, off_block.data(), block_dim, block_dim, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    S.assemble();

    if (rank == 0) std::cout << "Matrix Assembled. Starting isqrtm..." << std::endl;

    BlockSpMat<double> S_inv_sqrt = S.duplicate(true);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run isqrtm (maxit=0 to only measure Z0 init if possible? No, Z0 is part of it.
    // We can set maxit=0 or 1.
    // The user asked about Z0 initialization.
    // Let's modify isqrtm to allow stopping after init? 
    // Or just run 1 iteration.
    isqrtm(S, &S_inv_sqrt, 1e-7, 1e-7, 0, true);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    if (rank == 0) {
        std::cout << "isqrtm (maxit=0) took " << elapsed.count() << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
