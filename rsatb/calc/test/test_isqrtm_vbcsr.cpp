#include "../isqrtm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

// using namespace rsatb::backend;
using namespace rsatb::calc;
using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing Matrix ISQRT (VBCSR) with PD Matrix..." << std::endl;

    // 1. Setup Graph (1D Chain)
    int n_blocks = 2000; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) {
        global_adj[i].push_back(i);
        if (i > 0) global_adj[i].push_back(i-1);
        if (i < n_blocks - 1) global_adj[i].push_back(i+1);
    }
    
    DistGraph graph_S(MPI_COMM_WORLD);
    graph_S.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix S (1D Laplacian + Shift)
    // A = L + shift*I
    double shift = 1.0;
    BlockSpMat<double> S(&graph_S);
    int n_owned = graph_S.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph_S.owned_global_indices[i];
        
        double val = 2.0 + shift;
        S.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        val = -1.0;
        if (gid > 0) S.add_block(gid, gid-1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        if (gid < n_blocks - 1) S.add_block(gid, gid+1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    S.assemble();

    // 3. Compute S^{-1/2}
    double max_eig_L = 2.0 - 2.0 * std::cos(n_blocks * M_PI / (n_blocks + 1));
    double maxeig = max_eig_L + shift;
    double factor = 1.0 / std::sqrt(maxeig + 0.1);
    
    if (rank == 0) {
        std::cout << "  Max Eigenvalue (approx): " << maxeig << std::endl;
        std::cout << "  Calculated Factor: " << factor << std::endl;
    }

    BlockSpMat<double> S_inv_sqrt = S.duplicate(true);
    
    // isqrtm_vbcsr(S, S_inv_sqrt, tol, sptol, maxit, factor, verbose)
    rsatb::calc::inverse_sqrt(S, &S_inv_sqrt, {1e-7, 1e-7, 50}, true);
    // 4. Verify: S_inv_sqrt * S * S_inv_sqrt * v should be v
    double max_err = 0.0;
    int n_tests = 5;
    std::mt19937 gen(1337 + rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int t = 0; t < n_tests; ++t) {
        DistVector<double> v(&graph_S);
        for (int i = 0; i < v.local_size; ++i) v[i] = dis(gen);
        
        DistVector<double> tmp1(S_inv_sqrt.graph);
        DistVector<double> tmp2(&graph_S);
        DistVector<double> v_out(&graph_S);
        
        // tmp1 = S_inv_sqrt * v
        S_inv_sqrt.mult_optimized(v, tmp1);
        // tmp2 = S * tmp1
        S.mult_optimized(tmp1, tmp2);
        // v_out = S_inv_sqrt * tmp2
        S_inv_sqrt.mult_adjoint(tmp2, v_out);
        
        // Check v_out - v
        for (int i = 0; i < v.local_size; ++i) {
            max_err = std::max(max_err, std::abs(v_out[i] - v[i]));
        }
    }
    
    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  Verification (S^-1/2 * S * S^-1/2 * v - v) Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-4) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
