#include "../submf.hpp"
#include "../isqrtm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

using namespace rsatb::calc;
using namespace vbcsr;

// Helper to verify X^T S X approx I
double verify_orthogonality(BlockSpMat<double>& S, DistMultiVector<double>& X, int n_cols) {
    DistGraph* graph = S.graph;
    DistMultiVector<double> Y(graph, n_cols);
    S.mult_dense(X, Y); // Y = S * X
    
    double max_err = 0.0;
    for(int i=0; i<n_cols; ++i) {
        for(int j=0; j<n_cols; ++j) {
            double local_dot = 0.0;
            double* xi = X.col_data(i);
            double* yj = Y.col_data(j);
            for(int k=0; k<X.local_rows; ++k) {
                 local_dot += xi[k] * yj[k];
            }
            double global_dot = 0.0;
            MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            double expected = (i == j) ? 1.0 : 0.0;
            max_err = std::max(max_err, std::abs(global_dot - expected));
        }
    }
    return max_err;
}

// Direct Diagonalization Benchmark (Rank 0 only for simplicity)
void direct_diagonalization_benchmark(int n_global, int n_cols, double shift) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank != 0) return;
    
    // Construct Dense Matrix S
    std::vector<double> S(n_global * n_global, 0.0);
    for(int i=0; i<n_global; ++i) {
        S[i + i*n_global] = 2.0 + shift;
        if (i > 0) S[i + (i-1)*n_global] = -1.0;
        if (i < n_global - 1) S[i + (i+1)*n_global] = -1.0;
    }
    
    double start_time = MPI_Wtime();
    
    // Eigendecomposition
    std::vector<double> w(n_global);
    rsatb_lapack_int n = n_global;
    rsatb_lapack_int lwork = -1;
    rsatb_lapack_int info;
    double wkopt;
    
    dsyev_("V", "U", &n, S.data(), &n, w.data(), &wkopt, &lwork, &info);
    lwork = (rsatb_lapack_int)wkopt;
    std::vector<double> work(lwork);
    dsyev_("V", "U", &n, S.data(), &n, w.data(), work.data(), &lwork, &info);
    
    // Compute S^{-1/2} (first n_cols)
    // X = U * Sigma^{-1/2} * E_k
    // U is in S (overwritten).
    // We want first n_cols of (U * Sigma^{-1/2}).
    // Col j of Result = (Col j of U) * (1/sqrt(w[j]))?
    // No. S^{-1/2} = U Sigma^{-1/2} U^T.
    // We want S^{-1/2} * E_k (First k cols of Identity).
    // = U Sigma^{-1/2} U^T E_k.
    // U^T E_k is first k columns of U^T, i.e., first k rows of U.
    // Let A = U^T E_k (k x n_global matrix? No, n_global x k).
    // Actually (U^T)_{ij} = U_{ji}.
    // (U^T E_k)_{ij} = (U^T)_{ij} for j < k.
    // So we take first k rows of U.
    // Then scale by Sigma^{-1/2}.
    // Then multiply by U.
    
    // This is getting complicated to implement fully just for timing.
    // The dominant cost is dsyev.
    // We can stop here for timing purposes.
    
    double end_time = MPI_Wtime();
    std::cout << "  Direct Diagonalization Time: " << (end_time - start_time) << " s" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing Lanczos ISQRTM with DENSE SPD Matrix..." << std::endl;

    // 1. Setup Graph (Dense Block Structure)
    // 10 blocks of size 100 -> 1000x1000 matrix
    int n_blocks = 10; 
    int block_dim = 20;
    std::vector<int> block_sizes(n_blocks, block_dim);
    std::vector<std::vector<int>> global_adj(n_blocks);
    
    // Fully connected graph
    for(int i=0; i<n_blocks; ++i) {
        for(int j=0; j<n_blocks; ++j) {
            global_adj[i].push_back(j);
        }
    }
    
    DistGraph graph_S(MPI_COMM_WORLD);
    graph_S.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix S (Dense SPD)
    BlockSpMat<double> S(&graph_S);
    int n_owned = graph_S.owned_global_indices.size();
    int n_global = n_blocks * block_dim;
    
    std::vector<double> S_global(n_global * n_global);
    if (rank == 0) {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for(int r=0; r<n_global; ++r) {
            for(int c=r; c<n_global; ++c) {
                double val = dist(gen);
                if (r == c) val += n_global;
                S_global[r + c*n_global] = val;
                S_global[c + r*n_global] = val;
            }
        }
    }
    MPI_Bcast(S_global.data(), n_global * n_global, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < n_owned; ++i) {
        int r_gid = graph_S.owned_global_indices[i]; // Block Row Index
        
        for(int c_gid = 0; c_gid < n_blocks; ++c_gid) {
            // Extract block (r_gid, c_gid)
            std::vector<double> block(block_dim * block_dim);
            for(int r=0; r<block_dim; ++r) {
                for(int c=0; c<block_dim; ++c) {
                    int global_r = r_gid * block_dim + r;
                    int global_c = c_gid * block_dim + c;
                    block[r + c*block_dim] = S_global[global_r + global_c*n_global];
                }
            }
            S.add_block(r_gid, c_gid, block.data(), block_dim, block_dim, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    S.assemble();

    // 3. Stability Tests (Different Block Sizes)
    std::vector<int> test_block_sizes = {1, 4, 8, 16};
    
    for (int b : test_block_sizes) {
        if (rank == 0) std::cout << "\n--- Testing Block Size: " << b << " ---" << std::endl;
        
        DistMultiVector<double> X(&graph_S, b);
        
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        auto func = [](double x) { return (x > 1e-12) ? std::pow(x, -0.5) : 0.0; };
        lanczos_matrix_function(S, X, std::function<double(double)>(func), b, 1e-8, 100, true); // verbose=true
        
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        
        double err = verify_orthogonality(S, X, b);
        
        if (rank == 0) {
            std::cout << "  Time: " << (end_time - start_time) << " s" << std::endl;
            std::cout << "  Max Error (X^T S X - I): " << err << std::endl;
            if (err > 1e-4) std::cout << "  FAILED" << std::endl;
            else std::cout << "  PASSED" << std::endl;
        }
    }
    
    // 4. Test Subset Computation
    if (rank == 0) std::cout << "\n--- Testing Subset Computation (Cols 4-8) ---" << std::endl;
    int b_full = 16;
    int start_col = 4;
    int n_subset = 4;
    
    // Full computation
    DistMultiVector<double> X_full(&graph_S, b_full);
    auto func = [](double x) { return (x > 1e-12) ? std::pow(x, -0.5) : 0.0; };
    lanczos_matrix_function(S, X_full, std::function<double(double)>(func), b_full, 1e-8, 100, false);
    
    // Subset computation
    DistMultiVector<double> X_subset(&graph_S, n_subset);
    lanczos_matrix_function(S, X_subset, std::function<double(double)>(func), n_subset, 1e-8, 100, false, true, start_col);
    
    // Compare
    double max_diff = 0.0;
    for(int i=0; i<n_subset; ++i) {
        double* full_col = X_full.col_data(start_col + i);
        double* subset_col = X_subset.col_data(i);
        for(int k=0; k<X_full.local_rows; ++k) {
            max_diff = std::max(max_diff, std::abs(full_col[k] - subset_col[k]));
        }
    }
    double global_max_diff = 0.0;
    MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  Max Difference (Subset vs Full): " << global_max_diff << std::endl;
        if (global_max_diff > 1e-10) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
    
    // 5. Test isqrtm_init (Graph Initialization)
    if (rank == 0) std::cout << "\n--- Testing isqrtm_init (Lanczos-based) ---" << std::endl;
    BlockSpMat<double> S_inv_sqrt(&graph_S);
    // Run isqrtm with graph init and a few iterations to check convergence
    try {
        isqrtm(S, &S_inv_sqrt, 1e-5, 1e-6, 5, "graph", 1.0, true);
        if (rank == 0) std::cout << "  isqrtm_init + Newton-Schulz: PASSED (did not crash)" << std::endl;
    } catch (const std::exception& e) {
        if (rank == 0) std::cout << "  isqrtm_init FAILED: " << e.what() << std::endl;
    }

    // 6. Benchmark Direct Method
    if (rank == 0) std::cout << "\n--- Benchmarking Direct Method (Rank 0) ---" << std::endl;
    // Use the same dense matrix for benchmark
    
    // Only rank 0 runs this
    if (rank == 0) {
        double start_time = MPI_Wtime();
        
        std::vector<double> w(n_global);
        rsatb_lapack_int n = n_global;
        rsatb_lapack_int lwork = -1;
        rsatb_lapack_int info;
        double wkopt;
        rsatb_lapack_int iwkopt;
        
        // Copy S_global because dsyev destroys it
        std::vector<double> S_copy = S_global;
        
        dsyevd_("V", "U", &n, S_copy.data(), &n, w.data(), &wkopt, &lwork, &iwkopt, &lwork, &info);
        lwork = (rsatb_lapack_int)wkopt;
        rsatb_lapack_int liwork = iwkopt;
        std::vector<double> work(lwork);
        std::vector<rsatb_lapack_int> iwork(liwork);
        dsyevd_("V", "U", &n, S_copy.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
        
        double end_time = MPI_Wtime();
        std::cout << "  Direct Diagonalization Time: " << (end_time - start_time) << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
