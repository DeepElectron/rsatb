#include "../block_csr.hpp"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace rsatb::backend;

// Use BLAS Kernel for performance
using Kernel = BLASKernel;

// Deterministic value generators
double get_mat_val(int global_row, int global_col, int r_idx, int c_idx) {
    // Value at block (global_row, global_col), element (r_idx, c_idx)
    // Simple deterministic function
    return std::sin(global_row * 1000.0 + r_idx) * std::cos(global_col * 1000.0 + c_idx);
}

double get_vec_val(int global_col, int c_idx, int vec_idx = 0) {
    return std::cos(global_col * 1000.0 + c_idx + vec_idx);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Benchmark Parameters
    int n_global_blocks = 1000; // Reduced default for quick verification
    int block_size = 50;        
    int n_vecs = 5;             
    int n_iter = 10;            

    if (argc > 1) n_global_blocks = std::atoi(argv[1]);
    if (argc > 2) block_size = std::atoi(argv[2]);

    if (rank == 0) {
        std::cout << "Benchmark & Verification Configuration:" << std::endl;
        std::cout << "  Ranks: " << size << std::endl;
        std::cout << "  Global Blocks: " << n_global_blocks << std::endl;
        std::cout << "  Block Size: " << block_size << std::endl;
        std::cout << "  RHS Vectors: " << n_vecs << std::endl;
        std::cout << "  Iterations: " << n_iter << std::endl;
    }

    // 1. Distributed Graph Construction
    // 1D Stencil: i connected to i-1, i, i+1
    
    int blocks_per_rank = n_global_blocks / size;
    int remainder = n_global_blocks % size;
    
    int my_start = rank * blocks_per_rank + std::min(rank, remainder);
    int my_count = blocks_per_rank + (rank < remainder ? 1 : 0);
    int my_end = my_start + my_count;

    std::vector<int> my_owned_indices(my_count);
    std::vector<int> my_block_sizes(my_count, block_size);
    std::vector<std::vector<int>> my_adj(my_count);

    for (int i = 0; i < my_count; ++i) {
        int gid = my_start + i;
        my_owned_indices[i] = gid;
        
        // Neighbors
        if (gid > 0) my_adj[i].push_back(gid - 1);
        my_adj[i].push_back(gid); // Self
        if (gid < n_global_blocks - 1) my_adj[i].push_back(gid + 1);
    }

    double t0 = MPI_Wtime();
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_distributed(my_owned_indices, my_block_sizes, my_adj);
    double t_graph = MPI_Wtime() - t0;
    
    if (rank == 0) std::cout << "Graph Construction Time: " << t_graph << " s" << std::endl;

    // 2. Matrix Assembly
    BlockSpMat<double, Kernel> mat(&graph);
    
    // Fill with deterministic values
    int n_owned = graph.owned_global_indices.size();
    
    t0 = MPI_Wtime();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        for (int k = start; k < end; ++k) {
            size_t offset = mat.blk_ptr[k];
            int lid_c = mat.col_ind[k];
            
            // Resolve GID for column
            int gid_c;
            if (lid_c < n_owned) {
                gid_c = graph.owned_global_indices[lid_c];
            } else {
                gid_c = graph.ghost_global_indices[lid_c - n_owned];
            }
            
            int r_dim = block_size;
            int c_dim = block_size; // Uniform
            
            for (int c = 0; c < c_dim; ++c) {
                for (int r = 0; r < r_dim; ++r) {
                    mat.val[offset + c * r_dim + r] = get_mat_val(gid_r, gid_c, r, c);
                }
            }
        }
    }
    double t_assembly = MPI_Wtime() - t0;
    if (rank == 0) std::cout << "Matrix Assembly Time: " << t_assembly << " s" << std::endl;

    // 3. MatVec Benchmark & Verify
    DistVector<double> x(&graph), y(&graph);
    
    // Fill x
    double* x_ptr = x.local_data();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        for (int k = 0; k < block_size; ++k) {
            x_ptr[i * block_size + k] = get_vec_val(gid, k);
        }
    }
    y.set_constant(0.0);
    
    // Warmup & Verify
    mat.mult_optimized(x, y);
    
    // Verification
    double max_err = 0.0;
    double* y_ptr = y.local_data();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        
        // Compute reference for this block row
        std::vector<double> y_ref(block_size, 0.0);
        
        // Neighbors: gid_r-1, gid_r, gid_r+1
        std::vector<int> neighbors;
        if (gid_r > 0) neighbors.push_back(gid_r - 1);
        neighbors.push_back(gid_r);
        if (gid_r < n_global_blocks - 1) neighbors.push_back(gid_r + 1);
        
        for (int gid_c : neighbors) {
            for (int c = 0; c < block_size; ++c) {
                double x_val = get_vec_val(gid_c, c);
                for (int r = 0; r < block_size; ++r) {
                    double a_val = get_mat_val(gid_r, gid_c, r, c);
                    y_ref[r] += a_val * x_val;
                }
            }
        }
        
        for (int r = 0; r < block_size; ++r) {
            double err = std::abs(y_ptr[i * block_size + r] - y_ref[r]);
            max_err = std::max(max_err, err);
        }
    }
    
    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "MatVec Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-12) std::cout << "  VERIFICATION FAILED" << std::endl;
        else std::cout << "  VERIFICATION PASSED" << std::endl;
    }

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; ++i) {
        mat.mult_optimized(x, y);
    }
    double t_matvec = MPI_Wtime() - t0;
    double avg_matvec = t_matvec / n_iter;
    
    if (rank == 0) {
        std::cout << "MatVec Time (avg): " << avg_matvec << " s" << std::endl;
        double flops = (double)n_global_blocks * 3.0 * 2.0 * block_size * block_size;
        std::cout << "MatVec GFLOPS: " << (flops / avg_matvec) * 1e-9 << std::endl;
    }

    // 4. MatMat Benchmark & Verify
    DistMultiVector<double> X(&graph, n_vecs);
    DistMultiVector<double> Y(&graph, n_vecs);
    
    // Init X
    for (int v = 0; v < n_vecs; ++v) {
        double* col = X.col_data(v);
        for (int i = 0; i < n_owned; ++i) {
            int gid = graph.owned_global_indices[i];
            for (int k = 0; k < block_size; ++k) {
                col[i * block_size + k] = get_vec_val(gid, k, v);
            }
        }
    }
    
    // Warmup & Verify
    mat.mult_dense(X, Y);
    
    // Verification
    max_err = 0.0;
    for (int v = 0; v < n_vecs; ++v) {
        double* col_ptr = Y.col_data(v);
        for (int i = 0; i < n_owned; ++i) {
            int gid_r = graph.owned_global_indices[i];
            
            // Compute reference
            std::vector<double> y_ref(block_size, 0.0);
            std::vector<int> neighbors;
            if (gid_r > 0) neighbors.push_back(gid_r - 1);
            neighbors.push_back(gid_r);
            if (gid_r < n_global_blocks - 1) neighbors.push_back(gid_r + 1);
            
            for (int gid_c : neighbors) {
                for (int c = 0; c < block_size; ++c) {
                    double x_val = get_vec_val(gid_c, c, v);
                    for (int r = 0; r < block_size; ++r) {
                        double a_val = get_mat_val(gid_r, gid_c, r, c);
                        y_ref[r] += a_val * x_val;
                    }
                }
            }
            
            for (int r = 0; r < block_size; ++r) {
                double err = std::abs(col_ptr[i * block_size + r] - y_ref[r]);
                max_err = std::max(max_err, err);
            }
        }
    }
    
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "MatMat Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-12) std::cout << "  VERIFICATION FAILED" << std::endl;
        else std::cout << "  VERIFICATION PASSED" << std::endl;
    }
    
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; ++i) {
        mat.mult_dense(X, Y);
    }
    double t_matmat = MPI_Wtime() - t0;
    double avg_matmat = t_matmat / n_iter;
    
    if (rank == 0) {
        std::cout << "MatMat Time (avg): " << avg_matmat << " s" << std::endl;
        double flops = (double)n_global_blocks * 3.0 * 2.0 * block_size * block_size * n_vecs;
        std::cout << "MatMat GFLOPS: " << (flops / avg_matmat) * 1e-9 << std::endl;
    }

    MPI_Finalize();
    return 0;
}
