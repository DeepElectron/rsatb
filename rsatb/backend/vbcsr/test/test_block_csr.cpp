#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace rsatb::backend;

#include "block_csr.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>

using namespace rsatb::backend;

// Helper to fill vector with random numbers
void fill_random(std::vector<double>& v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto& val : v) val = dis(gen);
}

template <typename Kernel>
void run_test(const std::string& kernel_name, int rank, int size) {
    if (rank == 0) std::cout << "Testing " << kernel_name << "..." << std::endl;

    // 1. Setup Graph with Diagonals
    // 0: 0, 1
    // 1: 0, 1, 2
    // 2: 1, 2, 3
    // 3: 2, 3
    std::vector<std::vector<int>> global_adj = {
        {0, 1},
        {0, 1, 2},
        {1, 2, 3},
        {2, 3}
    };
    
    // Random block sizes
    std::vector<int> block_sizes = {3, 2, 4, 3}; 
    int n_blocks = block_sizes.size();
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix and Reference Data
    BlockSpMat<double, Kernel> mat(&graph);
    
    // Generate global matrix data (map<pair<r,c>, vector<double>>)
    std::map<std::pair<int,int>, std::vector<double>> global_mat_data;
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int r = 0; r < n_blocks; ++r) {
        for (int c : global_adj[r]) {
            int rows = block_sizes[r];
            int cols = block_sizes[c];
            std::vector<double> block(rows * cols);
            for (auto& val : block) val = dis(gen);
            global_mat_data[{r, c}] = block;
        }
    }

    // Fill Local Matrix
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            // We need GID of lid_c. 
            // DistGraph doesn't expose local->global map easily for ghosts?
            // Wait, DistGraph has global_to_local. We need reverse.
            // For owned it's easy. For ghosts, we stored them in ghost_global_indices.
            int gid_c;
            if (lid_c < n_owned) {
                gid_c = graph.owned_global_indices[lid_c];
            } else {
                gid_c = graph.ghost_global_indices[lid_c - n_owned];
            }
            
            const auto& block = global_mat_data[{gid_r, gid_c}];
            // Use add_block to handle transposition
            int rows = block_sizes[gid_r];
            int cols = block_sizes[gid_c];
            mat.add_block(gid_r, gid_c, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }

    // 3. Setup Vectors and Reference
    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    
    // Global vectors
    std::vector<std::vector<double>> global_x(n_blocks);
    std::vector<std::vector<double>> global_y_ref(n_blocks);
    
    for (int i = 0; i < n_blocks; ++i) {
        global_x[i].resize(block_sizes[i]);
        for (auto& val : global_x[i]) val = dis(gen);
        global_y_ref[i].resize(block_sizes[i], 0.0);
    }
    
    // Compute Reference Y = A * X
    for (int r = 0; r < n_blocks; ++r) {
        for (int c : global_adj[r]) {
            const auto& block = global_mat_data[{r, c}];
            int rows = block_sizes[r];
            int cols = block_sizes[c];
            // Block is RowMajor
            // Reference calculation: Y = A * X
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    global_y_ref[r][i] += block[i * cols + j] * global_x[c][j];
                }
            }
        }
    }

    // Fill Distributed X
    double* x_ptr = x.local_data();
    int offset = 0;
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        std::memcpy(x_ptr + offset, global_x[gid].data(), block_sizes[gid] * sizeof(double));
        offset += block_sizes[gid];
    }
    y.set_constant(0.0);

    // 4. Run MatVec
    mat.mult_optimized(x, y);

    // 5. Verify MatVec
    double* y_ptr = y.local_data();
    offset = 0;
    double max_err = 0.0;
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        for (int k = 0; k < block_sizes[gid]; ++k) {
            double err = std::abs(y_ptr[offset + k] - global_y_ref[gid][k]);
            max_err = std::max(max_err, err);
        }
        offset += block_sizes[gid];
    }
    
    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  MatVec Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-12) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }

    // 6. Test MatMat (mult_dense)
    int num_vecs = 3;
    DistMultiVector<double> X(&graph, num_vecs);
    DistMultiVector<double> Y(&graph, num_vecs);
    
    // Global MultiVectors
    std::vector<std::vector<std::vector<double>>> global_X(n_blocks); // [block][col][row]
    std::vector<std::vector<std::vector<double>>> global_Y_ref(n_blocks);
    
    for (int i = 0; i < n_blocks; ++i) {
        global_X[i].resize(num_vecs);
        global_Y_ref[i].resize(num_vecs);
        for (int v = 0; v < num_vecs; ++v) {
            global_X[i][v].resize(block_sizes[i]);
            for (auto& val : global_X[i][v]) val = dis(gen);
            global_Y_ref[i][v].resize(block_sizes[i], 0.0);
        }
    }
    
    // Compute Reference Y = A * X
    for (int r = 0; r < n_blocks; ++r) {
        for (int c : global_adj[r]) {
            const auto& block = global_mat_data[{r, c}];
            int rows = block_sizes[r];
            int cols = block_sizes[c];
            
            for (int v = 0; v < num_vecs; ++v) {
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        global_Y_ref[r][v][i] += block[i * cols + j] * global_X[c][v][j];
                    }
                }
            }
        }
    }
    
    // Fill Distributed X
    // X is ColMajor: X(row, col)
    // Local storage: [col 0][col 1]... where each col is contiguous local+ghost
    // DistMultiVector stores data as: col 0 (all rows), col 1 (all rows)...
    // We need to fill owned parts of each column.
    
    for (int v = 0; v < num_vecs; ++v) {
        double* col_ptr = X.col_data(v);
        offset = 0;
        for (int i = 0; i < n_owned; ++i) {
            int gid = graph.owned_global_indices[i];
            std::memcpy(col_ptr + offset, global_X[gid][v].data(), block_sizes[gid] * sizeof(double));
            offset += block_sizes[gid];
        }
    }
    
    // Initialize Y
    for (int v = 0; v < num_vecs; ++v) {
        double* col_ptr = Y.col_data(v);
        std::fill(col_ptr, col_ptr + Y.local_rows + Y.ghost_rows, 0.0);
    }
    
    mat.mult_dense(X, Y);
    
    // Verify MatMat
    max_err = 0.0;
    for (int v = 0; v < num_vecs; ++v) {
        double* col_ptr = Y.col_data(v);
        offset = 0;
        for (int i = 0; i < n_owned; ++i) {
            int gid = graph.owned_global_indices[i];
            for (int k = 0; k < block_sizes[gid]; ++k) {
                double err = std::abs(col_ptr[offset + k] - global_Y_ref[gid][v][k]);
                max_err = std::max(max_err, err);
            }
            offset += block_sizes[gid];
        }
    }
    
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  MatMat Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-12) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    run_test<NaiveKernel<double>>("NaiveKernel", rank, size);
    
    // Uncomment to test BLAS if available
    run_test<BLASKernel>("BLASKernel", rank, size);

    MPI_Finalize();
    return 0;
}
