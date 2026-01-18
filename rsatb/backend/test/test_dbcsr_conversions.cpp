#include "../vbcsr/block_csr.hpp"
#include "../dbcsr_conversions.hpp"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <random>
#include <cmath>
#include <typeinfo>

using namespace rsatb::backend;

template <typename T>
void run_test(int rank, int size) {
    if (rank == 0) std::cout << "Testing BlockSpMat <-> DBCSR Conversion for T=" << typeid(T).name() << "..." << std::endl;

    // 1. Setup Graph
    std::vector<std::vector<int>> global_adj = {
        {0, 1},
        {0, 1, 2},
        {1, 2, 3},
        {2, 3}
    };
    
    std::vector<int> block_sizes = {3, 2, 4, 3}; 
    int n_blocks = block_sizes.size();
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Original Matrix
    BlockSpMat<T, NaiveKernel<T>> mat(&graph);
    
    // Fill with random data
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            int gid_c;
            if (lid_c < n_owned) {
                gid_c = graph.owned_global_indices[lid_c];
            } else {
                gid_c = graph.ghost_global_indices[lid_c - n_owned];
            }
            
            int rows = block_sizes[gid_r];
            int cols = block_sizes[gid_c];
            std::vector<T> block(rows * cols);
            for (auto& val : block) {
                if constexpr (std::is_same<T, double>::value) {
                    val = dis(gen);
                } else {
                    val = std::complex<double>(dis(gen), dis(gen));
                }
            }
            
            // Add block (RowMajor input)
            mat.add_block(gid_r, gid_c, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    mat.assemble(); 
    
    // 3. Convert to DBCSR
    dbcsr_matrix m_dbcsr;
    BlockSpMat2DBCSR(mat, &m_dbcsr);
    
    // 4. Convert back to NEW BlockSpMat
    DistGraph graph2(MPI_COMM_WORLD);
    std::vector<int> my_owned = graph.owned_global_indices;
    std::vector<int> my_sizes;
    for(int gid : my_owned) my_sizes.push_back(block_sizes[gid]);
    std::vector<std::vector<int>> my_adj(n_owned); // Empty adj
    
    graph2.construct_distributed(my_owned, my_sizes, my_adj);
    
    BlockSpMat<T, NaiveKernel<T>> mat2(&graph2);
    DBCSR2BlockSpMat(m_dbcsr, mat2);
    
    // 5. Verify
    double max_err = 0.0;
    
    // Check structure match
    if (mat.row_ptr.size() != mat2.row_ptr.size()) {
        std::cout << "Rank " << rank << ": Row ptr size mismatch" << std::endl;
        max_err = 1.0;
    } else {
        for (size_t i = 0; i < mat.row_ptr.size(); ++i) {
            if (mat.row_ptr[i] != mat2.row_ptr[i]) {
                std::cout << "Rank " << rank << ": Row ptr mismatch at " << i << std::endl;
                max_err = 1.0;
                break;
            }
        }
    }
    
    if (max_err == 0.0) {
        // Robust check: iterate mat blocks, find in mat2
        for (int i = 0; i < n_owned; ++i) {
            int start = mat.row_ptr[i];
            int end = mat.row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int lid_c = mat.col_ind[k];
                int gid_c;
                if (lid_c < n_owned) {
                    gid_c = graph.owned_global_indices[lid_c];
                } else {
                    gid_c = graph.ghost_global_indices[lid_c - n_owned];
                }
                
                // Find in mat2 manually
                int idx2 = -1;
                int start2 = mat2.row_ptr[i];
                int end2 = mat2.row_ptr[i+1];
                for (int k2 = start2; k2 < end2; ++k2) {
                    int lid_c2 = mat2.col_ind[k2];
                    int gid_c2;
                    if (lid_c2 < mat2.graph->owned_global_indices.size()) { // Should be same size
                        gid_c2 = mat2.graph->owned_global_indices[lid_c2];
                    } else {
                        gid_c2 = mat2.graph->ghost_global_indices[lid_c2 - mat2.graph->owned_global_indices.size()];
                    }
                    
                    if (gid_c2 == gid_c) {
                        idx2 = k2;
                        break;
                    }
                }
                
                if (idx2 == -1) {
                    std::cout << "Rank " << rank << ": Block missing in mat2" << std::endl;
                    max_err = 1.0;
                } else {
                    // Compare values
                    T* data1 = mat.arena.get_ptr(mat.blk_handles[k]);
                    T* data2 = mat2.arena.get_ptr(mat2.blk_handles[idx2]);
                    size_t size = mat.blk_sizes[k];
                    
                    for (size_t j = 0; j < size; ++j) {
                        double diff = std::abs(data1[j] - data2[j]);
                        if (diff > 1e-12) {
                            max_err = std::max(max_err, diff);
                        }
                    }
                }
            }
        }
    }
    
    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-12) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
    
    c_dbcsr_release(&m_dbcsr);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    // Initialize DBCSR
    int io_unit = 0;
    c_dbcsr_init_lib(MPI_COMM_WORLD, &io_unit);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    run_test<double>(rank, size);
    run_test<std::complex<double>>(rank, size);
    
    c_dbcsr_finalize_lib();
    MPI_Finalize();
    return 0;
}
