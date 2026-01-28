#ifndef RSATB_CALC_GRAPHMF_HPP
#define RSATB_CALC_GRAPHMF_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <set>
#include <functional>
#include "submf.hpp"

using namespace vbcsr;

namespace rsatb {
namespace calc {


template <typename T, typename Kernel = DefaultKernel<T>>
void graph_matrix_function(BlockSpMat<T, Kernel>& A, BlockSpMat<T, Kernel>* Result, std::function<T(double)> func, std::string method = "lanczos", bool verbose = false) {
    
    DistGraph* graph = A.graph;
    int rank = graph->rank;
    int size = graph->size;
    MPI_Comm comm = graph->comm;

    // 1. Initialize Result
    if (rank == 0 && verbose) std::cout << "graph_matrix_function: Initializing using subgraph method (" << method << ")..." << std::endl;

    // check Result allocation
    if (Result == nullptr || Result == NULL) {
        *Result = A.duplicate();
    }

    if (Result->graph != A.graph) {
        // If pointers differ, check basic properties
        if (Result->graph->comm != A.graph->comm || 
            Result->graph->owned_global_indices.size() != A.graph->owned_global_indices.size()) {
            throw std::runtime_error("graph_matrix_function: Result matrix has incompatible graph structure");
        }
    }
    
    // Clear Result
    Result->fill(T(0));
    
    int n_owned = graph->owned_global_indices.size();
    int n_owned_max = 0;
    MPI_Allreduce(&n_owned, &n_owned_max, 1, MPI_INT, MPI_MAX, comm);
    
    // batch size
    int batch_size = std::max(1, omp_get_max_threads() / size);
    
    int nbatch = n_owned_max / batch_size;
    if (n_owned_max % batch_size != 0) nbatch++;

    std::vector<std::vector<int>> batch_indices(batch_size);

    for (int b = 0; b < nbatch; ++b) {
        // Avoid thread oversubscription in BLAS/LAPACK
        #ifdef RSATB_USE_MKL
        int one = 1;
        mkl_set_num_threads_(&one);
        #elif defined(RSATB_USE_OPENBLAS)
        openblas_set_num_threads(1);
        #endif

        batch_indices.clear();
        batch_indices.resize(batch_size);
        
        // Store neighbors for parallel phase
        std::vector<std::vector<int>> batch_neighbors(batch_size);

        for (int i=0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            
            if (idx < n_owned) {
                int global_row = graph->owned_global_indices[idx];
                // Identify Neighborhood C_i
                std::vector<int> neighbors;
                int start = A.row_ptr[idx]; // Use local index idx
                int end = A.row_ptr[idx+1];
                for (int k = start; k < end; ++k) {
                    int col_lid = A.col_ind[k];
                    int col_gid = graph->get_global_index(col_lid);
                    neighbors.push_back(col_gid);
                }
                // Ensure global_row is in neighbors (diagonal)
                bool has_diag = false;
                for(int gid : neighbors) if(gid == global_row) has_diag = true;
                if(!has_diag) neighbors.push_back(global_row);
                
                // Sort for consistency
                std::sort(neighbors.begin(), neighbors.end());

                batch_indices[i] = neighbors;
            } else {
                // Padding: Empty request
                batch_indices[i] = {};
            }
        }
        
        auto batch_blocks = A.fetch_blocks(batch_indices);

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            if (idx >= n_owned) continue;
            
            int global_row = graph->owned_global_indices[idx];
            const auto& neighbors = batch_indices[i];

            // Find block index of global_row
            auto it = std::find(neighbors.begin(), neighbors.end(), global_row);
            int block_idx = std::distance(neighbors.begin(), it);
            BlockSpMat<T, Kernel> sub_mat = A.construct_submatrix(neighbors, batch_blocks);
            
            int r_dim = sub_mat.graph->block_sizes[block_idx]; // getting from submat, where index is the neighbours order, safe

            // Convert to Dense
            std::vector<T> M = sub_mat.to_dense();
        
            // M is row-major, size (total_dim) x (total_dim)
            int total_dim = 0;
            for(size_t k=0; k<neighbors.size(); ++k) total_dim += sub_mat.graph->block_sizes[k];

            if (rank==0 || verbose) {
                std::cout << "working on atom: "<< idx << "/" << n_owned_max << " total dim: " << total_dim << std::endl;
            }
            
            // Calculate offset in dense matrix M
            int row_offset = 0;
            for(int k=0; k<block_idx; ++k) row_offset += sub_mat.graph->block_sizes[k];
            
            DistMultiVector<T> X(sub_mat.graph, r_dim);

            if (method == "lanczos") {
                // Compute f(M) using Lanczos
                // We want the columns of f(M) corresponding to global_row
                // These start at row_offset and have size r_dim
                
                // Use lanczos_matrix_function_dense with global_col_start = row_offset
                lanczos_matrix_function_dense(M, X, func, r_dim, sub_mat.graph, 1e-8, 100, verbose, false, row_offset);
            } else if (method == "dense") {
                // Full diagonalization
                // Optimized diagonalization (only compute needed columns)
                dense_matrix_function(total_dim, M, func, r_dim, row_offset);
                
                // M is now f(M)[:, row_offset:row_offset+r_dim]
                // Size: total_dim x r_dim
                // Layout: ColMajor (from dense_gemm)
                
                X.bind_to_graph(sub_mat.graph);
                // Copy M to X.data
                // X.data expects ColMajor storage for local_rows=total_dim, n_cols=r_dim
                // M is exactly that.
                if (X.data.size() != M.size()) {
                     // Should match if X was allocated with r_dim cols and total_dim rows
                     // X(sub_mat.graph, r_dim) -> local_rows = total_dim
                }
                std::copy(M.begin(), M.end(), X.data.begin());
            } else {
                if (rank == 0) std::cerr << "Unknown method: " << method << std::endl;
            }
            
            // Iterate over columns (neighbors)
            int col_offset = 0;
            for(size_t k=0; k<neighbors.size(); ++k) {
                int col_gid = neighbors[k];
                int c_dim = sub_mat.graph->block_sizes[k];
                
                // Extract block (r_dim x c_dim) from X
                // X contains f(M)(:, row_offset:row_offset+r_dim)
                // We want block (global_row, col_gid) of f(A)
                // In sub_mat, this is block (block_idx, k).
                // f(M)(row_offset:row_offset+r_dim, col_offset:col_offset+c_dim)
                // By symmetry = f(M)(col_offset:col_offset+c_dim, row_offset:row_offset+r_dim)^T
                // X has rows col_offset:col_offset+c_dim
                
                std::vector<T> block_data(r_dim * c_dim);
                for(int c=0; c<c_dim; ++c) {
                    for(int r=0; r<r_dim; ++r) {
                        // X(row, col) = X.data[row + col * total_dim]
                        // We want X(col_offset + c, r)
                        block_data[c * r_dim + r] = X.data[(col_offset + c) + r * total_dim];
                    }
                }
                
                Result->add_block(global_row, col_gid, block_data.data(), r_dim, c_dim, AssemblyMode::ADD, MatrixLayout::ColMajor);
                
                col_offset += c_dim;
            }
        }
    }
    
    // if (rank == 0 && verbose) std::cout << "graph_matrix_function: Loop Finished. Assembling..." << std::endl;
    // if (rank == 0) std::cout << "graph_matrix_function: remote_blocks size: " << Result->remote_blocks.size() << std::endl;

    Result->assemble();
    
}

}
}

#endif // RSATB_CALC_GRAPHMF_HPP