#ifndef RSATB_BACKEND_BLOCK_CSR_HPP
#define RSATB_BACKEND_BLOCK_CSR_HPP

#include "dist_graph.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "kernels.hpp"
#include <vector>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <type_traits>

namespace rsatb {
namespace backend {

enum class AssemblyMode {
    INSERT,
    ADD
};

enum class MatrixLayout {
    RowMajor,
    ColMajor
};

// Helper for Matrix Market output
template<typename T>
struct MMWriter {
    static void write(std::ostream& os, const T& v) {
        os << v;
    }
    static bool is_complex() { return false; }
};

template<typename T>
struct MMWriter<std::complex<T>> {
    static void write(std::ostream& os, const std::complex<T>& v) {
        os << v.real() << " " << v.imag();
    }
    static bool is_complex() { return true; }
};

template <typename T, typename Kernel = DefaultKernel<T>>
class BlockSpMat {
public:
    DistGraph* graph;
    using KernelType = Kernel;
    
    // CSR structure (local indices)
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    
    // Values (flattened blocks)
    std::vector<T> val;
    
    // Pointers to block starts in val
    std::vector<size_t> blk_ptr;
    
    // Block dimensions (cached for speed)
    // row_dims[i] = rows of local block row i
    // col_dims[j] = cols of local block col j (including ghosts)
    std::vector<int> row_dims;
    std::vector<int> col_dims;

    // Precomputed offsets for vector access
    std::vector<int> y_offsets; // For local rows
    std::vector<int> x_offsets; // For local cols (including ghosts)

    bool owns_graph = false;

public:
    BlockSpMat(DistGraph* g) : graph(g) {
        allocate_from_graph();
    }

    ~BlockSpMat() {
        if (owns_graph && graph) {
            delete graph;
        }
    }

    // Move constructor
    BlockSpMat(BlockSpMat&& other) noexcept : 
        graph(other.graph), 
        row_ptr(std::move(other.row_ptr)),
        col_ind(std::move(other.col_ind)),
        val(std::move(other.val)),
        blk_ptr(std::move(other.blk_ptr)),
        row_dims(std::move(other.row_dims)),
        col_dims(std::move(other.col_dims)),
        y_offsets(std::move(other.y_offsets)),
        x_offsets(std::move(other.x_offsets)),
        remote_blocks(std::move(other.remote_blocks)),
        owns_graph(other.owns_graph) 
    {
        other.graph = nullptr;
        other.owns_graph = false;
    }

    // Move assignment
    BlockSpMat& operator=(BlockSpMat&& other) noexcept {
        if (this != &other) {
            if (owns_graph && graph) delete graph;
            graph = other.graph;
            row_ptr = std::move(other.row_ptr);
            col_ind = std::move(other.col_ind);
            val = std::move(other.val);
            blk_ptr = std::move(other.blk_ptr);
            row_dims = std::move(other.row_dims);
            col_dims = std::move(other.col_dims);
            y_offsets = std::move(other.y_offsets);
            x_offsets = std::move(other.x_offsets);
            remote_blocks = std::move(other.remote_blocks);
            owns_graph = other.owns_graph;
            other.graph = nullptr;
            other.owns_graph = false;
        }
        return *this;
    }

    // Disable copy (use duplicate() instead)
    BlockSpMat(const BlockSpMat&) = delete;
    BlockSpMat& operator=(const BlockSpMat&) = delete;

    // Create a deep copy of the matrix
    BlockSpMat<T, Kernel> duplicate(bool independent_graph = true) const {
        DistGraph* new_graph = graph;
        bool new_owns_graph = false;
        if (independent_graph && graph) {
            new_graph = graph->duplicate();
            new_owns_graph = true;
        }
        BlockSpMat<T, Kernel> new_mat(new_graph);
        new_mat.owns_graph = new_owns_graph;
        new_mat.copy_from(*this);
        return new_mat;
    }

    void allocate_from_graph() {
        // 1. Get CSR structure
        graph->get_matrix_structure(row_ptr, col_ind);
        
        // 2. Cache dimensions
        int n_owned = graph->owned_global_indices.size();
        row_dims.resize(n_owned);
        for(int i=0; i<n_owned; ++i) row_dims[i] = graph->block_sizes[i];
        
        col_dims = graph->block_sizes; // All local blocks (owned + ghost)
        
        // 2.1 Compute offsets
        y_offsets.resize(n_owned + 1, 0);
        for(int i=0; i<n_owned; ++i) y_offsets[i+1] = y_offsets[i] + row_dims[i];
        
        x_offsets.resize(col_dims.size() + 1, 0);
        for(size_t i=0; i<col_dims.size(); ++i) x_offsets[i+1] = x_offsets[i] + col_dims[i];
        
        // 3. Calculate value size and block pointers
        int nnz = col_ind.size();
        blk_ptr.resize(nnz + 1);
        blk_ptr[0] = 0;
        
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = row_dims[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = col_dims[col];
                blk_ptr[k+1] = blk_ptr[k] + r_dim * c_dim;
            }
        }
        
        val.resize(blk_ptr[nnz], T(0));
    }

    // Buffer for remote assembly
    struct PendingBlock {
        int g_row;
        int g_col;
        int rows;
        int cols;
        AssemblyMode mode;
        std::vector<T> data;
    };
    // Map: Owner -> (GlobalRow, GlobalCol) -> PendingBlock
    std::map<int, std::map<std::pair<int, int>, PendingBlock>> remote_blocks;

    // Add a block (local or remote)
    // Input data layout is specified by `layout`
    void add_block(int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = MatrixLayout::ColMajor) {
        int owner = graph->find_owner(global_row);
        
        if (owner == graph->rank) {
            // Local: Try to update immediately
            if (graph->global_to_local.find(global_row) != graph->global_to_local.end()) {
                int local_row = graph->global_to_local.at(global_row);
                
                int local_col = -1;
                if (graph->global_to_local.count(global_col)) {
                    local_col = graph->global_to_local.at(global_col);
                }
                
                if (local_col != -1) {
                    // update_local_block handles layout
                    if (update_local_block(local_row, local_col, data, rows, cols, mode, layout)) {
                        return; // Success
                    }
                }
            }
            throw std::runtime_error("Block (row=" + std::to_string(global_row) + ", col=" + std::to_string(global_col) + ") not found in local graph");
        } 
        
        // Remote
        auto& blocks_map = remote_blocks[owner];
        std::pair<int, int> key = {global_row, global_col};
        auto it = blocks_map.find(key);
        
        if (it != blocks_map.end()) {
            // Block exists in buffer
            PendingBlock& pb = it->second;
            
            // Check dims
            if (pb.rows != rows || pb.cols != cols) {
                throw std::runtime_error("Dimension mismatch in add_block accumulation");
            }
            
            // We store pending blocks in ColMajor (canonical format for transport)
            
            if (mode == AssemblyMode::INSERT) {
                // Overwrite
                pb.mode = AssemblyMode::INSERT;
                if (layout == MatrixLayout::ColMajor) {
                    std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
                } else {
                    // Transpose RowMajor -> ColMajor
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[j * rows + i] = data[i * cols + j];
                        }
                    }
                }
            } else {
                // ADD
                // Accumulate
                if (layout == MatrixLayout::ColMajor) {
                    for (size_t i = 0; i < pb.data.size(); ++i) {
                        pb.data[i] += data[i];
                    }
                } else {
                    // Transpose add
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[j * rows + i] += data[i * cols + j];
                        }
                    }
                }
            }
        } else {
            // New block
            PendingBlock pb;
            pb.g_row = global_row;
            pb.g_col = global_col;
            pb.rows = rows;
            pb.cols = cols;
            pb.mode = mode;
            pb.data.resize(rows * cols);
            
            if (layout == MatrixLayout::ColMajor) {
                std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
            } else {
                // Transpose RowMajor -> ColMajor
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        pb.data[j * rows + i] = data[i * cols + j];
                    }
                }
            }
            blocks_map[key] = std::move(pb);
        }
    }

    // Finalize assembly by exchanging remote blocks
    void assemble() {
        int size = graph->size;
        
        // 1. Pack buffers
        // Protocol: g_row(int), g_col(int), rows(int), cols(int), mode(int), data(T...)
        // Data is always ColMajor in the protocol
        std::vector<std::vector<char>> send_bufs(size);
        
        for (auto& kv : remote_blocks) {
            int target = kv.first;
            auto& blocks_map = kv.second;
            for (auto& inner_kv : blocks_map) {
                auto& blk = inner_kv.second;
                size_t current_size = send_bufs[target].size();
                size_t data_bytes = blk.data.size() * sizeof(T);
                size_t packet_size = 4 * sizeof(int) + sizeof(int) + data_bytes;
                
                send_bufs[target].resize(current_size + packet_size);
                char* ptr = send_bufs[target].data() + current_size;
                
                std::memcpy(ptr, &blk.g_row, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, &blk.g_col, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, &blk.rows, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, &blk.cols, sizeof(int)); ptr += sizeof(int);
                int mode_int = static_cast<int>(blk.mode);
                std::memcpy(ptr, &mode_int, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, blk.data.data(), data_bytes);
            }
        }
        
        // 2. Exchange
        std::vector<int> send_counts(size);
        for(int i=0; i<size; ++i) send_counts[i] = send_bufs[i].size();
        
        std::vector<int> recv_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, graph->comm);
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<char> send_blob(sdispls[size]);
        for(int i=0; i<size; ++i) {
            std::memcpy(send_blob.data() + sdispls[i], send_bufs[i].data(), send_counts[i]);
        }
        
        std::vector<char> recv_blob(rdispls[size]);
        MPI_Alltoallv(send_blob.data(), send_counts.data(), sdispls.data(), MPI_BYTE,
                      recv_blob.data(), recv_counts.data(), rdispls.data(), MPI_BYTE, graph->comm);
                      
        // 3. Process received
        for(int i=0; i<size; ++i) {
            char* ptr = recv_blob.data() + rdispls[i];
            char* end = recv_blob.data() + rdispls[i+1];
            
            while(ptr < end) {
                int g_row, g_col, rows, cols, mode_int;
                std::memcpy(&g_row, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&g_col, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&rows, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&cols, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&mode_int, ptr, sizeof(int)); ptr += sizeof(int);
                AssemblyMode mode = static_cast<AssemblyMode>(mode_int);
                
                if (graph->global_to_local.find(g_row) == graph->global_to_local.end()) {
                     throw std::runtime_error("Received block for non-owned row");
                }
                int l_row = graph->global_to_local[g_row];
                
                if (graph->global_to_local.find(g_col) == graph->global_to_local.end()) {
                     throw std::runtime_error("Received block for unknown col");
                }
                int l_col = graph->global_to_local[g_col];
                
                size_t data_bytes = rows * cols * sizeof(T);
                
                // Update local block
                // Received data is ColMajor (from protocol)
                if (!update_local_block(l_row, l_col, (const T*)ptr, rows, cols, mode, MatrixLayout::ColMajor)) {
                    throw std::runtime_error("Received block not in graph");
                }
                
                ptr += data_bytes;
            }
        }
        
        remote_blocks.clear();
    }

    // Helper to update local block
    // Input data layout is specified by `layout`
    // Internal storage is ColMajor
    bool update_local_block(int local_row, int local_col, const T* data, int rows, int cols, AssemblyMode mode, MatrixLayout layout = MatrixLayout::ColMajor) {
        int start = row_ptr[local_row];
        int end = row_ptr[local_row+1];
        
        // optimizable
        for (int k = start; k < end; ++k) {
            if (col_ind[k] == local_col) {
                size_t offset = blk_ptr[k];
                // size_t size = blk_ptr[k+1] - offset; // This is total elements
                T* target = val.data() + offset;
                
                // Check dims
                
                if (row_dims[local_row] != rows || col_dims[local_col] != cols) {
                    std::stringstream ss;
                    ss << "\n Dimension mismatch in update_local_block (DEBUG): "
                       << "Row: " << local_row << " (Expected: " << row_dims[local_row] << ", Got: " << rows << ") \n"
                       << "Col: " << local_col << " (Expected: " << col_dims[local_col] << ", Got: " << cols << ") \n";
                    throw std::runtime_error(ss.str());
                }

                int r_dim = row_dims[local_row];
                int c_dim = col_dims[local_col];

                if (mode == AssemblyMode::INSERT) {
                    if (layout == MatrixLayout::ColMajor) {
                        // Direct copy: Input (ColMajor) -> Internal (ColMajor)
                        std::memcpy(target, data, r_dim * c_dim * sizeof(T));
                    } else {
                        // Transpose copy: Input (RowMajor) -> Internal (ColMajor)
                        // Internal[j*r_dim + i] = Input[i*c_dim + j]
                        for (int i = 0; i < r_dim; ++i) {
                            for (int j = 0; j < c_dim; ++j) {
                                target[j * r_dim + i] = data[i * c_dim + j];
                            }
                        }
                    }
                } else {
                    // ADD
                    if (layout == MatrixLayout::ColMajor) {
                        // Direct add
                         for (int i = 0; i < r_dim * c_dim; ++i) {
                            target[i] += data[i];
                        }
                    } else {
                        // Transpose copy and add: Internal[j*r_dim + i] += Input[i*c_dim + j]
                        for (int i = 0; i < r_dim; ++i) {
                            for (int j = 0; j < c_dim; ++j) {
                                target[j * r_dim + i] += data[i * c_dim + j];
                            }
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }

    // Legacy setter (defaults to INSERT)
    void set_local_block(int local_row, int local_col, const T* data) {
        if (!update_local_block(local_row, local_col, data, AssemblyMode::INSERT)) {
            throw std::runtime_error("Block not found in graph structure");
        }
    }

    // Matrix-Vector Multiplication
    void mult(DistVector<T>& x, DistVector<T>& y) {
        mult_optimized(x, y);
    }
    
    // Refined mult with offsets
    void mult_optimized(DistVector<T>& x, DistVector<T>& y) {
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        x.sync_ghosts();
        
        int n_rows = row_ptr.size() - 1;
        
        // Use precomputed offsets
        // y_offsets and x_offsets are members

        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = row_dims[i];
            T* y_val = y.local_data() + y_offsets[i];
            
            // Initialize y_val to 0? Or assume y is zeroed?
            // Usually mult implies y = A*x. So overwrite.
            std::memset(y_val, 0, r_dim * sizeof(T));
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = col_dims[col];
                const T* block_val = val.data() + blk_ptr[k];
                const T* x_val = x.data.data() + x_offsets[col]; // x.data includes ghosts
                
                // y_block += A_block * x_block
                Kernel::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_val, 1);
            }
        }
    }

    // Matrix-Matrix Multiplication (Dense RHS)
    void mult_dense(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        X.sync_ghosts();
        
        int n_rows = row_ptr.size() - 1;
        int num_vecs = X.num_vectors;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = row_dims[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            T* y_ptr = &Y(y_offsets[i], 0);
            int ldc = Y.local_rows + Y.ghost_rows;
            
            bool first = true;
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = col_dims[col];
                const T* block_val = val.data() + blk_ptr[k];
                const T* x_ptr = &X(x_offsets[col], 0);
                int ldb = X.local_rows + X.ghost_rows;
                
                T beta = first ? T(0) : T(1);
                Kernel::gemm(r_dim, num_vecs, c_dim, T(1), block_val, r_dim, x_ptr, ldb, beta, y_ptr, ldc);
                first = false;
            }
            
            if (first) {
                for (int v = 0; v < num_vecs; ++v) {
                    for (int r = 0; r < r_dim; ++r) {
                        y_ptr[v * ldc + r] = T(0);
                    }
                }
            }
        }
    }

    // Adjoint Matrix-Vector Multiplication: y = A^dagger * x
    void mult_adjoint(DistVector<T>& x, DistVector<T>& y) {
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        
        // Initialize y to 0 (including ghosts for accumulation)
        std::fill(y.data.begin(), y.data.end(), T(0));
        
        int n_rows = row_ptr.size() - 1;
        
        // We can't easily parallelize over rows because multiple rows contribute to the same y_j.
        // We need a thread-local y or atomic adds.
        // For simplicity and correctness, let's use a critical section or atomic if T supports it.
        // Or parallelize over rows but use a temporary per-thread y.
        
        #pragma omp parallel
        {
            std::vector<T> y_local(y.data.size(), T(0));
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int r_dim = row_dims[i];
                const T* x_val = x.local_data() + y_offsets[i];
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col = col_ind[k];
                    int c_dim = col_dims[col];
                    const T* block_val = val.data() + blk_ptr[k];
                    T* y_target = y_local.data() + x_offsets[col];
                    
                    // y_target += A_block^dagger * x_block
                    Kernel::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_target, 1, CblasConjTrans);
                }
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < y.data.size(); ++i) y.data[i] += y_local[i];
            }
        }
        
        y.reduce_ghosts();
    }

    // Adjoint Matrix-Matrix Multiplication: Y = A^dagger * X
    void mult_dense_adjoint(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        
        std::fill(Y.data.begin(), Y.data.end(), T(0));
        
        int n_rows = row_ptr.size() - 1;
        int num_vecs = X.num_vectors;
        
        #pragma omp parallel
        {
            std::vector<T> Y_local(Y.data.size(), T(0));
            int ldc_local = Y.local_rows + Y.ghost_rows;
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int r_dim = row_dims[i];
                const T* x_ptr = &X(y_offsets[i], 0);
                int ldb = X.local_rows + X.ghost_rows;
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col = col_ind[k];
                    int c_dim = col_dims[col];
                    const T* block_val = val.data() + blk_ptr[k];
                    T* y_ptr = &Y_local[col]; // This is wrong for column-major
                    // Correct pointer:
                    T* y_target = &Y_local[x_offsets[col]]; 
                    
                    // Y_target += A_block^dagger * X_block
                    // A_block is r_dim x c_dim. A_block^dagger is c_dim x r_dim.
                    // X_block is r_dim x num_vecs.
                    // Y_target is c_dim x num_vecs.
                    Kernel::gemm(c_dim, num_vecs, r_dim, T(1), block_val, r_dim, x_ptr, ldb, T(1), y_target, ldc_local, CblasConjTrans, CblasNoTrans);
                }
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < Y.data.size(); ++i) Y.data[i] += Y_local[i];
            }
        }
        
        Y.reduce_ghosts();
    }

    // Utilities
    void scale(T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < val.size(); ++i) {
            val[i] *= alpha;
        }
    }

    void copy_from(const BlockSpMat<T, Kernel>& other) {
        // If graphs are different, we should at least check compatibility
        if (graph != other.graph) {
            // Check if owned indices and block sizes match
            if (graph->owned_global_indices != other.graph->owned_global_indices ||
                graph->block_sizes != other.graph->block_sizes) {
                 throw std::runtime_error("Incompatible graph structure in copy_from");
            }
        }

        if (val.size() != other.val.size()) {
            throw std::runtime_error("Matrix value size mismatch in copy_from");
        }
        std::memcpy(val.data(), other.val.data(), val.size() * sizeof(T));
    }

    void axpy(T alpha, const BlockSpMat<T, Kernel>& other) {
        if (val.size() != other.val.size()) {
            throw std::runtime_error("Matrix structure mismatch in axpy");
        }
        #pragma omp parallel for
        for (size_t i = 0; i < val.size(); ++i) {
            val[i] += alpha * other.val[i];
        }
    }

    // Add alpha to diagonal elements
    void shift(T alpha) {
        int n_rows = row_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            // Find diagonal block (col == local_col corresponding to local_row i)
            // Local row i corresponds to global row G = graph->get_global_index(i)
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
                // Check if this local_col maps to global_row
                
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    size_t offset = blk_ptr[k];
                    int r_dim = row_dims[i];
                    int c_dim = col_dims[local_col];
                    
                    // Add alpha to diagonal of the block
                    // Block is ColMajor or RowMajor? Internal is ColMajor.
                    // Diagonal elements are at [j*r_dim + j] for j=0..min(r,c)
                    
                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        val[offset + j * r_dim + j] += alpha;
                    }
                    break; 
                }
            }
        }
    }

    // Compute C = [H, R] where R is diagonal (stored as DistVector)
    // C_ij = H_ij * (R_j - R_i)
    // Result C has same structure as H (this).
    void commutator_diagonal(const DistVector<T>& diag, BlockSpMat<T, Kernel>& result) {
        // Ensure result has same structure
        if (result.val.size() != val.size()) {
            result.allocate_from_graph(); // Or throw
        }
        
        const std::vector<T>& R = diag.data; // Includes ghosts
        
        int n_rows = row_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            T R_i = R[i]; // Local row i corresponds to R[i] (owned part)
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                T R_j = R[col]; // col is local index (owned or ghost), maps directly to R vector
                
                T diff = R_j - R_i;
                
                size_t offset = blk_ptr[k];
                int r_dim = row_dims[i];
                int c_dim = col_dims[col];
                int block_size = r_dim * c_dim;
                
                const T* H_ptr = val.data() + offset;
                T* C_ptr = result.val.data() + offset;
                
                for (int b = 0; b < block_size; ++b) {
                    C_ptr[b] = H_ptr[b] * diff;
                }
            }
        }
    }

    // Export to Matrix Market format (Serial only)
    void save_matrix_market(const std::string& filename) {
        if (graph->size > 1) {
            throw std::runtime_error("save_matrix_market only supported for serial execution (MPI size = 1)");
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        file << "%%MatrixMarket matrix coordinate " << (MMWriter<T>::is_complex() ? "complex" : "real") << " general\n";

        // Calculate total NNZ and dimensions
        size_t element_nnz = 0;
        int total_rows = 0;
        int total_cols = 0;
        
        int n_owned = row_ptr.size() - 1;
        
        // Calculate total rows
        for (int i = 0; i < n_owned; ++i) {
            total_rows += row_dims[i];
        }
        
        // Calculate total cols (assuming serial, so owned == all local cols)
        for (size_t i = 0; i < col_dims.size(); ++i) {
             total_cols += col_dims[i];
        }
        
        // Count NNZ
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = row_dims[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = col_dims[col];
                element_nnz += r_dim * c_dim;
            }
        }
        
        file << total_rows << " " << total_cols << " " << element_nnz << "\n";
        
        // Write data
        file << std::scientific << std::setprecision(16);
        
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = row_dims[i];
            int row_start_idx = y_offsets[i] + 1; // 1-based
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = col_dims[col];
                int col_start_idx = x_offsets[col] + 1; // 1-based
                
                size_t offset = blk_ptr[k];
                const T* block_data = val.data() + offset;
                
                // Block is stored in ColMajor
                for (int c = 0; c < c_dim; ++c) {
                    for (int r = 0; r < r_dim; ++r) {
                        T value = block_data[c * r_dim + r];
                        
                        // Write (row, col, val)
                        file << (row_start_idx + r) << " " << (col_start_idx + c) << " ";
                        MMWriter<T>::write(file, value);
                        file << "\n";
                    }
                }
            }
        }
    }
};

} // namespace backend
} // namespace rsatb

#endif
