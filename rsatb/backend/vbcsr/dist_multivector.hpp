#ifndef RSATB_BACKEND_DIST_MULTIVECTOR_HPP
#define RSATB_BACKEND_DIST_MULTIVECTOR_HPP

#include "dist_graph.hpp"
#include <vector>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <random>

#include <complex>
#include "scalar_traits.hpp"

namespace rsatb {
namespace backend {

template <typename T>
class DistMultiVector {
public:
    using value_type = T;
    DistGraph* graph;
    int num_vectors;
    std::vector<T> data; // Column-major: (local_size + ghost_size) x num_vectors
    int local_rows;
    int ghost_rows;

public:
    DistMultiVector(DistGraph* g, int n_vecs) : graph(g), num_vectors(n_vecs) {
        graph->get_vector_structure(local_rows, ghost_rows);
        data.resize((local_rows + ghost_rows) * num_vectors);
    }

    // Accessors
    // (row, col) -> data[col * total_rows + row]
    T& operator()(int row, int col) { 
        return data[col * (local_rows + ghost_rows) + row]; 
    }
    
    const T& operator()(int row, int col) const { 
        return data[col * (local_rows + ghost_rows) + row]; 
    }
    
    T* col_data(int col) {
        return data.data() + col * (local_rows + ghost_rows);
    }
    
    void conjugate() {
        for (int c = 0; c < num_vectors; ++c) {
            T* col = col_data(c);
            for (int i = 0; i < local_rows; ++i) {
                col[i] = ScalarTraits<T>::conjugate(col[i]);
            }
        }
    }

    // Bind to a new graph (must have same owned structure)
    void bind_to_graph(DistGraph* new_graph) {
        if (graph == new_graph) return;
        
        int new_local_rows, new_ghost_rows;
        new_graph->get_vector_structure(new_local_rows, new_ghost_rows);
        
        if (new_local_rows != local_rows) {
            throw std::runtime_error("Cannot bind to graph with different owned structure");
        }
        
        // Data is column-major: [col 0][col 1]...
        // Each col is [owned | ghost]
        // We need to resize EACH column.
        // This requires moving data if we just do resize on the flat vector.
        // Flat vector: [owned0 | ghost0 | owned1 | ghost1 ...]
        // If we change ghost size, owned1 moves!
        
        // We must repack.
        std::vector<T> new_data((new_local_rows + new_ghost_rows) * num_vectors);
        
        for (int c = 0; c < num_vectors; ++c) {
            T* src = col_data(c); // Old data
            T* dst = new_data.data() + c * (new_local_rows + new_ghost_rows);
            
            // Copy owned part
            std::memcpy(dst, src, local_rows * sizeof(T));
            // Ghosts are invalid/uninitialized in new buffer
        }
        
        data = std::move(new_data);
        graph = new_graph;
        ghost_rows = new_ghost_rows;
    }

    // Operations
    void set_constant(T val) {
        std::fill(data.begin(), data.end(), val);
    }

    void scale(T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) data[i] *= alpha;
    }

    void axpy(T alpha, const DistMultiVector<T>& x) {
        assert(x.num_vectors == num_vectors);
        assert(x.data.size() == data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) data[i] += alpha * x.data[i];
    }
    
    void axpby(T alpha, const DistMultiVector<T>& x, T beta) {
        assert(x.num_vectors == num_vectors);
        assert(x.data.size() == data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = alpha * x.data[i] + beta * data[i];
        }
    }

    void pointwise_mult(const DistMultiVector<T>& other) {
        assert(other.num_vectors == num_vectors);
        assert(other.data.size() == data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= other.data[i];
        }
    }

    void pointwise_mult(const DistVector<T>& other) {
        int total_rows = local_rows + ghost_rows;
        assert(other.full_size() == total_rows);
        
        #pragma omp parallel for
        for (int c = 0; c < num_vectors; ++c) {
            T* col = col_data(c);
            const T* vec_data = other.local_data();
            for (int i = 0; i < total_rows; ++i) {
                col[i] *= vec_data[i];
            }
        }
    }

    // Helper to get a column as a DistVector (view or copy)
    // Since DistVector owns its data, we return a copy for now.
    // Ideally, we'd have a DistVectorView, but for simplicity:
    DistVector<T> get_col(int col) {
        DistVector<T> vec(graph);
        T* src = col_data(col);
        // Copy owned part
        std::memcpy(vec.local_data(), src, local_rows * sizeof(T));
        // Ghosts are not synced
        return vec;
    }
    
    // Set a column from a DistVector
    void set_col(int col, const DistVector<T>& vec) {
        T* dst = col_data(col);
        const T* src = vec.local_data();
        std::memcpy(dst, src, local_rows * sizeof(T));
    }

    void copy_from(const DistMultiVector<T>& other) {
        if (local_rows != other.local_rows || num_vectors != other.num_vectors) {
            throw std::runtime_error("DistMultiVector::copy_from: size mismatch");
        }
        std::copy(other.data.begin(), other.data.end(), data.begin());
    }

    void swap(DistMultiVector<T>& other) {
        std::swap(data, other.data);
        std::swap(local_rows, other.local_rows);
        std::swap(num_vectors, other.num_vectors);
        std::swap(graph, other.graph);
    }
    
    // Batched dot product (global)
    // Returns a vector of size num_vectors
    std::vector<T> bdot(const DistMultiVector<T>& x) const {
        if (num_vectors != x.num_vectors) throw std::runtime_error("Dimension mismatch in bdot");
        
        std::vector<T> local_dots(num_vectors, 0);
        
        #pragma omp parallel for
        for (int c = 0; c < num_vectors; ++c) {
            const T* col_this = const_cast<DistMultiVector<T>*>(this)->col_data(c);
            const T* col_x = const_cast<DistMultiVector<T>&>(x).col_data(c);
            
            T sum = 0;
            for (int i = 0; i < local_rows; ++i) {
                sum += ScalarTraits<T>::conjugate(col_this[i]) * col_x[i];
            }
            local_dots[c] = sum;
        }
        
        std::vector<T> global_dots(num_vectors);
        MPI_Datatype type = get_mpi_type();
        MPI_Allreduce(local_dots.data(), global_dots.data(), num_vectors, type, MPI_SUM, graph->comm);
        
        return global_dots;
    }
    
    // Column-wise dot product (alias to bdot, matching implementation plan)
    void dot(const DistMultiVector<T>& other, std::vector<T>& results) const {
        results = bdot(other);
    }

    void set_random_normal(bool normalize = true) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> d(0.0, 1.0);
        
        for (auto& val : data) {
            if constexpr (std::is_same<T, std::complex<double>>::value) {
                val = std::complex<double>(d(gen), d(gen));
            } else {
                val = (T)d(gen);
            }
        }
        
        if (normalize) {
            std::vector<T> dots = this->bdot(*this);
            for (int c = 0; c < num_vectors; ++c) {
                double norm = std::sqrt(std::abs(dots[c]));
                T scale_factor = 1.0 / norm;
                T* col = col_data(c);
                int total_rows = local_rows + ghost_rows;
                for(int i=0; i<total_rows; ++i) col[i] *= scale_factor;
            }
        }
    }

    // Persistent buffers
    std::vector<T> send_buf;
    std::vector<T> recv_buf;

    // Sync ghosts for all vectors
    void sync_ghosts() {
        // Similar to DistVector but for multiple columns.
        // We can pack all columns for a block together to minimize latency.
        
        int total_rows = local_rows + ghost_rows;
        
        // Use cached block_offsets
        const auto& block_offsets = graph->block_offsets;
        const auto& send_counts_scalar = graph->send_counts_scalar;
        const auto& recv_counts_scalar = graph->recv_counts_scalar;
        const auto& send_displs_scalar = graph->send_displs_scalar;
        const auto& recv_displs_scalar = graph->recv_displs_scalar;
        
        // Resize persistent buffers
        int total_send_elements = send_displs_scalar[graph->size] * num_vectors;
        if (send_buf.size() < total_send_elements) send_buf.resize(total_send_elements);
        
        int total_recv_elements = recv_displs_scalar[graph->size] * num_vectors;
        if (recv_buf.size() < total_recv_elements) recv_buf.resize(total_recv_elements);
        
        // Pack Data
        // Data is column-major: data[col * total_rows + row]
        // But we want to pack blocks.
        // For each block, we pack all vectors.
        // Or we pack vector by vector?
        // MPI_Alltoallv expects contiguous buffer.
        // If we pack vector by vector, we need to adjust counts/displs.
        // Easier to pack block by block: Block 0 (all vecs), Block 1 (all vecs)...
        // But send_counts_scalar is in ELEMENTS (rows).
        // So we need to multiply counts/displs by num_vectors.
        
        std::vector<int> s_counts = send_counts_scalar;
        std::vector<int> r_counts = recv_counts_scalar;
        std::vector<int> s_displs = send_displs_scalar;
        std::vector<int> r_displs = recv_displs_scalar;
        
        for(auto& x : s_counts) x *= num_vectors;
        for(auto& x : r_counts) x *= num_vectors;
        for(auto& x : s_displs) x *= num_vectors;
        for(auto& x : r_displs) x *= num_vectors;
        
        int current_idx = 0;
        int buf_ptr = 0;
        
        // Pack: Block-major or Vector-major?
        // If we use the adjusted counts, we are sending a chunk of size (blk_size * num_vectors).
        // So we should pack Block 0 (all vecs), Block 1 (all vecs).
        
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->send_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];
                
                // Copy (blk_size x num_vectors) block
                // Source is strided.
                for (int v = 0; v < num_vectors; ++v) {
                    T* src = &(*this)(blk_offset, v);
                    std::memcpy(send_buf.data() + buf_ptr + v * blk_size, src, blk_size * sizeof(T));
                }
                buf_ptr += blk_size * num_vectors;
            }
        }
        
        // Exchange
        MPI_Datatype type = get_mpi_type();
        MPI_Alltoallv(send_buf.data(), s_counts.data(), s_displs.data(), type,
                      recv_buf.data(), r_counts.data(), r_displs.data(), type, graph->comm);
                      
        // Unpack
        current_idx = 0;
        buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->recv_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->recv_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];
                for (int v = 0; v < num_vectors; ++v) {
                    T* dst = &(*this)(blk_offset, v);
                    std::memcpy(dst, recv_buf.data() + buf_ptr + v * blk_size, blk_size * sizeof(T));
                }
                buf_ptr += blk_size * num_vectors;
            }
        }
    }

    void reduce_ghosts() {
        const auto& block_offsets = graph->block_offsets;
        const auto& send_counts_scalar = graph->send_counts_scalar;
        const auto& recv_counts_scalar = graph->recv_counts_scalar;
        const auto& send_displs_scalar = graph->send_displs_scalar;
        const auto& recv_displs_scalar = graph->recv_displs_scalar;
        
        int total_send_elements = recv_displs_scalar[graph->size] * num_vectors;
        std::vector<T> s_buf(total_send_elements);
        
        int current_idx = 0;
        int buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->recv_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->recv_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];
                for (int v = 0; v < num_vectors; ++v) {
                    T* src = &(*this)(blk_offset, v);
                    std::memcpy(s_buf.data() + buf_ptr + v * blk_size, src, blk_size * sizeof(T));
                }
                buf_ptr += blk_size * num_vectors;
            }
        }
        
        int total_recv_elements = send_displs_scalar[graph->size] * num_vectors;
        std::vector<T> r_buf(total_recv_elements);
        
        std::vector<int> s_counts = recv_counts_scalar;
        std::vector<int> r_counts = send_counts_scalar;
        std::vector<int> s_displs = recv_displs_scalar;
        std::vector<int> r_displs = send_displs_scalar;
        
        for(auto& x : s_counts) x *= num_vectors;
        for(auto& x : r_counts) x *= num_vectors;
        for(auto& x : s_displs) x *= num_vectors;
        for(auto& x : r_displs) x *= num_vectors;
        
        MPI_Datatype type = get_mpi_type();
        MPI_Alltoallv(s_buf.data(), s_counts.data(), s_displs.data(), type,
                      r_buf.data(), r_counts.data(), r_displs.data(), type, graph->comm);
                      
        current_idx = 0;
        buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->send_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];
                for (int v = 0; v < num_vectors; ++v) {
                    T* dst = &(*this)(blk_offset, v);
                    const T* src = r_buf.data() + buf_ptr + v * blk_size;
                    for (int i = 0; i < blk_size; ++i) dst[i] += src[i];
                }
                buf_ptr += blk_size * num_vectors;
            }
        }
    }

private:
    MPI_Datatype get_mpi_type() const;
};

template <> inline MPI_Datatype DistMultiVector<double>::get_mpi_type() const { return MPI_DOUBLE; }
template <> inline MPI_Datatype DistMultiVector<float>::get_mpi_type() const { return MPI_FLOAT; }
template <> inline MPI_Datatype DistMultiVector<int>::get_mpi_type() const { return MPI_INT; }
template <> inline MPI_Datatype DistMultiVector<std::complex<double>>::get_mpi_type() const { return MPI_CXX_DOUBLE_COMPLEX; }

} // namespace backend
} // namespace rsatb

#endif
