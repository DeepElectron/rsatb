#ifndef RSATB_BACKEND_DBCSR_CONVERSIONS_HPP
#define RSATB_BACKEND_DBCSR_CONVERSIONS_HPP

#include "vbcsr/block_csr.hpp"
#include "dbcsr_capi.h"
#include <dbcsr.h>
#include <vector>
#include <complex>
#include <mpi.h>

#include <type_traits>

namespace rsatb {
namespace backend {

// Convert BlockSpMat to DBCSR
// Output m_dbcsr is created and filled
template <typename T, typename Kernel>
void BlockSpMat2DBCSR(BlockSpMat<T, Kernel>& mat, dbcsr_matrix* m_dbcsr, char mat_type = 'N') {
    DistGraph* graph = mat.graph;
    MPI_Comm comm = graph->comm;
    int rank = graph->rank;
    int size = graph->size;

    // 1. Setup Distribution for Row-Based DBCSR (m_brd)
    // We want m_brd to have the same row distribution as BlockSpMat
    // BlockSpMat is row-distributed: each rank owns a set of rows.
    
    // We need global number of blocks and block sizes
    // DistGraph has this info?
    // DistGraph has `owned_global_indices` and `block_sizes` (local view of global sizes?)
    // Wait, DistGraph::block_sizes is for local blocks (owned + ghost).
    // We need global block sizes for all blocks in the matrix.
    // If DistGraph doesn't have it, we might need to gather it or assume it's available.
    // In `DMat2DBCSR`, it calculates `blk_size` array of size `NGlobal`.
    
    // Let's assume we can gather global block sizes.
    // Or maybe we can rely on what we know.
    // For now, let's gather global block sizes.
    
    // Determine NGlobal (total number of blocks)
    // Assuming indices are 0 to NGlobal-1 contiguous.
    // We can find max global index.
    
    int max_gid = -1;
    if (!graph->owned_global_indices.empty()) {
        max_gid = graph->owned_global_indices.back(); // Assumes sorted
    }
    int global_max_gid;
    MPI_Allreduce(&max_gid, &global_max_gid, 1, MPI_INT, MPI_MAX, comm);
    int NGlobal = global_max_gid + 1;
    
    std::vector<int> blk_size(NGlobal, 0);
    std::vector<int> dist_row_brd(NGlobal, 0);
    std::vector<int> dist_col_brd(NGlobal, 0); // Column distribution for BRD (usually 0 or ignored?)
    
    // Fill local info
    for (size_t i = 0; i < graph->owned_global_indices.size(); ++i) {
        int gid = graph->owned_global_indices[i];
        blk_size[gid] = mat.row_dims[i]; // row_dims corresponds to owned indices
        dist_row_brd[gid] = rank;
    }
    
    // Reduce to get global info
    MPI_Allreduce(MPI_IN_PLACE, blk_size.data(), NGlobal, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, dist_row_brd.data(), NGlobal, MPI_INT, MPI_MAX, comm);
    // dist_col_brd is 0 everywhere (replicated columns? or just default)
    
    // 2. Create m_brd
    dbcsr_distribution dist_brd;
    MPI_Comm comm_cart;
    int dims_comm[2] = {size, 1};
    int period[2] = {1, 1};
    MPI_Cart_create(comm, 2, dims_comm, period, 0, &comm_cart);
    
    c_dbcsr_distribution_new(&dist_brd, comm_cart, dist_row_brd.data(), NGlobal, dist_col_brd.data(), NGlobal);
    
    dbcsr_matrix m_brd;
    bool mutable_work = true;
    int data_type;
    if (std::is_same<T, double>::value) {
        data_type = dbcsr_type_real_8;
    } else {
        data_type = dbcsr_type_complex_8;
    }

    c_dbcsr_create_new(&m_brd, "BRDtmp", dist_brd, mat_type, blk_size.data(), NGlobal,
        blk_size.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);

    // 3. Fill m_brd from BlockSpMat
    // Iterate local blocks
    bool summation = false;
    std::complex<double> scale = 1.0;
    
    int n_owned = graph->owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph->owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        int r_dim = mat.row_dims[i];
        
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            // Need global col index
            int gid_c;
            if (lid_c < n_owned) {
                gid_c = graph->owned_global_indices[lid_c];
            } else {
                gid_c = graph->ghost_global_indices[lid_c - n_owned];
            }
            
            int c_dim = mat.col_dims[lid_c];
            const T* block_val = mat.val.data() + mat.blk_ptr[k];
            
            if constexpr (std::is_same<T, double>::value) {
                double scale = 1.0;
                // BlockSpMat is ColMajor. DBCSR expects ColMajor.
                // c_dbcsr_put_block2d_d expects ColMajor (standard DBCSR).
                c_dbcsr_put_block2d_d(m_brd, gid_r, gid_c, block_val, r_dim, c_dim, &summation, &scale);
            } else {
                std::complex<double> scale = 1.0;
                // Transpose ColMajor -> RowMajor for z_api
                std::vector<std::complex<double>> row_major_block(r_dim * c_dim);
                for (int r = 0; r < r_dim; ++r) {
                    for (int c = 0; c < c_dim; ++c) {
                        row_major_block[r * c_dim + c] = block_val[c * r_dim + r];
                    }
                }
                c_dbcsr_put_block2d_z_api(m_brd, gid_r, gid_c, row_major_block.data(), r_dim, c_dim, &summation, &scale);
            }
        }
    }
    
    c_dbcsr_finalize(m_brd);

    // 4. Create Target 2D-Mesh DBCSR (m_dbcsr)
    // Setup 2D distribution
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    MPI_Comm group;
    MPI_Cart_create(comm, 2, dims, period, 0, &group);
    
    std::vector<int> dist_row_dbcsr(NGlobal);
    std::vector<int> dist_col_dbcsr(NGlobal);
    for (int i = 0; i < NGlobal; ++i) {
        dist_row_dbcsr[i] = i % dims[0];
        dist_col_dbcsr[i] = i % dims[1];
    }
    
    dbcsr_distribution dist_dbcsr;
    c_dbcsr_distribution_new(&dist_dbcsr, group, dist_row_dbcsr.data(), NGlobal, dist_col_dbcsr.data(), NGlobal);
    
    c_dbcsr_create_new(m_dbcsr, "MATDBCSR", dist_dbcsr, mat_type, blk_size.data(), NGlobal,
        blk_size.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
        
    // 5. Redistribute
    bool keep_sparsity = false;
    c_dbcsr_complete_redistribute(m_brd, *m_dbcsr, &keep_sparsity, &summation);
    c_dbcsr_finalize(*m_dbcsr);
    
    // 6. Cleanup
    c_dbcsr_release(&m_brd);
    c_dbcsr_distribution_release(&dist_brd);
    // c_dbcsr_distribution_release(&dist_dbcsr);
    MPI_Comm_free(&comm_cart);
    // MPI_Comm_free(&group); // Do not free, used by m_dbcsr
}

// Convert DBCSR to BlockSpMat
// Output mat is allocated and filled
template <typename T, typename Kernel>
void DBCSR2BlockSpMat(dbcsr_matrix m_dbcsr, BlockSpMat<T, Kernel>& mat) {
    DistGraph* graph = mat.graph;
    MPI_Comm comm = graph->comm, group;
    c_dbcsr_get_group(m_dbcsr, &group);
    int rank = graph->rank;
    int size = graph->size;
    
    // 1. Get Global Info from DBCSR
    int NGlobal = c_dbcsr_get_num_blocks(m_dbcsr);
    
    int nblkrows_total = c_dbcsr_nblkrows_total(m_dbcsr);
    int nblkcols_total = c_dbcsr_nblkcols_total(m_dbcsr);
    NGlobal = nblkrows_total; 
    
    std::vector<int> blk_size(NGlobal);
    c_dbcsr_get_row_blk_size(m_dbcsr, blk_size.data(), NGlobal);
    
    // 2. Setup Distribution for Row-Based DBCSR (m_brd)
    std::vector<int> dist_row_brd(NGlobal, 0);
    // Fill based on graph
    for (size_t i = 0; i < graph->owned_global_indices.size(); ++i) {
        int gid = graph->owned_global_indices[i];
        dist_row_brd[gid] = rank;
    }
    MPI_Allreduce(MPI_IN_PLACE, dist_row_brd.data(), NGlobal, MPI_INT, MPI_MAX, comm);
    
    std::vector<int> dist_col_brd(NGlobal, 0);
    
    dbcsr_distribution dist_brd;
    MPI_Comm comm_cart;
    int dims_comm[2] = {size, 1};
    int period[2] = {1, 1};
    MPI_Cart_create(comm, 2, dims_comm, period, 0, &comm_cart);
    
    c_dbcsr_distribution_new(&dist_brd, comm_cart, dist_row_brd.data(), NGlobal, dist_col_brd.data(), NGlobal);
    
    dbcsr_matrix m_brd;
    char mat_type = c_dbcsr_get_matrix_type(m_dbcsr);
    int data_type = c_dbcsr_get_data_type(m_dbcsr);

    c_dbcsr_create_new(&m_brd, "BRDtmp", dist_brd, mat_type, blk_size.data(), NGlobal,
        blk_size.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, nullptr, nullptr);
        
    // 3. Redistribute DBCSR -> BRD
    bool keep_sparsity = false;
    bool summation = false;
    c_dbcsr_complete_redistribute(m_dbcsr, m_brd, &keep_sparsity, &summation);
    
    // 4. Iterate m_brd to count blocks and build graph
    dbcsr_iterator it;
    bool shared = false, dynamic = false, dynamic_byrow = false, continuous_pointer = false, read_only = true;
    c_dbcsr_iterator_start(&it, m_brd, &shared, &dynamic, &dynamic_byrow, &continuous_pointer, &read_only);
    
    int i_blk, j_blk, blk_id, blk_p;
    int nrow, ncol;
    bool tr, found;
    
    // Adjacency list for this rank's owned rows
    // Map: global_row -> set of global_cols
    std::map<int, std::vector<int>> local_adj;
    
    while (c_dbcsr_iterator_blocks_left(it)) {
        c_dbcsr_iterator_next_block_index(it, &i_blk, &j_blk, &blk_id, &blk_p);
        
        if (dist_row_brd[i_blk] == rank) {
            local_adj[i_blk].push_back(j_blk);
        }
    }
    c_dbcsr_iterator_stop(&it);
    
    // Update Graph
    std::vector<int> my_owned_indices;
    for (int i = 0; i < NGlobal; ++i) {
        if (dist_row_brd[i] == rank) {
            my_owned_indices.push_back(i);
        }
    }
    
    std::vector<int> my_block_sizes;
    for (int gid : my_owned_indices) {
        my_block_sizes.push_back(blk_size[gid]);
    }
    
    std::vector<std::vector<int>> adj_vec(my_owned_indices.size());
    for (size_t i = 0; i < my_owned_indices.size(); ++i) {
        int gid = my_owned_indices[i];
        if (local_adj.count(gid)) {
            adj_vec[i] = local_adj[gid];
        }
    }
    
    graph->construct_distributed(my_owned_indices, my_block_sizes, adj_vec);
    
    // Re-allocate Matrix
    mat.allocate_from_graph();
    
    // 5. Iterate m_brd again to fill values
    c_dbcsr_iterator_start(&it, m_brd, &shared, &dynamic, &dynamic_byrow, &continuous_pointer, &read_only);
    
    while (c_dbcsr_iterator_blocks_left(it)) {
        c_dbcsr_iterator_next_block_index(it, &i_blk, &j_blk, &blk_id, &blk_p);
        
        if (dist_row_brd[i_blk] == rank) {
            if (data_type == dbcsr_type_real_8) {
                double* v_real;
                c_dbcsr_get_block_p_d(m_brd, i_blk, j_blk, &v_real, &tr, &found, &nrow, &ncol);
                if (found) {
                     mat.add_block(i_blk, j_blk, reinterpret_cast<T*>(v_real), nrow, ncol, AssemblyMode::INSERT, rsatb::backend::MatrixLayout::ColMajor);
                }
            } else {
                double _Complex* v_complex;
                c_dbcsr_get_block_p_z(m_brd, i_blk, j_blk, &v_complex, &tr, &found, &nrow, &ncol);
                if (found) {
                    if constexpr (std::is_same<T, double>::value) {
                        std::vector<double> real_block(nrow * ncol);
                        for(int i=0; i<nrow*ncol; ++i) {
                            real_block[i] = __real__ v_complex[i];
                        }
                        mat.add_block(i_blk, j_blk, real_block.data(), nrow, ncol, AssemblyMode::INSERT, rsatb::backend::MatrixLayout::ColMajor);
                    } else {
                        mat.add_block(i_blk, j_blk, reinterpret_cast<T*>(v_complex), nrow, ncol, AssemblyMode::INSERT, rsatb::backend::MatrixLayout::ColMajor);
                    }
                }
            }
        }
    }
    
    c_dbcsr_iterator_stop(&it);
    
    // 5. Finalize BlockSpMat
    mat.assemble(); // Exchange any remote blocks if necessary (though here we iterated local BRD blocks which should match local BlockSpMat rows, but columns might be ghost)
    // Wait, if m_brd is row-distributed, we iterate blocks owned by this rank.
    // These blocks have global row indices owned by this rank.
    // So they are local rows.
    // The columns can be anything.
    // `add_block` handles local updates and remote buffering.
    // Since we are iterating local rows, `add_block` should mostly hit local updates.
    // But if `BlockSpMat` doesn't know about the column yet (ghost), it might need to add it.
    // `add_block` currently throws if local block not found in graph structure?
    // "Block not found in local graph"
    // So `BlockSpMat` structure must be pre-allocated or dynamic?
    // The prompt said: "first counting the blocks, preallocate the memory, and then copy the values."
    
    // We skipped the counting/preallocation step!
    // `BlockSpMat` assumes fixed structure from `DistGraph`.
    // If `DBCSR` has blocks that are NOT in `DistGraph`, `add_block` will fail.
    // We need to update `DistGraph` structure first.
    // But `DistGraph` structure is usually fixed.
    // If we are converting DBCSR -> BlockSpMat, we might be initializing a NEW BlockSpMat.
    // In that case, we need to build the graph from DBCSR blocks.
    
    // Let's implement the counting step.
    // We need to collect all (row, col) pairs from m_brd.
    // Then construct `DistGraph` (or update it).
    // `DistGraph` construction takes `global_adj`.
    
    // But `mat` is passed in. It has a `graph`.
    // If `mat` is empty, we might need to populate `graph`.
    // But `graph` is usually shared.
    
    // If the user intends to populate an EXISTING `BlockSpMat` with compatible structure, we are fine.
    // If the user wants to create a NEW one, we need to build the graph.
    // The prompt says: "DBCSR might gives additional blocks that are not in the original BlockSpMat, so you need to first conting the blocks, preallocate the memory, and then copy the values."
    
    // So we MUST update the graph or create a new one.
    // `BlockSpMat` holds a pointer to `DistGraph`.
    // We can't easily modify `DistGraph` in place if it's shared.
    // But maybe we can assume we can modify it or we are building a new one.
    
    // Let's assume we need to collect edges and call `graph->construct_...`?
    // `DistGraph` has `construct_serial` and `construct_distributed`.
    // We can build a local adjacency list and call `construct_distributed`?
    // Or just `construct_from_triplets`?
    
    // Let's collect edges first.
    // We need to iterate `m_brd` TWICE.
    // First pass: collect edges.
    // Second pass: fill values.
    
    // But `c_dbcsr_iterator` can be reset?
    // Yes, start/stop.
    
    // Let's add the counting/graph construction step.
    
    // 6. Cleanup
    dbcsr_distribution dist_dbcsr;
    c_dbcsr_get_distribution(m_dbcsr, &dist_dbcsr);
    c_dbcsr_release(&m_brd);
    // c_dbcsr_release(&m_dbcsr); // Do not release input matrix!
    c_dbcsr_distribution_release(&dist_brd);
    // c_dbcsr_distribution_release(&dist_dbcsr);
    MPI_Comm_free(&comm_cart);
    // MPI_Comm_free(&group); // Do not free, belongs to m_dbcsr
}

} // namespace backend
} // namespace rsatb

#endif
