#pragma once
#include "atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"
#include <map>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>

using namespace vbcsr;

namespace rsatb {
namespace backend {

enum class PhaseConvention {
    R_ONLY,          // exp(i * K * R)
    R_AND_POSITION   // exp(i * K * (R + r_j - r_i))
};

template <typename T, typename Kernel = DefaultKernel<T>>
class ImageContainer {
public:
    AtomicData* atom_data;
    DistGraph* base_graph; // Reference to atom_data->graph
    
    // Map R vector (rx, ry, rz) to its specific graph and matrix
    std::map<std::vector<int>, DistGraph*> image_graphs;
    std::map<std::vector<int>, BlockSpMat<T, Kernel>*> image_blocks;

    using ComplexT = std::complex<double>;
    // We might need a complex kernel for the result of sample_k if T is real
    // But usually T is already complex for Hamiltonians? 
    // If T is double, sample_k returns complex.
    // Let's assume for now we return BlockSpMat<std::complex<double>>.
    
    // Actually, if T is double, we can't store complex result in BlockSpMat<T>.
    // We need a separate type for the result.
    // Let's define ResultT as complex<double> always for now, or complex<real_part<T>>.
    
    using ResultT = std::complex<double>;
    using ResultKernel = DefaultKernel<ResultT>;

public:
    ImageContainer(AtomicData* data) : atom_data(data), base_graph(data->graph) {
        build_image_graphs();
    }

    ~ImageContainer() {
        for (auto& kv : image_graphs) {
            delete kv.second;
        }
        for (auto& kv : image_blocks) {
            delete kv.second;
        }
    }

    void build_image_graphs() {
        // 1. Identify all unique R vectors in edges
        // We need to scan all edges to find which R they belong to.
        // Edges in AtomicData are stored as: src (local), dst (local), rx, ry, rz.
        
        // We need to group edges by R.
        // Map: R -> adjacency list for that R
        // Adjacency list: src_lid -> list of dst_gid
        
        std::map<std::vector<int>, std::vector<std::vector<int>>> adj_by_r;
        
        int n_owned = atom_data->n_atom;
        
        // Initialize adj for known Rs? No, discover them.
        
        for (int i = 0; i < n_owned; ++i) {
            const auto& edges = atom_data->get_atom_edges(i);
            for (int edge_idx : edges) {
                int dst_lid = atom_data->get_edge_dst(edge_idx);
                int dst_gid = atom_data->get_global_index(dst_lid);
                
                int rx, ry, rz;
                atom_data->get_edge_shift_vec(edge_idx, &rx, &ry, &rz);
                std::vector<int> R = {rx, ry, rz};
                
                if (adj_by_r.find(R) == adj_by_r.end()) {
                    adj_by_r[R].resize(n_owned);
                }
                
                adj_by_r[R][i].push_back(dst_gid);
            }
        }
        
        // Ensure R=0 has diagonal blocks
        std::vector<int> R0 = {0, 0, 0};
        if (adj_by_r.find(R0) == adj_by_r.end()) {
            adj_by_r[R0].resize(n_owned);
        }
        for (int i = 0; i < n_owned; ++i) {
            int gid = atom_data->get_global_index(i);
            adj_by_r[R0][i].push_back(gid);
        }
        
        // Also need to ensure all ranks agree on the set of Rs?
        // DistGraph construction is collective. If a rank has no edges for a specific R, 
        // but other ranks do, it must still participate in DistGraph construction 
        // (passing empty adj is fine, but it needs to call the constructor).
        // So we need to gather all Rs from all ranks.
        
        // 1.1 Collect local Rs
        std::vector<int> local_Rs;
        for (auto& kv : adj_by_r) {
            local_Rs.push_back(kv.first[0]);
            local_Rs.push_back(kv.first[1]);
            local_Rs.push_back(kv.first[2]);
        }
        
        // 1.2 Allgather Rs
        int size = atom_data->size;
        std::vector<int> recv_counts(size);
        int my_count = local_Rs.size();
        MPI_Allgather(&my_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, atom_data->comm);
        
        std::vector<int> displs(size + 1, 0);
        for (int i = 0; i < size; ++i) displs[i+1] = displs[i] + recv_counts[i];
        
        std::vector<int> all_Rs(displs[size]);
        MPI_Allgatherv(local_Rs.data(), my_count, MPI_INT, all_Rs.data(), recv_counts.data(), displs.data(), MPI_INT, atom_data->comm);
        
        // 1.3 Unique Rs
        std::vector<std::vector<int>> unique_Rs;
        for (size_t i = 0; i < all_Rs.size(); i += 3) {
            unique_Rs.push_back({all_Rs[i], all_Rs[i+1], all_Rs[i+2]});
        }
        std::sort(unique_Rs.begin(), unique_Rs.end());
        unique_Rs.erase(std::unique(unique_Rs.begin(), unique_Rs.end()), unique_Rs.end());
        
        // 2. Construct graphs for each R
        std::vector<int> owned_indices(n_owned);
        for(int i=0; i<n_owned; ++i) owned_indices[i] = atom_data->get_global_index(i);
        
        std::vector<int> my_block_sizes(n_owned);
        for(int i=0; i<n_owned; ++i) {
            int norb;
            atom_data->get_atom_norb(i, &norb);
            my_block_sizes[i] = norb;
        }
        
        for (const auto& R : unique_Rs) {
            DistGraph* g = new DistGraph(atom_data->comm);
            
            // Get adj for this R (might be empty)
            std::vector<std::vector<int>> adj;
            if (adj_by_r.count(R)) {
                adj = adj_by_r[R];
            } else {
                adj.resize(n_owned);
            }
            
            // Remove duplicates in adj if any (AtomicData shouldn't have duplicate edges for same R, but good to be safe)
            for (auto& neighbors : adj) {
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            }
            
            g->construct_distributed(owned_indices, my_block_sizes, adj);
            
            image_graphs[R] = g;
            
            // Also allocate the matrix
            BlockSpMat<T, Kernel>* mat = new BlockSpMat<T, Kernel>(g);
            // mat->owns_graph = false; // We manage graph lifetime
            image_blocks[R] = mat;
        }
    }

    void add_block(const std::vector<int>& R, int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = MatrixLayout::ColMajor) {
        if (image_blocks.find(R) == image_blocks.end()) {
             // This should not happen if we built graphs for all Rs in AtomicData.
             // Unless the user is adding a block for an R that didn't exist in the original edges?
             // The requirement says: "if not, it should allocate the block_csr using the graph corresponding to that R."
             // But constructing a distributed graph is collective. We can't just do it on one rank.
             // If this R is new, we must assume all ranks call add_block or we have a mechanism to add new Rs collectively.
             // For now, let's assume R must exist or throw. 
             // Or, if the user meant "allocate the block in the matrix", that's handled by BlockSpMat.
             // If the user meant "create a new graph for a new R", that's complex.
             // Given "The base graph actually corresponding to a summation of graph for all shifted vector R",
             // it implies all possible edges are in base_graph (AtomicData).
             // So R must be in AtomicData.
             throw std::runtime_error("R vector not found in ImageContainer (must be present in AtomicData)");
        }
        
        image_blocks[R]->add_block(global_row, global_col, data, rows, cols, mode, layout);
    }
    
    void assemble() {
        for (auto& kv : image_blocks) {
            kv.second->assemble();
        }
    }

    // Sample K
    // Returns a new BlockSpMat allocated on the base_graph
    BlockSpMat<ResultT, ResultKernel>* sample_k(const std::vector<double>& K, PhaseConvention convention) {
        // 1. Allocate result matrix on base_graph
        // We need a BlockSpMat<ResultT> on base_graph.
        // base_graph is shared, so we don't own it.
        BlockSpMat<ResultT, ResultKernel>* result = new BlockSpMat<ResultT, ResultKernel>(base_graph);
        
        // 2. Iterate over all image blocks and accumulate
        // result = sum_R ( H(R) * phase(R) )
        
        // We need a map: image_graph[R]->col_ind[k] (local col) -> Global Col -> base_graph local col.
        
        int n_owned = base_graph->owned_global_indices.size();
        
        for (auto& kv : image_blocks) {
            const std::vector<int>& R_vec = kv.first;
            BlockSpMat<T, Kernel>* mat_R = kv.second;
            DistGraph* graph_R = mat_R->graph;
            
            double phase_R_val = -2.0 * M_PI * (K[0]*R_vec[0] + K[1]*R_vec[1] + K[2]*R_vec[2]);
            std::complex<double> exp_iKR = std::exp(std::complex<double>(0, phase_R_val));
            
            // Precompute col map for this graph_R
            // local_col_R -> local_col_Base
            int n_col_R = graph_R->block_sizes.size(); // owned + ghosts
            std::vector<int> col_map(n_col_R, -1);
            
            for (int lc = 0; lc < n_col_R; ++lc) {
                int gid = graph_R->get_global_index(lc);
                if (base_graph->global_to_local.count(gid)) {
                    col_map[lc] = base_graph->global_to_local.at(gid);
                } else {
                    // This shouldn't happen if base_graph is superset
                     throw std::runtime_error("Image graph column not found in base graph");
                }
            }
            
            // Iterate blocks
            #pragma omp parallel for
            for (int i = 0; i < n_owned; ++i) {
                int start = mat_R->row_ptr[i];
                int end = mat_R->row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col_R = mat_R->col_ind[k];
                    int col_Base = col_map[col_R];
                    
                    // Find block in result
                    // result->update_local_block(i, col_Base, ...)
                    // But update_local_block takes pointer to data.
                    // We need to compute data first.
                    
                    int n_elem = mat_R->blk_sizes[k];
                    int r_dim = mat_R->graph->block_sizes[i];
                    int c_dim = mat_R->graph->block_sizes[col_R];
                    const T* data_R = mat_R->arena.get_ptr(mat_R->blk_handles[k]);
                    
                    std::vector<ResultT> block_res(n_elem);
                    
                    if (convention == PhaseConvention::R_ONLY) {
                        for (int e = 0; e < n_elem; ++e) {
                            block_res[e] = static_cast<ResultT>(data_R[e]) * exp_iKR;
                        }
                    } else {
                        // R_AND_POSITION
                        
                        // Get positions
                        double ri[3], rj[3];
                        atom_data->get_pos(i, &ri[0], &ri[1], &ri[2]); // i is local owned
                        atom_data->get_pos(col_Base, &rj[0], &rj[1], &rj[2]);
                        
                        double dx = rj[0] - ri[0];
                        double dy = rj[1] - ri[1];
                        double dz = rj[2] - ri[2];
                        
                        // Convert to fractional
                        double fdx = dx, fdy = dy, fdz = dz;
                        atom_data->invert_cell(&fdx, &fdy, &fdz);
                        
                        double phase_pos = -2.0 * M_PI * (K[0]*fdx + K[1]*fdy + K[2]*fdz);
                        std::complex<double> total_phase = exp_iKR * std::exp(std::complex<double>(0, phase_pos));
                        
                        for (int e = 0; e < n_elem; ++e) {
                            block_res[e] = static_cast<ResultT>(data_R[e]) * total_phase;
                        }
                    }
                    
                    // Add to result
                    // We need a thread-safe way if multiple threads write to same block?
                    // Parallel over rows (i). Each row is independent in CSR.
                    // So it is thread safe for different i.
                    // But we are iterating i.
                    
                    // However, result->update_local_block might not be thread safe if it reallocates?
                    // No, update_local_block assumes structure exists.
                    // Does result have the structure?
                    // We initialized result(base_graph).
                    // allocate_from_graph() allocates val with 0.
                    // So structure is fixed.
                    // Writing to val is safe if blocks don't overlap.
                    // Since we iterate i, and for each i we iterate k (cols),
                    // we are writing to block (i, col_Base).
                    // Can multiple R's contribute to same (i, col_Base)?
                    // YES.
                    // Example: R=(0,0,0) connects 1->2. R=(1,0,0) connects 1->2 (periodic image).
                    // In base_graph, these are DIFFERENT edges if they are distinct in input.
                    // Wait.
                    // AtomicData: "Remove duplicates/R".
                    // "base graph actually corresponding to a summation of graph for all shifted vector R"
                    // If 1->2 exists with R1 and R2.
                    // In AtomicData, are they separate edges?
                    // "edges[k] = {src, dst, rx, ry, rz}"
                    // "iconn[src].push_back(k)"
                    // DistGraph adjacency: "adj[src].push_back(dst)"
                    // "Remove duplicates... unique"
                    // So in DistGraph, if 1->2 exists for R1 and R2, it appears ONCE in adj list.
                    // So there is ONE block (1, 2) in BlockSpMat.
                    // But we have contributions from R1 and R2.
                    // So we are summing into the SAME block.
                    // Race condition!
                    
                    // We need atomic add or critical section.
                    // Or accumulate locally and write once?
                    // But the loop is over R (outer) or i (inner)?
                    // I put loop over i inside loop over R.
                    // So multiple threads (handling different i) are safe.
                    // But different R loops run sequentially?
                    // "for (auto& kv : image_blocks)" is serial.
                    // So R1 loop finishes, then R2 loop starts.
                    // Within R1 loop, i is parallel.
                    // So no race condition between Rs.
                    // No race condition between is.
                    // Safe.
                    
                    result->update_local_block(i, col_Base, block_res.data(), r_dim, c_dim, AssemblyMode::ADD, MatrixLayout::ColMajor);
                }
            }
        }
        
        return result;
    }
};

} // namespace backend
} // namespace rsatb
