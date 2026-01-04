#pragma once
#include "vbcsr/dist_graph.hpp"
#include <vector>
#include <map>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>

namespace rsatb {
namespace backend {

class AtomicData {
public:
    DistGraph* graph;
    bool own_graph;

    // Atom attributes (Local + Ghost)
    // Indexed by local index from DistGraph (0 to n_owned+n_ghost-1)
    std::vector<int> atom_type;
    std::vector<int> atom_index; // Original ID from file
    std::vector<double> x, y, z;
    
    // Global info
    std::vector<int> type_norb;
    std::vector<double> cell; // 9 elements
    
    // Edge storage
    // We store edges as provided in input, mapped to local indices.
    struct Edge {
        int src_lid; // Local index of source (must be owned)
        int dst_lid; // Local index of dest (owned or ghost)
        int Rx, Ry, Rz;
    };
    std::vector<Edge> edges;
    
    // Connectivity: local atom index -> list of edge indices in 'edges' vector
    std::vector<std::vector<int>> iconn;

    // Offset for global indexing
    int atom_offset;
    int n_atom;
    int N_atom;
    int n_edge;
    int N_edge;
    
    // Offsets for block-sparse matrix
    std::vector<int> local_offsets; // Local offset of each atom's block
    std::vector<long long> global_offsets; // Global offset of each atom's block
    
    MPI_Comm comm;
    int rank, size;

public:
    AtomicData(DistGraph* g) : graph(g), own_graph(false) {
        comm = graph->comm;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        compute_offsets();
    }

    AtomicData(
      size_t n_atom_, size_t N_atom_, size_t atom_offset_, size_t n_edge_, size_t N_edge_,
      const int *atom_index_in, const int *atom_type_in, const int *edge_index_in, const int *type_norb_in, const int *edge_shift_vec_in,
      const double *cell_in, const double *pos_in,
      MPI_Comm comm_
    ) : atom_offset(atom_offset_), n_atom(n_atom_), N_atom(N_atom_), n_edge(n_edge_), N_edge(N_edge_), comm(comm_), own_graph(true) {
        
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // 1. Setup Graph
        // Prepare adjacency for DistGraph (remove duplicates/R)
        std::vector<int> owned_indices(n_atom);
        std::iota(owned_indices.begin(), owned_indices.end(), atom_offset);
        
        // Compute block sizes
        int max_type = 0;
        for(size_t i=0; i<n_atom; ++i) max_type = std::max(max_type, atom_type_in[i]);
        
        int my_max_type = max_type;
        int global_max_type;
        MPI_Allreduce(&my_max_type, &global_max_type, 1, MPI_INT, MPI_MAX, comm);
        
        // Assuming type_norb_in has enough size. 
        // In DMAtomic, it computes _n_type_ and uses it.
        // We'll trust the input pointer is valid up to global_max_type.
        type_norb.assign(type_norb_in, type_norb_in + global_max_type + 1);
        
        std::vector<int> my_block_sizes(n_atom);
        for(size_t i=0; i<n_atom; ++i) {
            my_block_sizes[i] = type_norb[atom_type_in[i]];
        }
        
        // Build adj
        std::vector<std::vector<int>> adj(n_atom);
        for(size_t k=0; k<n_edge; ++k) {
            int src_gid = edge_index_in[2*k];
            int dst_gid = edge_index_in[2*k+1];
            
            // src should be local
            if (src_gid < atom_offset || src_gid >= atom_offset + n_atom) {
                 throw std::runtime_error("Edge source not local");
            }
            int src_lid = src_gid - atom_offset;
            adj[src_lid].push_back(dst_gid);
        }
        
        // Remove duplicates
        for(auto& neighbors : adj) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        
        graph = new DistGraph(comm); 
        graph->construct_distributed(owned_indices, my_block_sizes, adj);
        
        compute_offsets();
        
        // 2. Store Local Data
        int n_owned = n_atom; 
        int n_ghost = graph->ghost_global_indices.size();
        int total_local = n_owned + n_ghost;
        
        atom_type.resize(total_local);
        atom_index.resize(total_local);
        x.resize(total_local);
        y.resize(total_local);
        z.resize(total_local);
        
        for(int i=0; i<n_owned; ++i) {
            atom_type[i] = atom_type_in[i];
            atom_index[i] = atom_index_in[i];
            x[i] = pos_in[3*i];
            y[i] = pos_in[3*i+1];
            z[i] = pos_in[3*i+2];
        }
        
        cell.assign(cell_in, cell_in + 9);
        
        // 3. Fetch Ghost Data
        exchange_attribute(atom_type);
        exchange_attribute(atom_index);
        exchange_attribute(x);
        exchange_attribute(y);
        exchange_attribute(z);
        
        // 4. Store Edges (mapped to local indices)
        edges.resize(n_edge);
        iconn.resize(total_local); 
        
        for(size_t k=0; k<n_edge; ++k) {
            int src_gid = edge_index_in[2*k];
            int dst_gid = edge_index_in[2*k+1];
            
            int src_lid = graph->global_to_local.at(src_gid);
            int dst_lid = graph->global_to_local.at(dst_gid); 
            
            edges[k] = {src_lid, dst_lid, edge_shift_vec_in[3*k], edge_shift_vec_in[3*k+1], edge_shift_vec_in[3*k+2]};
            
            iconn[src_lid].push_back(k);
        }
    }
    
    ~AtomicData() {
        if (own_graph && graph) delete graph;
    }

    // Accessors
    DistGraph* dm() { return graph; }
    const double* cell_ptr() const { return cell.data(); }
    int n_type() const { return type_norb.size(); } // Approximate
    
    void get_vertex_range_local(int *start, int *end) const {
        *start = 0; *end = n_atom + graph->ghost_global_indices.size();
    }

    void get_edge_range_local(int *start, int *end) const {
        *start = 0; *end = n_edge;
    }
    
    bool atom_is_ghost(int idx) const {
        return idx >= n_atom;
    }
    
    void get_atom_norb(int idx, int *norb) const {
        *norb = type_norb[atom_type[idx]];
    }
    
    void get_atom_id(int idx, int *aid) const {
        *aid = atom_index[idx];
    }
    
    void get_atom_type(int idx, int *tp) const {
        *tp = atom_type[idx];
    }
    
    void get_pos(int idx, double *rx, double *ry, double *rz) const {
        *rx = x[idx];
        *ry = y[idx];
        *rz = z[idx];
    }
    
    int get_global_index(int lid) const {
        if (lid < n_atom) return graph->owned_global_indices[lid];
        return graph->ghost_global_indices[lid - n_atom];
    }
    
    void get_connected_atoms(int edge_idx, int *i, int *j) const {
        *i = edges[edge_idx].src_lid;
        *j = edges[edge_idx].dst_lid;
    }
    
    void get_edge_shift_vec(int edge_idx, int *Rx, int *Ry, int *Rz) const {
        *Rx = edges[edge_idx].Rx;
        *Ry = edges[edge_idx].Ry;
        *Rz = edges[edge_idx].Rz;
    }
    
    void get_offset_local(int idx, int *offset) const {
        *offset = local_offsets[idx];
    }
    
    void get_offset_global(int idx, int *offset) const {
        *offset = static_cast<int>(global_offsets[idx]);
    }

    int get_edge_dst(int edge_idx) const {
        return edges[edge_idx].dst_lid;
    }

    const std::vector<int>& get_atom_edges(int atom_idx) const {
        return iconn[atom_idx];
    }

    // Three-body allocation
    // Returns a new DistGraph representing the 3-body connectivity
    void compute_offsets() {
        if (!graph) return;
        int n_local = graph->block_sizes.size();
        local_offsets.resize(n_local);
        global_offsets.resize(n_local);

        // 1. Local offsets (prefix sum of block sizes)
        int current_offset = 0;
        for(int i=0; i<n_local; ++i) {
            local_offsets[i] = current_offset;
            current_offset += graph->block_sizes[i];
        }

        // 2. Global offsets
        int n_owned = graph->owned_global_indices.size();
        long long my_owned_elements = 0;
        for(int i=0; i<n_owned; ++i) {
            my_owned_elements += graph->block_sizes[i];
        }

        long long rank_global_start = 0;
        MPI_Exscan(&my_owned_elements, &rank_global_start, 1, MPI_LONG_LONG, MPI_SUM, comm);
        if (rank == 0) rank_global_start = 0;

        // Fill owned
        for(int i=0; i<n_owned; ++i) {
            global_offsets[i] = rank_global_start + local_offsets[i];
        }

        // Fill ghosts (Communicate)
        std::vector<long long> send_buf(graph->send_indices.size());
        for(size_t i=0; i<graph->send_indices.size(); ++i) {
            int lid = graph->send_indices[i];
            send_buf[i] = global_offsets[lid];
        }

        std::vector<long long> recv_buf(graph->recv_indices.size());
        
        MPI_Alltoallv(send_buf.data(), graph->send_counts.data(), graph->send_displs.data(), MPI_LONG_LONG,
                      recv_buf.data(), graph->recv_counts.data(), graph->recv_displs.data(), MPI_LONG_LONG, comm);

        for(size_t i=0; i<graph->recv_indices.size(); ++i) {
            int lid = graph->recv_indices[i];
            global_offsets[lid] = recv_buf[i];
        }
    }

    void get_edge_vec(int edge_idx, double *rx, double *ry, double *rz) {
        int i = edges[edge_idx].src_lid;
        int j = edges[edge_idx].dst_lid;
        int Rx = edges[edge_idx].Rx;
        int Ry = edges[edge_idx].Ry;
        int Rz = edges[edge_idx].Rz;
        
        *rx = Rx*cell[0] + Ry*cell[3] + Rz*cell[6];
        *ry = Rx*cell[1] + Ry*cell[4] + Rz*cell[7];
        *rz = Rx*cell[2] + Ry*cell[5] + Rz*cell[8];
        
        *rx += x[j] - x[i];
        *ry += y[j] - y[i];
        *rz += z[j] - z[i];
    }

    void invert_cell(double *x, double *y, double *z) {
        double det = cell[0]*(cell[4]*cell[8] - cell[5]*cell[7]) -
                     cell[1]*(cell[3]*cell[8] - cell[5]*cell[6]) +
                     cell[2]*(cell[3]*cell[7] - cell[4]*cell[6]);
                     
        if (std::abs(det) < 1e-9) return;
        
        double inv[9];
        inv[0] = (cell[4]*cell[8] - cell[5]*cell[7]) / det;
        inv[1] = (cell[2]*cell[7] - cell[1]*cell[8]) / det;
        inv[2] = (cell[1]*cell[5] - cell[2]*cell[4]) / det;
        inv[3] = (cell[5]*cell[6] - cell[3]*cell[8]) / det;
        inv[4] = (cell[0]*cell[8] - cell[2]*cell[6]) / det;
        inv[5] = (cell[2]*cell[3] - cell[0]*cell[5]) / det;
        inv[6] = (cell[3]*cell[7] - cell[4]*cell[6]) / det;
        inv[7] = (cell[1]*cell[6] - cell[0]*cell[7]) / det;
        inv[8] = (cell[0]*cell[4] - cell[1]*cell[3]) / det;
        
        double a = inv[0]*(*x) + inv[1]*(*y) + inv[2]*(*z);
        double b = inv[3]*(*x) + inv[4]*(*y) + inv[5]*(*z);
        double c = inv[6]*(*x) + inv[7]*(*y) + inv[8]*(*z);
        
        *x = a;
        *y = b;
        *z = c;
    }

    DistGraph* allocate_three_body_hopping_matrix(const std::vector<double>& r_max_left, const std::vector<double>& r_max_right) {
        // 1. Build reduced connectivity (riconn)
        // Map: local_atom_i -> list of local_atom_j
        std::vector<std::vector<int>> riconn(atom_type.size());
        
        for(int i=0; i<n_atom; ++i) {
            for(int edge_idx : iconn[i]) {
                int j = edges[edge_idx].dst_lid;
                
                // Check distance
                double rx, ry, rz;
                get_edge_vec(edge_idx, &rx, &ry, &rz);
                double r = std::sqrt(rx*rx + ry*ry + rz*rz);
                
                int itype = atom_type[i];
                int jtype = atom_type[j];
                
                if (r <= (r_max_left[itype] + r_max_right[jtype]) * 0.5) {
                    riconn[i].push_back(j);
                }
            }
            // Sort and unique
            std::sort(riconn[i].begin(), riconn[i].end());
            riconn[i].erase(std::unique(riconn[i].begin(), riconn[i].end()), riconn[i].end());
        }
        
        // 2. Build 3-body connections
        // For each i, and for each j in riconn[i]:
        // We have a path j - i - k (where k is in riconn[i])
        // So j and k are connected in the 3-body matrix.
        // We need to add edges (j, k) for all pairs j, k in riconn[i].
        // Note: j and k can be ghosts.
        
        // We need to send these edges to the owners of j.
        // Map: Owner_of_j -> list of {Global_j, Global_k}
        std::map<int, std::vector<int>> send_map;
        
        for(int i=0; i<n_atom; ++i) {
            const auto& neighbors = riconn[i];
            for(size_t idx_j=0; idx_j<neighbors.size(); ++idx_j) {
                int j = neighbors[idx_j];
                int gid_j = get_gid(j);
                int owner_j = graph->find_owner(gid_j);
                
                for(size_t idx_k=0; idx_k<neighbors.size(); ++idx_k) {
                    int k = neighbors[idx_k];
                    int gid_k = get_gid(k);
                    
                    send_map[owner_j].push_back(gid_j);
                    send_map[owner_j].push_back(gid_k);
                }
            }
        }
        
        // 3. Exchange
        std::vector<int> send_counts(size, 0);
        for(auto& kv : send_map) send_counts[kv.first] = kv.second.size();
        
        std::vector<int> recv_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<int> send_buf(sdispls[size]);
        for(auto& kv : send_map) {
            std::copy(kv.second.begin(), kv.second.end(), send_buf.begin() + sdispls[kv.first]);
        }
        
        std::vector<int> recv_buf(rdispls[size]);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, comm);
                      
        // 4. Process received edges and build matrix structure
        // Received: pairs of (Global_j, Global_k). I own j.
        std::vector<std::vector<int>> matrix_adj(n_atom);
        
        for(size_t i=0; i<recv_buf.size(); i+=2) {
            int gid_j = recv_buf[i];
            int gid_k = recv_buf[i+1];
            
            if (graph->global_to_local.find(gid_j) == graph->global_to_local.end()) {
                 throw std::runtime_error("Received 3-body edge for non-owned atom");
            }
            int lid_j = graph->global_to_local.at(gid_j);
            
            matrix_adj[lid_j].push_back(gid_k);
        }
        
        // 5. Finalize structure
        for(int i=0; i<n_atom; ++i) {
            std::sort(matrix_adj[i].begin(), matrix_adj[i].end());
            matrix_adj[i].erase(std::unique(matrix_adj[i].begin(), matrix_adj[i].end()), matrix_adj[i].end());
        }
        
        // 6. Create new DistGraph
        DistGraph* new_graph = new DistGraph(comm);
        
        std::vector<int> owned_indices(n_atom);
        std::iota(owned_indices.begin(), owned_indices.end(), atom_offset);
        
        std::vector<int> my_block_sizes(n_atom);
        for(int i=0; i<n_atom; ++i) {
            my_block_sizes[i] = type_norb[atom_type[i]];
        }
        
        new_graph->construct_distributed(owned_indices, my_block_sizes, matrix_adj);
        
        return new_graph;
    }

private:
    template<typename T>
    void exchange_attribute(std::vector<T>& data) {
        // Exchange data for ghosts using graph's communication pattern
        // 1. Pack send buffer (what others need from me)
        std::vector<T> send_buf(graph->send_indices.size());
        for(size_t i=0; i<graph->send_indices.size(); ++i) {
            send_buf[i] = data[graph->send_indices[i]];
        }
        
        // 2. Prepare recv buffer
        std::vector<T> recv_buf(graph->recv_indices.size());
        
        // 3. Exchange
        // We need byte-wise exchange or type-aware.
        // MPI_Alltoallv needs displacements.
        // DistGraph has send_displs/recv_displs in terms of INDICES (count).
        // We need to scale by sizeof(T) if using MPI_BYTE, or use MPI Type.
        // Let's use MPI_BYTE for simplicity.
        
        size_t type_size = sizeof(T);
        std::vector<int> sdispls_bytes(size + 1);
        std::vector<int> rdispls_bytes(size + 1);
        std::vector<int> send_counts_bytes(size);
        std::vector<int> recv_counts_bytes(size);
        
        for(int i=0; i<size; ++i) {
            send_counts_bytes[i] = graph->send_counts[i] * type_size;
            recv_counts_bytes[i] = graph->recv_counts[i] * type_size;
            sdispls_bytes[i] = graph->send_displs[i] * type_size;
            rdispls_bytes[i] = graph->recv_displs[i] * type_size;
        }
        
        MPI_Alltoallv(send_buf.data(), send_counts_bytes.data(), sdispls_bytes.data(), MPI_BYTE,
                      recv_buf.data(), recv_counts_bytes.data(), rdispls_bytes.data(), MPI_BYTE, comm);
                      
        // 4. Unpack
        for(size_t i=0; i<graph->recv_indices.size(); ++i) {
            data[graph->recv_indices[i]] = recv_buf[i];
        }
    }
    
    int get_gid(int lid) const {
        if (lid < n_atom) return graph->owned_global_indices[lid];
        return graph->ghost_global_indices[lid - n_atom];
    }
    

};

}
}
