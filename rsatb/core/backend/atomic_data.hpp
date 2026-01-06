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
#include "neighbourlist.hpp"
#include "io.hpp"
#include <parmetis.h>

namespace rsatb {
namespace backend {

class AtomicData {
public:
    DistGraph* graph;
    bool own_graph;

    // Atom attributes (Local + Ghost)
    // Indexed by local index from DistGraph (0 to n_owned+n_ghost-1)
    std::vector<int> atom_type={};
    std::vector<int> atom_index={}; // Original ID from file
    std::vector<double> x={}; std::vector<double> y={}; std::vector<double> z={};
    
    // Global info
    std::vector<int> type_norb={};
    std::vector<double> cell={}; // 9 elements
    
    // Edge storage
    // We store edges as provided in input, mapped to local indices.
    struct Edge {
        int src; // Local index of source (must be owned)
        int dst; // Local index of dest (owned or ghost)
        int rx, ry, rz;
    };
    std::vector<Edge> edges={};
    
    // Connectivity: local atom index -> list of edge indices in 'edges' vector
    std::vector<std::vector<int>> iconn={};

    // Offset for global indexing
    int atom_offset=0;
    int n_atom=0;
    int N_atom=0;
    int n_edge=0;
    int N_edge=0;
    
    // Offsets for block-sparse matrix
    std::vector<int> local_offsets={}; // Local offset of each atom's block
    std::vector<long long> global_offsets={}; // Global offset of each atom's block
    
    MPI_Comm comm=MPI_COMM_WORLD;
    int rank=0, size=1;

private:
    // Internal struct for passing edge data during construction
    // struct EdgeInfo { int src, dst, rx, ry, rz; }; // Removed, use Edge instead


public:
    AtomicData(DistGraph* g) : graph(g), own_graph(false) {
        comm = graph->comm;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        compute_offsets();
    }

    AtomicData(MPI_Comm comm_) : comm(comm_), own_graph(false) {}

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
        std::vector<int> my_block_sizes(n_atom);
        std::vector<std::vector<int>> adj(n_atom);
        
        // Determine max_type and populate type_norb
        int max_type = 0;
        for(size_t i=0; i<n_atom; ++i) max_type = std::max(max_type, atom_type_in[i]);
        int my_max_type = max_type;
        int global_max_type;
        MPI_Allreduce(&my_max_type, &global_max_type, 1, MPI_INT, MPI_MAX, comm);
        type_norb.assign(type_norb_in, type_norb_in + global_max_type + 1);

        for(int i=0; i<n_atom; ++i) {
            owned_indices[i] = i + atom_offset;
            my_block_sizes[i] = type_norb[atom_type_in[i]];
        }

        // Build adj
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

        // add onsite edges
        for(int i=0; i<n_atom; ++i) {
            adj[i].push_back(i + atom_offset);
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
        iconn.resize(n_owned);
        
        for(size_t k=0; k<n_edge; ++k) {
            int src_gid = edge_index_in[2*k];
            int dst_gid = edge_index_in[2*k+1];
            
            int src_lid = graph->global_to_local.at(src_gid);
            int dst_lid = graph->global_to_local.at(dst_gid); 
            
            edges[k] = {src_lid, dst_lid, edge_shift_vec_in[3*k], edge_shift_vec_in[3*k+1], edge_shift_vec_in[3*k+2]};
            
            iconn[src_lid].push_back(k);
        }
    }
    
    static AtomicData* from_points(const std::vector<double>& pos, const std::vector<int>& z, 
                                   const std::vector<double>& cell, const std::vector<bool>& pbc, 
                                   const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb_in, 
                                   MPI_Comm comm) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // 1. Rank 0: Process Input & Initial Partition
        std::vector<int> type_norb = type_norb_in;
        std::vector<double> pos_sorted;
        std::vector<int> z_sorted;
        std::vector<int> types_sorted;
        std::vector<int> indices_sorted;
        std::vector<int> send_counts(size, 0);
        std::vector<int> send_displs(size + 1, 0);
        std::vector<std::vector<Edge>> edges_to_send(size);
        
        int n_global = 0;
        if (rank == 0) {
            n_global = z.size();
            process_input_rank0(size, pos, z, cell, pbc, r_max_per_type, type_norb,
                                pos_sorted, z_sorted, types_sorted, indices_sorted,
                                send_counts, send_displs, edges_to_send);
        }

        
        MPI_Bcast(&n_global, 1, MPI_INT, 0, comm);

        if (n_global == 0) {
            return new AtomicData(comm);
        }
        
        // 2. Scatter Atoms and Edges
        std::vector<double> my_pos;
        std::vector<int> my_z;
        std::vector<int> my_types;
        std::vector<int> my_indices;
        std::vector<int> my_edges_flat;
        int my_n_atom;
        
        scatter_initial_data(comm, rank, size, n_global, pos_sorted, z_sorted, types_sorted, indices_sorted,
                             send_counts, send_displs, edges_to_send, type_norb,
                             my_n_atom, my_pos, my_z, my_types, my_indices, my_edges_flat);
        
        // 3. ParMETIS Partitioning
        std::vector<int> vtxdist;
        std::vector<int> xadj;
        std::vector<int> adjncy;
        int my_start;
        
        // this step build temporary graph for partitioning
        build_parmetis_graph(comm, rank, size, my_n_atom, my_edges_flat, vtxdist, xadj, adjncy, my_start);
        
        std::vector<int> part(my_n_atom);
        std::fill(part.begin(), part.end(), rank);
        
        partition_graph(vtxdist, xadj, adjncy, size, part, comm);
        
        // 4. Redistribute Atoms
        std::vector<double> r_pos;
        std::vector<int> r_z;
        std::vector<int> r_types;
        std::vector<int> r_indices;
        int total_recv;
        
        std::vector<int> my_inter_indices(my_n_atom);
        for(int i=0; i<my_n_atom; ++i) my_inter_indices[i] = my_start + i;
        
        std::vector<int> r_inter_indices;
        redistribute_atoms(comm, rank, size, my_n_atom, part, my_pos, my_z, my_types, my_indices, my_inter_indices,
                           r_pos, r_z, r_types, r_indices, r_inter_indices, total_recv);
        
        // 5. Redistribute Edges
        std::vector<int> r_edges;
        redistribute_edges(comm, rank, size, my_start, my_edges_flat, part, r_edges);
        
        // 6. Re-map IDs to be contiguous on each rank
        std::vector<int> all_recv_counts(size);
        MPI_Allgather(&total_recv, 1, MPI_INT, all_recv_counts.data(), 1, MPI_INT, comm);
        std::vector<int> all_recv_displs(size + 1, 0);
        for(int i=0; i<size; ++i) all_recv_displs[i+1] = all_recv_displs[i] + all_recv_counts[i];
        
        std::vector<int> all_r_inter_indices(n_global);
        MPI_Allgatherv(r_inter_indices.data(), total_recv, MPI_INT, all_r_inter_indices.data(), all_recv_counts.data(), all_recv_displs.data(), MPI_INT, comm);
        
        std::vector<int> inter_to_final(n_global);
        for(int i=0; i<n_global; ++i) {
            inter_to_final[all_r_inter_indices[i]] = i;
        }
        
        // Update r_edges to use final IDs
        for(size_t k=0; k < r_edges.size() / 5; ++k) {
            r_edges[5*k] = inter_to_final[r_edges[5*k]];
            r_edges[5*k+1] = inter_to_final[r_edges[5*k+1]];
        }
                      
        // 7. Construct AtomicData
        return construct_final_object(comm, rank, size, cell, total_recv, r_indices, r_types, r_pos, r_edges, type_norb);
    }
    
    static AtomicData* from_file(const std::string& filename, const std::vector<double>& r_max_per_type, std::vector<int> type_norb, MPI_Comm comm, const std::string& format="") {
        int rank;
        MPI_Comm_rank(comm, &rank);
        
        std::vector<double> pos;
        std::vector<int> z;
        std::vector<double> cell;
        std::vector<bool> pbc;
        
        if (rank == 0) {
            io::StructureData data;
            std::string fmt = format;
            if (fmt.empty()) {
                if (filename.find(".vasp") != std::string::npos || filename.find("POSCAR") != std::string::npos) fmt = "POSCAR";
                else if (filename.find(".xyz") != std::string::npos) fmt = "XYZ";
                else throw std::runtime_error("Unknown file format");
            }
            
            if (fmt == "POSCAR") data = io::read_poscar(filename);
            else if (fmt == "XYZ") data = io::read_xyz(filename);
            else throw std::runtime_error("Unsupported format: " + fmt);
            
            pos = data.pos;
            z = data.z;
            cell = data.cell;
            pbc = data.pbc;
        }
        
        return from_points(pos, z, cell, pbc, r_max_per_type, type_norb, comm);
    }

    ~AtomicData() {
        if (own_graph && graph) delete graph;
    }

    // Accessors
    DistGraph* get_graph() { return graph; }
    const double* cell_ptr() const { return cell.data(); }
    int n_type() const { return type_norb.size(); } 
    
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
        *i = edges[edge_idx].src;
        *j = edges[edge_idx].dst;
    }
    
    void get_edge_shift_vec(int edge_idx, int *Rx, int *Ry, int *Rz) const {
        *Rx = edges[edge_idx].rx;
        *Ry = edges[edge_idx].ry;
        *Rz = edges[edge_idx].rz;
    }
    
    void get_offset_local(int idx, int *offset) const {
        *offset = local_offsets[idx];
    }
    
    void get_offset_global(int idx, int *offset) const {
        *offset = static_cast<int>(global_offsets[idx]);
    }

    int get_edge_dst(int edge_idx) const {
        return edges[edge_idx].dst;
    }

    const std::vector<int>& get_atom_edges(int atom_idx) const {
        return iconn[atom_idx];
    }

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
        int i = edges[edge_idx].src;
        int j = edges[edge_idx].dst;
        int Rx = edges[edge_idx].rx;
        int Ry = edges[edge_idx].ry;
        int Rz = edges[edge_idx].rz;
        
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

    // Compute global number of orbitals
    int norb() {
        int local_norb = 0;
        for(int i=0; i<n_atom; ++i) {
            local_norb += type_norb[atom_type[i]];
        }
        int global_norb = 0;
        MPI_Allreduce(&local_norb, &global_norb, 1, MPI_INT, MPI_SUM, comm);
        return global_norb;
    }

    // Compute cell volume
    double volume() {
        if (cell.size() < 9) return 0.0;
        double det = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
                   - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
                   + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
        return std::abs(det);
    }

    DistGraph* get_graph3b(const std::vector<double>& r_max_left, const std::vector<double>& r_max_right) {
        // 1. Build reduced connectivity (riconn)
        std::vector<std::vector<int>> riconn(atom_type.size());
        
        for(int i=0; i<n_atom; ++i) {
            for(int edge_idx : iconn[i]) {
                int j = edges[edge_idx].dst;
                
                double rx, ry, rz;
                get_edge_vec(edge_idx, &rx, &ry, &rz);
                double r = std::sqrt(rx*rx + ry*ry + rz*rz);
                
                int itype = atom_type[i];
                int jtype = atom_type[j];
                
                if (r <= r_max_left[itype] + r_max_right[jtype] + 1e-9) {
                    riconn[i].push_back(j);
                }
            }
            std::sort(riconn[i].begin(), riconn[i].end());
            riconn[i].erase(std::unique(riconn[i].begin(), riconn[i].end()), riconn[i].end());
        }

        
        std::map<int, std::vector<std::pair<int, int>>> send_map;
        
        for(int i=0; i<n_atom; ++i) {
            const auto& neighbors = riconn[i];
            for(size_t idx_j=0; idx_j<neighbors.size(); ++idx_j) {
                int j = neighbors[idx_j];
                int gid_j = get_gid(j);
                int owner_j = graph->find_owner(gid_j);
                
                for(size_t idx_k=0; idx_k<neighbors.size(); ++idx_k) {
                    int k = neighbors[idx_k];
                    int gid_k = get_gid(k);
                    send_map[owner_j].push_back({gid_j, gid_k});
                }
            }
        }

        for(int i=0; i<size; ++i) {
            std::sort(send_map[i].begin(), send_map[i].end());
            send_map[i].erase(std::unique(send_map[i].begin(), send_map[i].end()), send_map[i].end());
        }
        
        std::vector<int> send_counts(size, 0);
        for(auto& kv : send_map) send_counts[kv.first] = kv.second.size() * 2;
        
        std::vector<int> recv_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<int> send_buf(sdispls[size]);
        for(auto& kv : send_map) {
            int offset = sdispls[kv.first];
            for(const auto& p : kv.second) {
                send_buf[offset++] = p.first;
                send_buf[offset++] = p.second;
            }
        }
        
        std::vector<int> recv_buf(rdispls[size]);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, comm);
                      
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

        // add two-body and onsite edges
        for(int i=0; i<n_atom; ++i) {
            for(int j_lid : graph->adj_list[i]) {
                int j_gid = get_gid(j_lid);
                matrix_adj[i].push_back(j_gid);
            }
        }
        
        for(int i=0; i<n_atom; ++i) {
            std::sort(matrix_adj[i].begin(), matrix_adj[i].end());
            matrix_adj[i].erase(std::unique(matrix_adj[i].begin(), matrix_adj[i].end()), matrix_adj[i].end());
        }
        
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
        std::vector<T> send_buf(graph->send_indices.size());
        for(size_t i=0; i<graph->send_indices.size(); ++i) {
            send_buf[i] = data[graph->send_indices[i]];
        }
        
        std::vector<T> recv_buf(graph->recv_indices.size());
        
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
                      
        for(size_t i=0; i<graph->recv_indices.size(); ++i) {
            data[graph->recv_indices[i]] = recv_buf[i];
        }
    }
    
    int get_gid(int lid) const {
        if (lid < n_atom) return graph->owned_global_indices[lid];
        return graph->ghost_global_indices[lid - n_atom];
    }

    static void partition_graph(const std::vector<int>& vtxdist, const std::vector<int>& xadj, const std::vector<int>& adjncy, 
                                int nparts, std::vector<int>& part, MPI_Comm comm) {
        idx_t wgtflag = 0; 
        idx_t numflag = 0; 
        idx_t ncon = 1;    
        idx_t nparts_t = nparts;
        
        std::vector<real_t> tpwgts(ncon * nparts, 1.0 / nparts);
        real_t ubvec[1] = {1.05}; 
        
        idx_t options[3] = {0, 0, 0};
        idx_t edgecut;
        
        MPI_Comm comm_ = comm;
        
        if (vtxdist.back() == 0) return;

        std::vector<idx_t> vtxdist_t(vtxdist.begin(), vtxdist.end());
        std::vector<idx_t> xadj_t(xadj.begin(), xadj.end());
        std::vector<idx_t> adjncy_t(adjncy.begin(), adjncy.end());
        std::vector<idx_t> part_t(part.begin(), part.end());
        
        idx_t* adjncy_ptr = adjncy_t.empty() ? NULL : adjncy_t.data();
        // ParMETIS might complain if adjncy is NULL but xadj implies edges?
        // If no edges, xadj should be all 0 (or constant).
        // But if adjncy is empty, we should probably pass a valid pointer to avoid NULL check error if ParMETIS is picky.
        // Actually, if vtxdist is not 0, but no edges, adjncy is empty.
        // Let's pass a dummy variable if empty.
        idx_t dummy_adj = 0;
        if (adjncy_t.empty()) adjncy_ptr = &dummy_adj;

        ParMETIS_V3_RefineKway(vtxdist_t.data(), xadj_t.data(), adjncy_ptr, 
                               NULL, NULL, &wgtflag, &numflag, &ncon, &nparts_t, 
                               tpwgts.data(), ubvec, options, &edgecut, part_t.data(), &comm_);
                               
        for(size_t i=0; i<part.size(); ++i) part[i] = part_t[i];
    }

    static void process_input_rank0(
        int size,
        const std::vector<double>& pos, const std::vector<int>& z,
        const std::vector<double>& cell, const std::vector<bool>& pbc,
        const std::vector<double>& r_max_per_type,
        std::vector<int>& type_norb,
        std::vector<double>& pos_sorted,
        std::vector<int>& z_sorted,
        std::vector<int>& types_sorted,
        std::vector<int>& indices_sorted,
        std::vector<int>& send_counts,
        std::vector<int>& send_displs,
        std::vector<std::vector<Edge>>& edges_to_send
    ) {
        // this should only be called by rank 0
        int n_global = z.size();
        
        // 1.1 Types
        std::vector<int> unique_z = z;
        std::sort(unique_z.begin(), unique_z.end());
        unique_z.erase(std::unique(unique_z.begin(), unique_z.end()), unique_z.end());
        int n_types = unique_z.size();
        std::map<int, int> z_to_type;
        for(int i=0; i<n_types; ++i) z_to_type[unique_z[i]] = i;
        
        std::vector<int> atom_types(n_global);
        for(int i=0; i<n_global; ++i) atom_types[i] = z_to_type[z[i]];
        
        if (type_norb.empty()) type_norb.assign(n_types, 1);
        
        // 1.2 NeighborList
        double max_r = 0;
        for(double r : r_max_per_type) max_r = std::max(max_r, r);
        NeighborList nl;
        nl.build(pos, cell, pbc, max_r * 2.0); 
        
        // 1.3 Hilbert Sort
        double min_p[3] = {1e30, 1e30, 1e30};
        double max_p[3] = {-1e30, -1e30, -1e30};
        for(int i=0; i<n_global; ++i) {
            for(int k=0; k<3; ++k) {
                min_p[k] = std::min(min_p[k], pos[3*i+k]);
                max_p[k] = std::max(max_p[k], pos[3*i+k]);
            }
        }

        double range[3];
        for(int k=0; k<3; ++k) range[k] = max_p[k] - min_p[k] + 1e-9;
        
        std::vector<std::pair<uint64_t, int>> morton_codes(n_global);
        for(int i=0; i<n_global; ++i) {
            uint64_t code = 0;
            for(int k=0; k<3; ++k) {
                double n = (pos[3*i+k] - min_p[k]) / range[k];
                uint64_t u = (uint64_t)(n * 2097152.0); 
                for(int b=0; b<21; ++b) if ((u >> b) & 1) code |= (1ULL << (3*b + k));
            }
            morton_codes[i] = {code, i};
        }
        std::sort(morton_codes.begin(), morton_codes.end());
        
        // 1.4 Assign Ranks & Sort Data
        int atoms_per_rank = n_global / size;
        int remainder = n_global % size;
        int current_rank = 0;
        int current_count = 0;
        int target_count = atoms_per_rank + (0 < remainder ? 1 : 0);
        
        pos_sorted.resize(n_global * 3);
        z_sorted.resize(n_global);
        types_sorted.resize(n_global);
        indices_sorted.resize(n_global);
        std::vector<int> old_to_inter_gid(n_global);
        std::vector<int> inter_gid_to_rank(n_global);
        
        for(int i=0; i<n_global; ++i) {
            int old_idx = morton_codes[i].second;
            if (current_count >= target_count) {
                current_rank++;
                current_count = 0;
                target_count = atoms_per_rank + (current_rank < remainder ? 1 : 0);
            }
            send_counts[current_rank]++;
            current_count++;
            
            old_to_inter_gid[old_idx] = i;
            inter_gid_to_rank[i] = current_rank;
            
            pos_sorted[3*i] = pos[3*old_idx];
            pos_sorted[3*i+1] = pos[3*old_idx+1];
            pos_sorted[3*i+2] = pos[3*old_idx+2];
            z_sorted[i] = z[old_idx];
            types_sorted[i] = atom_types[old_idx];
            indices_sorted[i] = old_idx; 
        }
        
        // 1.5 Process Edges
        for(int old_i=0; old_i<n_global; ++old_i) {
            int inter_i = old_to_inter_gid[old_i];
            int rank_i = inter_gid_to_rank[inter_i];
            int type_i = atom_types[old_i];
            
            const auto& neighbors = nl.get_neighbors(old_i);
            for(const auto& n : neighbors) {
                int old_j = n.index;
                int inter_j = old_to_inter_gid[old_j];
                int type_j = atom_types[old_j];
                
                double r_cut = r_max_per_type[type_i] + r_max_per_type[type_j];
                
                double dx = pos[3*old_j] - pos[3*old_i] + n.rx*cell[0] + n.ry*cell[3] + n.rz*cell[6];
                double dy = pos[3*old_j+1] - pos[3*old_i+1] + n.rx*cell[1] + n.ry*cell[4] + n.rz*cell[7];
                double dz = pos[3*old_j+2] - pos[3*old_i+2] + n.rx*cell[2] + n.ry*cell[5] + n.rz*cell[8];
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                
                if (dist > r_cut + 1e-9) continue;
                
                edges_to_send[rank_i].push_back({inter_i, inter_j, n.rx, n.ry, n.rz});
            }
        }
        for(int i=0; i<size; ++i) send_displs[i+1] = send_displs[i] + send_counts[i];
    }

    static void scatter_initial_data(
        MPI_Comm comm, int rank, int size, int n_global,
        const std::vector<double>& pos_sorted,
        const std::vector<int>& z_sorted,
        const std::vector<int>& types_sorted,
        const std::vector<int>& indices_sorted,
        const std::vector<int>& send_counts,
        const std::vector<int>& send_displs,
        const std::vector<std::vector<Edge>>& edges_to_send,
        std::vector<int>& type_norb,
        int& my_n_atom,
        std::vector<double>& my_pos,
        std::vector<int>& my_z,
        std::vector<int>& my_types,
        std::vector<int>& my_indices,
        std::vector<int>& my_edges_flat
    ) {
        MPI_Bcast(&n_global, 1, MPI_INT, 0, comm);
        int n_types = type_norb.size();
        MPI_Bcast(&n_types, 1, MPI_INT, 0, comm);
        if (rank != 0) type_norb.resize(n_types);
        MPI_Bcast(type_norb.data(), n_types, MPI_INT, 0, comm);
        
        MPI_Scatter(send_counts.data(), 1, MPI_INT, &my_n_atom, 1, MPI_INT, 0, comm);
        
        my_pos.resize(my_n_atom * 3);
        my_z.resize(my_n_atom);
        my_types.resize(my_n_atom);
        my_indices.resize(my_n_atom);
        
        std::vector<int> send_counts_3(size), send_displs_3(size + 1);
        if (rank == 0) {
            for(int i=0; i<size; ++i) {
                send_counts_3[i] = send_counts[i] * 3;
                send_displs_3[i] = send_displs[i] * 3;
            }
            send_displs_3[size] = send_displs[size] * 3;
        }
        
        MPI_Scatterv(pos_sorted.data(), send_counts_3.data(), send_displs_3.data(), MPI_DOUBLE,
                     my_pos.data(), my_n_atom * 3, MPI_DOUBLE, 0, comm);
        MPI_Scatterv(z_sorted.data(), send_counts.data(), send_displs.data(), MPI_INT,
                     my_z.data(), my_n_atom, MPI_INT, 0, comm);
        MPI_Scatterv(types_sorted.data(), send_counts.data(), send_displs.data(), MPI_INT,
                     my_types.data(), my_n_atom, MPI_INT, 0, comm);
        MPI_Scatterv(indices_sorted.data(), send_counts.data(), send_displs.data(), MPI_INT,
                     my_indices.data(), my_n_atom, MPI_INT, 0, comm);

        // Scatter Edges
        int my_n_edge_tuples;
        std::vector<int> edge_send_counts(size);
        if (rank == 0) {
            for(int i=0; i<size; ++i) edge_send_counts[i] = edges_to_send[i].size() * 5; 
        }
        MPI_Scatter(edge_send_counts.data(), 1, MPI_INT, &my_n_edge_tuples, 1, MPI_INT, 0, comm);
        
        my_edges_flat.resize(my_n_edge_tuples);
        std::vector<int> edge_displs(size + 1, 0);
        std::vector<int> all_edges_flat;
        if (rank == 0) {
            for(int i=0; i<size; ++i) edge_displs[i+1] = edge_displs[i] + edge_send_counts[i];
            all_edges_flat.resize(edge_displs[size]);
            int offset = 0;
            for(int i=0; i<size; ++i) {
                for(const auto& e : edges_to_send[i]) {
                    all_edges_flat[offset++] = e.src;
                    all_edges_flat[offset++] = e.dst;
                    all_edges_flat[offset++] = e.rx;
                    all_edges_flat[offset++] = e.ry;
                    all_edges_flat[offset++] = e.rz;
                }
            }
        }
        MPI_Scatterv(all_edges_flat.data(), edge_send_counts.data(), edge_displs.data(), MPI_INT,
                     my_edges_flat.data(), my_n_edge_tuples, MPI_INT, 0, comm);
    }

    static void build_parmetis_graph(
        MPI_Comm comm, int rank, int size, int my_n_atom,
        const std::vector<int>& my_edges_flat,
        std::vector<int>& vtxdist,
        std::vector<int>& xadj,
        std::vector<int>& adjncy,
        int& my_start
    ) {
        std::vector<int> my_vtxdist(size);
        MPI_Allgather(&my_n_atom, 1, MPI_INT, my_vtxdist.data(), 1, MPI_INT, comm);
        vtxdist.assign(size + 1, 0);
        for(int i=0; i<size; ++i) vtxdist[i+1] = vtxdist[i] + my_vtxdist[i];
        
        my_start = vtxdist[rank];
        xadj.assign(my_n_atom + 1, 0);
        std::vector<std::vector<int>> adj_list(my_n_atom);
        
        int n_edges = my_edges_flat.size() / 5;
        for(int k=0; k<n_edges; ++k) {
            int src = my_edges_flat[5*k];
            int dst = my_edges_flat[5*k+1];
            int lid = src - my_start;
            if (lid < 0 || lid >= my_n_atom) throw std::runtime_error("Edge source mismatch");
            adj_list[lid].push_back(dst);
        }
        
        int current_offset = 0;
        for(int i=0; i<my_n_atom; ++i) {
            xadj[i] = current_offset;
            std::sort(adj_list[i].begin(), adj_list[i].end()); 
            for(int dst : adj_list[i]) adjncy.push_back(dst);
            current_offset += adj_list[i].size();
        }
        xadj[my_n_atom] = current_offset;
    }

    static void redistribute_atoms(
        MPI_Comm comm, int rank, int size, int my_n_atom,
        const std::vector<int>& part,
        const std::vector<double>& my_pos,
        const std::vector<int>& my_z,
        const std::vector<int>& my_types,
        const std::vector<int>& my_indices,
        const std::vector<int>& my_inter_indices,
        std::vector<double>& r_pos,
        std::vector<int>& r_z,
        std::vector<int>& r_types,
        std::vector<int>& r_indices,
        std::vector<int>& r_inter_indices,
        int& total_recv
    ) {
        std::vector<std::vector<int>> atoms_to_send(size);
        for(int i=0; i<my_n_atom; ++i) atoms_to_send[part[i]].push_back(i);
        
        std::vector<int> send_cnts(size), recv_cnts(size);
        for(int i=0; i<size; ++i) send_cnts[i] = atoms_to_send[i].size();
        MPI_Alltoall(send_cnts.data(), 1, MPI_INT, recv_cnts.data(), 1, MPI_INT, comm);
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_cnts[i];
            rdispls[i+1] = rdispls[i] + recv_cnts[i];
        }
        
        int total_send = sdispls[size];
        std::vector<double> s_pos(total_send * 3);
        std::vector<int> s_z(total_send);
        std::vector<int> s_types(total_send);
        std::vector<int> s_indices(total_send);
        std::vector<int> s_inter_indices(total_send);
        
        int offset = 0;
        for(int r=0; r<size; ++r) {
            for(int lid : atoms_to_send[r]) {
                s_pos[3*offset] = my_pos[3*lid];
                s_pos[3*offset+1] = my_pos[3*lid+1];
                s_pos[3*offset+2] = my_pos[3*lid+2];
                s_z[offset] = my_z[lid];
                s_types[offset] = my_types[lid];
                s_indices[offset] = my_indices[lid];
                s_inter_indices[offset] = my_inter_indices[lid];
                offset++;
            }
        }
        
        total_recv = rdispls[size];
        r_pos.resize(total_recv * 3);
        r_z.resize(total_recv);
        r_types.resize(total_recv);
        r_indices.resize(total_recv);
        r_inter_indices.resize(total_recv);
        
        std::vector<int> send_cnts_3(size), recv_cnts_3(size), sdispls_3(size+1), rdispls_3(size+1);
        for(int i=0; i<size; ++i) {
            send_cnts_3[i] = send_cnts[i] * 3;
            recv_cnts_3[i] = recv_cnts[i] * 3;
            sdispls_3[i] = sdispls[i] * 3;
            rdispls_3[i] = rdispls[i] * 3;
        }
        sdispls_3[size] = sdispls[size] * 3;
        rdispls_3[size] = rdispls[size] * 3;
        
        MPI_Alltoallv(s_pos.data(), send_cnts_3.data(), sdispls_3.data(), MPI_DOUBLE,
                      r_pos.data(), recv_cnts_3.data(), rdispls_3.data(), MPI_DOUBLE, comm);
        MPI_Alltoallv(s_z.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                      r_z.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
        MPI_Alltoallv(s_types.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                      r_types.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
        MPI_Alltoallv(s_indices.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                      r_indices.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
        MPI_Alltoallv(s_inter_indices.data(), send_cnts.data(), sdispls.data(), MPI_INT,
                      r_inter_indices.data(), recv_cnts.data(), rdispls.data(), MPI_INT, comm);
    }

    static void redistribute_edges(
        MPI_Comm comm, int rank, int size, int my_start,
        const std::vector<int>& my_edges_flat,
        const std::vector<int>& part,
        std::vector<int>& r_edges
    ) {
        std::vector<std::vector<int>> edges_to_send_final(size);
        std::vector<int> lid_to_rank = part;
        
        int n_edges = my_edges_flat.size() / 5;
        for(int k=0; k<n_edges; ++k) {
            int src = my_edges_flat[5*k];
            int dst = my_edges_flat[5*k+1];
            int rx = my_edges_flat[5*k+2];
            int ry = my_edges_flat[5*k+3];
            int rz = my_edges_flat[5*k+4];
            
            int lid = src - my_start;
            int target_rank = lid_to_rank[lid];
            
            edges_to_send_final[target_rank].push_back(src);
            edges_to_send_final[target_rank].push_back(dst);
            edges_to_send_final[target_rank].push_back(rx);
            edges_to_send_final[target_rank].push_back(ry);
            edges_to_send_final[target_rank].push_back(rz);
        }
        
        std::vector<int> e_send_cnts(size), e_recv_cnts(size);
        for(int i=0; i<size; ++i) e_send_cnts[i] = edges_to_send_final[i].size();
        MPI_Alltoall(e_send_cnts.data(), 1, MPI_INT, e_recv_cnts.data(), 1, MPI_INT, comm);
        
        std::vector<int> e_sdispls(size + 1, 0), e_rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            e_sdispls[i+1] = e_sdispls[i] + e_send_cnts[i];
            e_rdispls[i+1] = e_rdispls[i] + e_recv_cnts[i];
        }
        
        std::vector<int> s_edges(e_sdispls[size]);
        int offset = 0;
        for(int i=0; i<size; ++i) {
            std::copy(edges_to_send_final[i].begin(), edges_to_send_final[i].end(), s_edges.begin() + offset);
            offset += e_send_cnts[i];
        }
        
        r_edges.resize(e_rdispls[size]);
        MPI_Alltoallv(s_edges.data(), e_send_cnts.data(), e_sdispls.data(), MPI_INT,
                      r_edges.data(), e_recv_cnts.data(), e_rdispls.data(), MPI_INT, comm);
    }

    static AtomicData* construct_final_object(
        MPI_Comm comm, int rank, int size,
        std::vector<double> cell,
        int total_recv,
        const std::vector<int>& r_indices,
        const std::vector<int>& r_types,
        const std::vector<double>& r_pos,
        const std::vector<int>& r_edges,
        const std::vector<int>& type_norb
    ) {
        if (cell.empty()) cell.resize(9);
        MPI_Bcast(cell.data(), 9, MPI_DOUBLE, 0, comm);
        
        int my_final_n = total_recv;
        std::vector<int> final_counts(size);
        MPI_Allgather(&my_final_n, 1, MPI_INT, final_counts.data(), 1, MPI_INT, comm);
        
        int my_final_offset = 0;
        for(int i=0; i<rank; ++i) my_final_offset += final_counts[i];
        
        int total_atoms = 0;
        for(int c : final_counts) total_atoms += c;
        
        int my_final_n_edge = r_edges.size() / 5;
        int total_edges_global = 0;
        MPI_Allreduce(&my_final_n_edge, &total_edges_global, 1, MPI_INT, MPI_SUM, comm);
        
        std::vector<int> edge_indices(my_final_n_edge * 2);
        std::vector<int> edge_shifts(my_final_n_edge * 3);
        
        for(int k=0; k<my_final_n_edge; ++k) {
            edge_indices[2*k] = r_edges[5*k];
            edge_indices[2*k+1] = r_edges[5*k+1];
            edge_shifts[3*k] = r_edges[5*k+2];
            edge_shifts[3*k+1] = r_edges[5*k+3];
            edge_shifts[3*k+2] = r_edges[5*k+4];
        }
        
        return new AtomicData(my_final_n, total_atoms, my_final_offset, my_final_n_edge, total_edges_global,
                              r_indices.data(), r_types.data(), edge_indices.data(), type_norb.data(), edge_shifts.data(),
                              cell.data(), r_pos.data(), comm);
    }

};

}
}
