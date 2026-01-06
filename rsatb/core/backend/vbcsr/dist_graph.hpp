#ifndef RSATB_BACKEND_DIST_GRAPH_HPP
#define RSATB_BACKEND_DIST_GRAPH_HPP

#include <mpi.h>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace rsatb {
namespace backend {

class DistGraph {
public:
    MPI_Comm comm;
    int rank;
    int size;

    // Global indices of blocks owned by this process
    std::vector<int> owned_global_indices;
    
    // Global indices of remote blocks needed (ghosts)
    std::vector<int> ghost_global_indices;
    
    // Map global block index to local index
    // 0 to n_owned-1 are owned
    // n_owned to n_owned+n_ghost-1 are ghosts
    std::map<int, int> global_to_local;
    
    // Number of rows/cols (orbitals) for each local block (owned + ghost)
    std::vector<int> block_sizes;
    
    // Adjacency list for the block graph (local rows to column blocks)
    // Indices in adj_list are LOCAL indices
    std::vector<std::vector<int>> adj_list;

    // Communication pattern
    std::vector<int> send_counts;
    std::vector<int> recv_counts;
    std::vector<int> send_indices; // Local indices to send
    std::vector<int> recv_indices; // Local indices to receive (ghosts)
    std::vector<int> send_displs;
    std::vector<int> recv_displs;
    std::vector<int> send_ranks; // Ranks to send to (aligned with send_counts)
    std::vector<int> recv_ranks; // Ranks to receive from (aligned with recv_counts)
    
    // Global block displacements (rank r owns [block_displs[r], block_displs[r+1]))
    std::vector<int> block_displs;
    
    // Block offsets (prefix sum of block_sizes)
    std::vector<int> block_offsets;

    // Element-wise communication pattern (for scalar DistVector)
    std::vector<int> send_counts_scalar;
    std::vector<int> recv_counts_scalar;
    std::vector<int> send_displs_scalar;
    std::vector<int> recv_displs_scalar;
    
    // For neighbor collectives (if used)
    MPI_Comm neighbor_comm = MPI_COMM_NULL;

public:
    DistGraph(MPI_Comm c = MPI_COMM_WORLD) : comm(c) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    ~DistGraph() {
        if (neighbor_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&neighbor_comm);
        }
    }

    // Copy constructor (deep copy except for MPI_Comm handles that shouldn't be shared)
    DistGraph(const DistGraph& other) : comm(other.comm), rank(other.rank), size(other.size) {
        owned_global_indices = other.owned_global_indices;
        ghost_global_indices = other.ghost_global_indices;
        global_to_local = other.global_to_local;
        block_sizes = other.block_sizes;
        adj_list = other.adj_list;
        send_counts = other.send_counts;
        recv_counts = other.recv_counts;
        send_indices = other.send_indices;
        recv_indices = other.recv_indices;
        send_displs = other.send_displs;
        recv_displs = other.recv_displs;
        send_ranks = other.send_ranks;
        recv_ranks = other.recv_ranks;
        block_displs = other.block_displs;
        block_offsets = other.block_offsets;
        send_counts_scalar = other.send_counts_scalar;
        recv_counts_scalar = other.recv_counts_scalar;
        send_displs_scalar = other.send_displs_scalar;
        recv_displs_scalar = other.recv_displs_scalar;
        // neighbor_comm is NOT copied to avoid double-free or sharing managed handle
        neighbor_comm = MPI_COMM_NULL;
    }

    // Duplicate method for convenience (returns a pointer to a new deep copy)
    DistGraph* duplicate() const {
        return new DistGraph(*this);
    }

    // Construct from serial data (rank 0 distributes)
    // Simple 1D block partition for now
    void construct_serial(int n_global_blocks, const std::vector<int>& global_block_sizes, const std::vector<std::vector<int>>& global_adj) {
        // 1. Partitioning
        // Simple 1D partitioning: divide blocks equally among ranks
        int blocks_per_rank = n_global_blocks / size;
        int remainder = n_global_blocks % size;

        std::vector<int> count(size);
        std::vector<int> displ(size + 1, 0);

        for (int i = 0; i < size; ++i) {
            count[i] = blocks_per_rank + (i < remainder ? 1 : 0);
            displ[i + 1] = displ[i] + count[i];
        }
        block_displs = displ;

        // 2. Determine owned indices
        int my_start = displ[rank];
        int my_end = displ[rank + 1];
        int n_owned = my_end - my_start;

        owned_global_indices.resize(n_owned);
        std::iota(owned_global_indices.begin(), owned_global_indices.end(), my_start);

        // 3. Build local graph and identify ghosts
        // This requires scattering the global adjacency list. 
        // For simplicity in this serial construction, we assume all ranks have the full graph or rank 0 broadcasts.
        // But to be scalable, we should distribute. 
        // Here, assuming global_adj is available on rank 0, we scatter it.
        
        // ... Implementation detail: For now, let's assume all ranks have global_adj if passed, 
        // or we implement a proper scatter. 
        // Given the prompt "construct_serial", it implies rank 0 might have it.
        // Let's implement a broadcast of the full graph for simplicity if n_global_blocks is small, 
        // OR better, scatter the relevant parts.
        
        // Let's assume global_adj is valid on all ranks for "serial" construction convenience in tests,
        // or implement the scatter. 
        // To be safe and scalable-ish, let's assume only rank 0 has valid data if rank != 0.
        
        // Broadcast partition info (count, displ) is implicit since we computed it deterministically.
        
        // Scatter block sizes
        std::vector<int> my_block_sizes(n_owned);
        // If rank 0 has data, scatter.
        // Note: MPI_Scatterv requires sendcounts and displs arrays.
        
        if (rank == 0) {
             // Check sizes
             if (global_block_sizes.size() != n_global_blocks) throw std::runtime_error("Invalid block sizes size");
        }

        // We need to distribute global_block_sizes. 
        // Since we need ghost block sizes too, it might be better to broadcast all block sizes if memory allows,
        // or fetch on demand. For "Block Sparse", usually N is not huge (N_blocks), but N*BlockSize is.
        // Let's assume we can broadcast global_block_sizes for now.
        std::vector<int> all_block_sizes;
        if (rank == 0) all_block_sizes = global_block_sizes;
        all_block_sizes.resize(n_global_blocks);
        MPI_Bcast(all_block_sizes.data(), n_global_blocks, MPI_INT, 0, comm);

        // Scatter adjacency? 
        // Adjacency is a vector of vectors. Hard to scatter directly.
        // We can flatten and scatter, or just broadcast if not too large.
        // For a true distributed graph, we should use `construct_distributed`.
        // `construct_serial` is mostly for testing or small problems. 
        // Let's broadcast the adjacency list for simplicity in this method.
        
        // Flatten global_adj
        std::vector<int> flat_adj;
        std::vector<int> adj_offsets;
        if (rank == 0) {
            adj_offsets.push_back(0);
            for (const auto& neighbors : global_adj) {
                flat_adj.insert(flat_adj.end(), neighbors.begin(), neighbors.end());
                adj_offsets.push_back(flat_adj.size());
            }
        }
        
        // Broadcast sizes
        int total_edges = flat_adj.size();
        MPI_Bcast(&total_edges, 1, MPI_INT, 0, comm);
        flat_adj.resize(total_edges);
        adj_offsets.resize(n_global_blocks + 1);
        
        MPI_Bcast(flat_adj.data(), total_edges, MPI_INT, 0, comm);
        MPI_Bcast(adj_offsets.data(), n_global_blocks + 1, MPI_INT, 0, comm);
        
        // 3. Build local graph and identify ghosts
        global_to_local.clear();
        block_sizes.clear();
        adj_list.clear();
        ghost_global_indices.clear();

        // 3.1 Add owned blocks
        for (int i = 0; i < n_owned; ++i) {
            int gid = owned_global_indices[i];
            global_to_local[gid] = i;
            block_sizes.push_back(all_block_sizes[gid]);
        }

        // 3.2 Collect ghosts
        std::vector<int> ghost_candidates;
        for (int i = 0; i < n_owned; ++i) {
            int gid = owned_global_indices[i];
            int start = adj_offsets[gid];
            int end = adj_offsets[gid+1];
            
            for (int k = start; k < end; ++k) {
                int neighbor_gid = flat_adj[k];
                if (global_to_local.find(neighbor_gid) == global_to_local.end()) {
                    ghost_candidates.push_back(neighbor_gid);
                }
            }
        }
        
        // Sort and unique ghosts
        std::sort(ghost_candidates.begin(), ghost_candidates.end());
        ghost_candidates.erase(std::unique(ghost_candidates.begin(), ghost_candidates.end()), ghost_candidates.end());
        
        // Sort ghosts by owner
        std::sort(ghost_candidates.begin(), ghost_candidates.end(), [this](int a, int b) {
            int owner_a = this->find_owner(a);
            int owner_b = this->find_owner(b);
            if (owner_a != owner_b) return owner_a < owner_b;
            return a < b;
        });
        
        // Assign local IDs to ghosts
        for (int gid : ghost_candidates) {
            int local_id = global_to_local.size();
            global_to_local[gid] = local_id;
            ghost_global_indices.push_back(gid);
            block_sizes.push_back(all_block_sizes[gid]);
        }
        
        // 3.3 Build adjacency list
        adj_list.resize(n_owned);
        for (int i = 0; i < n_owned; ++i) {
            int gid = owned_global_indices[i];
            int start = adj_offsets[gid];
            int end = adj_offsets[gid+1];
            
            for (int k = start; k < end; ++k) {
                int neighbor_gid = flat_adj[k];
                adj_list[i].push_back(global_to_local[neighbor_gid]);
            }
        }
        
        // 4. Build communication pattern
        build_comm_pattern(displ);
    }

    // Construct from distributed data
    void construct_distributed(const std::vector<int>& my_owned_indices, 
                             const std::vector<int>& my_block_sizes, 
                             const std::vector<std::vector<int>>& my_adj) {
        owned_global_indices = my_owned_indices;
        int n_owned = owned_global_indices.size();
        
        // We need to know the global partition to locate owners of ghosts.
        int my_count = n_owned;
        std::vector<int> all_counts(size);
        MPI_Allgather(&my_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);
        
        std::vector<int> displ(size + 1, 0);
        for (int i = 0; i < size; ++i) {
            displ[i+1] = displ[i] + all_counts[i];
        }
        block_displs = displ;
        
        // Let's build the local graph
        global_to_local.clear();
        block_sizes.clear();
        adj_list.clear();
        ghost_global_indices.clear();

        // Add owned
        for (int i = 0; i < n_owned; ++i) {
            int gid = owned_global_indices[i];
            global_to_local[gid] = i;
            block_sizes.push_back(my_block_sizes[i]);
        }
        
        // Collect ghosts
        std::vector<int> ghost_candidates;
        for (int i = 0; i < n_owned; ++i) {
            for (int neighbor_gid : my_adj[i]) {
                if (global_to_local.find(neighbor_gid) == global_to_local.end()) {
                    ghost_candidates.push_back(neighbor_gid);
                }
            }
        }
        
        // Sort and unique
        std::sort(ghost_candidates.begin(), ghost_candidates.end());
        ghost_candidates.erase(std::unique(ghost_candidates.begin(), ghost_candidates.end()), ghost_candidates.end());
        
        // Sort by owner
        std::sort(ghost_candidates.begin(), ghost_candidates.end(), [this, &displ](int a, int b) {
            int owner_a = this->find_owner(a, displ);
            int owner_b = this->find_owner(b, displ);
            if (owner_a != owner_b) return owner_a < owner_b;
            return a < b;
        });
        
        // Assign local IDs
        for (int gid : ghost_candidates) {
            int local_id = global_to_local.size();
            global_to_local[gid] = local_id;
            ghost_global_indices.push_back(gid);
            // We don't know ghost block size yet! Need to fetch.
            block_sizes.push_back(0); // Placeholder
        }
        
        // Build adjacency list
        adj_list.resize(n_owned);
        for (int i = 0; i < n_owned; ++i) {
            for (int neighbor_gid : my_adj[i]) {
                adj_list[i].push_back(global_to_local[neighbor_gid]);
            }
        }
        
        // Fetch ghost block sizes
        fetch_ghost_block_sizes(displ);
        
        // Build comm pattern
        build_comm_pattern(displ);
    }

    void get_matrix_structure(std::vector<int>& row_ptr, std::vector<int>& col_ind) {
        int n_owned = owned_global_indices.size();
        row_ptr.resize(n_owned + 1);
        row_ptr[0] = 0;
        col_ind.clear();
        
        for (int i = 0; i < n_owned; ++i) {
            // Sort column indices for CSR
            std::vector<int> cols = adj_list[i];
            std::sort(cols.begin(), cols.end());
            
            col_ind.insert(col_ind.end(), cols.begin(), cols.end());
            row_ptr[i+1] = col_ind.size();
        }
    }
    
    void get_vector_structure(int& local_size, int& ghost_size) {
        local_size = 0;
        int n_owned = owned_global_indices.size();
        for(int i=0; i<n_owned; ++i) local_size += block_sizes[i];
        
        ghost_size = 0;
        for(size_t i=n_owned; i<block_sizes.size(); ++i) ghost_size += block_sizes[i];
    }

    // Find owner of a global block index using stored displacements
    int find_owner(int gid) const {
        if (block_displs.empty()) throw std::runtime_error("Graph not constructed");
        auto it = std::upper_bound(block_displs.begin(), block_displs.end(), gid);
        return std::distance(block_displs.begin(), it) - 1;
    }

    // Internal helper
    int find_owner(int gid, const std::vector<int>& displ) const {
        auto it = std::upper_bound(displ.begin(), displ.end(), gid); // find the first element that is greater than gid
        return std::distance(displ.begin(), it) - 1; // return the index of the first element that is greater than gid
    }

    // Get global index from local index
    int get_global_index(int local_index) const {
        int n_owned = owned_global_indices.size();
        if (local_index < n_owned) {
            return owned_global_indices[local_index];
        } else {
            if (local_index - n_owned < (int)ghost_global_indices.size()) {
                return ghost_global_indices[local_index - n_owned];
            } else {
                throw std::out_of_range("Local index out of range");
            }
        }
    }

private:
    void build_comm_pattern(const std::vector<int>& displ) {
        // 0. Compute block_offsets
        block_offsets.resize(block_sizes.size() + 1);
        block_offsets[0] = 0;
        for (size_t i = 0; i < block_sizes.size(); ++i) {
            block_offsets[i+1] = block_offsets[i] + block_sizes[i];
        }

        // 1. Identify who owns my ghosts (Recv from them)
        int n_ghosts = ghost_global_indices.size();
        std::map<int, std::vector<int>> ghosts_by_rank;
        
        for (int i = 0; i < n_ghosts; ++i) {
            int gid = ghost_global_indices[i];
            int owner = find_owner(gid, displ);
            ghosts_by_rank[owner].push_back(gid);
        }
        
        // 2. Tell owners I need these blocks (They will send to me)
        // We use MPI_Alltoall to exchange counts of requests
        std::vector<int> req_counts(size, 0);
        for (auto& kv : ghosts_by_rank) {
            req_counts[kv.first] = kv.second.size();
        }
        
        std::vector<int> incoming_req_counts(size);
        MPI_Alltoall(req_counts.data(), 1, MPI_INT, incoming_req_counts.data(), 1, MPI_INT, comm);
            // tasking the rank-th element and send to rank-th process
        
        // 3. Exchange the actual requests (which GIDs they want)
        std::vector<int> req_sdispls(size + 1, 0);
        std::vector<int> req_rdispls(size + 1, 0);
        for (int i = 0; i < size; ++i) {
            req_sdispls[i+1] = req_sdispls[i] + req_counts[i];
            req_rdispls[i+1] = req_rdispls[i] + incoming_req_counts[i];
        }
        
        std::vector<int> req_send_buf(req_sdispls[size]);
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            if (ghosts_by_rank.count(i)) {
                std::copy(ghosts_by_rank[i].begin(), ghosts_by_rank[i].end(), req_send_buf.begin() + offset);
            }
            offset += req_counts[i];
        }
        
        std::vector<int> req_recv_buf(req_rdispls[size]);
        MPI_Alltoallv(req_send_buf.data(), req_counts.data(), req_sdispls.data(), MPI_INT,
                      req_recv_buf.data(), incoming_req_counts.data(), req_rdispls.data(), MPI_INT, comm);
                      
        // 4. Build Send Lists (What I send to others)
        // req_recv_buf contains GIDs that others want from me
        send_counts = incoming_req_counts;
        recv_counts = req_counts;
        
        send_indices.clear();
        for (int i = 0; i < size; ++i) {
            int count = incoming_req_counts[i];
            int start = req_rdispls[i];
            for (int k = 0; k < count; ++k) {
                int gid = req_recv_buf[start + k];
                // Map GID to local index
                if (global_to_local.find(gid) == global_to_local.end()) {
                    throw std::runtime_error("Requested GID not owned by this rank");
                }
                send_indices.push_back(global_to_local[gid]);
            }
        }
        
        // 5. Build Recv Lists (Where to put what I receive)
        // I receive ghosts. The order corresponds to how I requested them (by rank, then by order in ghosts_by_rank)
        recv_indices.clear();
        for (int i = 0; i < size; ++i) {
            if (ghosts_by_rank.count(i)) {
                for (int gid : ghosts_by_rank[i]) {
                    recv_indices.push_back(global_to_local[gid]);
                }
            }
        }
        
        // 6. Setup displacements for Alltoallv
        send_displs.resize(size + 1, 0);
        recv_displs.resize(size + 1, 0);
        for (int i = 0; i < size; ++i) {
            send_displs[i+1] = send_displs[i] + send_counts[i];
            recv_displs[i+1] = recv_displs[i] + recv_counts[i];
        }
        
        // 7. Store ranks involved (optimization for Neighbor collectives if we upgrade)
        send_ranks.clear();
        recv_ranks.clear();
        for(int i=0; i<size; ++i) {
            if(send_counts[i] > 0) send_ranks.push_back(i);
            if(recv_counts[i] > 0) recv_ranks.push_back(i);
        }

        // 8. Compute scalar communication pattern
        send_counts_scalar.assign(size, 0);
        recv_counts_scalar.assign(size, 0);
        
        int current_idx = 0;
        for (int r = 0; r < size; ++r) {
            int n_blocks = send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = send_indices[current_idx++];
                send_counts_scalar[r] += block_sizes[blk_idx];
            }
        }
        
        current_idx = 0;
        for (int r = 0; r < size; ++r) {
            int n_blocks = recv_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = recv_indices[current_idx++];
                recv_counts_scalar[r] += block_sizes[blk_idx];
            }
        }
        
        send_displs_scalar.resize(size + 1, 0);
        recv_displs_scalar.resize(size + 1, 0);
        for (int i = 0; i < size; ++i) {
            send_displs_scalar[i+1] = send_displs_scalar[i] + send_counts_scalar[i];
            recv_displs_scalar[i+1] = recv_displs_scalar[i] + recv_counts_scalar[i];
        }
    }
    
    void fetch_ghost_block_sizes(const std::vector<int>& displ) {
        // std::cerr << "fetch_ghost_block_sizes called. n_ghosts: " << ghost_global_indices.size() << std::endl;
        // Similar to build_comm_pattern, but we exchange block sizes
        // 1. Identify owners of ghosts
        int n_ghosts = ghost_global_indices.size();
        std::map<int, std::vector<int>> ghosts_by_rank;
        for (int i = 0; i < n_ghosts; ++i) {
            int gid = ghost_global_indices[i];
            int owner = find_owner(gid, displ);
            ghosts_by_rank[owner].push_back(gid);
        }
        
        // 2. Exchange requests (counts)
        std::vector<int> req_counts(size, 0);
        for (auto& kv : ghosts_by_rank) req_counts[kv.first] = kv.second.size();
        
        std::vector<int> incoming_req_counts(size);
        MPI_Alltoall(req_counts.data(), 1, MPI_INT, incoming_req_counts.data(), 1, MPI_INT, comm);
        
        // 3. Exchange GIDs
        std::vector<int> req_sdispls(size + 1, 0), req_rdispls(size + 1, 0);
        for (int i = 0; i < size; ++i) {
            req_sdispls[i+1] = req_sdispls[i] + req_counts[i];
            req_rdispls[i+1] = req_rdispls[i] + incoming_req_counts[i];
        }
        
        std::vector<int> req_send_buf(req_sdispls[size]);
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            if (ghosts_by_rank.count(i)) {
                std::copy(ghosts_by_rank[i].begin(), ghosts_by_rank[i].end(), req_send_buf.begin() + offset);
            }
            offset += req_counts[i];
        }
        
        std::vector<int> req_recv_buf(req_rdispls[size]);
        MPI_Alltoallv(req_send_buf.data(), req_counts.data(), req_sdispls.data(), MPI_INT,
                      req_recv_buf.data(), incoming_req_counts.data(), req_rdispls.data(), MPI_INT, comm);
                      
        // 4. Prepare Block Sizes to send back
        std::vector<int> resp_send_buf(req_rdispls[size]);
        for (int i = 0; i < req_rdispls[size]; ++i) {
            int gid = req_recv_buf[i];
            if (global_to_local.find(gid) == global_to_local.end()) throw std::runtime_error("GID not found");
            int lid = global_to_local[gid];
            resp_send_buf[i] = block_sizes[lid];
        }
        
        // 5. Receive Block Sizes
        std::vector<int> resp_recv_buf(req_sdispls[size]);
        // Note: Send/Recv counts are swapped compared to request phase
        MPI_Alltoallv(resp_send_buf.data(), incoming_req_counts.data(), req_rdispls.data(), MPI_INT,
                      resp_recv_buf.data(), req_counts.data(), req_sdispls.data(), MPI_INT, comm);
                      
        // 6. Fill ghost block sizes
        offset = 0;
        for (int i = 0; i < size; ++i) {
            if (ghosts_by_rank.count(i)) {
                for (size_t k = 0; k < ghosts_by_rank[i].size(); ++k) {
                    int gid = ghosts_by_rank[i][k];
                    int recv_size = resp_recv_buf[offset + k];
                    int lid = global_to_local[gid];
                    // std::cerr << "[DEBUG] Rank " << rank << " filling ghost GID " << gid << " (lid=" << lid << ") with size " << recv_size << " from resp_recv_buf[" << (offset + k) << "]" << std::endl;
                    block_sizes[lid] = recv_size;
                }
            }
            offset += req_counts[i];
        }
        
        // Check for any remaining zeros in ghost block sizes
        int n_owned = owned_global_indices.size();
        for (size_t i = n_owned; i < block_sizes.size(); ++i) {
            if (block_sizes[i] == 0) {
                std::cerr << "[WARNING] Rank " << rank << " ghost block_sizes[" << i << "] is still ZERO! (GID=" << ghost_global_indices[i - n_owned] << ")" << std::endl;
            }
        }
    }
};

} // namespace backend
} // namespace rsatb

#endif
