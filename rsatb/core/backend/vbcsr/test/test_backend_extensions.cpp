#include "../block_csr.hpp"
#include "../dist_vector.hpp"
#include "../dist_multivector.hpp"
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

void test_dist_vector(DistGraph& graph, int rank) {
    if (rank == 0) std::cout << "Testing DistVector extensions..." << std::endl;

    DistVector<double> v1(&graph);
    DistVector<double> v2(&graph);
    
    // Fill v1, v2
    std::mt19937 gen(rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i=0; i<v1.local_size; ++i) v1[i] = dis(gen);
    for(int i=0; i<v2.local_size; ++i) v2[i] = dis(gen);
    
    // 1. Test duplicate
    DistVector<double> v3 = v1.duplicate();
    for(int i=0; i<v1.local_size; ++i) {
        if (v3[i] != v1[i]) {
            std::cerr << "DistVector::duplicate failed at " << i << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 2. Test axpby: v3 = 2*v1 + 3*v3 (v3 is copy of v1) -> v3 = 5*v1
    v3.axpby(2.0, v1, 3.0);
    for(int i=0; i<v1.local_size; ++i) {
        if (std::abs(v3[i] - 5.0 * v1[i]) > 1e-12) {
            std::cerr << "DistVector::axpby failed at " << i << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 3. Test pointwise_mult: v3 = v1 * v2
    v3 = v1.duplicate(); // Reset v3
    v3.pointwise_mult(v2);
    for(int i=0; i<v1.local_size; ++i) {
        if (std::abs(v3[i] - v1[i] * v2[i]) > 1e-12) {
            std::cerr << "DistVector::pointwise_mult failed at " << i << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    if (rank == 0) std::cout << "  DistVector extensions PASSED" << std::endl;
}

void test_dist_multivector(DistGraph& graph, int rank) {
    if (rank == 0) std::cout << "Testing DistMultiVector extensions..." << std::endl;
    
    int n_vecs = 2;
    DistMultiVector<double> mv1(&graph, n_vecs);
    DistMultiVector<double> mv2(&graph, n_vecs);
    
    std::mt19937 gen(rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for(size_t i=0; i<mv1.data.size(); ++i) mv1.data[i] = dis(gen);
    for(size_t i=0; i<mv2.data.size(); ++i) mv2.data[i] = dis(gen);
    
    // 1. Test scale
    DistMultiVector<double> mv3 = mv1; // Copy constructor (default)
    mv3.scale(2.0);
    for(size_t i=0; i<mv1.data.size(); ++i) {
        if (std::abs(mv3.data[i] - 2.0 * mv1.data[i]) > 1e-12) {
            std::cerr << "DistMultiVector::scale failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 2. Test axpy: mv3 = 2.0 * mv2 + mv3 (mv3 is 2*mv1) -> 2*mv2 + 2*mv1
    mv3.axpy(2.0, mv2);
    for(size_t i=0; i<mv1.data.size(); ++i) {
        if (std::abs(mv3.data[i] - 2.0 * (mv1.data[i] + mv2.data[i])) > 1e-12) {
            std::cerr << "DistMultiVector::axpy failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 3. Test axpby: mv3 = 2*mv1 + 3*mv2
    mv3 = mv2; // Reset to mv2
    mv3.axpby(2.0, mv1, 3.0);
    for(size_t i=0; i<mv1.data.size(); ++i) {
        if (std::abs(mv3.data[i] - (2.0 * mv1.data[i] + 3.0 * mv2.data[i])) > 1e-12) {
            std::cerr << "DistMultiVector::axpby failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 4. Test pointwise_mult
    mv3 = mv1;
    mv3.pointwise_mult(mv2);
    for(size_t i=0; i<mv1.data.size(); ++i) {
        if (std::abs(mv3.data[i] - mv1.data[i] * mv2.data[i]) > 1e-12) {
            std::cerr << "DistMultiVector::pointwise_mult failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 5. Test get_col
    DistVector<double> v_col = mv1.get_col(1);
    double* col1_ptr = mv1.col_data(1);
    for(int i=0; i<v_col.local_size; ++i) {
        if (v_col[i] != col1_ptr[i]) { // Only check owned
            std::cerr << "DistMultiVector::get_col failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 6. Test dot (bdot)
    std::vector<double> dots;
    mv1.dot(mv1, dots);
    // Manual check
    for(int v=0; v<n_vecs; ++v) {
        double local_dot = 0;
        double* col = mv1.col_data(v);
        for(int i=0; i<mv1.local_rows; ++i) local_dot += col[i] * col[i];
        
        double global_dot;
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (std::abs(dots[v] - global_dot) > 1e-12) {
            std::cerr << "DistMultiVector::dot failed for vec " << v << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    if (rank == 0) std::cout << "  DistMultiVector extensions PASSED" << std::endl;
}

void test_block_spmat(DistGraph& graph, int rank) {
    if (rank == 0) std::cout << "Testing BlockSpMat extensions..." << std::endl;
    
    BlockSpMat<double> mat(&graph);
    
    // Setup a diagonal matrix
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        int rows = graph.block_sizes[i]; // Assuming local index matches block_sizes index for simplicity in this test setup
        // Actually block_sizes is global? No, DistGraph::block_sizes is local (owned + ghost)
        // Wait, in test_block_csr.cpp, block_sizes was passed to construct_serial.
        // Let's assume we use the same setup as test_block_csr.cpp
        
        std::vector<double> block(rows * rows, 1.0); // Identity block
        mat.add_block(gid, gid, block.data(), rows, rows, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    mat.assemble();
    
    // 1. Test shift
    mat.shift(2.0); // Diagonals should become 3.0
    
    // Verify
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        bool found_diag = false;
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            if (graph.get_global_index(lid_c) == gid) {
                found_diag = true;
                size_t offset = mat.blk_ptr[k];
                int dim = mat.row_dims[i];
                for(int j=0; j<dim; ++j) {
                    if (std::abs(mat.val[offset + j*dim + j] - 3.0) > 1e-12) {
                         std::cerr << "BlockSpMat::shift failed" << std::endl;
                         MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
            }
        }
        if (!found_diag) {
             std::cerr << "BlockSpMat::shift failed (diag not found)" << std::endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 2. Test commutator_diagonal
    // C = [H, R] = H(R_j - R_i)
    // H is now 3*I. Commutator with diagonal should be 0!
    // Let's add off-diagonal blocks to test properly.
    // Add block (0, 1)
    if (rank == 0) { // Assuming rank 0 owns 0 and 1
        int r=0, c=1;
        int rows = graph.block_sizes[0]; // Assuming
        int cols = graph.block_sizes[1];
        std::vector<double> block(rows * cols, 1.0);
        mat.add_block(r, c, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    mat.assemble();
    
    DistVector<double> R(&graph);
    // Set R[i] = i
    double* r_ptr = R.local_data();
    for(int i=0; i<n_owned; ++i) {
        r_ptr[i] = (double)graph.owned_global_indices[i];
    }
    R.sync_ghosts();
    
    BlockSpMat<double> C(&graph);
    mat.commutator_diagonal(R, C);
    
    // Verify C_01 = H_01 * (R_1 - R_0) = 1.0 * (1 - 0) = 1.0
    // Verify C_00 = H_00 * (R_0 - R_0) = 0.0
    
    if (rank == 0) { // Check block (0, 1)
        // Find block (0, 1) in C
        // Need to find local row for 0.
        if (graph.global_to_local.count(0)) {
            int l_row = graph.global_to_local[0];
            int start = C.row_ptr[l_row];
            int end = C.row_ptr[l_row+1];
            for (int k = start; k < end; ++k) {
                int lid_c = C.col_ind[k];
                if (graph.get_global_index(lid_c) == 1) {
                    size_t offset = C.blk_ptr[k];
                    if (std::abs(C.val[offset] - 1.0) > 1e-12) {
                         std::cerr << "BlockSpMat::commutator_diagonal failed for off-diag" << std::endl;
                         MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                } else if (graph.get_global_index(lid_c) == 0) {
                     size_t offset = C.blk_ptr[k];
                     if (std::abs(C.val[offset]) > 1e-12) {
                         std::cerr << "BlockSpMat::commutator_diagonal failed for diag" << std::endl;
                         MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
            }
        }
    }
    
    if (rank == 0) std::cout << "  BlockSpMat extensions PASSED" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Setup Graph (Same as test_block_csr)
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

    test_dist_vector(graph, rank);
    test_dist_multivector(graph, rank);
    test_block_spmat(graph, rank);

    MPI_Finalize();
    return 0;
}
