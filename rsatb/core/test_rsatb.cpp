#include "rsatb.hpp"
#include <iostream>
#include <mpi.h>

using namespace rsatb;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Dummy Atomic Data
    int atom_index[] = {0};
    int atom_type[] = {0};
    int edge_index[] = {0};
    int type_norb[] = {10}; // 10 orbitals
    int edge_shift_vec[] = {0, 0, 0};
    double cell[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double pos[] = {0, 0, 0};
    
    // Instantiate RSATB
    RSATB solver(1, 1, 0, 1, 1, atom_index, atom_type, edge_index, type_norb, edge_shift_vec, cell, pos);
    
    if (rank == 0) std::cout << "RSATB instantiated." << std::endl;
    
    // Manually setup H (Diagonal)
    solver.setup_K(0, 0, 0);
    solver.init_matrices_manual();
    
    // Fill H with diagonal 0..9
    auto& H = solver.H();
    auto& S = solver.S();
    
    // BlockSpMat doesn't expose easy element access, but we can use set_local_block if we know the structure.
    // Or we can hack it since we know it's 1 atom, 1 block.
    // The graph has 1 vertex (0).
    // H has 1 block (0,0).
    
    std::vector<std::complex<double>> h_block(100, 0.0);
    std::vector<std::complex<double>> s_block(100, 0.0);
    
    for(int i=0; i<10; ++i) {
        h_block[i*10+i] = static_cast<double>(i);
        s_block[i*10+i] = 1.0;
    }
    
    // We need to use set_local_block.
    // But RSATB doesn't expose set_local_block directly.
    // But H() returns BlockSpMat&.
    // BlockSpMat has set_local_block.
    // We need to know local row/col indices.
    // With 1 process, 1 atom, local row 0 is global row 0.
    
    H.set_local_block(0, 0, h_block.data());
    S.set_local_block(0, 0, s_block.data());
    
    H.assemble();
    S.assemble();
    
    if (rank == 0) std::cout << "Matrices initialized." << std::endl;
    
    // Test scale_shift_H
    solver.scale_shift_H();
    if (rank == 0) std::cout << "H scaled and shifted." << std::endl;
    
    // Test compute_eigenvalues
    // Note: H is scaled now, so eigenvalues will be scaled.
    // But compute_eigenvalues uses H directly.
    // We should probably test eigenvalues BEFORE scaling or account for scaling.
    // Let's re-init for eigenvalue test or check scaled values.
    // Scaled: (E - shift) / scale.
    // E_min ~ 0, E_max ~ 9.
    // shift ~ 4.5. scale ~ 4.5 + 4 = 8.5.
    // Expected: (0-4.5)/8.5 ~ -0.53, (9-4.5)/8.5 ~ 0.53.
    
    std::vector<double> evals;
    solver.compute_eigenvalues(1, 1e-6, evals, true);
    
    if (rank == 0) {
        std::cout << "Computed eigenvalue: " << evals[0] << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
