#include "../dos.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <complex>

using namespace rsatb::backend;
using namespace rsatb::calc;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing DOS/PDOS (Time Propagation)..." << std::endl;

    // 1. Setup Graph
    int n_blocks = size * 100; // Larger size for better stochastic average
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i}; 
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix H (Diagonal)
    // H = diag(0.5, -0.5, ...)
    BlockSpMat<std::complex<double>> H(&graph);
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> val = (gid % 2 == 0) ? 0.5 : -0.5;
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    // 3. Setup Chebyshev
    Chebyshev<std::complex<double>> cheby(H, -1.0, 1.0);
    
    // 4. Setup Random Vector
    DistVector<std::complex<double>> psi_in(&graph);
    int seed = rank * 1000;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 1);
    
    std::complex<double>* data = psi_in.local_data();
    for (int i = 0; i < psi_in.local_size; ++i) {
        data[i] = (dis(gen) == 0) ? 1.0 : -1.0;
    }

    // 5. Setup Mask for PDOS (Even indices -> 0.5)
    DistVector<std::complex<double>> mask(&graph);
    std::complex<double>* m_data = mask.local_data();
    for(int i=0; i<n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        m_data[i] = (gid % 2 == 0) ? 1.0 : 0.0;
    }

    // 6. Run corr_dos
    int nt = 10;
    double dt = M_PI;
    std::vector<std::complex<double>> cdos, pcdos;
    
    corr_dos(cheby, psi_in, nt, cdos, &mask, &pcdos, true);
    
    // 7. Verify Results
    if (rank == 0) {
        std::cout << "Verifying Correlation Functions..." << std::endl;
        
        // Total Trace: N * cos(0.5 * t)
        // Projected Trace (Even): (N/2) * e^{-i * 0.5 * t}
        
        double max_err_dos = 0.0;
        double max_err_pdos = 0.0;
        
        for (int it = 0; it <= nt; ++it) {
            double t = it * dt;
            
            // Expected DOS
            std::complex<double> exp_dos = static_cast<double>(n_blocks) * std::cos(0.5 * t);
            
            // Expected PDOS
            std::complex<double> exp_pdos = (static_cast<double>(n_blocks) / 2.0) * std::exp(std::complex<double>(0, -0.5 * t));
            
            double err_dos = std::abs(cdos[it] - exp_dos) / n_blocks;
            double err_pdos = std::abs(pcdos[it] - exp_pdos) / n_blocks;
            
            std::cout << "  t=" << t << ": DOS=" << cdos[it] << " (Exp: " << exp_dos << ") Err=" << err_dos << std::endl;
            std::cout << "        PDOS=" << pcdos[it] << " (Exp: " << exp_pdos << ") Err=" << err_pdos << std::endl;
            
            max_err_dos = std::max(max_err_dos, err_dos);
            max_err_pdos = std::max(max_err_pdos, err_pdos);
        }
        
        // Stochastic error is roughly 1/sqrt(N). N=200 -> ~7%.
        // We check if error is reasonable.
        if (max_err_dos < 0.2 && max_err_pdos < 0.2) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (Error too large)" << std::endl;
        }
    }

    // 8. Test Multivector Version
    if (rank == 0) std::cout << "\nTesting Multivector Version..." << std::endl;
    
    int num_vecs = 4;
    DistMultiVector<std::complex<double>> psi_mv(&graph, num_vecs);
    
    // Fill with random data
    for (int v = 0; v < num_vecs; ++v) {
        std::complex<double>* col_data = psi_mv.col_data(v);
        for (int i = 0; i < psi_mv.local_rows; ++i) {
            col_data[i] = (dis(gen) == 0) ? 1.0 : -1.0;
        }
    }
    
    std::vector<std::complex<double>> cdos_mv, pcdos_mv;
    corr_dos(cheby, psi_mv, nt, cdos_mv, &mask, &pcdos_mv, true);
    
    if (rank == 0) {
        std::cout << "Verifying Multivector Correlation Functions..." << std::endl;
        
        double max_err_dos = 0.0;
        double max_err_pdos = 0.0;
        
        for (int it = 0; it <= nt; ++it) {
            double t = it * dt;
            
            std::complex<double> exp_dos = static_cast<double>(n_blocks) * std::cos(0.5 * t);
            std::complex<double> exp_pdos = (static_cast<double>(n_blocks) / 2.0) * std::exp(std::complex<double>(0, -0.5 * t));
            
            double err_dos = std::abs(cdos_mv[it] - exp_dos) / n_blocks;
            double err_pdos = std::abs(pcdos_mv[it] - exp_pdos) / n_blocks;
            
            std::cout << "  t=" << t << ": DOS=" << cdos_mv[it] << " (Exp: " << exp_dos << ") Err=" << err_dos << std::endl;
            std::cout << "        PDOS=" << pcdos_mv[it] << " (Exp: " << exp_pdos << ") Err=" << err_pdos << std::endl;
            
            max_err_dos = std::max(max_err_dos, err_dos);
            max_err_pdos = std::max(max_err_pdos, err_pdos);
        }
        
        if (max_err_dos < 0.2 && max_err_pdos < 0.2) {
            std::cout << "MULTIVECTOR PASSED" << std::endl;
        } else {
            std::cout << "MULTIVECTOR FAILED (Error too large)" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
