#include "fermi_dirac.hpp"
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

    if (rank == 0) std::cout << "Testing Fermi-Dirac KPM..." << std::endl;

    // 1. Setup Graph
    int n_blocks = size * 10; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i}; 
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix H (Diagonal with eigenvalues in [-1, 1])
    // H = diag(e_i)
    BlockSpMat<std::complex<double>> H(&graph);
    int n_owned = graph.owned_global_indices.size();
    std::vector<double> eigenvalues(n_owned);
    
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        // Spread eigenvalues from -0.9 to 0.9
        double val = -0.9 + 1.8 * (double)gid / (n_blocks - 1);
        eigenvalues[i] = val;
        std::complex<double> cval = val;
        H.add_block(gid, gid, &cval, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    // 3. Setup Chebyshev
    Chebyshev<std::complex<double>> cheby(H, -1.0, 1.0);
    
    // 4. Setup Input Vector (All ones)
    DistVector<std::complex<double>> psi_in(&graph);
    psi_in.set_constant(1.0);
    
    DistVector<std::complex<double>> psi_out(&graph);

    // 5. Run Fermi-Dirac KPM
    double temp = 300.0; // K
    double mu = 0.0; // Fermi level at 0
    double shift = 0.0;
    int n_kpm = 500; // Moments
    
    fermi_dirac_kpm(cheby, psi_in, psi_out, temp, mu, n_kpm);
    
    // 6. Verify Results
    // y_i = f(E_i) * x_i = f(E_i) * 1
    // f(E) = 1 / (exp((E-mu)/kT) + 1)
    
    double kb = 8.617333262e-5;
    double beta = 1.0 / (temp * kb);
    
    double max_err = 0.0;
    const std::complex<double>* out_data = psi_out.local_data();
    
    for (int i = 0; i < n_owned; ++i) {
        double E = eigenvalues[i];
        double exact = 1.0 / (std::exp((E - mu) * beta) + 1.0);
        double computed = out_data[i].real();
        
        double err = std::abs(computed - exact);
        max_err = std::max(max_err, err);
        
        // if (rank == 0 && i < 5) {
        //     std::cout << "E=" << E << " Exact=" << exact << " Computed=" << computed << " Err=" << err << std::endl;
        // }
    }
    
    double global_max_err;
    MPI_Allreduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Max Error: " << global_max_err << std::endl;
        if (global_max_err < 1e-2) { // KPM approximation error
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
