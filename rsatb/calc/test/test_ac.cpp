#include "../ac.hpp"
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

    if (rank == 0) std::cout << "Testing AC Conductivity..." << std::endl;

    // 1. Setup Graph
    int n_blocks = size * 10; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i}; 
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix H (Diagonal)
    BlockSpMat<std::complex<double>> H(&graph);
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> val = -0.5 + (double)gid / n_blocks;
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    // 3. Setup Velocity Matrices (Identity for simplicity)
    BlockSpMat<std::complex<double>> Vx(&graph), Vy(&graph), Vz(&graph);
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> val = 1.0;
        Vx.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        Vy.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        Vz.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    Vx.assemble();
    Vy.assemble();
    Vz.assemble();

    // 4. Setup Chebyshev
    Chebyshev<std::complex<double>> cheby(H, -1.0, 1.0);
    
    // 5. Setup Input Vector
    DistVector<std::complex<double>> psi_in(&graph);
    psi_in.set_constant(1.0);
    
    // 6. Run corr_ac
    int nt = 5;
    double temp = 300.0;
    double mu = 0.0;
    int n_kpm = 50;
    
    std::vector<std::complex<double>> cac((nt + 1) * 6);
    
    corr_ac(cheby, Vx, Vy, Vz, psi_in, nt, temp, mu, 1.0, 0.0, n_kpm, 1e-14, cac.data());
    
    if (rank == 0) {
        std::cout << "AC Correlation Results:" << std::endl;
        for (int t = 0; t <= nt; ++t) {
            std::cout << "t=" << t << ": ";
            for (int i = 0; i < 6; ++i) {
                std::cout << cac[t * 6 + i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "PASSED" << std::endl;
    }

    // 7. Test Multivector Version
    if (rank == 0) std::cout << "\nTesting Multivector Version..." << std::endl;
    
    int num_vecs = 4;
    DistMultiVector<std::complex<double>> psi_mv(&graph, num_vecs);
    psi_mv.set_constant(1.0);
    
    std::vector<std::complex<double>> cac_mv((nt + 1) * 6);
    corr_ac(cheby, Vx, Vy, Vz, psi_mv, nt, temp, mu, 1.0, 0.0, n_kpm, 1e-14, cac_mv.data());
    
    if (rank == 0) {
        std::cout << "AC Correlation Results (Multivector):" << std::endl;
        double max_diff = 0.0;
        for (int t = 0; t <= nt; ++t) {
            std::cout << "t=" << t << ": ";
            for (int i = 0; i < 6; ++i) {
                std::complex<double> val = cac_mv[t * 6 + i];
                std::complex<double> ref = cac[t * 6 + i];
                std::cout << val << " ";
                max_diff = std::max(max_diff, std::abs(val - ref));
            }
            std::cout << std::endl;
        }
        if (max_diff < 1e-10) {
            std::cout << "MULTIVECTOR PASSED (Max Diff: " << max_diff << ")" << std::endl;
        } else {
            std::cout << "MULTIVECTOR FAILED (Max Diff: " << max_diff << ")" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
