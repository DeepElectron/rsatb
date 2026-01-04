#include "eigen.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <complex>
#include <mpi.h>

using namespace rsatb::backend;
using namespace rsatb::calc;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing PrimmeEigenSolver..." << std::endl;

    // 1. Setup Graph
    int n_blocks = size * 50; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i}; 
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix H (Diagonal: 1, 2, ..., N)
    BlockSpMat<double> H(&graph);
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        double val = static_cast<double>(gid + 1);
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    // 3. Setup Matrix S (Diagonal: 2, 2, ..., 2)
    BlockSpMat<double> S(&graph);
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        double val = 2.0;
        S.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    S.assemble();

    // 4. Test Standard Eigenvalue Problem (H x = lambda x)
    // Largest eigenvalues should be N, N-1, ...
    {
        if (rank == 0) std::cout << "\nTesting Standard Eigenvalue Problem (Largest)..." << std::endl;
        PrimmeEigenSolver<double> solver;
        std::vector<double> evals;
        std::vector<DistVector<double>> evecs;
        int num_evals = 5;
        
        solver.solve_largest(H, nullptr, nullptr, num_evals, 1e-6, evals, evecs, true);
        
        if (rank == 0) {
            for (int i = 0; i < num_evals; ++i) {
                double expected = static_cast<double>(n_blocks - i);
                std::cout << "Eval " << i << ": " << evals[i] << " (Expected: " << expected << ")" << std::endl;
                if (std::abs(evals[i] - expected) > 1e-4) {
                    std::cout << "FAILED" << std::endl;
                }
            }
        }
    }

    // 5. Test Generalized Eigenvalue Problem (H x = lambda S x)
    // Eigenvalues should be (gid+1)/2. Largest: N/2, (N-1)/2...
    {
        if (rank == 0) std::cout << "\nTesting Generalized Eigenvalue Problem (Largest)..." << std::endl;
        PrimmeEigenSolver<double> solver;
        std::vector<double> evals;
        std::vector<DistVector<double>> evecs;
        int num_evals = 5;
        
        solver.solve(H, &S, nullptr, num_evals, primme_largest, 1e-6, evals, evecs, true);
        
        if (rank == 0) {
            for (int i = 0; i < num_evals; ++i) {
                double expected = static_cast<double>(n_blocks - i) / 2.0;
                std::cout << "Eval " << i << ": " << evals[i] << " (Expected: " << expected << ")" << std::endl;
                if (std::abs(evals[i] - expected) > 1e-4) {
                    std::cout << "FAILED" << std::endl;
                }
            }
        }
    }

    // 6. Test Extremal (Smallest)
    {
        if (rank == 0) std::cout << "\nTesting Extremal (Smallest)..." << std::endl;
        PrimmeEigenSolver<double> solver;
        double emin, emax;
        
        solver.solve_extremal(H, nullptr, nullptr, 1e-6, emin, emax, true);
        
        if (rank == 0) {
            std::cout << "Emin: " << emin << " (Expected: 1.0)" << std::endl;
            std::cout << "Emax: " << emax << " (Expected: " << n_blocks << ")" << std::endl;
            if (std::abs(emin - 1.0) > 1e-4 || std::abs(emax - n_blocks) > 1e-4) {
                std::cout << "FAILED" << std::endl;
            } else {
                std::cout << "PASSED" << std::endl;
            }
        }
    }

    // 7. Test Composed Operator (Left Preconditioned: M*H*x = lambda*x)
    // H is diagonal (1..N). M is BlockJacobian (approx H^-1).
    // Since H is diagonal, M = H^-1 exactly.
    // So M*H = I. Eigenvalues should be 1.0.
    {
        if (rank == 0) std::cout << "\nTesting Composed Operator (Left: M*H)..." << std::endl;
        PrimmeEigenSolver<double> solver;
        std::vector<double> evals;
        std::vector<DistVector<double>> evecs;
        int num_evals = 5;
        
        BlockJacobianPreconditioner<double> M(H);
        
        solver.solve(H, nullptr, &M, num_evals, primme_largest, 1e-6, evals, evecs, true, PreconditionerType::Left);
        
        if (rank == 0) {
            for (int i = 0; i < num_evals; ++i) {
                double expected = 1.0;
                std::cout << "Eval " << i << ": " << evals[i] << " (Expected: " << expected << ")" << std::endl;
                if (std::abs(evals[i] - expected) > 1e-4) {
                    std::cout << "FAILED" << std::endl;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
