#include "../solver.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

using namespace rsatb::calc;

// Helper to fill vector with random numbers
void fill_random(std::vector<double>& v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto& val : v) val = dis(gen);
}

void test_diagonal_solver(int rank, int size) {
    if (rank == 0) std::cout << "Testing Diagonal Solver..." << std::endl;

    // 1. Setup Graph (Diagonal matrix for simplicity)
    int n_blocks = size * 2;
    std::vector<int> block_sizes(n_blocks, 2);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i}; // Diagonal only
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix A (SPD)
    BlockSpMat<double> A(&graph);
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        int rows = 2;
        // Make it SPD: 2*I
        std::vector<double> block = {2.0, 0.0, 0.0, 2.0};
        A.add_block(gid, gid, block.data(), rows, rows, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    A.assemble();

    // 3. Setup RHS b (Random)
    DistVector<double> b(&graph);
    std::mt19937 gen(rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i=0; i<b.local_size; ++i) b[i] = dis(gen);
    
    // 4. Setup Initial Guess x (Zero)
    DistVector<double> x(&graph);
    x.set_constant(0.0);
    
    // 5. Solve Ax = b
    LinearSolver<double> solver(1e-8, 100, false); // Quiet
    solver.solve(A, b, x);
    
    // 6. Verify: x should be b / 2
    double max_err = 0.0;
    for(int i=0; i<b.local_size; ++i) {
        double err = std::abs(x[i] - b[i] / 2.0);
        max_err = std::max(max_err, err);
    }
    
    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "  Diagonal CG Max Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

void test_laplacian_solver(int rank, int size) {
    if (rank == 0) std::cout << "Testing Laplacian Solver (1D)..." << std::endl;

    // 1. Setup Graph (1D Grid)
    int n_blocks = size * 10; // 10 points per rank
    std::vector<int> block_sizes(n_blocks, 1); // 1x1 blocks
    std::vector<std::vector<int>> global_adj(n_blocks);
    
    for(int i=0; i<n_blocks; ++i) {
        global_adj[i].push_back(i);
        if (i > 0) global_adj[i].push_back(i-1);
        if (i < n_blocks - 1) global_adj[i].push_back(i+1);
    }
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix A (1D Laplacian)
    // A = 2*I - 1*off_diag
    BlockSpMat<double> A(&graph);
    int n_owned = graph.owned_global_indices.size();
    
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        double val = 2.0;
        A.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        val = -1.0;
        if (gid > 0) {
            A.add_block(gid, gid-1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        if (gid < n_blocks - 1) {
            A.add_block(gid, gid+1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    A.assemble();

    // 3. Setup RHS b (Random)
    DistVector<double> b(&graph);
    std::mt19937 gen(rank + 100);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i=0; i<b.local_size; ++i) b[i] = dis(gen);
    
    // 4. Setup Initial Guess x (Zero)
    DistVector<double> x(&graph);
    x.set_constant(0.0);
    
    // 5. Solve Ax = b
    LinearSolver<double> solver(1e-8, 1000, true); // Verbose
    solver.solve(A, b, x);
    
    // 6. Verify: Check residual ||Ax - b||
    DistVector<double> Ax(&graph);
    A.mult_optimized(x, Ax);
    Ax.axpby(1.0, b, -1.0); // Ax = Ax - b (Residual)
    
    double r_norm_sq = Ax.dot(Ax);
    double r_norm = std::sqrt(r_norm_sq);
    
    if (rank == 0) {
        std::cout << "  Laplacian Residual Norm: " << r_norm << std::endl;
        if (r_norm > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

#include "../preconditioner.hpp"

// Manual preconditioners removed in favor of rsatb::calc::BlockJacobianPreconditioner
// and rsatb::calc::ISQRTMPreconditioner

void test_preconditioned_solver(int rank, int size) {
    if (rank == 0) std::cout << "Testing Preconditioned Solver (Symmetric Jacobi)..." << std::endl;

    // 1. Setup Graph (1D Grid)
    int n_blocks = size * 10; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    
    for(int i=0; i<n_blocks; ++i) {
        global_adj[i].push_back(i);
        if (i > 0) global_adj[i].push_back(i-1);
        if (i < n_blocks - 1) global_adj[i].push_back(i+1);
    }
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix A (1D Laplacian)
    BlockSpMat<double> A(&graph);
    int n_owned = graph.owned_global_indices.size();
    
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        double val = 10.0 * 0.25;
        A.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        // val = -1.0 * 0.1;
        // if (gid > 0) A.add_block(gid, gid-1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        // if (gid < n_blocks - 1) A.add_block(gid, gid+1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    A.assemble();

    // 3. Setup RHS b (Random)
    DistVector<double> b(&graph);
    std::mt19937 gen(rank + 200);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i=0; i<b.local_size; ++i) b[i] = dis(gen);
    
    // 4. Setup Preconditioner (ISQRTM -> P = A^-1/2)
    // Scale factor 0.5 for stability
    ISQRTMPreconditioner<double> P(A, 50, 1e-5, 0.2, /*verbose=*/true);
    
    // 5. Solve P A P y = P b, x = P y
    DistVector<double> x(&graph);
    x.set_constant(0.0);
    
    LinearSolver<double> solver(1e-8, 1000, true);
    solver.solve(A, b, x, P, PreconditionerType::Symmetric);
    
    // 6. Verify: Check residual ||Ax - b||
    DistVector<double> Ax(&graph);
    A.mult_optimized(x, Ax);
    Ax.axpby(1.0, b, -1.0);
    
    double r_norm_sq = Ax.dot(Ax);
    double r_norm = std::sqrt(r_norm_sq);
    
    if (rank == 0) {
        std::cout << "  Preconditioned Residual Norm: " << r_norm << std::endl;
        if (r_norm > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

void test_left_preconditioned_solver(int rank, int size) {
    if (rank == 0) std::cout << "Testing Left Preconditioned Solver (PCG)..." << std::endl;

    // 1. Setup Graph (1D Grid)
    int n_blocks = size * 10; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    
    for(int i=0; i<n_blocks; ++i) {
        global_adj[i].push_back(i);
        if (i > 0) global_adj[i].push_back(i-1);
        if (i < n_blocks - 1) global_adj[i].push_back(i+1);
    }
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix A (1D Laplacian)
    BlockSpMat<double> A(&graph);
    int n_owned = graph.owned_global_indices.size();
    
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        
        // Variable diagonal to ensure P and A don't commute
        // Use deterministic randomness based on gid
        double diag_val = 2.0 + 0.5 * std::sin(gid); 
        
        A.add_block(gid, gid, &diag_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        double val = -1.0;
        if (gid > 0) A.add_block(gid, gid-1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        if (gid < n_blocks - 1) A.add_block(gid, gid+1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    A.assemble();

    // 3. Setup RHS b (Random)
    DistVector<double> b(&graph);
    std::mt19937 gen(rank + 300);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i=0; i<b.local_size; ++i) b[i] = dis(gen);
    
    // 4. Setup Preconditioner (Block Jacobian -> P = D^-1)
    BlockJacobianPreconditioner<double> P(A);
    
    // 5. Solve Ax = b with PCG
    DistVector<double> x(&graph);
    x.set_constant(0.0);
    
    LinearSolver<double> solver(1e-8, 1000, true);
    solver.solve(A, b, x, P, PreconditionerType::Left);
    
    // 6. Verify: Check residual ||Ax - b||
    DistVector<double> Ax(&graph);
    A.mult_optimized(x, Ax);
    Ax.axpby(1.0, b, -1.0);
    
    double r_norm_sq = Ax.dot(Ax);
    double r_norm = std::sqrt(r_norm_sq);
    
    if (rank == 0) {
        std::cout << "  Left Preconditioned Residual Norm: " << r_norm << std::endl;
        if (r_norm > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

void test_batched_preconditioned_solver(int rank, int size) {
    if (rank == 0) std::cout << "Testing Batched Preconditioned Solver (PCG)..." << std::endl;

    // 1. Setup Graph (1D Grid)
    int n_blocks = size * 10; 
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    
    for(int i=0; i<n_blocks; ++i) {
        global_adj[i].push_back(i);
        if (i > 0) global_adj[i].push_back(i-1);
        if (i < n_blocks - 1) global_adj[i].push_back(i+1);
    }
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup Matrix A (1D Laplacian with Variable Diagonal)
    BlockSpMat<double> A(&graph);
    int n_owned = graph.owned_global_indices.size();
    
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        
        // Variable diagonal
        double diag_val = 2.0 + 0.5 * std::sin(gid); 
        
        A.add_block(gid, gid, &diag_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        double val = -1.0;
        if (gid > 0) A.add_block(gid, gid-1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        if (gid < n_blocks - 1) A.add_block(gid, gid+1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    A.assemble();

    // 3. Setup RHS B (Random, 3 vectors)
    int num_vecs = 3;
    DistMultiVector<double> B(&graph, num_vecs);
    std::mt19937 gen(rank + 400);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    // Fill only the num_vectors lanes of each row: the row-major buffer has
    // ld >= num_vectors and the trailing padding lanes must stay zero.
    for (int i = 0; i < B.local_rows + B.ghost_rows; ++i) {
        double* row = B.row_data(i);
        for (int v = 0; v < num_vecs; ++v) row[v] = dis(gen);
    }
    
    // 4. Setup Preconditioner (Block Jacobian -> P = D^-1)
    BlockJacobianPreconditioner<double> P(A);
    
    // 5. Solve AX = B with Batched PCG
    DistMultiVector<double> X(&graph, num_vecs);
    std::fill(X.data.begin(), X.data.end(), 0.0);
    
    LinearSolver<double> solver(1e-8, 1000, true);
    solver.solve(A, B, X, P, PreconditionerType::Left);
    
    // 6. Verify: Check max residual ||AX - B||
    DistMultiVector<double> AX(&graph, num_vecs);
    A.mult_dense(X, AX);
    AX.axpby(1.0, B, -1.0); // AX = AX - B
    
    std::vector<double> r_norm_sq;
    AX.dot(AX, r_norm_sq);
    
    double max_err = 0.0;
    for(auto v : r_norm_sq) max_err = std::max(max_err, std::sqrt(v));
    
    if (rank == 0) {
        std::cout << "  Batched PCG Max Residual: " << max_err << std::endl;
        if (max_err > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    test_diagonal_solver(rank, size);
    test_laplacian_solver(rank, size);
    test_preconditioned_solver(rank, size);
    test_left_preconditioned_solver(rank, size);
    test_batched_preconditioned_solver(rank, size);

    MPI_Finalize();
    return 0;
}
