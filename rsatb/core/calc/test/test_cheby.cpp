#include "../cheby.hpp"
#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include <cmath>

using namespace rsatb::backend;
using namespace rsatb::calc;

void test_timeprop_coeff() {
    std::cout << "Testing timeprop_coeff..." << std::endl;
    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(1.0, 1e-10, true, coeffs);
    std::cout << "  Generated " << coeffs.size() << " coefficients." << std::endl;
    assert(coeffs.size() > 0);
    // J0(1.0) is approx 0.765
    std::cout << "  coeff[0] (J0): " << coeffs[0] << " (expected ~0.765)" << std::endl;
    std::cout << "  PASSED" << std::endl;
}

void test_cheby_propagate(int rank, int size) {
    if (rank == 0) std::cout << "Testing Chebyshev Propagation (Normal Mode)..." << std::endl;

    // 1. Setup Graph (Diagonal)
    int n_blocks = 4;
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i};
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup H (Diagonal: 0, 1, 2, 3)
    BlockSpMat<std::complex<double>> H(&graph);
    for (int i = 0; i < (int)graph.owned_global_indices.size(); ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> val = static_cast<double>(gid);
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    // 3. Chebyshev Setup
    // Scale H to [-1, 1]. Range [0, 3] -> center=1.5, half_width=1.5
    Chebyshev<std::complex<double>> cheby(H, 0.0, 3.0);

    // 4. Input Vector v = [1, 1, 1, 1]
    DistVector<std::complex<double>> v_in(&graph);
    v_in.set_constant(1.0);
    DistVector<std::complex<double>> v_out(&graph);

    // 5. Propagate with e^{-iHt}
    double t = 1.0;
    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(t * cheby.half_width, 1e-12, true, coeffs);
    
    cheby.propagate(v_in, v_out, coeffs, ChebyMode::NORMAL);
    
    std::complex<double> phase = std::exp(std::complex<double>(0, -cheby.center * t));
    v_out.scale(phase);

    // 6. Verify
    double max_err = 0.0;
    for (int i = 0; i < v_out.local_size; ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> expected = std::exp(std::complex<double>(0, -static_cast<double>(gid) * t));
        double err = std::abs(v_out[i] - expected);
        max_err = std::max(max_err, err);
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  Max Propagation Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

void test_cheby_adjoint(int rank, int size) {
    if (rank == 0) std::cout << "Testing Chebyshev Adjoint Mode..." << std::endl;

    // 1. Setup Graph
    int n_blocks = 4;
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i};
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup H and S (Diagonal)
    BlockSpMat<std::complex<double>> H(&graph);
    BlockSpMat<std::complex<double>> S(&graph);
    for (int i = 0; i < (int)graph.owned_global_indices.size(); ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> h_val = static_cast<double>(gid + 1); // 1, 2, 3, 4
        std::complex<double> s_val = 2.0;
        H.add_block(gid, gid, &h_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        S.add_block(gid, gid, &s_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();
    S.assemble();

    // 3. Chebyshev Setup
    // Op = H S^-1. Values: 0.5, 1.0, 1.5, 2.0. Range [0.5, 2.0]
    Chebyshev<std::complex<double>> cheby(H, S, 0.5, 2.0);

    // 4. Input v = [1, 1, 1, 1]
    DistVector<std::complex<double>> v_in(&graph);
    v_in.set_constant(1.0);
    DistVector<std::complex<double>> v_out(&graph);

    // 5. Propagate with Adjoint mode
    double t = 1.0;
    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(t * cheby.half_width, 1e-12, true, coeffs);
    
    cheby.propagate(v_in, v_out, coeffs, ChebyMode::ADJOINT);
    
    std::complex<double> phase = std::exp(std::complex<double>(0, -cheby.center * t));
    v_out.scale(phase);

    // 6. Verify: v_out[j] = e^{-i (H_jj/S_jj) t} * v_in[j]
    double max_err = 0.0;
    for (int i = 0; i < v_out.local_size; ++i) {
        int gid = graph.owned_global_indices[i];
        double val = static_cast<double>(gid + 1) / 2.0;
        std::complex<double> expected = std::exp(std::complex<double>(0, -val * t));
        double err = std::abs(v_out[i] - expected);
        max_err = std::max(max_err, err);
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  Max Adjoint Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

void test_cheby_preconditioned(int rank, int size) {
    if (rank == 0) std::cout << "Testing Chebyshev Preconditioned Mode..." << std::endl;

    // 1. Setup Graph
    int n_blocks = 4;
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i};
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup H and S (Diagonal)
    BlockSpMat<std::complex<double>> H(&graph);
    BlockSpMat<std::complex<double>> S(&graph);
    for (int i = 0; i < (int)graph.owned_global_indices.size(); ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> h_val = static_cast<double>(gid + 1); // 1, 2, 3, 4
        std::complex<double> s_val = 2.0;
        H.add_block(gid, gid, &h_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        S.add_block(gid, gid, &s_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();
    S.assemble();

    // 3. Preconditioner Setup
    BlockJacobianPreconditioner<std::complex<double>> M(S);

    // 4. Chebyshev Setup
    // Op = H S^-1. Values: 0.5, 1.0, 1.5, 2.0. Range [0.5, 2.0]
    Chebyshev<std::complex<double>> cheby(H, S, M, rsatb::calc::PreconditionerType::Left, 0.5, 2.0);

    // 5. Input v = [1, 1, 1, 1]
    DistVector<std::complex<double>> v_in(&graph);
    v_in.set_constant(1.0);
    DistVector<std::complex<double>> v_out(&graph);

    // 6. Propagate
    double t = 1.0;
    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(t * cheby.half_width, 1e-12, true, coeffs);
    
    cheby.propagate(v_in, v_out, coeffs, ChebyMode::NORMAL);
    
    std::complex<double> phase = std::exp(std::complex<double>(0, -cheby.center * t));
    v_out.scale(phase);

    // 7. Verify
    double max_err = 0.0;
    for (int i = 0; i < v_out.local_size; ++i) {
        int gid = graph.owned_global_indices[i];
        double val = static_cast<double>(gid + 1) / 2.0;
        std::complex<double> expected = std::exp(std::complex<double>(0, -val * t));
        double err = std::abs(v_out[i] - expected);
        max_err = std::max(max_err, err);
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  Max Preconditioned Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-6) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

void test_cheby_direct_preconditioned(int rank, int size) {
    if (rank == 0) std::cout << "Testing Chebyshev Direct Preconditioned Mode (S=null, M!=null)..." << std::endl;

    // 1. Setup Graph
    int n_blocks = 4;
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for(int i=0; i<n_blocks; ++i) global_adj[i] = {i};
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    // 2. Setup H (Diagonal: 1, 2, 3, 4)
    BlockSpMat<std::complex<double>> H(&graph);
    BlockSpMat<std::complex<double>> S_dummy(&graph); // To build M
    for (int i = 0; i < (int)graph.owned_global_indices.size(); ++i) {
        int gid = graph.owned_global_indices[i];
        std::complex<double> h_val = static_cast<double>(gid + 1);
        std::complex<double> s_val = 4.0; // M will be 1/4
        H.add_block(gid, gid, &h_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        S_dummy.add_block(gid, gid, &s_val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();
    S_dummy.assemble();

    // 3. Preconditioner Setup (M = 1/4)
    BlockJacobianPreconditioner<std::complex<double>> M(S_dummy);

    // 4. Chebyshev Setup
    // Op = M (H - center) M / half_width. 
    // H range [1, 4], M = 1/4.
    // M H M range [1/16, 4/16] = [0.0625, 0.25]
    double emin = 1.0/16.0;
    double emax = 4.0/16.0;
    
    Chebyshev<std::complex<double>> cheby(H, M, rsatb::calc::PreconditionerType::Symmetric, emin, emax);

    // 5. Input v = [1, 1, 1, 1]
    DistVector<std::complex<double>> v_in(&graph);
    v_in.set_constant(1.0);
    DistVector<std::complex<double>> v_out(&graph);

    // 6. Propagate (just to check if it runs)
    double t = 1.0;
    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(t * cheby.half_width, 1e-12, true, coeffs);
    
    cheby.propagate(v_in, v_out, coeffs, ChebyMode::NORMAL);
    
    // 7. Verify apply_op directly.
    DistVector<std::complex<double>> y(&graph);
    cheby.apply_op(v_in, y, ChebyMode::NORMAL);
    
    double max_err = 0.0;
    for (int i = 0; i < y.local_size; ++i) {
        int gid = graph.owned_global_indices[i];
        double h_val = static_cast<double>(gid + 1);
        double m_val = 1.0/4.0;
        double op_val = (m_val * h_val * m_val - cheby.center) / cheby.half_width;
        double err = std::abs(y[i] - op_val);
        max_err = std::max(max_err, err);
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  Max Direct Preconditioned Apply Error: " << global_max_err << std::endl;
        if (global_max_err > 1e-12) std::cout << "  FAILED" << std::endl;
        else std::cout << "  PASSED" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) test_timeprop_coeff();
    test_cheby_propagate(rank, size);
    test_cheby_adjoint(rank, size);
    test_cheby_preconditioned(rank, size);
    test_cheby_direct_preconditioned(rank, size);

    MPI_Finalize();
    return 0;
}
