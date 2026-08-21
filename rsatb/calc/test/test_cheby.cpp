#include "../cheby.hpp"
#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include <cmath>

using namespace vbcsr;
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

// The property ChebyMode exists to provide: <y, A x> == <A^H y, x>.
//
// Checked on a system where S^-1 H != H S^-1, which the diagonal fixture above
// cannot do -- there the two shapes coincide and a swap of them is invisible.
void test_cheby_adjoint_identity(int rank, int size) {
    if (rank == 0) std::cout << "Testing Chebyshev adjoint identity..." << std::endl;
    if (size != 1) {
        if (rank == 0) std::cout << "  SKIPPED (serial fixture)" << std::endl;
        return;
    }
    const int n = 4;
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        adj[i].push_back(i);
        if (i > 0) adj[i].push_back(i - 1);
        if (i + 1 < n) adj[i].push_back(i + 1);
    }
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n, block_sizes, adj);

    // Tridiagonal complex HERMITIAN H (real diagonal, H(i,i+1) = conj(H(i+1,i)))
    // and SPD tridiagonal S -- the operators this class is defined for. The
    // site-dependent diagonal keeps H from commuting with S, so S^-1 H and
    // H S^-1 remain genuinely different operators.
    //
    // Complex matters even though H is Hermitian: A^H = A here but
    // A^T = conj(A) does not, so an implementation that conjugated instead of
    // reversing the order still fails this. Only a REAL fixture would be
    // vacuous, which is why the AC test's real one cannot stand in for it.
    BlockSpMat<std::complex<double>> H(&graph), S(&graph);
    for (int i = 0; i < n; ++i) {
        std::complex<double> d(2.0 + 0.3 * i, 0.0);
        H.add_block(i, i, &d, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        std::complex<double> sd(2.0, 0.0);
        S.add_block(i, i, &sd, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        if (i > 0) {
            std::complex<double> lo(0.4, 0.7), so(-0.5, 0.0);
            H.add_block(i, i - 1, &lo, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            S.add_block(i, i - 1, &so, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        if (i + 1 < n) {
            std::complex<double> up(0.4, -0.7), so(-0.5, 0.0);
            H.add_block(i, i + 1, &up, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            S.add_block(i, i + 1, &so, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    H.assemble();
    S.assemble();

    DistVector<std::complex<double>> x(&graph), y(&graph), Ax(&graph), Ahy(&graph);
    for (int i = 0; i < n; ++i) {
        x[i] = std::complex<double>(0.3 + i, -0.7 + 0.2 * i);
        y[i] = std::complex<double>(-1.1 + 0.5 * i, 0.4 - 0.3 * i);
    }

    double worst = 0.0;
    auto check = [&](const char* name, Chebyshev<std::complex<double>>& cheby) {
        cheby.apply_op(x, Ax, ChebyMode::NORMAL);
        cheby.apply_op(y, Ahy, ChebyMode::ADJOINT);

        // <y, A x> against <A^H y, x>; dot() is conjugate-linear in its own
        // argument, so both sides are formed the same way.
        std::complex<double> lhs = y.dot(Ax);
        std::complex<double> rhs = Ahy.dot(x);
        const double err = std::abs(lhs - rhs) / std::max(1e-30, std::abs(lhs));
        worst = std::max(worst, err);
        if (rank == 0) {
            std::cout << "  " << name << "  <y,Ax>=" << lhs << "  <A^H y,x>=" << rhs
                      << "  rel err " << err << std::endl;
        }
    };

    // Every configurable form, each built through its own constructor. With
    // `form` const there is no back door to set it after the fact, so this is
    // also the construction path production takes.
    {
        Chebyshev<std::complex<double>> cheby(H, 0.1, 4.0);
        check("Orthogonal", cheby);
    }
    {
        Chebyshev<std::complex<double>> cheby(H, S, 0.1, 4.0);
        check("Solve     ", cheby);
    }
    {
        // Any Z will do here: Z^H H Z is self-adjoint-consistent for every Z,
        // so what this actually pins down is that the sandwich branch daggers
        // H (it is not Hermitian) and pairs apply with apply_adjoint.
        BlockJacobianPreconditioner<std::complex<double>> Z(S);
        Chebyshev<std::complex<double>> cheby(H, Z, rsatb::calc::PreconditionerType::Symmetric,
                                              0.1, 4.0);
        check("Sandwich  ", cheby);
    }
    if (rank == 0) {
        std::cout << "  Max adjoint-identity error: " << worst << std::endl;
        std::cout << (worst > 1e-8 ? "  FAILED" : "  PASSED") << std::endl;
    }
}

// The solve form under ADJOINT must be H S^-1 -- the factors REVERSED, not
// merely conjugated in place.
//
// Checked against a reference built from the forward kernel and CG, because
// the bilinear identity above cannot separate these two: S^-1 H^H is a
// perfectly good operator that simply is not the adjoint of S^-1 H, and it was
// what this branch applied before the mode and the form were pulled apart. The
// fixture is non-commuting, so the two differ by a wide margin -- asserted here
// as well, since a fixture where they agreed would make this test vacuous.
//
// The reference uses mult, never mult_adjoint, so it stays an independent
// check of the path under test rather than a restatement of it.
void test_cheby_solve_adjoint_shape(int rank, int size) {
    if (rank == 0) std::cout << "Testing solve-form ADJOINT is H S^-1 ..." << std::endl;
    if (size != 1) {
        if (rank == 0) std::cout << "  SKIPPED (serial fixture)" << std::endl;
        return;
    }
    const int n = 4;
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        adj[i].push_back(i);
        if (i > 0) adj[i].push_back(i - 1);
        if (i + 1 < n) adj[i].push_back(i + 1);
    }
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n, block_sizes, adj);

    // Same complex Hermitian fixture as the identity test above.
    BlockSpMat<std::complex<double>> H(&graph), S(&graph);
    for (int i = 0; i < n; ++i) {
        std::complex<double> d(2.0 + 0.3 * i, 0.0), sd(2.0, 0.0);
        H.add_block(i, i, &d, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        S.add_block(i, i, &sd, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        if (i > 0) {
            std::complex<double> lo(0.4, 0.7), so(-0.5, 0.0);
            H.add_block(i, i - 1, &lo, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            S.add_block(i, i - 1, &so, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        if (i + 1 < n) {
            std::complex<double> up(0.4, -0.7), so(-0.5, 0.0);
            H.add_block(i, i + 1, &up, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            S.add_block(i, i + 1, &so, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    H.assemble();
    S.assemble();

    DistVector<std::complex<double>> x(&graph);
    for (int i = 0; i < n; ++i) x[i] = std::complex<double>(1.0 + i, 0.5 - 0.2 * i);

    Chebyshev<std::complex<double>> cheby(H, S, 0.1, 4.0);
    DistVector<std::complex<double>> got(&graph);
    cheby.apply_op(x, got, ChebyMode::ADJOINT);

    LinearSolver<std::complex<double>> ref(1e-13, 2000, false);
    DistVector<std::complex<double>> t(&graph), want(&graph);

    // want = (H S^-1 x - center x) / half_width
    t.set_constant(std::complex<double>(0.0));
    ref.solve(S, x, t);
    H.mult(t, want);
    want.axpy(-cheby.center, x);
    want.scale(1.0 / cheby.half_width);

    // wrong = (S^-1 H x - center x) / half_width -- the factors left in their
    // original order, which is what NORMAL applies.
    DistVector<std::complex<double>> hx(&graph), wrong(&graph);
    H.mult(x, hx);
    wrong.set_constant(std::complex<double>(0.0));
    ref.solve(S, hx, wrong);
    wrong.axpy(-cheby.center, x);
    wrong.scale(1.0 / cheby.half_width);

    double err = 0.0, separation = 0.0;
    for (int i = 0; i < n; ++i) {
        err = std::max(err, std::abs(got[i] - want[i]));
        separation = std::max(separation, std::abs(want[i] - wrong[i]));
    }
    if (rank == 0) {
        std::cout << "  max |apply_op(ADJOINT) - H S^-1 x| = " << err << std::endl;
        std::cout << "  max |H S^-1 x - S^-1 H x|          = " << separation << std::endl;
        if (separation < 1e-6) {
            std::cout << "  FAILED (fixture commutes; test proves nothing)" << std::endl;
        } else {
            std::cout << (err > 1e-8 ? "  FAILED" : "  PASSED") << std::endl;
        }
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
    test_cheby_adjoint_identity(rank, size);
    test_cheby_solve_adjoint_shape(rank, size);
    test_cheby_preconditioned(rank, size);
    test_cheby_direct_preconditioned(rank, size);

    MPI_Finalize();
    return 0;
}
