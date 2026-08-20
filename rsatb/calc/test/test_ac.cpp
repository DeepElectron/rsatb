// AC correlator tests: sequential vs concurrent (shared-Chebyshev-state)
// evolution, block restarts, and the alias-free sampling.
//
// Model: pairs of levels at -0.75 and +0.75 (span [-1, 1]), Vx coupling the
// two members of each pair, mu = 0 at low temperature. Then
//   C_xx(t) = 0.5 * e^{-i 1.5 t}
// exactly (|psi_i|^2 is exactly 1/N, so there is no stochastic error): a
// pure tone at 1.5 -- ABOVE what dt = pi sampling can represent
// (e^{-1.5 i pi k} = e^{+0.5 i pi k}: on that grid it aliases onto -0.5
// with a sign flip in the sine transform). Both production routes sample at
// the Nyquist-exact dt = pi/2 (kAcOversample); the defect of the historical
// pi grid is demonstrated here by subsampling the correct correlator back
// onto it. The model is simultaneously a machine-precision equivalence
// check and that physical demonstration.
#include "../ac.hpp"
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>

using namespace rsatb::calc;

namespace {

using cd = std::complex<double>;

// The sine transform ComputeAc applies, at one frequency.
double sine_transform(double omega, const std::vector<cd>& cac, int ns,
                      double t_step, int component = 0) {
    double val = 0.0;
    for (int it = 0; it <= ns; ++it) {
        val += 2.0 * std::sin(omega * it * t_step) *
               std::exp(-2.0 * std::pow((double)(it + 1) / (ns + 1), 2)) *
               cac[it * 6 + component].imag();
    }
    return val * t_step;
}

double max_component_diff(const std::vector<cd>& a, const std::vector<cd>& b,
                          int n_samples, int stride_a, int stride_b) {
    double max_diff = 0.0;
    for (int k = 0; k <= n_samples; ++k) {
        for (int c = 0; c < 6; ++c) {
            max_diff = std::max(max_diff,
                                std::abs(a[(size_t)k * stride_a * 6 + c] -
                                         b[(size_t)k * stride_b * 6 + c]));
        }
    }
    return max_diff;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing AC correlator (sequential vs concurrent)..." << std::endl;

    // Pairs of coupled levels: sites 2p (E = -0.75) and 2p+1 (E = +0.75).
    int n_pairs = size * 8;
    int n_blocks_graph = 2 * n_pairs;
    std::vector<int> block_sizes(n_blocks_graph, 1);
    std::vector<std::vector<int>> global_adj(n_blocks_graph);
    for (int p = 0; p < n_pairs; ++p) {
        global_adj[2 * p] = {2 * p, 2 * p + 1};
        global_adj[2 * p + 1] = {2 * p, 2 * p + 1};
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks_graph, block_sizes, global_adj);

    BlockSpMat<cd> H(&graph), Vx(&graph), Vy(&graph), Vz(&graph);
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        cd e = (gid % 2 == 0) ? -0.75 : 0.75;
        cd one = 1.0, zero = 0.0;
        int partner = (gid % 2 == 0) ? gid + 1 : gid - 1;
        H.add_block(gid, gid, &e, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        Vx.add_block(gid, partner, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        Vy.add_block(gid, gid, &zero, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        Vz.add_block(gid, gid, &zero, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();
    Vx.assemble();
    Vy.assemble();
    Vz.assemble();

    Chebyshev<cd> cheby(H, -1.0, 1.0);

    DistVector<cd> psi_in(&graph);
    const double inv_sqrt_n = 1.0 / std::sqrt((double)graph.global_scalar_rows());
    cd* data = psi_in.local_data();
    for (int i = 0; i < psi_in.local_size; ++i) {
        int gid = graph.owned_global_indices[i];
        data[i] = ((gid * 7 + 3) % 5 < 2 ? 1.0 : -1.0) * inv_sqrt_n;
    }

    const int nt = 64;
    const double temp = 30.0;
    const double mu = 0.0;
    const int n_kpm = 400;
    bool all_ok = true;

    const int ns = nt * kAcOversample;

    // 1. Sequential baseline (now also at the Nyquist step pi/2).
    std::vector<cd> cac_seq;
    corr_ac(cheby, Vx, Vy, Vz, psi_in, nt, temp, mu, 1.0, 0.0, n_kpm, 1e-14, &cac_seq);

    // 2. Concurrent, one full block: same estimator, same sampling -- must
    // agree with sequential to rounding + expansion truncation.
    std::vector<cd> cac_conc;
    corr_ac_concurrent(cheby, Vx, Vy, Vz, psi_in, nt, temp, mu, n_kpm, 1e-14, &cac_conc,
                       /*block=*/0);
    // 3. Concurrent with a block size that does not divide nt: exercises the
    // block-restart path.
    std::vector<cd> cac_cb;
    corr_ac_concurrent(cheby, Vx, Vy, Vz, psi_in, nt, temp, mu, n_kpm, 1e-14, &cac_cb,
                       /*block=*/7);

    if (rank == 0) {
        double d_seq = max_component_diff(cac_seq, cac_conc, ns, 1, 1);
        double d_block = max_component_diff(cac_conc, cac_cb, ns, 1, 1);
        std::cout << "  |sequential - concurrent|      = " << d_seq << std::endl;
        std::cout << "  |full block - blocked(7)|      = " << d_block << std::endl;

        // Analytic correlator: C_xx(t) = 0.5 e^{-1.5 i t}. Limited by the
        // Fermi-operator expansion, not by propagation.
        double d_exact = 0.0;
        for (int k = 0; k <= ns; ++k) {
            double t = k * M_PI / kAcOversample;
            d_exact = std::max(d_exact, std::abs(cac_conc[(size_t)k * 6] -
                                                 0.5 * std::exp(cd(0.0, -1.5 * t))));
        }
        std::cout << "  |C_xx - 0.5 e^{-1.5it}|        = " << d_exact << std::endl;

        // Aliasing: the tone at 1.5 must NOT produce a response at 0.5. The
        // historical dt = pi grid is reconstructed by subsampling the even
        // samples of the correct correlator.
        std::vector<cd> cac_coarse((size_t)(nt + 1) * 6);
        for (int k = 0; k <= nt; ++k)
            for (int c = 0; c < 6; ++c)
                cac_coarse[(size_t)k * 6 + c] = cac_conc[(size_t)(2 * k) * 6 + c];
        double t_coarse = sine_transform(0.5, cac_coarse, nt, M_PI);
        double t_alias = sine_transform(0.5, cac_conc, ns, M_PI / kAcOversample);
        double t_true = sine_transform(1.5, cac_conc, ns, M_PI / kAcOversample);
        std::cout << "  sine transform @0.5: pi grid " << t_coarse
                  << " (spurious), pi/2 grid " << t_alias << std::endl;
        std::cout << "  sine transform @1.5: pi/2 grid " << t_true
                  << " (true peak)" << std::endl;

        bool ok = d_seq < 1e-10 && d_block < 1e-10 &&
                  d_exact < 1e-3 &&
                  std::abs(t_alias) < 0.05 * std::abs(t_coarse) &&
                  std::abs(t_true) > std::abs(t_coarse);
        std::cout << (ok ? "CONCURRENT PASSED" : "CONCURRENT FAILED") << std::endl;
        all_ok = all_ok && ok;
    }

    // 5. Multivector: identical columns must reproduce the single-vector
    // result through the averaged dots.
    if (rank == 0) std::cout << "\nTesting Multivector Concurrent..." << std::endl;
    int num_vecs = 3;
    DistMultiVector<cd> psi_mv(&graph, num_vecs);
    for (int i = 0; i < psi_mv.local_rows; ++i) {
        cd* row_data = psi_mv.row_data(i);
        for (int v = 0; v < num_vecs; ++v) row_data[v] = data[i];
    }
    std::vector<cd> cac_mv;
    corr_ac_concurrent(cheby, Vx, Vy, Vz, psi_mv, nt, temp, mu, n_kpm, 1e-14, &cac_mv,
                       /*block=*/0);
    if (rank == 0) {
        double d_mv = max_component_diff(cac_conc, cac_mv, ns, 1, 1);
        std::cout << "  |single - multivector| = " << d_mv << std::endl;
        bool ok = d_mv < 1e-10;
        std::cout << (ok ? "MULTIVECTOR PASSED" : "MULTIVECTOR FAILED") << std::endl;
        all_ok = all_ok && ok;
    }

    // 6. Operator-form dressing of an auxiliary operator.
    //
    // The velocity operator has to be carried into whichever representation
    // the recursion iterates:
    //
    //     solve      S^-1 V c          on coefficients c
    //     sandwich   Z^H V Z eta       on eta, with c = Z eta
    //
    // These are the same physical operator, so Z (Z^H V Z eta) and
    // S^-1 V (Z eta) agree identically. Before Chebyshev recorded its form,
    // the sandwich case was selected by "S == nullptr" -- the test an
    // orthogonal basis also passes -- and applied a bare V, dropping the Z on
    // both sides. The overlap below is deliberately NOT diagonal, so that
    // failure cannot hide as a scale factor; the second number printed is the
    // margin by which the test would catch it.
    if (rank == 0) std::cout << "\nTesting operator-form dressing of V..." << std::endl;
    {
        BlockSpMat<cd> S(&graph);
        for (int i = 0; i < (int)graph.owned_global_indices.size(); ++i) {
            int gid = graph.owned_global_indices[i];
            int partner = (gid % 2 == 0) ? gid + 1 : gid - 1;
            cd diag = 1.0, off = 0.3;
            S.add_block(gid, gid, &diag, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            S.add_block(gid, partner, &off, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        S.assemble();

        BlockSpMat<cd> Z(S.duplicate(true));
        inverse_sqrt(S, &Z, InverseSqrtOptions{1e-12, 1e-13, 50});
        ISQRTMPreconditioner<cd> Mz(Z, kBorrowFactor);

        Chebyshev<cd> cheby_solve(H, S, Mz, PreconditionerType::Symmetric, -1.0, 1.0);
        Chebyshev<cd> cheby_sand(H, Mz, PreconditionerType::Symmetric, -1.0, 1.0);
        cheby_solve.solver = LinearSolver<cd>(1e-13, 2000, false);

        DistVector<cd> eta(&graph), c(&graph), lhs(&graph), rhs(&graph);
        DistVector<cd> dressed(&graph), bare(&graph);
        for (int i = 0; i < eta.local_size; ++i) {
            int gid = graph.owned_global_indices[i];
            eta[i] = cd(std::sin(0.7 * gid + 0.2), std::cos(0.3 * gid + 1.1));
        }
        Mz.apply(eta, c);                            // c = Z eta

        cheby_sand.apply_dual(Vx, eta, dressed);     // Z^H Vx Z eta
        Mz.apply(dressed, lhs);                      // Z (Z^H Vx Z eta)
        cheby_solve.apply_dual(Vx, c, rhs);          // S^-1 Vx (Z eta)
        Vx.mult(eta, bare);                          // what the bug applied

        double local[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < lhs.local_size; ++i) {
            local[0] = std::max(local[0], std::abs(lhs[i] - rhs[i]));
            local[1] = std::max(local[1], std::abs(dressed[i] - bare[i]));
            local[2] = std::max(local[2], std::abs(rhs[i]));
        }
        double g[3] = {0.0, 0.0, 0.0};
        MPI_Allreduce(local, g, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "  |Z (Z^H V Z eta) - S^-1 V (Z eta)| = " << g[0] << std::endl;
            std::cout << "  |Z^H V Z eta - V eta| = " << g[1]
                      << " (gap a dropped Z would open)" << std::endl;
            bool ok = g[0] < 1e-9 * g[2] && g[1] > 0.1 * g[2];
            std::cout << (ok ? "DRESSING PASSED" : "DRESSING FAILED") << std::endl;
            all_ok = all_ok && ok;
        }
    }

    MPI_Finalize();
    return (rank == 0 && !all_ok) ? 1 : 0;
}
