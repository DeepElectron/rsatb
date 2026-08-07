// DOS moment-machinery tests.
//
// Model: N levels split evenly between -0.5 and +0.5 on a diagonal H, with
// sample entries +-1/sqrt(N). The stochastic estimator then has ZERO error:
// |psi_i|^2 is exactly 1/N, so mu_n = (T_n(0.5) + T_n(-0.5))/2 = cos(n pi/3)
// for even n and 0 for odd n, and the correlation series is exactly
// C(t) = cos(0.5 t). That makes machine-precision assertions possible.
//
// The stepwise-propagation route survives here only as an INLINE
// cross-check (cheby.propagate + dots -- the operations the removed
// corr_dos function performed): corr_from_moments must reproduce it to
// expansion truncation on both a diagonal and a complex-hopping operator.
// The equivalence that justified removing corr_dos was measured at 3.8e-14
// (diagonal) and 5.8e-14 (non-diagonal), serial and 3 ranks.
#include "../dos.hpp"
#include "../kpm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <complex>

using namespace rsatb::calc;

namespace {

using cd = std::complex<double>;

// What corr_dos used to do: stepwise e^{-i Op pi} propagation with an
// overlap against the initial state at every step.
template <typename ChebyT, typename VecT>
std::vector<cd> stepwise_corr(ChebyT& cheby, const VecT& psi_in, int nt) {
    std::vector<cd> coeffs;
    timeprop_coeff(M_PI, 1e-14, true, coeffs);
    VecT cur(psi_in.graph), out(psi_in.graph);
    cur.copy_from(psi_in);
    std::vector<cd> cdos(nt + 1);
    cdos[0] = static_cast<cd>(psi_in.dot(cur));
    for (int it = 0; it < nt; ++it) {
        cheby.propagate(cur, out, coeffs, ChebyMode::NORMAL);
        cdos[it + 1] = static_cast<cd>(psi_in.dot(out));
        cur.copy_from(out);
    }
    return cdos;
}

std::vector<cd> moments_corr(Chebyshev<cd>& cheby, const DistVector<cd>& psi,
                             int nt) {
    std::vector<cd> coeffs_tot;
    timeprop_coeff(nt * M_PI, 1e-14, true, coeffs_tot);
    std::vector<cd> mu = kpm_dos_moments(cheby, psi, static_cast<int>(coeffs_tot.size()));
    std::vector<cd> cdos;
    corr_from_moments(mu, nt, cdos);
    return cdos;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Testing DOS moment machinery..." << std::endl;
    bool all_ok = true;

    // 1. Diagonal two-level model: H = diag(+-0.5).
    int n_blocks = size * 100;
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> global_adj(n_blocks);
    for (int i = 0; i < n_blocks; ++i) global_adj[i] = {i};

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, global_adj);

    BlockSpMat<cd> H(&graph);
    int n_owned = graph.owned_global_indices.size();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        cd val = (gid % 2 == 0) ? 0.5 : -0.5;
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    Chebyshev<cd> cheby(H, -1.0, 1.0);

    // Deterministic +-1/sqrt(N) samples (independent of rank layout).
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(graph.global_scalar_rows()));
    DistVector<cd> psi_in(&graph);
    cd* data = psi_in.local_data();
    for (int i = 0; i < psi_in.local_size; ++i) {
        int gid = graph.owned_global_indices[i];
        data[i] = (((gid * 7 + 3) % 5 < 2) ? 1.0 : -1.0) * inv_sqrt_n;
    }

    const int nt = 10;
    const int n_moments = 64;

    // 2. KPM moments against the analytic values, and the doubled
    // (two-moments-per-matvec) path against the plain recurrence.
    if (!cheby.op_is_hermitian()) {
        if (rank == 0) std::cout << "FAILED (orthogonal operator not detected as Hermitian)" << std::endl;
        all_ok = false;
    }

    std::vector<cd> mu = kpm_dos_moments(cheby, psi_in, n_moments);
    std::vector<cd> mu_plain = cheby.compute_moments(psi_in, n_moments);

    if (rank == 0) {
        double max_err_exact = 0.0;
        double max_err_paths = 0.0;
        for (int n = 0; n < n_moments; ++n) {
            double exact = (n % 2 == 0) ? std::cos(n * M_PI / 3.0) : 0.0;
            max_err_exact = std::max(max_err_exact, std::abs(mu[n] - exact));
            max_err_paths = std::max(max_err_paths, std::abs(mu[n] - mu_plain[n]));
        }
        std::cout << "  max |mu - exact| = " << max_err_exact
                  << ", max |doubled - plain| = " << max_err_paths << std::endl;

        // Jackson density: integral mu_0 = 1, peaks at the eigenvalues.
        int n_grid = 400;
        std::vector<double> x_grid(n_grid);
        for (int i = 0; i < n_grid; ++i) x_grid[i] = -1.0 + 2.0 * (i + 0.5) / n_grid;
        std::vector<double> rho = evaluate_kpm_density(mu, x_grid, 1.0 / n_grid);
        double integral = 0.0;
        double peak_pos = 0.0, peak_val = 0.0;
        for (int i = 0; i < n_grid; ++i) {
            integral += rho[i] * 2.0 / n_grid;
            if (x_grid[i] > 0 && rho[i] > peak_val) { peak_val = rho[i]; peak_pos = x_grid[i]; }
        }
        std::cout << "  integral = " << integral << ", positive peak at x = " << peak_pos << std::endl;

        bool ok = max_err_exact < 1e-10 && max_err_paths < 1e-10 &&
                  std::abs(integral - 1.0) < 0.02 && std::abs(peak_pos - 0.5) < 0.05;
        std::cout << (ok ? "KPM PASSED" : "KPM FAILED") << std::endl;
        all_ok = all_ok && ok;
    }

    // 3. Correlation series from moments against the analytic cos(0.5 t)
    // (exact here), single vector and multivector.
    std::vector<cd> cdos_mom = moments_corr(cheby, psi_in, nt);

    int num_vecs = 4;
    DistMultiVector<cd> psi_mv(&graph, num_vecs);
    std::mt19937 gen(12345);
    std::uniform_int_distribution<> dis(0, 1);
    for (int i = 0; i < psi_mv.local_rows; ++i) {
        cd* row_data = psi_mv.row_data(i);
        for (int v = 0; v < num_vecs; ++v) {
            row_data[v] = ((dis(gen) == 0) ? 1.0 : -1.0) * inv_sqrt_n;
        }
    }
    std::vector<cd> coeffs_tot;
    timeprop_coeff(nt * M_PI, 1e-14, true, coeffs_tot);
    std::vector<cd> mu_mv = kpm_dos_moments(cheby, psi_mv, static_cast<int>(coeffs_tot.size()));
    std::vector<cd> cdos_mv;
    corr_from_moments(mu_mv, nt, cdos_mv);

    if (rank == 0) {
        double err_sv = 0.0, err_mv = 0.0;
        for (int it = 0; it <= nt; ++it) {
            cd exact = std::cos(0.5 * it * M_PI);
            err_sv = std::max(err_sv, std::abs(cdos_mom[it] - exact));
            err_mv = std::max(err_mv, std::abs(cdos_mv[it] - exact));
        }
        std::cout << "\n  |C(t) - cos(0.5 t)|: single " << err_sv
                  << ", multivector " << err_mv << std::endl;
        bool ok = err_sv < 1e-10 && err_mv < 1e-10;
        std::cout << (ok ? "CORR ANALYTIC PASSED" : "CORR ANALYTIC FAILED") << std::endl;
        all_ok = all_ok && ok;
    }

    // 4. Moments vs inline stepwise propagation on the diagonal model.
    {
        std::vector<cd> cdos_step = stepwise_corr(cheby, psi_in, nt);
        if (rank == 0) {
            double max_err = 0.0;
            for (int it = 0; it <= nt; ++it) {
                max_err = std::max(max_err, std::abs(cdos_mom[it] - cdos_step[it]));
            }
            std::cout << "  |moments - stepwise| (diagonal H) = " << max_err << std::endl;
            bool ok = max_err < 1e-12;
            std::cout << (ok ? "CORR STEPWISE PASSED" : "CORR STEPWISE FAILED") << std::endl;
            all_ok = all_ok && ok;
        }
    }

    // 5. Same identity on a NON-diagonal Hermitian operator (pair-coupled
    // levels with complex hopping): genuine Chebyshev dynamics.
    {
        std::vector<std::vector<int>> pair_adj(n_blocks);
        for (int p = 0; p < n_blocks / 2; ++p) {
            pair_adj[2 * p] = {2 * p, 2 * p + 1};
            pair_adj[2 * p + 1] = {2 * p, 2 * p + 1};
        }
        DistGraph graph2(MPI_COMM_WORLD);
        graph2.construct_serial(n_blocks, block_sizes, pair_adj);

        BlockSpMat<cd> H2(&graph2);
        int n_owned2 = graph2.owned_global_indices.size();
        for (int i = 0; i < n_owned2; ++i) {
            int gid = graph2.owned_global_indices[i];
            cd e = (gid % 2 == 0) ? -0.3 : 0.3;
            // Hermitian pair: v on (even, odd), conj(v) on (odd, even).
            cd v = (gid % 2 == 0) ? cd(0.35, 0.1) : cd(0.35, -0.1);
            int partner = (gid % 2 == 0) ? gid + 1 : gid - 1;
            H2.add_block(gid, gid, &e, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            H2.add_block(gid, partner, &v, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        H2.assemble();
        Chebyshev<cd> cheby2(H2, -1.0, 1.0);

        DistVector<cd> psi2(&graph2);
        cd* d2 = psi2.local_data();
        for (int i = 0; i < psi2.local_size; ++i) {
            int gid = graph2.owned_global_indices[i];
            d2[i] = (((gid * 7 + 3) % 5 < 2) ? 1.0 : -1.0) * inv_sqrt_n;
        }

        std::vector<cd> cdos_step = stepwise_corr(cheby2, psi2, nt);
        std::vector<cd> cdos2_mom = moments_corr(cheby2, psi2, nt);
        if (rank == 0) {
            double max_err = 0.0;
            for (int it = 0; it <= nt; ++it) {
                max_err = std::max(max_err, std::abs(cdos2_mom[it] - cdos_step[it]));
            }
            std::cout << "  |moments - stepwise| (complex hopping) = " << max_err << std::endl;
            bool ok = max_err < 1e-12;
            std::cout << (ok ? "CORR NONDIAG PASSED" : "CORR NONDIAG FAILED") << std::endl;
            all_ok = all_ok && ok;
        }
    }

    // 6. Bessel expansion sanity at production-scale total time
    // (nt = 2000 -> tau ~ 6283): finite coefficients, length ~ tau + tail.
    if (rank == 0) {
        std::vector<cd> coeffs_big;
        timeprop_coeff(2000 * M_PI, 1e-14, true, coeffs_big);
        bool finite = true;
        for (const auto& c : coeffs_big) {
            if (!std::isfinite(c.real()) || !std::isfinite(c.imag())) finite = false;
        }
        std::cout << "  timeprop_coeff(2000*pi): " << coeffs_big.size()
                  << " terms, finite = " << finite << std::endl;
        bool ok = finite && coeffs_big.size() > 6283 && coeffs_big.size() < 7000;
        std::cout << (ok ? "BESSEL PASSED" : "BESSEL FAILED") << std::endl;
        all_ok = all_ok && ok;
    }

    MPI_Finalize();
    return (rank == 0 && !all_ok) ? 1 : 0;
}
