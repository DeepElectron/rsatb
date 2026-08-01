// Validation driver for the adaptive tapered (ACE-FD) Fermi expansion.
//
// Diagonal Hamiltonians give exact references: for eigenvalues {E_i} and the
// normalized uniform vector psi, <psi| f_N(H) |psi> must reproduce
// sum_i f(E_i)/n to the requested weak-form tolerance.
//
// Checks:
//   1. Dense (quasi-continuum) spectrum: the adaptive degree converges, the
//      error meets tolerance, and the degree is essentially temperature
//      independent between 300 K and 4 K (the whole point of the method).
//   2. Kernel-weight sanity: Jackson path of kpm_kernel_weights matches the
//      original jackson_kernel; tapers reproduce the leading half exactly.
//   3. Safety on discrete spectra (including an isolated level just above
//      mu, the published false-convergence counterexample for the naive
//      two-level indicator): whenever the indicator fires, the error is
//      actually small; otherwise the cap was reported honestly. That holds
//      in the regime the method is stated over -- for what happens outside
//      it, see benchmarks/fermi_dirac_diamond.
//
// Run with 1 or more MPI ranks.
#include "../fermi_dirac.hpp"
#include "../occupation.hpp"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace rsatb::calc;
using cdbl = std::complex<double>;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, int rank) {
    if (!ok) {
        ++g_failures;
        if (rank == 0) std::cout << "  FAIL: " << what << "\n";
    } else if (rank == 0) {
        std::cout << "  ok:   " << what << "\n";
    }
}

struct CaseResult {
    double error = 0.0;
    FermiExpansionReport report;
};

// Diagonal H = diag(eigs); psi = uniform, normalized. Returns the error of
// <psi|f_N(H)|psi> against the exact discrete sum, plus the adaptive report.
CaseResult run_case(MPI_Comm comm, const std::vector<double>& eigs,
                    double temp, double mu, const FermiExpansionOptions& opt) {
    const int n = static_cast<int>(eigs.size());
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) adj[i] = {i};

    DistGraph graph(comm);
    graph.construct_serial(n, block_sizes, adj);

    BlockSpMat<cdbl> H(&graph);
    const int n_owned = static_cast<int>(graph.owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gid = graph.owned_global_indices[i];
        cdbl val = eigs[static_cast<size_t>(gid)];
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    Chebyshev<cdbl> cheby(H, -1.0, 1.0);

    DistVector<cdbl> psi_in(&graph);
    psi_in.set_constant(1.0 / std::sqrt(static_cast<double>(n)));
    DistVector<cdbl> psi_out(&graph);

    CaseResult res;
    fermi_dirac_kpm(cheby, psi_in, psi_out, temp, mu, opt, ChebyMode::NORMAL,
                    &res.report);

    const double q_est = std::real(psi_in.dot(psi_out));

    const double kb = 8.617333262e-5;
    const double beta = (temp > 0.0) ? 1.0 / (temp * kb)
                                     : std::numeric_limits<double>::infinity();
    double q_exact = 0.0;
    for (double e : eigs) q_exact += occupation_projector_func(e, beta, mu);
    q_exact /= static_cast<double>(n);

    res.error = std::abs(q_est - q_exact);
    return res;
}

// Chebyshev-node spectrum on [-0.92, 0.92]: a smooth quasi-continuum band.
std::vector<double> dense_band(int n) {
    std::vector<double> eigs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        eigs[static_cast<size_t>(i)] =
            -0.92 * std::cos(M_PI * (static_cast<double>(i) + 0.5) / n);
    }
    return eigs;
}

// Trace moments of a diagonal system with a uniform normalized sample.
TraceMoments trace_case(MPI_Comm comm, const std::vector<double>& eigs,
                        double temp, double mu_anchor,
                        const FermiExpansionOptions& opt) {
    const int n = static_cast<int>(eigs.size());
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) adj[i] = {i};

    DistGraph graph(comm);
    graph.construct_serial(n, block_sizes, adj);

    BlockSpMat<cdbl> H(&graph);
    const int n_owned = static_cast<int>(graph.owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gid = graph.owned_global_indices[i];
        cdbl val = eigs[static_cast<size_t>(gid)];
        H.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    H.assemble();

    Chebyshev<cdbl> cheby(H, -1.0, 1.0);
    DistVector<cdbl> psi(&graph);
    psi.set_constant(1.0 / std::sqrt(static_cast<double>(n)));
    return adaptive_trace_moments(cheby, psi, temp, mu_anchor, opt);
}

// Exact trace against the discrete spectrum: sum_i w(E_i) f(E_i) / n.
double exact_trace(const std::vector<double>& eigs, double temp, double mu,
                   bool energy_weighted) {
    const double kb = 8.617333262e-5;
    const double beta = (temp > 0.0) ? 1.0 / (temp * kb)
                                     : std::numeric_limits<double>::infinity();
    double q = 0.0;
    for (double e : eigs) {
        const double f = occupation_projector_func(e, beta, mu);
        q += energy_weighted ? e * f : f;
    }
    return q / static_cast<double>(eigs.size());
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) std::cout << "Testing adaptive ACE-FD Fermi expansion...\n";

    // --- kernel-weight sanity -------------------------------------------
    {
        const int n = 257;
        const auto gj = kpm_kernel_weights(KpmKernel::Jackson, n);
        const auto gj_ref = jackson_kernel(n);
        double dmax = 0.0;
        for (int i = 0; i < n; ++i) dmax = std::max(dmax, std::abs(gj[i] - gj_ref[i]));
        check(dmax == 0.0, "Jackson weights match jackson_kernel", rank);

        for (KpmKernel k : {KpmKernel::CosTaper, KpmKernel::VpTaper}) {
            const auto g = kpm_kernel_weights(k, n);
            bool head_exact = true;
            for (int i = 0; i < n / 2; ++i) head_exact = head_exact && g[i] == 1.0;
            check(head_exact && g[n - 1] < 0.05 && g[0] == 1.0,
                  "taper reproduces leading half, rolls off tail", rank);
        }
    }

    // --- dense spectrum: accuracy and T-independence --------------------
    {
        const auto eigs = dense_band(20000);
        const double mu = -0.3;
        FermiExpansionOptions opt; // defaults: CosTaper, adaptive, tol 1e-4

        const CaseResult r300 = run_case(MPI_COMM_WORLD, eigs, 300.0, mu, opt);
        const CaseResult r4 = run_case(MPI_COMM_WORLD, eigs, 4.0, mu, opt);
        // 1 K: beta_scaled ~ 1.2e4 >> the old fixed 4096-point coefficient
        // grid — regression test for the thermal-aware grid rule.
        const CaseResult r1 = run_case(MPI_COMM_WORLD, eigs, 1.0, mu, opt);
        // Step limit: beta_scaled far beyond the grid cap exercises the
        // exact analytic T=0 coefficient branch.
        const CaseResult r0 = run_case(MPI_COMM_WORLD, eigs, 1e-3, mu, opt);
        if (rank == 0) {
            std::cout << "  dense band: N(300K) = " << r300.report.n_used
                      << " err = " << r300.error
                      << " | N(4K) = " << r4.report.n_used
                      << " err = " << r4.error
                      << " | N(1K) = " << r1.report.n_used
                      << " err = " << r1.error
                      << " | N(step) = " << r0.report.n_used
                      << " err = " << r0.error << "\n";
        }
        // The recurrence that picks the degree is the one that sums the
        // series, so the whole run costs N - 1 matvecs -- what a fixed-degree
        // expansion would cost had the degree been known up front. Anything
        // above that means a pass was thrown away.
        if (rank == 0) {
            std::cout << "  single pass: N(300K) = " << r300.report.n_used
                      << " matvecs = " << r300.report.matvecs
                      << " (probe-then-expand would be " << 2 * r300.report.n_used << ")"
                      << std::endl;
        }
        check(r300.report.matvecs == r300.report.n_used - 1 &&
              r4.report.matvecs == r4.report.n_used - 1,
              "degree search and summation share one recurrence", rank);
        check(r300.report.indicator_fired, "300 K adaptive run: indicator fired", rank);
        check(r4.report.indicator_fired, "4 K adaptive run: indicator fired", rank);
        check(r300.error < 10.0 * opt.indicator_threshold, "300 K error within tolerance", rank);
        check(r4.error < 10.0 * opt.indicator_threshold, "4 K error within tolerance", rank);
        check(r1.report.indicator_fired && r1.error < 10.0 * opt.indicator_threshold,
              "1 K accurate (thermal-aware coefficient grid)", rank);
        check(r0.report.indicator_fired && r0.error < 10.0 * opt.indicator_threshold,
              "step-limit branch accurate (analytic coefficients)", rank);
        check(r4.report.n_used <= 4 * r300.report.n_used &&
                  r1.report.n_used <= 4 * r300.report.n_used &&
                  r0.report.n_used <= 4 * r300.report.n_used && r1.report.n_used <= 4096,
              "degree is temperature independent (no beta*W growth)", rank);
        // For reference: the pointwise sigma <= kT Jackson degree at 4 K is
        // pi*half_width/kT ~ 9100; the adaptive degree must be far below it.
        check(r4.report.n_used < 9100 / 4,
              "adaptive degree well below pointwise degree at 4 K", rank);
    }

    // --- the worst case is capped at Jackson parity ---------------------
    //
    // METHOD.md Sec. 5.2 and the Sec. 9 cost table rest the whole worst case on
    // this: when the indicator never fires, the run stops at the pointwise
    // degree, where a tapered sum is Jackson-grade by construction -- so cost
    // never exceeds current practice. That holds only if the check ladder hits
    // the cap exactly. A ladder built up from n_min overshoots it by up to 2x
    // whenever the cap is not a rung, which silently doubles the bound.
    {
        // A single isolated level at mu defeats the indicator by design (the
        // measure is as far from self-averaging as it gets), so these runs go
        // all the way to the cap and expose where it actually lands.
        int worst_failures = 0;
        for (double temp : {300.0, 30.0, 4.0}) {
            const double kb = 8.617333262e-5;
            const double beta_scaled = 1.0 / (temp * kb);  // half_width = 1
            const int cap = static_cast<int>(2.0 * M_PI * beta_scaled) + 1;

            FermiExpansionOptions opt;
            opt.indicator_threshold = 1e-16;  // unreachable: force the run to the cap
            const CaseResult r = run_case(MPI_COMM_WORLD, {0.0}, temp, 0.0, opt);
            if (r.report.n_used > cap) {
                ++worst_failures;
                if (rank == 0) {
                    std::cout << "    T = " << temp << " K: N = " << r.report.n_used
                              << " exceeds the pointwise cap " << cap << "\n";
                }
            }
        }
        check(worst_failures == 0,
              "capped runs never exceed the pointwise degree (Jackson parity)", rank);
    }

    // --- discrete spectra: honesty of the report ------------------------
    {
        // (a) 200 well-separated levels at 1 K: level spacing >> kT. The
        // expansion may legitimately need to resolve levels; whatever
        // happens, a fired indicator must mean a small true error here.
        const auto eigs = dense_band(200);
        const CaseResult r = run_case(MPI_COMM_WORLD, eigs, 1.0, -0.3,
                                      FermiExpansionOptions{});
        if (rank == 0) {
            std::cout << "  discrete band: N = " << r.report.n_used
                      << " err = " << r.error << " converged = "
                      << r.report.indicator_fired << "\n";
        }
        check(!r.report.indicator_fired || r.error < 2e-3,
              "discrete band: converged implies accurate", rank);

        // (b) Isolated level just above mu at T = 0 — the counterexample
        // that defeats the naive |q(N)-q(N/2)| rule (which stops at N = 256
        // with a 3e-2 error). The hardened indicator must not do that.
        std::vector<double> spiky = {-0.9, -0.8, 0.0229845371117, 0.9};
        const CaseResult r2 = run_case(MPI_COMM_WORLD, spiky, 0.0, 0.0,
                                       FermiExpansionOptions{});
        if (rank == 0) {
            std::cout << "  isolated level: N = " << r2.report.n_used
                      << " err = " << r2.error << " converged = "
                      << r2.report.indicator_fired << "\n";
        }
        check(!r2.report.indicator_fired || r2.error < 2e-3,
              "isolated level near mu: no false convergence", rank);
    }

    // --- trace moments: identities, (mu, T) reuse, and the mu-search ----
    {
        const auto eigs = dense_band(20000);
        FermiExpansionOptions opt; // adaptive defaults
        const double anchor_T = 4.0;
        const double anchor_mu = -0.3;
        const TraceMoments tm =
            trace_case(MPI_COMM_WORLD, eigs, anchor_T, anchor_mu, opt);
        if (rank == 0) {
            std::cout << "  trace moments: N = " << tm.n_used
                      << " indicator = " << tm.indicator << "\n";
        }
        check(tm.indicator_fired && tm.n_used <= 4096,
              "trace moments converged at T-independent degree", rank);

        // Doubling identities against exact spectral moments
        // ref[m] = (1/n) sum_i T_m(E_i), by direct recurrence on eigenvalues.
        const int n_check = std::min(64, tm.n_used);
        std::vector<double> ref(static_cast<size_t>(n_check), 0.0);
        {
            std::vector<double> tp(eigs.size(), 1.0);
            std::vector<double> tc(eigs.begin(), eigs.end());
            ref[0] = 1.0;
            for (int m = 1; m < n_check; ++m) {
                double s = 0.0;
                for (double v : tc) s += v;
                ref[static_cast<size_t>(m)] = s / static_cast<double>(eigs.size());
                for (size_t i = 0; i < eigs.size(); ++i) {
                    const double next = 2.0 * eigs[i] * tc[i] - tp[i];
                    tp[i] = tc[i];
                    tc[i] = next;
                }
            }
        }
        double dmax = 0.0;
        for (int m = 0; m < n_check; ++m) {
            dmax = std::max(dmax, std::abs(tm.moments[static_cast<size_t>(m)] -
                                           ref[static_cast<size_t>(m)]));
        }
        check(dmax < 1e-10, "doubling-trick moments match exact spectral moments", rank);

        // (mu, T) reuse: evaluate far from the adaptivity anchor, zero MVPs.
        const double err_ne = std::abs(fermi_trace_sum(tm, 300.0, -0.5) -
                                       exact_trace(eigs, 300.0, -0.5, false));
        const double err_eb = std::abs(
            fermi_trace_sum(tm, 300.0, -0.5, KpmKernel::CosTaper, true) -
            exact_trace(eigs, 300.0, -0.5, true));
        check(err_ne < 1e-3 && err_eb < 1e-3 &&
                  fermi_trace_indicator(tm, 300.0, -0.5) < 1e-3,
              "moment reuse at a different (mu, T) stays accurate", rank);

        // Chemical-potential recovery: target the exact count at mu = -0.2.
        const double target = exact_trace(eigs, anchor_T, -0.2, false);
        const double mu_j = bisect_fermi_level(
            [&](double m) { return fermi_trace_sum(tm, anchor_T, m, KpmKernel::Jackson); },
            -1.0, 1.0, target);
        const double ne_at_mu = fermi_trace_sum(tm, anchor_T, mu_j);
        if (rank == 0) {
            std::cout << "  mu-search: mu = " << mu_j << " (exact -0.2), ne err = "
                      << std::abs(ne_at_mu - target) << "\n";
        }
        check(std::abs(mu_j - (-0.2)) < 5e-3 && std::abs(ne_at_mu - target) < 5e-4,
              "Fermi-level bisection recovers mu in coefficient space", rank);
    }

    int global_failures = 0;
    MPI_Allreduce(&g_failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << (global_failures == 0 ? "PASSED" : "FAILED") << "\n";
    }

    MPI_Finalize();
    return global_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
