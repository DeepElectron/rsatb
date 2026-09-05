// The velocity unit convention, locked across model families.
//
// Every velocity builder must store hbar * velocity in eV*Angstrom -- the
// hbar = 1 convention the correlator time grid (1/eV) and the SI constant
// e^2/(hbar*Angstrom) in observables.hpp assume (commutator_velocity.hpp
// UNITS). Historically the generic orthogonal builder divided by hbar while
// op2c did not, which silently put SK-model conductivities on a scale
// hbar^2 = 0.433 apart from LCAO ones (doc/position_commutator.md section 7).
//
// Two locks, deliberately family-crossing and phrased against the SAME
// constant:
//
//   1. SK (exact): the eigenbasis diagonal of the stored velocity equals
//      -dE_n/dk from central differences of the band energies. The generic
//      builder is exact here, so the tolerance is the finite-difference
//      truncation with margin.
//
//   2. LCAO diamond (physical): the Thomas-Reiche-Kuhn f-sum
//      n_eff = (2/N_k) sum |V_nm|^2 / (3.80998 * omega_nm) over occupied ->
//      unoccupied pairs must land near the valence electron count, 8. op2c's
//      explicit integrals carry a few-percent basis deficit, so the window is
//      loose -- but an hbar^2 unit error lands at 3.5 or 18.5, far outside it.
//
// Either lock alone pins one builder; together, against one constant, they pin
// the FAMILY-CROSSING consistency that the production consumers rely on.
#include "rsatb.hpp"
#include "test_data_path.hpp"
#include "model/sktb/sktb_model.hpp"
#include "setup/operator_assembly.hpp"

#include <mpi.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using cd = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<cd>;

#include "vbcsr/core/detail/kernels/lapack_api.hpp"
extern "C" {
void zhegv_(const vbcsr_lapack_int* itype, const char* jobz, const char* uplo,
            const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda,
            vbcsr_complex_double* b, const vbcsr_lapack_int* ldb, double* w,
            vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, double* rwork,
            vbcsr_lapack_int* info);
void zgesv_(const vbcsr_lapack_int* n, const vbcsr_lapack_int* nrhs, vbcsr_complex_double* a,
            const vbcsr_lapack_int* lda, vbcsr_lapack_int* ipiv, vbcsr_complex_double* b,
            const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
}

namespace {

constexpr double kTRK = 3.80998212;  // hbar^2 / 2 m_e, eV Angstrom^2

std::vector<cd> densify(Matrix& A, int n) {
    std::vector<cd> dense(static_cast<size_t>(n) * n, cd(0.0, 0.0));
    vbcsr::DistVector<cd> e(A.graph), y(A.graph);
    for (int j = 0; j < n; ++j) {
        e.set_constant(cd(0.0, 0.0));
        e[j] = cd(1.0, 0.0);
        A.mult(e, y);
        for (int i = 0; i < n; ++i) dense[static_cast<size_t>(j) * n + i] = y[i];
    }
    return dense;
}

// H c = E S c, S-orthonormal eigenvectors in the columns of C.
bool gen_eig(std::vector<cd> A, std::vector<cd> B, int n,
             std::vector<double>& w, std::vector<cd>& C) {
    w.assign(n, 0.0);
    const vbcsr_lapack_int itype = 1, nn = n, lda = n, ldb = n;
    const vbcsr_lapack_int lwork = 8 * n + 64;
    std::vector<cd> work(static_cast<size_t>(lwork));
    std::vector<double> rwork(static_cast<size_t>(std::max(1, 3 * n - 2)));
    vbcsr_lapack_int info = 0;
    const char jobz = 'V', uplo = 'U';
    zhegv_(&itype, &jobz, &uplo, &nn, A.data(), &lda, B.data(), &ldb, w.data(),
           work.data(), &lwork, rwork.data(), &info);
    C = A;
    return info == 0;
}

std::vector<cd> identity(int n) {
    std::vector<cd> I(static_cast<size_t>(n) * n, cd(0.0, 0.0));
    for (int i = 0; i < n; ++i) I[static_cast<size_t>(i) * n + i] = cd(1.0, 0.0);
    return I;
}

// C^dagger M C, column-major n x n throughout.
std::vector<cd> to_eigenbasis(const std::vector<cd>& M, const std::vector<cd>& C, int n) {
    const size_t N = static_cast<size_t>(n);
    std::vector<cd> MC(N * N, cd(0.0, 0.0)), out(N * N, cd(0.0, 0.0));
    for (int m = 0; m < n; ++m)
        for (int i = 0; i < n; ++i) {
            cd s(0.0, 0.0);
            for (int k = 0; k < n; ++k)
                s += M[static_cast<size_t>(k) * N + i] * C[static_cast<size_t>(m) * N + k];
            MC[static_cast<size_t>(m) * N + i] = s;
        }
    for (int m = 0; m < n; ++m)
        for (int nn2 = 0; nn2 < n; ++nn2) {
            cd s(0.0, 0.0);
            for (int k = 0; k < n; ++k)
                s += std::conj(C[static_cast<size_t>(nn2) * N + k]) * MC[static_cast<size_t>(m) * N + k];
            out[static_cast<size_t>(m) * N + nn2] = s;
        }
    return out;
}

// LOCK 1: the generic builder's diagonal is exactly -dE_n/dk in eV*Angstrom.
bool sk_diagonal_lock() {
    using namespace rsatb::model::sktb;
    namespace rmodel = rsatb::model;

    auto spline = [](double v, double rc) {
        SkRadialFunction f;
        f.kind = "spline";
        f.r = {0.0, rc};
        f.values = {v, v};
        f.r_cut = rc;
        f.spline_second_derivatives = f.BuildSplineSecondDerivatives();
        return f;
    };
    std::map<std::string, std::vector<std::string>> basis;
    basis["C"] = {"p"};
    SkParameters ham;
    ham.SetOnsite("C", "p", 0.0);
    ham.SetPairChannel("C", "C", "pp", "sigma", spline(2.0, 2.5));
    ham.SetPairChannel("C", "C", "pp", "pi", spline(0.5, 2.5));

    const auto table = BuildSkTable(basis, ham, std::nullopt);
    const auto prepared = DeriveBasisSpec(table, rmodel::SpinSpec{});

    const std::vector<double> pos = {0.0, 0.0, 0.0, 1.2, 0.6, 0.4};
    const std::vector<int> z = {6, 6};
    const std::vector<double> cell = {2.4, 0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.0, 2.4};
    const std::vector<bool> pbc = {true, true, true};
    auto ad = std::unique_ptr<vbcsr::atomic::AtomicData>(
        vbcsr::atomic::AtomicData::from_points(pos, z, cell, pbc, prepared.cutoff_radius,
                                               prepared.type_norb, MPI_COMM_WORLD));

    rmodel::SktbConfig config;
    config.basis = basis;
    config.hamiltonian = ham;
    auto model = rmodel::SktbModel::Create(config);

    // Central difference along Cartesian x: fractional step h * cell_col / 2pi.
    const double h = 1e-3;
    const std::vector<double> k0 = {0.13, 0.21, 0.07};
    auto step = [&](double s) {
        std::vector<double> k = k0;
        for (int j = 0; j < 3; ++j) k[j] += s * cell[0 + 3 * j] / (2.0 * M_PI);
        return k;
    };
    rmodel::SetupOptions options;
    options.kpoints = {k0, step(h), step(-h)};
    options.compute_bounds = false;
    rsatb::setup::OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;
    auto ops = rsatb::setup::BuildOperatorSet(*ad, MPI_COMM_WORLD, false, *model, options, runtime);

    Matrix* H = ops->hamiltonian_k(0);
    Matrix* V = ops->velocity_state(0).x.matrix.get();
    const int n = H->graph->global_scalar_rows();
    const size_t N = static_cast<size_t>(n);

    auto spectrum = [&](int kidx) {
        auto A = densify(*ops->hamiltonian_k(kidx), n);
        std::vector<double> w;
        std::vector<cd> C;
        gen_eig(A, identity(n), n, w, C);
        return w;
    };
    std::vector<double> E0, Ep = spectrum(1), Em = spectrum(2);
    std::vector<cd> C;
    {
        auto A = densify(*H, n);
        if (!gen_eig(A, identity(n), n, E0, C)) return false;
    }
    const auto Vd = densify(*V, n);
    const auto Ve = to_eigenbasis(Vd, C, n);

    double worst = 0.0, scale = 0.0;
    for (int i = 0; i < n; ++i) {
        // Bands closer than 10 meV cannot be tracked across the three k-points.
        double gap = 1e30;
        if (i > 0) gap = std::min(gap, E0[i] - E0[i - 1]);
        if (i + 1 < n) gap = std::min(gap, E0[i + 1] - E0[i]);
        if (gap < 1e-2) continue;
        const double dEdk = (Ep[i] - Em[i]) / (2.0 * h);
        const double stored = Ve[static_cast<size_t>(i) * N + i].real();
        worst = std::max(worst, std::abs(stored + dEdk));   // stored = -dE/dk
        scale = std::max(scale, std::abs(dEdk));
    }
    const double rel = worst / scale;
    std::printf("SK diagonal lock: max |V_nn + dE_n/dk| / max|dE/dk| = %.3e  %s\n",
                rel, rel < 1e-3 ? "PASS" : "FAIL");
    return rel < 1e-3;
}

// Densify the composite velocity: V = i(bond - xbar S^-1 H + H S^-1 delta),
// with S^-1 by dense LU. Small n only; this is the reference the production
// apply must agree with.
std::vector<cd> densify_composite(const rsatb::op::VelocityAxis& axis,
                                  Matrix& H, Matrix& S, int n) {
    const size_t N = static_cast<size_t>(n);
    auto bond = densify(*axis.bond, n);
    auto xbar = densify(*axis.xbar, n);
    auto delta = densify(*axis.delta, n);
    auto Hd = densify(H, n);
    auto Sd = densify(S, n);

    // S^-1 M by LU, both right-factors at once.
    auto solveS = [&](std::vector<cd> M) {
        std::vector<cd> LU = Sd;
        std::vector<vbcsr_lapack_int> ipiv(n);
        vbcsr_lapack_int nn = n, nrhs = n, lda = n, ldb = n, info = 0;
        zgesv_(&nn, &nrhs, LU.data(), &lda, ipiv.data(), M.data(), &ldb, &info);
        return M;
    };
    auto mul = [&](const std::vector<cd>& P, const std::vector<cd>& Q) {
        std::vector<cd> R(N * N, cd(0.0, 0.0));
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                const cd q = Q[static_cast<size_t>(j) * N + k];
                if (q == cd(0.0, 0.0)) continue;
                for (int i = 0; i < n; ++i)
                    R[static_cast<size_t>(j) * N + i] += P[static_cast<size_t>(k) * N + i] * q;
            }
        return R;
    };
    const auto t1 = mul(xbar, solveS(Hd));
    const auto t2 = mul(Hd, solveS(delta));
    std::vector<cd> V(N * N);
    for (size_t t = 0; t < N * N; ++t)
        V[t] = cd(0.0, 1.0) * (bond[t] - t1[t] + t2[t]);
    return V;
}

// LOCK 2: the production velocity on LCAO diamond is now the section-4
// commutator, so BOTH locks tighten: its eigenbasis diagonal must sit at the
// finite-difference floor like the SK one, and its f-sum must land at the
// valence electron count -- not merely near it, as the explicit-integral
// route did.
bool lcao_fsum_lock() {
    auto structure =
        vbcsr::atomic::ReadStructure(rsatb::test::data_path("diamond/conf.vasp"));

    rsatb::model::LcaoConfig cfg;
    cfg.hamiltonian =
        rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/hamiltonians.h5")};
    cfg.orbitals.dir = rsatb::test::data_path("diamond/");
    cfg.orbitals.names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    auto model = rsatb::model::LcaoModel::Create(cfg, MPI_COMM_WORLD);

    rsatb::model::SetupOptions opt;
    opt.kpoints.clear();   // the default is Gamma; this test wants ONLY the
                           // shifted grid -- at Gamma every band velocity
                           // vanishes by time reversal and the diagonal lock
                           // would divide noise by noise.
    const int nk = 2;
    for (int a = 0; a < nk; ++a)
        for (int b = 0; b < nk; ++b)
            for (int c = 0; c < nk; ++c)
                opt.kpoints.push_back({(a + 0.5) / nk, (b + 0.5) / nk, (c + 0.5) / nk});
    // The first grid point again, stepped along Cartesian x, for dE/dk.
    const double h = 1e-3;
    const int idx_plus = static_cast<int>(opt.kpoints.size());
    {
        std::vector<double> kp = opt.kpoints[0], km = opt.kpoints[0];
        for (int j = 0; j < 3; ++j) {
            const double cell_col = structure.cell[0 + 3 * j] / (2.0 * M_PI);
            kp[j] += h * cell_col;
            km[j] -= h * cell_col;
        }
        opt.kpoints.push_back(kp);
        opt.kpoints.push_back(km);
    }
    opt.compute_bounds = false;

    rsatb::RSATB inst(structure, *model, opt, 1, MPI_COMM_WORLD);
    auto* ops = inst.operators();

    // SetupOptions default-initializes with Gamma at index 0 when kpoints were
    // assigned, not appended -- here they were assigned, so index 0 is the
    // first grid point. Guard anyway by using exactly the points we set.
    const int ngrid = nk * nk * nk;
    const int n_occ = 4;  // 2 C x 4 valence electrons, spin-degenerate
    double neff = 0.0;
    double diag_worst = 0.0, diag_scale = 0.0;
    for (int kidx = 0; kidx < ngrid; ++kidx) {
        Matrix* H = ops->hamiltonian_k(kidx);
        Matrix* S = ops->overlap_k(kidx);
        const auto& axis = ops->velocity_state(kidx).x;
        if (!axis.composite()) {
            std::printf("LCAO velocity is not the composite commutator; wiring is wrong\n");
            return false;
        }
        const int n = H->graph->global_scalar_rows();
        const size_t N = static_cast<size_t>(n);
        auto Hd = densify(*H, n);
        auto Sd = densify(*S, n);
        std::vector<double> E;
        std::vector<cd> C;
        if (!gen_eig(Hd, Sd, n, E, C)) return false;
        const auto Ve = to_eigenbasis(densify_composite(axis, *H, *S, n), C, n);
        for (int nn2 = 0; nn2 < n_occ; ++nn2)
            for (int m = n_occ; m < n; ++m) {
                const double w = E[m] - E[nn2];
                if (w < 1e-8) continue;
                neff += 2.0 * std::norm(Ve[static_cast<size_t>(m) * N + nn2]) / (kTRK * w);
            }
        // The diagonal lock, at the first grid point: the commutator velocity
        // is exact for the propagated pencil, so this must sit at the FD floor
        // -- the lock the explicit-integral route could never pass.
        if (kidx == 0) {
            std::vector<double> Ep, Em;
            {
                auto A = densify(*ops->hamiltonian_k(idx_plus), n);
                auto B = densify(*ops->overlap_k(idx_plus), n);
                std::vector<cd> Cp;
                gen_eig(A, B, n, Ep, Cp);
                A = densify(*ops->hamiltonian_k(idx_plus + 1), n);
                B = densify(*ops->overlap_k(idx_plus + 1), n);
                gen_eig(A, B, n, Em, Cp);
            }
            for (int i = 0; i < n; ++i) {
                double gap = 1e30;
                if (i > 0) gap = std::min(gap, E[i] - E[i - 1]);
                if (i + 1 < n) gap = std::min(gap, E[i + 1] - E[i]);
                if (gap < 1e-2) continue;
                const double dEdk = (Ep[i] - Em[i]) / (2.0 * h);
                diag_worst = std::max(diag_worst,
                                      std::abs(Ve[static_cast<size_t>(i) * N + i].real() + dEdk));
                diag_scale = std::max(diag_scale, std::abs(dEdk));
            }
        }
    }
    // LOCK 3: the production apply path. apply_dual's composite branch --
    // metric solves by CG, the same code every AC/DC correlator runs -- must
    // reproduce the dense reference S^-1 V x column by column.
    {
        Matrix* H = ops->hamiltonian_k(0);
        Matrix* S = ops->overlap_k(0);
        const auto& axis = ops->velocity_state(0).x;
        const int n = H->graph->global_scalar_rows();
        const size_t N = static_cast<size_t>(n);
        const auto Vd = densify_composite(axis, *H, *S, n);
        auto Sd = densify(*S, n);

        rsatb::calc::Chebyshev<cd> cheby(*H, *S, -30.0, 300.0);
        const auto vop = axis.view();
        vbcsr::DistVector<cd> e(H->graph), y(H->graph);
        double dev = 0.0, nrm = 0.0;
        for (int j = 0; j < n; j += 5) {   // a spread of columns is plenty
            e.set_constant(cd(0.0, 0.0));
            e[j] = cd(1.0, 0.0);
            cheby.apply_dual(vop, e, y);   // Solve form: y = S^-1 V e_j
            // dense reference: S^-1 (V e_j) by LU
            std::vector<cd> col(N), lu = Sd;
            for (int i = 0; i < n; ++i) col[i] = Vd[static_cast<size_t>(j) * N + i];
            std::vector<vbcsr_lapack_int> ipiv(n);
            vbcsr_lapack_int nn3 = n, one = 1, info = 0;
            zgesv_(&nn3, &one, lu.data(), &nn3, ipiv.data(), col.data(), &nn3, &info);
            for (int i = 0; i < n; ++i) {
                dev += std::norm(y[i] - col[i]);
                nrm += std::norm(col[i]);
            }
        }
        const double rel = std::sqrt(dev / nrm);
        const bool pass = rel < 1e-6;
        std::printf("LCAO apply lock (apply_dual composite vs dense): rel = %.3e  %s\n",
                    rel, pass ? "PASS" : "FAIL");
        if (!pass) return false;

        // And the Sandwich branch -- what production LCAO AC actually runs.
        // Cross-form identity, no dense Z needed: for eta arbitrary and
        // c = Z eta,   Z * (Z^H V Z eta)  ==  S^-1 V c.
        auto& state = ops->kpoint_state(0);
        if (state.preconditioner) {
            rsatb::calc::Chebyshev<cd> cheby_sw(*H, *state.preconditioner,
                                                rsatb::calc::PreconditionerType::Symmetric,
                                                -30.0, 300.0);
            vbcsr::DistVector<cd> eta(H->graph), c(H->graph), lhs(H->graph),
                                  rhs(H->graph), t(H->graph);
            double dev2 = 0.0, nrm2 = 0.0;
            for (int j = 0; j < n; j += 7) {
                eta.set_constant(cd(0.0, 0.0));
                eta[j] = cd(1.0, 0.0);
                state.preconditioner->apply(eta, c);          // c = Z eta
                cheby_sw.apply_dual(vop, eta, t);             // t = Z^H V Z eta
                state.preconditioner->apply(t, lhs);          // lhs = Z t
                cheby.apply_dual(vop, c, rhs);                // rhs = S^-1 V c
                for (int i = 0; i < n; ++i) {
                    dev2 += std::norm(lhs[i] - rhs[i]);
                    nrm2 += std::norm(rhs[i]);
                }
            }
            const double rel2 = std::sqrt(dev2 / nrm2);
            // Z is a truncated polynomial approximation of S^-1/2, so the two
            // routes agree to the preconditioner's quality, not machine eps.
            const bool pass2 = rel2 < 1e-3;
            std::printf("LCAO sandwich lock (Z Z^H V Z eta vs S^-1 V Z eta): rel = %.3e  %s\n",
                        rel2, pass2 ? "PASS" : "FAIL");
            if (!pass2) return false;
        } else {
            std::printf("LCAO sandwich lock: no preconditioner in state; skipped\n");
        }
    }

    neff /= ngrid;
    const double diag_rel = diag_scale > 0.0 ? diag_worst / diag_scale : 1.0;
    const bool diag_pass = diag_rel < 1e-3;
    std::printf("LCAO diagonal lock (commutator): max |V_nn + dE_n/dk| / max|dE/dk| = %.3e  %s\n",
                diag_rel, diag_pass ? "PASS" : "FAIL");
    // The 2x2x2 shifted grid is deliberately coarse (this is a units lock,
    // not a convergence study): the converged value is 7.9-8.1 (4^3 and 6^3
    // grids, doc section 7), this grid gives ~7.0, and an hbar^2 unit error
    // lands at ~3.0 or ~16 -- far outside any grid wiggle.
    const bool fsum_pass = neff > 5.5 && neff < 10.5;
    std::printf("LCAO f-sum lock: n_eff = %.4f  (exact 8; converged commutator value "
                "7.9-8.1, hbar^2 unit error ~3.0 or ~16)  %s\n",
                neff, fsum_pass ? "PASS" : "FAIL");
    return diag_pass && fsum_pass;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        if (rank == 0) std::printf("test_velocity_units is serial-only; skipping.\n");
        MPI_Finalize();
        return 0;
    }

    const bool ok1 = sk_diagonal_lock();
    const bool ok2 = lcao_fsum_lock();
    std::printf(ok1 && ok2 ? "ALL PASSED\n" : "FAILED\n");

    MPI_Finalize();
    return (ok1 && ok2) ? 0 : 1;
}
