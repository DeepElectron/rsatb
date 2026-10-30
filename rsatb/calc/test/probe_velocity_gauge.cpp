// TEMPORARY diagnostic: which velocity operator?  (doc/position_commutator.md §7)
//
// Two routes exist for the same physical operator:
//   (a) explicit integrals -- op2c's gradient part plus the 3-body beta-projector
//       commutator [V_NL, x]. Needs pseudopotentials.
//   (b) the commutator route, which in the eigenbasis of the finite-basis pencil
//       is exactly (i/hbar)(E_n - E_m) <n|X|m>.
//
// Diagonalising the SAME pencil cannot arbitrate between them: each Chebyshev run
// converges to its own reference, because
//     <n|[H,x]|m> = (E_n - E_m) <n|X|m>
// holds exactly in a finite basis. What DOES arbitrate, from one diagonalisation:
//
//   1. r_nm = v_a(n,m) / [ (E_n - E_m) x_nm ].  Were the routes equal this would
//      be the constant i/hbar for every pair; its SPREAD is the basis-
//      incompleteness error, and it is free of unit conventions.
//   2. The Thomas-Reiche-Kuhn sum rule sum_m (E_m - E_n)|x_nm|^2 = hbar^2/2m_e
//      = 3.80998 eV A^2, evaluated from each route. An absolute check against
//      physics rather than against the other route.
//
// FIRST ATTEMPT, and why it failed -- kept because the failure is the point.
// Evaluating route (b) as (E_n-E_m)<n|X|m> with X = position_state().x.matrix
// gave TRK sums of 1e5..1e9 against a target of 3.81. That shortcut needs the
// TRUE position operator, and under PBC the two-centre matrix sum_R X(R)e^{ikR}
// is NOT it: the i d/dk content is missing, precisely the first-lattice-moment
// term. Route (b) has to be built from the formula instead,
//     C = -S0^-1 (H1 - S1 A0) + [ S0^-1 (X0 - S1), A0 ],
// which is what this file now does, at Gamma where that formula is stated.
#include "rsatb.hpp"
#include "test_data_path.hpp"

#include <fstream>
#include "operator/lattice_moment.hpp"
#include "model/sktb/sktb_model.hpp"
#include "setup/operator_assembly.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include <optional>
#include <vector>

using cd = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<cd>;

// Same convention as calc/lapack_isqrt.hpp: vbcsr's int width and
// std::complex<double>, so this does not collide with MKL's own prototype.
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

constexpr double kTRK = 3.80998212;  // hbar^2 / 2 m_e, in eV * Angstrom^2
constexpr double kHbar = 0.6582119569;   // eV fs

// Column j of A from A e_j. n is tiny here and this needs no sparsity knowledge.
std::vector<cd> densify(Matrix& A, int n) {
    std::vector<cd> dense(static_cast<size_t>(n) * n, cd(0.0, 0.0));
    vbcsr::DistVector<cd> e(A.graph), y(A.graph);
    for (int j = 0; j < n; ++j) {
        e.set_constant(cd(0.0, 0.0));
        e[j] = cd(1.0, 0.0);
        A.mult(e, y);
        for (int i = 0; i < n; ++i) dense[static_cast<size_t>(j) * n + i] = y[i];  // col-major
    }
    return dense;
}

// out(n,m) = C_n^dagger M C_m, C column-major n x n.
std::vector<cd> to_eigenbasis(const std::vector<cd>& M, const std::vector<cd>& C, int n) {
    const size_t N = static_cast<size_t>(n);
    std::vector<cd> MC(N * N, cd(0.0, 0.0)), out(N * N, cd(0.0, 0.0));
    for (int m = 0; m < n; ++m)
        for (int i = 0; i < n; ++i) {
            cd s(0.0, 0.0);
            for (int k = 0; k < n; ++k) s += M[static_cast<size_t>(k) * N + i] * C[static_cast<size_t>(m) * N + k];
            MC[static_cast<size_t>(m) * N + i] = s;
        }
    for (int m = 0; m < n; ++m)
        for (int nn = 0; nn < n; ++nn) {
            cd s(0.0, 0.0);
            for (int k = 0; k < n; ++k) s += std::conj(C[static_cast<size_t>(nn) * N + k]) * MC[static_cast<size_t>(m) * N + k];
            out[static_cast<size_t>(m) * N + nn] = s;
        }
    return out;
}

}  // namespace

// The comparison, shared by both models. S may be null (orthogonal basis), in
// which case S0 = I and S1 = 0. X0 may come as a full matrix (non-orthogonal)
// or as a diagonal (orthogonal).
void compare_routes(const char* title,
                    Matrix* H, Matrix* S, Matrix* Vx, Matrix* X0mat,
                    vbcsr::DistVector<cd>* X0diag,
                    Matrix* H1m, Matrix* S1m, int n) {
    const size_t N = static_cast<size_t>(n);
    auto eye = [&]() {
        std::vector<cd> I(N * N, cd(0.0, 0.0));
        for (int i = 0; i < n; ++i) I[static_cast<size_t>(i) * N + i] = cd(1.0, 0.0);
        return I;
    };
    auto H0 = densify(*H, n);
    auto S0 = S ? densify(*S, n) : eye();
    auto Vd = densify(*Vx, n);
    auto H1 = densify(*H1m, n);
    auto S1 = S1m ? densify(*S1m, n) : std::vector<cd>(N * N, cd(0.0, 0.0));
    std::vector<cd> X0(N * N, cd(0.0, 0.0));
    if (X0mat) X0 = densify(*X0mat, n);
    else for (int i = 0; i < n; ++i) X0[static_cast<size_t>(i) * N + i] = (*X0diag)[i];

    auto solveS = [&](const std::vector<cd>& M) {
        std::vector<cd> B = M, LU = S0;
        std::vector<vbcsr_lapack_int> ipiv(n);
        vbcsr_lapack_int nn = n, nrhs = n, lda = n, ldb = n, info = 0;
        zgesv_(&nn, &nrhs, LU.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
        if (info != 0) std::printf("  zgesv info=%lld\n", (long long)info);
        return B;
    };
    auto mul = [&](const std::vector<cd>& P, const std::vector<cd>& Q) {
        std::vector<cd> R(N * N, cd(0.0, 0.0));
        for (int j = 0; j < n; ++j)
            for (int k2 = 0; k2 < n; ++k2) {
                const cd q = Q[static_cast<size_t>(j) * N + k2];
                if (q == cd(0.0, 0.0)) continue;
                for (int i = 0; i < n; ++i)
                    R[static_cast<size_t>(j) * N + i] += P[static_cast<size_t>(k2) * N + i] * q;
            }
        return R;
    };

    const auto A0 = solveS(H0);
    std::vector<cd> t(N * N);
    { const auto sa = mul(S1, A0); for (size_t i = 0; i < N * N; ++i) t[i] = H1[i] - sa[i]; }
    const auto cell_term = solveS(t);                       // S0^-1 (H1 - S1 A0)
    std::vector<cd> D0(N * N);
    for (size_t i = 0; i < N * N; ++i) D0[i] = X0[i] - S1[i];
    const auto SD0 = solveS(D0);
    const auto c1 = mul(SD0, A0), c2 = mul(A0, SD0);
    std::vector<cd> C(N * N);
    for (size_t i = 0; i < N * N; ++i) C[i] = -cell_term[i] + (c1[i] - c2[i]);
    const auto SC = mul(S0, C);

    auto nrm = [&](const std::vector<cd>& M) {
        double x = 0.0; for (auto& v : M) x += std::norm(v); return std::sqrt(x);
    };
    cd num(0.0, 0.0); double den = 0.0, tn = 0.0;
    for (size_t i = 0; i < N * N; ++i) {
        num += std::conj(SC[i]) * Vd[i];
        den += std::norm(SC[i]);
        tn  += std::norm(Vd[i]);
    }
    const cd kappa = (den > 0) ? num / den : cd(0.0, 0.0);
    double res = 0.0;
    for (size_t i = 0; i < N * N; ++i) res += std::norm(Vd[i] - kappa * SC[i]);
    const double rel = (tn > 0) ? std::sqrt(res / tn) : 0.0;

    std::printf("%s\n", title);
    std::printf("  n=%d  ||v_a||=%.4e  ||S0*C||=%.4e  ||H1||=%.4e  ||S1||=%.4e  ||X0||=%.4e\n",
                n, nrm(Vd), nrm(SC), nrm(H1), nrm(S1), nrm(X0));
    std::printf("  fitted kappa = (%.6f, %.6f)\n", kappa.real(), kappa.imag());
    std::printf("  RESIDUAL |v_a - kappa*S0*C| / |v_a| = %.6e   %s\n\n", rel,
                rel < 1e-8 ? "<-- routes AGREE" : "<-- routes DISAGREE");
}

namespace {

// Defined further down, next to the other eigenbasis diagnostics.
void time_reversal_at_gamma(const char* label, Matrix* Hm, Matrix* Sm, Matrix* Vm, int n);

// A PERIODIC orthogonal Slater-Koster carbon-p model. Periodic on purpose: with
// pbc off there are no lattice images, H1 vanishes identically, and the cell
// term -- the half of the formula most likely to carry a sign error -- would
// never be exercised. Two inequivalent sites so the dipole term is non-zero too.
void validate_on_orthogonal_sk() {
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
    const double rc = 2.5;
    std::map<std::string, std::vector<std::string>> basis;
    basis["C"] = {"p"};
    SkParameters ham;
    ham.SetOnsite("C", "p", 0.0);
    ham.SetPairChannel("C", "C", "pp", "sigma", spline(2.0, rc));
    ham.SetPairChannel("C", "C", "pp", "pi", spline(0.5, rc));

    const auto table = BuildSkTable(basis, ham, std::nullopt);
    const auto prepared = DeriveBasisSpec(table, rmodel::SpinSpec{});

    // a = 2.4 A cubic: self-images at 2.4 < rc give lattice moments along all
    // three axes, and the two sites sit at different positions.
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

    rmodel::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.compute_bounds = false;
    rsatb::setup::OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;
    auto ops = rsatb::setup::BuildOperatorSet(*ad, MPI_COMM_WORLD, false, *model, options, runtime);

    auto img_H_held = model->BuildHamiltonianImages(*ad);
    auto* img_H = &img_H_held.real();
    auto H1m = rsatb::op::BuildLatticeMomentAxis(*ad, *img_H, 0);

    Matrix* H = ops->hamiltonian_k(0);
    Matrix* Vx = ops->velocity_state(0).x.matrix.get();
    auto* Xdiag = ops->position_state(0).x.orthogonal_diagonal.get();
    Matrix* Xmat = ops->position_state(0).x.matrix.get();
    if (!H || !Vx || (!Xdiag && !Xmat)) {
        std::printf("orthogonal SK: operator missing (H=%p V=%p Xdiag=%p Xmat=%p)\n",
                    (void*)H, (void*)Vx, (void*)Xdiag, (void*)Xmat);
        return;
    }
    if (ops->has_overlap()) { std::printf("orthogonal SK: unexpected overlap\n"); return; }

    time_reversal_at_gamma("orthogonal SK, route (a) = bond-vector", H, nullptr, Vx,
                           H->graph->global_scalar_rows());
    compare_routes("VALIDATION -- periodic orthogonal SK (C p-only, a=2.4 A, 2 sites)\n"
                   "  S = I, so route (b) reduces to -H1 + [diag(r), H0], which IS the\n"
                   "  bond-vector velocity route (a) builds. These must agree exactly.",
                   H, nullptr, Vx, Xmat, Xdiag, H1m.get(), nullptr,
                   H->graph->global_scalar_rows());
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------
// THE ABSOLUTE ARBITER: does the velocity reproduce the band velocity?
//
// Comparing route (a) against route (b) can only ever say that they differ, not
// which is right. There IS a neutral referee, and it is not diagonalisation of
// a reference Hamiltonian -- it is the Hellmann-Feynman theorem for the pencil
// actually being propagated. For (H(k) - E S(k)) c = 0 with c^dagger S c = 1,
//
//     dE_n/dk_alpha = c_n^dagger ( dH/dk_alpha - E_n dS/dk_alpha ) c_n
//
// holds EXACTLY -- it is variational, and assumes nothing about basis
// completeness. With the convention H(k) = sum_R H(R) e^{-i k.R} that is
// -i (c^dagger H1 c - E c^dagger S1 c), which is precisely the DIAGONAL of the
// section-4 operator: the gauge/dipole term [S0^-1 (X0 - S1), A0] cancels
// between its two orderings the moment A0 acts on an eigenvector,
//
//     c^dagger S0 D A0 c - c^dagger S0 A0 D c = E c^dagger S0 D c - E c^dagger S0 D c = 0,
//
// so the band velocity probes the cell term alone and needs no position
// operator, no gauge choice and no pseudopotential.
//
// The point: dE_n/dk is a property of the Hamiltonian being propagated, and it
// is obtainable from H(k), S(k) alone by finite differences. A velocity
// operator paired with that Hamiltonian MUST reproduce it, or the current it
// generates is not the current of the dynamics it is inserted into. That is an
// absolute standard, not a preference between two constructions.

// Fractional k displaced by `h` along CARTESIAN axis `axis`.
//
// The convention is exp(-2 pi i k_frac . R_int) = exp(-i k_cart . R_cart) with
// k_cart = 2 pi L^-T k_frac and L the cell matrix holding lattice vectors in
// its columns (cell[axis + 3j] = a_j[axis]). Inverting, a cartesian step h
// along `axis` is the fractional step h * L[axis][j] / 2 pi. Diamond's
// primitive cell is not cubic, so this cannot be shortcut.
std::vector<double> k_step(const std::vector<double>& cell, const std::vector<double>& k,
                           int axis, double h) {
    std::vector<double> out = k;
    for (int j = 0; j < 3; ++j) out[j] += h * cell[axis + 3 * j] / (2.0 * M_PI);
    return out;
}

// D[A]_alpha = sum_R (R_alpha + r_j,alpha - r_i,alpha) A(R) e^{-2 pi i k.(R + r_j - r_i)}
//
// This is i * dA(k)/dk_alpha in the ATOMIC gauge, which is the gauge
// operator_set.hpp samples H and S in (PhaseConvention::R_AND_POSITION). The
// lattice-only first moment of operator/lattice_moment.hpp is the same object
// in the OTHER gauge (R_ONLY); the two agree at Gamma and nowhere else, which
// is exactly what made the first run of this test fail -- the Gamma comparison
// higher up could never have caught it. Structurally a copy of
// BuildCommutatorVelocity, minus the i/hbar and for one axis, so that D[S] --
// which has no builder anywhere -- is obtained the same way as D[H].
std::unique_ptr<Matrix> bond_moment_at_k(vbcsr::atomic::AtomicData& ad,
                                         vbcsr::atomic::ImageContainer<double>& images,
                                         int axis, const std::vector<double>& kpoint) {
    auto out = std::make_unique<Matrix>(ad.get_graph());
    vbcsr::DistGraph* ref = ad.get_graph();
    const int n_owned = static_cast<int>(ref->owned_global_indices.size());

    for (const auto& entry : images.image_blocks) {
        const auto& shift = entry.first;
        auto* im = entry.second;
        auto* ig = im->graph;
        const cd image_phase = rsatb::op::BlochPhase(shift[0], shift[1], shift[2],
                                                     kpoint[0], kpoint[1], kpoint[2]);
        const double shift_alpha = rsatb::op::CellShiftComponent(ad, shift, axis);

        const int n_image_columns = static_cast<int>(ig->block_sizes.size());
        std::vector<int> column_map(n_image_columns, -1);
        for (int lc = 0; lc < n_image_columns; ++lc)
            column_map[lc] = ref->global_to_local.at(ig->get_global_index(lc));

        for (int lr = 0; lr < n_owned; ++lr) {
            double ri[3];
            ad.get_pos(lr, &ri[0], &ri[1], &ri[2]);
            const int start = im->row_ptr()[lr], end = im->row_ptr()[lr + 1];
            const int row_dim = ig->block_sizes[lr];
            for (int kk = start; kk < end; ++kk) {
                const int ic = im->col_ind()[kk];
                const int rc = column_map[ic];
                const int col_dim = ig->block_sizes[ic];
                double rj[3];
                ad.get_pos(rc, &rj[0], &rj[1], &rj[2]);
                double d[3] = {rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]};
                const double displacement = shift_alpha + d[axis];
                // invert_cell mutates in place, so the cartesian displacement
                // above has to be taken first.
                ad.invert_cell(&d[0], &d[1], &d[2]);
                if (std::abs(displacement) <= 1e-12) continue;
                const cd phase = image_phase * rsatb::op::BlochPhase(d[0], d[1], d[2],
                                                                     kpoint[0], kpoint[1], kpoint[2]);
                const int n_elem = static_cast<int>(im->block_size_elements(kk));
                const auto block = rsatb::op::ScaleBlock(im->block_data(kk), n_elem,
                                                         phase * cd(displacement, 0.0));
                out->update_local_block(lr, rc, block.data(), row_dim, col_dim,
                                        vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
            }
        }
    }
    out->assemble();
    return out;
}

// u_i = exp(-2 pi i k . f_i), the diagonal unitary relating the two Bloch
// gauges: H_atomic = U^dagger H_lattice U.
//
// operator_set.hpp samples H and S with PhaseConvention::R_AND_POSITION, but
// op2c phases its velocity through op::AddLocalBlock, which uses the lattice
// shift alone (block_assembly.hpp: BlochPhase(rx, ry, rz, ...)). The two
// coincide at Gamma and differ everywhere else, so an expectation value taken
// with atomic-gauge eigenvectors against a lattice-gauge velocity is comparing
// two different objects. Correcting for it is a precondition for reading
// anything physical out of the comparison.
std::vector<cd> atomic_gauge(vbcsr::atomic::AtomicData& ad, vbcsr::DistGraph* g,
                             const std::vector<double>& k, int n) {
    std::vector<cd> u(static_cast<size_t>(n), cd(1.0, 0.0));
    const int nb = static_cast<int>(g->owned_global_indices.size());
    for (int b = 0; b < nb; ++b) {
        double r[3];
        ad.get_pos(b, &r[0], &r[1], &r[2]);
        ad.invert_cell(&r[0], &r[1], &r[2]);
        const cd ub = rsatb::op::BlochPhase(r[0], r[1], r[2], k[0], k[1], k[2]);
        const int off = g->block_offsets[b], sz = g->block_sizes[b];
        for (int s = 0; s < sz; ++s) u[static_cast<size_t>(off + s)] = ub;
    }
    return u;
}

std::vector<cd> identity(int n) {
    std::vector<cd> I(static_cast<size_t>(n) * n, cd(0.0, 0.0));
    for (int i = 0; i < n; ++i) I[static_cast<size_t>(i) * n + i] = cd(1.0, 0.0);
    return I;
}

// H c = E S c with S-orthonormal columns. info > n means S was not positive
// definite, which for a hand-built overlap is the failure worth seeing.
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
    if (info != 0) std::printf("  zhegv info=%lld\n", (long long)info);
    return info == 0;
}

std::vector<double> spectrum(Matrix* H, Matrix* S, int n) {
    auto A = densify(*H, n);
    auto B = S ? densify(*S, n) : identity(n);
    std::vector<double> w;
    std::vector<cd> C;
    if (!gen_eig(A, B, n, w, C)) w.assign(n, 0.0);
    return w;
}

// TIME REVERSAL AT GAMMA. Every band velocity vanishes at a time-reversal
// invariant momentum. That statement is exact and basis-independent -- no
// completeness assumption enters it, and it holds for the projected model just
// as for the continuum one, because H(R) real implies E(k) = E(-k) whatever the
// basis. So this separates the two candidate explanations for a route that
// misses dE/dk: a construction that violates it is defective, while one that
// satisfies it and still misses dE/dk away from Gamma is merely incomplete.
//
// Inside a degenerate multiplet the individual diagonal elements depend on the
// arbitrary basis LAPACK returns, so it is the subspace trace that must vanish.
void time_reversal_at_gamma(const char* label, Matrix* Hm, Matrix* Sm, Matrix* Vm, int n) {
    const size_t N = static_cast<size_t>(n);
    auto H0 = densify(*Hm, n);
    auto S0 = Sm ? densify(*Sm, n) : identity(n);
    auto Vd = densify(*Vm, n);
    std::vector<double> E;
    std::vector<cd> C;
    if (!gen_eig(H0, S0, n, E, C)) return;
    const auto va = to_eigenbasis(Vd, C, n);

    double vnorm = 0.0;
    for (const auto& x : Vd) vnorm += std::norm(x);
    vnorm = std::sqrt(vnorm / static_cast<double>(N * N));

    double worst = 0.0;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && E[j + 1] - E[i] < 1e-6) ++j;
        cd tr(0.0, 0.0);
        for (int b = i; b <= j; ++b) tr += va[static_cast<size_t>(b) * N + b];
        worst = std::max(worst, std::abs(tr) / static_cast<double>(j - i + 1));
        i = j + 1;
    }
    std::printf("  TR at Gamma (%s): max |subspace-mean <n|v|n>| = %.6e   rms|v| = %.4e   ratio %.3e\n",
                label, worst, vnorm, vnorm > 0.0 ? worst / vnorm : 0.0);
}

void band_velocity_test(const char* title,
                        Matrix* H0m, Matrix* S0m, Matrix* Vm,
                        Matrix* H1m, Matrix* S1m,
                        const std::vector<double>& Ep, const std::vector<double>& Em,
                        double h, int n, const std::vector<cd>* gauge = nullptr) {
    const size_t N = static_cast<size_t>(n);
    auto H0 = densify(*H0m, n);
    auto S0 = S0m ? densify(*S0m, n) : identity(n);
    auto H1 = densify(*H1m, n);
    auto S1 = S1m ? densify(*S1m, n) : std::vector<cd>(N * N, cd(0.0, 0.0));

    std::printf("%s\n", title);
    std::vector<double> E0;
    std::vector<cd> C;
    if (!gen_eig(H0, S0, n, E0, C)) { std::printf("  eigensolve failed\n\n"); return; }

    const auto h1 = to_eigenbasis(H1, C, n);
    const auto s1 = to_eigenbasis(S1, C, n);
    std::vector<cd> va;
    if (Vm) { auto Vd = densify(*Vm, n); va = to_eigenbasis(Vd, C, n); }

    // Central-difference reference, and a mask: two bands closer than 1 meV
    // cannot be tracked band-by-band across the three k-points.
    std::vector<double> dE(n, 0.0);
    std::vector<char> ok(n, 1);
    for (int i = 0; i < n; ++i) {
        dE[i] = (Ep[i] - Em[i]) / (2.0 * h);
        double gap = 1e30;
        if (i > 0) gap = std::min(gap, E0[i] - E0[i - 1]);
        if (i + 1 < n) gap = std::min(gap, E0[i + 1] - E0[i]);
        if (gap < 1e-2) ok[i] = 0;
    }
    int used = 0;
    double refnorm = 0.0;
    for (int i = 0; i < n; ++i) if (ok[i]) { ++used; refnorm += dE[i] * dE[i]; }
    refnorm = std::sqrt(refnorm);

    // route (b): the cell term alone -- the dipole term cancels on the diagonal.
    std::vector<cd> b(N * N, cd(0.0, 0.0));
    for (int i = 0; i < n; ++i)
        b[static_cast<size_t>(i) * N + i] =
            -(h1[static_cast<size_t>(i) * N + i] - cd(E0[i], 0.0) * s1[static_cast<size_t>(i) * N + i]);

    std::printf("  n=%d  bands used=%d  h=%.1e 1/A  ||dE/dk||=%.6e eV.A\n", n, used, h, refnorm);

    auto fit = [&](const std::vector<cd>& diag, const char* name) {
        cd num(0.0, 0.0);
        double den = 0.0, res = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!ok[i]) continue;
            const cd d = diag[static_cast<size_t>(i) * N + i];
            num += std::conj(d) * cd(dE[i], 0.0);
            den += std::norm(d);
        }
        const cd kap = den > 0.0 ? num / den : cd(0.0, 0.0);
        for (int i = 0; i < n; ++i) {
            if (!ok[i]) continue;
            res += std::norm(cd(dE[i], 0.0) - kap * diag[static_cast<size_t>(i) * N + i]);
        }
        const double rel = refnorm > 0.0 ? std::sqrt(res) / refnorm : 0.0;
        std::printf("    %-26s kappa=(% .6f,% .6f)  rel.err vs dE/dk = %.6e  %s\n",
                    name, kap.real(), kap.imag(), rel,
                    rel < 1e-4 ? "<== REPRODUCES the band velocity (to FD truncation)"
                               : "<== DOES NOT");
        return kap;
    };

    const cd kb = fit(b, "route (b) commutator");
    cd ka(0.0, 0.0);
    std::vector<cd> vg;
    if (Vm) {
        ka = fit(va, "route (a) as stored");
        if (gauge) {
            auto Vd = densify(*Vm, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    Vd[static_cast<size_t>(j) * N + i] =
                        std::conj((*gauge)[i]) * Vd[static_cast<size_t>(j) * N + i] * (*gauge)[j];
            vg = to_eigenbasis(Vd, C, n);
            ka = fit(vg, "route (a) gauge-corrected");
        }
    } else {
        std::printf("    %-26s unavailable for this model\n", "route (a) explicit integrals");
    }
    const std::vector<cd>& shown_a = vg.empty() ? va : vg;

    std::printf("    %-4s %14s %16s %16s\n", "band", "dE/dk [eV.A]", "route(b)", "route(a)");
    int shown = 0;
    for (int i = 0; i < n && shown < 8; ++i) {
        if (!ok[i]) continue;
        const cd pb = kb * b[static_cast<size_t>(i) * N + i];
        std::printf("    %-4d % 14.6f % 16.6f", i, dE[i], pb.real());
        if (Vm) { const cd pa = ka * shown_a[static_cast<size_t>(i) * N + i]; std::printf(" % 16.6f", pa.real()); }
        std::printf("\n");
        ++shown;
    }
    std::printf("\n");
}

}  // namespace

namespace {

constexpr int kAxis = 0;
constexpr double kStepH = 1e-3;                            // 1/Angstrom
const std::vector<double> kGeneralK = {0.13, 0.21, 0.07};  // low symmetry on purpose

// A Slater-Koster carbon-p model, periodic, optionally non-orthogonal. H and S
// come from the SAME table, so provenance is held fixed and anything the test
// sees is formalism rather than a mismatch of inputs.
void sk_band_velocity(bool with_overlap, bool triclinic = false) {
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
    const double rc = 2.5;
    std::map<std::string, std::vector<std::string>> basis;
    basis["C"] = {"p"};
    SkParameters ham;
    ham.SetOnsite("C", "p", 0.0);
    ham.SetPairChannel("C", "C", "pp", "sigma", spline(2.0, rc));
    ham.SetPairChannel("C", "C", "pp", "pi", spline(0.5, rc));

    std::optional<SkParameters> ovl;
    if (with_overlap) {
        SkParameters s;
        s.SetOnsite("C", "p", 1.0);
        // Small enough that S stays positive definite with this many neighbours;
        // large enough that S1 is not a rounding error next to H1.
        s.SetPairChannel("C", "C", "pp", "sigma", spline(0.05, rc));
        s.SetPairChannel("C", "C", "pp", "pi", spline(-0.02, rc));
        ovl = s;
    }

    const auto table = BuildSkTable(basis, ham, ovl);
    const auto prepared = DeriveBasisSpec(table, rmodel::SpinSpec{});

    const std::vector<double> pos = {0.0, 0.0, 0.0, 1.2, 0.6, 0.4};
    const std::vector<int> z = {6, 6};
    // The triclinic variant matters: a symmetric cell cannot distinguish
    // C^-1 from its transpose in invert_cell, so only a sheared cell
    // exercises the fractional-coordinate convention in the Bloch phases of
    // sample_k and the commutator velocity.
    const std::vector<double> cell = triclinic
        ? std::vector<double>{2.4, 0.0, 0.0, 1.1, 2.2, 0.0, 0.7, 0.5, 2.3}
        : std::vector<double>{2.4, 0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.0, 2.4};
    const std::vector<bool> pbc = {true, true, true};
    auto ad = std::unique_ptr<vbcsr::atomic::AtomicData>(
        vbcsr::atomic::AtomicData::from_points(pos, z, cell, pbc, prepared.cutoff_radius,
                                               prepared.type_norb, MPI_COMM_WORLD));

    rmodel::SktbConfig config;
    config.basis = basis;
    config.hamiltonian = ham;
    config.overlap = ovl;
    auto model = rmodel::SktbModel::Create(config);

    rmodel::SetupOptions options;
    options.kpoints = {kGeneralK,
                       k_step(cell, kGeneralK, kAxis, kStepH),
                       k_step(cell, kGeneralK, kAxis, -kStepH)};
    options.compute_bounds = false;
    rsatb::setup::OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;
    auto ops = rsatb::setup::BuildOperatorSet(*ad, MPI_COMM_WORLD, false, *model, options, runtime);

    auto img_H_held = model->BuildHamiltonianImages(*ad);
    auto* img_H = &img_H_held.real();
    auto img_S = with_overlap ? model->BuildOverlapImages(*ad) : nullptr;
    auto H1 = bond_moment_at_k(*ad, *img_H, kAxis, kGeneralK);
    auto S1 = img_S ? bond_moment_at_k(*ad, *img_S, kAxis, kGeneralK) : nullptr;

    Matrix* H = ops->hamiltonian_k(0);
    Matrix* S = with_overlap ? ops->overlap_k(0) : nullptr;
    const int n = H->graph->global_scalar_rows();

    // A non-orthogonal Slater-Koster model declares no velocity at all
    // (sktb_model.hpp: OperatorSource::kUnavailable), which is exactly the hole
    // the section-4 construction fills.
    Matrix* V = nullptr;
    if (!with_overlap) V = ops->velocity_state(0).x.matrix.get();

    const auto Ep = spectrum(ops->hamiltonian_k(1), with_overlap ? ops->overlap_k(1) : nullptr, n);
    const auto Em = spectrum(ops->hamiltonian_k(2), with_overlap ? ops->overlap_k(2) : nullptr, n);

    const char* title;
    if (triclinic)
        title = with_overlap ? "BAND VELOCITY -- non-orthogonal SK, TRICLINIC cell"
                             : "BAND VELOCITY -- orthogonal SK, TRICLINIC cell";
    else
        title = with_overlap ? "BAND VELOCITY -- non-orthogonal SK (S from the same table, S1 != 0)"
                             : "BAND VELOCITY -- orthogonal SK (S = I, S1 = 0)";
    band_velocity_test(title, H, S, V, H1.get(), S1.get(), Ep, Em, kStepH, n);
}

// Diamond, LCAO. Here H and S are read from hamiltonians.h5 while route (a) is
// integrated by op2c from its own orbital and projector data -- the one case
// where the two routes can genuinely disagree.
void lcao_band_velocity(double kscale) {
    auto structure = vbcsr::atomic::ReadStructure(rsatb::test::data_path("diamond/conf.vasp"));

    rsatb::model::LcaoConfig cfg;
    cfg.hamiltonian = rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/hamiltonians.h5")};
    cfg.orbitals.dir = rsatb::test::data_path("diamond/");
    cfg.orbitals.names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    cfg.pseudopotentials.dir = rsatb::test::data_path("diamond/");
    cfg.pseudopotentials.names = {"C_ONCV_PBE-1.0.upf"};
    auto model = rsatb::model::LcaoModel::Create(cfg, MPI_COMM_WORLD);

    rsatb::model::SetupOptions opt;
    std::vector<double> kbase = kGeneralK;
    for (auto& c : kbase) c *= kscale;
    std::vector<double> kG = kbase;
    kG[0] += 1.0;  // k + G: same physical point, different gauge behaviour
    opt.kpoints = {kbase,
                   k_step(structure.cell, kbase, kAxis, kStepH),
                   k_step(structure.cell, kbase, kAxis, -kStepH),
                   kG};
    opt.compute_bounds = false;

    rsatb::RSATB inst(structure, *model, opt, 1, MPI_COMM_WORLD);
    auto* ops = inst.operators();
    auto* ad = inst.atomic_data();

    auto img_H_held = model->BuildHamiltonianImages(*ad);
    auto* img_H = &img_H_held.real();
    auto img_S = model->BuildOverlapImages(*ad);
    auto H1 = bond_moment_at_k(*ad, *img_H, kAxis, kbase);
    auto S1 = bond_moment_at_k(*ad, *img_S, kAxis, kbase);

    Matrix* H = ops->hamiltonian_k(0);
    Matrix* S = ops->overlap_k(0);
    auto Vt = model->BuildVelocity(*ad, kbase);       // route (a): op2c explicit
    auto VtG = model->BuildVelocity(*ad, kG);
    Matrix* V = Vt->x.matrix.get();
    const int n = H->graph->global_scalar_rows();

    const auto Ep = spectrum(ops->hamiltonian_k(1), ops->overlap_k(1), n);
    const auto Em = spectrum(ops->hamiltonian_k(2), ops->overlap_k(2), n);

    // PROVENANCE. op2c computes S from its own orbital integrals, but the
    // fixture also ships overlaps.h5 -- the overlap that came WITH the
    // Hamiltonian, from the code that produced it. If those two disagree then H
    // and the op2c integrals are not the same basis, and no comparison here
    // says anything about formalism. Reading it as if it were a Hamiltonian is
    // the cheapest way in: the file layout is the same.
    {
        const size_t N = static_cast<size_t>(n);
        rsatb::model::LcaoConfig ref = cfg;
        ref.hamiltonian = rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/overlaps.h5")};
        auto ref_model = rsatb::model::LcaoModel::Create(ref, MPI_COMM_WORLD);
        auto img_file_S_held = ref_model->BuildHamiltonianImages(*ad);
        auto* img_file_S = &img_file_S_held.real();
        std::unique_ptr<Matrix> Sfile(
            img_file_S->sample_k(kbase, vbcsr::atomic::PhaseConvention::R_AND_POSITION));
        const auto A = densify(*Sfile, n);
        const auto B = densify(*S, n);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < N * N; ++i) { num += std::norm(A[i] - B[i]); den += std::norm(B[i]); }
        std::printf("  provenance: |S_file - S_op2c| / |S_op2c| = %.6e\n", std::sqrt(num / den));
    }

    // WHICH GAUGE IS THE VELOCITY IN? Shifting k by a reciprocal lattice vector
    // leaves exp(-2 pi i k.R) untouched but not exp(-2 pi i k.(R + f_j - f_i)).
    // So an operator is in the lattice gauge iff it is invariant under k -> k+G.
    // This settles it by measurement rather than by reading the assembly code.
    {
        const size_t N = static_cast<size_t>(n);
        auto rel = [&](Matrix* A, Matrix* B) {
            const auto a = densify(*A, n), b = densify(*B, n);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < N * N; ++i) { num += std::norm(a[i] - b[i]); den += std::norm(b[i]); }
            return std::sqrt(num / den);
        };
        const auto uG = atomic_gauge(*ad, H->graph, kG, n);
        const auto u0 = atomic_gauge(*ad, H->graph, kbase, n);
        auto rotated_matches = [&](Matrix* AkG, Matrix* Ak) {
            const auto a = densify(*AkG, n), b = densify(*Ak, n);
            double num = 0.0, den = 0.0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    // U(k+G) U(k)^dagger is the residual rotation between the two.
                    const cd w = std::conj(uG[i] * std::conj(u0[i])) * b[static_cast<size_t>(j) * N + i]
                               * (uG[j] * std::conj(u0[j]));
                    num += std::norm(a[static_cast<size_t>(j) * N + i] - w);
                    den += std::norm(a[static_cast<size_t>(j) * N + i]);
                }
            return std::sqrt(num / den);
        };
        std::printf("  gauge of V: |A(k+G) - U_G^H A(k) U_G| / |A|   H %.3e   V %.3e   (0 => atomic gauge)\n",
                    rotated_matches(ops->hamiltonian_k(3), H), rotated_matches(VtG->x.matrix.get(), V));
        std::printf("  (raw k->k+G difference: H %.3e  V %.3e)\n",
                    rel(ops->hamiltonian_k(3), H), rel(VtG->x.matrix.get(), V));
        // Localise it: the position operator goes through the same two-centre
        // assembly but has no three-body projector term.
        Matrix* X0k = ops->position_state(0).x.matrix.get();
        Matrix* XGk = ops->position_state(3).x.matrix.get();
        if (X0k && XGk)
            std::printf("  same law, LCAO position operator X: %.3e   overlap S: %.3e\n",
                        rotated_matches(XGk, X0k), rotated_matches(ops->overlap_k(3), S));
    }

    const auto gauge = atomic_gauge(*ad, H->graph, kbase, n);

    // Validate the gauge vector and its DIRECTION before using it on anything:
    // sampling H in the lattice gauge and rotating must reproduce the very H
    // that operator_set sampled in the atomic gauge. If this is not ~1e-15 the
    // correction below is meaningless.
    {
        const size_t N = static_cast<size_t>(n);
        std::unique_ptr<Matrix> Hl(
            img_H->sample_k(kbase, vbcsr::atomic::PhaseConvention::R_ONLY));
        const auto A = densify(*Hl, n);
        const auto B = densify(*H, n);
        double fwd = 0.0, rev = 0.0, den = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const cd f = std::conj(gauge[i]) * A[static_cast<size_t>(j) * N + i] * gauge[j];
                const cd r = gauge[i] * A[static_cast<size_t>(j) * N + i] * std::conj(gauge[j]);
                fwd += std::norm(f - B[static_cast<size_t>(j) * N + i]);
                rev += std::norm(r - B[static_cast<size_t>(j) * N + i]);
                den += std::norm(B[static_cast<size_t>(j) * N + i]);
            }
        std::printf("  gauge check: |U^H H_lat U - H_atomic|/|H| = %.3e   (other direction %.3e)\n",
                    std::sqrt(fwd / den), std::sqrt(rev / den));
    }

    char title[256];
    std::snprintf(title, sizeof(title),
                  "BAND VELOCITY -- diamond LCAO (H,S from file; route (a) from op2c)  k = %.3g * k0", kscale);
    band_velocity_test(title,
                       H, S, V, H1.get(), S1.get(), Ep, Em, kStepH, n, &gauge);
}

}  // namespace

namespace {

// FORENSICS: measure the phase exponent op2c actually applies, element by
// element. With pbc OFF there are no lattice images, so every (i,j) matrix
// element is a SINGLE phased term and the ratio A(k+G)_ij / A(k)_ij equals
// e^{-2 pi i G . w_ij} exactly, where w_ij is whatever exponent the assembly
// used. The covariance law A(k+G) = U_G^H A(k) U_G holds mechanically -- for
// ANY block values -- iff w_ij = f_j - f_i + integer. So:
//   * pbc=false, all ratios match e^{-2 pi i G.(f_j - f_i)}  => the basic
//     2-body and 3-body phasing is consistent, and any pbc=true failure must
//     come from image handling;
//   * a deviating element here names the mis-phased term directly.
void lcao_gauge_forensics() {
    auto structure = vbcsr::atomic::ReadStructure(rsatb::test::data_path("diamond/conf.vasp"));
    structure.pbc = {false, false, false};   // kill every lattice image

    rsatb::model::LcaoConfig cfg;
    cfg.hamiltonian = rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/hamiltonians.h5")};
    cfg.orbitals.dir = rsatb::test::data_path("diamond/");
    cfg.orbitals.names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    cfg.pseudopotentials.dir = rsatb::test::data_path("diamond/");
    cfg.pseudopotentials.names = {"C_ONCV_PBE-1.0.upf"};
    auto model = rsatb::model::LcaoModel::Create(cfg, MPI_COMM_WORLD);

    const std::vector<double> k0 = {0.13, 0.21, 0.07};
    std::vector<double> kG = k0;
    kG[0] += 1.0;

    rsatb::model::SetupOptions opt;
    opt.kpoints = {k0, kG};
    opt.compute_bounds = false;

    rsatb::RSATB inst(structure, *model, opt, 1, MPI_COMM_WORLD);
    auto* ops = inst.operators();
    auto* ad = inst.atomic_data();

    // velocity_state now serves the production commutator; this probe's
    // route (a) is op2c's explicit integrals, fetched from the model itself.
    auto V0t = model->BuildVelocity(*ad, k0);
    auto VGt = model->BuildVelocity(*ad, kG);
    Matrix* V0 = V0t->x.matrix.get();
    Matrix* VG = VGt->x.matrix.get();
    Matrix* X0m = ops->position_state(0).x.matrix.get();
    Matrix* XGm = ops->position_state(1).x.matrix.get();
    Matrix* H0m = ops->hamiltonian_k(0);
    Matrix* HGm = ops->hamiltonian_k(1);
    const int n = H0m->graph->global_scalar_rows();
    const size_t N = static_cast<size_t>(n);

    // Predicted exponent per scalar index: the fractional coordinate f_i.
    std::vector<double> fx(n, 0.0);
    {
        vbcsr::DistGraph* g = H0m->graph;
        const int nb = static_cast<int>(g->owned_global_indices.size());
        for (int b = 0; b < nb; ++b) {
            double r[3];
            ad->get_pos(b, &r[0], &r[1], &r[2]);
            ad->invert_cell(&r[0], &r[1], &r[2]);
            for (int s = 0; s < g->block_sizes[b]; ++s)
                fx[static_cast<size_t>(g->block_offsets[b] + s)] = r[0];
        }
    }

    auto forensic = [&](const char* name, Matrix* AG, Matrix* A0) {
        const auto a = densify(*AG, n), b = densify(*A0, n);
        // G = (1,0,0): predicted ratio exponent is f_j - f_i along x.
        double worst = 0.0;
        int worst_i = -1, worst_j = -1;
        double worst_meas = 0.0, worst_pred = 0.0;
        int checked = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const cd num = a[static_cast<size_t>(j) * N + i];
                const cd den = b[static_cast<size_t>(j) * N + i];
                if (std::abs(den) < 1e-10 || std::abs(num) < 1e-10) continue;
                ++checked;
                const double meas = -std::arg(num / den) / (2.0 * M_PI);   // w_ij mod 1
                const double pred = fx[j] - fx[i];
                double d = meas - pred;
                d -= std::round(d);                                        // mod 1, into [-1/2,1/2)
                // |ratio| must be 1 too; fold magnitude error in.
                const double mag = std::abs(std::abs(num / den) - 1.0);
                const double err = std::max(std::abs(d), mag);
                if (err > worst) { worst = err; worst_i = i; worst_j = j; worst_meas = meas; worst_pred = pred; }
            }
        std::printf("  %-3s pbc=off: %4d elements, worst |w_meas - (f_j - f_i)| mod 1 = %.3e",
                    name, checked, worst);
        if (worst > 1e-8)
            std::printf("   at (%d,%d): measured %.6f, predicted %.6f", worst_i, worst_j, worst_meas, worst_pred);
        std::printf("\n");
    };

    std::printf("FORENSICS -- diamond C2, pbc OFF (no images; every element single-term)\n");
    forensic("H", HGm, H0m);
    forensic("V", VG, V0);
    if (X0m && XGm) forensic("X", XGm, X0m);
    std::printf("\n");
}

// Fold a matrix to dense by READING ITS STORED BLOCKS, mapping every (possibly
// ghost) column to its home atom by global id. This bypasses vbcsr's mult and
// halo exchange entirely. If this folding satisfies the covariance law while
// the mult-based densify does not, the assembly is innocent and the defect
// lives in the measurement path; if both fail the same way, the stored blocks
// are genuinely mis-phased.
std::vector<cd> densify_direct(Matrix& A, int n) {
    std::vector<cd> dense(static_cast<size_t>(n) * n, cd(0.0, 0.0));
    vbcsr::DistGraph* g = A.graph;
    const int n_owned = static_cast<int>(g->owned_global_indices.size());
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_ind();
    std::vector<int> scal(n_owned + 1, 0);
    for (int b = 0; b < n_owned; ++b) scal[b + 1] = scal[b] + g->block_sizes[b];
    for (int i = 0; i < n_owned; ++i) {
        const int ri = g->block_sizes[i];
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            const int c = ci[k];
            const int gj = g->get_global_index(c);
            const int cj = g->block_sizes[c];
            const cd* bd = A.block_data(k);
            if (bd == nullptr) continue;
            // Canonical vbcsr block layout is column-major within the block.
            for (int col = 0; col < cj; ++col)
                for (int row = 0; row < ri; ++row)
                    dense[static_cast<size_t>(scal[gj] + col) * n + (scal[i] + row)] +=
                        bd[static_cast<size_t>(col) * ri + row];
        }
    }
    return dense;
}

// Same structure, pbc ON. Elements are now image sums, so instead of reading
// off a single exponent, test the covariance law per element and map where it
// breaks: which atom pair, diagonal vs off-diagonal, and how the deviation is
// distributed. Any coherent pattern names the mis-phased contribution.
void lcao_gauge_forensics_pbc() {
    auto structure = vbcsr::atomic::ReadStructure(rsatb::test::data_path("diamond/conf.vasp"));

    rsatb::model::LcaoConfig cfg;
    cfg.hamiltonian = rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/hamiltonians.h5")};
    cfg.orbitals.dir = rsatb::test::data_path("diamond/");
    cfg.orbitals.names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    cfg.pseudopotentials.dir = rsatb::test::data_path("diamond/");
    cfg.pseudopotentials.names = {"C_ONCV_PBE-1.0.upf"};
    auto model = rsatb::model::LcaoModel::Create(cfg, MPI_COMM_WORLD);

    const std::vector<double> k0 = {0.13, 0.21, 0.07};
    std::vector<double> kG = k0;
    kG[0] += 1.0;

    rsatb::model::SetupOptions opt;
    opt.kpoints = {k0, kG};
    opt.compute_bounds = false;

    rsatb::RSATB inst(structure, *model, opt, 1, MPI_COMM_WORLD);
    auto* ops = inst.operators();
    auto* ad = inst.atomic_data();

    auto V0t = model->BuildVelocity(*ad, k0);
    auto VGt = model->BuildVelocity(*ad, kG);
    Matrix* V0 = V0t->x.matrix.get();
    Matrix* VG = VGt->x.matrix.get();
    Matrix* H0m = ops->hamiltonian_k(0);
    const int n = H0m->graph->global_scalar_rows();
    const size_t N = static_cast<size_t>(n);

    std::vector<double> fx(n, 0.0);
    std::vector<int> atom_of(n, 0);
    {
        vbcsr::DistGraph* g = H0m->graph;
        const int nb = static_cast<int>(g->owned_global_indices.size());
        for (int b = 0; b < nb; ++b) {
            double r[3];
            ad->get_pos(b, &r[0], &r[1], &r[2]);
            ad->invert_cell(&r[0], &r[1], &r[2]);
            for (int s = 0; s < g->block_sizes[b]; ++s) {
                fx[static_cast<size_t>(g->block_offsets[b] + s)] = r[0];
                atom_of[static_cast<size_t>(g->block_offsets[b] + s)] = b;
            }
        }
    }

    const auto a = densify(*VG, n), b = densify(*V0, n);
    {
        const auto ad2 = densify_direct(*VG, n), bd2 = densify_direct(*V0, n);
        double d_mult = 0.0, d_dir = 0.0, x_ab = 0.0, nn2 = 0.0;
        for (size_t t = 0; t < static_cast<size_t>(n) * n; ++t) {
            x_ab += std::norm(a[t] - ad2[t]);
            nn2 += std::norm(a[t]);
        }
        auto law = [&](const std::vector<cd>& A2, const std::vector<cd>& B2) {
            double num = 0.0, den = 0.0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const cd pred = std::exp(cd(0.0, -2.0 * M_PI * (fx[j] - fx[i]))) * B2[static_cast<size_t>(j) * N + i];
                    num += std::norm(A2[static_cast<size_t>(j) * N + i] - pred);
                    den += std::norm(A2[static_cast<size_t>(j) * N + i]);
                }
            return std::sqrt(num / den);
        };
        d_mult = law(a, b);
        d_dir = law(ad2, bd2);
        std::printf("FORENSICS -- fold comparison: |densify_mult - densify_direct|/|.| = %.3e\n", std::sqrt(x_ab / nn2));
        std::printf("  covariance law on op2c V:  mult fold %.3e   direct fold %.3e\n", d_mult, d_dir);
    }

    // PER-STORED-BLOCK forensics on the position operator: it violates the law
    // like V does, is pure 2-body, and lives on the atomic graph, where every
    // ghost column's image position is recoverable. Each stored block belongs
    // to ONE displacement, so BG = e^{-2 pi i G.w} B0 must hold block-by-block
    // with w = fractional displacement of that column. A block whose measured w
    // is not its column's displacement names the mis-phased deposit directly.
    {
        Matrix* X0m = ops->position_state(0).x.matrix.get();
        Matrix* XGm = ops->position_state(1).x.matrix.get();
        if (X0m && XGm && X0m->graph == XGm->graph) {
            vbcsr::DistGraph* g = X0m->graph;
            const int n_owned = static_cast<int>(g->owned_global_indices.size());
            const auto& rp = X0m->row_ptr();
            const auto& ci = X0m->col_ind();
            std::printf("FORENSICS -- per stored block of op2c X (pbc ON), G=(1,0,0)\n");
            int bad = 0, total = 0, shown = 0;
            for (int i = 0; i < n_owned; ++i) {
                double pi[3];
                ad->get_pos(i, &pi[0], &pi[1], &pi[2]);
                for (int k = rp[i]; k < rp[i + 1]; ++k) {
                    const int c = ci[k];
                    double pc[3];
                    ad->get_pos(c, &pc[0], &pc[1], &pc[2]);
                    double d[3] = {pc[0] - pi[0], pc[1] - pi[1], pc[2] - pi[2]};
                    ad->invert_cell(&d[0], &d[1], &d[2]);
                    const int ri = g->block_sizes[i], cj = g->block_sizes[c];
                    const cd* b0 = X0m->block_data(k);
                    const cd* bg = XGm->block_data(k);
                    if (!b0 || !bg) continue;
                    cd num(0.0, 0.0);
                    double den = 0.0, tot = 0.0;
                    for (int t = 0; t < ri * cj; ++t) {
                        num += std::conj(b0[t]) * bg[t];
                        den += std::norm(b0[t]);
                        tot += std::norm(bg[t]);
                    }
                    if (den < 1e-16 || tot < 1e-16) continue;
                    ++total;
                    const cd ratio = num / den;
                    double coher = 0.0;   // is the block really single-phase?
                    for (int t = 0; t < ri * cj; ++t) coher += std::norm(bg[t] - ratio * b0[t]);
                    coher = std::sqrt(coher / tot);
                    const double w_meas = -std::arg(ratio) / (2.0 * M_PI);
                    double dev = w_meas - d[0];
                    dev -= std::round(dev);
                    const bool off = std::abs(dev) > 1e-8 || std::abs(std::abs(ratio) - 1.0) > 1e-8;
                    if (off) ++bad;
                    if (off && shown < 12) {
                        ++shown;
                        std::printf("    block row %d -> col local %d (home %d): disp_frac_x %+ .4f   w_meas %+ .4f   dev %+ .4f   |ratio| %.6f   coher %.2e\n",
                                    i, c, g->get_global_index(c), d[0], w_meas, dev, std::abs(ratio), coher);
                    }
                }
            }
            std::printf("    %d of %d stored blocks off-family\n\n", bad, total);
        } else {
            std::printf("FORENSICS -- X graphs differ between k-points; per-block match skipped\n\n");
        }
    }
    std::printf("FORENSICS -- diamond sc111, pbc ON: covariance deviation of op2c V by atom block\n");
    double dev[2][2] = {{0, 0}, {0, 0}}, nrm[2][2] = {{0, 0}, {0, 0}};
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const cd pred = std::exp(cd(0.0, -2.0 * M_PI * (fx[j] - fx[i]))) * b[static_cast<size_t>(j) * N + i];
            const cd got = a[static_cast<size_t>(j) * N + i];
            dev[atom_of[i]][atom_of[j]] += std::norm(got - pred);
            nrm[atom_of[i]][atom_of[j]] += std::norm(got);
        }
    for (int bi = 0; bi < 2; ++bi)
        for (int bj = 0; bj < 2; ++bj)
            std::printf("    block (%d,%d): rel dev %.3e   (|block| %.3e)\n", bi, bj,
                        std::sqrt(dev[bi][bj] / std::max(nrm[bi][bj], 1e-300)), std::sqrt(nrm[bi][bj]));

    // The sharpest single number: the worst element, with its measured vs
    // predicted ratio, in case one parasitic term dominates it.
    double worst = 0.0;
    int wi = -1, wj = -1;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const cd pred = std::exp(cd(0.0, -2.0 * M_PI * (fx[j] - fx[i]))) * b[static_cast<size_t>(j) * N + i];
            const double e = std::abs(a[static_cast<size_t>(j) * N + i] - pred);
            if (e > worst) { worst = e; wi = i; wj = j; }
        }
    const cd ra = a[static_cast<size_t>(wj) * N + wi], rb = b[static_cast<size_t>(wj) * N + wi];
    std::printf("    worst element (%d,%d): |dev| %.3e   V(k+G)=(%.6f,%.6f)  V(k)=(%.6f,%.6f)  |ratio|=%.6f  arg/2pi=%.6f  pred=%.6f\n\n",
                wi, wj, worst, ra.real(), ra.imag(), rb.real(), rb.imag(),
                std::abs(ra / rb), -std::arg(ra / rb) / (2.0 * M_PI), fx[wj] - fx[wi]);
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------
// SPECTRAL-CONVENTION STUDY (--spectral)
//
// The band-velocity test above arbitrates only the DIAGONAL of the velocity
// in the eigenbasis. Spectral functions -- AC, DC, MSD, quantum metric -- are
// built from |v_nm|^2 across pairs, so choosing a convention needs the
// off-diagonals measured too. This section (1) validates the section-4
// assembly off-diagonally on SK models, where route (a) and route (b) must be
// THE SAME matrix, and (2) dumps per-pair data on diamond LCAO over a k-grid
// for offline analysis: f-sum rule, energy-resolved deviation, sigma(omega).
//
// Units: both routes are exported as vtilde = hbar * v in eV*Angstrom, so
// vtilde_nn = dE_n/dk exactly. Since the unit unification (every builder now
// stores hbar * velocity in eV*Angstrom -- commutator_velocity.hpp UNITS):
//   vtilde_a = -V_stored             (kappa_a = -1 on SK, both builders)
//   vtilde_b = i * (S0 C)            (kappa_b = +i on every model)
constexpr double kTRKdenominator = 3.80998212;  // hbar^2 / 2 m_e, eV A^2

// Assemble S0*C(k) densely from the section-4 pieces at one k-point.
std::vector<cd> assemble_S0C(vbcsr::atomic::AtomicData& ad,
                             Matrix* Hm, Matrix* Sm,
                             Matrix* X0mat, vbcsr::DistVector<cd>* X0diag,
                             vbcsr::atomic::ImageContainer<double>* img_H,
                             vbcsr::atomic::ImageContainer<double>* img_S,
                             const std::vector<double>& k, int n,
                             bool include_dipole = true) {
    const size_t N = static_cast<size_t>(n);
    auto eye = [&]() {
        std::vector<cd> I(N * N, cd(0.0, 0.0));
        for (int i = 0; i < n; ++i) I[static_cast<size_t>(i) * N + i] = cd(1.0, 0.0);
        return I;
    };
    auto H0 = densify(*Hm, n);
    auto S0 = Sm ? densify(*Sm, n) : eye();
    auto DH_m = bond_moment_at_k(ad, *img_H, 0, k);
    auto DH = densify(*DH_m, n);
    std::vector<cd> DS(N * N, cd(0.0, 0.0));
    if (img_S) {
        auto DS_m = bond_moment_at_k(ad, *img_S, 0, k);
        DS = densify(*DS_m, n);
    }
    std::vector<cd> X0(N * N, cd(0.0, 0.0));
    if (X0mat) X0 = densify(*X0mat, n);
    else if (X0diag) for (int i = 0; i < n; ++i) X0[static_cast<size_t>(i) * N + i] = (*X0diag)[i];

    // Cartesian x of each scalar row's atom. With BOND moments the dipole term
    // must be the KET-CENTERED dipole Delta = X0 - D[S] - diag(r) S0: the bond
    // weighting already carries the inter-site displacement, so leaving the
    // monopole r_i S inside X0 double-counts it. At Gamma this form reduces
    // exactly to the section-4 lattice-moment form (the residual [r, A0]
    // cancels), and on an orthogonal SK model Delta vanishes identically,
    // leaving C = -D[H] -- the bond-vector velocity, as it must.
    std::vector<double> rx(n, 0.0);
    {
        vbcsr::DistGraph* g = Hm->graph;
        const int nb = static_cast<int>(g->owned_global_indices.size());
        for (int b = 0; b < nb; ++b) {
            double r[3];
            ad.get_pos(b, &r[0], &r[1], &r[2]);
            for (int s = 0; s < g->block_sizes[b]; ++s)
                rx[static_cast<size_t>(g->block_offsets[b] + s)] = r[0];
        }
    }

    auto solveS = [&](const std::vector<cd>& M) {
        std::vector<cd> B = M, LU = S0;
        std::vector<vbcsr_lapack_int> ipiv(n);
        vbcsr_lapack_int nn = n, nrhs = n, lda = n, ldb = n, info = 0;
        zgesv_(&nn, &nrhs, LU.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
        if (info != 0) std::printf("  assemble_S0C: zgesv info=%lld\n", (long long)info);
        return B;
    };
    auto mul = [&](const std::vector<cd>& P, const std::vector<cd>& Q) {
        std::vector<cd> R(N * N, cd(0.0, 0.0));
        for (int j = 0; j < n; ++j)
            for (int k2 = 0; k2 < n; ++k2) {
                const cd q = Q[static_cast<size_t>(j) * N + k2];
                if (q == cd(0.0, 0.0)) continue;
                for (int i = 0; i < n; ++i)
                    R[static_cast<size_t>(j) * N + i] += P[static_cast<size_t>(k2) * N + i] * q;
            }
        return R;
    };

    const auto A0 = solveS(H0);
    std::vector<cd> t(N * N);
    { const auto sa = mul(DS, A0); for (size_t i = 0; i < N * N; ++i) t[i] = DH[i] - sa[i]; }
    const auto cell_term = solveS(t);
    // Delta = X0 - D[S] - diag(r) S0 is the KET-CENTERED intra-atomic dipole:
    // what remains of the position operator once the monopole (which atom) and
    // the bond displacement (how far to the ket atom) -- both obtainable from
    // S(R) and geometry alone -- have been removed. It is the ONLY ingredient
    // in the whole construction that needs real orbital integrals. Setting
    // include_dipole = false measures what a model without them loses: it is
    // exactly the approximation a non-orthogonal SK/DFTB model is forced into,
    // and it changes nothing on the diagonal (the commutator vanishes there),
    // so its whole effect is interband -- i.e. visible in the f-sum rule.
    std::vector<cd> C(N * N);
    if (!include_dipole) {
        for (size_t i = 0; i < N * N; ++i) C[i] = -cell_term[i];
        return mul(S0, C);
    }
    std::vector<cd> D0(N * N);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            D0[static_cast<size_t>(j) * N + i] = X0[static_cast<size_t>(j) * N + i]
                - DS[static_cast<size_t>(j) * N + i]
                - rx[i] * S0[static_cast<size_t>(j) * N + i];
    const auto SD0 = solveS(D0);
    const auto c1 = mul(SD0, A0), c2 = mul(A0, SD0);
    for (size_t i = 0; i < N * N; ++i) C[i] = -cell_term[i] + (c1[i] - c2[i]);
    return mul(S0, C);
}

// Route (a) and route (b) are the SAME operator on a Slater-Koster model, so
// on SK the full matrices must agree at a general k -- this is the off-diagonal
// validation the band-velocity test cannot give.
void sk_offdiag_check(bool triclinic) {
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
    const double rc = 2.5;
    std::map<std::string, std::vector<std::string>> basis;
    basis["C"] = {"p"};
    SkParameters ham;
    ham.SetOnsite("C", "p", 0.0);
    ham.SetPairChannel("C", "C", "pp", "sigma", spline(2.0, rc));
    ham.SetPairChannel("C", "C", "pp", "pi", spline(0.5, rc));

    const auto table = BuildSkTable(basis, ham, std::nullopt);
    const auto prepared = DeriveBasisSpec(table, rmodel::SpinSpec{});

    const std::vector<double> pos = {0.0, 0.0, 0.0, 1.2, 0.6, 0.4};
    const std::vector<int> z = {6, 6};
    const std::vector<double> cell = triclinic
        ? std::vector<double>{2.4, 0.0, 0.0, 1.1, 2.2, 0.0, 0.7, 0.5, 2.3}
        : std::vector<double>{2.4, 0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.0, 2.4};
    const std::vector<bool> pbc = {true, true, true};
    auto ad = std::unique_ptr<vbcsr::atomic::AtomicData>(
        vbcsr::atomic::AtomicData::from_points(pos, z, cell, pbc, prepared.cutoff_radius,
                                               prepared.type_norb, MPI_COMM_WORLD));

    rmodel::SktbConfig config;
    config.basis = basis;
    config.hamiltonian = ham;
    auto model = rmodel::SktbModel::Create(config);

    rmodel::SetupOptions options;
    options.kpoints = {kGeneralK};
    options.compute_bounds = false;
    rsatb::setup::OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;
    auto ops = rsatb::setup::BuildOperatorSet(*ad, MPI_COMM_WORLD, false, *model, options, runtime);
    auto img_H_held = model->BuildHamiltonianImages(*ad);
    auto* img_H = &img_H_held.real();

    Matrix* H = ops->hamiltonian_k(0);
    Matrix* Va = ops->velocity_state(0).x.matrix.get();
    auto* Xdiag = ops->position_state(0).x.orthogonal_diagonal.get();
    const int n = H->graph->global_scalar_rows();
    const size_t N = static_cast<size_t>(n);

    const auto S0C = assemble_S0C(*ad, H, nullptr, nullptr, Xdiag, img_H, nullptr, kGeneralK, n);
    const auto va = densify(*Va, n);

    double num = 0.0, den = 0.0;
    for (size_t t = 0; t < N * N; ++t) {
        const cd vb = cd(0.0, 1.0) * S0C[t];               // vtilde_b
        const cd vA = -va[t];                              // vtilde_a (stored is eV*A)
        num += std::norm(vA - vb);
        den += std::norm(vA);
    }
    std::printf("  SK %-10s full-matrix |vtilde_a - vtilde_b| / |vtilde_a| = %.6e   %s\n",
                triclinic ? "triclinic" : "cubic", std::sqrt(num / den),
                std::sqrt(num / den) < 1e-10 ? "<== off-diagonals AGREE" : "<== DISAGREE");
}

// Dump per-pair eigenbasis matrix elements of both conventions on diamond
// LCAO over a Gamma-centered shifted k-grid, x component.
void lcao_spectral_dump(int nk, const char* outfile) {
    auto structure = vbcsr::atomic::ReadStructure(rsatb::test::data_path("diamond/conf.vasp"));

    rsatb::model::LcaoConfig cfg;
    cfg.hamiltonian = rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/hamiltonians.h5")};
    cfg.orbitals.dir = rsatb::test::data_path("diamond/");
    cfg.orbitals.names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    cfg.pseudopotentials.dir = rsatb::test::data_path("diamond/");
    cfg.pseudopotentials.names = {"C_ONCV_PBE-1.0.upf"};
    auto model = rsatb::model::LcaoModel::Create(cfg, MPI_COMM_WORLD);

    rsatb::model::SetupOptions opt;
    for (int a = 0; a < nk; ++a)
        for (int b = 0; b < nk; ++b)
            for (int c = 0; c < nk; ++c)
                opt.kpoints.push_back({(a + 0.5) / nk, (b + 0.5) / nk, (c + 0.5) / nk});
    opt.compute_bounds = false;

    rsatb::RSATB inst(structure, *model, opt, 1, MPI_COMM_WORLD);
    auto* ops = inst.operators();
    auto* ad = inst.atomic_data();
    auto img_H_held = model->BuildHamiltonianImages(*ad);
    auto* img_H = &img_H_held.real();
    auto img_S = model->BuildOverlapImages(*ad);

    std::FILE* out = std::fopen(outfile, "w");
    std::fprintf(out, "# diamond LCAO 2s2p1d, %dx%dx%d shifted grid, x component\n", nk, nk, nk);
    std::fprintf(out, "# kidx n m E_n E_m Re(va) Im(va) Re(vb) Im(vb) Re(vb_nodipole) Im(vb_nodipole)\n");

    const int nkpt = static_cast<int>(opt.kpoints.size());
    for (int kidx = 0; kidx < nkpt; ++kidx) {
        Matrix* H = ops->hamiltonian_k(kidx);
        Matrix* S = ops->overlap_k(kidx);
        auto Vat = model->BuildVelocity(*ad, opt.kpoints[kidx]);  // route (a): op2c
        Matrix* Va = Vat->x.matrix.get();
        Matrix* X0 = ops->position_state(kidx).x.matrix.get();
        const int n = H->graph->global_scalar_rows();
        const size_t N = static_cast<size_t>(n);

        auto Hd = densify(*H, n);
        auto Sd = densify(*S, n);
        std::vector<double> E;
        std::vector<cd> Cv;
        if (!gen_eig(Hd, Sd, n, E, Cv)) { std::printf("  k %d eigensolve failed\n", kidx); continue; }

        const auto S0C = assemble_S0C(*ad, H, S, X0, nullptr, img_H, img_S.get(),
                                      opt.kpoints[kidx], n);
        const auto S0C_nd = assemble_S0C(*ad, H, S, X0, nullptr, img_H, img_S.get(),
                                         opt.kpoints[kidx], n, /*include_dipole=*/false);
        std::vector<cd> vnd(N * N);
        for (size_t t = 0; t < N * N; ++t) vnd[t] = cd(0.0, 1.0) * S0C_nd[t];
        const auto vnd_e = to_eigenbasis(vnd, Cv, n);
        auto vad = densify(*Va, n);
        std::vector<cd> vb(N * N), va2(N * N);
        for (size_t t = 0; t < N * N; ++t) {
            vb[t] = cd(0.0, 1.0) * S0C[t];
            // UNITS: every velocity builder now stores hbar * velocity in
            // eV*Angstrom (commutator_velocity.hpp UNITS), so vtilde is -V
            // uniformly. Historically the generic builder divided by hbar
            // (A/fs) while op2c did not, which mis-scaled cross-family
            // comparisons and production sigma by hbar^2 = 0.433 until the
            // convention was unified.
            va2[t] = -vad[t];
        }
        const auto va_e = to_eigenbasis(va2, Cv, n);
        const auto vb_e = to_eigenbasis(vb, Cv, n);

        for (int m = 0; m < n; ++m)
            for (int nn2 = 0; nn2 < n; ++nn2) {
                const cd A = va_e[static_cast<size_t>(m) * N + nn2];
                const cd B = vb_e[static_cast<size_t>(m) * N + nn2];
                const cd D = vnd_e[static_cast<size_t>(m) * N + nn2];
                std::fprintf(out, "%d %d %d %.10f %.10f %.6e %.6e %.6e %.6e %.6e %.6e\n",
                             kidx, nn2, m, E[nn2], E[m],
                             A.real(), A.imag(), B.real(), B.imag(), D.real(), D.imag());
            }
        if ((kidx + 1) % 8 == 0) std::printf("  spectral dump: %d / %d k-points\n", kidx + 1, nkpt);
    }
    std::fclose(out);
    std::printf("  wrote %s\n", outfile);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        if (rank == 0) std::printf("serial only\n");
        MPI_Finalize();
        return 0;
    }
    // Route (a) is op2c's explicit-integral velocity, which needs the beta
    // projectors of a pseudopotential file the test data does not ship
    // (production velocity is the commutator route and needs none).
    if (!std::ifstream(rsatb::test::data_path("diamond/C_ONCV_PBE-1.0.upf")).good()) {
        std::printf("probe_velocity_gauge: needs test/data/diamond/C_ONCV_PBE-1.0.upf "
                    "for the explicit-integral route; nothing to do.\n");
        MPI_Finalize();
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == std::string("--spectral")) {
        std::printf("SPECTRAL CONVENTION STUDY\n");
        sk_offdiag_check(false);
        sk_offdiag_check(true);
        lcao_spectral_dump(argc > 2 ? std::atoi(argv[2]) : 4,
                           argc > 3 ? argv[3] : "spectral_pairs.txt");
        MPI_Finalize();
        return 0;
    }

    validate_on_orthogonal_sk();
    sk_band_velocity(/*with_overlap=*/false);
    sk_band_velocity(/*with_overlap=*/true);
    sk_band_velocity(/*with_overlap=*/false, /*triclinic=*/true);
    sk_band_velocity(/*with_overlap=*/true, /*triclinic=*/true);
    lcao_gauge_forensics();
    lcao_gauge_forensics_pbc();
    lcao_band_velocity(1.0);
    lcao_band_velocity(0.1);
    lcao_band_velocity(0.01);

    std::vector<std::vector<double>> kpoints = {{0.0, 0.0, 0.0}};

    rsatb::model::LcaoConfig cfg;
    cfg.hamiltonian = rsatb::io::HamiltonianFile{rsatb::test::data_path("diamond/hamiltonians.h5")};
    cfg.orbitals.dir = rsatb::test::data_path("diamond/");
    cfg.orbitals.names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    cfg.pseudopotentials.dir = rsatb::test::data_path("diamond/");
    cfg.pseudopotentials.names = {"C_ONCV_PBE-1.0.upf"};
    auto model = rsatb::model::LcaoModel::Create(cfg, MPI_COMM_WORLD);

    rsatb::model::SetupOptions opt;
    opt.kpoints = kpoints;
    opt.compute_bounds = false;

    rsatb::RSATB inst(vbcsr::atomic::ReadStructure(rsatb::test::data_path("diamond/conf.vasp")),
                      *model, opt, 1, MPI_COMM_WORLD);
    auto* ops = inst.operators();

    auto* ad = inst.atomic_data();
    auto img_H_held = model->BuildHamiltonianImages(*ad);
    auto* img_H = &img_H_held.real();
    auto img_S = model->BuildOverlapImages(*ad);
    auto H1m = rsatb::op::BuildLatticeMomentAxis(*ad, *img_H, 0);
    auto S1m = rsatb::op::BuildLatticeMomentAxis(*ad, *img_S, 0);

    Matrix* H = ops->hamiltonian_k(0);
    Matrix* S = ops->overlap_k(0);
    auto Vxt = model->BuildVelocity(*ad, kpoints[0]);  // route (a): op2c explicit
    Matrix* Vx = Vxt->x.matrix.get();
    Matrix* Xx = ops->position_state(0).x.matrix.get();
    if (!H || !S || !Vx || !Xx) { std::printf("missing operator\n"); MPI_Finalize(); return 1; }

    time_reversal_at_gamma("diamond LCAO, route (a) = op2c", H, S, Vx,
                           H->graph->global_scalar_rows());

    // Are these matrices even on the same row ordering? The velocity is built
    // on a denser graph; if its vertices are ordered differently, densifying
    // each on its own graph scrambles rows and every comparison below is noise.
    std::printf("graphs: H=%p S=%p V=%p X=%p  rows H=%d V=%d X=%d\n",
                (void*)H->graph, (void*)S->graph, (void*)Vx->graph, (void*)Xx->graph,
                H->graph->global_scalar_rows(), Vx->graph->global_scalar_rows(),
                Xx->graph->global_scalar_rows());
    {
        const auto& a = H->graph->block_offsets;
        const auto& b = Vx->graph->block_offsets;
        bool same = a.size() == b.size();
        if (same) for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) { same = false; break; }
        std::printf("block_offsets H vs V identical: %s (sizes %zu, %zu)\n",
                    same ? "yes" : "NO", a.size(), b.size());
        const auto& ga = H->graph->owned_global_indices;
        const auto& gb = Vx->graph->owned_global_indices;
        bool sameg = ga.size() == gb.size();
        if (sameg) for (size_t i = 0; i < ga.size(); ++i) if (ga[i] != gb[i]) { sameg = false; break; }
        std::printf("owned_global_indices H vs V identical: %s\n", sameg ? "yes" : "NO");
    }

    const int n = H->graph->global_scalar_rows();
    const size_t N = static_cast<size_t>(n);
    auto H0 = densify(*H, n), S0 = densify(*S, n), Vd = densify(*Vx, n), X0 = densify(*Xx, n);
    auto H1 = densify(*H1m, n), S1 = densify(*S1m, n);

    // A0 = S0^-1 H0, and generally S0^-1 M, by LU solve.
    auto solveS = [&](const std::vector<cd>& M) {
        std::vector<cd> B = M, LU = S0;
        std::vector<vbcsr_lapack_int> ipiv(n);
        vbcsr_lapack_int nn = n, nrhs = n, lda = n, ldb = n, info = 0;
        zgesv_(&nn, &nrhs, LU.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
        if (info != 0) std::printf("  zgesv info=%lld\n", (long long)info);
        return B;
    };
    auto mul = [&](const std::vector<cd>& P, const std::vector<cd>& Q) {
        std::vector<cd> R(N * N, cd(0.0, 0.0));
        for (int j = 0; j < n; ++j)
            for (int k2 = 0; k2 < n; ++k2) {
                const cd q = Q[static_cast<size_t>(j) * N + k2];
                if (q == cd(0.0, 0.0)) continue;
                for (int i = 0; i < n; ++i) R[static_cast<size_t>(j) * N + i] += P[static_cast<size_t>(k2) * N + i] * q;
            }
        return R;
    };

    const auto A0 = solveS(H0);
    std::vector<cd> H1_S1A0(N * N);
    for (size_t i = 0; i < N * N; ++i) H1_S1A0[i] = H1[i];
    { const auto t = mul(S1, A0); for (size_t i = 0; i < N * N; ++i) H1_S1A0[i] -= t[i]; }
    const auto cell_term = solveS(H1_S1A0);            // S0^-1 (H1 - S1 A0)

    std::vector<cd> D0(N * N);
    for (size_t i = 0; i < N * N; ++i) D0[i] = X0[i] - S1[i];   // D0 = X0 - S1
    const auto SD0 = solveS(D0);
    const auto comm1 = mul(SD0, A0), comm2 = mul(A0, SD0);

    std::vector<cd> C(N * N);
    for (size_t i = 0; i < N * N; ++i) C[i] = -cell_term[i] + (comm1[i] - comm2[i]);

    // v_explicit and C are the same operator up to one unit constant. Fit it by
    // least squares and report the residual: that residual IS the disagreement.
    auto fit_residual = [&](const std::vector<cd>& target, const std::vector<cd>& model_m,
                            const char* label) {
        cd num(0.0, 0.0); double den = 0.0, tn = 0.0;
        for (size_t i = 0; i < N * N; ++i) {
            num += std::conj(model_m[i]) * target[i];
            den += std::norm(model_m[i]);
            tn  += std::norm(target[i]);
        }
        const cd kappa = num / den;
        double res = 0.0;
        for (size_t i = 0; i < N * N; ++i) res += std::norm(target[i] - kappa * model_m[i]);
        std::printf("  %-28s kappa = (%.6f, %.6f)   residual %.4e  (relative to ||v_a||)\n",
                    label, kappa.real(), kappa.imag(), std::sqrt(res / tn));
    };

    std::printf("\n# diamond sc111 (2 C, 2s2p1d) at Gamma, LCAO non-orthogonal, n = %d\n", n);
    std::printf("# v_explicit (op2c: grad + beta 3-body) vs the commutator route, fitted\n");
    std::printf("# up to one unit constant. Residual 0 => the two routes are the same operator.\n\n");

    auto nrm = [&](const std::vector<cd>& M) {
        double t = 0.0; for (auto& v : M) t += std::norm(v); return std::sqrt(t);
    };
    auto herm = [&](const std::vector<cd>& M) {   // ||M - M^H|| / ||M||
        double t = 0.0; for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
            t += std::norm(M[static_cast<size_t>(j)*N+i] - std::conj(M[static_cast<size_t>(i)*N+j]));
        const double d = nrm(M); return d > 0 ? std::sqrt(t)/d : 0.0;
    };
    std::printf("  norms  H0 %.4e  S0 %.4e  X0 %.4e  H1 %.4e  S1 %.4e  v_a %.4e  C %.4e\n",
                nrm(H0), nrm(S0), nrm(X0), nrm(H1), nrm(S1), nrm(Vd), nrm(C));
    // C is the operator on COEFFICIENTS; v_a is the operator's MATRIX
    // <phi|v|phi>. They differ by the metric: the coefficient operator of v is
    // S^-1 V, so V = const * S0 C is the relation to test, not V = const * C.
    const auto SC = mul(S0, C);
    std::printf("  herm dev:  v_a %.3e   S0*C %.3e   (both should be ~0 up to the i factor)\n",
                herm(Vd), herm(SC));
    std::printf("  first v_a entries: (%.5f,%.5f) (%.5f,%.5f)   C: (%.5f,%.5f) (%.5f,%.5f)\n",
                Vd[0].real(), Vd[0].imag(), Vd[N].real(), Vd[N].imag(),
                C[0].real(), C[0].imag(), C[N].real(), C[N].imag());

    fit_residual(Vd, SC, "S0 * full C  (doc eq. 4)");

    // The sign of R in the image convention, and of the S1 subtraction in
    // D0 = X0 -/+ S1, are conventions rather than physics. Sweep the four
    // combinations and let the residual pick; a clear winner identifies the
    // convention, four equally bad ones mean the disagreement is elsewhere.
    // Independent cross-check of C itself: the bond-vector construction
    // D[H] that operator/commutator_velocity.hpp assembles. It is route (c) --
    // the diagonal-position approximation -- so it should be CLOSE to C, not
    // equal. Wildly different would mean C is simply wrong.
    {
        auto builder = rsatb::op::MakeCommutatorVelocityBuilder(img_H_held);
        auto triplet = builder(*ad, rsatb::op::KPoint{{0.0, 0.0, 0.0}});
        if (triplet && triplet->x.matrix) {
            auto Dh = densify(*triplet->x.matrix, n);
            fit_residual(Dh, SC, "D[H] (route c) vs S0*C");
            fit_residual(Vd, Dh, "v_explicit vs D[H] (route c)");
        } else {
            std::printf("  commutator-velocity builder produced nothing\n");
        }
    }

    std::printf("\n  sign sweep  (cellsign, dipolesign) -> residual\n");
    for (double cs : {-1.0, 1.0})
        for (double ds : {-1.0, 1.0}) {
            std::vector<cd> Dv(N * N), Cv(N * N);
            for (size_t i = 0; i < N * N; ++i) Dv[i] = X0[i] + ds * S1[i];
            const auto SDv = solveS(Dv);
            const auto a1 = mul(SDv, A0), a2 = mul(A0, SDv);
            for (size_t i = 0; i < N * N; ++i) Cv[i] = cs * cell_term[i] + (a1[i] - a2[i]);
            char lbl[64];
            std::snprintf(lbl, sizeof lbl, "cell %+.0f, D0 = X0 %+.0f*S1", cs, ds);
            fit_residual(Vd, mul(S0, Cv), lbl);
        }
    std::printf("\n");

    // Ablations: which term carries the difference?
    std::vector<cd> C_noS1(N * N), C_noDipole(N * N), C_XonlyCommutator(N * N);
    {
        const auto cell_noS1 = solveS(H1);                       // drop S1 in the cell term
        const auto SX0 = solveS(X0);                             // and use X0, not X0 - S1
        const auto c1 = mul(SX0, A0), c2 = mul(A0, SX0);
        for (size_t i = 0; i < N * N; ++i) {
            C_noS1[i] = -cell_noS1[i] + (c1[i] - c2[i]);
            C_noDipole[i] = -cell_term[i];
            C_XonlyCommutator[i] = c1[i] - c2[i];
        }
    }
    fit_residual(Vd, mul(S0, C_noS1), "S0 * C, S1 dropped");
    fit_residual(Vd, mul(S0, C_noDipole), "S0 * cell term only");
    fit_residual(Vd, mul(S0, C_XonlyCommutator), "S0 * [S^-1 X0, A0] only");

    MPI_Finalize();
    return 0;
}
