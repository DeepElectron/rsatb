// Which congruence the Newton-Schulz refinement should iterate on:
//
//   adjoint:  X = Z^H S Z   (Hermitian by construction, what isqrtm.hpp does)
//   plain:    X = Z  S Z    (Hermitian only if Z is)
//
// The two coincide when Z is Hermitian. The graph seed is NOT exactly
// Hermitian: block row i is solved on row i's subgraph and block row j on
// row j's, so Z0[i][j] and Z0[j][i]^H come from two different truncated
// problems and differ at the truncation error. This experiment measures how
// far, and what each variant then does to (a) its own residual, (b) the
// OTHER variant's residual, and (c) the Hermiticity of Z itself -- because
// the downstream contract is Z^H S Z = I (that is what makes Z^H H Z a
// similarity transform of S^-1 H), so a variant that converges in its own
// metric while leaving the adjoint metric high is not converging at all.
//
//   OMP_NUM_THREADS=... ./experiment_ns_variants [--supercell 4]
//                                                [--steps 6] [--sptol 1e-5]
//                                                [--kpoint kx ky kz] [--only NAME]
//
// Measured outcomes (diamond sc888, seed residual 1.9e-2, herm defect 1.4e-2):
//
//   adjoint  Z^H S Z   converges 4 steps to 4.5e-9, at Gamma and at complex k
//                      alike; leaves Z S Z - I at ~0.2 (unitary gauge freedom).
//   plain    Z S Z     improves for one step, then the anti-Hermitian mode
//                      compounds: 8.3e-3 -> 0.11 -> 2.1 -> 54. Diverges.
//   trans    Z^T S Z   identical to adjoint at Gamma (real). At complex k the
//                      SEED already violates it by 5.5 -- the seed approximates
//                      the Hermitian S^-1/2, and W^T S W = conj(W) S W != I for
//                      complex Hermitian W -- so the iteration is outside the
//                      Newton-Schulz basin before the first step and explodes.
//   sym      untested  (dropped: adds a full transpose per step).
//
// Conclusion: the conjugate congruence is the only variant that works at every
// k, and it is what vbcsr's transpose()/mult_adjoint implement throughout.

#include "../isqrtm.hpp"
#include "../../model/lcao/op2c_engine.hpp"
#include "../../test_data_path.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/io.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using Scalar = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<Scalar>;

namespace {

double maxabs_block_norm(Matrix& m) {
    double norm = 0.0;
    for (double n : m.get_block_norms()) norm = std::max(norm, n);
    double global = norm;
    MPI_Allreduce(&norm, &global, 1, MPI_DOUBLE, MPI_MAX, m.graph->comm);
    return global;
}

// ||M - I||_maxabs without disturbing M.
double residual_of(Matrix& m) {
    m.shift(Scalar(-1.0));
    const double r = maxabs_block_norm(m);
    m.shift(Scalar(1.0));
    return r;
}

// ||Z - Z^H||_maxabs.
double hermiticity_defect(const Matrix& z) {
    Matrix zh = z.transpose();
    zh.axpby(Scalar(1.0), z, Scalar(-1.0));  // zh = Z - Z^H
    return maxabs_block_norm(zh);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int repeat = 4;
    int max_steps = 6;
    double sptol = 1e-5;
    std::vector<double> kpoint = {0.0, 0.0, 0.0};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--supercell") == 0 && i + 1 < argc) repeat = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) max_steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sptol") == 0 && i + 1 < argc) sptol = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kpoint") == 0 && i + 3 < argc) {
            kpoint = {std::atof(argv[i + 1]), std::atof(argv[i + 2]), std::atof(argv[i + 3])};
            i += 3;
        }
    }

    // ---- overlap, exactly as benchmark_isqrt_diamond builds it ------------
    const std::string data = rsatb::test::data_dir() + "diamond/";
    vbcsr::atomic::Structure st = vbcsr::atomic::ReadPoscar(data + "conf.vasp");
    st.supercell = {{repeat, repeat, repeat}};
    st = st.BuildSupercell();
    const double rcut_ang = 7.0 * 0.529177210903;
    vbcsr::atomic::NeighborList nl;
    nl.build(st.pos, st.cell, {true, true, true}, 2.0 * rcut_ang + 1e-6);

    const int n_atom = st.n_atom;
    std::vector<int> atom_index(n_atom), atom_type(n_atom, 0);
    for (int i = 0; i < n_atom; ++i) atom_index[i] = i;
    const int type_norb[1] = {13};
    std::vector<int> edge_index, edge_shift;
    for (int i = 0; i < n_atom; ++i) {
        for (const auto& nb : nl.neighbors[i]) {
            edge_index.push_back(i);
            edge_index.push_back(nb.index);
            edge_shift.push_back(nb.rx);
            edge_shift.push_back(nb.ry);
            edge_shift.push_back(nb.rz);
        }
    }
    const std::size_t n_edge = edge_index.size() / 2;
    vbcsr::atomic::AtomicData ad(n_atom, n_atom, 0, n_edge, n_edge,
                                 atom_index.data(), atom_type.data(), edge_index.data(),
                                 type_norb, edge_shift.data(), st.cell.data(), st.pos.data(),
                                 MPI_COMM_WORLD, {true, true, true});

    rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
    op.init(1, 1, false, data, {"C_gga_7au_100Ry_2s2p1d.orb"}, "", {},
            "experiment_ns_variants_op2c.log");
    std::unique_ptr<vbcsr::atomic::ImageContainer<double>> img = op.build_overlap(ad);
    std::unique_ptr<Matrix> S(
        img->sample_k(kpoint, vbcsr::atomic::PhaseConvention::R_AND_POSITION));
    std::printf("diamond %dx%dx%d: %d atoms, sptol %.0e, k = (%.3f, %.3f, %.3f)\n",
                repeat, repeat, repeat, n_atom, sptol, kpoint[0], kpoint[1], kpoint[2]);

    // ---- one seed per variant (deterministic, so all start identical) ----
    // 0: adjoint (X = Z^H S Z), 1: plain (X = Z S Z), 2: plain with Z
    // re-Hermitized each step -- the classical stabilization of the plain
    // form, and the only variant whose limit satisfies BOTH congruences.
    const char* only = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[i + 1];
    }
    for (const int mode : {0, 1, 2, 3}) {
        const char* name = mode == 0 ? "adjoint" : (mode == 1 ? "plain" : (mode == 2 ? "sym" : "trans"));
        if (only != nullptr && std::strcmp(only, name) != 0) continue;
        Matrix Z(S->graph);
        rsatb::calc::graph_inverse_sqrt(*S, &Z, {}, /*verbose=*/false);

        std::printf("\n== variant: %s\n",
                    mode == 0 ? "X = Z^H S Z (adjoint)"
                              : (mode == 1 ? "X = Z S Z (plain)"
                                           : (mode == 2 ? "X = Z S Z, Z re-Hermitized each step (sym)"
                                                        : "X = Z^T S Z (plain transpose, unconjugated)")));
        std::printf("%4s %12s %12s %12s %10s\n",
                    "step", "r_adjoint", "r_driver", "herm(Z)", "density");

        for (int step = 0; step <= max_steps; ++step) {
            // The adjoint metric is always reported -- it is the contract the
            // apply path (mult_adjoint = conjugate transpose) relies on --
            // alongside whichever congruence drives this variant.
            Matrix SZ = S->spmm(Z, sptol);
            Matrix X_adj = Z.spmm(SZ, sptol, /*transA=*/true, false);
            const double r_adj = residual_of(X_adj);
            Matrix X_drv(S->graph);
            double r_drv = r_adj;
            if (mode == 1 || mode == 2) {
                X_drv = Z.spmm(SZ, sptol, false, false);
                r_drv = residual_of(X_drv);
            } else if (mode == 3) {
                Matrix Zt = Z.transpose();  // Z^H
                Zt.conjugate();             // Z^T
                X_drv = Zt.spmm(SZ, sptol, false, false);
                r_drv = residual_of(X_drv);
            }
            const double herm = hermiticity_defect(Z);
            std::printf("%4d %12.4e %12.4e %12.4e %10.4f\n",
                        step, r_adj, r_drv, herm, Z.get_block_density());
            std::fflush(stdout);
            if (step == max_steps) break;
            if (std::min(r_adj, r_drv) > 1e2) {
                std::printf("     diverged; stopping this variant\n");
                break;
            }

            Matrix& X = (mode == 0) ? X_adj : X_drv;
            Matrix ZX = Z.spmm(X, sptol);
            ZX.axpby(Scalar(1.5), Z, Scalar(-0.5));  // 1.5 Z - 0.5 Z X
            ZX.filter_blocks(sptol);
            Z = std::move(ZX);
            if (mode == 2) {
                Matrix Zh = Z.transpose();
                Z.axpby(Scalar(0.5), Zh, Scalar(0.5));  // Z <- (Z + Z^H)/2
            }
        }
    }

    MPI_Finalize();
    return 0;
}
