// ACE-FD against exact diagonalization on the real non-orthogonal problem.
//
// benchmarks/fermi_dirac measures the method on prescribed spectra with exact
// moments. This driver removes every idealization at once: diamond, the
// Hamiltonian read from the DeePTB file, the overlap built by the op2c
// two-centre integral engine on the same graph, S^-1/2 from the subgraph
// matrix function used as the preconditioner, and S^-1 applied by the
// preconditioned solve -- i.e. Op = S^-1 H exactly as production forms it.
// f(Op) is then applied to a random vector and compared against the dense
// generalized eigenproblem.
//
// Two quantities are reported per run because METHOD.md Sec. 7 says they
// behave differently, and that claim is the method's scope boundary:
//
//   weak   |<psi|S y_N> - <psi|S y>| / |<psi|S y>|   a linear contraction:
//                                                    what the method is for
//   vector ||y_N - y||_S / ||y||_S                   the filtered vector
//                                                    itself: what it is not for
//
// The chemical potential is swept too, because diamond is an insulator and a
// gap-centred mu makes f smooth on the whole spectrum -- an easy case that
// flatters every method equally. Putting mu inside the valence band restores
// the sharp edge the expansion actually has to resolve, which is the metallic
// situation the method targets.
//
//   OMP_NUM_THREADS=8 ./benchmark_fermi_diamond [--poscar diamond_sc444.vasp]

#include "../fermi_dirac.hpp"
#include "../preconditioner.hpp"
#include "../../model/lcao/op2c_engine.hpp"
#include "../../model/hamiltonian_images.hpp"
#include "../../io/hamiltonian_readers.hpp"
#include "../../test_data_path.hpp"
#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/atomic/io.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

extern "C" {
void dsygv_(const int* itype, const char* jobz, const char* uplo, const int* n,
            double* a, const int* lda, double* b, const int* ldb, double* w,
            double* work, const int* lwork, int* info);
}

using namespace vbcsr;
using rsatb::calc::FermiExpansionOptions;
using rsatb::calc::FermiExpansionReport;
using rsatb::calc::KpmKernel;

namespace {

constexpr double kBoltzmann = 8.617333262e-5;  // eV/K

// The diamond problem: one graph, H from the file, S from the integrals.
struct DiamondProblem {
    std::unique_ptr<atomic::AtomicData> ad;
    std::unique_ptr<atomic::ImageContainer<double>> s_img, h_img;
    std::unique_ptr<BlockSpMat<double>> S, H;
    int n_orb = 0;
    std::vector<int> row_offset;  // global orbital offset of each block row

    explicit DiamondProblem(const std::string& poscar) {
        const std::string data = rsatb::test::data_dir() + "diamond/";
        atomic::Structure st = atomic::ReadPoscar(data + poscar);

        // 2s2p1d, r_cut = 7 au. The overlap reaches 2 r_cut, and that same
        // graph has to hold the file Hamiltonian -- any H block outside it is
        // dropped, so the count is reported below rather than assumed zero.
        const double rcut_ang = 7.0 * 0.529177210903;
        atomic::NeighborList nl;
        nl.build(st.pos, st.cell, {true, true, true}, 2.0 * rcut_ang + 1e-6);

        const int n_atom = st.n_atom;
        std::vector<int> atom_index(n_atom), atom_type(n_atom, 0);
        for (int i = 0; i < n_atom; ++i) atom_index[i] = i;
        const int type_norb[1] = {13};

        // No explicit (i, i, 0) edges: build_overlap does the on-site block in
        // its own vertex loop and a duplicate would double the diagonal.
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
        const size_t n_edge = edge_index.size() / 2;

        ad.reset(new atomic::AtomicData(
            n_atom, n_atom, 0, n_edge, n_edge, atom_index.data(), atom_type.data(),
            edge_index.data(), type_norb, edge_shift.data(), st.cell.data(),
            st.pos.data(), MPI_COMM_WORLD, {true, true, true}));

        rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
        op.init(1, 1, false, data, {"C_gga_7au_100Ry_2s2p1d.orb"}, "", {},
                "fermi_diamond_opconst.log");
        s_img = op.build_overlap(*ad);

        const std::string h5 = data + "H_" + poscar.substr(0, poscar.size() - 5) + ".h5";
        auto reader = rsatb::io::OpenHamiltonianFile(
            rsatb::io::HamiltonianFile{h5, rsatb::io::HamiltonianFormat::kDeeptbHdf5});
        h_img = rsatb::model::BuildHamiltonianImagesFromFile(*ad, *reader);

        S.reset(RealAtGamma(*s_img));
        H.reset(RealAtGamma(*h_img));

        const DistGraph* g = S->graph;
        const int n_block = static_cast<int>(g->owned_global_indices.size());
        row_offset.assign(n_block, 0);
        for (int i = 0; i < n_block; ++i) {
            row_offset[i] = n_orb;
            n_orb += g->block_sizes[i];
        }
    }

    // Sums the images to k = 0 and drops the (numerically zero) imaginary part.
    static BlockSpMat<double>* RealAtGamma(atomic::ImageContainer<double>& img) {
        std::unique_ptr<BlockSpMat<std::complex<double>>> k0(
            img.sample_k({0.0, 0.0, 0.0}, atomic::PhaseConvention::R_AND_POSITION));
        auto* out = new BlockSpMat<double>(k0->graph);
        const auto& row_ptr = k0->row_ptr();
        const auto& col_ind = k0->col_ind();
        std::vector<double> buf;
        for (int i = 0; i + 1 < static_cast<int>(row_ptr.size()); ++i) {
            const int gr = k0->graph->get_global_index(i);
            const int r = k0->graph->block_sizes[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int lc = col_ind[k];
                const int c = k0->graph->block_sizes[lc];
                const std::complex<double>* blk = k0->block_data(k);
                buf.resize(static_cast<size_t>(r) * c);
                for (size_t t = 0; t < buf.size(); ++t) buf[t] = blk[t].real();
                out->add_block(gr, k0->graph->get_global_index(lc), buf.data(), r, c,
                               AssemblyMode::INSERT, MatrixLayout::RowMajor);
            }
        }
        out->assemble();
        return out;
    }

    // Column-major dense copy, for the reference eigensolve.
    std::vector<double> Dense(const BlockSpMat<double>& A) const {
        std::vector<double> out(static_cast<size_t>(n_orb) * n_orb, 0.0);
        const auto& row_ptr = A.row_ptr();
        const auto& col_ind = A.col_ind();
        for (int i = 0; i + 1 < static_cast<int>(row_ptr.size()); ++i) {
            const int r = A.graph->block_sizes[i];
            const int ro = row_offset[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int lc = col_ind[k];
                const int c = A.graph->block_sizes[lc];
                const int co = row_offset[lc];
                const double* blk = A.block_data(k);  // row-major r x c
                for (int p = 0; p < r; ++p) {
                    for (int q = 0; q < c; ++q) {
                        out[static_cast<size_t>(co + q) * n_orb + (ro + p)] =
                            blk[static_cast<size_t>(p) * c + q];
                    }
                }
            }
        }
        return out;
    }
};

double FermiOcc(double e, double mu, double temp) {
    if (temp <= 0.0) return e <= mu ? 1.0 : 0.0;
    const double x = (e - mu) / (temp * kBoltzmann);
    return x > 0.0 ? std::exp(-x) / (1.0 + std::exp(-x)) : 1.0 / (1.0 + std::exp(x));
}

struct Errors {
    double weak = 0.0;
    double vector = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // Line-buffered even when redirected. A sweep over the larger cells runs
    // for hours, and with the default block buffering on a redirected stdout
    // none of it is visible until the process exits -- so a run that is making
    // progress and one that is wedged look identical from outside.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string poscar = "diamond_sc444.vasp";
    bool adaptive_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--poscar") == 0 && i + 1 < argc) poscar = argv[++i];
        if (std::strcmp(argv[i], "--adaptive-only") == 0) adaptive_only = true;
    }

    std::printf("Building %s: H from the DeePTB file, S from op2c integrals.\n",
                poscar.c_str());
    DiamondProblem prob(poscar);
    std::printf("  %d orbitals, %d graph edges\n", prob.n_orb,
                static_cast<int>(prob.ad->n_edge));

    // --- exact reference: H C = S C eps, with C^T S C = I --------------------
    std::vector<double> A = prob.Dense(*prob.H);
    std::vector<double> B = prob.Dense(*prob.S);
    std::vector<double> eps(prob.n_orb);
    {
        const int n = prob.n_orb, itype = 1;
        int lwork = 64 * n, info = 0;
        std::vector<double> work(lwork);
        dsygv_(&itype, "V", "U", &n, A.data(), &n, B.data(), &n, eps.data(),
               work.data(), &lwork, &info);
        if (info != 0) {
            std::printf("dsygv failed, info = %d\n", info);
            MPI_Finalize();
            return 1;
        }
    }
    // A now holds C in columns.
    const int n_occ = prob.ad->n_atom * 4 / 2;  // carbon: 4 valence electrons
    const double gap_mu = 0.5 * (eps[n_occ - 1] + eps[n_occ]);
    std::printf("  spectrum [%.3f, %.3f] eV, gap %.3f eV at mu = %.3f\n",
                eps.front(), eps.back(), eps[n_occ] - eps[n_occ - 1], gap_mu);

    // --- Op = S^-1 H, with Z ~ S^-1/2 from the subgraph matrix function ------
    std::printf("Building the S^-1/2 preconditioner (graph matrix function)...\n");
    rsatb::calc::ISQRTMPreconditioner<double> Z(*prob.S, /*max_iter=*/0, 1e-5, 1e-5, "graph");

    const double pad = 0.02 * (eps.back() - eps.front());
    rsatb::calc::Chebyshev<double> cheby(*prob.H, *prob.S, Z,
                                         rsatb::calc::PreconditionerType::Symmetric,
                                         eps.front() - pad, eps.back() + pad);
    cheby.solver = rsatb::calc::LinearSolver<double>(1e-12, 2000, false);

    // --- the vector, and everything the reference needs about it ------------
    DistVector<double> psi(prob.S->graph);
    psi.set_random_normal(true);
    DistVector<double> Spsi(prob.S->graph);
    prob.S->mult(psi, Spsi);

    // d = C^T S psi. Then <psi|S f(Op)|psi> = sum_k f(eps_k) d_k^2 exactly, and
    // f(Op) psi = C f(eps) d.
    std::vector<double> d(prob.n_orb, 0.0);
    for (int k = 0; k < prob.n_orb; ++k) {
        double acc = 0.0;
        for (int i = 0; i < prob.n_orb; ++i) {
            acc += A[static_cast<size_t>(k) * prob.n_orb + i] * Spsi.data[i];
        }
        d[k] = acc;
    }

    DistVector<double> y(prob.S->graph), y_exact(prob.S->graph), Sy(prob.S->graph);

    const auto reference = [&](double mu, double temp) {
        double q = 0.0;
        std::vector<double> coeff(prob.n_orb);
        for (int k = 0; k < prob.n_orb; ++k) {
            const double f = FermiOcc(eps[k], mu, temp);
            coeff[k] = f * d[k];
            q += f * d[k] * d[k];
        }
        for (int i = 0; i < prob.n_orb; ++i) {
            double acc = 0.0;
            for (int k = 0; k < prob.n_orb; ++k) {
                acc += A[static_cast<size_t>(k) * prob.n_orb + i] * coeff[k];
            }
            y_exact.data[i] = acc;
        }
        return q;
    };

    const auto measure = [&](double q_ref) {
        prob.S->mult(y, Sy);
        double q = 0.0, num = 0.0, den = 0.0;
        for (int i = 0; i < prob.n_orb; ++i) q += psi.data[i] * Sy.data[i];
        // ||.||_S via the S-inner product of the difference.
        DistVector<double> diff(prob.S->graph), Sdiff(prob.S->graph);
        for (int i = 0; i < prob.n_orb; ++i) diff.data[i] = y.data[i] - y_exact.data[i];
        prob.S->mult(diff, Sdiff);
        for (int i = 0; i < prob.n_orb; ++i) num += diff.data[i] * Sdiff.data[i];
        prob.S->mult(y_exact, Sdiff);
        for (int i = 0; i < prob.n_orb; ++i) den += y_exact.data[i] * Sdiff.data[i];
        Errors e;
        e.weak = std::abs(q - q_ref) / std::max(std::abs(q_ref), 1e-300);
        e.vector = std::sqrt(num / std::max(den, 1e-300));
        return e;
    };

    const std::vector<int> degrees = {32, 64, 128, 256, 512, 1024, 2048};
    const std::vector<double> temps = {300.0, 30.0, 3.0};
    struct MuCase { const char* name; double mu; };
    const std::vector<MuCase> mus = {{"valence band (metal-like)", eps[n_occ / 2]},
                                     {"gap centre (insulating)", gap_mu}};

    for (const auto& mc : mus) {
        std::printf("\n================================================================\n");
        std::printf("mu = %.4f eV -- %s\n", mc.mu, mc.name);
        std::printf("================================================================\n");
        for (double temp : temps) {
            const double q_ref = reference(mc.mu, temp);
            const double n_pointwise =
                M_PI * (cheby.half_width) / (temp * kBoltzmann);
            std::printf("\n  T = %6.1f K   (pointwise degree ~ %.0f)   <psi|S f|psi> = %.10f\n",
                        temp, n_pointwise, q_ref);
            if (!adaptive_only)
            std::printf("    %-8s %-24s %-24s\n", "N", "Jackson (weak/vector)",
                        "CosTaper (weak/vector)");
            for (int n : (adaptive_only ? std::vector<int>{} : degrees)) {
                FermiExpansionOptions oj;
                oj.n_fixed = n;
                oj.summation = KpmKernel::Jackson;
                fermi_dirac_kpm(cheby, psi, y, temp, mc.mu, oj);
                const Errors ej = measure(q_ref);

                FermiExpansionOptions oc;
                oc.n_fixed = n;
                oc.summation = KpmKernel::CosTaper;
                fermi_dirac_kpm(cheby, psi, y, temp, mc.mu, oc);
                const Errors ec = measure(q_ref);

                std::printf("    %-8d %9.2e / %-11.2e %9.2e / %-11.2e\n", n, ej.weak,
                            ej.vector, ec.weak, ec.vector);
            }
            // Adaptive, swept over the tolerance asked for. The degree it picks
            // is only meaningful against a tolerance the system can actually
            // deliver: with K levels the granular part of the spectral measure
            // contributes ~ (N K)^-1/2 to the weak form (METHOD.md Sec. 3.2),
            // and asking below that floor forces the run to the cap no matter
            // how smooth the density is.
            for (double tol : {1e-2, 1e-3, 1e-4, 1e-5}) {
                FermiExpansionOptions oa;  // CosTaper, degree measured
                oa.indicator_threshold = tol;
                FermiExpansionReport rep;
                fermi_dirac_kpm(cheby, psi, y, temp, mc.mu, oa,
                                rsatb::calc::ChebyMode::NORMAL, &rep);
                const Errors ea = measure(q_ref);
                std::printf("    adaptive tol %.0e -> N = %-6d %-14s weak %.2e / vector %.2e\n",
                            tol, rep.n_used,
                            rep.indicator_fired ? "(indicator fired)" : "(hit the cap)", ea.weak,
                            ea.vector);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
