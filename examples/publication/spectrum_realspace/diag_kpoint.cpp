// k-sampled PRIMITIVE-cell reference: DOS, PDOS, density and LDOS from the
// 2-atom cell on a Monkhorst-Pack mesh, folded onto the supercell grid.
//
// WHY THIS EXISTS. A Gamma-only calculation on an n x n x n supercell and the
// primitive cell on an n x n x n Gamma-centred k-mesh are the SAME eigenproblem
// -- band folding. So this program and a dense diagonalization of the
// supercell must agree exactly, while sharing nothing but the file readers:
// one diagonalizes a 13312 x 13312 real pencil once, the other a 26 x 26
// complex pencil 512 times. Agreement between them, and of both with the
// stochastic stack, closes the loop three ways.
//
// It is also the cheap route. sc888 costs two N=13312 dense diagonalizations
// (~80 s each and 1.4 GB per matrix); the same physics here is 512 diagonal-
// izations of a 26 x 26 matrix -- milliseconds, kilobytes.
//
// CONVENTIONS.
//   * Eigenvalues and Lowdin weights are gauge invariant: the two phase
//     conventions (R_ONLY vs R_AND_POSITION) differ by a diagonal unitary
//     diag(e^{i k tau_mu}), which cannot move an eigenvalue and cannot move
//     |W_in|^2 either. DOS and PDOS are therefore convention-free, and are
//     reported per PRIMITIVE CELL so they compare directly against a supercell
//     result divided by the image count.
//   * The real-space fold is NOT convention-free, so R_ONLY is used and the
//     fold is SELF-CHECKED: the same fold applied to S(k) must reproduce the
//     supercell overlap built independently from the images. If that
//     check passes the density fold is trustworthy; if it fails the program
//     stops rather than emitting a plausible wrong map.
//
// Lowdin at each k: S(k) = U diag(s) U^H, H_perp(k) = S^-1/2 H S^-1/2, then a
// standard Hermitian eigenproblem whose eigenvectors ARE the Lowdin
// coefficients -- the dense supercell route, in complex arithmetic.

#include "calc/collocation.hpp"
#include "calc/real_space_grid.hpp"
#include "io/hamiltonian_readers.hpp"
#include "model/hamiltonian_images.hpp"
#include "model/lcao/op2c_engine.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/io.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using cd = std::complex<double>;

extern "C" {
void zheevd_(const char* jobz, const char* uplo, const int* n, cd* a, const int* lda,
             double* w, cd* work, const int* lwork, double* rwork, const int* lrwork,
             int* iwork, const int* liwork, int* info);
void zgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const cd* alpha, const cd* a, const int* lda, const cd* b,
            const int* ldb, const cd* beta, cd* c, const int* ldc);
}

namespace {

using vbcsr::BlockSpMat;
using vbcsr::atomic::AtomicData;

constexpr double kKB = 8.617333262e-5;

double wall() { return MPI_Wtime(); }

void say(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    std::vprintf(fmt, a);
    va_end(a);
    std::fflush(stdout);
}

struct Args {
    std::string poscar, ham, orbdir, orb, out = ".";
    int nk = 4;          // Gamma-centred nk x nk x nk mesh
    double electrons = 0.0, temp = 300.0, spacing = 0.4, eta = 0.25;
    std::vector<double> ldos_energies;
    bool realspace = false;

    static Args parse(int argc, char** argv) {
        Args a;
        auto next = [&](int& i) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + argv[i]);
            return std::string(argv[++i]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string f = argv[i];
            if (f == "--poscar") a.poscar = next(i);
            else if (f == "--ham") a.ham = next(i);
            else if (f == "--orbdir") a.orbdir = next(i);
            else if (f == "--orb") a.orb = next(i);
            else if (f == "--out") a.out = next(i);
            else if (f == "--kmesh") a.nk = std::atoi(next(i).c_str());
            else if (f == "--electrons") a.electrons = std::atof(next(i).c_str());
            else if (f == "--temp") a.temp = std::atof(next(i).c_str());
            else if (f == "--spacing") a.spacing = std::atof(next(i).c_str());
            else if (f == "--eta") a.eta = std::atof(next(i).c_str());
            else if (f == "--realspace") a.realspace = true;
            else if (f == "--ldos") {
                std::string v = next(i), tok;
                std::stringstream ss(v);
                while (std::getline(ss, tok, ',')) a.ldos_energies.push_back(std::atof(tok.c_str()));
            } else throw std::runtime_error("unknown flag: " + f);
        }
        if (a.poscar.empty() || a.ham.empty() || a.orb.empty()) {
            throw std::runtime_error("need --poscar, --ham, --orb, --orbdir");
        }
        return a;
    }
};

/// Dense complex, column-major, from a k-sampled block matrix.
std::vector<cd> densify(const BlockSpMat<cd>& M, int N, int norb) {
    std::vector<cd> a(static_cast<size_t>(N) * N, cd(0.0, 0.0));
    const auto& rp = M.row_ptr();
    const auto& ci = M.col_ind();
    const int nr = static_cast<int>(rp.size()) - 1;
    for (int i = 0; i < nr; ++i) {
        const int gr = M.graph->get_global_index(i);
        const int r = M.graph->block_sizes[i];
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            const int gc = M.graph->get_global_index(ci[k]);
            const int c = M.graph->block_sizes[ci[k]];
            const cd* blk = M.block_data(k);
            for (int p = 0; p < r; ++p)
                for (int q = 0; q < c; ++q)
                    a[static_cast<size_t>(gr * norb + p) +
                      static_cast<size_t>(gc * norb + q) * N] = blk[static_cast<size_t>(p) * c + q];
        }
    }
    return a;
}

void eighc(std::vector<cd>& a, int n, std::vector<double>& w) {
    w.assign(n, 0.0);
    int info = 0, lwork = -1, lrwork = -1, liwork = -1;
    cd wq;
    double rwq;
    int iwq;
    zheevd_("V", "U", &n, a.data(), &n, w.data(), &wq, &lwork, &rwq, &lrwork, &iwq,
            &liwork, &info);
    lwork = static_cast<int>(wq.real());
    lrwork = static_cast<int>(rwq);
    liwork = iwq;
    std::vector<cd> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);
    zheevd_("V", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, rwork.data(),
            &lrwork, iwork.data(), &liwork, &info);
    if (info != 0) throw std::runtime_error("zheevd failed, info=" + std::to_string(info));
}

std::vector<cd> gemmc(const std::vector<cd>& A, const char* ta, const std::vector<cd>& B,
                      const char* tb, int n) {
    std::vector<cd> C(static_cast<size_t>(n) * n, cd(0.0, 0.0));
    const cd one(1.0, 0.0), zero(0.0, 0.0);
    zgemm_(ta, tb, &n, &n, &n, &one, A.data(), &n, B.data(), &n, &zero, C.data(), &n);
    return C;
}

double fermi(double e, double mu, double kT) {
    const double u = (e - mu) / kT;
    return u > 0.0 ? std::exp(-u) / (1.0 + std::exp(-u)) : 1.0 / (1.0 + std::exp(u));
}

template <typename T>
void write_raw(const std::string& p, const std::vector<T>& v) {
    std::ofstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + p);
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
}


/// Reorders an owned-point map (box-major, the order ContractOntoBoxes fills
/// it) into GLOBAL LINEAR grid order, so the written file reshapes directly to
/// the `shape` recorded in the grid JSON. The two orders differ by a
/// permutation across the grid's boxes, which leaves every integral untouched
/// and scrambles every pointwise comparison -- exactly how it stayed hidden.
std::vector<double> ToGlobalOrder(const rsatb::calc::RealSpaceGrid& grid,
                                  const rsatb::calc::GridDomain& domain,
                                  const std::vector<double>& owned) {
    const int64_t n = domain.n_point();
    if (n == 0) return owned;
    const int64_t n_map = static_cast<int64_t>(owned.size()) / n;
    std::vector<double> out(owned.size(), 0.0);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t g = rsatb::calc::GlobalIndexOfOwnedPoint(grid, domain, k);
        for (int64_t m = 0; m < n_map; ++m) out[m * n + g] = owned[m * n + k];
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        std::fprintf(stderr, "diag_kpoint: serial only\n");
        MPI_Finalize();
        return 1;
    }
    try {
        const Args args = Args::parse(argc, argv);
        const int nk = args.nk, n_kpt = nk * nk * nk;

        // Fail on an unusable --out NOW rather than at the first write, which
        // happens only after the k loop and the fold: a missing directory used
        // to discard the entire computation (minutes at nk = 4, hours at 12).
        {
            const std::string probe = args.out + "/.diag_kpoint_write_test";
            std::ofstream f(probe);
            if (!f) {
                throw std::runtime_error(
                    "cannot write into the --out directory '" + args.out +
                    "'; create it before running (nothing is computed until it exists)");
            }
            f.close();
            std::remove(probe.c_str());
        }

        const rsatb::calc::OrbitalCollocation collocation(
            std::vector<std::string>{args.orbdir + "/" + args.orb}, MPI_COMM_WORLD);
        const double rcut = collocation.rcut_angstrom(0);
        const int norb = collocation.nphi(0);

        // ---- primitive cell -------------------------------------------------
        vbcsr::atomic::Structure prim = vbcsr::atomic::ReadPoscar(args.poscar);
        std::unique_ptr<AtomicData> pad(AtomicData::from_points(
            prim.pos, prim.z, prim.cell, prim.pbc, {rcut}, {norb}, MPI_COMM_WORLD));
        pad->primitive_n_atom = prim.n_atom;
        pad->structure_supercell = {{1, 1, 1}};
        const int n_prim = pad->n_atom;
        const int Np = n_prim * norb;
        say("primitive cell: %d atoms, %d orbitals; %d x %d x %d mesh = %d k-points\n",
            n_prim, Np, nk, nk, nk, n_kpt);

        rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
        op.init(1, 1, false, args.orbdir, {args.orb}, "", {}, "diag_kpoint_op2c.log");
        auto s_img = op.build_overlap(*pad);
        rsatb::io::HamiltonianFile hf;
        hf.path = args.ham;
        hf.format = rsatb::io::HamiltonianFormat::kDeeptbHdf5;
        auto reader = rsatb::io::OpenHamiltonianFile(hf);
        auto h_img = rsatb::model::BuildHamiltonianImagesFromFile(*pad, *reader);

        // ---- k loop ----------------------------------------------------------
        // Gamma-centred mesh, the one commensurate with an nk x nk x nk
        // supercell: k = (i/nk, j/nk, l/nk) in reciprocal lattice units, which
        // is exactly the set the supercell folds onto.
        double t0 = wall();
        std::vector<double> all_eps;                       // sorted, for the DOS
        std::vector<double> eps_k;                         // in k order, for the fold
        // Slot = orbital offset within the atom block, which names the same
        // shell on every atom ONLY while there is one type. rsatb's own
        // grouping keys on (type, slot) for exactly this reason; here the
        // assumption is asserted instead of carried silently, since this
        // reference is only ever pointed at elemental cells.
        if (pad->n_type() != 1) {
            throw std::runtime_error(
                "slot-resolved weights assume a single species; this cell has "
                "more than one atom type, so slot k is not one shell.");
        }
        std::vector<double> w_slot(static_cast<size_t>(norb) * n_kpt * Np, 0.0);
        std::vector<std::vector<cd>> W_of_k, Sinv_of_k;    // kept for the fold
        all_eps.reserve(static_cast<size_t>(n_kpt) * Np);
        double worst_norm = 0.0;
        std::vector<std::array<double, 3>> klist;

        for (int i = 0; i < nk; ++i)
        for (int j = 0; j < nk; ++j)
        for (int l = 0; l < nk; ++l) {
            const std::array<double, 3> k{{static_cast<double>(i) / nk,
                                           static_cast<double>(j) / nk,
                                           static_cast<double>(l) / nk}};
            klist.push_back(k);
            std::unique_ptr<BlockSpMat<cd>> sk(
                s_img->sample_k({k[0], k[1], k[2]}, vbcsr::atomic::PhaseConvention::R_ONLY));
            std::unique_ptr<BlockSpMat<cd>> hk(
                h_img->sample_k({k[0], k[1], k[2]}, vbcsr::atomic::PhaseConvention::R_ONLY));

            std::vector<cd> U = densify(*sk, Np, norb);
            std::vector<double> s;
            eighc(U, Np, s);
            if (!(s.front() > 0.0)) {
                throw std::runtime_error("overlap not positive definite at a k-point");
            }
            std::vector<cd> tmp(static_cast<size_t>(Np) * Np);
            for (int c = 0; c < Np; ++c) {
                const double f = 1.0 / std::sqrt(s[c]);
                for (int r = 0; r < Np; ++r)
                    tmp[static_cast<size_t>(r) + static_cast<size_t>(c) * Np] =
                        U[static_cast<size_t>(r) + static_cast<size_t>(c) * Np] * f;
            }
            std::vector<cd> Sinvhalf = gemmc(tmp, "N", U, "C", Np);

            std::vector<cd> Hd = densify(*hk, Np, norb);
            tmp = gemmc(Sinvhalf, "N", Hd, "N", Np);
            std::vector<cd> Hperp = gemmc(tmp, "N", Sinvhalf, "N", Np);
            std::vector<double> eps;
            eighc(Hperp, Np, eps);  // Hperp -> W(k), the Lowdin coefficients

            const size_t koff = all_eps.size();
            all_eps.insert(all_eps.end(), eps.begin(), eps.end());
            eps_k.insert(eps_k.end(), eps.begin(), eps.end());
            for (int n = 0; n < Np; ++n) {
                const cd* col = Hperp.data() + static_cast<size_t>(n) * Np;
                double sum = 0.0;
                for (int r = 0; r < Np; ++r) {
                    const double v = std::norm(col[r]);
                    w_slot[static_cast<size_t>(r % norb) * (n_kpt * Np) + koff + n] += v;
                    sum += v;
                }
                worst_norm = std::max(worst_norm, std::abs(sum - 1.0));
            }
            if (args.realspace) {
                W_of_k.push_back(std::move(Hperp));
                Sinv_of_k.push_back(std::move(Sinvhalf));
            }
        }
        say("k loop: %d diagonalizations in %.2f s; Lowdin norm error %.2e\n", n_kpt,
            wall() - t0, worst_norm);
        // The weights were accumulated in K-VISIT order, so sorting the
        // energies alone would pair each state's weights with a DIFFERENT
        // state's energy -- which leaves every per-state and per-slot total
        // intact (they are sums over states) and destroys only the
        // energy-resolved PDOS. Sort a permutation and apply it to both.
        {
            const size_t nstate = eps_k.size();
            std::vector<int> ord(nstate);
            for (size_t t = 0; t < nstate; ++t) ord[t] = static_cast<int>(t);
            std::stable_sort(ord.begin(), ord.end(),
                             [&](int a, int b) { return eps_k[a] < eps_k[b]; });
            std::vector<double> w_sorted(w_slot.size(), 0.0);
            for (size_t t = 0; t < nstate; ++t) {
                all_eps[t] = eps_k[ord[t]];
                for (int g = 0; g < norb; ++g) {
                    w_sorted[static_cast<size_t>(g) * nstate + t] =
                        w_slot[static_cast<size_t>(g) * nstate +
                               static_cast<size_t>(ord[t])];
                }
            }
            w_slot.swap(w_sorted);
        }
        say("folded spectrum [%.6f, %.6f] eV over %zu states\n", all_eps.front(),
            all_eps.back(), all_eps.size());

        // ---- Fermi level, per the same state-count convention ---------------
        const double kT = kKB * args.temp;
        double mu = 0.0;
        if (args.electrons > 0.0) {
            double lo = all_eps.front() - 1.0, hi = all_eps.back() + 1.0;
            for (int it = 0; it < 200; ++it) {
                mu = 0.5 * (lo + hi);
                double ne = 0.0;
                for (double e : all_eps) ne += 2.0 * fermi(e, mu, kT);
                (ne < args.electrons ? lo : hi) = mu;
            }
            say("Fermi level: mu = %.6f eV for %.1f electrons in the %dx%dx%d cell\n", mu,
                args.electrons, nk, nk, nk);
        }

        const std::string o = args.out;
        write_raw(o + "/k_eigenvalues.f64", all_eps);
        write_raw(o + "/k_lowdin_weight_slot.f64", w_slot);
        {
            std::ofstream j(o + "/kpoint_meta.json");
            j << "{\n  \"n_prim_atom\": " << n_prim << ",\n  \"n_orb_per_atom\": " << norb
              << ",\n  \"n_basis_prim\": " << Np << ",\n  \"nk\": " << nk
              << ",\n  \"n_kpoint\": " << n_kpt << ",\n  \"n_states_total\": "
              << all_eps.size() << ",\n  \"mu\": " << mu << ",\n  \"temp_k\": " << args.temp
              << ",\n  \"eps_min\": " << all_eps.front() << ",\n  \"eps_max\": "
              << all_eps.back() << ",\n  \"lowdin_norm_error\": " << worst_norm << "\n}\n";
        }
        say("wrote folded eigenvalues and Lowdin weights to %s\n", o.c_str());

        if (!args.realspace) {
            MPI_Finalize();
            return 0;
        }

        // ---- fold onto the nk x nk x nk supercell ---------------------------
        // Supercell atom ordering follows Structure::BuildSupercell: image
        // (ix, iy, iz) in that nesting, primitive atoms inside each image, so
        //     supercell atom = ((ix*nk + iy)*nk + iz) * n_prim + a
        // and the image's lattice translation is (ix, iy, iz).
        vbcsr::atomic::Structure sc = prim;
        sc.supercell = {{nk, nk, nk}};
        const auto expanded = sc.BuildSupercell();
        std::unique_ptr<AtomicData> sad(AtomicData::from_points(
            expanded.pos, expanded.z, expanded.cell, expanded.pbc, {rcut}, {norb},
            MPI_COMM_WORLD));
        sad->primitive_n_atom = prim.n_atom;
        sad->structure_supercell = sc.supercell;
        const int Ns = sad->n_atom * norb;
        say("supercell: %d atoms, N = %d\n", sad->n_atom, Ns);

        std::vector<std::array<int, 3>> image(n_kpt);
        for (int ix = 0; ix < nk; ++ix)
        for (int iy = 0; iy < nk; ++iy)
        for (int iz = 0; iz < nk; ++iz)
            image[(ix * nk + iy) * nk + iz] = {{ix, iy, iz}};

        // M_super[(I,a,mu),(J,b,nu)] = (1/Nk) sum_k e^{-i k . (R_J - R_I)} M(k)_{a mu, b nu}
        // -- the inverse DFT of the k-sampled block, which for a mesh
        // commensurate with the supercell IS the supercell matrix at Gamma.
        // The block at image pair (I, J) depends on the image DIFFERENCE
        // R_J - R_I alone, so the k sum is evaluated once per difference --
        // (2 nk - 1)^3 of them -- and then scattered to the n_kpt^2 pairs that
        // share it. That is O(N_k^2) where summing inside the pair loop was
        // O(N_k^3): ~250x at nk = 12, on what was the entire cost of a
        // --realspace run (the k loop itself is 2.5 s at nk = 8; one fold was
        // minutes, and there are five of them per run).
        //
        // Keyed on the SIGNED difference, not on d mod nk: the two give
        // mathematically equal phases, but only the signed key reproduces the
        // pair loop's arguments exactly. With the q sum in the same order,
        // every phase and every addition is the one the old code performed,
        // so the fold is BITWISE identical to the O(N_k^3) version -- which is
        // what the k8 regression against the stored reference checks.
        const int n_diff = 2 * nk - 1;
        const size_t np_blk = static_cast<size_t>(Np) * Np;
        auto diff_index = [n_diff, nk](int dx, int dy, int dz) {
            return (static_cast<size_t>(dx + nk - 1) * n_diff +
                    static_cast<size_t>(dy + nk - 1)) * n_diff +
                   static_cast<size_t>(dz + nk - 1);
        };
        auto fold = [&](const std::vector<std::vector<cd>>& Mk) {
            std::vector<std::vector<cd>> Md(static_cast<size_t>(n_diff) * n_diff * n_diff);
            for (int dx = -(nk - 1); dx <= nk - 1; ++dx)
            for (int dy = -(nk - 1); dy <= nk - 1; ++dy)
            for (int dz = -(nk - 1); dz <= nk - 1; ++dz) {
                std::vector<cd>& acc = Md[diff_index(dx, dy, dz)];
                acc.assign(np_blk, cd(0.0, 0.0));
                for (int q = 0; q < n_kpt; ++q) {
                    // sample_k uses exp(-2*pi*i*k.R) (image_container.hpp:468),
                    // so the inverse transform carries +2*pi. The sign was
                    // wrong here first; the S self-check below is what caught
                    // it, at max|dS| = 0.59 on a matrix of scale 1.
                    const double ph = 2.0 * M_PI * (klist[q][0] * dx + klist[q][1] * dy +
                                                    klist[q][2] * dz);
                    const cd w = std::polar(1.0 / n_kpt, ph);
                    const std::vector<cd>& m = Mk[q];
                    for (size_t t = 0; t < np_blk; ++t) acc[t] += w * m[t];
                }
            }
            std::vector<double> out(static_cast<size_t>(Ns) * Ns, 0.0);
            for (int I = 0; I < n_kpt; ++I) {
                for (int J = 0; J < n_kpt; ++J) {
                    const std::vector<cd>& acc = Md[diff_index(image[J][0] - image[I][0],
                                                               image[J][1] - image[I][1],
                                                               image[J][2] - image[I][2])];
                    for (int p = 0; p < Np; ++p) {
                        for (int r = 0; r < Np; ++r) {
                            const int row = I * Np + p, col = J * Np + r;
                            out[static_cast<size_t>(row) + static_cast<size_t>(col) * Ns] =
                                acc[static_cast<size_t>(p) + static_cast<size_t>(r) * Np].real();
                        }
                    }
                }
            }
            return out;
        };

        // SELF-CHECK: fold S(k) and compare against the supercell overlap built
        // independently by op2c. Passing this pins the phase convention and the
        // atom ordering; failing it means the density fold below would be a
        // plausible wrong answer, so stop instead.
        {
            t0 = wall();
            std::vector<std::vector<cd>> Sk(n_kpt);
            for (int q = 0; q < n_kpt; ++q) {
                std::unique_ptr<BlockSpMat<cd>> sk(s_img->sample_k(
                    {klist[q][0], klist[q][1], klist[q][2]},
                    vbcsr::atomic::PhaseConvention::R_ONLY));
                Sk[q] = densify(*sk, Np, norb);
            }
            const std::vector<double> S_folded = fold(Sk);

            auto s_img_sc = op.build_overlap(*sad);
            std::unique_ptr<BlockSpMat<cd>> s_sc(s_img_sc->sample_k(
                {0.0, 0.0, 0.0}, vbcsr::atomic::PhaseConvention::R_ONLY));
            const std::vector<cd> S_direct = densify(*s_sc, Ns, norb);
            double worst = 0.0, scale = 0.0;
            for (size_t t = 0; t < S_folded.size(); ++t) {
                worst = std::max(worst, std::abs(S_folded[t] - S_direct[t].real()));
                scale = std::max(scale, std::abs(S_direct[t].real()));
            }
            say("fold self-check: max |S_folded - S_supercell| = %.3e (scale %.3f, %.1f s)\n",
                worst, scale, wall() - t0);
            if (worst > 1e-8 * std::max(scale, 1.0)) {
                throw std::runtime_error(
                    "the k-space fold does not reproduce the supercell overlap; the phase "
                    "convention or the image ordering is wrong, and any density folded the "
                    "same way would be silently incorrect");
            }
        }

        // AO coefficients per k: C(k) = S^{-1/2}(k) W(k).
        std::vector<std::vector<cd>> Ck(n_kpt);
        for (int q = 0; q < n_kpt; ++q) Ck[q] = gemmc(Sinv_of_k[q], "N", W_of_k[q], "N", Np);

        auto grid = rsatb::calc::MakeRealSpaceGrid(sad->cell, args.spacing, 0);
        std::vector<int> boxes(grid.n_box());
        for (int b = 0; b < grid.n_box(); ++b) boxes[b] = b;
        auto domain = rsatb::calc::MakeGridDomain(grid, boxes);
        std::vector<double> centers, radii;
        for (int b = 0; b < grid.n_box(); ++b) {
            const auto box = grid.BoxAt(b);
            centers.insert(centers.end(), {box.center[0], box.center[1], box.center[2]});
            radii.push_back(box.radius);
        }
        const auto set = rsatb::calc::BuildOrbitalSetForSpheres(*sad, centers, radii, {rcut},
                                                                MPI_COMM_WORLD);
        const int n_set = set.n_orb();
        std::vector<int> scalar(n_set);
        for (int a = 0; a < set.n_atom(); ++a)
            for (int ob = 0; ob < set.atom_norb[a]; ++ob)
                scalar[set.orb_offset[a] + ob] = set.atoms[a] * norb + ob;
        say("grid %d x %d x %d, orbital set %d\n", grid.shape[0], grid.shape[1],
            grid.shape[2], n_set);

        // The weighted density matrix at each k, then one fold.
        auto contract = [&](const std::vector<double>& weight_of_state,
                            std::vector<double>* out) {
            std::vector<std::vector<cd>> Dk(n_kpt);
            for (int q = 0; q < n_kpt; ++q) {
                std::vector<cd> scaled(static_cast<size_t>(Np) * Np);
                for (int n = 0; n < Np; ++n) {
                    const cd* col = Ck[q].data() + static_cast<size_t>(n) * Np;
                    cd* dst = scaled.data() + static_cast<size_t>(n) * Np;
                    const double g = weight_of_state[static_cast<size_t>(q) * Np + n];
                    for (int r = 0; r < Np; ++r) dst[r] = col[r] * g;
                }
                Dk[q] = gemmc(scaled, "N", Ck[q], "C", Np);
            }
            const std::vector<double> D = fold(Dk);
            std::vector<cd> block(static_cast<size_t>(n_set) * n_set);
            for (int i = 0; i < n_set; ++i)
                for (int j = 0; j < n_set; ++j)
                    block[static_cast<size_t>(i) * n_set + j] =
                        D[static_cast<size_t>(scalar[i]) + static_cast<size_t>(scalar[j]) * Ns];
            collocation.ContractOntoBoxes(*sad, set, block, grid, domain, out);
            *out = ToGlobalOrder(grid, domain, *out);
        };

        if (args.electrons > 0.0) {
            t0 = wall();
            std::vector<double> occ(eps_k.size());
            for (size_t t = 0; t < eps_k.size(); ++t) occ[t] = 2.0 * fermi(eps_k[t], mu, kT);
            std::vector<double> dens;
            contract(occ, &dens);
            const std::vector<double>& c = sad->cell;
            const double det = c[0] * (c[4] * c[8] - c[5] * c[7]) -
                               c[1] * (c[3] * c[8] - c[5] * c[6]) +
                               c[2] * (c[3] * c[7] - c[4] * c[6]);
            double integral = 0.0;
            for (double v : dens) integral += v;
            integral *= std::abs(det) / static_cast<double>(grid.n_point());
            say("density (k-sampled, folded): integral %.6f electrons (%.1f s)\n", integral,
                wall() - t0);
            write_raw(o + "/k_density.f64", dens);
        }

        if (!args.ldos_energies.empty()) {
            t0 = wall();
            std::vector<double> maps;
            const double pref = 1.0 / (args.eta * std::sqrt(2.0 * M_PI));
            for (double energy : args.ldos_energies) {
                std::vector<double> g(eps_k.size());
                for (size_t t = 0; t < eps_k.size(); ++t) {
                    const double d = (energy - eps_k[t]) / args.eta;
                    g[t] = pref * std::exp(-0.5 * d * d);
                }
                std::vector<double> one;
                contract(g, &one);
                maps.insert(maps.end(), one.begin(), one.end());
            }
            say("ldos (k-sampled, folded): %zu energies (%.1f s)\n", args.ldos_energies.size(),
                wall() - t0);
            write_raw(o + "/k_ldos.f64", maps);
            write_raw(o + "/k_ldos_energies.f64", args.ldos_energies);
        }
        {
            std::ofstream j(o + "/k_grid.json");
            j << "{\n  \"shape\": [" << grid.shape[0] << ", " << grid.shape[1] << ", "
              << grid.shape[2] << "],\n  \"n_point\": " << grid.n_point()
              << ",\n  \"spacing\": " << args.spacing << ",\n  \"eta\": " << args.eta
              << ",\n  \"nk\": " << nk << "\n}\n";
        }
        say("wrote folded real-space maps to %s\n", o.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "diag_kpoint: %s\n", e.what());
        MPI_Finalize();
        return 2;
    }
    MPI_Finalize();
    return 0;
}
