// Benchmark of the thin inverse-square-root action on the real production
// problem: the diamond sc888 overlap.
//
// The synthetic sweep in benchmark_isqrt_action.cpp certifies the kernel against
// prescribed spectra. This driver removes the guesswork about which spectrum an
// LCAO overlap actually has: it builds the real diamond overlap at Gamma with
// the op2c two-center integral engine (the path LcaoModel::BuildOverlapImages
// uses), extracts the very subgraph problems the graph loop feeds the kernel -
// one atom's neighbourhood, ~292 atoms x 13 orbitals ~ 3800 rows - and for each
// sampled atom reports
//
//   * the measured spectrum of that subgraph (lambda_min, lambda_max, kappa and
//     how many eigenvalues sit near zero), i.e. which regime the dispatcher is
//     actually in;
//   * wall clock and deviation for each route against the production kernel
//     (vbcsr::dense_matrix_function on the complex matrix);
//   * the block density and bandwidth of the subgraph, which decide whether a
//     sparse or banded factorization could replace the dense one.
//
// The overlap is built from integrals, not from the R-truncated S(R) in the
// DeePTB files, so it is SPD by construction, as in production.
//
// Single rank by design: this measures per-problem kernel time, which is how the
// graph loop schedules the work (one dense problem per thread, single-threaded
// BLAS). Run it as
//
//   OMP_NUM_THREADS=1 ./benchmark_isqrt_diamond [--poscar conf.vasp]
//                                              [--atoms 3] [--skip-spectrum]
//                                              [--route legacy|eigen|band|auto]
//                                              [--ns N] [--sptol X] [--tol X]
//                                              [--supercell nx ny nz]
//
// --route runs exactly one route per sampled atom, for isolated timing.
// --graph runs the whole graph_inverse_sqrt instead (use several OMP threads
// for this mode; it is the production loop). --ns N runs the full production
// inverse_sqrt with N Newton-Schulz steps, with a crash-forensics handler.

#include "../isqrtm.hpp"
#include "vbcsr/core/atomic/io.hpp"
#include "../../model/lcao/op2c_engine.hpp"
#include "../../test_data_path.hpp"
#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <csignal>
#include <cstdio>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>

namespace isq = rsatb::calc::isqrt;
namespace det = rsatb::calc::isqrt::detail;

using Scalar = std::complex<double>;  // the production scalar type
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Subgraph {
    int atom = 0;
    int dim = 0;          // total_dim of the dense problem
    int nrhs = 0;         // the central atom's orbital count
    int row_offset = 0;   // where the central atom's block starts
    std::vector<Scalar> dense;
    // Structure of the subgraph, to judge whether a sparse or banded
    // factorization could replace the dense one (see the "next" section of
    // the vbcsr tree's doc/design_isqrt_action.md).
    double block_density = 0.0;    // present blocks / all blocks
    double band_fraction = 0.0;    // half-bandwidth / block count, current order
};

// Exactly the neighbourhood the graph loop assembles for owned row `idx`.
Subgraph extract_subgraph(vbcsr::BlockSpMat<Scalar>& S, int idx) {
    vbcsr::DistGraph* graph = S.graph;
    const int global_row = graph->owned_global_indices[idx];

    std::vector<int> neighbors;
    for (int k = S.row_ptr()[idx]; k < S.row_ptr()[idx + 1]; ++k) {
        neighbors.push_back(graph->get_global_index(S.col_ind()[k]));
    }
    if (std::find(neighbors.begin(), neighbors.end(), global_row) == neighbors.end()) {
        neighbors.push_back(global_row);
    }
    std::sort(neighbors.begin(), neighbors.end());

    vbcsr::BlockSpMat<Scalar> sub = S.extract_submatrix(neighbors);
    const auto it = std::find(neighbors.begin(), neighbors.end(), global_row);
    const int block_idx = static_cast<int>(std::distance(neighbors.begin(), it));

    Subgraph out;
    out.atom = global_row;
    out.nrhs = sub.graph->block_sizes[block_idx];
    out.dim = 0;
    for (std::size_t k = 0; k < neighbors.size(); ++k) out.dim += sub.graph->block_sizes[k];
    out.row_offset = 0;
    for (int k = 0; k < block_idx; ++k) out.row_offset += sub.graph->block_sizes[k];
    out.dense = sub.to_dense();

    const int nb = static_cast<int>(neighbors.size());
    long long present = 0;
    int half_band = 0;
    for (int i = 0; i < nb; ++i) {
        for (int k = sub.graph->adj_ptr[i]; k < sub.graph->adj_ptr[i + 1]; ++k) {
            ++present;
            half_band = std::max(half_band, std::abs(i - sub.graph->adj_ind[k]));
        }
    }
    out.block_density = static_cast<double>(present) / (static_cast<double>(nb) * nb);
    out.band_fraction = static_cast<double>(half_band) / std::max(1, nb);
    return out;
}

struct Spectrum {
    double lambda_min = 0.0;
    double lambda_max = 0.0;
    int below_1e6 = 0;
    int below_1e4 = 0;
    int below_1e2 = 0;
};

// The dense problem's own spectrum, for the record: this is what decides
// whether the Krylov fast path can work at all.
Spectrum measure_spectrum(const Subgraph& sg) {
    // heevd reads the 'U' triangle only, so the plain triangle copy suffices.
    std::vector<Scalar> a = det::copy_triangle(sg.dense.data(), sg.dim, sg.dim, 'U');
    std::vector<double> w(static_cast<std::size_t>(sg.dim));
    Spectrum s;
    if (det::heevd<Scalar>('U', sg.dim, a.data(), sg.dim, w.data()) != 0) return s;
    s.lambda_min = w.front();
    s.lambda_max = w.back();
    for (double v : w) {
        if (v < 1.0e-6 * w.back()) ++s.below_1e6;
        if (v < 1.0e-4 * w.back()) ++s.below_1e4;
        if (v < 1.0e-2 * w.back()) ++s.below_1e2;
    }
    return s;
}

struct Outcome {
    double seconds = 0.0;
    isq::Stats stats;
    std::vector<Scalar> x;
};

int g_band_width = 0;  // --kd override for the band route

Outcome run_route(const Subgraph& sg, isq::Method method) {
    isq::Options opt;
    opt.method = method;
    opt.uplo = 'U';
    opt.relative_tolerance = 1.0e-8;
    opt.band_width = g_band_width;

    Outcome out;
    out.x.assign(static_cast<std::size_t>(sg.dim) * sg.nrhs, Scalar(0));
    const auto t0 = Clock::now();
    isq::inverse_sqrt_selector<Scalar>(sg.dense.data(), sg.dim, sg.dim, sg.row_offset, sg.nrhs,
                                       out.x.data(), sg.dim, opt, nullptr, &out.stats);
    out.seconds = seconds_since(t0);
    return out;
}

// The production kernel as it stands today: one dense diagonalization of the
// complex subgraph matrix, keeping only the requested columns.
Outcome run_legacy(const Subgraph& sg) {
    Outcome out;
    std::vector<Scalar> m = sg.dense;
    auto func = [](double x) { return (x > 1e-12) ? std::pow(x, -0.5) : 0.0; };
    const auto t0 = Clock::now();
    vbcsr::dense_matrix_function(sg.dim, m, std::function<Scalar(double)>(func), sg.nrhs,
                                sg.row_offset);
    out.seconds = seconds_since(t0);
    out.x = std::move(m);
    return out;
}

double relative_deviation(const std::vector<Scalar>& x, const std::vector<Scalar>& ref) {
    long double num = 0.0L, den = 0.0L;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += static_cast<long double>(det::abs2<Scalar>(x[i] - ref[i]));
        den += static_cast<long double>(det::abs2<Scalar>(ref[i]));
    }
    if (!(den > 0.0L)) return 0.0;
    return static_cast<double>(std::sqrt(num / den));
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "benchmark_isqrt_diamond measures per-problem kernel time; "
                         "run it on a single rank.\n");
        }
        MPI_Finalize();
        return 2;
    }

    std::string poscar = "conf.vasp";
    std::string only_route;
    int n_atoms_sampled = 3;
    bool with_spectrum = true;
    bool graph_mode = false;
    int ns_steps = 0;  // --ns N: run the full inverse_sqrt with N Newton-Schulz steps
    // The drop threshold decides how dense every matrix in the loop is, so it
    // decides the memory the run takes -- which makes it the one setting a
    // memory measurement has to be able to state. It was hardcoded at 1e-5,
    // and the published figures in doc/design_isqrtm_memory.md section 8 are at
    // 1e-4, so no run of this benchmark could be compared against them.
    // Sizes come from repeating the base structure, NOT from pre-tiled POSCARs:
    // one structure on disk, and the supercell path gets exercised as a side
    // effect. This is how the figures in doc/design_isqrtm_memory.md section 8
    // were produced, and without it this benchmark could not reproduce them.
    // The default is the 1024-atom cell those figures used.
    std::array<int, 3> supercell{{8, 8, 8}};
    double ns_sptol = 1e-5;
    double ns_tol = 1e-5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--poscar") == 0 && i + 1 < argc) poscar = argv[++i];
        else if (std::strcmp(argv[i], "--atoms") == 0 && i + 1 < argc) n_atoms_sampled = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--skip-spectrum") == 0) with_spectrum = false;
        else if (std::strcmp(argv[i], "--route") == 0 && i + 1 < argc) only_route = argv[++i];
        else if (std::strcmp(argv[i], "--kd") == 0 && i + 1 < argc) g_band_width = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--graph") == 0) graph_mode = true;
        else if (std::strcmp(argv[i], "--ns") == 0 && i + 1 < argc) ns_steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--supercell") == 0 && i + 3 < argc) {
            for (int axis = 0; axis < 3; ++axis) supercell[axis] = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--sptol") == 0 && i + 1 < argc) ns_sptol = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) ns_tol = std::atof(argv[++i]);
    }
    if (!only_route.empty()) with_spectrum = false;  // keep the RSS reading clean

    // ---------------------------------------------------------------- overlap
    const std::string data = rsatb::test::data_dir() + "diamond/";
    vbcsr::atomic::Structure st = vbcsr::atomic::ReadPoscar(data + poscar);
    if (supercell[0] * supercell[1] * supercell[2] > 1) {
        st.supercell = supercell;
        st = st.BuildSupercell();
    }

    // 2s2p1d orbital, r_cut = 7 au; the overlap pattern reaches 2 r_cut.
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

    std::printf("diamond %s: %d atoms, %zu edges, overlap cutoff %.3f Ang\n", poscar.c_str(),
                n_atom, n_edge, 2.0 * rcut_ang);
    std::fflush(stdout);

    vbcsr::atomic::AtomicData ad(n_atom, n_atom, /*atom_offset=*/0, n_edge, n_edge,
                                 atom_index.data(), atom_type.data(), edge_index.data(),
                                 type_norb, edge_shift.data(), st.cell.data(), st.pos.data(),
                                 MPI_COMM_WORLD, {true, true, true});

    rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
    op.init(1, 1, false, data, {"C_gga_7au_100Ry_2s2p1d.orb"}, "", {},
            "isqrt_diamond_opconst.log");
    const auto t_build = Clock::now();
    std::unique_ptr<vbcsr::atomic::ImageContainer<double>> img = op.build_overlap(ad);
    std::unique_ptr<vbcsr::BlockSpMat<Scalar>> S(
        img->sample_k({0.0, 0.0, 0.0}, vbcsr::atomic::PhaseConvention::R_AND_POSITION));
    std::printf("overlap built in %.2fs\n", seconds_since(t_build));

    // At Gamma the overlap is real; how real decides whether a real-arithmetic
    // specialisation of the kernel would be worth adding.
    {
        double max_imag = 0.0, max_real = 0.0;
        const auto& row_ptr = S->row_ptr();
        for (int i = 0; i + 1 < static_cast<int>(row_ptr.size()); ++i) {
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int r = S->graph->block_sizes[i];
                const int c = S->graph->block_sizes[S->col_ind()[k]];
                const Scalar* blk = S->block_data(k);
                for (int t = 0; t < r * c; ++t) {
                    max_imag = std::max(max_imag, std::abs(blk[t].imag()));
                    max_real = std::max(max_real, std::abs(blk[t].real()));
                }
            }
        }
        std::printf("max |Im S| / max |Re S| at Gamma = %.2e\n", max_imag / max_real);
    }
    std::fflush(stdout);

    // -------------------------------------- ns mode (production inverse_sqrt)
    if (ns_steps > 0) {
        // Crash forensics for the load-dependent SIGBUS: fault address, native
        // backtrace, and the memory map, straight to stderr (async-signal-safe
        // primitives only).
        struct Forensics {
            static void handler(int sig, siginfo_t* info, void*) {
                char line[128];
                int n = std::snprintf(line, sizeof line, "\nFATAL signal %d at address %p\n",
                                      sig, info->si_addr);
                (void)!write(2, line, n);
                void* frames[64];
                const int depth = backtrace(frames, 64);
                backtrace_symbols_fd(frames, depth, 2);
                const int mfd = open("/proc/self/maps", O_RDONLY);
                if (mfd >= 0) {
                    char buf[4096];
                    ssize_t got;
                    while ((got = read(mfd, buf, sizeof buf)) > 0) (void)!write(2, buf, got);
                    close(mfd);
                }
                _exit(97);
            }
        };
        static char alt_stack[SIGSTKSZ * 4];
        stack_t ss{alt_stack, 0, sizeof alt_stack};
        sigaltstack(&ss, nullptr);
        struct sigaction sa{};
        sa.sa_sigaction = &Forensics::handler;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigaction(SIGBUS, &sa, nullptr);
        sigaction(SIGSEGV, &sa, nullptr);

        // Per-step progress, which is where the neighbour counts, the route
        // taken and the running peak are reported. Without it this mode prints
        // one total and nothing about where the memory went.
        rsatb::util::Log::level() = rsatb::util::LogLevel::kProgress;
        // The peak in those progress lines comes from MemorySampler, which
        // only Rsatb::begin_profile() ever started. Standalone, every line
        // read "peak 0 GB" -- an actively wrong number, not a missing one --
        // and the real peak had to be polled from outside via VmHWM.
        rsatb::util::MemorySampler::instance().start();

        rsatb::calc::InverseSqrtOptions opts;
        opts.tol = ns_tol;
        opts.sptol = ns_sptol;
        opts.maxit = ns_steps;
        std::printf("ns mode: %d atoms, maxit %d, sptol %g, tol %g\n",
                    n_atom, ns_steps, ns_sptol, ns_tol);
        vbcsr::BlockSpMat<Scalar> Z(S->graph);
        const auto tn = Clock::now();
        rsatb::calc::inverse_sqrt(*S, &Z, opts, true);
        std::printf("inverse_sqrt (maxit %d) completed in %.2fs\n", ns_steps,
                    seconds_since(tn));
        MPI_Finalize();
        return 0;
    }

    // ------------------------------------------------- graph mode (whole loop)
    if (graph_mode) {
        rsatb::calc::GraphInverseSqrtReport report;
        vbcsr::BlockSpMat<Scalar> Z0(S->graph);
        const auto tg = Clock::now();
        rsatb::calc::graph_inverse_sqrt(*S, &Z0, {}, false, &report);
        std::printf("graph seed (%d threads): %.2fs (krylov %lld, band %lld, "
                    "eigen %lld, failed %lld)\n",
                    omp_get_max_threads(), seconds_since(tg), report.krylov, report.band,
                    report.eigen, report.failed);
        MPI_Finalize();
        return 0;
    }

    // ------------------------------------------------------------- benchmark
    const int n_owned = static_cast<int>(S->graph->owned_global_indices.size());
    n_atoms_sampled = std::max(1, std::min(n_atoms_sampled, n_owned));
    // Spread the samples across the cell rather than taking the first atoms.
    const int stride = std::max(1, n_owned / n_atoms_sampled);

    double total_legacy = 0.0, total_eigen = 0.0, total_auto = 0.0;
    int fast_accepted = 0;

    for (int s = 0; s < n_atoms_sampled; ++s) {
        const int idx = std::min(n_owned - 1, s * stride);
        const auto t_extract = Clock::now();
        const Subgraph sg = extract_subgraph(*S, idx);
        const double extract_seconds = seconds_since(t_extract);

        std::printf("\natom %d: subgraph dim %d, block %d, offset %d (assembled in %.2fs)\n",
                    sg.atom, sg.dim, sg.nrhs, sg.row_offset, extract_seconds);
        std::printf("  structure: block density %.2f, half-bandwidth %.2f of the block count "
                    "(atom-index order)\n",
                    sg.block_density, sg.band_fraction);
        std::fflush(stdout);

        if (with_spectrum) {
            const Spectrum sp = measure_spectrum(sg);
            std::printf("  spectrum: lambda in [%.3e, %.3e], kappa = %.2e; eigenvalues below "
                        "1e-2/1e-4/1e-6 of lambda_max: %d/%d/%d\n",
                        sp.lambda_min, sp.lambda_max,
                        sp.lambda_max / std::max(sp.lambda_min, 1.0e-300), sp.below_1e2,
                        sp.below_1e4, sp.below_1e6);
            std::fflush(stdout);
        }

        if (!only_route.empty()) {
            Outcome single;
            if (only_route == "legacy") single = run_legacy(sg);
            else if (only_route == "eigen") single = run_route(sg, isq::Method::Eigen);
            else if (only_route == "band") single = run_route(sg, isq::Method::BandZolotarev);
            else single = run_route(sg, isq::Method::Auto);
            std::printf("  %-10s %8.3fs\n", only_route.c_str(), single.seconds);
            std::fflush(stdout);
            continue;
        }

        const Outcome legacy = run_legacy(sg);
        const Outcome eigen = run_route(sg, isq::Method::Eigen);
        const Outcome band = run_route(sg, isq::Method::BandZolotarev);
        const Outcome autorun = run_route(sg, isq::Method::Auto);

        total_legacy += legacy.seconds;
        (void)band;
        total_eigen += eigen.seconds;
        total_auto += autorun.seconds;
        if (autorun.stats.used_method == isq::Method::InverseBlockLanczos) ++fast_accepted;

        // Deviations and speedups are both against the production kernel.
        std::printf("  legacy dsyevd   %8.3fs  (complex arithmetic, full diagonalization)\n",
                    legacy.seconds);
        std::printf("  eigen           %8.3fs                             dev %8.2e  speedup %5.2fx\n",
                    eigen.seconds, relative_deviation(eigen.x, legacy.x),
                    legacy.seconds / std::max(1.0e-9, eigen.seconds));
        std::printf("  band            %8.3fs  kd %3d poles %2d           dev %8.2e  speedup %5.2fx\n",
                    band.seconds, band.stats.band_width, band.stats.rational_poles,
                    relative_deviation(band.x, legacy.x),
                    legacy.seconds / std::max(1.0e-9, band.seconds));
        std::printf("  auto            %8.3fs  %-22s m %3d  dev %8.2e  speedup %5.2fx\n",
                    autorun.seconds, isq::method_name(autorun.stats.used_method),
                    autorun.stats.krylov_blocks, relative_deviation(autorun.x, legacy.x),
                    legacy.seconds / std::max(1.0e-9, autorun.seconds));
        std::fflush(stdout);
    }

    if (!only_route.empty()) {
        MPI_Finalize();
        return 0;
    }
    std::printf("\n%d subgraph problems: legacy %.2fs | eigen %.2fs (%.2fx), "
                "auto %.2fs (%.2fx); fast path accepted on %d/%d\n",
                n_atoms_sampled, total_legacy, total_eigen,
                total_legacy / std::max(1.0e-9, total_eigen), total_auto,
                total_legacy / std::max(1.0e-9, total_auto), fast_accepted, n_atoms_sampled);
    std::printf("(auto pays for a refused Krylov trial on every problem here; in the graph loop "
                "isqrt::Router stops trying after a few atoms, so the run-level figure is the "
                "representative one.)\n");

    MPI_Finalize();
    return 0;
}
