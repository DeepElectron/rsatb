// Wall-clock of the diffusion-spectrum MSD loop, whose cost is the per-step
// KPM moment recurrence (n_steps * n_kpm applies of the scaled operator; the
// time propagation is the small term -- see the metric_density_moments note
// in msd.hpp). Two fixtures, matching the two operator regimes:
//
//   orthogonal   3D tight-binding lattice, diagonal position, SpMV applies.
//                Production-shaped ratio: dt_fs = 10 puts the propagation at
//                ~2 x 140 coefficient applies per step against n_kpm = 1000
//                moment applies.
//   solve        same lattice with S = 1 + 0.08 * adjacency, explicit
//                position matrix: every apply is a CG solve, the regime
//                where halving the apply count pays the most.
//
// Deterministic seeds and a printed spectrum checksum, so two builds of this
// file can be compared value-for-value as well as second-for-second.
//
//   OMP_NUM_THREADS=8 ./benchmark_msd_moments

#include "../msd.hpp"
#include "../msd_spectral.hpp"
#include "../../utils/profile.hpp"

#include <mpi.h>

#include <complex>
#include <cstdio>
#include <vector>

using namespace vbcsr;
using namespace rsatb::calc;

namespace {

using Scalar = std::complex<double>;

struct Lattice {
    int L = 0;
    int n = 0;
    std::vector<std::vector<int>> adj;  // open boundaries, self + neighbours

    explicit Lattice(int side) : L(side), n(side * side * side), adj(n) {
        auto id = [&](int x, int y, int z) { return (x * L + y) * L + z; };
        for (int x = 0; x < L; ++x)
            for (int y = 0; y < L; ++y)
                for (int z = 0; z < L; ++z) {
                    auto& row = adj[id(x, y, z)];
                    row.push_back(id(x, y, z));
                    if (x > 0) row.push_back(id(x - 1, y, z));
                    if (x + 1 < L) row.push_back(id(x + 1, y, z));
                    if (y > 0) row.push_back(id(x, y - 1, z));
                    if (y + 1 < L) row.push_back(id(x, y + 1, z));
                    if (z > 0) row.push_back(id(x, y, z - 1));
                    if (z + 1 < L) row.push_back(id(x, y, z + 1));
                }
    }
};

// Small deterministic LCG so both builds see identical matrices and seeds.
double lcg_uniform(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
}

DistGraph make_graph(const Lattice& lat, MPI_Comm comm) {
    std::vector<int> block_sizes(lat.n, 1);
    DistGraph graph(comm);
    graph.construct_serial(lat.n, block_sizes, lat.adj);
    return graph;
}

void fill_hamiltonian(BlockSpMat<Scalar>& H, const Lattice& lat) {
    unsigned rng = 12345u;
    for (int local = 0; local < static_cast<int>(H.graph->owned_global_indices.size()); ++local) {
        const int row = H.graph->owned_global_indices[local];
        for (int col : lat.adj[row]) {
            if (col == row) {
                Scalar onsite(lcg_uniform(rng) - 0.5, 0.0);
                // rng must advance identically on every row regardless of
                // ownership, so the disorder is rank-count independent; with
                // construct_serial every rank owns a contiguous slice and the
                // loop order is global order, so a plain draw is fine here.
                H.add_block(row, col, &onsite, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            } else if (col > row) {
                Scalar hop(-1.0, 0.0);
                H.add_block(row, col, &hop, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
                H.add_block(col, row, &hop, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            }
        }
    }
    H.assemble();
}

void fill_overlap(BlockSpMat<Scalar>& S, const Lattice& lat, double off) {
    for (int local = 0; local < static_cast<int>(S.graph->owned_global_indices.size()); ++local) {
        const int row = S.graph->owned_global_indices[local];
        for (int col : lat.adj[row]) {
            if (col == row) {
                Scalar one(1.0, 0.0);
                S.add_block(row, col, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            } else if (col > row) {
                Scalar v(off, 0.0);
                S.add_block(row, col, &v, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
                S.add_block(col, row, &v, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            }
        }
    }
    S.assemble();
}

// x-coordinate of each site, as diagonal data (Angstrom-flavoured spacing).
void fill_position(DistVector<Scalar>& x, const Lattice& lat) {
    for (int local = 0; local < x.local_size; ++local) {
        const int row = x.graph->owned_global_indices[local];
        x[local] = Scalar(2.5 * static_cast<double>(row / (lat.L * lat.L)), 0.0);
    }
}

void fill_position_matrix(BlockSpMat<Scalar>& X, const Lattice& lat) {
    for (int local = 0; local < static_cast<int>(X.graph->owned_global_indices.size()); ++local) {
        const int row = X.graph->owned_global_indices[local];
        Scalar v(2.5 * static_cast<double>(row / (lat.L * lat.L)), 0.0);
        X.add_block(row, row, &v, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    X.assemble();
}

void fill_samples(DistMultiVector<Scalar>& psi, unsigned seed = 777u) {
    unsigned rng = seed;
    for (int col = 0; col < psi.num_vectors; ++col) {
        unsigned col_rng = rng + 1000003u * static_cast<unsigned>(col);
        for (int local = 0; local < psi.local_rows; ++local) {
            psi(local, col) = Scalar(lcg_uniform(col_rng) - 0.5, lcg_uniform(col_rng) - 0.5);
        }
    }
}

double checksum(const std::vector<double>& msd) {
    double acc = 0.0;
    for (size_t i = 0; i < msd.size(); ++i) {
        acc += msd[i] * static_cast<double>((i % 97) + 1) * 1e-3;
    }
    return acc;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const Lattice lat(20);  // 8000 sites, orthogonal fixture
    {
        DistGraph graph = make_graph(lat, MPI_COMM_WORLD);
        BlockSpMat<Scalar> H(&graph);
        fill_hamiltonian(H, lat);
        DistVector<Scalar> x(&graph);
        fill_position(x, lat);
        DistMultiVector<Scalar> psi(&graph, 8);
        fill_samples(psi);

        Chebyshev<Scalar> cheby(H, -7.0, 7.0);
        std::vector<double> msd, energy;
        const double t0 = rsatb::util::wall_time();
        corr_msd(cheby, x, psi, 8, 10.0, 1000, msd, energy);
        const double t1 = rsatb::util::wall_time();
        std::printf("orthogonal  n=%d nv=8 steps=8 n_kpm=1000 dt=10fs   %8.2f s   checksum % .12e\n",
                    lat.n, t1 - t0, checksum(msd));

        // Moment-level validation at this size: the doubled recurrence
        // against an inline plain one-moment-per-apply chain on the same
        // seeds. (The spectrum checksum above divides by the density, which
        // is Jackson-tiny at the window edges and amplifies rounding there;
        // the moments themselves are the like-for-like comparison.)
        {
            const int n_kpm = 1000;
            std::vector<std::complex<double>> doubled =
                rsatb::calc::detail::metric_density_moments(cheby, psi, n_kpm);

            DistMultiVector<Scalar> t0v(&graph, psi.num_vectors), t1v(&graph, psi.num_vectors),
                t2v(&graph, psi.num_vectors);
            std::vector<Scalar> dots(static_cast<size_t>(psi.num_vectors));
            auto averaged = [&]() {
                std::complex<double> sum(0.0, 0.0);
                for (const auto& d : dots) sum += static_cast<std::complex<double>>(d);
                return sum / static_cast<double>(psi.num_vectors);
            };
            std::vector<std::complex<double>> plain(n_kpm);
            t0v.copy_from(psi);
            t0v.dot(t0v, dots);
            plain[0] = averaged();
            cheby.apply_op(t0v, t1v, ChebyMode::NORMAL);
            psi.dot(t1v, dots);
            plain[1] = averaged();
            for (int n = 2; n < n_kpm; ++n) {
                cheby.apply_op(t1v, t2v, ChebyMode::NORMAL);
                t2v.axpby(Scalar(-1.0), t0v, Scalar(2.0));
                psi.dot(t2v, dots);
                plain[static_cast<size_t>(n)] = averaged();
                t0v.swap(t1v);
                t1v.swap(t2v);
            }
            double max_diff = 0.0, max_mag = 0.0;
            for (int n = 0; n < n_kpm; ++n) {
                max_diff = std::max(max_diff, std::abs(doubled[static_cast<size_t>(n)] -
                                                       plain[static_cast<size_t>(n)]));
                max_mag = std::max(max_mag, std::abs(plain[static_cast<size_t>(n)]));
            }
            std::printf("            moments doubled vs plain: max |diff| %.3e (max |mu| %.3e)\n",
                        max_diff, max_mag);
        }
    }

    const Lattice small(12);  // 1728 sites, solve fixture (CG per apply)
    {
        DistGraph graph = make_graph(small, MPI_COMM_WORLD);
        BlockSpMat<Scalar> H(&graph);
        fill_hamiltonian(H, small);
        BlockSpMat<Scalar> S(&graph);
        fill_overlap(S, small, 0.08);
        BlockSpMat<Scalar> X(&graph);
        fill_position_matrix(X, small);
        DistMultiVector<Scalar> psi(&graph, 4);
        fill_samples(psi);

        Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
        std::vector<double> msd, energy;
        const double t0 = rsatb::util::wall_time();
        corr_msd(cheby, X, psi, 3, 1.0, 300, msd, energy);
        const double t1 = rsatb::util::wall_time();
        std::printf("solve       n=%d nv=4 steps=3 n_kpm=300  dt=1fs    %8.2f s   checksum % .12e\n",
                    small.n, t1 - t0, checksum(msd));
    }

    // Stepwise vs spectral on the Solve form, in the regime the spectral
    // route targets: expensive applies (CG each) and a fine output grid.
    // The commutator (stepwise) and VAC (spectral) routes run on the SAME
    // velocity operator, so the two checksums describe the same estimator
    // family; small differences are the per-sample estimator arrangement.
    {
        DistGraph graph = make_graph(small, MPI_COMM_WORLD);
        BlockSpMat<Scalar> H(&graph);
        fill_hamiltonian(H, small);
        BlockSpMat<Scalar> S(&graph);
        fill_overlap(S, small, 0.08);
        // V = i[H, X] for the diagonal lattice X: V_ij = i H_ij (x_j - x_i),
        // Hermitian, same sparsity as H.
        BlockSpMat<Scalar> V(&graph);
        {
            auto xcoord = [&](int row) {
                return 2.5 * static_cast<double>(row / (small.L * small.L));
            };
            for (int local = 0; local < static_cast<int>(V.graph->owned_global_indices.size()); ++local) {
                const int row = V.graph->owned_global_indices[local];
                for (int col : small.adj[row]) {
                    if (col == row) continue;
                    Scalar hop(-1.0, 0.0);
                    Scalar val = Scalar(0.0, 1.0) * hop * Scalar(xcoord(col) - xcoord(row), 0.0);
                    V.add_block(row, col, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
                }
            }
            V.assemble();
        }
        DistMultiVector<Scalar> psi(&graph, 2);
        fill_samples(psi);

        const int n_steps = 48;
        const double dt_fs = 0.25;
        const int n_kpm = 300;

        std::vector<double> msd_a, msd_b, energy;
        double t_step, t_spec;
        {
            Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
            const double t0 = rsatb::util::wall_time();
            corr_msd(cheby, VelocityOp<Scalar>(V), psi, n_steps, dt_fs, n_kpm, msd_a, energy);
            t_step = rsatb::util::wall_time() - t0;
        }
        {
            Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
            const double t0 = rsatb::util::wall_time();
            corr_msd_spectral(cheby, VelocityOp<Scalar>(V), psi, n_steps, dt_fs, n_kpm, msd_b, energy);
            t_spec = rsatb::util::wall_time() - t0;
        }
        std::printf("solve fine grid  n=%d nv=2 steps=%d n_kpm=%d dt=%.2ffs\n",
                    small.n, n_steps, n_kpm, dt_fs);
        std::printf("  stepwise  %8.2f s   checksum % .12e\n", t_step, checksum(msd_a));
        std::printf("  spectral  %8.2f s   checksum % .12e   speedup %.2fx\n",
                    t_spec, checksum(msd_b), t_step / t_spec);
        // Route agreement where the density has weight (middle half of the
        // window, final step). The routes are different unbiased estimators
        // of one trace, so this is finite-sample scatter, not truncation.
        double max_rel = 0.0;
        for (int i = n_kpm / 4; i < 3 * n_kpm / 4; ++i) {
            const size_t idx = static_cast<size_t>(n_steps) * n_kpm + static_cast<size_t>(i);
            const double denom = std::max(std::abs(msd_a[idx]), std::abs(msd_b[idx]));
            if (denom > 1e-8) {
                max_rel = std::max(max_rel, std::abs(msd_a[idx] - msd_b[idx]) / denom);
            }
        }
        std::printf("  mid-spectrum route deviation (final step): %.3f\n", max_rel);
        // Noise floor: the SAME stepwise route on an independent sample set,
        // same metric. Cross-route deviation at or below this level is
        // finite-sample scatter, not a systematic difference.
        {
            DistMultiVector<Scalar> psi2(&graph, 2);
            fill_samples(psi2, 424243u);
            Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
            std::vector<double> msd_c;
            corr_msd(cheby, VelocityOp<Scalar>(V), psi2, n_steps, dt_fs, n_kpm, msd_c, energy);
            double floor_rel = 0.0;
            for (int i = n_kpm / 4; i < 3 * n_kpm / 4; ++i) {
                const size_t idx = static_cast<size_t>(n_steps) * n_kpm + static_cast<size_t>(i);
                const double denom = std::max(std::abs(msd_a[idx]), std::abs(msd_c[idx]));
                if (denom > 1e-8) {
                    floor_rel = std::max(floor_rel, std::abs(msd_a[idx] - msd_c[idx]) / denom);
                }
            }
            std::printf("  stepwise seed-to-seed noise floor:         %.3f\n", floor_rel);
        }
        // Moment-level LP fidelity at lattice scale: extrapolate the density
        // moments 100 -> 300 and compare against the directly computed tail
        // (same seeds; per-moment stochastic noise is percent-level, unlike
        // the per-bin spectra, so this isolates the model error).
        {
            Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
            std::vector<std::complex<double>> full300 =
                rsatb::calc::detail::metric_density_moments(cheby, psi, 300);
            std::vector<std::complex<double>> head(full300.begin(), full300.begin() + 100);
            // Signal vs noise in the tail itself: an independent sample
            // set's moments differ from these by pure stochastic noise, so
            // band-wise |difference| / |mean| reads the noise fraction.
            {
                DistMultiVector<Scalar> psi3(&graph, 2);
                fill_samples(psi3, 987651u);
                std::vector<std::complex<double>> alt300 =
                    rsatb::calc::detail::metric_density_moments(cheby, psi3, 300);
                for (int q0 : {10, 60, 120, 200, 260}) {
                    double sig = 0.0, noi = 0.0;
                    for (int q = q0; q < q0 + 30; ++q) {
                        sig += std::norm(0.5 * (full300[static_cast<size_t>(q)] + alt300[static_cast<size_t>(q)]));
                        noi += std::norm(full300[static_cast<size_t>(q)] - alt300[static_cast<size_t>(q)]);
                    }
                    std::printf("  moment band q=%3d..%3d: |mean| %.3e, seed-diff %.3e\n",
                                q0, q0 + 29, std::sqrt(sig / 30), std::sqrt(noi / 30));
                }
            }
            for (int order : {25, 40}) {
                std::vector<std::complex<double>> lp =
                    extrapolate_moments_lp(head, 300, order);
                double num = 0.0, den = 0.0;
                for (int q = 100; q < 300; ++q) {
                    num += std::norm(lp[static_cast<size_t>(q)] - full300[static_cast<size_t>(q)]);
                    den += std::norm(full300[static_cast<size_t>(q)]);
                }
                std::printf("  density-moment LP tail (order %d): rel err %.4f\n",
                            order, std::sqrt(num / den));
            }
        }
        // Linear-prediction knob at lattice scale, same seeds as the full
        // runs above, so the deviation IS the model error (no stochastic
        // confound): compute a third of the moments per step, extrapolate.
        {
            const int n_compute = 100;
            Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
            std::vector<double> msd_lp;
            const double t0 = rsatb::util::wall_time();
            corr_msd(cheby, VelocityOp<Scalar>(V), psi, n_steps, dt_fs, n_kpm, msd_lp, energy,
                     n_compute);
            const double t_lp = rsatb::util::wall_time() - t0;
            double dev = 0.0;
            for (int i = n_kpm / 4; i < 3 * n_kpm / 4; ++i) {
                const size_t idx = static_cast<size_t>(n_steps) * n_kpm + static_cast<size_t>(i);
                const double denom = std::max(std::abs(msd_a[idx]), std::abs(msd_lp[idx]));
                if (denom > 1e-8) dev = std::max(dev, std::abs(msd_a[idx] - msd_lp[idx]) / denom);
            }
            std::printf("  stepwise + LP (compute %d of %d)  %8.2f s   mid-spectrum dev vs full %.4f   speedup %.2fx\n",
                        n_compute, n_kpm, t_lp, dev, t_step / t_lp);
        }
        {
            const int n_compute = 100;
            Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
            std::vector<double> msd_lp;
            const double t0 = rsatb::util::wall_time();
            corr_msd_spectral(cheby, VelocityOp<Scalar>(V), psi, n_steps, dt_fs, n_kpm, msd_lp,
                              energy, 2, 0, n_compute);
            const double t_lp = rsatb::util::wall_time() - t0;
            double dev = 0.0;
            for (int i = n_kpm / 4; i < 3 * n_kpm / 4; ++i) {
                const size_t idx = static_cast<size_t>(n_steps) * n_kpm + static_cast<size_t>(i);
                const double denom = std::max(std::abs(msd_b[idx]), std::abs(msd_lp[idx]));
                if (denom > 1e-8) dev = std::max(dev, std::abs(msd_b[idx] - msd_lp[idx]) / denom);
            }
            std::printf("  spectral + LP (compute %d of %d)  %8.2f s   mid-spectrum dev vs full %.4f   speedup %.2fx\n",
                        n_compute, n_kpm, t_lp, dev, t_spec / t_lp);
        }

        // Sampling-density sweep at the same fixed physical window (12 fs):
        // the stepwise cost scales with the number of output samples, the
        // spectral cost is set by window + resolution alone. This is the
        // input dependence the route choice hinges on -- the crossover sits
        // near dt ~ n_kpm * hbar / (5 * half_width).
        for (const auto& cfg : {std::pair<int, double>{12, 1.0},
                                std::pair<int, double>{3, 4.0},
                                std::pair<int, double>{48, 4.0},
                                std::pair<int, double>{192, 1.0}}) {
            const int ns = cfg.first;
            const double dt = cfg.second;
            std::vector<double> ma, mb;
            double ta, tb;
            {
                Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
                const double t0 = rsatb::util::wall_time();
                corr_msd(cheby, VelocityOp<Scalar>(V), psi, ns, dt, n_kpm, ma, energy);
                ta = rsatb::util::wall_time() - t0;
            }
            {
                Chebyshev<Scalar> cheby(H, S, -13.0, 13.0);
                const double t0 = rsatb::util::wall_time();
                corr_msd_spectral(cheby, VelocityOp<Scalar>(V), psi, ns, dt, n_kpm, mb, energy);
                tb = rsatb::util::wall_time() - t0;
            }
            std::printf("  sweep steps=%2d dt=%.2ffs: stepwise %7.2f s, spectral %7.2f s, ratio %.2fx\n",
                        ns, dt, ta, tb, ta / tb);
        }
    }

    MPI_Finalize();
    return 0;
}
