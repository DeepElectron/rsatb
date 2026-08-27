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

void fill_samples(DistMultiVector<Scalar>& psi) {
    unsigned rng = 777u;
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

    MPI_Finalize();
    return 0;
}
