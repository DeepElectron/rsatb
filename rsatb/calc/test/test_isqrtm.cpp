// Newton-Schulz S^{-1/2} (calc/isqrtm.hpp), end to end on a 1D Laplacian chain.
//
// Assertions must stay active in this Release build. That is load-bearing
// twice: the checks below are assert()-based, AND vbcsr's padding-invariant
// guards then fire on both sides of every apply the reported probe runs --
// which is what shows the explicit zero_padding() calls that used to sit in
// that code were redundant rather than load-bearing.
#undef NDEBUG

#include "../isqrtm.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vbcsr;

namespace {

// S = 1D Laplacian + shift*I. Eigenvalues lie in [shift, 4+shift].
BlockSpMat<double> build_chain(DistGraph& graph, int n_blocks, double shift) {
    BlockSpMat<double> S(&graph);
    const int n_owned = static_cast<int>(graph.owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gid = graph.owned_global_indices[i];
        double val = 2.0 + shift;
        S.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        val = -1.0;
        if (gid > 0) {
            S.add_block(gid, gid - 1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        if (gid < n_blocks - 1) {
            S.add_block(gid, gid + 1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    S.assemble();
    return S;
}

// ||Z^H S Z v - v||_2 over independent random v, computed here rather than
// taken from the library: this is the cross-check on inverse_sqrt's own
// reported probe, and it runs the DistVector path where the library's runs the
// DistMultiVector one, so agreement is two implementations meeting.
double congruence_error(BlockSpMat<double>& S, BlockSpMat<double>& Z, DistGraph& graph, int rank) {
    double worst = 0.0;
    std::mt19937 gen(1337 + rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int t = 0; t < 5; ++t) {
        DistVector<double> v(&graph), zv(Z.graph), szv(&graph), out(&graph);
        for (int i = 0; i < v.local_size; ++i) v[i] = dis(gen);
        Z.mult_optimized(v, zv);    // Z v
        S.mult_optimized(zv, szv);  // S Z v
        Z.mult_adjoint(szv, out);   // Z^H S Z v
        double num = 0.0, den = 0.0;
        for (int i = 0; i < v.local_size; ++i) {
            num += (out[i] - v[i]) * (out[i] - v[i]);
            den += v[i] * v[i];
        }
        double g[2] = {0.0, 0.0};
        const double local[2] = {num, den};
        MPI_Allreduce(local, g, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        worst = std::max(worst, std::sqrt(g[0] / g[1]));
    }
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rsatb::util::Log::level() = rsatb::util::LogLevel::kProgress;

    const int n_blocks = 2000;
    DistGraph graph(MPI_COMM_WORLD);
    std::vector<int> block_sizes(n_blocks, 1);
    std::vector<std::vector<int>> adj(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        adj[i].push_back(i);
        if (i > 0) adj[i].push_back(i - 1);
        if (i < n_blocks - 1) adj[i].push_back(i + 1);
    }
    graph.construct_serial(n_blocks, block_sizes, adj);

    BlockSpMat<double> S = build_chain(graph, n_blocks, /*shift=*/1.0);
    BlockSpMat<double> Z = S.duplicate(true);

    rsatb::calc::InverseSqrtOptions options;
    options.tol = 1e-7;
    options.sptol = 1e-8;
    options.maxit = 50;

    rsatb::calc::InverseSqrtReport report;
    rsatb::calc::inverse_sqrt(S, &Z, options, /*verbose=*/false, &report);

    const double independent = congruence_error(S, Z, graph, rank);
    if (rank == 0) {
        std::printf("\n  steps=%d converged=%d  loop metric=%.3e\n",
                    report.steps, static_cast<int>(report.converged), report.residual);
        std::printf("  ||(Z^H S Z - I) v|| : reported %.3e, independent %.3e\n",
                    report.applied_residual, independent);
    }

    // The iteration did what it said.
    assert(report.converged);
    assert(!report.hit_maxiter);
    assert(report.residual <= options.tol);

    // The number inverse_sqrt reports is the one a caller experiences: the
    // preconditioner applied to a vector lands back within this of where it
    // started. Measured 1.3e-07 against a 1e-07 loop target.
    assert(std::isfinite(report.applied_residual));
    assert(report.applied_residual <= 1e-5);

    // ... and it is the RIGHT number, not merely a finite one. The library
    // probes one vector through the multivector kernels; the check above probes
    // five others through the plain vector kernels. Different code, different
    // vectors, so they will not match to the digit -- but a factor of ten apart
    // would mean one of the two is computing something else.
    assert(independent <= 1e-5);
    const double ratio = report.applied_residual / independent;
    assert(ratio > 0.1 && ratio < 10.0);

    if (rank == 0) std::printf("test_isqrtm PASSED\n");
    MPI_Finalize();
    return 0;
}
