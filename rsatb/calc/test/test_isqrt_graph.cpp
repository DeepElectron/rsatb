// Graph-level equivalence test for the accelerated S^{-1/2} initialisation.
//
// graph_inverse_sqrt builds one block row per atom from the dense subgraph of its
// neighbourhood. The accelerated routes (Cholesky + inverse block Lanczos,
// dense-to-band Zolotarev) must reproduce, to the requested tolerance, the
// matrix the legacy per-subgraph diagonalization produces - on a real
// BlockSpMat, for both scalar types, and including the S^{-1/2} S S^{-1/2} = I
// check on the assembled result.
//
// Standalone MPI driver: mpirun -n <p> ./test_isqrt_graph

#include "../isqrtm.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace vbcsr;
using namespace rsatb::calc;

namespace {

// Overlap-like test matrix: a 1D chain of atoms with `radius` neighbours on each
// side, identity diagonal blocks and exponentially decaying off-diagonal blocks
// drawn from a fixed random symmetric pattern. The decay amplitude is scaled so
// the matrix stays symmetric positive definite by diagonal dominance, with the
// margin `1 - dominance` controlling the condition number.
template <typename T>
BlockSpMat<T> build_overlap(DistGraph& graph, int n_blocks, int block_size, int radius,
                            double dominance, bool complex_phase = false) {
    std::mt19937_64 rng(20240729u);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // One symmetric pattern per neighbour distance, normalised to unit
    // max-row-sum so the dominance bookkeeping below is exact.
    // Real and (optionally) imaginary parts of the pattern. The imaginary part
    // makes the assembled matrix genuinely complex Hermitian rather than a real
    // matrix stored in a complex type, which is what distinguishes a transpose
    // from a conjugate transpose anywhere in the pipeline.
    std::vector<std::vector<double>> pattern(radius + 1);
    std::vector<std::vector<double>> pattern_imag(radius + 1);
    std::vector<double> weight(radius + 1, 0.0);
    double total = 0.0;
    for (int d = 1; d <= radius; ++d) {
        pattern[d].assign(static_cast<std::size_t>(block_size) * block_size, 0.0);
        pattern_imag[d].assign(static_cast<std::size_t>(block_size) * block_size, 0.0);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                pattern[d][i * block_size + j] = gauss(rng);
                if (complex_phase) pattern_imag[d][i * block_size + j] = gauss(rng);
            }
        }
        double row_max = 0.0;
        for (int i = 0; i < block_size; ++i) {
            double s = 0.0;
            for (int j = 0; j < block_size; ++j) {
                s += std::hypot(pattern[d][i * block_size + j],
                                pattern_imag[d][i * block_size + j]);
            }
            row_max = std::max(row_max, s);
        }
        for (auto& v : pattern[d]) v /= row_max;
        for (auto& v : pattern_imag[d]) v /= row_max;
        weight[d] = std::exp(-0.9 * (d - 1));
        total += 2.0 * weight[d];
    }
    for (int d = 1; d <= radius; ++d) weight[d] *= dominance / total;

    BlockSpMat<T> S(&graph);
    const int n_owned = static_cast<int>(graph.owned_global_indices.size());
    std::vector<T> block(static_cast<std::size_t>(block_size) * block_size);
    for (int i = 0; i < n_owned; ++i) {
        const int gid = graph.owned_global_indices[i];
        std::fill(block.begin(), block.end(), T(0));
        for (int k = 0; k < block_size; ++k) block[k * block_size + k] = T(1);
        S.add_block(gid, gid, block.data(), block_size, block_size, AssemblyMode::INSERT,
                    kCanonicalBlockLayout);
        for (int d = 1; d <= radius; ++d) {
            for (int sign = -1; sign <= 1; sign += 2) {
                const int other = gid + sign * d;
                if (other < 0 || other >= n_blocks) continue;
                // Block (gid, other) = w (P + iQ) and block (other, gid) =
                // w (P + iQ)^H, so the assembled matrix is exactly Hermitian.
                for (int r = 0; r < block_size; ++r) {
                    for (int c = 0; c < block_size; ++c) {
                        const int fwd = r * block_size + c;
                        const int bwd = c * block_size + r;
                        const double re = (sign > 0) ? pattern[d][fwd] : pattern[d][bwd];
                        const double im =
                            (sign > 0) ? pattern_imag[d][fwd] : -pattern_imag[d][bwd];
                        if constexpr (std::is_same<T, std::complex<double>>::value) {
                            block[r * block_size + c] =
                                T(weight[d] * re, weight[d] * im);
                        } else {
                            block[r * block_size + c] = T(weight[d] * re);
                        }
                    }
                }
                S.add_block(gid, other, block.data(), block_size, block_size,
                            AssemblyMode::INSERT, kCanonicalBlockLayout);
            }
        }
    }
    S.assemble();
    return S;
}

template <typename T>
double max_block_norm(const BlockSpMat<T>& A) {
    double local = 0.0;
    for (double v : A.get_block_norms()) local = std::max(local, v);
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, A.graph->comm);
    return global;
}

// max |Z^H S Z - I| over blocks, with Z = S^{-1/2}.
template <typename T>
double identity_residual(BlockSpMat<T>& S, BlockSpMat<T>& Z, double sptol) {
    BlockSpMat<T> t = Z.spmm(S, sptol, true, false);
    t = t.spmm(Z, sptol, false, false);
    t.shift(T(-1.0));
    return max_block_norm(t);
}

int failures = 0;

template <typename T>
void run(const char* label, int n_blocks, int block_size, int radius, double dominance,
         bool verbose, bool complex_phase = false) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> block_sizes(n_blocks, block_size);
    std::vector<std::vector<int>> adj(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        for (int d = -radius; d <= radius; ++d) {
            const int j = i + d;
            if (j >= 0 && j < n_blocks) adj[i].push_back(j);
        }
    }
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, adj);

    BlockSpMat<T> S = build_overlap<T>(graph, n_blocks, block_size, radius, dominance,
                                       complex_phase);

    const int subgraph_dim = static_cast<int>(adj[n_blocks / 2].size()) * block_size;
    if (rank == 0) {
        std::printf("[%s] atoms %d, block %d, radius %d, subgraph dim %d\n", label, n_blocks,
                    block_size, radius, subgraph_dim);
    }

    struct Route { const char* env; const char* name; };
    const Route routes[] = {{"eigen", "eigen (legacy)"}, {"auto", "auto"}, {"band", "band"},
                            {"krylov", "krylov"}};

    BlockSpMat<T> reference = S.duplicate(true);
    double reference_scale = 0.0;

    for (const Route& route : routes) {
        setenv("RSATB_ISQRT_INIT", route.env, 1);
        BlockSpMat<T> Z = S.duplicate(true);
        const double t0 = MPI_Wtime();
        graph_inverse_sqrt(S, &Z, isqrt::Options{}, verbose);
        const double elapsed = MPI_Wtime() - t0;

        const double residual = identity_residual(S, Z, 1e-14);

        double deviation = 0.0;
        if (route.env == std::string("eigen")) {
            reference = Z.duplicate(true);
            reference_scale = std::max(1.0e-30, max_block_norm(reference));
        } else {
            BlockSpMat<T> diff = Z.duplicate(true);
            diff.axpby(T(-1.0), reference, T(1.0));
            deviation = max_block_norm(diff) / reference_scale;
        }

        if (rank == 0) {
            std::printf("    %-15s %7.3fs   |Z^H S Z - I| = %8.2e   rel. dev. vs legacy = %8.2e\n",
                        route.name, elapsed, residual, deviation);
        }
        // The graph approximation itself is not an exact inverse square root, so
        // the residual is a sanity bound only; agreement with the legacy route
        // is the actual assertion.
        if (deviation > 1.0e-7 || !(residual < 1.0e-1)) {
            if (rank == 0) std::printf("    FAILED (%s)\n", route.name);
            ++failures;
        }
    }
    unsetenv("RSATB_ISQRT_INIT");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose") verbose = true;
    }

    // Subgraph dimensions and block widths in the range the accelerated kernel
    // targets (one atom's orbitals against a whole neighbourhood).
    run<double>("real  b=13", 120, 13, 15, 0.98, verbose);
    run<std::complex<double>>("cplx  b=13", 120, 13, 15, 0.98, verbose);
    run<double>("real  b=4 ", 200, 4, 24, 0.995, verbose);
    // Genuinely complex Hermitian input (non-zero imaginary part): the identity
    // residual below only stays small if every transpose/conjugate-transpose in
    // the pipeline is the right one.
    run<std::complex<double>>("cplx  b=13 herm", 120, 13, 15, 0.98, verbose, true);

    if (rank == 0) {
        std::printf("\n%s (%d failing route%s)\n",
                    failures == 0 ? "GRAPH-LEVEL ROUTES AGREE" : "FAILURES", failures,
                    failures == 1 ? "" : "s");
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
