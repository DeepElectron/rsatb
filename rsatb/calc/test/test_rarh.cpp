// vbcsr's fused triple product rarh_upper: C = A B A^H, upper triangle only,
// computed without ever forming A B.
//
// This kernel is on the Newton-Schulz critical path (isqrtm.hpp, phase
// setup/isqrt_ns_P_X_P) and replaced a two-call route, so what needs pinning is
// not just "it multiplies" but the two properties that justified fusing:
//
//   (1) it is EXACT at threshold 0, to a dense reference -- the row-at-a-time
//       accumulation and the A[j,l]^H == A[l,j] shortcut must not cost
//       accuracy;
//   (2) it is exactly Hermitian once completed, which everything downstream
//       assumes; and
//   (3) thresholded, it is no LESS accurate than the two-call route at the same
//       kept-block count. Not at the same threshold: the fused kernel drops
//       once, on the finished result, where the two-call route drops twice, so
//       the same threshold leaves the two holding different numbers of blocks
//       and comparing there measures the block count, not the method. Matching
//       nnz first is what makes it a comparison.
//
// Synthetic matrices, no integral engine: milliseconds, and every value is a
// deterministic function of its GLOBAL orbital indices, so the reference is
// identical at any rank count. Runs at any comm size; the CMake registration
// drives it at 1, 2, 3 and 4.

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_graph.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using Scalar = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<Scalar>;

namespace {

constexpr int kBlocks = 6;

// Deterministic pseudo-random in [-1, 1) from an integer key. Not an RNG
// stream: a stream would hand different values to different ranks depending on
// how many blocks each one built, and then the "reference" would disagree with
// the matrix by construction.
double jitter(unsigned key, unsigned salt) {
    unsigned k = key * 2654435761u + salt * 40503u;
    k ^= k >> 15;
    k *= 2246822519u;
    k ^= k >> 13;
    k *= 3266489917u;
    k ^= k >> 16;
    return static_cast<double>(k % 100000u) / 50000.0 - 1.0;
}

// Entry (R, C) of a Hermitian matrix, R and C being GLOBAL orbital indices.
// Hermitian by construction rather than by mirroring afterwards: the key is
// symmetric in (R, C) and the imaginary part changes sign with the order, so
// M[R][C] == conj(M[C][R]) holds exactly, including a real diagonal.
//
// decay makes the magnitude fall away from the diagonal, so that a threshold
// has something to drop. A flat random matrix would make the thresholded case
// below vacuous -- every block would survive or none would.
Scalar hermitian_entry(int R, int C, unsigned salt, double decay) {
    const int lo = std::min(R, C);
    const int hi = std::max(R, C);
    const unsigned key = static_cast<unsigned>(lo) * 7919u + static_cast<unsigned>(hi);
    const double scale = std::exp(-decay * static_cast<double>(hi - lo));
    const double re = jitter(key, salt) * scale;
    if (R == C) return Scalar(re, 0.0);
    const double im = jitter(key, salt + 1u) * scale;
    return Scalar(re, R < C ? im : -im);
}

// Global scalar offset of each block row, and the total dimension.
std::vector<int> scalar_offsets(const std::vector<int>& bs) {
    std::vector<int> off(bs.size() + 1, 0);
    for (size_t i = 0; i < bs.size(); ++i) off[i + 1] = off[i] + bs[i];
    return off;
}

// A contiguous slice of the block rows per rank, full adjacency. The remainder
// is spread over the low ranks rather than dumped on the last one, so at 4
// ranks the split is 2/2/1/1 and the rank counts the suite drives do not all
// happen to divide kBlocks evenly.
// `empty_row` >= 0 carves a pattern-Hermitian hole: that block row keeps no
// blocks and no other row keeps a block in that column. The row still EXISTS
// (it is owned, it has a size, it partitions like any other) -- it just stores
// nothing, which is a state filter_blocks can produce and a distributed kernel
// has to answer for.
void build_graph(vbcsr::DistGraph& graph, const std::vector<int>& bs, MPI_Comm comm,
                 int empty_row = -1) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int base = kBlocks / size;
    const int extra = kBlocks % size;
    const int begin = rank * base + std::min(rank, extra);
    const int end = begin + base + (rank < extra ? 1 : 0);

    std::vector<int> owned, sizes;
    std::vector<std::vector<int>> adj;
    for (int g = begin; g < end; ++g) {
        owned.push_back(g);
        sizes.push_back(bs[g]);
        std::vector<int> row;
        if (g != empty_row) {
            for (int j = 0; j < kBlocks; ++j) {
                if (j != empty_row) row.push_back(j);
            }
        }
        adj.push_back(std::move(row));
    }
    graph.construct_distributed(owned, sizes, adj);
}

// Values follow the graph's own adjacency, so the same generator fills a full
// graph and a holed one.
Matrix build(vbcsr::DistGraph* graph, const std::vector<int>& bs, unsigned salt,
             double decay) {
    const std::vector<int> off = scalar_offsets(bs);
    Matrix m(graph);
    const int n_owned = static_cast<int>(graph->owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gi = graph->owned_global_indices[i];
        const int r_dim = bs[gi];
        for (int k = graph->adj_ptr[i]; k < graph->adj_ptr[i + 1]; ++k) {
            const int gj = graph->get_global_index(graph->adj_ind[k]);
            const int c_dim = bs[gj];
            std::vector<Scalar> block(static_cast<size_t>(r_dim) * c_dim);
            for (int r = 0; r < r_dim; ++r)
                for (int c = 0; c < c_dim; ++c)
                    block[static_cast<size_t>(r) * c_dim + c] =
                        hermitian_entry(off[gi] + r, off[gj] + c, salt, decay);
            m.add_block(gi, gj, block.data(), r_dim, c_dim);
        }
    }
    m.assemble();
    return m;
}

// The whole matrix as dense, on every rank.
//
// The Allreduce is the point. BlockSpMat::to_dense() returns this rank's OWNED
// rows over its LOCAL columns (ghosts included, in local order) -- a different
// shape and a different column ordering on every rank. Comparing that against a
// global reference fails at every rank count including one, and the failure
// looks like a kernel bug rather than a harness bug. Each block row has exactly
// one owner, so summing the per-rank contributions is exact, not approximate.
std::vector<Scalar> gather_dense(const Matrix& m, const std::vector<int>& bs) {
    const std::vector<int> off = scalar_offsets(bs);
    const int total = off.back();
    std::vector<Scalar> dense(static_cast<size_t>(total) * total, Scalar(0.0));

    vbcsr::DistGraph* g = m.graph;
    const int n_owned = static_cast<int>(g->owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gi = g->owned_global_indices[i];
        for (int k = g->adj_ptr[i]; k < g->adj_ptr[i + 1]; ++k) {
            const int lc = g->adj_ind[k];
            // Local index convention: owned first, then ghosts.
            const int gj = (lc < n_owned) ? g->owned_global_indices[lc]
                                          : g->ghost_global_indices[lc - n_owned];
            const int r_dim = bs[gi];
            const int c_dim = bs[gj];
            const std::vector<Scalar> block = m.get_block(i, lc);
            if (block.empty()) continue;
            for (int r = 0; r < r_dim; ++r)
                for (int c = 0; c < c_dim; ++c)
                    dense[static_cast<size_t>(off[gi] + r) * total + off[gj] + c] =
                        block[static_cast<size_t>(r) * c_dim + c];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, dense.data(), 2 * total * total, MPI_DOUBLE, MPI_SUM,
                  g->comm);
    return dense;
}

// C = A B A^H, dense, from the same generators the sparse matrices were built
// from -- an independent path to the answer, not a rearrangement of the kernel.
// `empty_block` >= 0 zeroes A's rows and columns over that block's scalar
// range, matching the holed graph above; B is never holed.
// Every block filled with one real constant. Hermitian trivially (a real
// constant equals its own conjugate transpose), and -- unlike the decaying
// pseudo-random matrices above -- every product in the contraction adds with
// the same sign, so nothing cancels. That is what makes it a probe of the
// pruning bound rather than of the arithmetic.
Matrix build_constant(vbcsr::DistGraph* graph, const std::vector<int>& bs, double value) {
    Matrix m(graph);
    const int n_owned = static_cast<int>(graph->owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gi = graph->owned_global_indices[i];
        const int r_dim = bs[gi];
        for (int k = graph->adj_ptr[i]; k < graph->adj_ptr[i + 1]; ++k) {
            const int gj = graph->get_global_index(graph->adj_ind[k]);
            const int c_dim = bs[gj];
            std::vector<Scalar> block(static_cast<size_t>(r_dim) * c_dim, Scalar(value, 0.0));
            m.add_block(gi, gj, block.data(), r_dim, c_dim);
        }
    }
    m.assemble();
    return m;
}

std::vector<Scalar> dense_reference(const std::vector<int>& bs, unsigned salt_a,
                                    double decay_a, unsigned salt_b, double decay_b,
                                    int empty_block = -1) {
    const std::vector<int> off = scalar_offsets(bs);
    const int total = off.back();
    const int hole_lo = empty_block >= 0 ? off[empty_block] : 0;
    const int hole_hi = empty_block >= 0 ? off[empty_block + 1] : 0;
    auto in_hole = [&](int s) { return empty_block >= 0 && s >= hole_lo && s < hole_hi; };
    std::vector<Scalar> A(static_cast<size_t>(total) * total);
    std::vector<Scalar> B(static_cast<size_t>(total) * total);
    for (int r = 0; r < total; ++r) {
        for (int c = 0; c < total; ++c) {
            A[static_cast<size_t>(r) * total + c] =
                (in_hole(r) || in_hole(c)) ? Scalar(0.0)
                                           : hermitian_entry(r, c, salt_a, decay_a);
            B[static_cast<size_t>(r) * total + c] = hermitian_entry(r, c, salt_b, decay_b);
        }
    }
    std::vector<Scalar> AB(static_cast<size_t>(total) * total, Scalar(0.0));
    for (int i = 0; i < total; ++i)
        for (int k = 0; k < total; ++k) {
            const Scalar a = A[static_cast<size_t>(i) * total + k];
            for (int j = 0; j < total; ++j)
                AB[static_cast<size_t>(i) * total + j] +=
                    a * B[static_cast<size_t>(k) * total + j];
        }
    std::vector<Scalar> C(static_cast<size_t>(total) * total, Scalar(0.0));
    for (int i = 0; i < total; ++i)
        for (int k = 0; k < total; ++k) {
            const Scalar t = AB[static_cast<size_t>(i) * total + k];
            for (int j = 0; j < total; ++j)
                // A^H[k][j] = conj(A[j][k]).
                C[static_cast<size_t>(i) * total + j] +=
                    t * std::conj(A[static_cast<size_t>(j) * total + k]);
        }
    return C;
}

double max_abs(const std::vector<Scalar>& v) {
    double n = 0.0;
    for (const Scalar& s : v) n = std::max(n, std::abs(s));
    return n;
}

double max_abs_diff(const std::vector<Scalar>& a, const std::vector<Scalar>& b) {
    double n = 0.0;
    for (size_t i = 0; i < a.size(); ++i) n = std::max(n, std::abs(a[i] - b[i]));
    return n;
}

// ||M - M^H||_maxabs on a dense global matrix.
double hermiticity_defect(const std::vector<Scalar>& m, int total) {
    double n = 0.0;
    for (int r = 0; r < total; ++r)
        for (int c = 0; c < total; ++c)
            n = std::max(n, std::abs(m[static_cast<size_t>(r) * total + c] -
                                     std::conj(m[static_cast<size_t>(c) * total + r])));
    return n;
}

// Global count of stored blocks.
long long global_blocks(const Matrix& m) {
    long long local = static_cast<long long>(m.graph->adj_ind.size());
    long long total = local;
    MPI_Allreduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, m.graph->comm);
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int failures = 0;
    auto expect = [&](bool ok, const char* what, const char* kind) {
        if (rank == 0) {
            std::printf("%-8s %-46s %s\n", kind, what, ok ? "PASS" : "FAIL");
        }
        if (!ok) ++failures;
    };

    // One case per matrix kind: uniform blocks (BSR), variable blocks (VBCSR),
    // scalar blocks (CSR). The kernel dispatches on kind, so a bug confined to
    // one backend is invisible from the other two.
    struct Case { const char* name; std::vector<int> bs; };
    // The last two exist for the vendor grouped-batch dispatch, which only
    // engages at bs >= 16 (common.hpp's vendor_batch_profitable) and only on
    // uniform square products: "big" takes that path, "bigvar" is over the
    // size threshold but ragged, so it must fall back to the per-pair kernel
    // and agree with it. Without them the batch path ships untested, since
    // every LCAO block in rsatb is 13.
    const std::vector<Case> cases = {
        {"bsr",    std::vector<int>(kBlocks, 3)},
        {"vbcsr",  {2, 5, 3, 4, 2, 3}},
        {"csr",    std::vector<int>(kBlocks, 1)},
        {"big",    std::vector<int>(kBlocks, 20)},
        {"bigvar", {20, 24, 20, 18, 24, 20}},
    };

    const unsigned kSaltA = 11u, kSaltB = 29u;
    // Enough decay that the product has a real spread of block norms. Flat
    // random operands give a result whose blocks are all the same size, and
    // then no threshold short of "drop everything" distinguishes the two
    // routes -- the accuracy comparison below would have nothing to measure.
    const double kDecayA = 0.55, kDecayB = 0.40;

    for (const auto& test_case : cases) {
        const std::vector<int>& bs = test_case.bs;
        const int total = scalar_offsets(bs).back();

        vbcsr::DistGraph graph(MPI_COMM_WORLD);
        build_graph(graph, bs, MPI_COMM_WORLD);

        Matrix A = build(&graph, bs, kSaltA, kDecayA);
        Matrix B = build(&graph, bs, kSaltB, kDecayB);

        const std::vector<Scalar> reference =
            dense_reference(bs, kSaltA, kDecayA, kSaltB, kDecayB);
        const double scale = max_abs(reference);

        // (1) Exact at threshold 0, against the dense reference.
        // (2) Exactly Hermitian after completion.
        {
            Matrix upper = A.rarh_upper(B, 0.0);
            Matrix C = upper.complete_hermitian();
            const std::vector<Scalar> got = gather_dense(C, bs);
            expect(max_abs_diff(got, reference) < 1e-13 * scale,
                   "A B A^H matches dense reference", test_case.name);
            // Exactly zero, not nearly: the lower triangle is a conjugated copy
            // of the upper, not an independently accumulated sum, so any
            // nonzero defect is a mirroring bug rather than round-off.
            expect(hermiticity_defect(got, total) == 0.0,
                   "completed result exactly Hermitian", test_case.name);
        }

        // (3) Thresholded: fused must be no less accurate than the two-call
        // route AT MATCHED KEPT-BLOCK COUNT. The two-call route is what
        // production ran before the fusion -- A B by spmm, then (A B) A by
        // spmm_hermitian, using A^H == A.
        {
            // The threshold is SEARCHED FOR, not hardcoded. A hardcoded one is
            // what makes this kind of test quietly vacuous: below the smallest
            // surviving block norm both routes return the exact answer and the
            // comparison measures round-off rather than the method. Sweep
            // upward until the fused result has actually lost a quarter of its
            // blocks, and fail loudly if no threshold in range does that.
            const long long full_blocks = static_cast<long long>(kBlocks) * kBlocks;
            double fused_threshold = -1.0;
            for (int step = 1; step <= 200; ++step) {
                const double t = 0.01 * scale * static_cast<double>(step);
                if (global_blocks(A.rarh_upper(B, t).complete_hermitian()) * 4 <=
                    full_blocks * 3) {
                    fused_threshold = t;
                    break;
                }
            }
            expect(fused_threshold > 0.0, "a threshold that truncates exists",
                   test_case.name);
            if (fused_threshold <= 0.0) continue;

            Matrix fused = A.rarh_upper(B, fused_threshold).complete_hermitian();
            const long long fused_blocks = global_blocks(fused);
            const double fused_error = max_abs_diff(gather_dense(fused, bs), reference);

            // Find the two-call threshold that keeps as close to the same
            // number of blocks as possible without keeping fewer -- so the
            // comparison never flatters the fused kernel by handing the
            // baseline less information than it has itself.
            double best_error = -1.0;
            long long best_blocks = -1;
            for (int step = 0; step <= 80; ++step) {
                const double s = 4.0 * fused_threshold * static_cast<double>(step) / 80.0;
                Matrix AB = A.spmm(B, s);
                Matrix two = AB.spmm_hermitian(A, s);
                const long long blocks = global_blocks(two);
                if (blocks < fused_blocks) continue;
                if (best_blocks < 0 || blocks < best_blocks) {
                    best_blocks = blocks;
                    best_error = max_abs_diff(gather_dense(two, bs), reference);
                }
            }

            if (best_blocks < 0) {
                expect(false, "two-call baseline found at matched nnz", test_case.name);
            } else {
                if (rank == 0) {
                    std::printf("%-8s   thr %.3e | fused %lld blocks err %.3e"
                                " | two-call %lld blocks err %.3e\n",
                                test_case.name, fused_threshold, fused_blocks,
                                fused_error, best_blocks, best_error);
                }
                expect(fused_error <= best_error,
                       "fused no less accurate at matched nnz", test_case.name);
                // The bounded-neglect contract, which is the ONE thing the
                // norm-guided pair pruning gives up against an exact
                // intermediate. Budget: <= threshold skipped across the two
                // stages, plus <= threshold for a block dropped at the end.
                // Assert it rather than trusting the derivation -- a wrong
                // per-row eps would still produce a plausible-looking answer.
                expect(fused_error <= 2.0 * fused_threshold,
                       "error within the stated neglect bound", test_case.name);
                expect(hermiticity_defect(gather_dense(fused, bs), total) == 0.0,
                       "thresholded result exactly Hermitian", test_case.name);
            }
        }

        // (4) A Hermitian A with a globally EMPTY block row -- a state
        // filter_blocks can produce. Its contribution to the contraction is
        // zero, and the kernel must say so rather than throw: an empty row's
        // fetched pattern arrives empty, which is an ANSWER, not an absence.
        // Regression for a distributed-only false positive ("left operand is
        // not Hermitian" on a valid operand); at one rank every row is local
        // and the case cannot fire, so only multi-rank runs exercise it.
        {
            const int kEmptyRow = 2;
            vbcsr::DistGraph holed(MPI_COMM_WORLD);
            build_graph(holed, bs, MPI_COMM_WORLD, kEmptyRow);
            Matrix Ah = build(&holed, bs, kSaltA, kDecayA);

            bool threw = false;
            double err = -1.0;
            try {
                Matrix C = Ah.rarh_upper(B, 0.0).complete_hermitian();
                err = max_abs_diff(gather_dense(C, bs),
                                   dense_reference(bs, kSaltA, kDecayA, kSaltB, kDecayB,
                                                   kEmptyRow));
            } catch (const std::exception&) {
                // The kernel Allreduces its verdict before throwing, so every
                // rank lands here together and no one is left in a collective.
                threw = true;
            }
            expect(!threw, "empty row of A does not throw", test_case.name);
            expect(!threw && err >= 0.0 && err < 1e-13 * scale,
                   "empty row of A matches dense reference", test_case.name);
        }
    }

    // Coherent accumulation: the stage-one pruning bound, on data where the
    // contraction cannot cancel.
    //
    // 6x6 scalar blocks, A all 1, B all 0.08, threshold 1. Every stage-one pair
    // norm is 0.08. The bound this kernel used to carry was
    // threshold / (2 * n_A * a_max_norm) = 1/12 = 0.0833, so every pair was
    // skipped and the kernel returned zero -- while the exact answer is
    // 36 * 0.08 = 2.88 per entry. An error of 2.88 against a claimed bound of 1.
    // The gate now carries the second fan-out (|B row k|) as well, which is what
    // this asserts.
    {
        const std::vector<int> bs(kBlocks, 1);
        vbcsr::DistGraph graph(MPI_COMM_WORLD);
        build_graph(graph, bs, MPI_COMM_WORLD);
        Matrix A = build_constant(&graph, bs, 1.0);
        Matrix B = build_constant(&graph, bs, 0.08);

        const double thr = 1.0;
        Matrix C = A.rarh_upper(B, thr).complete_hermitian();
        const std::vector<Scalar> got = gather_dense(C, bs);

        const double exact = static_cast<double>(kBlocks) * static_cast<double>(kBlocks) * 0.08;
        double worst = 0.0;
        for (const Scalar& v : got) worst = std::max(worst, std::abs(v - Scalar(exact, 0.0)));
        if (rank == 0) {
            std::printf("%-8s   exact %.4f | worst deviation %.4f | bound %.4f\n",
                        "coherent", exact, worst, thr);
        }
        expect(worst <= thr, "coherent accumulation within the neglect bound", "coherent");
    }

    // Halo tiling: the distributed path fetches remote payloads a tile of local
    // rows at a time so a rank cannot accumulate a copy of both operands. The
    // tiling changes only WHICH payloads are resident together -- the per-row
    // arithmetic is untouched -- so a tiled run must agree with an untiled one
    // exactly, not approximately.
    //
    // Forced through the budget, because no test-sized problem has a halo over
    // the 2 GiB default and the path would otherwise never run here.
    if (size > 1) {
        const std::vector<int> bs(kBlocks, 3);
        vbcsr::DistGraph graph(MPI_COMM_WORLD);
        build_graph(graph, bs, MPI_COMM_WORLD);
        Matrix A = build(&graph, bs, kSaltA, kDecayA);
        Matrix B = build(&graph, bs, kSaltB, kDecayB);

        unsetenv("VBCSR_RARH_HALO_BUDGET_BYTES");
        const std::vector<Scalar> untiled = gather_dense(A.rarh_upper(B, 0.0).complete_hermitian(), bs);
        // 512 bytes is below this cell's halo (a 3x3 complex block alone is 144),
        // so the row range is split as far as it goes -- one row per tile. A
        // megabyte-scale budget would leave the whole thing in one tile and the
        // check would pass without ever entering the path it exists for.
        setenv("VBCSR_RARH_HALO_BUDGET_BYTES", "512", 1);
        const std::vector<Scalar> tiled = gather_dense(A.rarh_upper(B, 0.0).complete_hermitian(), bs);
        unsetenv("VBCSR_RARH_HALO_BUDGET_BYTES");

        expect(max_abs_diff(tiled, untiled) == 0.0,
               "halo tiling changes nothing about the answer", "tiling");
    }

    // Every rank counted its own failures against the same deterministic data,
    // so they should agree -- but reduce anyway, because a kernel bug that only
    // shows on the rank owning a particular row is exactly the bug this file
    // exists to catch, and rank 0 alone would not see it.
    int global_failures = failures;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("\ntest_rarh: %d rank(s), %s\n", size,
                    global_failures == 0 ? "all checks PASSED" : "FAILURES");
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
