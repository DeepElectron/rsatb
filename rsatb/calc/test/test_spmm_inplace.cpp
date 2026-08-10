// spmm_inplace (A <- A B) against the ordinary product it replaces.
//
// The in-place form releases A's storage as the numeric phase walks past it,
// which is only sound because row i of A B reads row i of A and rows of B and
// nothing else. If that ever stops holding, the symptom is not a crash but a
// WRONG ANSWER in the rows processed after the release -- so the check that
// matters is equality with the out-of-place product, not merely "it ran".
//
// Multi-rank matters here beyond the usual reason: A is the left operand and
// its blocks are never fetched by another rank, which is the second half of the
// licence to release. A distributed run is what tests that half.

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_graph.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using Scalar = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<Scalar>;

namespace {

constexpr int kBlocks = 12;

double jitter(unsigned key, unsigned salt) {
    unsigned k = key * 2654435761u + salt * 40503u;
    k ^= k >> 15; k *= 2246822519u; k ^= k >> 13; k *= 3266489917u; k ^= k >> 16;
    return static_cast<double>(k % 100000u) / 50000.0 - 1.0;
}

// Deterministic in the GLOBAL indices, so the matrix is the same however the
// rows are split across ranks.
Matrix build(vbcsr::DistGraph* graph, const std::vector<int>& bs, unsigned salt) {
    std::vector<int> off(bs.size() + 1, 0);
    for (size_t i = 0; i < bs.size(); ++i) off[i + 1] = off[i] + bs[i];
    Matrix m(graph);
    const int n_owned = static_cast<int>(graph->owned_global_indices.size());
    for (int i = 0; i < n_owned; ++i) {
        const int gi = graph->owned_global_indices[i];
        for (int k = graph->adj_ptr[i]; k < graph->adj_ptr[i + 1]; ++k) {
            const int gj = graph->get_global_index(graph->adj_ind[k]);
            const int r = bs[gi], c = bs[gj];
            std::vector<Scalar> block(static_cast<size_t>(r) * c);
            for (int x = 0; x < r; ++x)
                for (int y = 0; y < c; ++y)
                    block[static_cast<size_t>(x) * c + y] =
                        Scalar(jitter(static_cast<unsigned>((off[gi] + x) * 131 + off[gj] + y), salt),
                               jitter(static_cast<unsigned>((off[gi] + x) * 131 + off[gj] + y), salt + 7u));
            m.add_block(gi, gj, block.data(), r, c);
        }
    }
    m.assemble();
    return m;
}

void build_graph(vbcsr::DistGraph& graph, const std::vector<int>& bs, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int base = kBlocks / size, extra = kBlocks % size;
    const int begin = rank * base + std::min(rank, extra);
    const int end = begin + base + (rank < extra ? 1 : 0);
    std::vector<int> owned, sizes;
    std::vector<std::vector<int>> adj;
    for (int g = begin; g < end; ++g) {
        owned.push_back(g);
        sizes.push_back(bs[g]);
        // Banded, not full: a full pattern makes every row identical and hides
        // a release boundary that is off by a row.
        std::vector<int> row;
        for (int j = 0; j < kBlocks; ++j)
            if (std::abs(j - g) <= 3) row.push_back(j);
        adj.push_back(std::move(row));
    }
    graph.construct_distributed(owned, sizes, adj);
}

// Largest |difference| over the blocks the two matrices share, plus a check
// that they agree on which blocks exist at all.
double compare(Matrix& expected, Matrix& got, int& pattern_mismatch) {
    pattern_mismatch = 0;
    vbcsr::DistGraph* ge = expected.graph;
    vbcsr::DistGraph* gg = got.graph;
    const int rows_e = static_cast<int>(ge->adj_ptr.size()) - 1;
    const int rows_g = static_cast<int>(gg->adj_ptr.size()) - 1;
    if (rows_e != rows_g) { pattern_mismatch = 1; return 0.0; }
    double worst = 0.0;
    for (int i = 0; i < rows_e; ++i) {
        const int ne = ge->adj_ptr[i + 1] - ge->adj_ptr[i];
        const int ng = gg->adj_ptr[i + 1] - gg->adj_ptr[i];
        if (ne != ng) { pattern_mismatch = 1; continue; }
        for (int k = ge->adj_ptr[i]; k < ge->adj_ptr[i + 1]; ++k) {
            const int gcol = ge->get_global_index(ge->adj_ind[k]);
            int match = -1;
            for (int q = gg->adj_ptr[i]; q < gg->adj_ptr[i + 1]; ++q) {
                if (gg->get_global_index(gg->adj_ind[q]) == gcol) { match = gg->adj_ind[q]; break; }
            }
            if (match < 0) { pattern_mismatch = 1; continue; }
            const std::vector<Scalar> be = expected.get_block(i, ge->adj_ind[k]);
            const std::vector<Scalar> bg = got.get_block(i, match);
            if (be.size() != bg.size()) { pattern_mismatch = 1; continue; }
            for (size_t e = 0; e < be.size(); ++e)
                worst = std::max(worst, std::abs(be[e] - bg[e]));
        }
    }
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int failures = 0;

    auto expect = [&](bool ok, const char* what, const char* kind) {
        if (rank == 0) std::printf("%-8s %-46s %s\n", kind, what, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    };

    struct Case { const char* name; std::vector<int> bs; };
    const std::vector<Case> cases = {
        {"bsr",   std::vector<int>(kBlocks, 13)},   // the path that releases
        {"vbcsr", {2, 5, 3, 4, 2, 3, 5, 2, 4, 3, 2, 5}},
        {"csr",   std::vector<int>(kBlocks, 1)},
    };

    // 0.0 and a threshold that actually drops blocks: the release boundary is
    // computed from A's row pointer, which the drop does not change, but the
    // result's pattern does change and the two must still agree.
    const std::vector<double> thresholds = {0.0, 0.35};

    for (const auto& c : cases) {
        for (double tol : thresholds) {
            vbcsr::DistGraph graph(MPI_COMM_WORLD);
            build_graph(graph, c.bs, MPI_COMM_WORLD);
            Matrix A = build(&graph, c.bs, 11u);
            Matrix B = build(&graph, c.bs, 29u);

            Matrix expected = A.spmm(B, tol);
            Matrix in_place = A.duplicate();
            in_place.spmm_inplace(B, tol);

            int mismatch = 0;
            const double worst = compare(expected, in_place, mismatch);
            int any_mismatch = mismatch;
            MPI_Allreduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            double worst_all = worst;
            MPI_Allreduce(&worst, &worst_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            double scale = 0.0;
            for (double n : expected.get_block_norms()) scale = std::max(scale, n);
            double scale_all = scale;
            MPI_Allreduce(&scale, &scale_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            char label[80];
            std::snprintf(label, sizeof(label), "tol %.2f: same pattern as spmm", tol);
            expect(any_mismatch == 0, label, c.name);
            std::snprintf(label, sizeof(label), "tol %.2f: matches spmm (%.1e rel)", tol,
                          scale_all > 0.0 ? worst_all / scale_all : worst_all);
            // Round-off, not bit-identity. Two pre-existing reasons the two
            // sides can differ in the last digits without either being wrong:
            // at one rank BSR's ordinary spmm takes the vendor route while the
            // in-place form always takes the generic one, and VBCSR groups its
            // products into batches by whichever thread claimed each row, so
            // its summation order is not reproducible run to run. Neither can
            // hide the failure this test exists for -- a prematurely released
            // page gives a wrong answer of order one, not of order 1e-14.
            // (Multi-rank BSR, where both sides run the same generic path, does
            // come out bit-identical.)
            expect(worst_all <= 1e-12 * std::max(scale_all, 1e-30), label, c.name);
        }
    }

    // Aliasing must be refused rather than silently corrupting B.
    {
        vbcsr::DistGraph graph(MPI_COMM_WORLD);
        build_graph(graph, cases[0].bs, MPI_COMM_WORLD);
        Matrix A = build(&graph, cases[0].bs, 3u);
        bool threw = false;
        try {
            A.spmm_inplace(A, 0.0);
        } catch (const std::exception&) {
            threw = true;
        }
        expect(threw, "self-aliased operand is rejected", "guard");
    }

    int global_failures = failures;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("\ntest_spmm_inplace: %d rank(s), %s\n", size,
                    global_failures == 0 ? "all checks PASSED" : "FAILURES");
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
