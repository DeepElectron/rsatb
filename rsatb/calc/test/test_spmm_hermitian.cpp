// vbcsr's spmm_hermitian against the full product, exercised the way the
// Newton-Schulz loop uses it: the congruence Z^H (S Z) and the square of a
// Hermitian matrix. Synthetic serial matrices -- no integral engine, so this
// runs in milliseconds and pins the contract:
//   (1) equal to the full spmm at threshold scale,
//   (2) exactly Hermitian, which the full product is not guaranteed to be
//       once the threshold drops blocks asymmetrically.

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_graph.hpp"

#include <mpi.h>

#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using Scalar = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<Scalar>;

namespace {

constexpr int kBlocks = 6;

// A serial full-pattern block matrix over the given block sizes; Hermitian
// when asked. Off-diagonal blocks are rectangular when the sizes differ,
// which is what makes this exercise the VBCSR kind.
Matrix build(vbcsr::DistGraph* graph, const std::vector<int>& bs,
             std::mt19937& gen, bool hermitian) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<std::vector<Scalar>> upper(kBlocks * kBlocks);
    Matrix m(graph);
    for (int i = 0; i < kBlocks; ++i) {
        for (int j = 0; j < kBlocks; ++j) {
            const int r_dim = bs[i];
            const int c_dim = bs[j];
            std::vector<Scalar> block(static_cast<size_t>(r_dim) * c_dim);
            if (hermitian && j < i) {
                const auto& src = upper[j * kBlocks + i];  // c_dim x r_dim
                for (int r = 0; r < r_dim; ++r)
                    for (int c = 0; c < c_dim; ++c)
                        block[r * c_dim + c] = std::conj(src[c * r_dim + r]);
            } else {
                for (auto& v : block) v = Scalar(dist(gen), dist(gen));
                if (hermitian && i == j) {
                    for (int r = 0; r < r_dim; ++r)
                        for (int c = 0; c < r; ++c)
                            block[c * c_dim + r] = std::conj(block[r * c_dim + c]);
                    for (int r = 0; r < r_dim; ++r)
                        block[r * c_dim + r] = Scalar(block[r * c_dim + r].real(), 0.0);
                }
                upper[i * kBlocks + j] = block;
            }
            m.add_block(i, j, block.data(), r_dim, c_dim);
        }
    }
    m.assemble();
    return m;
}

double max_diff(Matrix& a, Matrix& b) {
    Matrix d = a.transpose();  // conj(a^T)
    d = d.transpose();         // back to a: a deep copy via double transpose
    d.axpby(Scalar(-1.0), b, Scalar(1.0));
    double norm = 0.0;
    for (double n : d.get_block_norms()) norm = std::max(norm, n);
    return norm;
}

double hermiticity_defect(const Matrix& m) {
    Matrix mh = m.transpose();
    mh.axpby(Scalar(1.0), m, Scalar(-1.0));
    double norm = 0.0;
    for (double n : mh.get_block_norms()) norm = std::max(norm, n);
    return norm;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int failures = 0;

    auto expect = [&](bool ok, const char* what, const char* kind) {
        std::printf("%-8s %-44s %s\n", kind, what, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    };

    // One case per matrix kind: uniform blocks (BSR), variable blocks
    // (VBCSR), scalar blocks (CSR). Same checks each time.
    struct Case { const char* name; std::vector<int> bs; };
    const std::vector<Case> cases = {
        {"bsr",   std::vector<int>(kBlocks, 3)},
        {"vbcsr", {2, 5, 3, 4, 2, 3}},
        {"csr",   std::vector<int>(kBlocks, 1)},
    };

    for (const auto& test_case : cases) {
        std::vector<std::vector<int>> adj(kBlocks);
        for (int i = 0; i < kBlocks; ++i)
            for (int j = 0; j < kBlocks; ++j) adj[i].push_back(j);

        vbcsr::DistGraph graph(MPI_COMM_WORLD);
        graph.construct_serial(kBlocks, test_case.bs, adj);

        std::mt19937 gen(7);
        Matrix S = build(&graph, test_case.bs, gen, /*hermitian=*/true);
        Matrix Z = build(&graph, test_case.bs, gen, /*hermitian=*/false);

        {
            // Congruence: X = Z^H (S Z).
            Matrix SZ = S.spmm(Z, 0.0);
            Matrix full = Z.spmm(SZ, 0.0, /*transA=*/true, false);
            Matrix half = Z.spmm_hermitian(SZ, 0.0, /*transA=*/true);
            const double scale = [&] {
                double n = 0.0;
                for (double v : full.get_block_norms()) n = std::max(n, v);
                return n;
            }();
            expect(max_diff(full, half) < 1e-12 * scale,
                   "congruence matches full product", test_case.name);
            expect(hermiticity_defect(half) == 0.0,
                   "congruence exactly Hermitian", test_case.name);
        }

        {
            // Square of a Hermitian matrix: X * X.
            Matrix full = S.spmm(S, 0.0);
            Matrix half = S.spmm_hermitian(S, 0.0);
            const double scale = [&] {
                double n = 0.0;
                for (double v : full.get_block_norms()) n = std::max(n, v);
                return n;
            }();
            expect(max_diff(full, half) < 1e-12 * scale,
                   "Hermitian square matches full product", test_case.name);
            expect(hermiticity_defect(half) == 0.0,
                   "Hermitian square exactly Hermitian", test_case.name);
        }

        {
            // Thresholded, which is how production calls it. The upper
            // triangle's drop decisions are identical to the full product's
            // upper triangle; the mirrored lower may differ from the full
            // product's lower (that asymmetry is what the mirror fixes), so
            // the comparison is bounded by the drop scale rather than machine
            // epsilon -- and the result must STILL be exactly Hermitian.
            const double threshold = 0.05;
            Matrix SZ = S.spmm(Z, threshold);
            Matrix full = Z.spmm(SZ, threshold, /*transA=*/true, false);
            Matrix half = Z.spmm_hermitian(SZ, threshold, /*transA=*/true);
            const double scale = [&] {
                double n = 0.0;
                for (double v : full.get_block_norms()) n = std::max(n, v);
                return n;
            }();
            expect(max_diff(full, half) < 4.0 * threshold * scale,
                   "thresholded within drop scale of full", test_case.name);
            expect(hermiticity_defect(half) == 0.0,
                   "thresholded exactly Hermitian", test_case.name);
        }
    }

    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
