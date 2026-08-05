// SpGEMM cost breakdown: full product vs Hermitian-halved, batched vendor
// GEMM vs the per-pair kernel, at the two block densities the Newton-Schulz
// loop actually visits (the 0.29-dense seed regime and the fully dense one).
//
// Synthetic BSR matrices shaped like the sc888 problem (1024 block rows of
// 13), so the numbers translate directly. Phase breakdown comes from the
// executor itself (VBCSR_PROFILE_BSR_SPGEMM on stderr); this driver reports
// wall time and effective GFLOP/s per configuration.
//
//   OMP_NUM_THREADS=... ./benchmark_spgemm [--blocks 1024] [--bs 13]
//                                          [--density 1.0] [--reps 3]

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_graph.hpp"

#include <mpi.h>
#include <omp.h>

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using Scalar = std::complex<double>;
using Matrix = vbcsr::BlockSpMat<Scalar>;

namespace {

Matrix build(vbcsr::DistGraph* graph, int n_blocks, const std::vector<int>& bs,
             double density, std::mt19937& gen, bool hermitian) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    // Symmetric pattern: keep (i, j) and (j, i) together, diagonal always.
    std::vector<std::vector<int>> cols(n_blocks);
    for (int i = 0; i < n_blocks; ++i) cols[i].push_back(i);
    for (int i = 0; i < n_blocks; ++i) {
        for (int j = i + 1; j < n_blocks; ++j) {
            if (coin(gen) < density) {
                cols[i].push_back(j);
                cols[j].push_back(i);
            }
        }
    }

    Matrix m(graph);
    std::vector<Scalar> block;
    std::vector<Scalar> mirror;
    for (int i = 0; i < n_blocks; ++i) {
        for (int j : cols[i]) {
            if (hermitian && j < i) continue;  // added as the mirror below
            const int r_dim = bs[i];
            const int c_dim = bs[j];
            block.resize(static_cast<size_t>(r_dim) * c_dim);
            for (auto& v : block) v = Scalar(dist(gen), dist(gen));
            if (hermitian && i == j) {
                for (int r = 0; r < r_dim; ++r) {
                    block[r * c_dim + r] = Scalar(block[r * c_dim + r].real(), 0.0);
                    for (int c = r + 1; c < c_dim; ++c)
                        block[c * c_dim + r] = std::conj(block[r * c_dim + c]);
                }
            }
            m.add_block(i, j, block.data(), r_dim, c_dim);
            if (hermitian && j > i) {
                mirror.resize(block.size());
                for (int r = 0; r < r_dim; ++r)
                    for (int c = 0; c < c_dim; ++c)
                        mirror[c * r_dim + r] = std::conj(block[r * c_dim + c]);
                m.add_block(j, i, mirror.data(), c_dim, r_dim);
            }
        }
    }
    m.assemble();
    return m;
}

double flops_of(const Matrix& A, const Matrix& B, int bs, bool halved) {
    // Complex fused multiply-adds per block product: 8 real flops per entry.
    // Product count = sum over k of nnzcol_A(k) * nnzrow_B(k); with the
    // symmetric synthetic pattern nnzcol_A ~ nnzrow_A.
    double products = 0.0;
    const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
    for (int k = 0; k < n_rows; ++k) {
        const double a_col = static_cast<double>(A.row_ptr()[k + 1] - A.row_ptr()[k]);
        const double b_row = static_cast<double>(B.row_ptr()[k + 1] - B.row_ptr()[k]);
        products += a_col * b_row;
    }
    if (halved) products *= 0.5;
    return products * 8.0 * bs * bs * bs;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int n_blocks = 1024, bs = 13, reps = 3;
    double density = 1.0;
    bool mixed = false;  // variable block sizes: exercises the VBCSR executor
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mixed") == 0) mixed = true;
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) n_blocks = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bs") == 0 && i + 1 < argc) bs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--density") == 0 && i + 1 < argc) density = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
    }

    std::vector<int> block_sizes(n_blocks, bs);
    if (mixed) {
        // 9/13/17 rotation around the same mean: VBCSR kind, comparable flops.
        const int sizes[3] = {9, 13, 17};
        for (int i = 0; i < n_blocks; ++i) block_sizes[i] = sizes[i % 3];
    }
    std::vector<std::vector<int>> adj(n_blocks);
    for (int i = 0; i < n_blocks; ++i)
        for (int j = 0; j < n_blocks; ++j) adj[i].push_back(j);
    vbcsr::DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, adj);

    std::mt19937 gen(7);
    Matrix S = build(&graph, n_blocks, block_sizes, density, gen, /*hermitian=*/true);
    Matrix Z = build(&graph, n_blocks, block_sizes, density, gen, /*hermitian=*/false);
    Matrix SZ = S.spmm(Z, 1e-14);

    std::printf("spgemm benchmark: %d blocks x %s, density %.2f, %d threads\n",
                n_blocks, mixed ? "mixed(9/13/17)" : std::to_string(bs).c_str(),
                density, omp_get_max_threads());
    std::printf("%-34s %10s %10s %9s\n", "operation", "wall(s)", "GFLOP/s", "speedup");

    double reference = 0.0;
    auto bench = [&](const char* label, bool batch, bool hermitian) {
        setenv("VBCSR_SPGEMM_BATCH", batch ? "1" : "0", 1);
        // Warm-up once, then the timed repetitions.
        double best = 1e300;
        for (int r = 0; r <= reps; ++r) {
            const double t0 = omp_get_wtime();
            Matrix X = hermitian ? Z.spmm_hermitian(SZ, 1e-14, /*transA=*/true)
                                 : Z.spmm(SZ, 1e-14, /*transA=*/true, false);
            const double dt = omp_get_wtime() - t0;
            if (r > 0) best = std::min(best, dt);
        }
        const double flops = flops_of(Z, SZ, bs, hermitian);  // mixed: bs-mean approximation
        if (reference == 0.0) reference = best;
        std::printf("%-34s %10.3f %10.1f %8.2fx\n", label, best,
                    flops / best / 1e9, reference / best);
    };

    if (argc > 1 && std::string(argv[1]) == "--mirror-breakdown") {
        // The spmm_hermitian steps, timed separately: where does the
        // difference between half the flops and the measured time go?
        setenv("VBCSR_SPGEMM_BATCH", "1", 1);
        double t = omp_get_wtime();
        Matrix U = vbcsr::detail::BSRSpMMExecutor<Matrix>::run(Z, SZ, 1e-14, false)
                       .transpose();  // placeholder to warm caches
        std::printf("(warm-up %.2fs)\n", omp_get_wtime() - t);

        t = omp_get_wtime();
        Matrix ZT = Z.transpose();
        const double t_at = omp_get_wtime() - t;
        t = omp_get_wtime();
        Matrix Uh = vbcsr::detail::BSRSpMMExecutor<Matrix>::run(ZT, SZ, 1e-14, /*upper_only=*/true);
        const double t_run = omp_get_wtime() - t;
        std::printf("upper result density %.4f (full would be ~%.4f)\n",
                    Uh.get_block_density(), density);
        setenv("VBCSR_SPGEMM_BATCH", "0", 1);
        t = omp_get_wtime();
        Matrix Uh2 = vbcsr::detail::BSRSpMMExecutor<Matrix>::run(ZT, SZ, 1e-14, /*upper_only=*/true);
        std::printf("upper product (no batch) %8.2fs\n", omp_get_wtime() - t);
        t = omp_get_wtime();
        Matrix F2 = vbcsr::detail::BSRSpMMExecutor<Matrix>::run(ZT, SZ, 1e-14, /*upper_only=*/false);
        std::printf("full  product (no batch) %8.2fs\n", omp_get_wtime() - t);
        setenv("VBCSR_SPGEMM_BATCH", "1", 1);
        t = omp_get_wtime();
        Matrix L = Uh.transpose();
        const double t_tr = omp_get_wtime() - t;
        t = omp_get_wtime();
        L.zero_diagonal_blocks();
        const double t_zd = omp_get_wtime() - t;
        t = omp_get_wtime();
        Uh.axpby(Scalar(1), L, Scalar(1));
        const double t_ax = omp_get_wtime() - t;
        t = omp_get_wtime();
        Uh.hermitize_diagonal_blocks();
        const double t_hd = omp_get_wtime() - t;
        std::printf("A^H materialize   %8.2fs\n", t_at);
        std::printf("upper product     %8.2fs\n", t_run);
        std::printf("mirror transpose  %8.2fs\n", t_tr);
        std::printf("zero diagonal     %8.2fs\n", t_zd);
        std::printf("union axpby       %8.2fs\n", t_ax);
        std::printf("hermitize diag    %8.2fs\n", t_hd);
        MPI_Finalize();
        return 0;
    }

    bench("full product, per-pair kernel", false, false);
    bench("full product, batched",         true,  false);
    bench("hermitian half, per-pair",      false, true);
    bench("hermitian half, batched",       true,  true);

    MPI_Finalize();
    return 0;
}
