#include "../fermi_dirac.hpp"
#include "../kpm.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using namespace rsatb::calc;

namespace {

using Scalar = std::complex<double>;
using Matrix = BlockSpMat<Scalar>;
using Vector = DistVector<Scalar>;

DistGraph make_diagonal_graph(MPI_Comm comm, int n) {
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        adj[i] = {i};
    }

    DistGraph graph(comm);
    graph.construct_serial(n, block_sizes, adj);
    return graph;
}

void fill_diagonal_matrix(Matrix& mat, const std::vector<double>& diagonal) {
    for (int local_row = 0; local_row < static_cast<int>(mat.graph->owned_global_indices.size()); ++local_row) {
        const int global_row = mat.graph->owned_global_indices[local_row];
        Scalar val(diagonal[static_cast<size_t>(global_row)], 0.0);
        mat.add_block(global_row, global_row, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    mat.assemble();
}

void fill_constant(Vector& vec, Scalar value) {
    vec.set_constant(value);
}

} // namespace

TEST(KPMExtensionsTest, ReconstructedSpectrumUsesAscendingGrid) {
    std::vector<std::complex<double>> spectrum;
    std::vector<double> eps;
    reconstruct_spectrum_from_moments(
        std::vector<std::complex<double>>{
            {1.0, 0.0}, {0.25, 0.0}, {-0.1, 0.0}, {0.05, 0.0},
            {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        spectrum,
        &eps,
        true);

    ASSERT_EQ(spectrum.size(), 8u);
    ASSERT_EQ(eps.size(), 8u);
    EXPECT_TRUE(std::is_sorted(eps.begin(), eps.end()));
    EXPECT_LT(eps.front(), 0.0);
    EXPECT_GT(eps.back(), 0.0);
}

TEST(KPMExtensionsTest, ZeroTemperatureProjectorMatchesHeavisideAwayFromMu) {
    DistGraph graph = make_diagonal_graph(MPI_COMM_WORLD, 32);
    Matrix H(&graph);
    std::vector<double> diagonal(32, 0.0);
    for (int i = 0; i < 32; ++i) {
        diagonal[static_cast<size_t>(i)] = -0.95 + 1.9 * static_cast<double>(i) / 31.0;
    }
    fill_diagonal_matrix(H, diagonal);

    Chebyshev<Scalar> cheby(H, -1.0, 1.0);
    Vector psi_in(&graph);
    Vector psi_out(&graph);
    fill_constant(psi_in, Scalar(1.0, 0.0));

    FermiExpansionOptions opt;
    opt.n_fixed = 512;
    opt.summation = KpmKernel::Jackson;
    fermi_dirac_kpm(cheby, psi_in, psi_out, 0.0, 0.0, opt);

    const Scalar* out = psi_out.local_data();
    for (int local_row = 0; local_row < psi_out.local_size; ++local_row) {
        const int global_row = psi_out.graph->owned_global_indices[local_row];
        const double energy = diagonal[static_cast<size_t>(global_row)];
        if (std::abs(energy) < 0.2) {
            continue;
        }

        const double expected = energy <= 0.0 ? 1.0 : 0.0;
        EXPECT_NEAR(out[local_row].real(), expected, 5e-2);
        EXPECT_NEAR(out[local_row].imag(), 0.0, 1e-10);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
