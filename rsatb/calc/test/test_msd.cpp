#include "../msd.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <vector>

using namespace rsatb::calc;

namespace {

using Scalar = std::complex<double>;
using Matrix = BlockSpMat<Scalar>;
using Vector = DistVector<Scalar>;
using MultiVector = DistMultiVector<Scalar>;

DistGraph make_graph(MPI_Comm comm, int n, bool dense) {
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        if (dense) {
            adj[i].reserve(n);
            for (int j = 0; j < n; ++j) {
                adj[i].push_back(j);
            }
        } else {
            adj[i] = {i};
        }
    }

    DistGraph graph(comm);
    graph.construct_serial(n, block_sizes, adj);
    return graph;
}

void fill_matrix(Matrix& mat, const std::vector<std::vector<Scalar>>& values) {
    for (int local_row = 0; local_row < static_cast<int>(mat.graph->owned_global_indices.size()); ++local_row) {
        const int global_row = mat.graph->owned_global_indices[local_row];
        for (int j = 0; j < static_cast<int>(values[global_row].size()); ++j) {
            if (std::abs(values[global_row][j]) > 1e-14) {
                Scalar val = values[global_row][j];
                mat.add_block(global_row, j, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            }
        }
    }
    mat.assemble();
}

void fill_vector(Vector& vec, const std::vector<Scalar>& values) {
    vec.set_constant(Scalar(0.0, 0.0));
    for (int local_row = 0; local_row < vec.local_size; ++local_row) {
        const int global_row = vec.graph->owned_global_indices[local_row];
        vec[local_row] = values[static_cast<size_t>(global_row)];
    }
}

} // namespace

TEST(MSDTest, CommutingDiagonalOperatorsGiveZeroSpectrumWithOverlap) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, false);
    Matrix H(&graph);
    Matrix S(&graph);
    Matrix X(&graph);

    fill_matrix(H, {
        {Scalar(-0.4, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(0.35, 0.0)}
    });
    fill_matrix(S, {
        {Scalar(1.2, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(0.9, 0.0)}
    });
    fill_matrix(X, {
        {Scalar(-0.7, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(1.1, 0.0)}
    });

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.3, -0.4)});

    Chebyshev<Scalar> cheby(H, S, -1.0, 1.0);
    std::vector<double> msd;
    std::vector<double> energy;
    corr_msd(cheby, X, sample, 3, 0.5, 64, msd, energy, false);

    ASSERT_EQ(energy.size(), 64u);
    ASSERT_EQ(msd.size(), static_cast<size_t>(4 * 64));
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }
}

TEST(MSDTest, RepeatedSamplesInMultiVectorMatchSingleVectorRun) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph);
    Matrix X(&graph);

    fill_matrix(H, {
        {Scalar(0.0, 0.0), Scalar(0.35, 0.0)},
        {Scalar(0.35, 0.0), Scalar(0.1, 0.0)}
    });
    fill_matrix(X, {
        {Scalar(-0.5, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(0.5, 0.0)}
    });

    Vector sample_a(&graph);
    fill_vector(sample_a, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.0, 1.0 / std::sqrt(2.0))});

    std::vector<double> msd_single;
    std::vector<double> energy_single;
    std::vector<double> energy_mv;

    {
        Chebyshev<Scalar> cheby(H, -1.0, 1.0);
        corr_msd(cheby, X, sample_a, 2, 0.4, 48, msd_single, energy_single, false);
    }

    MultiVector samples(&graph, 2);
    samples.set_col(0, sample_a);
    samples.set_col(1, sample_a);

    std::vector<double> msd_mv;
    {
        Chebyshev<Scalar> cheby(H, -1.0, 1.0);
        corr_msd(cheby, X, samples, 2, 0.4, 48, msd_mv, energy_mv, false);
    }

    ASSERT_EQ(msd_mv.size(), msd_single.size());
    ASSERT_EQ(energy_single.size(), energy_mv.size());
    for (size_t i = 0; i < energy_single.size(); ++i) {
        EXPECT_NEAR(energy_single[i], energy_mv[i], 1e-12);
    }
    for (size_t i = 0; i < msd_single.size(); ++i) {
        EXPECT_NEAR(msd_mv[i], msd_single[i], 1e-10);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
