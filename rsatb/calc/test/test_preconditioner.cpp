// Regression tests for the block-layout contract of the preconditioners.
//
// vbcsr's canonical within-block storage is row-major (kCanonicalBlockLayout);
// element (i,j) of an r x c block lives at block[i*c + j]. Code that reads a
// block via block_data() and writes it back must therefore declare RowMajor --
// declaring ColMajor silently stores the transpose.
//
// test_solver.cpp only exercises symmetric blocks (2*I), where a transpose is
// invisible, which is exactly how the ColMajor tag survived unnoticed. These
// tests use deliberately NON-symmetric blocks.

#include "../preconditioner.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace vbcsr;
using namespace rsatb::calc;

namespace {

DistGraph make_single_block_graph(int block_size) {
    std::vector<int> block_sizes(1, block_size);
    std::vector<std::vector<int>> adj(1);
    adj[0] = {0};

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(1, block_sizes, adj);
    return graph;
}

} // namespace

// A = [[1, 2], [3, 4]] (row-major), det = -2, so A^-1 = [[-2, 1], [1.5, -0.5]].
// A^-1 is non-symmetric, so storing its transpose is detectable.
TEST(BlockJacobianPreconditionerTest, StoresInverseNotItsTransposeForNonSymmetricBlock) {
    DistGraph graph = make_single_block_graph(2);

    BlockSpMat<double> A(&graph);
    std::vector<double> blk = {1.0, 2.0, 3.0, 4.0};
    A.add_block(0, 0, blk.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    A.assemble();

    BlockJacobianPreconditioner<double> P(A);

    const double* got = P.P.block_data(0);
    ASSERT_NE(got, nullptr);

    const double expected[4] = {-2.0, 1.0, 1.5, -0.5}; // A^-1, canonical row-major
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-12) << "flat index " << i;
    }
}

// Guard the round trip at the level that actually matters: P must behave as
// A^-1 under application, i.e. A * P = I for the diagonal block.
TEST(BlockJacobianPreconditionerTest, InverseBlockReconstructsIdentity) {
    DistGraph graph = make_single_block_graph(2);

    BlockSpMat<double> A(&graph);
    std::vector<double> blk = {4.0, 7.0, 2.0, 6.0}; // det = 10
    A.add_block(0, 0, blk.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    A.assemble();

    BlockJacobianPreconditioner<double> P(A);
    const double* inv = P.P.block_data(0);
    ASSERT_NE(inv, nullptr);

    // (A * A^-1)[i][j] with both operands in row-major storage.
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 2; ++k) {
                acc += blk[i * 2 + k] * inv[k * 2 + j];
            }
            EXPECT_NEAR(acc, (i == j) ? 1.0 : 0.0, 1e-12) << "(" << i << "," << j << ")";
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
