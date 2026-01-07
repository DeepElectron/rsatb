#include "../block_csr.hpp"
#include "../dist_graph.hpp"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <fstream>
#include <complex>

using namespace rsatb::backend;

TEST(BlockCSRTest, ExportMatrixMarketReal) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 1) {
        std::cout << "Skipping serial export test in parallel environment" << std::endl;
        return;
    }

    // 1. Create Graph
    DistGraph graph(MPI_COMM_WORLD);
    int n_global = 2;
    std::vector<int> block_sizes = {2, 3}; // Block 0: 2x2, Block 1: 3x3
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}}; // Full connectivity
    graph.construct_serial(n_global, block_sizes, adj);

    // 2. Create Matrix
    BlockSpMat<double> mat(&graph);

    // 3. Fill Matrix
    // Block (0,0) [2x2]
    std::vector<double> b00 = {1.0, 2.0, 3.0, 4.0}; // ColMajor: 1,2 (col 0), 3,4 (col 1) -> [1 3; 2 4]
    mat.add_block(0, 0, b00.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);

    // Block (0,1) [2x3]
    std::vector<double> b01 = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0}; // ColMajor: 5,6 (col 0), 7,8 (col 1), 9,10 (col 2)
    mat.add_block(0, 1, b01.data(), 2, 3, AssemblyMode::INSERT, MatrixLayout::ColMajor);

    // Block (1,0) [3x2]
    std::vector<double> b10(6, 0.0); // Zeros
    mat.add_block(1, 0, b10.data(), 3, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);

    // Block (1,1) [3x3]
    std::vector<double> b11(9, 1.0); // Ones
    mat.add_block(1, 1, b11.data(), 3, 3, AssemblyMode::INSERT, MatrixLayout::ColMajor);

    // 4. Export
    std::string filename = "test_export_real.mtx";
    mat.save_matrix_market(filename);

    // 5. Verify
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "%%MatrixMarket matrix coordinate real general");

    int rows, cols, nnz;
    file >> rows >> cols >> nnz;
    EXPECT_EQ(rows, 5); // 2+3
    EXPECT_EQ(cols, 5); // 2+3
    EXPECT_EQ(nnz, 2*2 + 2*3 + 3*2 + 3*3); // 4 + 6 + 6 + 9 = 25

    // Check some values
    // We can read all and verify, but let's just check file existence and header for now.
    // Or read a few lines.
    
    // Clean up
    // std::remove(filename.c_str());
}

TEST(BlockCSRTest, ExportMatrixMarketComplex) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 1) return;

    DistGraph graph(MPI_COMM_WORLD);
    int n_global = 1;
    std::vector<int> block_sizes = {2};
    std::vector<std::vector<int>> adj = {{0}};
    graph.construct_serial(n_global, block_sizes, adj);

    BlockSpMat<std::complex<double>> mat(&graph);
    std::vector<std::complex<double>> b00 = {{1.0, 0.5}, {2.0, 1.5}, {3.0, 2.5}, {4.0, 3.5}};
    mat.add_block(0, 0, b00.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);

    std::string filename = "test_export_complex.mtx";
    mat.save_matrix_market(filename);

    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());
    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "%%MatrixMarket matrix coordinate complex general");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
