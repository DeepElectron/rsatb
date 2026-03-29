#include "hdf5.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>

#include <complex>
#include <cstdio>
#include <string>
#include <vector>

using namespace rsatb::io;

namespace {

std::string make_test_file() {
    const std::string path = "/tmp/rsatb_hdf5_io_test.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("0");

    {
        const hsize_t dims[2] = {2, 2};
        H5::DataSpace space(2, dims);
        H5::DataSet dataset = group.createDataSet("1_1_0_0_0", H5::PredType::NATIVE_DOUBLE, space);
        const double values[4] = {1.0, 2.0, 3.0, 4.0};
        dataset.write(values, H5::PredType::NATIVE_DOUBLE);
    }

    {
        const hsize_t dims[3] = {2, 2, 2};
        H5::DataSpace space(3, dims);
        H5::DataSet dataset = group.createDataSet("3_2_0_1_0", H5::PredType::NATIVE_DOUBLE, space);
        const double values[8] = {
            1.0,  0.5,
            2.0, -0.5,
            3.0,  1.5,
            4.0, -1.5
        };
        dataset.write(values, H5::PredType::NATIVE_DOUBLE);
    }

    return path;
}

} // namespace

TEST(ioTest, ReadsLegacyRealBlocks) {
    const std::string path = make_test_file();
    HDF5Handler hd(path);

    int m = 0, n = 0;
    double* values = nullptr;
    bool found = false;

    ASSERT_EQ(hd.read_from_ijR_dense(1, 1, 0, 0, 0, &m, &n, &values, &found), 0);
    ASSERT_TRUE(found);
    ASSERT_NE(values, nullptr);
    EXPECT_EQ(m, 2);
    EXPECT_EQ(n, 2);
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 2.0);
    EXPECT_DOUBLE_EQ(values[2], 3.0);
    EXPECT_DOUBLE_EQ(values[3], 4.0);
    delete[] values;
}

TEST(ioTest, ReadsComplexBlocksAndTransposeFallback) {
    const std::string path = make_test_file();
    HDF5Handler hd(path);

    int m = 0, n = 0;
    std::complex<double>* values = nullptr;
    bool found = false;

    ASSERT_EQ(hd.read_from_ijR_dense_complex(2, 3, 0, -1, 0, &m, &n, &values, &found), 0);
    ASSERT_TRUE(found);
    ASSERT_NE(values, nullptr);
    EXPECT_EQ(m, 2);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(values[0], std::complex<double>(1.0, 0.5));
    EXPECT_EQ(values[1], std::complex<double>(3.0, 1.5));
    EXPECT_EQ(values[2], std::complex<double>(2.0, -0.5));
    EXPECT_EQ(values[3], std::complex<double>(4.0, -1.5));
    delete[] values;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    std::remove("/tmp/rsatb_hdf5_io_test.h5");
    return result;
}
