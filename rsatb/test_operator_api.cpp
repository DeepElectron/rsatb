#include "rsatb.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string make_hdf5_hamiltonian_file(const std::complex<double>& value) {
    const std::string path = "/tmp/rsatb_operator_api_test.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("0");

    const hsize_t dims[3] = {1, 1, 2};
    H5::DataSpace space(3, dims);
    H5::DataSet dataset = group.createDataSet("0_0_0_0_0", H5::PredType::NATIVE_DOUBLE, space);
    const double raw[2] = {value.real(), value.imag()};
    dataset.write(raw, H5::PredType::NATIVE_DOUBLE);
    return path;
}

std::unique_ptr<rsatb::RSATB> make_single_atom_rsatb(const std::string& hfile, const std::vector<double>& pos = {0.0, 0.0, 0.0}) {
    const std::vector<int> z = {1};
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const std::vector<double> r_max = {1.0};
    const std::vector<int> type_norb = {1};

    auto rsatb_inst = std::make_unique<rsatb::RSATB>(1, pos, z, cell, pbc, r_max, type_norb, false, MPI_COMM_WORLD);
    rsatb_inst->setup(hfile, {{0.0, 0.0, 0.0}}, {}, "", {}, "", 1e-7, 1e-5, 20, "graph", "None", 1, true);
    return rsatb_inst;
}

} // namespace

TEST(RSATBOperatorApiTest, ComputeACFailsClearlyWithoutVelocityProvider) {
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(2.0, 0.0));
    auto rsatb_inst = make_single_atom_rsatb(path);

    std::vector<double> ac;
    std::vector<double> omega;

    try {
        rsatb_inst->compute_ac(4, 8, 300.0, 0.0, ac, omega);
        FAIL() << "compute_ac should require a velocity-capable provider";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("velocity-capable operator provider"), std::string::npos);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeDCSpectrumFailsClearlyWithoutVelocityProvider) {
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(2.0, 0.0));
    auto rsatb_inst = make_single_atom_rsatb(path);

    std::vector<double> sigma;
    std::vector<double> energy;

    try {
        rsatb_inst->compute_dc_spectrum(8, sigma, energy);
        FAIL() << "compute_dc_spectrum should require a velocity-capable provider";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("velocity-capable operator provider"), std::string::npos);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumUsesFallbackPositionProviderForCommutingSingleSite) {
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(2.0, 0.0));
    auto rsatb_inst = make_single_atom_rsatb(path, {1.25, -0.5, 0.75});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    rsatb_inst->compute_msd_spectrum(2, 1.0, 16, msd, energy, time, "x");

    ASSERT_EQ(msd.size(), 3u * 16u);
    ASSERT_EQ(energy.size(), 16u);
    ASSERT_EQ(time.size(), 3u);
    EXPECT_DOUBLE_EQ(time[0], 0.0);
    EXPECT_DOUBLE_EQ(time[1], 1.0);
    EXPECT_DOUBLE_EQ(time[2], 2.0);
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeQuantumMetricFailsClearlyWithoutVelocityProvider) {
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(2.0, 0.0));
    auto rsatb_inst = make_single_atom_rsatb(path);

    std::vector<double> gxx;
    std::vector<double> sigma_xx;
    std::vector<double> omega;

    try {
        rsatb_inst->compute_quantum_metric(4, 8, 0.0, 0.0, gxx, sigma_xx, omega);
        FAIL() << "compute_quantum_metric should require a velocity-capable provider";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("velocity-capable operator provider"), std::string::npos);
    }

    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
