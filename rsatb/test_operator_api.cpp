#include "rsatb.hpp"
#include "test_data_path.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>

#include <cstdio>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
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

void write_complex_block_dataset(H5::Group& group, const std::string& name, int norb, const std::complex<double>& diagonal_value) {
    const hsize_t dims[3] = {static_cast<hsize_t>(norb), static_cast<hsize_t>(norb), 2};
    H5::DataSpace space(3, dims);
    H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);

    std::vector<double> raw(static_cast<size_t>(norb) * static_cast<size_t>(norb) * 2u, 0.0);
    for (int i = 0; i < norb; ++i) {
        const size_t offset = (static_cast<size_t>(i) * static_cast<size_t>(norb) + static_cast<size_t>(i)) * 2u;
        raw[offset] = diagonal_value.real();
        raw[offset + 1] = diagonal_value.imag();
    }
    dataset.write(raw.data(), H5::PredType::NATIVE_DOUBLE);
}

std::string make_periodic_hdf5_hamiltonian_file(int norb, const std::complex<double>& onsite, const std::complex<double>& hop) {
    const std::string path = "/tmp/rsatb_periodic_operator_api_test.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("0");

    write_complex_block_dataset(group, "0_0_0_0_0", norb, onsite);
    write_complex_block_dataset(group, "0_0_1_0_0", norb, hop);
    write_complex_block_dataset(group, "0_0_-1_0_0", norb, hop);
    return path;
}

void setup_lcao_hdf5(
    rsatb::RSATB& rsatb_inst,
    const std::string& hfile,
    const std::vector<std::vector<double>>& kpoints,
    const std::vector<std::string>& orb_names = {},
    const std::string& orb_dir = "",
    const std::vector<std::string>& psd_names = {},
    const std::string& psd_dir = "",
    const std::string& precond = "None",
    int nsamples = 1,
    bool compute_bounds = true
) {
    rsatb::model::SetupOptions options;
    options.kpoints = kpoints;
    options.nsamples = nsamples;
    options.compute_bounds = compute_bounds;
    options.preconditioner.mode = precond;

    rsatb::model::LCAOModelConfig model;
    model.input = rsatb::model::Hdf5Input{};
    model.input->filename = hfile;
    model.integrals.orb_names = orb_names;
    model.integrals.orb_dir = orb_dir;
    model.integrals.psd_names = psd_names;
    model.integrals.psd_dir = psd_dir;

    rsatb_inst.setup(model, options);
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
    setup_lcao_hdf5(*rsatb_inst, hfile, {{0.0, 0.0, 0.0}});
    return rsatb_inst;
}

std::unique_ptr<rsatb::RSATB> make_periodic_single_atom_rsatb(
    const std::string& hfile,
    int norb,
    const std::vector<std::vector<double>>& kpoints,
    const std::vector<std::string>& orb_names = {},
    const std::string& orb_dir = ""
) {
    const size_t n_atom = 1;
    const size_t N_atom = 1;
    const size_t atom_offset = 0;
    const size_t n_edge = 2;
    const size_t N_edge = 2;

    const int atom_index[] = {0};
    const int atom_type[] = {0};
    const int edge_index[] = {
        0, 0,
        0, 0
    };
    const int type_norb[] = {norb};
    const int edge_shift_vec[] = {
         1, 0, 0,
        -1, 0, 0
    };
    const double cell[] = {
        1.0, 0.0, 0.0,
        0.0, 8.0, 0.0,
        0.0, 0.0, 8.0
    };
    const double pos[] = {0.0, 0.0, 0.0};

    auto rsatb_inst = std::make_unique<rsatb::RSATB>(
        1,
        n_atom, N_atom, atom_offset, n_edge, N_edge,
        atom_index, atom_type, edge_index, type_norb, std::vector<std::string>{}, edge_shift_vec,
        cell, pos, false, MPI_COMM_WORLD);

    setup_lcao_hdf5(*rsatb_inst, hfile, kpoints, orb_names, orb_dir);
    return rsatb_inst;
}

rsatb::backend::SKTBModel make_sktb_pz_model(bool with_overlap = false, double cutoff = 1.5) {
    rsatb::backend::SKTBInput source;
    source.basis["C"] = {"pz"};
    source.hamiltonian.onsite["C"] = {{"pz", 0.0}};

    rsatb::backend::SKPairInput pair;
    pair.species = {"C", "C"};
    rsatb::backend::SKRadialFunction radial;
    radial.kind = "spline";
    radial.r = {0.0, cutoff};
    radial.values = {0.2, 0.2};
    radial.r_cut = cutoff;
    radial.spline_second_derivatives = radial.build_spline_second_derivatives();
    pair.channels["pp_pi"] = radial;
    source.hamiltonian.pairs.push_back(pair);

    if (with_overlap) {
        source.overlap = rsatb::backend::SKMatrixInput{};
        source.overlap->pairs.push_back(pair);
    }

    return rsatb::backend::build_sktb_model(source, {"C"});
}

rsatb::model::SetupOptions make_pairwise_setup_options(bool compute_bounds) {
    rsatb::model::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.compute_bounds = compute_bounds;
    options.preconditioner.mode = "None";
    return options;
}

std::unique_ptr<rsatb::RSATB> make_sktb_points_rsatb(
    const rsatb::backend::SKTBModel& source,
    bool with_overlap = false
) {
    const std::vector<double> pos = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    };
    const std::vector<int> z = {6, 6};
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const auto prepared = rsatb::backend::prepare_sktb_atomic_data(source);

    auto rsatb_inst = std::make_unique<rsatb::RSATB>(
        1, pos, z, cell, pbc, prepared.r_max_per_type, prepared.type_norb, false, MPI_COMM_WORLD);

    rsatb::model::SKTBModelConfig model;
    model.native = source;
    rsatb_inst->setup(model, make_pairwise_setup_options(true));
    (void)with_overlap;
    return rsatb_inst;
}

} // namespace

TEST(RSATBOperatorApiTest, ComputeACUsesOrthogonalVelocityFallbackForHdf5Hamiltonian) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.125, 0.0, 0.0}});

    std::vector<double> ac;
    std::vector<double> omega;
    rsatb_inst->compute_ac(4, 8, 300.0, 0.0, ac, omega);

    ASSERT_FALSE(ac.empty());
    ASSERT_FALSE(omega.empty());

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeDCSpectrumUsesOrthogonalVelocityFallbackForHdf5Hamiltonian) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.125, 0.0, 0.0}});

    std::vector<double> sigma;
    std::vector<double> energy;
    rsatb_inst->compute_dc_spectrum(8, sigma, energy);

    ASSERT_FALSE(sigma.empty());
    ASSERT_FALSE(energy.empty());

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

TEST(RSATBOperatorApiTest, ComputeQuantumMetricUsesOrthogonalVelocityFallbackForHdf5Hamiltonian) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.125, 0.0, 0.0}});

    std::vector<double> gxx;
    std::vector<double> sigma_xx;
    std::vector<double> omega;
    rsatb_inst->compute_quantum_metric(4, 8, 0.0, 0.0, gxx, sigma_xx, omega);

    ASSERT_FALSE(gxx.empty());
    ASSERT_FALSE(sigma_xx.empty());
    ASSERT_FALSE(omega.empty());

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, AutomaticBoundsSkipsZeroHamiltonian) {
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(0.0, 0.0));
    auto rsatb_inst = make_single_atom_rsatb(path);

    ASSERT_NE(rsatb_inst->operators(), nullptr);
    const auto bounds = rsatb_inst->operators()->bounds();
    EXPECT_NEAR(bounds.first, -0.5, 1e-12);
    EXPECT_NEAR(bounds.second, 0.5, 1e-12);

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, SetupUsesExplicitEnergyBounds) {
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(2.0, 0.0));
    const std::vector<int> z = {1};
    const std::vector<double> pos = {0.0, 0.0, 0.0};
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const std::vector<double> r_max = {1.0};
    const std::vector<int> type_norb = {1};

    rsatb::RSATB rsatb_inst(1, pos, z, cell, pbc, r_max, type_norb, false, MPI_COMM_WORLD);

    rsatb::model::LCAOModelConfig model;
    model.input = rsatb::model::Hdf5Input{};
    model.input->filename = path;

    rsatb::model::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.energy_bounds = std::make_pair(-3.0, 4.0);
    options.preconditioner.mode = "None";
    rsatb_inst.setup(model, options);

    ASSERT_NE(rsatb_inst.operators(), nullptr);
    const auto bounds = rsatb_inst.operators()->bounds();
    EXPECT_DOUBLE_EQ(bounds.first, -3.0);
    EXPECT_DOUBLE_EQ(bounds.second, 4.0);

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumUsesGammaAffinePathForPeriodicAxisWithoutPositionProvider) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.0, 0.0, 0.0}});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "x");

    ASSERT_EQ(msd.size(), 3u * 16u);
    ASSERT_EQ(energy.size(), 16u);
    ASSERT_EQ(time.size(), 3u);
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumUsesGammaSettingForNonPeriodicAxisWhenOtherKPointsExist) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.125, 0.0, 0.0}, {0.0, 0.0, 0.0}});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "z");

    ASSERT_EQ(msd.size(), 3u * 16u);
    ASSERT_EQ(energy.size(), 16u);
    ASSERT_EQ(time.size(), 3u);
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumFailsClearlyWithoutGammaKPoint) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.125, 0.0, 0.0}});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;

    try {
        rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "z");
        FAIL() << "compute_msd_spectrum should reject MSD when Gamma is not configured";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("Gamma-point"), std::string::npos);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumFailsClearlyForPeriodicOverlapSystems) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(13, std::complex<double>(0.0, 0.0), std::complex<double>(0.05, 0.0));
    const std::string orb_dir = rsatb::test::data_path("diamond/");
    const std::vector<std::string> orb_names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 13, {{0.0, 0.0, 0.0}}, orb_names, orb_dir);

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;

    try {
        rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "x");
        FAIL() << "compute_msd_spectrum should reject periodic overlap-present MSD in this pass";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("orthogonal systems only"), std::string::npos);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, SetupRejectsSKTBWhenStoredCutoffIsTooSmall) {
    const std::vector<double> pos = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    };
    const std::vector<int> z = {6, 6};
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const std::vector<double> r_max = {0.1};
    const std::vector<int> type_norb = {1};

    rsatb::RSATB rsatb_inst(1, pos, z, cell, pbc, r_max, type_norb, false, MPI_COMM_WORLD);
    auto source = make_sktb_pz_model(false, 1.5);
    rsatb::model::SKTBModelConfig model;
    model.native = source;

    try {
        rsatb_inst.setup(model, make_pairwise_setup_options(false));
        FAIL() << "setup should reject SKTB when the stored graph cutoff is too small";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("graph cutoff"), std::string::npos);
    }
}

TEST(RSATBOperatorApiTest, ComputeACWorksForOrthogonalSKTBProvider) {
    auto rsatb_inst = make_sktb_points_rsatb(make_sktb_pz_model(false));

    std::vector<double> ac;
    std::vector<double> omega;
    rsatb_inst->compute_ac(4, 8, 300.0, 0.0, ac, omega);

    ASSERT_FALSE(ac.empty());
    ASSERT_FALSE(omega.empty());
}

TEST(RSATBOperatorApiTest, OverlapPresentSKTBStillRejectsTransportWithoutVelocity) {
    auto rsatb_inst = make_sktb_points_rsatb(make_sktb_pz_model(true), true);

    std::vector<double> ac;
    std::vector<double> omega;
    try {
        rsatb_inst->compute_ac(4, 8, 300.0, 0.0, ac, omega);
        FAIL() << "compute_ac should reject overlap-present SKTB until a nonorthogonal velocity model exists";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("velocity-capable model/operator setup"), std::string::npos);
    }
}

TEST(RSATBOperatorApiTest, PeriodicOrthogonalSKTBUsesGammaAffineMSDPath) {
    const size_t n_atom = 1;
    const size_t N_atom = 1;
    const size_t atom_offset = 0;
    const size_t n_edge = 2;
    const size_t N_edge = 2;

    const int atom_index[] = {0};
    const int atom_type[] = {0};
    const int edge_index[] = {0, 0, 0, 0};
    const int type_norb[] = {1};
    const int edge_shift_vec[] = {1, 0, 0, -1, 0, 0};
    const double cell[] = {
        1.0, 0.0, 0.0,
        0.0, 8.0, 0.0,
        0.0, 0.0, 8.0
    };
    const double pos[] = {0.0, 0.0, 0.0};

    rsatb::RSATB rsatb_inst(
        1,
        n_atom, N_atom, atom_offset, n_edge, N_edge,
        atom_index, atom_type, edge_index, type_norb, std::vector<std::string>{"C"}, edge_shift_vec,
        cell, pos, false, MPI_COMM_WORLD);

    auto source = make_sktb_pz_model(false, 1.1);
    rsatb::model::SKTBModelConfig model;
    model.native = source;
    rsatb_inst.setup(model, make_pairwise_setup_options(true));

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    rsatb_inst.compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "x");

    ASSERT_EQ(msd.size(), 3u * 16u);
    ASSERT_EQ(energy.size(), 16u);
    ASSERT_EQ(time.size(), 3u);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
