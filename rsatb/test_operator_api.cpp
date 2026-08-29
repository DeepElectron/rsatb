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

rsatb::model::SetupOptions make_file_setup_options(
    const std::vector<std::vector<double>>& kpoints,
    int nsamples = 1,
    bool compute_bounds = true
) {
    rsatb::model::SetupOptions options;
    options.kpoints = kpoints;
    options.nsamples = nsamples;
    options.compute_bounds = compute_bounds;
    return options;
}

/// Orthogonal tight-binding model backed by a synthetic Hamiltonian file.
std::unique_ptr<rsatb::model::TightBindingFileModel> make_file_model(
    const std::string& hfile,
    int norb = 1,
    const std::string& symbol = "H",
    double cutoff = 1.0
) {
    rsatb::model::TightBindingFileConfig config;
    config.hamiltonian = rsatb::io::HamiltonianFile{hfile};
    config.type_symbols = {symbol};
    config.type_norb = {norb};
    config.cutoff_radius = {cutoff};
    return rsatb::model::TightBindingFileModel::Create(std::move(config), MPI_COMM_WORLD);
}

struct FileBackedRsatb {
    std::unique_ptr<rsatb::model::TightBindingFileModel> model;
    std::unique_ptr<rsatb::RSATB> rsatb;

    rsatb::RSATB* operator->() const { return rsatb.get(); }
    rsatb::RSATB& operator*() const { return *rsatb; }
};

FileBackedRsatb make_single_atom_rsatb(
    const std::string& hfile,
    const std::vector<double>& pos = {0.0, 0.0, 0.0}
) {
    vbcsr::atomic::Structure structure;
    structure.n_atom = 1;
    structure.pos = pos;
    structure.z = {1};
    structure.cell = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    structure.pbc = {false, false, false};

    FileBackedRsatb result;
    result.model = make_file_model(hfile);
    result.rsatb = std::make_unique<rsatb::RSATB>(
        structure, *result.model, make_file_setup_options({{0.0, 0.0, 0.0}}),
        1, MPI_COMM_WORLD);
    return result;
}

FileBackedRsatb make_periodic_single_atom_rsatb(
    const std::string& hfile,
    int norb,
    const std::vector<std::vector<double>>& kpoints
) {
    static const int atom_index[] = {0};
    static const int atom_type[] = {0};
    static const int edge_index[] = {0, 0, 0, 0};
    static const int edge_shift_vec[] = {1, 0, 0, -1, 0, 0};
    static const double cell[] = {
        1.0, 0.0, 0.0,
        0.0, 8.0, 0.0,
        0.0, 0.0, 8.0
    };
    static const double pos[] = {0.0, 0.0, 0.0};

    rsatb::RSATB::AtomicGraphArrays graph;
    graph.n_atom = 1;
    graph.N_atom = 1;
    graph.atom_offset = 0;
    graph.n_edge = 2;
    graph.N_edge = 2;
    graph.atom_index = atom_index;
    graph.atom_type = atom_type;
    graph.edge_index = edge_index;
    graph.edge_shift_vec = edge_shift_vec;
    graph.cell = cell;
    graph.pos = pos;

    FileBackedRsatb result;
    result.model = make_file_model(hfile, norb);
    result.rsatb = std::make_unique<rsatb::RSATB>(
        graph, *result.model, make_file_setup_options(kpoints), 1, MPI_COMM_WORLD);
    return result;
}

rsatb::model::SktbConfig make_sktb_pz_config(bool with_overlap = false, double cutoff = 1.5) {
    rsatb::model::SktbConfig config;
    config.basis["C"] = {"pz"};
    config.hamiltonian.onsite["C"] = {{"pz", 0.0}};

    const auto radial = rsatb::model::sktb::MakeSpline({0.0, cutoff}, {0.2, 0.2}, cutoff);
    config.hamiltonian.SetPairChannel("C", "C", "pp", "pi", radial);
    if (with_overlap) {
        rsatb::model::sktb::SkParameters overlap;
        overlap.SetPairChannel("C", "C", "pp", "pi", radial);
        config.overlap = overlap;
    }
    return config;
}

rsatb::model::SetupOptions make_pairwise_setup_options(bool compute_bounds) {
    rsatb::model::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.compute_bounds = compute_bounds;
    return options;
}

struct SktbBackedRsatb {
    std::unique_ptr<rsatb::model::SktbModel> model;
    std::unique_ptr<rsatb::RSATB> rsatb;

    rsatb::RSATB* operator->() const { return rsatb.get(); }
    rsatb::RSATB& operator*() const { return *rsatb; }
};

SktbBackedRsatb make_sktb_points_rsatb(const rsatb::model::SktbConfig& config) {
    vbcsr::atomic::Structure structure;
    structure.n_atom = 2;
    structure.pos = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    structure.z = {6, 6};
    structure.cell = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    structure.pbc = {false, false, false};

    SktbBackedRsatb result;
    result.model = rsatb::model::SktbModel::Create(config);
    result.rsatb = std::make_unique<rsatb::RSATB>(
        structure, *result.model, make_pairwise_setup_options(true), 1, MPI_COMM_WORLD);
    return result;
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

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumRunsOnTheVelocityFallbackForCommutingSingleSite) {
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

    vbcsr::atomic::Structure structure;
    structure.n_atom = 1;
    structure.pos = {0.0, 0.0, 0.0};
    structure.z = {1};
    structure.cell = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    structure.pbc = {false, false, false};

    auto model = make_file_model(path);

    rsatb::model::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.energy_bounds = std::make_pair(-3.0, 4.0);

    rsatb::RSATB rsatb_inst(structure, *model, options, 1, MPI_COMM_WORLD);

    ASSERT_NE(rsatb_inst.operators(), nullptr);
    const auto bounds = rsatb_inst.operators()->bounds();
    EXPECT_DOUBLE_EQ(bounds.first, -3.0);
    EXPECT_DOUBLE_EQ(bounds.second, 4.0);

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumRunsOnTheVelocityFallbackForPeriodicAxis) {
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
        // The message is observables.hpp's: "requires a Gamma point in the
        // setup k-point list". Asserted on the un-hyphenated spelling the
        // code actually uses.
        EXPECT_NE(std::string(error.what()).find("Gamma point"), std::string::npos);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumFailsClearlyForOverlapWithoutDipoles) {
    // The periodic MSD runs on non-orthogonal bases through the commutator
    // velocity -- which a non-orthogonal Slater-Koster model cannot supply,
    // since it has no dipole integrals. The refusal must say so by name.
    auto rsatb_inst = make_sktb_points_rsatb(make_sktb_pz_config(true));

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;

    try {
        rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "x");
        FAIL() << "compute_msd_spectrum should reject a velocity-less non-orthogonal model";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_TRUE(message.find("commutator velocity") != std::string::npos ||
                    message.find("position-capable") != std::string::npos)
            << message;
    }
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumSpectralMethodRunsOnThePeriodicVelocityFallback) {
    // The spectral route on the generic commutator velocity an orthogonal
    // periodic model falls back to. One site per cell means every operator
    // commutes, so the spectrum must vanish like the stepwise routes'.
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.0, 0.0, 0.0}});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "x", "spectral");

    ASSERT_EQ(msd.size(), 3u * 16u);
    ASSERT_EQ(energy.size(), 16u);
    ASSERT_EQ(time.size(), 3u);
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumRejectsUnknownMethod) {
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.0, 0.0, 0.0}});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    try {
        rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "x", "concurrent");
        FAIL() << "compute_msd_spectrum should reject an unknown method";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("method"), std::string::npos);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, ComputeMSDSpectrumSpectralRunsOnNonPeriodicAxis) {
    // The velocity formalism is boundary-condition-agnostic: a non-periodic
    // axis carries a single R = 0 bond-moment image, which IS the
    // open-boundary [H, X], so method 'spectral' runs there like anywhere
    // else. Along z this fixture has no bonds at all, so the spectrum
    // vanishes.
    const std::string path = make_periodic_hdf5_hamiltonian_file(1, std::complex<double>(0.0, 0.0), std::complex<double>(0.2, 0.0));
    auto rsatb_inst = make_periodic_single_atom_rsatb(path, 1, {{0.0, 0.0, 0.0}});

    std::vector<double> msd;
    std::vector<double> energy;
    std::vector<double> time;
    rsatb_inst->compute_msd_spectrum(2, 0.5, 16, msd, energy, time, "z", "spectral");

    ASSERT_EQ(msd.size(), 3u * 16u);
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }

    std::remove(path.c_str());
}

TEST(RSATBOperatorApiTest, SktbGraphCutoffIsDerivedFromThePairCutoff) {
    // This replaces SetupRejectsSKTBWhenStoredCutoffIsTooSmall. That test
    // checked a cross-check between a caller-supplied graph cutoff and the
    // model's pair cutoffs; the graph cutoff now comes from the model, so the
    // two cannot disagree. What is worth asserting is the derivation itself.
    auto model = rsatb::model::SktbModel::Create(make_sktb_pz_config(false, 1.5));
    ASSERT_EQ(model->basis().cutoff_radius.size(), 1u);
    EXPECT_DOUBLE_EQ(model->basis().cutoff_radius[0], 0.75);
    EXPECT_EQ(model->basis().type_norb, std::vector<int>{1});
}

TEST(RSATBOperatorApiTest, ComputeACWorksForOrthogonalSKTBProvider) {
    auto rsatb_inst = make_sktb_points_rsatb(make_sktb_pz_config(false));

    std::vector<double> ac;
    std::vector<double> omega;
    rsatb_inst->compute_ac(4, 8, 300.0, 0.0, ac, omega);

    ASSERT_FALSE(ac.empty());
    ASSERT_FALSE(omega.empty());
}

TEST(RSATBOperatorApiTest, OverlapPresentSKTBStillRejectsTransportWithoutVelocity) {
    auto rsatb_inst = make_sktb_points_rsatb(make_sktb_pz_config(true));

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
    const int atom_index[] = {0};
    const int atom_type[] = {0};
    const int edge_index[] = {0, 0, 0, 0};
    const int edge_shift_vec[] = {1, 0, 0, -1, 0, 0};
    const double cell[] = {
        1.0, 0.0, 0.0,
        0.0, 8.0, 0.0,
        0.0, 0.0, 8.0
    };
    const double pos[] = {0.0, 0.0, 0.0};

    rsatb::RSATB::AtomicGraphArrays graph;
    graph.n_atom = 1;
    graph.N_atom = 1;
    graph.atom_offset = 0;
    graph.n_edge = 2;
    graph.N_edge = 2;
    graph.atom_index = atom_index;
    graph.atom_type = atom_type;
    graph.edge_index = edge_index;
    graph.edge_shift_vec = edge_shift_vec;
    graph.cell = cell;
    graph.pos = pos;

    auto model = rsatb::model::SktbModel::Create(make_sktb_pz_config(false, 1.1));
    rsatb::RSATB rsatb_inst(
        graph, *model, make_pairwise_setup_options(true), 1, MPI_COMM_WORLD);

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
