// Round-trip tests for the DFTB .skf reader.
//
// These validate the parser, the column-to-channel mapping, and the unit
// conversion, by writing a file with known integrals and checking that the
// resulting Hamiltonian matches an equivalent hand-built SKTB model.
//
// They do NOT validate physical correctness against a released parameter set:
// that needs a DFTB+ reference, which is also what would settle the
// heteronuclear file orientation currently gated by SkfHeteronuclearPolicy.

#include "../sktb_model.hpp"
#include "../../../io/skf.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

using namespace rsatb::model::sktb;

namespace rmodel = rsatb::model;
namespace rio = rsatb::io;

namespace {

constexpr double kHartree = rio::kElectronVoltPerHartree;
constexpr double kBohr = rio::kAngstromPerBohrLength;

struct SkfIntegrals {
    double ss_sigma_h = 0.0;
    double sp_sigma_h = 0.0;
    double pp_sigma_h = 0.0;
    double pp_pi_h = 0.0;
    double ss_sigma_s = 0.0;
    double sp_sigma_s = 0.0;
    double pp_sigma_s = 0.0;
    double pp_pi_s = 0.0;
};

/// Writes a homonuclear .skf whose integrals are constant on the grid.
///
/// Values are given in rsatb units (eV, Angstrom) and written out in the
/// file's units (Hartree, Bohr), so the test exercises the conversion in the
/// direction a real file would.
std::string WriteConstantSkf(
    const std::string& path,
    double grid_dist_angstrom,
    int n_grid,
    const SkfIntegrals& v,
    double e_s_ev,
    double e_p_ev,
    bool homonuclear = true
) {
    std::ofstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    // Full round-trip precision: the default six significant digits would show
    // up as a ~1e-6 discrepancy that looks like a unit-conversion error.
    file << std::setprecision(17);

    file << (grid_dist_angstrom / kBohr) << " " << n_grid << "\n";
    // The onsite line exists only in homonuclear files.
    if (homonuclear) {
        // Ed Ep Es  SPE  Ud Up Us  fd fp fs
        file << 0.0 << " " << (e_p_ev / kHartree) << " " << (e_s_ev / kHartree)
             << " 0.0  0.0 0.3 0.4  0.0 2.0 2.0\n";
    }
    // mass + repulsive polynomial, unused
    file << "12.01 " << "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n";

    for (int row = 0; row < n_grid; ++row) {
        // Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0
        file << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " "
             << (v.pp_sigma_h / kHartree) << " " << (v.pp_pi_h / kHartree) << " "
             << 0.0 << " " << (v.sp_sigma_h / kHartree) << " " << (v.ss_sigma_h / kHartree) << " "
        // Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0
             << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " "
             << v.pp_sigma_s << " " << v.pp_pi_s << " "
             << 0.0 << " " << v.sp_sigma_s << " " << v.ss_sigma_s << "\n";
    }
    file.close();
    return path;
}

std::unique_ptr<AtomicData> MakeTwoAtomData(const rmodel::BasisSpec& basis, double separation) {
    const std::vector<double> pos = {0.0, 0.0, 0.0, 0.0, 0.0, separation};
    const std::vector<int> z = {6, 6};
    const std::vector<double> cell = {
        20.0, 0.0, 0.0,
        0.0, 20.0, 0.0,
        0.0, 0.0, 20.0
    };
    const std::vector<bool> pbc = {false, false, false};
    return std::unique_ptr<AtomicData>(AtomicData::from_points(
        pos, z, cell, pbc, basis.cutoff_radius, basis.type_norb, MPI_COMM_WORLD));
}

std::vector<std::complex<double>> BlockAt(
    BlockSpMat<std::complex<double>>& matrix,
    int row,
    int col
) {
    const int start = matrix.row_ptr()[row];
    const int end = matrix.row_ptr()[row + 1];
    for (int k = start; k < end; ++k) {
        if (matrix.col_ind()[k] != col) continue;
        const size_t n = matrix.block_size_elements(k);
        const std::complex<double>* data = matrix.block_data(k);
        return std::vector<std::complex<double>>(data, data + n);
    }
    return {};
}

std::vector<std::complex<double>> GammaBlock(
    const rmodel::Model& model,
    AtomicData& atomic_data,
    int row,
    int col
) {
    auto images = model.BuildHamiltonianImages(atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images.sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const int start = gamma->row_ptr()[row];
    const int end = gamma->row_ptr()[row + 1];
    for (int k = start; k < end; ++k) {
        if (gamma->col_ind()[k] != col) continue;
        const size_t n = gamma->block_size_elements(k);
        const std::complex<double>* data = gamma->block_data(k);
        return std::vector<std::complex<double>>(data, data + n);
    }
    return {};
}

}  // namespace

TEST(SkfTest, ParsesGridUnitsAndChannelColumns) {
    const std::string path = "test_skf_constant_C-C.skf";
    SkfIntegrals v;
    v.ss_sigma_h = -1.5;
    v.sp_sigma_h = 0.75;
    v.pp_sigma_h = 2.25;
    v.pp_pi_h = -0.5;
    v.ss_sigma_s = 0.11;
    v.sp_sigma_s = 0.22;
    v.pp_sigma_s = 0.33;
    v.pp_pi_s = 0.44;
    WriteConstantSkf(path, 0.1, 30, v, -13.5, -5.25);

    const auto table = rio::ReadSkfFile(path, "C", "C");

    // Grid: 30 points at 0.1 A spacing, first point one step out.
    ASSERT_EQ(table.r.size(), 30u);
    EXPECT_NEAR(table.r.front(), 0.1, 1e-12);
    EXPECT_NEAR(table.r.back(), 3.0, 1e-12);
    EXPECT_NEAR(table.cutoff(), 3.0, 1e-12);

    // Onsite energies come back in eV, and in s, p, d order rather than the
    // file's d, p, s.
    ASSERT_TRUE(table.has_onsite);
    EXPECT_NEAR(table.onsite.energy[0], -13.5, 1e-9);
    EXPECT_NEAR(table.onsite.energy[1], -5.25, 1e-9);
    EXPECT_NEAR(table.onsite.energy[2], 0.0, 1e-12);


    const size_t ss = static_cast<size_t>(rio::SkfChannel::kSsSigma);
    const size_t sp = static_cast<size_t>(rio::SkfChannel::kSpSigma);
    const size_t pps = static_cast<size_t>(rio::SkfChannel::kPpSigma);
    const size_t ppp = static_cast<size_t>(rio::SkfChannel::kPpPi);

    // H is converted Hartree -> eV; S is dimensionless and must not be scaled.
    EXPECT_NEAR(table.h[ss].front(), -1.5, 1e-9);
    EXPECT_NEAR(table.h[sp].front(), 0.75, 1e-9);
    EXPECT_NEAR(table.h[pps].front(), 2.25, 1e-9);
    EXPECT_NEAR(table.h[ppp].front(), -0.5, 1e-9);
    EXPECT_NEAR(table.s[ss].front(), 0.11, 1e-12);
    EXPECT_NEAR(table.s[sp].front(), 0.22, 1e-12);
    EXPECT_NEAR(table.s[pps].front(), 0.33, 1e-12);
    EXPECT_NEAR(table.s[ppp].front(), 0.44, 1e-12);

    std::remove(path.c_str());
}

TEST(SkfTest, SkfModelMatchesEquivalentHandBuiltSktbModel) {
    const std::string dir = ".";
    const std::string path = "C-C.skf";
    SkfIntegrals v;
    v.ss_sigma_h = -1.5;
    v.sp_sigma_h = 0.75;
    v.pp_sigma_h = 2.25;
    v.pp_pi_h = -0.5;
    v.ss_sigma_s = 0.11;
    v.sp_sigma_s = 0.22;
    v.pp_sigma_s = 0.33;
    v.pp_pi_s = 0.44;
    const double e_s = -13.5;
    const double e_p = -5.25;
    WriteConstantSkf(path, 0.1, 30, v, e_s, e_p);

    rmodel::DftbSkfConfig skf_config;
    skf_config.skf_dir = dir;
    skf_config.basis["C"] = {"s", "p"};
    auto skf_model = rmodel::DftbModel::FromSkfDirectory(skf_config);

    // The same parameters written directly as an SKTB-style model. A constant
    // spline over the same grid reproduces the file's constant integrals.
    std::vector<double> r(30);
    for (int i = 0; i < 30; ++i) r[static_cast<size_t>(i)] = 0.1 * (i + 1);
    const auto constant = [&](double value) {
        return MakeSpline(r, std::vector<double>(30, value), 3.0);
    };

    rmodel::DftbConfig hand_config;
    hand_config.basis["C"] = {"s", "p"};
    hand_config.hamiltonian.SetOnsite("C", "s", e_s);
    hand_config.hamiltonian.SetOnsite("C", "p", e_p);
    hand_config.hamiltonian.SetPairChannel("C", "C", "ss", "sigma", constant(v.ss_sigma_h));
    hand_config.hamiltonian.SetPairChannel("C", "C", "sp", "sigma", constant(v.sp_sigma_h));
    hand_config.hamiltonian.SetPairChannel("C", "C", "pp", "sigma", constant(v.pp_sigma_h));
    hand_config.hamiltonian.SetPairChannel("C", "C", "pp", "pi", constant(v.pp_pi_h));
    hand_config.overlap.SetPairChannel("C", "C", "ss", "sigma", constant(v.ss_sigma_s));
    hand_config.overlap.SetPairChannel("C", "C", "sp", "sigma", constant(v.sp_sigma_s));
    hand_config.overlap.SetPairChannel("C", "C", "pp", "sigma", constant(v.pp_sigma_s));
    hand_config.overlap.SetPairChannel("C", "C", "pp", "pi", constant(v.pp_pi_s));
    auto hand_model = rmodel::DftbModel::Create(hand_config);

    // Same derived basis.
    EXPECT_EQ(skf_model->basis().type_symbols, hand_model->basis().type_symbols);
    EXPECT_EQ(skf_model->basis().type_norb, hand_model->basis().type_norb);
    EXPECT_EQ(skf_model->basis().type_norb, (std::vector<int>{4}));
    EXPECT_NEAR(skf_model->basis().cutoff_radius[0], hand_model->basis().cutoff_radius[0], 1e-12);

    // Same Hamiltonian, onsite and off-diagonal.
    auto skf_data = MakeTwoAtomData(skf_model->basis(), 1.5);
    auto hand_data = MakeTwoAtomData(hand_model->basis(), 1.5);

    const auto skf_onsite = GammaBlock(*skf_model, *skf_data, 0, 0);
    const auto hand_onsite = GammaBlock(*hand_model, *hand_data, 0, 0);
    ASSERT_EQ(skf_onsite.size(), 16u);
    ASSERT_EQ(skf_onsite.size(), hand_onsite.size());
    for (size_t i = 0; i < skf_onsite.size(); ++i) {
        EXPECT_NEAR(skf_onsite[i].real(), hand_onsite[i].real(), 1e-9) << "onsite index " << i;
    }
    // Onsite diagonal is (e_s, e_p, e_p, e_p) in the declared basis order.
    EXPECT_NEAR(skf_onsite[0].real(), e_s, 1e-9);
    EXPECT_NEAR(skf_onsite[5].real(), e_p, 1e-9);

    const auto skf_pair = GammaBlock(*skf_model, *skf_data, 0, 1);
    const auto hand_pair = GammaBlock(*hand_model, *hand_data, 0, 1);
    ASSERT_EQ(skf_pair.size(), 16u);
    ASSERT_EQ(skf_pair.size(), hand_pair.size());
    for (size_t i = 0; i < skf_pair.size(); ++i) {
        EXPECT_NEAR(skf_pair[i].real(), hand_pair[i].real(), 1e-9) << "pair index " << i;
        EXPECT_NEAR(skf_pair[i].imag(), hand_pair[i].imag(), 1e-12) << "pair index " << i;
    }

    std::remove(path.c_str());
}

// A C(s,p) / H(s) pair: the classic DFTB two-species system, and the case where
// the declared bases force a unique reading of the sp channel.
TEST(SkfTest, TwoSpeciesCarbonHydrogenBuildsThePhysicalSigmaBond) {
    SkfIntegrals cc;
    cc.ss_sigma_h = -1.0;
    cc.pp_sigma_h = 1.5;
    cc.pp_pi_h = -0.4;
    cc.ss_sigma_s = 0.05;
    cc.pp_sigma_s = 0.06;
    cc.pp_pi_s = 0.07;
    WriteConstantSkf("C-C.skf", 0.1, 30, cc, -13.5, -5.25);

    SkfIntegrals hh;
    hh.ss_sigma_h = -0.9;
    hh.ss_sigma_s = 0.08;
    WriteConstantSkf("H-H.skf", 0.1, 30, hh, -6.4, 0.0);

    // The only C-H coupling a p-less hydrogen allows is <p(C)|H|s(H)>, carried
    // by the sp column. Give it a distinctive value so it is unmistakable.
    SkfIntegrals ch;
    ch.ss_sigma_h = -2.0;
    ch.sp_sigma_h = 3.0;
    ch.ss_sigma_s = 0.11;
    ch.sp_sigma_s = 0.13;
    WriteConstantSkf("C-H.skf", 0.1, 30, ch, 0.0, 0.0, /*homonuclear=*/false);
    WriteConstantSkf("H-C.skf", 0.1, 30, ch, 0.0, 0.0, /*homonuclear=*/false);

    rmodel::DftbSkfConfig config;
    config.skf_dir = ".";
    config.basis["C"] = {"s", "p"};
    config.basis["H"] = {"s"};
    auto model = rmodel::DftbModel::FromSkfDirectory(config);

    // Ascending atomic number: H is type 0, C is type 1.
    ASSERT_EQ(model->basis().type_symbols, (std::vector<std::string>{"H", "C"}));
    ASSERT_EQ(model->basis().type_norb, (std::vector<int>{1, 4}));

    // A C-H dimer along z.
    const std::vector<double> pos = {0.0, 0.0, 0.0, 0.0, 0.0, 1.1};
    const std::vector<int> z = {6, 1};
    const std::vector<double> cell = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
    const std::vector<bool> pbc = {false, false, false};
    auto atomic_data = std::unique_ptr<AtomicData>(AtomicData::from_points(
        pos, z, cell, pbc, model->basis().cutoff_radius, model->basis().type_norb, MPI_COMM_WORLD));

    auto images = model->BuildHamiltonianImages(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images.sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    // Locate the carbon (4 orbitals) and hydrogen (1 orbital) rows.
    int carbon = -1;
    int hydrogen = -1;
    for (int i = 0; i < 2; ++i) {
        int norb = 0;
        atomic_data->get_atom_norb(i, &norb);
        if (norb == 4) carbon = i;
        if (norb == 1) hydrogen = i;
    }
    ASSERT_GE(carbon, 0);
    ASSERT_GE(hydrogen, 0);

    const auto ch_block = BlockAt(*gamma, carbon, hydrogen);
    ASSERT_EQ(ch_block.size(), 4u) << "carbon row x hydrogen column is 4 x 1";

    // Carbon's basis is s, py, pz, px in DeePTB magnetic order, and the bond
    // runs along z. The decisive assertion: pz couples to hydrogen's s, while
    // py and px do not. If the sp column had been read onto the hydrogen side
    // it would have been dropped -- hydrogen has no p -- and this coupling
    // would be zero, which is the failure this test exists to catch.
    EXPECT_GT(std::abs(ch_block[2]), 1e-6) << "p_z on carbon must bond to s on hydrogen";
    EXPECT_NEAR(std::abs(ch_block[1]), 0.0, 1e-12) << "p_y is pi to a sigma-only partner";
    EXPECT_NEAR(std::abs(ch_block[3]), 0.0, 1e-12) << "p_x is pi to a sigma-only partner";
    EXPECT_GT(std::abs(ch_block[0]), 1e-6) << "s-s sigma coupling must be present";

    // Hermiticity: the reverse block is the conjugate transpose of this one.
    const auto hc_block = BlockAt(*gamma, hydrogen, carbon);
    ASSERT_EQ(hc_block.size(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(hc_block[i].real(), ch_block[i].real(), 1e-9) << "index " << i;
        EXPECT_NEAR(hc_block[i].imag(), -ch_block[i].imag(), 1e-9) << "index " << i;
    }

    for (const char* name : {"C-C.skf", "H-H.skf", "C-H.skf", "H-C.skf"}) {
        std::remove(name);
    }
}

TEST(SkfTest, GenuinelyAmbiguousMixedShellPairIsRejected) {
    // Both species carry both shells, so the single sp column could be either
    // <s(C)|H|p(N)> or <p(C)|H|s(N)> and the file does not say which.
    SkfIntegrals v;
    v.ss_sigma_h = -1.0;
    v.sp_sigma_h = 2.0;
    v.pp_sigma_h = 1.0;
    v.ss_sigma_s = 0.1;
    WriteConstantSkf("C-C.skf", 0.1, 30, v, -13.5, -5.25);
    WriteConstantSkf("N-N.skf", 0.1, 30, v, -18.0, -7.0);
    WriteConstantSkf("C-N.skf", 0.1, 30, v, 0.0, 0.0, /*homonuclear=*/false);
    WriteConstantSkf("N-C.skf", 0.1, 30, v, 0.0, 0.0, /*homonuclear=*/false);

    rmodel::DftbSkfConfig config;
    config.skf_dir = ".";
    config.basis["C"] = {"s", "p"};
    config.basis["N"] = {"s", "p"};

    try {
        rmodel::DftbModel::FromSkfDirectory(config);
        FAIL() << "an ambiguous mixed-shell channel should be refused";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("ambiguous"), std::string::npos) << message;
        EXPECT_NE(message.find("sp_sigma"), std::string::npos) << message;
    }

    for (const char* name : {"C-C.skf", "N-N.skf", "C-N.skf", "N-C.skf"}) {
        std::remove(name);
    }
}

TEST(SkfTest, MissingFileReportsThePathItLookedFor) {
    rmodel::DftbSkfConfig config;
    config.skf_dir = "/nonexistent-skf-dir";
    config.basis["C"] = {"s"};

    try {
        rmodel::DftbModel::FromSkfDirectory(config);
        FAIL() << "a missing .skf should be reported";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("C-C.skf"), std::string::npos) << message;
    }
}

TEST(SkfTest, ExtendedFOrbitalFilesAreRejected) {
    const std::string path = "test_skf_extended_C-C.skf";
    {
        std::ofstream file(path);
        file << "@ extended format\n0.1 10\n";
    }

    try {
        rio::ReadSkfFile(path, "C", "C");
        FAIL() << "extended (f-orbital) files should be refused";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("Extended"), std::string::npos);
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
