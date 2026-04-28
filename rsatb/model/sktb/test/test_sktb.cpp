#include "../../operator_set.hpp"
#include "../sktb.hpp"

#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cmath>
#include <memory>
#include <vector>

using namespace rsatb::backend;

namespace rmodel = rsatb::model;

namespace {

SKRadialFunction constant_spline(double value, double cutoff = 2.5) {
    SKRadialFunction cfg;
    cfg.kind = "spline";
    cfg.r = {0.0, cutoff};
    cfg.values = {value, value};
    cfg.r_cut = cutoff;
    cfg.spline_second_derivatives = cfg.build_spline_second_derivatives();
    return cfg;
}

SKRadialFunction exp_poly_constant(double value, double r_on, double r_cut) {
    SKRadialFunction cfg;
    cfg.kind = "exp_poly";
    cfg.coeffs = {value};
    cfg.lambda_value = 0.0;
    cfg.r_on = r_on;
    cfg.r_cut = r_cut;
    return cfg;
}

std::unique_ptr<AtomicData> make_points_data(
    const std::vector<double>& pos,
    const std::vector<int>& z,
    const SKTBModel& sktb
) {
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const auto prepared = prepare_sktb_atomic_data(sktb);
    return std::unique_ptr<AtomicData>(
        AtomicData::from_points(pos, z, cell, pbc, prepared.r_max_per_type, prepared.type_norb, MPI_COMM_WORLD));
}

std::unique_ptr<AtomicData> make_points_data(
    const std::vector<double>& pos,
    const std::vector<int>& z,
    const SKTBInput& sktb
) {
    const auto model = build_sktb_model(sktb, unique_type_symbols_from_atomic_numbers(z));
    return make_points_data(pos, z, model);
}

std::unique_ptr<AtomicData> make_periodic_single_atom_data(double cell_x, const SKTBModel& sktb) {
    const auto prepared = prepare_sktb_atomic_data(sktb);

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
    const int edge_shift_vec[] = {
         1, 0, 0,
        -1, 0, 0
    };
    const double cell[] = {
        cell_x, 0.0, 0.0,
        0.0, 8.0, 0.0,
        0.0, 0.0, 8.0
    };
    const double pos[] = {0.25, -0.5, 0.75};

    return std::make_unique<AtomicData>(
        n_atom, N_atom, atom_offset, n_edge, N_edge,
        atom_index, atom_type, edge_index, prepared.type_norb.data(), edge_shift_vec,
        cell, pos, MPI_COMM_WORLD);
}

std::unique_ptr<AtomicData> make_directed_two_type_data(
    const std::vector<double>& pos,
    const SKTBModel& sktb
) {
    const auto prepared = prepare_sktb_atomic_data(sktb);

    const size_t n_atom = 2;
    const size_t N_atom = 2;
    const size_t atom_offset = 0;
    const size_t n_edge = 1;
    const size_t N_edge = 1;

    const int atom_index[] = {0, 1};
    const int atom_type[] = {0, 1};
    const int edge_index[] = {0, 1};
    const int edge_shift_vec[] = {0, 0, 0};
    const double cell[] = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };

    return std::make_unique<AtomicData>(
        n_atom, N_atom, atom_offset, n_edge, N_edge,
        atom_index, atom_type, edge_index, prepared.type_norb.data(), edge_shift_vec,
        cell, pos.data(), MPI_COMM_WORLD);
}

template <typename T>
std::vector<T> block_values(const BlockSpMat<T>& matrix, int local_row, int local_col) {
    const int start = matrix.row_ptr()[local_row];
    const int end = matrix.row_ptr()[local_row + 1];
    for (int k = start; k < end; ++k) {
        if (matrix.col_ind()[k] == local_col) {
            const int nrow = matrix.graph->block_sizes[local_row];
            const int ncol = matrix.graph->block_sizes[local_col];
            const T* ptr = matrix.block_data(k);
            return std::vector<T>(ptr, ptr + nrow * ncol);
        }
    }
    return {};
}

std::vector<std::string> orbital_labels(const SKAtomType& atom_type) {
    std::vector<std::string> labels;
    labels.reserve(atom_type.orbitals.size());
    for (const auto& orbital : atom_type.orbitals) {
        labels.push_back(orbital.label);
    }
    return labels;
}

std::array<double, 3> normalized_direction(const std::array<double, 3>& value) {
    const double r = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    return {value[0] / r, value[1] / r, value[2] / r};
}

std::vector<double> p_sigma_coefficients(const std::array<double, 3>& n) {
    return {n[1], n[2], n[0]}; // py, pz, px
}

std::vector<double> d_sigma_coefficients(const std::array<double, 3>& n) {
    const double nx = n[0];
    const double ny = n[1];
    const double nz = n[2];
    return {
        std::sqrt(3.0) * nx * ny,
        std::sqrt(3.0) * ny * nz,
        0.5 * (3.0 * nz * nz - 1.0),
        std::sqrt(3.0) * nz * nx,
        0.5 * std::sqrt(3.0) * (nx * nx - ny * ny)
    };
}

std::vector<double> pp_ground_truth(
    const std::array<double, 3>& n,
    double sigma,
    double pi
) {
    const auto p = p_sigma_coefficients(n);
    std::vector<double> block(9, 0.0);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            block[static_cast<size_t>(row) * 3 + static_cast<size_t>(col)] =
                (row == col ? pi : 0.0) + (sigma - pi) * p[row] * p[col];
        }
    }
    return block;
}

std::vector<double> outer_product(
    const std::vector<double>& row,
    const std::vector<double>& col,
    double scale
) {
    std::vector<double> block(row.size() * col.size(), 0.0);
    for (size_t i = 0; i < row.size(); ++i) {
        for (size_t j = 0; j < col.size(); ++j) {
            block[i * col.size() + j] = scale * row[i] * col[j];
        }
    }
    return block;
}

std::vector<double> transpose_block(const std::vector<double>& block, size_t nrow, size_t ncol) {
    std::vector<double> result(nrow * ncol, 0.0);
    for (size_t row = 0; row < nrow; ++row) {
        for (size_t col = 0; col < ncol; ++col) {
            result[col * nrow + row] = block[row * ncol + col];
        }
    }
    return result;
}

std::vector<double> block_storage_from_row_major(
    const std::vector<double>& row_major,
    size_t nrow,
    size_t ncol
) {
    std::vector<double> result(nrow * ncol, 0.0);
    for (size_t row = 0; row < nrow; ++row) {
        for (size_t col = 0; col < ncol; ++col) {
            result[col * nrow + row] = row_major[row * ncol + col];
        }
    }
    return result;
}

void expect_real_block(
    const std::vector<std::complex<double>>& block,
    const std::vector<double>& expected,
    double tol = 1e-12
) {
    ASSERT_EQ(block.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(block[i].real(), expected[i], tol) << "flat index " << i;
        EXPECT_NEAR(block[i].imag(), 0.0, tol) << "flat index " << i;
    }
}

void expect_real_diagonal_block(
    const std::vector<std::complex<double>>& block,
    const std::vector<double>& diagonal
) {
    const size_t n = diagonal.size();
    ASSERT_EQ(block.size(), n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double expected = (i == j) ? diagonal[i] : 0.0;
            EXPECT_NEAR(block[i * n + j].real(), expected, 1e-12);
            EXPECT_NEAR(block[i * n + j].imag(), 0.0, 1e-12);
        }
    }
}

SKTBModel make_carbon_p_model(
    double pp_sigma,
    double pp_pi,
    bool with_overlap = false,
    double cutoff = 2.5
) {
    SKTBInput source;
    source.basis["C"] = {"p"};
    source.set_onsite("H", "C", "p", 0.0);
    source.set_pair_channel("H", "C", "C", "pp", "sigma", constant_spline(pp_sigma, cutoff));
    source.set_pair_channel("H", "C", "C", "pp", "pi", constant_spline(pp_pi, cutoff));
    if (with_overlap) {
        source.set_pair_channel("S", "C", "C", "pp", "sigma", constant_spline(pp_sigma, cutoff));
        source.set_pair_channel("S", "C", "C", "pp", "pi", constant_spline(pp_pi, cutoff));
    }
    return build_sktb_model(source, {"C"});
}

} // namespace

TEST(SKTBTest, BasisExpansionAndAtomicPreparation) {
    SKTBInput config;
    config.basis["C"] = {"pz"};
    config.basis["Si"] = {"s", "p", "d"};
    config.hamiltonian.onsite["C"] = {{"pz", 0.0}};
    config.hamiltonian.onsite["Si"] = {{"s", -8.0}, {"p", 0.0}, {"d", 6.0}};

    SKPairInput pair;
    pair.species = {"C", "Si"};
    pair.channels["sp_sigma"] = constant_spline(-2.0, 3.7);
    config.hamiltonian.pairs.push_back(pair);

    const auto model = build_sktb_model(config, {"C", "Si"});
    const std::vector<std::string> expected_c_basis = {"pz"};
    const std::vector<std::string> expected_si_basis = {
        "s", "py", "pz", "px", "dxy", "dyz", "dz2", "dzx", "dx2-y2"
    };
    ASSERT_EQ(orbital_labels(model.atom_types[0]), expected_c_basis);
    ASSERT_EQ(orbital_labels(model.atom_types[1]), expected_si_basis);

    const auto prepared = prepare_sktb_atomic_data_from_atomic_numbers({14, 6, 14}, config);
    ASSERT_EQ(prepared.type_symbols.size(), 2u);
    EXPECT_EQ(prepared.type_symbols[0], "C");
    EXPECT_EQ(prepared.type_symbols[1], "Si");
    ASSERT_EQ(prepared.type_norb.size(), 2u);
    EXPECT_EQ(prepared.type_norb[0], 1);
    EXPECT_EQ(prepared.type_norb[1], 9);
    ASSERT_EQ(prepared.r_max_per_type.size(), 2u);
    EXPECT_DOUBLE_EQ(prepared.r_max_per_type[0], 3.7);
    EXPECT_DOUBLE_EQ(prepared.r_max_per_type[1], 3.7);
}

TEST(SKTBTest, DuplicateBasisLabelsAreRejected) {
    SKTBInput config;
    config.basis["C"] = {"p", "pz"};
    config.hamiltonian.onsite["C"] = {{"p", 0.0}};

    EXPECT_THROW(
        prepare_sktb_atomic_data_from_atomic_numbers({6}, config),
        std::runtime_error);
}

TEST(SKTBTest, RadialFunctionsEvaluateSplineAndExpPoly) {
    const auto spline_constant = constant_spline(2.0, 3.0);
    EXPECT_DOUBLE_EQ(spline_constant.evaluate(0.0), 2.0);
    EXPECT_DOUBLE_EQ(spline_constant.evaluate(1.5), 2.0);
    EXPECT_DOUBLE_EQ(spline_constant.evaluate(3.5), 0.0);

    SKRadialFunction spline;
    spline.kind = "spline";
    spline.r = {0.0, 1.0, 2.0};
    spline.values = {0.0, 1.0, 0.0};
    spline.r_cut = 2.0;
    spline.validate("spline");
    spline.spline_second_derivatives = spline.build_spline_second_derivatives();
    EXPECT_DOUBLE_EQ(spline.evaluate(0.0), 0.0);
    EXPECT_NEAR(spline.evaluate(1.0), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(spline.evaluate(2.5), 0.0);

    const auto exp_poly = exp_poly_constant(4.0, 2.0, 4.0);
    EXPECT_DOUBLE_EQ(exp_poly.evaluate(1.0), 4.0);
    EXPECT_NEAR(exp_poly.evaluate(3.0), 2.0, 1e-12);
    EXPECT_DOUBLE_EQ(exp_poly.evaluate(4.1), 0.0);
}

TEST(SKTBTest, TableKindIsRejectedInFavorOfSpline) {
    SKRadialFunction legacy = constant_spline(1.0, 2.0);
    legacy.kind = "table";

    try {
        legacy.validate("legacy");
        FAIL() << "Expected table kind to be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("use 'spline'"), std::string::npos);
    }
}

TEST(SKTBTest, PPEvaluatorBuildsExpectedBlockAlongBondDirection) {
    auto model = make_carbon_p_model(5.0, 1.0);
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         1.0, 0.0, 0.0},
        {6, 6},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto block = block_values(*gamma, 0, 1);
    expect_real_diagonal_block(block, {1.0, 1.0, 5.0});
}

TEST(SKTBTest, PPEvaluatorFollowsMagneticOrderForAxisDirections) {
    auto model = make_carbon_p_model(5.0, 1.0);
    const std::vector<std::vector<double>> positions = {
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };
    const std::vector<std::vector<double>> expected = {
        {1.0, 1.0, 5.0},
        {5.0, 1.0, 1.0},
        {1.0, 5.0, 1.0}
    };

    for (size_t axis = 0; axis < positions.size(); ++axis) {
        auto atomic_data = make_points_data(positions[axis], {6, 6}, model);
        SKTBConstruction construction(model);
        auto images = construction.build_hamiltonian_images(*atomic_data);
        std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
            images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

        const auto block = block_values(*gamma, 0, 1);
        expect_real_diagonal_block(block, expected[axis]);
    }
}

TEST(SKTBTest, PPRotationMatchesDirectionCosineFormulaForSkewBond) {
    const double sigma = 5.0;
    const double pi = 1.0;
    auto model = make_carbon_p_model(sigma, pi, false, 10.0);
    const std::array<double, 3> displacement = {1.0, 2.0, 3.0};
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         displacement[0], displacement[1], displacement[2]},
        {6, 6},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto block = block_values(*gamma, 0, 1);
    const auto expected = pp_ground_truth(normalized_direction(displacement), sigma, pi);
    expect_real_block(block, expected, 1e-12);
}

TEST(SKTBTest, DDEvaluatorBuildsExpectedDiagonalChannelsAlongZ) {
    SKTBInput config;
    config.basis["Si"] = {"d"};
    config.hamiltonian.onsite["Si"] = {{"d", 0.0}};
    SKPairInput pair;
    pair.species = {"Si", "Si"};
    pair.channels["dd_sigma"] = constant_spline(6.0);
    pair.channels["dd_pi"] = constant_spline(4.0);
    pair.channels["dd_delta"] = constant_spline(2.0);
    config.hamiltonian.pairs.push_back(pair);

    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         0.0, 0.0, 1.5},
        {14, 14},
        config);

    const auto model = build_sktb_model(config, {"Si"}, atomic_data->type_norb);
    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto block = block_values(*gamma, 0, 1);
    expect_real_diagonal_block(block, {2.0, 4.0, 6.0, 4.0, 2.0});
}

TEST(SKTBTest, DDSigmaRotationMatchesQuadrupoleOuterProductForSkewBond) {
    const double sigma = 6.0;
    SKTBInput config;
    config.basis["Si"] = {"d"};
    config.hamiltonian.onsite["Si"] = {{"d", 0.0}};
    config.set_pair_channel("H", "Si", "Si", "dd", "sigma", constant_spline(sigma, 10.0));

    const auto model = build_sktb_model(config, {"Si"});
    const std::array<double, 3> displacement = {2.0, -1.0, 3.0};
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         displacement[0], displacement[1], displacement[2]},
        {14, 14},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto d = d_sigma_coefficients(normalized_direction(displacement));
    const auto expected = outer_product(d, d, sigma);
    const auto block = block_values(*gamma, 0, 1);
    expect_real_block(block, expected, 1e-12);
}

TEST(SKTBTest, HeteronuclearSPCouplingIsTransposeConsistent) {
    SKTBInput config;
    config.basis["H"] = {"s"};
    config.basis["C"] = {"pz"};
    config.hamiltonian.onsite["H"] = {{"s", 0.0}};
    config.hamiltonian.onsite["C"] = {{"pz", 0.0}};
    SKPairInput pair;
    pair.species = {"H", "C"};
    pair.channels["sp_sigma"] = constant_spline(3.0);
    config.hamiltonian.pairs.push_back(pair);

    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {1, 6},
        config);

    const auto model = build_sktb_model(config, {"H", "C"}, atomic_data->type_norb);
    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto hc = block_values(*gamma, 0, 1);
    const auto ch = block_values(*gamma, 1, 0);
    ASSERT_EQ(hc.size(), 1u);
    ASSERT_EQ(ch.size(), 1u);
    EXPECT_NEAR(hc[0].real(), 3.0, 1e-12);
    EXPECT_NEAR(ch[0].real(), 3.0, 1e-12);
}

TEST(SKTBTest, SDEvaluatorPlacesSigmaOnDz2InMagneticOrder) {
    SKTBInput config;
    config.basis["H"] = {"s"};
    config.basis["Si"] = {"d"};
    config.hamiltonian.onsite["H"] = {{"s", 0.0}};
    config.hamiltonian.onsite["Si"] = {{"d", 0.0}};
    SKPairInput pair;
    pair.species = {"H", "Si"};
    pair.channels["sd_sigma"] = constant_spline(7.0);
    config.hamiltonian.pairs.push_back(pair);

    const auto model = build_sktb_model(config, {"H", "Si"});
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {1, 14},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto hs = block_values(*gamma, 0, 1);
    ASSERT_EQ(hs.size(), 5u);
    EXPECT_NEAR(hs[0].real(), 0.0, 1e-12);
    EXPECT_NEAR(hs[1].real(), 0.0, 1e-12);
    EXPECT_NEAR(hs[2].real(), 7.0, 1e-12);
    EXPECT_NEAR(hs[3].real(), 0.0, 1e-12);
    EXPECT_NEAR(hs[4].real(), 0.0, 1e-12);
}

TEST(SKTBTest, PDEvaluatorBuildsAlignedZBlockInMagneticOrder) {
    SKTBInput config;
    config.basis["C"] = {"p"};
    config.basis["Si"] = {"d"};
    config.hamiltonian.onsite["C"] = {{"p", 0.0}};
    config.hamiltonian.onsite["Si"] = {{"d", 0.0}};
    SKPairInput pair;
    pair.species = {"C", "Si"};
    pair.channels["pd_sigma"] = constant_spline(6.0);
    pair.channels["pd_pi"] = constant_spline(4.0);
    config.hamiltonian.pairs.push_back(pair);

    const auto model = build_sktb_model(config, {"C", "Si"});
    auto atomic_data = make_directed_two_type_data(
        {0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto block = block_values(*gamma, 0, 1);
    ASSERT_EQ(block.size(), 15u);
    for (size_t i = 0; i < block.size(); ++i) {
        double expected = 0.0;
        if (i == 3) expected = 4.0;       // py-dzx in the directed edge frame
        if (i == 7) expected = 6.0;       // pz-dz2
        if (i == 11) expected = 4.0;      // px-dyz in the directed edge frame
        EXPECT_NEAR(block[i].real(), expected, 1e-12) << "flat index " << i;
        EXPECT_NEAR(block[i].imag(), 0.0, 1e-12) << "flat index " << i;
    }
}

TEST(SKTBTest, PDSigmaRotationMatchesOuterProductForSkewBond) {
    const double sigma = 6.0;
    SKTBInput config;
    config.basis["C"] = {"p"};
    config.basis["Si"] = {"d"};
    config.hamiltonian.onsite["C"] = {{"p", 0.0}};
    config.hamiltonian.onsite["Si"] = {{"d", 0.0}};
    config.set_pair_channel("H", "C", "Si", "pd", "sigma", constant_spline(sigma, 10.0));

    const auto model = build_sktb_model(config, {"C", "Si"});
    const std::array<double, 3> displacement = {1.0, -2.0, 2.5};
    auto atomic_data = make_directed_two_type_data(
        {0.0, 0.0, 0.0,
         displacement[0], displacement[1], displacement[2]},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto n = normalized_direction(displacement);
    const auto expected = outer_product(p_sigma_coefficients(n), d_sigma_coefficients(n), sigma);
    const auto block = block_values(*gamma, 0, 1);
    expect_real_block(block, block_storage_from_row_major(expected, 3, 5), 1e-12);
}

TEST(SKTBTest, ReverseLookupTransposesPDSkewBondBlock) {
    const double sigma = 6.0;
    SKTBInput config;
    config.basis["C"] = {"p"};
    config.basis["Si"] = {"d"};
    config.hamiltonian.onsite["C"] = {{"p", 0.0}};
    config.hamiltonian.onsite["Si"] = {{"d", 0.0}};
    config.set_pair_channel("H", "C", "Si", "pd", "sigma", constant_spline(sigma, 10.0));

    const auto model = build_sktb_model(config, {"C", "Si"});
    const std::array<double, 3> displacement = {1.0, -2.0, 2.5};
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         displacement[0], displacement[1], displacement[2]},
        {6, 14},
        model);

    SKTBConstruction construction(model);
    auto images = construction.build_hamiltonian_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> gamma(
        images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto n = normalized_direction(displacement);
    const auto direct_expected = outer_product(p_sigma_coefficients(n), d_sigma_coefficients(n), sigma);
    const auto reverse_expected = transpose_block(direct_expected, 3, 5);
    expect_real_block(block_values(*gamma, 0, 1), block_storage_from_row_major(direct_expected, 3, 5), 1e-12);
    expect_real_block(block_values(*gamma, 1, 0), block_storage_from_row_major(reverse_expected, 5, 3), 1e-12);
}

TEST(SKTBTest, AlignedPDBlockUsesMagneticMOrderBeforeRotation) {
    auto sigma_block = sktb_detail::build_aligned_shell_block(1, 2, 0, 6.0);
    auto pi_block = sktb_detail::build_aligned_shell_block(1, 2, 1, 4.0);
    ASSERT_EQ(sigma_block.size(), pi_block.size());

    std::vector<double> block(sigma_block.size(), 0.0);
    for (size_t i = 0; i < block.size(); ++i) {
        block[i] = sigma_block[i] + pi_block[i];
    }
    ASSERT_EQ(block.size(), 15u);
    for (size_t i = 0; i < block.size(); ++i) {
        double expected = 0.0;
        if (i == 1) expected = 4.0;       // py-dyz, m = -1
        if (i == 7) expected = 6.0;       // pz-dz2, m = 0
        if (i == 13) expected = 4.0;      // px-dzx, m = +1
        EXPECT_NEAR(block[i], expected, 1e-12) << "flat index " << i;
    }
}

TEST(SKTBTest, ExplicitPzBasisSelectsMagneticMZeroChannel) {
    SKTBInput config;
    config.basis["C"] = {"pz"};
    config.hamiltonian.onsite["C"] = {{"pz", 0.0}};
    SKPairInput pair;
    pair.species = {"C", "C"};
    pair.channels["pp_sigma"] = constant_spline(5.0);
    pair.channels["pp_pi"] = constant_spline(1.0);
    config.hamiltonian.pairs.push_back(pair);

    const auto model = build_sktb_model(config, {"C"});
    SKTBConstruction construction(model);

    auto z_data = make_points_data(
        {0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {6, 6},
        model);
    auto z_images = construction.build_hamiltonian_images(*z_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> z_gamma(
        z_images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));
    const auto z_block = block_values(*z_gamma, 0, 1);
    ASSERT_EQ(z_block.size(), 1u);
    EXPECT_NEAR(z_block[0].real(), 5.0, 1e-12);

    auto x_data = make_points_data(
        {0.0, 0.0, 0.0,
         1.0, 0.0, 0.0},
        {6, 6},
        model);
    auto x_images = construction.build_hamiltonian_images(*x_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> x_gamma(
        x_images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));
    const auto x_block = block_values(*x_gamma, 0, 1);
    ASSERT_EQ(x_block.size(), 1u);
    EXPECT_NEAR(x_block[0].real(), 1.0, 1e-12);
}

TEST(SKTBTest, OverlapUsesSeparateRadialChannelsFromHamiltonian) {
    SKTBInput config;
    config.basis["C"] = {"pz"};
    config.set_onsite("H", "C", "pz", 0.0);
    config.set_pair_channel("H", "C", "C", "pp", "pi", constant_spline(0.20, 3.0));
    config.set_pair_channel("S", "C", "C", "pp", "pi", constant_spline(0.05, 3.0));

    const auto model = build_sktb_model(config, {"C"});
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         1.0, 0.0, 0.0},
        {6, 6},
        model);

    SKTBConstruction construction(model);
    auto h_images = construction.build_hamiltonian_images(*atomic_data);
    auto s_images = construction.build_overlap_images(*atomic_data);
    std::unique_ptr<BlockSpMat<std::complex<double>>> h_gamma(
        h_images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));
    std::unique_ptr<BlockSpMat<std::complex<double>>> s_gamma(
        s_images->sample_k({0.0, 0.0, 0.0}, PhaseConvention::R_AND_POSITION));

    const auto h_block = block_values(*h_gamma, 0, 1);
    const auto s_block = block_values(*s_gamma, 0, 1);
    ASSERT_EQ(h_block.size(), 1u);
    ASSERT_EQ(s_block.size(), 1u);
    EXPECT_NEAR(h_block[0].real(), 0.20, 1e-12);
    EXPECT_NEAR(s_block[0].real(), 0.05, 1e-12);
}

TEST(SKTBTest, OrthogonalProviderBuildsVelocityAndFallbackPosition) {
    auto native_model = make_carbon_p_model(2.0, 0.5);
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         1.0, 0.0, 0.0},
        {6, 6},
        native_model);

    rmodel::SKTBModelConfig model;
    model.native = native_model;
    rmodel::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.preconditioner.mode = "None";
    options.compute_bounds = false;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    auto operators = build_operator_set(*atomic_data, MPI_COMM_WORLD, false, rmodel::AnyModelConfig(model), options, {"C"}, runtime);
    ASSERT_TRUE(operators->has_hamiltonian());
    ASSERT_FALSE(operators->has_overlap());
    ASSERT_TRUE(operators->has_velocity());
    ASSERT_TRUE(operators->has_position());

    const auto& velocity = operators->velocity_state(0);
    EXPECT_GT(velocity.x->local_block_nnz(), 0u);
    EXPECT_GT(velocity.y->local_block_nnz(), 0u);
    EXPECT_GT(velocity.z->local_block_nnz(), 0u);
    const auto vx_block = block_values(*velocity.x, 0, 1);
    ASSERT_EQ(vx_block.size(), 9u);
    EXPECT_NEAR(vx_block[0].real(), 0.0, 1e-12);
    EXPECT_NEAR(vx_block[4].real(), 0.0, 1e-12);
    EXPECT_NEAR(vx_block[8].real(), 0.0, 1e-12);
    EXPECT_NEAR(vx_block[0].imag(), 0.5 / 0.6582119569, 1e-12);
    EXPECT_NEAR(vx_block[4].imag(), 0.5 / 0.6582119569, 1e-12);
    EXPECT_NEAR(vx_block[8].imag(), 2.0 / 0.6582119569, 1e-12);
    const auto vy_block = block_values(*velocity.y, 0, 1);
    const auto vz_block = block_values(*velocity.z, 0, 1);
    ASSERT_EQ(vy_block.size(), 9u);
    ASSERT_EQ(vz_block.size(), 9u);
    for (const auto& value : vy_block) {
        EXPECT_NEAR(std::abs(value), 0.0, 1e-12);
    }
    for (const auto& value : vz_block) {
        EXPECT_NEAR(std::abs(value), 0.0, 1e-12);
    }

    const auto& position = operators->position_state(0);
    ASSERT_TRUE(position.x.orthogonal_diagonal);
    EXPECT_FALSE(position.x.matrix);
}

TEST(SKTBTest, OverlapPresentProviderBuildsOverlapWithoutVelocity) {
    auto native_model = make_carbon_p_model(2.0, 0.5, true);
    auto atomic_data = make_points_data(
        {0.0, 0.0, 0.0,
         1.0, 0.0, 0.0},
        {6, 6},
        native_model);

    rmodel::SKTBModelConfig model;
    model.native = native_model;
    rmodel::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.preconditioner.mode = "None";
    options.compute_bounds = false;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    auto operators = build_operator_set(*atomic_data, MPI_COMM_WORLD, false, rmodel::AnyModelConfig(model), options, {"C"}, runtime);
    ASSERT_TRUE(operators->has_hamiltonian());
    ASSERT_TRUE(operators->has_overlap());
    ASSERT_FALSE(operators->has_velocity());
    ASSERT_FALSE(operators->has_position());
    ASSERT_NE(operators->overlap_k(0), nullptr);
}

TEST(SKTBTest, PeriodicOrthogonalModelBuildsFirstLatticeMoment) {
    SKTBInput config;
    config.basis["C"] = {"pz"};
    config.hamiltonian.onsite["C"] = {{"pz", 0.0}};
    SKPairInput pair;
    pair.species = {"C", "C"};
    pair.channels["pp_pi"] = constant_spline(0.2, 3.5);
    config.hamiltonian.pairs.push_back(pair);

    const auto native_model = build_sktb_model(config, {"C"});
    auto atomic_data = make_periodic_single_atom_data(3.0, native_model);

    rmodel::SKTBModelConfig model;
    model.native = native_model;
    rmodel::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.preconditioner.mode = "None";
    options.compute_bounds = false;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    auto operators = build_operator_set(*atomic_data, MPI_COMM_WORLD, false, rmodel::AnyModelConfig(model), options, {"C"}, runtime);
    const auto& first_lattice_moment = operators->first_lattice_moment_state(0);
    ASSERT_TRUE(first_lattice_moment.x.first_moment);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
