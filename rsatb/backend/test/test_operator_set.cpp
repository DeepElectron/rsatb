#include "../operator_set.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace rsatb::backend;

namespace {

std::unique_ptr<AtomicData> make_single_atom_data() {
    const std::vector<double> pos = {0.0, 0.0, 0.0};
    const std::vector<int> z = {1};
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const std::vector<double> r_max = {1.0};
    const std::vector<int> type_norb = {1};
    return std::unique_ptr<AtomicData>(AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD));
}

std::unique_ptr<AtomicData> make_single_atom_data(const std::vector<double>& pos, int norb) {
    const std::vector<int> z = {1};
    const std::vector<double> cell = {
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
        0.0, 0.0, 10.0
    };
    const std::vector<bool> pbc = {false, false, false};
    const std::vector<double> r_max = {1.0};
    const std::vector<int> type_norb = {norb};
    return std::unique_ptr<AtomicData>(AtomicData::from_points(pos, z, cell, pbc, r_max, type_norb, MPI_COMM_WORLD));
}

std::string make_hdf5_hamiltonian_file(const std::complex<double>& value) {
    const std::string path = "/tmp/rsatb_operator_set_test.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("0");

    const hsize_t dims[3] = {1, 1, 2};
    H5::DataSpace space(3, dims);
    H5::DataSet dataset = group.createDataSet("0_0_0_0_0", H5::PredType::NATIVE_DOUBLE, space);
    const double raw[2] = {value.real(), value.imag()};
    dataset.write(raw, H5::PredType::NATIVE_DOUBLE);
    return path;
}

template <typename T>
T first_block_value(const BlockSpMat<T>& matrix) {
    EXPECT_FALSE(matrix.blk_handles.empty());
    return matrix.arena.get_ptr(matrix.blk_handles[0])[0];
}

std::unique_ptr<ImageContainer<double>> make_scalar_h_images(AtomicData& atomic_data, double value) {
    auto images = std::make_unique<ImageContainer<double>>(&atomic_data);
    std::vector<int> r = {0, 0, 0};
    images->add_block(r, 0, 0, &value, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    images->assemble();
    return images;
}

std::unique_ptr<ImageContainer<double>> make_scalar_s_images(AtomicData& atomic_data, double value) {
    auto images = std::make_unique<ImageContainer<double>>(&atomic_data);
    std::vector<int> r = {0, 0, 0};
    images->add_block(r, 0, 0, &value, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    images->assemble();
    return images;
}

std::unique_ptr<ImageContainer<double>> make_diagonal_h_images(AtomicData& atomic_data, const std::vector<double>& diagonal) {
    auto images = std::make_unique<ImageContainer<double>>(&atomic_data);
    std::vector<double> block(diagonal.size() * diagonal.size(), 0.0);
    for (size_t i = 0; i < diagonal.size(); ++i) {
        block[i * diagonal.size() + i] = diagonal[i];
    }
    std::vector<int> r = {0, 0, 0};
    images->add_block(
        r,
        0,
        0,
        block.data(),
        static_cast<int>(diagonal.size()),
        static_cast<int>(diagonal.size()),
        AssemblyMode::INSERT,
        MatrixLayout::RowMajor);
    images->assemble();
    return images;
}

template <typename T>
std::vector<T> first_block_values(const BlockSpMat<T>& matrix) {
    EXPECT_FALSE(matrix.blk_handles.empty());
    const int local_row = 0;
    const int local_col = 0;
    const int norb = matrix.graph->block_sizes[local_row];
    const T* block = matrix.arena.get_ptr(matrix.blk_handles[0]);
    return std::vector<T>(block, block + norb * norb);
}

class CountingVelocityBuilder : public IVelocityBuilder {
public:
    explicit CountingVelocityBuilder(int* calls) : calls_(calls) {}

    std::unique_ptr<VelocityTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>&
    ) const override {
        ++(*calls_);
        auto triplet = std::make_unique<VelocityTriplet>();
        triplet->x = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
        triplet->y = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
        triplet->z = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());

        const std::complex<double> vx(1.0, 0.0);
        const std::complex<double> vy(2.0, 0.0);
        const std::complex<double> vz(3.0, 0.0);
        triplet->x->add_block(0, 0, &vx, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        triplet->y->add_block(0, 0, &vy, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        triplet->z->add_block(0, 0, &vz, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        triplet->x->assemble();
        triplet->y->assemble();
        triplet->z->assemble();
        return triplet;
    }

private:
    int* calls_;
};

class MockProvider : public IOperatorProvider {
public:
    explicit MockProvider(int* velocity_calls) : velocity_calls_(velocity_calls) {}

    std::string name() const override {
        return "MockProvider";
    }

    ProviderCapabilities capabilities() const override {
        return ProviderCapabilities{true, true, true, false};
    }

    void populate(AtomicData& atomic_data, OperatorSetBuilder& builder) const override {
        builder.set_hamiltonian_images(make_scalar_h_images(atomic_data, 2.5));
        builder.set_overlap_images(make_scalar_s_images(atomic_data, 1.5));
        builder.set_velocity_builder(std::make_unique<CountingVelocityBuilder>(velocity_calls_));
    }

private:
    int* velocity_calls_;
};

} // namespace

TEST(OperatorSetTest, Hdf5ProviderBuildsHamiltonianAndBounds) {
    auto atomic_data = make_single_atom_data();
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(4.0, 0.0));

    OperatorSetupConfig config;
    config.kpoints = {{0.0, 0.0, 0.0}};
    config.sources.hdf5.filename = path;
    config.preconditioner.mode = "None";
    config.compute_bounds = true;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    auto operators = build_operator_set(*atomic_data, MPI_COMM_WORLD, false, config, runtime);

    ASSERT_TRUE(operators->has_hamiltonian());
    ASSERT_FALSE(operators->has_overlap());
    ASSERT_TRUE(operators->has_bounds());
    ASSERT_EQ(operators->global_nk(), 1);

    auto* hk = operators->hamiltonian_k(0);
    ASSERT_NE(hk, nullptr);
    EXPECT_EQ(first_block_value(*hk), std::complex<double>(4.0, 0.0));
    EXPECT_NEAR(operators->emin(), 3.5, 1e-8);
    EXPECT_NEAR(operators->emax(), 4.5, 1e-8);

    std::remove(path.c_str());
}

TEST(OperatorSetTest, MockProviderBuildsCachedVelocityLazily) {
    auto atomic_data = make_single_atom_data();
    int velocity_calls = 0;

    OperatorSetupConfig config;
    config.kpoints = {{0.0, 0.0, 0.0}};
    config.preconditioner.mode = "None";
    config.compute_bounds = true;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    OperatorSetBuilder builder(*atomic_data, MPI_COMM_WORLD, false);
    MockProvider provider(&velocity_calls);
    builder.note_capabilities(provider.capabilities());
    provider.populate(*atomic_data, builder);
    auto operators = builder.build(config, runtime);

    ASSERT_TRUE(operators->has_hamiltonian());
    ASSERT_TRUE(operators->has_overlap());
    ASSERT_TRUE(operators->has_velocity());
    ASSERT_TRUE(operators->has_bounds());

    auto* hk_first = operators->hamiltonian_k(0);
    auto* hk_second = operators->hamiltonian_k(0);
    ASSERT_EQ(hk_first, hk_second);
    ASSERT_NE(hk_first, nullptr);
    EXPECT_EQ(first_block_value(*hk_first), std::complex<double>(2.5, 0.0));

    auto* sk_first = operators->overlap_k(0);
    auto* sk_second = operators->overlap_k(0);
    ASSERT_EQ(sk_first, sk_second);
    ASSERT_NE(sk_first, nullptr);
    EXPECT_EQ(first_block_value(*sk_first), std::complex<double>(1.5, 0.0));

    const auto& velocity_first = operators->velocity_state(0);
    const auto& velocity_second = operators->velocity_state(0);
    EXPECT_EQ(&velocity_first, &velocity_second);
    EXPECT_EQ(velocity_calls, 1);
    EXPECT_EQ(first_block_value(*velocity_first.x), std::complex<double>(1.0, 0.0));
    EXPECT_EQ(first_block_value(*velocity_first.y), std::complex<double>(2.0, 0.0));
    EXPECT_EQ(first_block_value(*velocity_first.z), std::complex<double>(3.0, 0.0));

    ASSERT_TRUE(operators->has_position());
    const auto& position = operators->position_state(0);
    EXPECT_EQ(first_block_value(*position.x), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(first_block_value(*position.y), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(first_block_value(*position.z), std::complex<double>(0.0, 0.0));
}

TEST(OperatorSetTest, DiagonalFallbackPositionUsesAtomicCoordinatesForAllOrbitals) {
    auto atomic_data = make_single_atom_data({1.25, -0.5, 0.75}, 2);

    OperatorSetupConfig config;
    config.kpoints = {{0.0, 0.0, 0.0}};
    config.preconditioner.mode = "None";
    config.compute_bounds = false;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    OperatorSetBuilder builder(*atomic_data, MPI_COMM_WORLD, false);
    builder.set_hamiltonian_images(make_diagonal_h_images(*atomic_data, {2.0, 3.0}));
    auto operators = builder.build(config, runtime);

    ASSERT_TRUE(operators->has_position());
    const auto& position = operators->position_state(0);

    const auto bx = first_block_values(*position.x);
    const auto by = first_block_values(*position.y);
    const auto bz = first_block_values(*position.z);
    ASSERT_EQ(bx.size(), 4u);
    ASSERT_EQ(by.size(), 4u);
    ASSERT_EQ(bz.size(), 4u);

    EXPECT_EQ(bx[0], std::complex<double>(1.25, 0.0));
    EXPECT_EQ(bx[3], std::complex<double>(1.25, 0.0));
    EXPECT_EQ(by[0], std::complex<double>(-0.5, 0.0));
    EXPECT_EQ(by[3], std::complex<double>(-0.5, 0.0));
    EXPECT_EQ(bz[0], std::complex<double>(0.75, 0.0));
    EXPECT_EQ(bz[3], std::complex<double>(0.75, 0.0));
    EXPECT_EQ(bx[1], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(by[1], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(bz[1], std::complex<double>(0.0, 0.0));
}

TEST(OperatorSetTest, AndersonDisorderIsReproducibleAndConsistentPerAtom) {
    auto atomic_data = make_single_atom_data({0.0, 0.0, 0.0}, 2);

    OperatorSetupConfig config;
    config.kpoints = {{0.0, 0.0, 0.0}};
    config.preconditioner.mode = "None";
    config.compute_bounds = false;
    config.disorder.kind = "anderson";
    config.disorder.strength_ev = 1.5;
    config.disorder.seed = 1234;
    config.disorder.per_atom = true;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    OperatorSetBuilder builder_a(*atomic_data, MPI_COMM_WORLD, false);
    builder_a.set_hamiltonian_images(make_diagonal_h_images(*atomic_data, {2.5, 4.0}));
    auto operators_a = builder_a.build(config, runtime);

    OperatorSetBuilder builder_b(*atomic_data, MPI_COMM_WORLD, false);
    builder_b.set_hamiltonian_images(make_diagonal_h_images(*atomic_data, {2.5, 4.0}));
    auto operators_b = builder_b.build(config, runtime);

    OperatorSetupConfig config_diff_seed = config;
    config_diff_seed.disorder.seed = 9876;
    OperatorSetBuilder builder_c(*atomic_data, MPI_COMM_WORLD, false);
    builder_c.set_hamiltonian_images(make_diagonal_h_images(*atomic_data, {2.5, 4.0}));
    auto operators_c = builder_c.build(config_diff_seed, runtime);

    const auto block_a = first_block_values(*operators_a->hamiltonian_k(0));
    const auto block_b = first_block_values(*operators_b->hamiltonian_k(0));
    const auto block_c = first_block_values(*operators_c->hamiltonian_k(0));

    const double shift_a_0 = block_a[0].real() - 2.5;
    const double shift_a_1 = block_a[3].real() - 4.0;
    const double shift_b_0 = block_b[0].real() - 2.5;
    const double shift_b_1 = block_b[3].real() - 4.0;
    const double shift_c_0 = block_c[0].real() - 2.5;

    EXPECT_NEAR(shift_a_0, shift_a_1, 1e-12);
    EXPECT_NEAR(shift_a_0, shift_b_0, 1e-12);
    EXPECT_NEAR(shift_a_1, shift_b_1, 1e-12);
    EXPECT_NE(block_a[0].real(), block_c[0].real());
    EXPECT_EQ(block_a[1], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(block_a[2], std::complex<double>(0.0, 0.0));
}

TEST(OperatorSetTest, Op2cProviderBuildsOverlapAndVelocityCapabilities) {
    auto atomic_data = std::unique_ptr<AtomicData>(
        AtomicData::from_file(
            "/home/zhanghao/softwares/rsatb/test/data/diamond/diamond_sc111.vasp",
            std::vector<double>{3.7},
            std::vector<int>{13},
            MPI_COMM_WORLD));

    OperatorSetupConfig config;
    config.kpoints = {{0.0, 0.0, 0.0}};
    config.sources.op2c.orb_dir = "/home/zhanghao/softwares/rsatb/test/data/diamond/";
    config.sources.op2c.orb_names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    config.sources.op2c.psd_dir = "/home/zhanghao/softwares/rsatb/test/data/diamond/";
    config.sources.op2c.psd_names = {"C_ONCV_PBE-1.0.upf"};
    config.preconditioner.mode = "None";
    config.compute_bounds = false;

    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;

    auto operators = build_operator_set(*atomic_data, MPI_COMM_WORLD, false, config, runtime);

    ASSERT_TRUE(operators->has_overlap());
    ASSERT_TRUE(operators->has_velocity());
    ASSERT_TRUE(operators->has_position());
    ASSERT_NE(operators->overlap_k(0), nullptr);

    const auto& velocity = operators->velocity_state(0);
    EXPECT_FALSE(velocity.x->blk_handles.empty());
    EXPECT_FALSE(velocity.y->blk_handles.empty());
    EXPECT_FALSE(velocity.z->blk_handles.empty());

    const auto& position = operators->position_state(0);
    EXPECT_FALSE(position.x->blk_handles.empty());
    EXPECT_FALSE(position.y->blk_handles.empty());
    EXPECT_FALSE(position.z->blk_handles.empty());
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
