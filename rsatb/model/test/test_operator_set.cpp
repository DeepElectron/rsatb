#include "../operator_set.hpp"

#include <gtest/gtest.h>
#include <H5Cpp.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace rsatb::backend;

namespace rmodel = rsatb::model;

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

std::unique_ptr<AtomicData> make_periodic_single_atom_data(double cell_x, int norb) {
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
        cell_x, 0.0, 0.0,
        0.0, 8.0, 0.0,
        0.0, 0.0, 8.0
    };
    const double pos[] = {0.25, -0.5, 0.75};

    return std::make_unique<AtomicData>(
        n_atom, N_atom, atom_offset, n_edge, N_edge,
        atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
        cell, pos, MPI_COMM_WORLD);
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
    EXPECT_GT(matrix.local_block_nnz(), 0u);
    return matrix.block_data(0)[0];
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

std::unique_ptr<ImageContainer<double>> make_periodic_scalar_h_images(
    AtomicData& atomic_data,
    double onsite,
    double positive_hop,
    double negative_hop
) {
    auto images = std::make_unique<ImageContainer<double>>(&atomic_data);
    std::vector<int> r0 = {0, 0, 0};
    std::vector<int> rp = {1, 0, 0};
    std::vector<int> rm = {-1, 0, 0};
    images->add_block(r0, 0, 0, &onsite, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    images->add_block(rp, 0, 0, &positive_hop, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    images->add_block(rm, 0, 0, &negative_hop, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    images->assemble();
    return images;
}

std::unique_ptr<ImageContainer<double>> make_periodic_scalar_h_images(AtomicData& atomic_data, double onsite, double hop) {
    return make_periodic_scalar_h_images(atomic_data, onsite, hop, hop);
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

rmodel::SetupOptions make_options(bool compute_bounds = false) {
    rmodel::SetupOptions options;
    options.kpoints = {{0.0, 0.0, 0.0}};
    options.preconditioner.mode = "None";
    options.compute_bounds = compute_bounds;
    return options;
}

OperatorRuntimeContext make_runtime() {
    OperatorRuntimeContext runtime;
    runtime.npar = 1;
    runtime.world_rank = 0;
    runtime.world_size = 1;
    return runtime;
}

} // namespace

TEST(OperatorSetTest, Hdf5ModelBuildsHamiltonianAndBounds) {
    auto atomic_data = make_single_atom_data();
    const std::string path = make_hdf5_hamiltonian_file(std::complex<double>(4.0, 0.0));

    rmodel::LCAOModelConfig model;
    model.input = rmodel::Hdf5Input{path};
    auto operators = build_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        rmodel::AnyModelConfig(model),
        make_options(true),
        {},
        make_runtime());

    ASSERT_TRUE(operators->has_hamiltonian());
    ASSERT_FALSE(operators->has_overlap());
    ASSERT_TRUE(operators->has_velocity());
    ASSERT_TRUE(operators->has_bounds());
    ASSERT_EQ(operators->global_nk(), 1);

    auto* hk = operators->hamiltonian_k(0);
    ASSERT_NE(hk, nullptr);
    EXPECT_EQ(first_block_value(*hk), std::complex<double>(4.0, 0.0));
    EXPECT_NEAR(operators->emin(), 3.5, 1e-8);
    EXPECT_NEAR(operators->emax(), 4.5, 1e-8);

    std::remove(path.c_str());
}

TEST(OperatorSetTest, OrthogonalVelocityFallbackUsesHamiltonianImageDisplacements) {
    auto atomic_data = make_periodic_single_atom_data(3.0, 1);

    ModelArtifacts artifacts;
    artifacts.capabilities.hamiltonian = true;
    artifacts.hamiltonian_images = make_periodic_scalar_h_images(*atomic_data, 0.2, 0.75, 0.25);

    auto operators = finalize_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        std::move(artifacts),
        make_options(false),
        make_runtime());
    ASSERT_TRUE(operators->has_velocity());

    const auto& velocity = operators->velocity_state(0);
    ASSERT_GT(velocity.x->local_block_nnz(), 0u);
    EXPECT_GT(velocity.y->local_block_nnz(), 0u);
    EXPECT_GT(velocity.z->local_block_nnz(), 0u);

    const std::complex<double> expected_vx(0.0, 3.0 * (0.75 - 0.25) / 0.6582119569);
    EXPECT_NEAR(first_block_value(*velocity.x).real(), expected_vx.real(), 1e-10);
    EXPECT_NEAR(first_block_value(*velocity.x).imag(), expected_vx.imag(), 1e-10);
    EXPECT_NEAR(std::abs(first_block_value(*velocity.y)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(first_block_value(*velocity.z)), 0.0, 1e-12);
}

TEST(OperatorSetTest, ManualArtifactsBuildCachedVelocityLazily) {
    auto atomic_data = make_single_atom_data();
    int velocity_calls = 0;

    ModelArtifacts artifacts;
    artifacts.capabilities.hamiltonian = true;
    artifacts.capabilities.overlap = true;
    artifacts.capabilities.velocity = true;
    artifacts.hamiltonian_images = make_scalar_h_images(*atomic_data, 2.5);
    artifacts.overlap_images = make_scalar_s_images(*atomic_data, 1.5);
    artifacts.velocity_builder = std::make_unique<CountingVelocityBuilder>(&velocity_calls);

    auto operators = finalize_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        std::move(artifacts),
        make_options(true),
        make_runtime());

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

    EXPECT_FALSE(operators->has_position());
}

TEST(OperatorSetTest, FirstLatticeMomentStateBuildsAndCachesAxisData) {
    auto atomic_data = make_periodic_single_atom_data(3.0, 1);

    ModelArtifacts artifacts;
    artifacts.capabilities.hamiltonian = true;
    artifacts.hamiltonian_images = make_periodic_scalar_h_images(*atomic_data, 0.2, 0.5);

    auto operators = finalize_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        std::move(artifacts),
        make_options(true),
        make_runtime());
    const auto& moment_first = operators->first_lattice_moment_state(0);
    const auto& moment_second = operators->first_lattice_moment_state(0);
    EXPECT_EQ(&moment_first, &moment_second);

    ASSERT_TRUE(moment_first.x.first_moment);
    EXPECT_FALSE(moment_first.y.first_moment);
    EXPECT_FALSE(moment_first.z.first_moment);
    EXPECT_EQ(first_block_value(*moment_first.x.first_moment), std::complex<double>(0.0, 0.0));

    const auto& position = operators->position_state(0);
    ASSERT_TRUE(position.x.orthogonal_diagonal);
    ASSERT_TRUE(position.y.orthogonal_diagonal);
    ASSERT_TRUE(position.z.orthogonal_diagonal);
    EXPECT_FALSE(position.x.matrix);
    EXPECT_FALSE(position.y.matrix);
    EXPECT_FALSE(position.z.matrix);
    EXPECT_EQ((*position.x.orthogonal_diagonal)[0], std::complex<double>(0.25, 0.0));
    EXPECT_EQ((*position.y.orthogonal_diagonal)[0], std::complex<double>(-0.5, 0.0));
    EXPECT_EQ((*position.z.orthogonal_diagonal)[0], std::complex<double>(0.75, 0.0));
}

TEST(OperatorSetTest, FirstLatticeMomentUsesImageShiftWeights) {
    auto atomic_data = make_periodic_single_atom_data(3.0, 1);

    ModelArtifacts artifacts;
    artifacts.capabilities.hamiltonian = true;
    artifacts.hamiltonian_images = make_periodic_scalar_h_images(*atomic_data, 0.2, 0.75, 0.25);

    auto operators = finalize_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        std::move(artifacts),
        make_options(false),
        make_runtime());
    const auto& first_lattice_moment = operators->first_lattice_moment_state(0);

    ASSERT_TRUE(first_lattice_moment.x.first_moment);
    EXPECT_EQ(first_block_value(*first_lattice_moment.x.first_moment), std::complex<double>(1.5, 0.0));
}

TEST(OperatorSetTest, DiagonalFallbackPositionUsesAtomicCoordinatesForAllOrbitals) {
    auto atomic_data = make_single_atom_data({1.25, -0.5, 0.75}, 2);

    ModelArtifacts artifacts;
    artifacts.capabilities.hamiltonian = true;
    artifacts.hamiltonian_images = make_diagonal_h_images(*atomic_data, {2.0, 3.0});

    auto operators = finalize_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        std::move(artifacts),
        make_options(false),
        make_runtime());

    ASSERT_TRUE(operators->has_position());
    const auto& position = operators->position_state(0);

    EXPECT_FALSE(position.x.matrix);
    EXPECT_FALSE(position.y.matrix);
    EXPECT_FALSE(position.z.matrix);
    ASSERT_TRUE(position.x.orthogonal_diagonal);
    ASSERT_TRUE(position.y.orthogonal_diagonal);
    ASSERT_TRUE(position.z.orthogonal_diagonal);
    EXPECT_EQ((*position.x.orthogonal_diagonal)[0], std::complex<double>(1.25, 0.0));
    EXPECT_EQ((*position.x.orthogonal_diagonal)[1], std::complex<double>(1.25, 0.0));
    EXPECT_EQ((*position.y.orthogonal_diagonal)[0], std::complex<double>(-0.5, 0.0));
    EXPECT_EQ((*position.y.orthogonal_diagonal)[1], std::complex<double>(-0.5, 0.0));
    EXPECT_EQ((*position.z.orthogonal_diagonal)[0], std::complex<double>(0.75, 0.0));
    EXPECT_EQ((*position.z.orthogonal_diagonal)[1], std::complex<double>(0.75, 0.0));
}

TEST(OperatorSetTest, LcaoModelBuildsOverlapVelocityAndPositionCapabilities) {
    auto atomic_data = std::unique_ptr<AtomicData>(
        AtomicData::from_file(
            "/home/zhanghao/softwares/rsatb/test/data/diamond/diamond_sc111.vasp",
            std::vector<double>{3.7},
            std::vector<int>{13},
            MPI_COMM_WORLD));

    rmodel::LCAOModelConfig model;
    model.input = rmodel::Hdf5Input{"/home/zhanghao/softwares/rsatb/test/data/diamond/C2.0/hamiltonians.h5"};
    model.integrals.orb_dir = "/home/zhanghao/softwares/rsatb/test/data/diamond/";
    model.integrals.orb_names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    model.integrals.psd_dir = "/home/zhanghao/softwares/rsatb/test/data/diamond/";
    model.integrals.psd_names = {"C_ONCV_PBE-1.0.upf"};

    auto operators = build_operator_set(
        *atomic_data,
        MPI_COMM_WORLD,
        false,
        rmodel::AnyModelConfig(model),
        make_options(false),
        {},
        make_runtime());

    ASSERT_TRUE(operators->has_overlap());
    ASSERT_TRUE(operators->has_velocity());
    ASSERT_TRUE(operators->has_position());
    ASSERT_NE(operators->overlap_k(0), nullptr);

    const auto& velocity = operators->velocity_state(0);
    EXPECT_GT(velocity.x->local_block_nnz(), 0u);
    EXPECT_GT(velocity.y->local_block_nnz(), 0u);
    EXPECT_GT(velocity.z->local_block_nnz(), 0u);

    const auto& position = operators->position_state(0);
    ASSERT_TRUE(position.x.matrix);
    ASSERT_TRUE(position.y.matrix);
    ASSERT_TRUE(position.z.matrix);
    ASSERT_TRUE(position.x.orthogonal_diagonal);
    ASSERT_TRUE(position.y.orthogonal_diagonal);
    ASSERT_TRUE(position.z.orthogonal_diagonal);
    EXPECT_GT(position.x.matrix->local_block_nnz(), 0u);
    EXPECT_GT(position.y.matrix->local_block_nnz(), 0u);
    EXPECT_GT(position.z.matrix->local_block_nnz(), 0u);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
