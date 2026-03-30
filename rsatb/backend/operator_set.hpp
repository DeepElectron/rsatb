#ifndef RSATB_BACKEND_OPERATOR_SET_HPP
#define RSATB_BACKEND_OPERATOR_SET_HPP

#include "operator_construction.hpp"
#include "../calc/eigen.hpp"
#include "../calc/preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rsatb {
namespace backend {

struct ProviderCapabilities {
    bool hamiltonian = false;
    bool overlap = false;
    bool velocity = false;
    bool position = false;
};

struct PreconditionerConfig {
    std::string mode = "H/S";
    double tol = 1e-7;
    double sptol = 1e-5;
    int maxiter = 20;
    std::string method = "graph";

    std::string normalized_mode() const {
        if (mode == "H/S") {
            // The historical default meant "keep the generalized H/S form".
            return "S";
        }
        return mode;
    }
};

struct Hdf5HamiltonianSourceConfig {
    std::string filename;
    bool lspinorb = false;
    int nspin = 1;
    bool noncolin = false;

    bool enabled() const {
        return !filename.empty();
    }
};

struct Op2cIntegralSourceConfig {
    std::vector<std::string> orb_names;
    std::string orb_dir;
    std::vector<std::string> psd_names;
    std::string psd_dir;
    int nspin = 1;
    bool lspinorb = false;
    bool noncolin = false;
    
    std::string log_file = "op2c.log";

    bool has_overlap_inputs() const {
        return !orb_names.empty() && !orb_dir.empty();
    }

    bool has_velocity_inputs() const {
        return has_overlap_inputs() && !psd_names.empty() && !psd_dir.empty();
    }

    bool enabled() const {
        return has_overlap_inputs();
    }
};

struct OperatorSourceConfig {
    Hdf5HamiltonianSourceConfig hdf5;
    Op2cIntegralSourceConfig op2c;
};

struct DisorderConfig {
    std::string kind = "none";
    double strength_ev = 0.0;
    std::uint64_t seed = 0;
    bool per_atom = true;

    bool enabled() const {
        return kind != "none" && std::abs(strength_ev) > 0.0;
    }
};

struct OperatorSetupConfig {
    std::vector<std::vector<double>> kpoints;
    OperatorSourceConfig sources;
    PreconditionerConfig preconditioner;
    DisorderConfig disorder;
    bool compute_bounds = true;
};

struct OperatorRuntimeContext {
    int npar = 1;
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm master_comm = MPI_COMM_NULL;
};

class OperatorSetBuilder;
class DiagonalPositionBuilder;

class IOperatorProvider {
public:
    virtual ~IOperatorProvider() = default;
    virtual std::string name() const = 0;
    virtual ProviderCapabilities capabilities() const = 0;
    virtual void populate(AtomicData& atomic_data, OperatorSetBuilder& builder) const = 0;
};

struct VelocityTriplet {
    using Scalar = std::complex<double>;
    using Matrix = BlockSpMat<Scalar>;

    std::unique_ptr<Matrix> x;
    std::unique_ptr<Matrix> y;
    std::unique_ptr<Matrix> z;
};

class IVelocityBuilder {
public:
    virtual ~IVelocityBuilder() = default;
    virtual std::unique_ptr<VelocityTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const = 0;
};

struct PositionTriplet {
    using Scalar = std::complex<double>;
    using Matrix = BlockSpMat<Scalar>;

    std::unique_ptr<Matrix> x;
    std::unique_ptr<Matrix> y;
    std::unique_ptr<Matrix> z;
};

class IPositionBuilder {
public:
    virtual ~IPositionBuilder() = default;
    virtual std::unique_ptr<PositionTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const = 0;
};

class OperatorSet {
public:
    using Scalar = std::complex<double>;
    using Matrix = BlockSpMat<Scalar>;
    using Preconditioner = rsatb::calc::Preconditioner<Scalar>;

    struct KPointOperatorState {
        int k_idx = -1;
        std::vector<double> kpoint;
        std::unique_ptr<Matrix> hamiltonian;
        std::unique_ptr<Matrix> overlap;
        std::unique_ptr<Preconditioner> preconditioner;
        std::unique_ptr<VelocityTriplet> velocity;
        std::unique_ptr<PositionTriplet> position;
    };

    const std::vector<std::pair<int, std::vector<double>>>& local_kpoints() const {
        return local_kpoints_;
    }

    int global_nk() const {
        return global_nk_;
    }

    bool has_hamiltonian() const {
        return static_cast<bool>(hamiltonian_images_);
    }

    bool has_overlap() const {
        return static_cast<bool>(overlap_images_);
    }

    bool has_velocity() const {
        return static_cast<bool>(velocity_builder_);
    }

    bool has_position() const {
        return static_cast<bool>(position_builder_);
    }

    bool has_bounds() const {
        return bounds_ready_;
    }

    const ImageContainer<double>* hamiltonian_images() const {
        return hamiltonian_images_.get();
    }

    const ImageContainer<double>* overlap_images() const {
        return overlap_images_.get();
    }

    double emin() const {
        return emin_;
    }

    double emax() const {
        return emax_;
    }

    std::pair<double, double> bounds() const {
        return std::make_pair(emin_, emax_);
    }

    bool has_preconditioner(int k_idx) const {
        const auto* state = try_kpoint_state(k_idx);
        return state != nullptr && static_cast<bool>(state->preconditioner);
    }

    KPointOperatorState& kpoint_state(int k_idx) {
        auto it = kpoint_states_.find(k_idx);
        if (it == kpoint_states_.end()) {
            throw std::out_of_range("Requested k-point is not owned by this rank");
        }
        return it->second;
    }

    const KPointOperatorState& kpoint_state(int k_idx) const {
        auto it = kpoint_states_.find(k_idx);
        if (it == kpoint_states_.end()) {
            throw std::out_of_range("Requested k-point is not owned by this rank");
        }
        return it->second;
    }

    const KPointOperatorState* try_kpoint_state(int k_idx) const {
        auto it = kpoint_states_.find(k_idx);
        return (it == kpoint_states_.end()) ? nullptr : &it->second;
    }

    Matrix* hamiltonian_k(int k_idx) {
        auto* state = try_kpoint_state(k_idx);
        return (state == nullptr) ? nullptr : state->hamiltonian.get();
    }

    Matrix* overlap_k(int k_idx) {
        auto* state = try_kpoint_state(k_idx);
        return (state == nullptr) ? nullptr : state->overlap.get();
    }

    const VelocityTriplet& velocity_state(int k_idx) {
        auto& state = kpoint_state(k_idx);
        if (!state.velocity) {
            if (!velocity_builder_) {
                throw std::runtime_error(
                    "No velocity-capable operator provider is configured for this OperatorSet.");
            }
            state.velocity = velocity_builder_->build(*atomic_data_, state.kpoint);
        }
        return *state.velocity;
    }

    const PositionTriplet& position_state(int k_idx) {
        auto& state = kpoint_state(k_idx);
        if (!state.position) {
            if (!position_builder_) {
                throw std::runtime_error(
                    "No position-capable operator provider is configured for this OperatorSet.");
            }
            state.position = position_builder_->build(*atomic_data_, state.kpoint);
        }
        return *state.position;
    }

private:
    friend class OperatorSetBuilder;

    OperatorSet(AtomicData& atomic_data, MPI_Comm comm, bool verbose)
        : atomic_data_(&atomic_data), comm_(comm), verbose_(verbose) {}

    void initialize(const OperatorSetupConfig& config, const OperatorRuntimeContext& runtime) {
        global_nk_ = static_cast<int>(config.kpoints.size());
        local_kpoints_.clear();
        kpoint_states_.clear();
        bounds_ready_ = false;
        emin_ = -1.0;
        emax_ = 1.0;

        const int world_size = std::max(1, runtime.world_size);
        const int npar = std::max(1, std::min(runtime.npar, world_size));
        const int my_color = (runtime.world_rank * npar) / world_size;

        for (size_t i = 0; i < config.kpoints.size(); ++i) {
            if (static_cast<int>(i) % npar != my_color) {
                continue;
            }

            KPointOperatorState state;
            state.k_idx = static_cast<int>(i);
            state.kpoint = config.kpoints[i];

            if (hamiltonian_images_) {
                state.hamiltonian.reset(hamiltonian_images_->sample_k(state.kpoint, PhaseConvention::R_AND_POSITION));
            }
            if (overlap_images_) {
                state.overlap.reset(overlap_images_->sample_k(state.kpoint, PhaseConvention::R_AND_POSITION));
            }
            if (state.hamiltonian) {
                apply_disorder_if_requested(state, config.disorder);
            }
            if (state.overlap) {
                build_preconditioner(state, config.preconditioner);
            }

            local_kpoints_.push_back(std::make_pair(state.k_idx, state.kpoint));
            kpoint_states_.emplace(state.k_idx, std::move(state));
        }

        compute_bounds_if_requested(config.compute_bounds, runtime.master_comm);
    }

    void build_preconditioner(KPointOperatorState& state, const PreconditionerConfig& config) {
        Matrix* overlap = state.overlap.get();
        if (overlap == nullptr) {
            return;
        }

        const std::string mode = config.normalized_mode();
        if (mode == "None") {
            return;
        }

        // if preconditioner on "H", the preconditioner is the exact inverse square root of the overlap S, and can use to transform the "H" to orthogonal basis,
        // so as to act as a preconditioner for each "H" application, as PHP.
        // if preconditioner on "S", the preconditioner is the approximate inverse square root of the overlap S, then, each time the operator will be applied using linear solvers as S-1H, and the S-1 is preconditioned by the P
        if (mode != "H" && mode != "S") {
            throw std::runtime_error("Preconditioner supports H, S, H/S or None");
        }

        rsatb::calc::PrimmeEigenSolver<Scalar, typename Matrix::KernelType> esolver;
        std::vector<double> evals;
        std::vector<DistVector<Scalar>> evecs;
        esolver.solve_largest(*overlap, nullptr, nullptr, 1, 1e-3, evals, evecs, verbose_);

        const double max_eig = evals.empty() ? 1.0 : evals[0];
        const double factor = 1.0 / std::sqrt(max_eig + 0.1);
        state.preconditioner = std::make_unique<rsatb::calc::ISQRTMPreconditioner<Scalar>>(
            *overlap, config.maxiter, config.tol, config.sptol, config.method, factor, verbose_);

        if (mode == "H") {
            state.overlap.reset();
        }
    }

    static std::uint64_t splitmix64(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    static double uniform_from_hash(std::uint64_t value) {
        const std::uint64_t mantissa = splitmix64(value) >> 11;
        return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);
    }

    void apply_disorder_if_requested(KPointOperatorState& state, const DisorderConfig& config) {
        if (!config.enabled() || !state.hamiltonian) {
            return;
        }
        if (config.kind != "anderson") {
            throw std::runtime_error("Unsupported disorder kind: " + config.kind);
        }

        int start = 0;
        int end = 0;
        atomic_data_->get_vertex_range_local(&start, &end);
        for (int atom = start; atom < end; ++atom) {
            if (atomic_data_->atom_is_ghost(atom)) {
                continue;
            }

            const int global_atom = atomic_data_->get_global_index(atom);
            auto row_it = state.hamiltonian->graph->global_to_local.find(global_atom);
            if (row_it == state.hamiltonian->graph->global_to_local.end()) {
                continue;
            }
            const int local_row = row_it->second;
            const int norb = state.hamiltonian->graph->block_sizes[local_row];
            std::vector<Scalar> block(static_cast<size_t>(norb) * static_cast<size_t>(norb), Scalar(0.0, 0.0));

            for (int orb = 0; orb < norb; ++orb) {
                const std::uint64_t orbital_key = config.per_atom ? 0ULL : static_cast<std::uint64_t>(orb + 1);
                const double disorder =
                    config.strength_ev *
                    (uniform_from_hash(config.seed ^ (static_cast<std::uint64_t>(global_atom + 1) << 32) ^ orbital_key) - 0.5);
                block[static_cast<size_t>(orb) * static_cast<size_t>(norb) + static_cast<size_t>(orb)] = Scalar(disorder, 0.0);
            }

            state.hamiltonian->update_local_block(
                local_row, local_row, block.data(), norb, norb, AssemblyMode::ADD, MatrixLayout::ColMajor);
        }
    }

    void compute_bounds_if_requested(bool compute_bounds, MPI_Comm master_comm) {
        if (!compute_bounds || !has_hamiltonian()) {
            return;
        }

        double local_emin = std::numeric_limits<double>::max();
        double local_emax = std::numeric_limits<double>::lowest();

        for (const auto& entry : kpoint_states_) {
            const KPointOperatorState& state = entry.second;
            if (!state.hamiltonian) {
                continue;
            }

            rsatb::calc::PrimmeEigenSolver<Scalar> esolver;
            std::vector<double> evals_min;
            std::vector<double> evals_max;
            std::vector<DistVector<Scalar>> evecs;

            Matrix* overlap = state.overlap.get();
            Preconditioner* preconditioner = state.preconditioner.get();
            const auto pc_type = preconditioner ? preconditioner->get_type() : rsatb::calc::PreconditionerType::Left;

            esolver.solve(*state.hamiltonian, overlap, preconditioner, 1, primme_smallest, 1e-3, evals_min, evecs, verbose_, pc_type);
            esolver.solve(*state.hamiltonian, overlap, preconditioner, 1, primme_largest, 1e-3, evals_max, evecs, verbose_, pc_type);

            if (!evals_min.empty()) {
                local_emin = std::min(local_emin, evals_min[0]);
            }
            if (!evals_max.empty()) {
                local_emax = std::max(local_emax, evals_max[0]);
            }
        }

        double global_emin = local_emin;
        double global_emax = local_emax;

        if (master_comm != MPI_COMM_NULL) {
            MPI_Allreduce(&local_emin, &global_emin, 1, MPI_DOUBLE, MPI_MIN, master_comm);
            MPI_Allreduce(&local_emax, &global_emax, 1, MPI_DOUBLE, MPI_MAX, master_comm);
        }

        MPI_Bcast(&global_emin, 1, MPI_DOUBLE, 0, comm_);
        MPI_Bcast(&global_emax, 1, MPI_DOUBLE, 0, comm_);

        emin_ = global_emin - 0.5;
        emax_ = global_emax + 0.5;
        bounds_ready_ = true;
    }

    AtomicData* atomic_data_ = nullptr;
    MPI_Comm comm_ = MPI_COMM_NULL;
    bool verbose_ = false;

    ProviderCapabilities capabilities_;
    std::unique_ptr<ImageContainer<double>> hamiltonian_images_;
    std::unique_ptr<ImageContainer<double>> overlap_images_;
    std::unique_ptr<IVelocityBuilder> velocity_builder_;
    std::unique_ptr<IPositionBuilder> position_builder_;

    std::vector<std::pair<int, std::vector<double>>> local_kpoints_;
    std::map<int, KPointOperatorState> kpoint_states_;
    int global_nk_ = 0;
    bool bounds_ready_ = false;
    double emin_ = -1.0;
    double emax_ = 1.0;
};

class OperatorSetBuilder {
public:
    using Scalar = OperatorSet::Scalar;

    OperatorSetBuilder(AtomicData& atomic_data, MPI_Comm comm, bool verbose = false)
        : atomic_data_(atomic_data), comm_(comm), verbose_(verbose) {}

    MPI_Comm comm() const {
        return comm_;
    }

    void note_capabilities(const ProviderCapabilities& capabilities) {
        capabilities_.hamiltonian = capabilities_.hamiltonian || capabilities.hamiltonian;
        capabilities_.overlap = capabilities_.overlap || capabilities.overlap;
        capabilities_.velocity = capabilities_.velocity || capabilities.velocity;
        capabilities_.position = capabilities_.position || capabilities.position;
    }

    void set_hamiltonian_images(std::unique_ptr<ImageContainer<double>> images) {
        if (hamiltonian_images_) {
            throw std::runtime_error("Hamiltonian images are already registered");
        }
        hamiltonian_images_ = std::move(images);
    }

    void set_overlap_images(std::unique_ptr<ImageContainer<double>> images) {
        if (overlap_images_) {
            throw std::runtime_error("Overlap images are already registered");
        }
        overlap_images_ = std::move(images);
    }

    void set_velocity_builder(std::unique_ptr<IVelocityBuilder> builder) {
        if (velocity_builder_) {
            throw std::runtime_error("Velocity builder is already registered");
        }
        velocity_builder_ = std::move(builder);
    }

    void set_position_builder(std::unique_ptr<IPositionBuilder> builder) {
        if (position_builder_) {
            throw std::runtime_error("Position builder is already registered");
        }
        position_builder_ = std::move(builder);
    }

    std::unique_ptr<OperatorSet> build(const OperatorSetupConfig& config, const OperatorRuntimeContext& runtime);

private:
    AtomicData& atomic_data_;
    MPI_Comm comm_;
    bool verbose_ = false;
    ProviderCapabilities capabilities_;
    std::unique_ptr<ImageContainer<double>> hamiltonian_images_;
    std::unique_ptr<ImageContainer<double>> overlap_images_;
    std::unique_ptr<IVelocityBuilder> velocity_builder_;
    std::unique_ptr<IPositionBuilder> position_builder_;
};

class Hdf5HamiltonianProvider : public IOperatorProvider {
public:
    explicit Hdf5HamiltonianProvider(Hdf5HamiltonianSourceConfig config)
        : config_(std::move(config)) {}

    std::string name() const override {
        return "Hdf5HamiltonianProvider";
    }

    ProviderCapabilities capabilities() const override {
        return ProviderCapabilities{true, false, false};
    }

    void populate(AtomicData& atomic_data, OperatorSetBuilder& builder) const override {
        rsatb::io::HDF5Handler h5(config_.filename);
        auto construction = OperatorConstruction(builder.comm());
        builder.set_hamiltonian_images(construction.build_from_hdf5(atomic_data, h5));
    }

private:
    Hdf5HamiltonianSourceConfig config_;
};

class Op2cVelocityBuilder : public IVelocityBuilder {
public:
    explicit Op2cVelocityBuilder(std::shared_ptr<OperatorConstruction> construction)
        : construction_(std::move(construction)) {}

    std::unique_ptr<VelocityTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const override {
        if (kpoint.size() != 3) {
            throw std::runtime_error("Velocity construction requires 3-component k-points");
        }

        DistGraph* graph = construction_->velocity_graph(atomic_data);
        auto triplet = std::make_unique<VelocityTriplet>();
        triplet->x = std::make_unique<VelocityTriplet::Matrix>(graph);
        triplet->y = std::make_unique<VelocityTriplet::Matrix>(graph);
        triplet->z = std::make_unique<VelocityTriplet::Matrix>(graph);

        construction_->build_overlap_velocity(
            triplet->x.get(), triplet->y.get(), triplet->z.get(),
            atomic_data, kpoint[0], kpoint[1], kpoint[2]);

        return triplet;
    }

private:
    std::shared_ptr<OperatorConstruction> construction_;
};

class Op2cPositionBuilder : public IPositionBuilder {
public:
    explicit Op2cPositionBuilder(std::shared_ptr<OperatorConstruction> construction)
        : construction_(std::move(construction)) {}

    std::unique_ptr<PositionTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const override {
        if (kpoint.size() != 3) {
            throw std::runtime_error("Position construction requires 3-component k-points");
        }

        auto triplet = std::make_unique<PositionTriplet>();
        triplet->x = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->y = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->z = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());

        construction_->build_position(
            triplet->x.get(), triplet->y.get(), triplet->z.get(),
            atomic_data, kpoint[0], kpoint[1], kpoint[2]);

        return triplet;
    }

private:
    std::shared_ptr<OperatorConstruction> construction_;
};

class DiagonalPositionBuilder : public IPositionBuilder {
public:
    std::unique_ptr<PositionTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>&
    ) const override {
        auto triplet = std::make_unique<PositionTriplet>();
        triplet->x = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->y = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->z = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());

        int start = 0;
        int end = 0;
        atomic_data.get_vertex_range_local(&start, &end);
        for (int atom = start; atom < end; ++atom) {
            if (atomic_data.atom_is_ghost(atom)) {
                continue;
            }

            int norb = 0;
            atomic_data.get_atom_norb(atom, &norb);
            const int global_atom = atomic_data.get_global_index(atom);

            double rx = 0.0;
            double ry = 0.0;
            double rz = 0.0;
            atomic_data.get_pos(atom, &rx, &ry, &rz);

            std::vector<std::complex<double>> bx(static_cast<size_t>(norb) * static_cast<size_t>(norb), std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> by(static_cast<size_t>(norb) * static_cast<size_t>(norb), std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> bz(static_cast<size_t>(norb) * static_cast<size_t>(norb), std::complex<double>(0.0, 0.0));
            for (int orb = 0; orb < norb; ++orb) {
                bx[static_cast<size_t>(orb) * static_cast<size_t>(norb) + static_cast<size_t>(orb)] = std::complex<double>(rx, 0.0);
                by[static_cast<size_t>(orb) * static_cast<size_t>(norb) + static_cast<size_t>(orb)] = std::complex<double>(ry, 0.0);
                bz[static_cast<size_t>(orb) * static_cast<size_t>(norb) + static_cast<size_t>(orb)] = std::complex<double>(rz, 0.0);
            }

            triplet->x->add_block(global_atom, global_atom, bx.data(), norb, norb, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            triplet->y->add_block(global_atom, global_atom, by.data(), norb, norb, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            triplet->z->add_block(global_atom, global_atom, bz.data(), norb, norb, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }

        triplet->x->assemble();
        triplet->y->assemble();
        triplet->z->assemble();
        return triplet;
    }
};

inline std::unique_ptr<OperatorSet> OperatorSetBuilder::build(
    const OperatorSetupConfig& config,
    const OperatorRuntimeContext& runtime
) {
    if (!position_builder_) {
        position_builder_ = std::make_unique<DiagonalPositionBuilder>();
        capabilities_.position = true;
    }
    auto set = std::unique_ptr<OperatorSet>(new OperatorSet(atomic_data_, comm_, verbose_));
    set->capabilities_ = capabilities_;
    set->hamiltonian_images_ = std::move(hamiltonian_images_);
    set->overlap_images_ = std::move(overlap_images_);
    set->velocity_builder_ = std::move(velocity_builder_);
    set->position_builder_ = std::move(position_builder_);
    set->initialize(config, runtime);
    return set;
}

class Op2cIntegralProvider : public IOperatorProvider {
public:
    explicit Op2cIntegralProvider(Op2cIntegralSourceConfig config)
        : config_(std::move(config)) {}

    std::string name() const override {
        return "Op2cIntegralProvider";
    }

    ProviderCapabilities capabilities() const override {
        return ProviderCapabilities{false, config_.has_overlap_inputs(), config_.has_velocity_inputs(), config_.has_overlap_inputs()};
    }

    void populate(AtomicData& atomic_data, OperatorSetBuilder& builder) const override {
        auto construction = std::make_shared<OperatorConstruction>(builder.comm());
        construction->init(
            static_cast<size_t>(atomic_data.n_type()),
            config_.nspin,
            config_.lspinorb,
            config_.orb_dir,
            config_.orb_names,
            config_.psd_dir,
            config_.psd_names,
            config_.log_file);

        builder.set_overlap_images(construction->build_overlap(atomic_data));
        builder.set_position_builder(std::make_unique<Op2cPositionBuilder>(construction));
        if (config_.has_velocity_inputs() && construction->has_beta_projectors()) {
            builder.set_velocity_builder(std::make_unique<Op2cVelocityBuilder>(construction));
        }
    }

private:
    Op2cIntegralSourceConfig config_;
};

inline std::vector<std::unique_ptr<IOperatorProvider>> make_operator_providers(const OperatorSourceConfig& sources) {
    std::vector<std::unique_ptr<IOperatorProvider>> providers;
    if (sources.hdf5.enabled()) {
        providers.emplace_back(std::make_unique<Hdf5HamiltonianProvider>(sources.hdf5));
    }
    if (sources.op2c.enabled()) {
        providers.emplace_back(std::make_unique<Op2cIntegralProvider>(sources.op2c));
    }
    return providers;
}

inline std::unique_ptr<OperatorSet> build_operator_set(
    AtomicData& atomic_data,
    MPI_Comm comm,
    bool verbose,
    const OperatorSetupConfig& config,
    const OperatorRuntimeContext& runtime) {
    OperatorSetBuilder builder(atomic_data, comm, verbose);
    auto providers = make_operator_providers(config.sources);
    for (const auto& provider : providers) {
        builder.note_capabilities(provider->capabilities());
        provider->populate(atomic_data, builder);
    }
    return builder.build(config, runtime);
}

} // namespace backend
} // namespace rsatb

#endif
