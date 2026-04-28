#ifndef RSATB_BACKEND_OPERATOR_SET_HPP
#define RSATB_BACKEND_OPERATOR_SET_HPP

#include "setup.hpp"
#include "operator_construction.hpp"
#include "sktb/sktb_construction.hpp"
#include "../calc/eigen.hpp"
#include "../calc/preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rsatb {
namespace backend {

struct OperatorCapabilities {
    bool hamiltonian = false;
    bool overlap = false;
    bool velocity = false;
    bool position = false;
};

struct OperatorRuntimeContext {
    int npar = 1;
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm master_comm = MPI_COMM_NULL;
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

struct PositionAxisData {
    using Scalar = std::complex<double>;
    using Matrix = BlockSpMat<Scalar>;
    using DiagonalVector = DistVector<Scalar>;

    std::unique_ptr<Matrix> matrix;
    std::unique_ptr<DiagonalVector> orthogonal_diagonal;
};

struct PositionTriplet {
    using Scalar = PositionAxisData::Scalar;
    using Matrix = PositionAxisData::Matrix;
    using DiagonalVector = PositionAxisData::DiagonalVector;

    PositionAxisData x;
    PositionAxisData y;
    PositionAxisData z;
};

class IPositionBuilder {
public:
    virtual ~IPositionBuilder() = default;
    virtual std::unique_ptr<PositionTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const = 0;
};

struct ModelArtifacts;

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
        std::unique_ptr<FirstLatticeMomentTriplet> first_lattice_moment;
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
                    "No velocity-capable model/operator path is configured for this OperatorSet.");
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
                    "No position-capable model/operator path is configured for this OperatorSet.");
            }
            state.position = position_builder_->build(*atomic_data_, state.kpoint);
        }
        attach_orthogonal_diagonal_position(*state.position);
        return *state.position;
    }

    const FirstLatticeMomentTriplet& first_lattice_moment_state(int k_idx) {
        auto& state = kpoint_state(k_idx);
        if (!state.first_lattice_moment) {
            if (!hamiltonian_images_) {
                throw std::runtime_error(
                    "No Hamiltonian image data is configured for first-lattice-moment construction.");
            }
            state.first_lattice_moment =
                build_first_lattice_moment_triplet(*atomic_data_, *hamiltonian_images_);
        }
        return *state.first_lattice_moment;
    }

private:
    friend std::unique_ptr<OperatorSet> finalize_operator_set(
        AtomicData& atomic_data,
        MPI_Comm comm,
        bool verbose,
        ModelArtifacts artifacts,
        const rsatb::model::SetupOptions& options,
        const OperatorRuntimeContext& runtime);

    OperatorSet(AtomicData& atomic_data, MPI_Comm comm, bool verbose)
        : atomic_data_(&atomic_data), comm_(comm), verbose_(verbose) {}

    void initialize(const rsatb::model::SetupOptions& options, const OperatorRuntimeContext& runtime) {
        global_nk_ = static_cast<int>(options.kpoints.size());
        local_kpoints_.clear();
        kpoint_states_.clear();
        bounds_ready_ = false;
        emin_ = -1.0;
        emax_ = 1.0;

        const int world_size = std::max(1, runtime.world_size);
        const int npar = std::max(1, std::min(runtime.npar, world_size));
        const int my_color = (runtime.world_rank * npar) / world_size;

        for (size_t i = 0; i < options.kpoints.size(); ++i) {
            if (static_cast<int>(i) % npar != my_color) {
                continue;
            }

            KPointOperatorState state;
            state.k_idx = static_cast<int>(i);
            state.kpoint = options.kpoints[i];

            if (hamiltonian_images_) {
                state.hamiltonian.reset(hamiltonian_images_->sample_k(state.kpoint, PhaseConvention::R_AND_POSITION));
            }
            if (overlap_images_) {
                state.overlap.reset(overlap_images_->sample_k(state.kpoint, PhaseConvention::R_AND_POSITION));
            }
            if (state.overlap) {
                build_preconditioner(state, options.preconditioner);
            }

            local_kpoints_.push_back(std::make_pair(state.k_idx, state.kpoint));
            kpoint_states_.emplace(state.k_idx, std::move(state));
        }

        compute_bounds_if_requested(options, runtime.master_comm);
    }

    void build_preconditioner(
        KPointOperatorState& state,
        const rsatb::model::PreconditionerOptions& config
    ) {
        Matrix* overlap = state.overlap.get();
        if (overlap == nullptr) {
            return;
        }

        const std::string mode = config.normalized_mode();
        if (mode == "None") {
            return;
        }

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

    void attach_orthogonal_diagonal_position(PositionTriplet& position) const {
        if (!position.x.orthogonal_diagonal) {
            position.x.orthogonal_diagonal = std::make_unique<PositionTriplet::DiagonalVector>(
                detail::make_orthogonal_diagonal_coordinate_vector(*atomic_data_, 0));
        }
        if (!position.y.orthogonal_diagonal) {
            position.y.orthogonal_diagonal = std::make_unique<PositionTriplet::DiagonalVector>(
                detail::make_orthogonal_diagonal_coordinate_vector(*atomic_data_, 1));
        }
        if (!position.z.orthogonal_diagonal) {
            position.z.orthogonal_diagonal = std::make_unique<PositionTriplet::DiagonalVector>(
                detail::make_orthogonal_diagonal_coordinate_vector(*atomic_data_, 2));
        }
    }

    static std::string kpoint_to_string(const std::vector<double>& kpoint) {
        std::ostringstream out;
        out << "[";
        for (size_t i = 0; i < kpoint.size(); ++i) {
            if (i > 0) out << ", ";
            out << kpoint[i];
        }
        out << "]";
        return out.str();
    }

    void set_explicit_energy_bounds(const std::pair<double, double>& bounds) {
        if (!std::isfinite(bounds.first) || !std::isfinite(bounds.second) || !(bounds.first < bounds.second)) {
            throw std::runtime_error("setup energy_bounds must be finite and satisfy emin < emax.");
        }
        emin_ = bounds.first;
        emax_ = bounds.second;
        bounds_ready_ = true;
    }

    void compute_bounds_if_requested(const rsatb::model::SetupOptions& options, MPI_Comm master_comm) {
        if (options.energy_bounds.has_value()) {
            set_explicit_energy_bounds(*options.energy_bounds);
            return;
        }

        if (!options.compute_bounds || !has_hamiltonian()) {
            return;
        }
        if (options.bounds_zero_tol < 0.0) {
            throw std::runtime_error("setup bounds_zero_tol must be non-negative.");
        }
        if (options.bounds_max_matvecs < 0) {
            throw std::runtime_error("setup bounds_max_matvecs must be non-negative.");
        }

        double local_emin = std::numeric_limits<double>::max();
        double local_emax = std::numeric_limits<double>::lowest();

        for (const auto& entry : kpoint_states_) {
            const KPointOperatorState& state = entry.second;
            if (!state.hamiltonian) {
                continue;
            }

            const double h_max_abs = state.hamiltonian->maxabs();
            if (!std::isfinite(h_max_abs)) {
                throw std::runtime_error(
                    "Automatic bounds estimation found a non-finite Hamiltonian entry at k-point " +
                    kpoint_to_string(state.kpoint) + ".");
            }
            if (h_max_abs <= options.bounds_zero_tol) {
                local_emin = std::min(local_emin, 0.0);
                local_emax = std::max(local_emax, 0.0);
                continue;
            }

            rsatb::calc::PrimmeEigenSolver<Scalar> esolver;
            std::vector<double> evals_min;
            std::vector<double> evals_max;
            std::vector<DistVector<Scalar>> evecs;

            Matrix* overlap = state.overlap.get();
            Preconditioner* preconditioner = state.preconditioner.get();
            const auto pc_type = preconditioner ? preconditioner->get_type() : rsatb::calc::PreconditionerType::Left;

            try {
                esolver.solve(
                    *state.hamiltonian,
                    overlap,
                    preconditioner,
                    1,
                    primme_smallest,
                    1e-3,
                    evals_min,
                    evecs,
                    verbose_,
                    pc_type,
                    options.bounds_max_matvecs);
                esolver.solve(
                    *state.hamiltonian,
                    overlap,
                    preconditioner,
                    1,
                    primme_largest,
                    1e-3,
                    evals_max,
                    evecs,
                    verbose_,
                    pc_type,
                    options.bounds_max_matvecs);
            } catch (const std::exception& error) {
                if (overlap == nullptr && preconditioner == nullptr) {
                    const double fallback_bound = state.hamiltonian->frobenius_norm();
                    if (!std::isfinite(fallback_bound)) {
                        throw std::runtime_error(
                            "Automatic bounds fallback found a non-finite Hamiltonian norm at k-point " +
                            kpoint_to_string(state.kpoint) + ".");
                    }
                    local_emin = std::min(local_emin, -fallback_bound);
                    local_emax = std::max(local_emax, fallback_bound);
                    if (verbose_ && state.hamiltonian->graph != nullptr && state.hamiltonian->graph->rank == 0) {
                        std::cerr
                            << "Warning: PRIMME bounds solve failed at k-point "
                            << kpoint_to_string(state.kpoint)
                            << "; using conservative Frobenius norm bound. Underlying error: "
                            << error.what() << std::endl;
                    }
                    continue;
                }

                throw std::runtime_error(
                    "Automatic bounds estimation failed at k-point " + kpoint_to_string(state.kpoint) +
                    ". Provide SetupOptions.energy_bounds or increase bounds_max_matvecs. Underlying error: " +
                    error.what());
            }

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

        if (comm_ != MPI_COMM_NULL) {
            MPI_Bcast(&global_emin, 1, MPI_DOUBLE, 0, comm_);
            MPI_Bcast(&global_emax, 1, MPI_DOUBLE, 0, comm_);
        }

        if (!std::isfinite(global_emin) || !std::isfinite(global_emax) || global_emin > global_emax) {
            throw std::runtime_error("Automatic bounds estimation did not produce finite energy bounds.");
        }

        emin_ = global_emin - 0.5;
        emax_ = global_emax + 0.5;
        bounds_ready_ = true;
    }

    AtomicData* atomic_data_ = nullptr;
    MPI_Comm comm_ = MPI_COMM_NULL;
    bool verbose_ = false;

    OperatorCapabilities capabilities_;
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

struct ModelArtifacts {
    OperatorCapabilities capabilities;
    std::unique_ptr<ImageContainer<double>> hamiltonian_images;
    std::unique_ptr<ImageContainer<double>> overlap_images;
    std::unique_ptr<IVelocityBuilder> velocity_builder;
    std::unique_ptr<IPositionBuilder> position_builder;
};

inline std::unique_ptr<OperatorConstruction> make_op2c_construction(
    MPI_Comm comm,
    AtomicData& atomic_data,
    const rsatb::model::LCAOIntegralInput& integrals
) {
    auto construction = std::make_unique<OperatorConstruction>(comm);
    construction->init(
        static_cast<size_t>(atomic_data.n_type()),
        integrals.nspin,
        integrals.lspinorb,
        integrals.orb_dir,
        integrals.orb_names,
        integrals.psd_dir,
        integrals.psd_names,
        integrals.log_file);
    return construction;
}

class Op2cVelocityBuilder : public IVelocityBuilder {
public:
    Op2cVelocityBuilder(MPI_Comm comm, rsatb::model::LCAOIntegralInput integrals)
        : comm_(comm), integrals_(std::move(integrals)) {}

    std::unique_ptr<VelocityTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const override {
        if (kpoint.size() != 3) {
            throw std::runtime_error("Velocity construction requires 3-component k-points");
        }

        auto construction = make_op2c_construction(comm_, atomic_data, integrals_);
        DistGraph* graph = construction->velocity_graph(atomic_data);
        auto triplet = std::make_unique<VelocityTriplet>();
        triplet->x = std::make_unique<VelocityTriplet::Matrix>(graph);
        triplet->y = std::make_unique<VelocityTriplet::Matrix>(graph);
        triplet->z = std::make_unique<VelocityTriplet::Matrix>(graph);

        construction->build_overlap_velocity(
            triplet->x.get(), triplet->y.get(), triplet->z.get(),
            atomic_data, kpoint[0], kpoint[1], kpoint[2]);

        return triplet;
    }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    rsatb::model::LCAOIntegralInput integrals_;
};

class Op2cPositionBuilder : public IPositionBuilder {
public:
    Op2cPositionBuilder(MPI_Comm comm, rsatb::model::LCAOIntegralInput integrals)
        : comm_(comm), integrals_(std::move(integrals)) {}

    std::unique_ptr<PositionTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const override {
        if (kpoint.size() != 3) {
            throw std::runtime_error("Position construction requires 3-component k-points");
        }

        auto triplet = std::make_unique<PositionTriplet>();
        triplet->x.matrix = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->y.matrix = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->z.matrix = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());

        auto construction = make_op2c_construction(comm_, atomic_data, integrals_);
        construction->build_position(
            triplet->x.matrix.get(), triplet->y.matrix.get(), triplet->z.matrix.get(),
            atomic_data, kpoint[0], kpoint[1], kpoint[2]);

        return triplet;
    }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    rsatb::model::LCAOIntegralInput integrals_;
};

class OrthogonalImageVelocityBuilder : public IVelocityBuilder {
public:
    explicit OrthogonalImageVelocityBuilder(const ImageContainer<double>* hamiltonian_images)
        : hamiltonian_images_(hamiltonian_images) {}

    std::unique_ptr<VelocityTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>& kpoint
    ) const override {
        if (kpoint.size() != 3) {
            throw std::runtime_error("Orthogonal velocity construction requires 3-component k-points");
        }
        if (hamiltonian_images_ == nullptr) {
            throw std::runtime_error("Orthogonal velocity construction requires Hamiltonian image data.");
        }
        if (hamiltonian_images_->atom_data != &atomic_data || hamiltonian_images_->base_graph != atomic_data.graph) {
            throw std::runtime_error(
                "Orthogonal velocity construction requires Hamiltonian images built from the same AtomicData graph.");
        }
        // TODO: I think the VBCSR have a function called the commutator_diagonal that can simplify the velocity construction.

        auto triplet = std::make_unique<VelocityTriplet>();
        triplet->x = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
        triplet->y = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
        triplet->z = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());

        constexpr double hbar_ev_fs = 0.6582119569;
        DistGraph* reference_graph = atomic_data.get_graph();
        const int n_owned = static_cast<int>(reference_graph->owned_global_indices.size());
        const std::complex<double> imag_unit(0.0, 1.0);

        for (const auto& entry : hamiltonian_images_->image_blocks) {
            const auto& shift = entry.first;
            auto* image_matrix = entry.second;
            auto* image_graph = image_matrix->graph;

            if (image_graph->owned_global_indices != reference_graph->owned_global_indices) {
                throw std::runtime_error(
                    "Orthogonal velocity construction requires image graphs aligned with the AtomicData base graph.");
            }

            const double phase_r = -2.0 * M_PI * (
                kpoint[0] * static_cast<double>(shift[0]) +
                kpoint[1] * static_cast<double>(shift[1]) +
                kpoint[2] * static_cast<double>(shift[2]));
            const std::complex<double> image_phase = std::exp(std::complex<double>(0.0, phase_r));

            const double shift_x = detail::cell_shift_component(atomic_data, shift, 0);
            const double shift_y = detail::cell_shift_component(atomic_data, shift, 1);
            const double shift_z = detail::cell_shift_component(atomic_data, shift, 2);

            const int n_col_r = static_cast<int>(image_graph->block_sizes.size());
            std::vector<int> col_map(n_col_r, -1);
            for (int local_col = 0; local_col < n_col_r; ++local_col) {
                const int gid = image_graph->get_global_index(local_col);
                auto it = reference_graph->global_to_local.find(gid);
                if (it == reference_graph->global_to_local.end()) {
                    throw std::runtime_error("Orthogonal velocity image column not found in the AtomicData base graph.");
                }
                col_map[local_col] = it->second;
            }

            for (int local_row = 0; local_row < n_owned; ++local_row) {
                double ri[3];
                atomic_data.get_pos(local_row, &ri[0], &ri[1], &ri[2]);

                const int start = image_matrix->row_ptr()[local_row];
                const int end = image_matrix->row_ptr()[local_row + 1];
                const int row_dim = image_graph->block_sizes[local_row];

                for (int k = start; k < end; ++k) {
                    const int local_col_r = image_matrix->col_ind()[k];
                    const int local_col_ref = col_map[local_col_r];
                    const int col_dim = image_graph->block_sizes[local_col_r];

                    double rj[3];
                    atomic_data.get_pos(local_col_ref, &rj[0], &rj[1], &rj[2]);

                    const double delta_x = rj[0] - ri[0];
                    const double delta_y = rj[1] - ri[1];
                    const double delta_z = rj[2] - ri[2];

                    double frac_dx = delta_x;
                    double frac_dy = delta_y;
                    double frac_dz = delta_z;
                    atomic_data.invert_cell(&frac_dx, &frac_dy, &frac_dz);

                    const double phase_pos = -2.0 * M_PI * (
                        kpoint[0] * frac_dx +
                        kpoint[1] * frac_dy +
                        kpoint[2] * frac_dz);
                    const std::complex<double> phase = image_phase * std::exp(std::complex<double>(0.0, phase_pos));

                    const double displacement_x = shift_x + delta_x;
                    const double displacement_y = shift_y + delta_y;
                    const double displacement_z = shift_z + delta_z;

                    const int n_elem = static_cast<int>(image_matrix->block_size_elements(k));
                    const double* block_data = image_matrix->block_data(k);

                    auto add_axis_block = [&](VelocityTriplet::Matrix& matrix, double displacement) {
                        if (std::abs(displacement) <= 1e-12) {
                            return;
                        }
                        const std::complex<double> factor = phase * imag_unit * (displacement / hbar_ev_fs);
                        std::vector<std::complex<double>> block(static_cast<size_t>(n_elem));
                        for (int idx = 0; idx < n_elem; ++idx) {
                            block[static_cast<size_t>(idx)] =
                                static_cast<std::complex<double>>(block_data[idx]) * factor;
                        }
                        matrix.update_local_block(
                            local_row,
                            local_col_ref,
                            block.data(),
                            row_dim,
                            col_dim,
                            AssemblyMode::ADD,
                            MatrixLayout::ColMajor);
                    };

                    add_axis_block(*triplet->x, displacement_x);
                    add_axis_block(*triplet->y, displacement_y);
                    add_axis_block(*triplet->z, displacement_z);
                }
            }
        }

        triplet->x->assemble();
        triplet->y->assemble();
        triplet->z->assemble();

        return triplet;
    }

private:
    const ImageContainer<double>* hamiltonian_images_ = nullptr;
};

class OrthogonalDiagonalPositionBuilder : public IPositionBuilder {
public:
    std::unique_ptr<PositionTriplet> build(
        AtomicData& atomic_data,
        const std::vector<double>&
    ) const override {
        (void)atomic_data;
        return std::make_unique<PositionTriplet>();
    }
};

inline std::unique_ptr<ImageContainer<double>> build_hdf5_hamiltonian_images(
    AtomicData& atomic_data,
    MPI_Comm comm,
    const rsatb::model::Hdf5Input& input
) {
    rsatb::io::HDF5Handler h5(input.filename);
    OperatorConstruction construction(comm);
    return construction.build_from_hdf5(atomic_data, h5);
}

inline ModelArtifacts build_lcao_artifacts(
    AtomicData& atomic_data,
    MPI_Comm comm,
    const rsatb::model::LCAOModelConfig& config
) {
    ModelArtifacts artifacts;

    if (config.input.has_value() && config.input->enabled()) {
        artifacts.hamiltonian_images = build_hdf5_hamiltonian_images(atomic_data, comm, *config.input);
        artifacts.capabilities.hamiltonian = true;
    }

    if (config.integrals.enabled()) {
        auto construction = make_op2c_construction(comm, atomic_data, config.integrals);

        // position and velocity is delayed until requested
        artifacts.overlap_images = construction->build_overlap(atomic_data);
        artifacts.position_builder = std::make_unique<Op2cPositionBuilder>(comm, config.integrals);
        artifacts.capabilities.overlap = true;
        artifacts.capabilities.position = true;

        if (config.integrals.has_velocity_inputs() && construction->has_beta_projectors()) {
            artifacts.velocity_builder = std::make_unique<Op2cVelocityBuilder>(comm, config.integrals);
            artifacts.capabilities.velocity = true;
        }
    }

    if (!artifacts.hamiltonian_images) {
        throw std::runtime_error(
            "LCAO model setup requires Hamiltonian input. "
            "Provide model.input.filename or use a model family with native Hamiltonian construction.");
    }

    return artifacts;
}

inline ModelArtifacts build_pairwise_tb_artifacts(
    AtomicData& atomic_data,
    MPI_Comm comm,
    const std::optional<rsatb::model::Hdf5Input>& input,
    const SKTBModel& native_config,
    bool require_native_overlap,
    const std::string& model_name
) {
    ModelArtifacts artifacts;

    if (input.has_value() && input->enabled()) {
        artifacts.hamiltonian_images = build_hdf5_hamiltonian_images(atomic_data, comm, *input);
        artifacts.capabilities.hamiltonian = true;
    }

    if (native_config.enabled()) {
        if (require_native_overlap && !native_config.has_overlap) {
            throw std::runtime_error(
                model_name + " native setup requires both Hamiltonian and overlap parameterization.");
        }

        auto construction = std::make_shared<SKTBConstruction>(native_config);
        if (!artifacts.hamiltonian_images) {
            artifacts.hamiltonian_images = construction->build_hamiltonian_images(atomic_data);
            artifacts.capabilities.hamiltonian = true;
        }
        if (native_config.has_overlap) {
            artifacts.overlap_images = construction->build_overlap_images(atomic_data);
            artifacts.capabilities.overlap = true;
        }
    }

    if (!artifacts.hamiltonian_images) {
        throw std::runtime_error(
            model_name + " model setup requires either model.input.filename or model.native parameterization.");
    }

    return artifacts;
}

inline ModelArtifacts build_sktb_artifacts(
    AtomicData& atomic_data,
    MPI_Comm comm,
    const rsatb::model::SKTBModelConfig& config
) {
    return build_pairwise_tb_artifacts(
        atomic_data,
        comm,
        config.input,
        config.native,
        false,
        "SKTB");
}

inline ModelArtifacts build_dftb_artifacts(
    AtomicData& atomic_data,
    MPI_Comm comm,
    const rsatb::model::DFTBModelConfig& config
) {
    return build_pairwise_tb_artifacts(
        atomic_data,
        comm,
        config.input,
        config.native,
        config.native.enabled(),
        "DFTB");
}

inline ModelArtifacts build_model_artifacts(
    AtomicData& atomic_data,
    MPI_Comm comm,
    const rsatb::model::AnyModelConfig& model_config,
    const std::vector<std::string>& type_symbols
) {
    return std::visit(
        [&](const auto& cfg) -> ModelArtifacts {
            using Config = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<Config, rsatb::model::LCAOModelConfig>) {
                return build_lcao_artifacts(atomic_data, comm, cfg);
            } else if constexpr (std::is_same_v<Config, rsatb::model::SKTBModelConfig>) {
                (void)type_symbols;
                return build_sktb_artifacts(atomic_data, comm, cfg);
            } else {
                (void)type_symbols;
                return build_dftb_artifacts(atomic_data, comm, cfg);
            }
        },
        model_config);
}

inline void install_default_fallbacks(ModelArtifacts& artifacts) {
    if (!artifacts.position_builder && !artifacts.overlap_images) {
        artifacts.position_builder = std::make_unique<OrthogonalDiagonalPositionBuilder>();
        artifacts.capabilities.position = true;
    }
    if (!artifacts.velocity_builder && !artifacts.overlap_images && artifacts.hamiltonian_images) {
        artifacts.velocity_builder =
            std::make_unique<OrthogonalImageVelocityBuilder>(artifacts.hamiltonian_images.get());
        artifacts.capabilities.velocity = true;
    }
}

inline std::unique_ptr<OperatorSet> finalize_operator_set(
    AtomicData& atomic_data,
    MPI_Comm comm,
    bool verbose,
    ModelArtifacts artifacts,
    const rsatb::model::SetupOptions& options,
    const OperatorRuntimeContext& runtime
) {
    install_default_fallbacks(artifacts);

    auto set = std::unique_ptr<OperatorSet>(new OperatorSet(atomic_data, comm, verbose));
    set->capabilities_ = artifacts.capabilities;
    set->hamiltonian_images_ = std::move(artifacts.hamiltonian_images);
    set->overlap_images_ = std::move(artifacts.overlap_images);
    set->velocity_builder_ = std::move(artifacts.velocity_builder);
    set->position_builder_ = std::move(artifacts.position_builder);
    set->initialize(options, runtime);
    return set;
}

inline std::unique_ptr<OperatorSet> build_operator_set(
    AtomicData& atomic_data,
    MPI_Comm comm,
    bool verbose,
    const rsatb::model::AnyModelConfig& model_config,
    const rsatb::model::SetupOptions& options,
    const std::vector<std::string>& type_symbols,
    const OperatorRuntimeContext& runtime
) {
    auto artifacts = build_model_artifacts(atomic_data, comm, model_config, type_symbols);
    return finalize_operator_set(atomic_data, comm, verbose, std::move(artifacts), options, runtime);
}

} // namespace backend
} // namespace rsatb

#endif
