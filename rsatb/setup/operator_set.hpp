#ifndef RSATB_SETUP_OPERATOR_SET_HPP
#define RSATB_SETUP_OPERATOR_SET_HPP

// The operators a calculation runs against, and the state that goes with them.
//
// This is a container, not a construction site. It owns the real-space images,
// the per-k-point matrices, the preconditioner and the spectral window, and it
// builds velocity and position lazily through builders it was handed. Deciding
// WHICH builder belongs to a model is operator_assembly.hpp's job; producing
// the operators is the model's and operator/'s.
//
// It used to hold both: ~280 lines of construction -- the commutator velocity,
// the diagonal position, the model adapters and the assembly factories -- lived
// in this header alongside the cache. They are now in operator/ and in
// operator_assembly.hpp, and what is left matches the name.

#include "../utils/profile.hpp"
#include "../calc/preconditioner.hpp"
#include "../calc/eigen.hpp"
#include "../model/setup.hpp"
#include "../operator/builders.hpp"
#include "../operator/lattice_moment.hpp"
#include "../operator/types.hpp"
#include "kpoint_partition.hpp"
#include "spectral_bounds.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <complex>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace vbcsr;
using namespace vbcsr::atomic;

namespace rsatb {
namespace setup {

/// The MPI layout the k-points are distributed over.
struct OperatorRuntimeContext {
    int npar = 1;
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm master_comm = MPI_COMM_NULL;
};

using op::KPoint;
using op::PositionBuilder;
using op::VelocityBuilder;
using op::PositionAxisData;
using op::PositionTriplet;
using op::VelocityTriplet;
using op::LatticeMomentTriplet;

/// The pieces an OperatorSet is assembled from.
///
/// There is no capability flag here. Which operators a set holds is not a
/// separate fact to be declared and kept in step -- it is whether the artifact
/// is present, which is what every `has_*` below reports. The model states its
/// sources once (model/operator_source.hpp); the assembler turns them into
/// artifacts; the set answers from the artifacts.
///
/// In production this is filled from a Model by operator_assembly.hpp, which
/// is the only thing that reads a model's operator sources. It is a named type
/// rather than a hidden step because the set is worth testing on its own: a
/// test can hand it images it built by hand and a stub builder, and assert the
/// caching and the bounds without a model, a file, or op2c.
struct OperatorAssembly {
    std::unique_ptr<ImageContainer<double>> hamiltonian_images;
    std::unique_ptr<ImageContainer<double>> overlap_images;
    VelocityBuilder velocity_builder;
    PositionBuilder position_builder;
};

class OperatorSet {
public:
    /// Assembles a set from its parts and initializes it.
    ///
    /// The only way to make one, and a member rather than a free function so
    /// that nothing outside needs access to the internals -- the free version
    /// needed a `friend`, and the friend declaration was the only reason this
    /// header ever had to know the Model type at all.
    ///
    /// Reading a Model's operator sources and filling `parts` from them is
    /// setup/operator_assembly.hpp's job, which is what keeps the family
    /// knowledge in one file.
    static std::unique_ptr<OperatorSet> FromParts(
        vbcsr::atomic::AtomicData& atomic_data,
        MPI_Comm comm,
        bool verbose,
        OperatorAssembly parts,
        const rsatb::model::SetupOptions& options,
        const OperatorRuntimeContext& runtime
    ) {
        auto set = std::unique_ptr<OperatorSet>(new OperatorSet(atomic_data, comm, verbose));
        set->hamiltonian_images_ = std::move(parts.hamiltonian_images);
        set->overlap_images_ = std::move(parts.overlap_images);
        set->velocity_builder_ = std::move(parts.velocity_builder);
        set->position_builder_ = std::move(parts.position_builder);
        set->initialize(options, runtime);
        return set;
    }

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
        std::unique_ptr<LatticeMomentTriplet> first_lattice_moment;
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
            state.velocity = velocity_builder_(*atomic_data_, state.kpoint);
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
            state.position = position_builder_(*atomic_data_, state.kpoint);
        }
        return *state.position;
    }

    const LatticeMomentTriplet& first_lattice_moment_state(int k_idx) {
        auto& state = kpoint_state(k_idx);
        if (!state.first_lattice_moment) {
            if (!hamiltonian_images_) {
                throw std::runtime_error(
                    "No Hamiltonian image data is configured for first-lattice-moment construction.");
            }
            state.first_lattice_moment =
                op::BuildLatticeMoment(*atomic_data_, *hamiltonian_images_);
        }
        return *state.first_lattice_moment;
    }

private:
    OperatorSet(AtomicData& atomic_data, MPI_Comm comm, bool verbose)
        : atomic_data_(&atomic_data), comm_(comm), verbose_(verbose) {}

    void initialize(const rsatb::model::SetupOptions& options, const OperatorRuntimeContext& runtime) {
        local_kpoints_.clear();
        kpoint_states_.clear();
        bounds_ready_ = false;
        emin_ = -1.0;
        emax_ = 1.0;

        rsatb::setup::KPointPartitionContext partition_context;
        partition_context.npar = runtime.npar;
        partition_context.world_rank = runtime.world_rank;
        partition_context.world_size = runtime.world_size;
        const auto partition = rsatb::setup::PartitionKPoints(options.kpoints, partition_context);

        global_nk_ = partition.global_nk;
        local_kpoints_ = partition.local;

        for (const auto& entry : partition.local) {
            KPointOperatorState state;
            state.k_idx = entry.first;
            state.kpoint = entry.second;

            // Separate phases: together they were 43% of a multi-k setup, and
            // one merged row could not say whether H or S sampling carried it.
            if (hamiltonian_images_) {
                RSATB_PHASE("setup/hamiltonian_sample_k");
                state.hamiltonian.reset(hamiltonian_images_->sample_k(state.kpoint, PhaseConvention::R_AND_POSITION));
            }
            if (overlap_images_) {
                RSATB_PHASE("setup/overlap_sample_k");
                state.overlap.reset(overlap_images_->sample_k(state.kpoint, PhaseConvention::R_AND_POSITION));
            }
            if (state.overlap) {
                build_preconditioner(state, options.preconditioner);
            }

            kpoint_states_.emplace(state.k_idx, std::move(state));
        }

        resolve_spectral_bounds(options, runtime.master_comm);
    }

    /// Adopts explicit bounds, or estimates them from the built operators.
    void resolve_spectral_bounds(const rsatb::model::SetupOptions& options, MPI_Comm master_comm) {
        if (options.energy_bounds.has_value()) {
            const auto bounds = rsatb::setup::MakeExplicitBounds(
                options.energy_bounds->first, options.energy_bounds->second);
            emin_ = bounds.emin;
            emax_ = bounds.emax;
            bounds_ready_ = bounds.ready;
            return;
        }
        if (!options.compute_bounds || !has_hamiltonian()) {
            return;
        }

        std::vector<rsatb::setup::BoundsKPointView> views;
        views.reserve(kpoint_states_.size());
        for (const auto& entry : kpoint_states_) {
            const KPointOperatorState& state = entry.second;
            rsatb::setup::BoundsKPointView view;
            view.kpoint = state.kpoint;
            view.hamiltonian = state.hamiltonian.get();
            view.overlap = state.overlap.get();
            view.preconditioner = state.preconditioner.get();
            // Atom input indices key the solver's pinned start, so the
            // estimate no longer moves with the partition.
            view.stable_block_ids = atomic_data_->atom_index.data();
            views.push_back(std::move(view));
        }

        rsatb::setup::BoundsOptions bounds_options;
        bounds_options.compute = options.compute_bounds;
        bounds_options.zero_tol = options.bounds_zero_tol;
        bounds_options.max_matvecs = options.bounds_max_matvecs;

        rsatb::setup::BoundsComms comms;
        comms.master = master_comm;
        comms.sub = comm_;

        RSATB_PHASE("setup/spectral_bounds");
        const auto bounds =
            rsatb::setup::EstimateSpectralBounds(views, bounds_options, comms, verbose_);
        emin_ = bounds.emin;
        emax_ = bounds.emax;
        bounds_ready_ = bounds.ready;
    }

    void build_preconditioner(
        KPointOperatorState& state,
        const rsatb::model::PreconditionerOptions& config
    ) {
        Matrix* overlap = state.overlap.get();
        if (overlap == nullptr) {
            return;
        }
        if (config.target == rsatb::model::PreconditionerTarget::kNone) {
            return;
        }

        // Wraps the seed and refinement phases nested inside it, so this row
        // reports only the preconditioner's own overhead -- if it is ever large,
        // that is news.
        RSATB_PHASE("setup/preconditioner");
        state.preconditioner = std::make_unique<rsatb::calc::ISQRTMPreconditioner<Scalar>>(
            *overlap, config.maxiter, config.tol, config.sptol, verbose_);

        // The sandwich form propagates Z H Z and never touches S again;
        // releasing it here is what routes MakeChebyshev to that branch.
        // The solve form keeps S: matvecs CG-solve S y = H x instead.
        if (config.operator_form == rsatb::model::OperatorForm::kSandwich) {
            state.overlap.reset();
        }
    }

    AtomicData* atomic_data_ = nullptr;
    MPI_Comm comm_ = MPI_COMM_NULL;
    bool verbose_ = false;

    std::unique_ptr<ImageContainer<double>> hamiltonian_images_;
    std::unique_ptr<ImageContainer<double>> overlap_images_;
    VelocityBuilder velocity_builder_;
    PositionBuilder position_builder_;

    std::vector<std::pair<int, std::vector<double>>> local_kpoints_;
    std::map<int, KPointOperatorState> kpoint_states_;
    int global_nk_ = 0;
    bool bounds_ready_ = false;
    double emin_ = -1.0;
    double emax_ = 1.0;
};

}  // namespace setup
}  // namespace rsatb

#endif  // RSATB_SETUP_OPERATOR_SET_HPP
