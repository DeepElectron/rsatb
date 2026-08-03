#ifndef RSATB_SETUP_OPERATOR_ASSEMBLY_HPP
#define RSATB_SETUP_OPERATOR_ASSEMBLY_HPP

// Turning a Model into an OperatorSet.
//
// This is the ONE place that reads a model's operator sources and installs the
// matching builders, and the only place in the tree where the two routes meet.
// It does not know any family by name: it reads the table each model declares
// (model/operator_source.hpp) and wires it up. Adding a family therefore means
// adding a model file, not another arm here.
//
// It replaces BuildModelArtifacts -> ModelArtifacts -> finalize_operator_set ->
// build_operator_set. What remains is this function and OperatorSet::FromParts,
// which is a member: "assemble a set from parts" belongs to the set, and
// "read a model's table" belongs here. That split is why this header depends on
// model/ and operator_set.hpp does not.

#include "operator_set.hpp"

#include "../model/model.hpp"
#include "../model/operator_source.hpp"
#include "../operator/builders.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace rsatb {
namespace setup {

/// Assembles the operators a Model describes.
///
/// The real-space images are built now, because everything downstream needs
/// them; velocity and position are deferred to the builders, because most runs
/// never ask for either.
///
/// Args:
///   atomic_data: The graph every operator is indexed against.
///   comm: The sub-communicator this rank's k-points live on.
///   verbose: Whether setup reports progress.
///   model: Supplies the operators and names their sources.
///   options: k-points, spectral bounds and preconditioner settings.
///   runtime: The MPI layout the k-points are distributed over.
/// Returns:
///   An initialized OperatorSet.
/// Throws:
///   std::runtime_error if the model's sources are inconsistent, or if it
///   declares an operator it then fails to produce.
inline std::unique_ptr<OperatorSet> BuildOperatorSet(
    vbcsr::atomic::AtomicData& atomic_data,
    MPI_Comm comm,
    bool verbose,
    const rsatb::model::Model& model,
    const rsatb::model::SetupOptions& options,
    const OperatorRuntimeContext& runtime
) {
    using rsatb::model::OperatorSource;

    const auto sources = model.operator_sources();
    sources.Validate(model.name());

    OperatorAssembly parts;

    // --- H(R): always the model, always now.
    parts.hamiltonian_images = model.BuildHamiltonianImages(atomic_data);
    if (!parts.hamiltonian_images) {
        throw std::runtime_error(model.name() + " model produced no Hamiltonian images.");
    }

    // --- S(R): the model, or the basis is orthogonal.
    if (sources.overlap == OperatorSource::kModel) {
        parts.overlap_images = model.BuildOverlapImages(atomic_data);
        if (!parts.overlap_images) {
            throw std::runtime_error(
                model.name() + " declares a non-orthogonal basis but produced no overlap images.");
        }
    }

    // --- v and r: the model's integrals, the generic route, or absent.
    switch (sources.velocity) {
        case OperatorSource::kModel:
            // The model's own integrals. A lambda over the model rather than
            // an adapter class: both are one captured pointer and one call.
            parts.velocity_builder = [&model](vbcsr::atomic::AtomicData& atomic_data,
                                              const op::KPoint& kpoint) {
                return model.BuildVelocity(atomic_data, kpoint);
            };
            break;
        case OperatorSource::kGeneric:
            parts.velocity_builder = op::MakeCommutatorVelocityBuilder(
                parts.hamiltonian_images.get());
            break;
        case OperatorSource::kUnavailable:
            break;
    }

    switch (sources.position) {
        case OperatorSource::kModel:
            parts.position_builder = [&model](vbcsr::atomic::AtomicData& atomic_data,
                                              const op::KPoint& kpoint) {
                return model.BuildPosition(atomic_data, kpoint);
            };
            break;
        case OperatorSource::kGeneric:
            parts.position_builder = op::MakeDiagonalPositionBuilder();
            break;
        case OperatorSource::kUnavailable:
            break;
    }

    return OperatorSet::FromParts(
        atomic_data, comm, verbose, std::move(parts), options, runtime);
}

}  // namespace setup
}  // namespace rsatb

#endif  // RSATB_SETUP_OPERATOR_ASSEMBLY_HPP
