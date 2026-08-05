#ifndef RSATB_OPERATOR_BUILDERS_HPP
#define RSATB_OPERATOR_BUILDERS_HPP

// The generic operator builders: what an orthogonal basis gets for free.
//
// A builder is a `std::function` (operator/types.hpp), so these are factories
// rather than classes -- each captures the one thing its construction needs and
// forwards to the free function that does the work. Which builder a model gets
// is decided in setup/operator_assembly.hpp from the source table the model
// declares; nothing is guessed here.
//
// Both are valid EXACTLY when the basis is orthogonal, which is the condition
// OperatorSources::Validate enforces once, at model construction.

#include "commutator_velocity.hpp"
#include "diagonal_position.hpp"
#include "types.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"

#include <stdexcept>

namespace rsatb {
namespace op {

/// v = (i/hbar) [H, r], from images the caller keeps alive.
inline VelocityBuilder MakeCommutatorVelocityBuilder(
    const vbcsr::atomic::ImageContainer<double>* hamiltonian_images
) {
    if (hamiltonian_images == nullptr) {
        throw std::runtime_error("Commutator velocity construction requires Hamiltonian images.");
    }
    return [hamiltonian_images](vbcsr::atomic::AtomicData& atomic_data, const KPoint& kpoint) {
        return BuildCommutatorVelocity(atomic_data, *hamiltonian_images, kpoint);
    };
}

/// r as a diagonal over atom coordinates.
///
/// Independent of k, so the k-point is accepted and ignored; the operator set
/// caches per k-point and this returns the same operator each time.
inline PositionBuilder MakeDiagonalPositionBuilder() {
    return [](vbcsr::atomic::AtomicData& atomic_data, const KPoint&) {
        return BuildDiagonalPosition(atomic_data);
    };
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_BUILDERS_HPP
