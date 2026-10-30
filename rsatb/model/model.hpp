#ifndef RSATB_MODEL_MODEL_HPP
#define RSATB_MODEL_MODEL_HPP

// The interface every model family implements: LCAO, SKTB, DFTB.
//
// A Model answers two kinds of question. First, what basis does it work in --
// which is what lets RSATB build the atomic graph without the caller supplying
// orbital counts or cutoff radii. Second, how are its operators built -- H(R)
// and S(R) once, velocity and position lazily per k-point.
//
// Everything a family needs to distinguish itself lives behind this interface,
// so adding one means adding a subclass rather than another arm of a variant
// and another branch in the setup path.

#include "basis_spec.hpp"
#include "operator_source.hpp"

#include "../operator/images.hpp"
#include "../operator/types.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {

class Model {
public:
    using ImageContainer = vbcsr::atomic::ImageContainer<double>;
    using HamiltonianImages = op::HamiltonianImages;
    using AtomicData = vbcsr::atomic::AtomicData;
    using KPoint = op::KPoint;
    using VelocityTriplet = op::VelocityTriplet;
    using PositionTriplet = op::PositionTriplet;

    virtual ~Model() = default;

    /// Returns the family name used in diagnostics, e.g. "LCAO".
    virtual std::string name() const = 0;

    /// Returns the basis this model works in, ordered by ascending atomic number.
    virtual const BasisSpec& basis() const = 0;

    virtual const SpinSpec& spin() const = 0;

    /// Returns where each of this model's operators comes from.
    ///
    /// This is the whole of what distinguishes one family from another at the
    /// operator level, and it is a table rather than a set of scattered
    /// branches. `setup/operator_assembly.hpp` reads it and installs the
    /// matching builder; nothing else inspects a family by name.
    virtual OperatorSources operator_sources() const = 0;

    /// Returns which operators are available once this model is set up.
    ///
    /// Derived from `operator_sources()` rather than declared alongside it, so
    /// the two cannot disagree.
    Capabilities capabilities() const { return operator_sources().capabilities(); }

    /// Returns the radial orbital files backing the basis, or an empty list.
    ///
    /// Real-space quantities -- LDOS on a grid, and later the density -- need
    /// the actual radial functions, not just the orbital counts and cutoffs
    /// that BasisSpec carries. A model that has them returns one file per atom
    /// type in basis order; one that does not (a Slater-Koster or file-based
    /// tight-binding model, whose basis is an abstract label set) returns an
    /// empty list, and the real-space observables refuse rather than guess a
    /// radial shape.
    virtual std::vector<std::string> RealSpaceOrbitalFiles() const { return {}; }

    /// Builds the real-space Hamiltonian images H(R): real for a
    /// spin-degenerate basis, complex for a spinor one (op/images.hpp).
    virtual HamiltonianImages BuildHamiltonianImages(AtomicData& atomic_data) const = 0;

    /// Builds the real-space overlap images S(R).
    ///
    /// Returns nullptr if and only if `capabilities().overlap` is false, i.e.
    /// the basis is orthogonal.
    virtual std::unique_ptr<ImageContainer> BuildOverlapImages(
        AtomicData& atomic_data) const = 0;

    /// Builds the Cartesian velocity operators at one k-point.
    ///
    /// Called only when this model names itself as the velocity source. A
    /// model that names the generic route never sees this, so it does not
    /// implement it -- which is why both have a default that refuses.
    virtual std::unique_ptr<VelocityTriplet> BuildVelocity(
        AtomicData& atomic_data,
        const KPoint& kpoint) const {
        (void)atomic_data;
        (void)kpoint;
        throw std::runtime_error(
            name() + " does not build its own velocity operator; its source is " +
            ToString(operator_sources().velocity) + ".");
    }

    /// Builds the Cartesian position operators at one k-point.
    ///
    /// Called only when this model names itself as the position source.
    virtual std::unique_ptr<PositionTriplet> BuildPosition(
        AtomicData& atomic_data,
        const KPoint& kpoint) const {
        (void)atomic_data;
        (void)kpoint;
        throw std::runtime_error(
            name() + " does not build its own position operator; its source is " +
            ToString(operator_sources().position) + ".");
    }

protected:
    /// Rejects a k-point that is not a 3-vector.
    static void RequireThreeComponentKPoint(const KPoint& kpoint, const std::string& context) {
        if (kpoint.size() != 3) {
            throw std::runtime_error(context + " requires 3-component k-points.");
        }
    }
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_MODEL_HPP
