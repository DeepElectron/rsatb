#ifndef RSATB_MODEL_OPERATOR_SOURCE_HPP
#define RSATB_MODEL_OPERATOR_SOURCE_HPP

// Where each of a model's operators comes from.
//
// This is the one statement that distinguishes the families from one another
// at the operator level, and every model makes it in its own file:
//
//   family              H              S              v              r
//   LCAO                file           op2c           generic        op2c
//   SKTB (orthogonal)   SK tables      --             generic        generic
//   SKTB (+ overlap)    SK tables      SK tables      --             --
//   DFTB                SK tables      SK tables      --             --
//   tb_file             file           --             generic        generic
//
// "generic" velocity means the commutator i[H, x] built in operator/ -- the
// bare bond moment when the basis is orthogonal, the section-4 overlap-dressed
// composite when it is not. The composite needs the model's position matrix
// (the dipole integrals), which is why a non-orthogonal family without a
// position stays velocity-less: doc/position_commutator.md section 7 measures
// that the dipole term cannot be dropped. op2c's explicit-integral velocity
// remains available on the model itself as a cross-check, but it is not what
// the operator set serves: the commutator is the convention every spectral
// consumer propagates.
//
// It replaces a `bool ProvidesOperatorBuilders()` that selected between two
// whole implementations of velocity and position without naming either, and
// that had to agree by hand with a separately written Capabilities. Here the
// capabilities are DERIVED, so the two cannot disagree.

#include "basis_spec.hpp"

#include <stdexcept>
#include <string>

namespace rsatb {
namespace model {

/// Where one operator comes from.
enum class OperatorSource {
    /// This model cannot supply it. An observable that needs it refuses, by name.
    kUnavailable,
    /// The model's own integrals -- op2c for LCAO, the SK tables for SKTB/DFTB.
    kModel,
    /// Built in operator/ from the model's images and geometry: the commutator
    /// velocity (plus the model's position matrix when an overlap dresses it)
    /// and the diagonal position. The generic POSITION is correct only in an
    /// orthogonal basis; the generic VELOCITY is correct whenever its dipole
    /// ingredient exists.
    kGeneric,
};

inline const char* ToString(OperatorSource source) {
    switch (source) {
        case OperatorSource::kUnavailable: return "unavailable";
        case OperatorSource::kModel:       return "model";
        case OperatorSource::kGeneric:     return "generic";
    }
    return "unknown";
}

/// The source of every operator, stated once per family.
struct OperatorSources {
    /// Always the model: a model that cannot produce H is not a model.
    OperatorSource hamiltonian = OperatorSource::kModel;
    OperatorSource overlap = OperatorSource::kUnavailable;
    OperatorSource velocity = OperatorSource::kUnavailable;
    OperatorSource position = OperatorSource::kUnavailable;

    /// Returns which operators are available, derived rather than declared.
    Capabilities capabilities() const {
        return Capabilities{
            /*overlap=*/overlap != OperatorSource::kUnavailable,
            /*velocity=*/velocity != OperatorSource::kUnavailable,
            /*position=*/position != OperatorSource::kUnavailable,
        };
    }

    /// Rejects a combination that cannot be built.
    ///
    /// One rule, checked once here instead of assumed at every call site: the
    /// generic route derives an operator from H(R) and the atom coordinates,
    /// which is the velocity -- and the position -- only when the basis is
    /// orthogonal. Naming it alongside an overlap is a contradiction.
    ///
    /// Args:
    ///   family: Family name, used in the message.
    /// Throws:
    ///   std::runtime_error naming the family and the offending operator.
    void Validate(const std::string& family) const {
        if (hamiltonian != OperatorSource::kModel) {
            throw std::runtime_error(
                family + " must build its own Hamiltonian; no other source exists.");
        }
        if (overlap == OperatorSource::kGeneric) {
            throw std::runtime_error(
                family + " names a generic overlap, which does not exist: an overlap "
                "comes from the model's integrals or the basis is orthogonal.");
        }
        const bool non_orthogonal = overlap != OperatorSource::kUnavailable;
        if (!non_orthogonal) {
            return;
        }
        if (position == OperatorSource::kGeneric) {
            throw std::runtime_error(
                family + " names the generic position together with an overlap. The "
                "diagonal position is correct only in an orthogonal basis; a "
                "non-orthogonal model must supply its own.");
        }
        if (velocity == OperatorSource::kGeneric && position == OperatorSource::kUnavailable) {
            throw std::runtime_error(
                family + " names the generic velocity together with an overlap but "
                "supplies no position. The overlap-dressed commutator needs the "
                "model's dipole matrix; without it there is no velocity -- the dipole "
                "term is not droppable (doc/position_commutator.md section 7).");
        }
    }
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_OPERATOR_SOURCE_HPP
