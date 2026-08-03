#ifndef RSATB_MODEL_OPERATOR_SOURCE_HPP
#define RSATB_MODEL_OPERATOR_SOURCE_HPP

// Where each of a model's operators comes from.
//
// This is the one statement that distinguishes the families from one another
// at the operator level, and every model makes it in its own file:
//
//   family              H              S              v              r
//   LCAO                file           op2c           op2c           op2c
//   SKTB (orthogonal)   SK tables      --             generic        generic
//   SKTB (+ overlap)    SK tables      SK tables      --             --
//   DFTB                SK tables      SK tables      --             --
//   tb_file             file           --             generic        generic
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
    /// Built from H(R) and the geometry alone: the commutator velocity and the
    /// diagonal position in operator/. Correct only in an orthogonal basis.
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
        for (const auto& entry : {std::make_pair("velocity", velocity),
                                  std::make_pair("position", position)}) {
            if (entry.second == OperatorSource::kGeneric) {
                throw std::runtime_error(
                    family + " names the generic " + entry.first + " together with an "
                    "overlap. The generic route is built from H(R) and the atom "
                    "coordinates, which is correct only in an orthogonal basis; a "
                    "non-orthogonal model must supply its own.");
            }
        }
    }
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_OPERATOR_SOURCE_HPP
