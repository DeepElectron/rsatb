#ifndef RSATB_MODEL_BASIS_SPEC_HPP
#define RSATB_MODEL_BASIS_SPEC_HPP

// What a model tells RSATB about the basis it works in.
//
// These three structs are the contract that removes the circular
// initialization: previously the caller had to supply `type_norb` and
// `r_max_per_type` to the RSATB constructor even though both are properties of
// the model, which only arrived later in setup(). A Model now reports its own
// BasisSpec, and RSATB builds the AtomicData from it.

#include "vbcsr/core/atomic/periodic_table.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {

/// How spin enters the basis.
struct SpinSpec {
    enum class Treatment {
        kNone,          // spin-degenerate, one component per orbital
        kCollinear,     // independent spin channels, one component each
        kNonCollinear,  // spinor basis, two components per orbital
        kSpinOrbit,     // spinor basis with spin-orbit coupling
    };

    Treatment treatment = Treatment::kNone;

    /// Returns the basis-size multiplier for one spatial orbital.
    int Components() const {
        switch (treatment) {
            case Treatment::kNone:
            case Treatment::kCollinear:
                return 1;
            case Treatment::kNonCollinear:
            case Treatment::kSpinOrbit:
                return 2;
        }
        throw std::runtime_error("Unhandled SpinSpec::Treatment.");
    }

    /// Returns the number of independent spin channels solved separately.
    int Channels() const {
        return treatment == Treatment::kCollinear ? 2 : 1;
    }

    bool IsSpinor() const { return Components() == 2; }
    bool HasSpinOrbit() const { return treatment == Treatment::kSpinOrbit; }
};

inline std::string ToString(SpinSpec::Treatment treatment) {
    switch (treatment) {
        case SpinSpec::Treatment::kNone: return "none";
        case SpinSpec::Treatment::kCollinear: return "collinear";
        case SpinSpec::Treatment::kNonCollinear: return "noncollinear";
        case SpinSpec::Treatment::kSpinOrbit: return "spinorbit";
    }
    throw std::runtime_error("Unhandled SpinSpec::Treatment.");
}

/// Parses a user-facing spin treatment name.
inline SpinSpec ParseSpinSpec(const std::string& raw_name) {
    const std::string name = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_name));
    if (name.empty() || name == "none") return {SpinSpec::Treatment::kNone};
    if (name == "collinear") return {SpinSpec::Treatment::kCollinear};
    if (name == "noncollinear" || name == "noncolin") return {SpinSpec::Treatment::kNonCollinear};
    if (name == "spinorbit" || name == "soc" || name == "lspinorb") {
        return {SpinSpec::Treatment::kSpinOrbit};
    }
    throw std::runtime_error(
        "Unsupported spin treatment '" + raw_name +
        "'. Supported: none, collinear, noncollinear, spinorbit.");
}

/// Everything AtomicData needs in order to build the atomic graph.
///
/// Produced by the Model, never typed by the user. `type_symbols` must be
/// ordered by ascending atomic number, because that is how AtomicData assigns
/// atom type indices; `Validate` enforces it.
struct BasisSpec {
    std::vector<std::string> type_symbols;   // index == atom type
    std::vector<int> type_norb;              // orbitals per type, spin-expanded
    std::vector<double> cutoff_radius;       // Angstrom; edge iff d <= r_i + r_j

    size_t n_type() const { return type_symbols.size(); }

    /// Checks internal consistency and the AtomicData type ordering.
    ///
    /// Args:
    ///   context: Prefix used in error messages, e.g. the model name.
    void Validate(const std::string& context) const {
        if (type_symbols.empty()) {
            throw std::runtime_error(context + ": basis declares no atom types.");
        }
        if (type_norb.size() != type_symbols.size() ||
            cutoff_radius.size() != type_symbols.size()) {
            throw std::runtime_error(
                context + ": basis arrays have inconsistent lengths (" +
                std::to_string(type_symbols.size()) + " symbols, " +
                std::to_string(type_norb.size()) + " orbital counts, " +
                std::to_string(cutoff_radius.size()) + " cutoffs).");
        }

        int previous_z = 0;
        for (size_t type = 0; type < type_symbols.size(); ++type) {
            const int z = vbcsr::atomic::AtomicNumber(type_symbols[type]);
            if (z <= previous_z) {
                throw std::runtime_error(
                    context + ": basis type symbols must be ordered by ascending atomic number "
                    "to match AtomicData's type indexing, but " + type_symbols[type] +
                    " follows " + type_symbols[type - 1] + ".");
            }
            previous_z = z;

            if (type_norb[type] <= 0) {
                throw std::runtime_error(
                    context + ": species " + type_symbols[type] + " has no basis orbitals.");
            }
            if (!(cutoff_radius[type] > 0.0)) {
                throw std::runtime_error(
                    context + ": species " + type_symbols[type] +
                    " has a non-positive cutoff radius.");
            }
        }
    }

    /// Checks that this basis covers every species present in a structure.
    ///
    /// Args:
    ///   structure_symbols: Distinct species of the structure, ascending Z.
    ///   context: Prefix used in error messages.
    void RequireCovers(
        const std::vector<std::string>& structure_symbols,
        const std::string& context
    ) const {
        if (structure_symbols == type_symbols) {
            return;
        }
        std::string expected;
        for (const auto& symbol : structure_symbols) {
            if (!expected.empty()) expected += ", ";
            expected += symbol;
        }
        std::string actual;
        for (const auto& symbol : type_symbols) {
            if (!actual.empty()) actual += ", ";
            actual += symbol;
        }
        throw std::runtime_error(
            context + ": the structure contains species [" + expected +
            "] but the model basis defines [" + actual + "].");
    }
};

/// Which operators a model can produce.
///
/// Declared by the model rather than inferred from whether some configuration
/// string happens to be non-empty, so a missing input is a setup error instead
/// of a silently different calculation.
struct Capabilities {
    bool overlap = false;   // false means the basis is orthogonal
    bool velocity = false;
    bool position = false;

    bool orthogonal() const { return !overlap; }
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_BASIS_SPEC_HPP
