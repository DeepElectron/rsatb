#ifndef RSATB_MODEL_SKTB_SK_TABLE_HPP
#define RSATB_MODEL_SKTB_SK_TABLE_HPP

// Slater-Koster parameters: what the user writes, and what construction reads.
//
// Input side: `SkParameters` holds the onsite energies and pair channels for
// ONE matrix. The Hamiltonian and the overlap are separate SkParameters
// objects, because they are separate physics -- there is no combined "H and S
// in one dict" form to disambiguate, and so no shape-guessing anywhere.
//
// Resolved side: `SkTable` keys everything by atom type index rather than by
// species symbol, which is what the construction inner loops need.

#include "orbital_labels.hpp"
#include "radial.hpp"
#include "../basis_spec.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rsatb {
namespace model {
namespace sktb {

/// Radial channels for one ordered species pair, keyed by channel name.
struct SkPair {
    std::string row_species;
    std::string col_species;
    std::map<std::string, SkRadialFunction> channels;

    /// Adds one channel using the public shell-pair / bond-type vocabulary.
    ///
    /// Args:
    ///   shell_pair: e.g. "pp".
    ///   bond_type: one of "sigma", "pi", "delta".
    ///   radial: Distance dependence for that channel.
    SkPair& SetChannel(
        const std::string& shell_pair,
        const std::string& bond_type,
        const SkRadialFunction& radial
    ) {
        channels[MakeChannelName(shell_pair, bond_type)] = radial;
        return *this;
    }
};

/// Onsite energies and pair channels for one matrix (Hamiltonian or overlap).
struct SkParameters {
    /// species symbol -> orbital or shell label -> value
    std::map<std::string, std::map<std::string, double>> onsite;
    std::vector<SkPair> pairs;

    bool IsEnabled() const {
        return !onsite.empty() || !pairs.empty();
    }

    SkParameters& SetOnsite(
        const std::string& species,
        const std::string& orbital,
        double value
    ) {
        onsite[vbcsr::atomic::CanonicalSymbol(species)][CanonicalOrbitalLabel(orbital)] = value;
        return *this;
    }

    /// Returns the editable entry for one ordered species pair, creating it if needed.
    SkPair& Pair(const std::string& row_species, const std::string& col_species) {
        const std::string row_symbol = vbcsr::atomic::CanonicalSymbol(row_species);
        const std::string col_symbol = vbcsr::atomic::CanonicalSymbol(col_species);
        for (auto& entry : pairs) {
            if (vbcsr::atomic::CanonicalSymbol(entry.row_species) == row_symbol &&
                vbcsr::atomic::CanonicalSymbol(entry.col_species) == col_symbol) {
                return entry;
            }
        }
        SkPair entry;
        entry.row_species = row_symbol;
        entry.col_species = col_symbol;
        pairs.push_back(std::move(entry));
        return pairs.back();
    }

    SkParameters& SetPairChannel(
        const std::string& row_species,
        const std::string& col_species,
        const std::string& shell_pair,
        const std::string& bond_type,
        const SkRadialFunction& radial
    ) {
        Pair(row_species, col_species).SetChannel(shell_pair, bond_type, radial);
        return *this;
    }
};

/// Basis and onsite data for one atom type.
struct SkAtomType {
    std::string symbol;
    std::vector<SkOrbital> orbitals;
    std::map<std::string, int> orbital_index;
    std::vector<double> h_onsite;
    std::vector<double> s_onsite;
};

/// One resolved sigma/pi/delta channel for an ordered shell pair.
struct SkPairChannel {
    int row_l = 0;
    int col_l = 0;
    int channel = 0;
    SkRadialFunction h;
    SkRadialFunction s;
    bool has_h = false;
    bool has_s = false;
};

/// All radial channels for one ordered atom-type pair.
struct SkAtomPair {
    int row_type = 0;
    int col_type = 0;
    std::vector<SkPairChannel> channels;
    double h_max_cutoff = 0.0;
    double s_max_cutoff = 0.0;

    double MaxCutoff() const {
        return std::max(h_max_cutoff, s_max_cutoff);
    }
};

/// Resolved Slater-Koster parameters ready for operator construction.
struct SkTable {
    std::vector<SkAtomType> atom_types;
    std::vector<SkAtomPair> atom_pairs;
    std::map<std::pair<int, int>, size_t> pair_index_by_type;
    bool has_overlap = false;

    bool IsEnabled() const {
        return !atom_types.empty();
    }

    /// Returns true when no explicit overlap matrix is parameterized.
    bool orthogonal() const {
        return !has_overlap;
    }

    const SkAtomPair* FindPair(int row_type, int col_type) const {
        auto it = pair_index_by_type.find(std::make_pair(row_type, col_type));
        return (it == pair_index_by_type.end()) ? nullptr : &atom_pairs[it->second];
    }

    SkAtomPair& Pair(int row_type, int col_type) {
        const auto key = std::make_pair(row_type, col_type);
        auto it = pair_index_by_type.find(key);
        if (it != pair_index_by_type.end()) {
            return atom_pairs[it->second];
        }

        SkAtomPair pair_entry;
        pair_entry.row_type = row_type;
        pair_entry.col_type = col_type;
        atom_pairs.push_back(std::move(pair_entry));
        const size_t index = atom_pairs.size() - 1;
        pair_index_by_type.emplace(key, index);
        return atom_pairs.back();
    }
};

/// Converts pair/bond cutoffs into AtomicData atom-wise cutoff radii.
///
/// AtomicData builds a graph edge for atom types `(i, j)` when
/// `distance <= r_i + r_j`, while SK radial functions carry a direct bond
/// cutoff. The radius below is half of the largest bond cutoff touching each
/// type, which is tight for single-species models and guarantees every SK pair
/// cutoff is covered.
inline std::vector<double> DeriveAtomwiseCutoffRadii(const SkTable& table) {
    std::vector<double> radii(table.atom_types.size(), 0.0);
    for (const auto& pair : table.atom_pairs) {
        const double radius = 0.5 * pair.MaxCutoff();
        radii[pair.row_type] = std::max(radii[pair.row_type], radius);
        radii[pair.col_type] = std::max(radii[pair.col_type], radius);
    }
    return radii;
}

/// Expands shell aliases such as `p` and `d` into an atom-type basis.
inline SkAtomType BuildAtomTypeBasis(
    const std::vector<std::string>& raw_labels,
    const std::string& species_symbol
) {
    SkAtomType atom_type;
    atom_type.symbol = species_symbol;
    for (const auto& raw_label : raw_labels) {
        for (const auto& label : ExpandBasisLabel(raw_label)) {
            if (atom_type.orbital_index.count(label) != 0) {
                throw std::runtime_error(
                    "SKTB basis for species " + species_symbol + " contains a duplicate orbital: " + label);
            }
            const int index = static_cast<int>(atom_type.orbitals.size());
            atom_type.orbitals.push_back(MakeOrbital(label));
            atom_type.orbital_index.emplace(label, index);
        }
    }
    if (atom_type.orbitals.empty()) {
        throw std::runtime_error("SKTB basis for species " + species_symbol + " must not be empty.");
    }
    return atom_type;
}

/// Expands onsite shell/orbital entries into one diagonal value per basis orbital.
inline std::vector<double> ExpandOnsiteDiagonal(
    const std::map<std::string, double>& raw_entries,
    const SkAtomType& atom_type,
    bool require_complete,
    bool default_identity
) {
    std::vector<double> diagonal(atom_type.orbitals.size(), default_identity ? 1.0 : 0.0);
    std::vector<bool> assigned(atom_type.orbitals.size(), default_identity);

    for (const auto& entry : raw_entries) {
        for (const auto& label : ExpandBasisLabel(entry.first)) {
            auto it = atom_type.orbital_index.find(label);
            if (it == atom_type.orbital_index.end()) {
                throw std::runtime_error(
                    "SKTB onsite specification for species " + atom_type.symbol +
                    " references orbital " + label + " which is not in the declared basis.");
            }
            if (assigned[it->second]) {
                throw std::runtime_error(
                    "SKTB onsite specification for species " + atom_type.symbol +
                    " assigns orbital " + label + " more than once.");
            }
            diagonal[it->second] = entry.second;
            assigned[it->second] = true;
        }
    }

    if (require_complete) {
        for (size_t i = 0; i < assigned.size(); ++i) {
            if (!assigned[i]) {
                throw std::runtime_error(
                    "SKTB onsite specification for species " + atom_type.symbol +
                    " is missing orbital " + atom_type.orbitals[i].label + ".");
            }
        }
    }

    return diagonal;
}

/// Canonicalizes and validates a declared species order.
inline std::vector<std::string> ResolveTypeSymbols(const std::vector<std::string>& raw_symbols) {
    std::vector<std::string> symbols;
    symbols.reserve(raw_symbols.size());
    for (const auto& raw_symbol : raw_symbols) {
        symbols.push_back(vbcsr::atomic::CanonicalSymbol(raw_symbol));
    }
    if (symbols.empty()) {
        throw std::runtime_error("SKTB setup requires at least one species.");
    }

    std::map<std::string, int> counts;
    for (const auto& symbol : symbols) {
        if (++counts[symbol] > 1) {
            throw std::runtime_error("SKTB species must not contain duplicates: " + symbol + ".");
        }
    }
    return symbols;
}

/// Resolves declared parameters into a table keyed by atom type index.
///
/// The species order is derived from the declared basis rather than supplied
/// separately: `AtomicData` indexes atom types by ascending atomic number, and
/// sorting here is what keeps the two in agreement.
///
/// Args:
///   basis: species symbol -> basis labels, e.g. {"C": {"pz"}}.
///   hamiltonian: Onsite energies and pair channels for H.
///   overlap: Same for S; absent means an orthogonal basis.
/// Returns:
///   The resolved table.
inline SkTable BuildSkTable(
    const std::map<std::string, std::vector<std::string>>& basis,
    const SkParameters& hamiltonian,
    const std::optional<SkParameters>& overlap
) {
    if (basis.empty()) {
        throw std::runtime_error("SKTB setup requires a basis definition.");
    }
    if (!hamiltonian.IsEnabled()) {
        throw std::runtime_error("SKTB setup requires Hamiltonian parameters.");
    }

    std::vector<std::string> declared_symbols;
    declared_symbols.reserve(basis.size());
    for (const auto& entry : basis) {
        declared_symbols.push_back(entry.first);
    }
    // Ascending atomic number, to match AtomicData's atom type indexing.
    std::sort(declared_symbols.begin(), declared_symbols.end(),
              [](const std::string& lhs, const std::string& rhs) {
                  return vbcsr::atomic::AtomicNumber(lhs) < vbcsr::atomic::AtomicNumber(rhs);
              });
    const auto resolved_symbols = ResolveTypeSymbols(declared_symbols);

    SkTable table;
    const int n_types = static_cast<int>(resolved_symbols.size());
    table.atom_types.resize(n_types);
    table.has_overlap = overlap.has_value() && overlap->IsEnabled();

    std::map<std::string, int> symbol_to_type;
    for (int type = 0; type < n_types; ++type) {
        symbol_to_type.emplace(resolved_symbols[type], type);
    }

    for (int type = 0; type < n_types; ++type) {
        const std::string& symbol = resolved_symbols[type];
        table.atom_types[type] = BuildAtomTypeBasis(basis.at(declared_symbols[type]), symbol);

        auto h_onsite = hamiltonian.onsite.find(symbol);
        if (h_onsite == hamiltonian.onsite.end()) {
            throw std::runtime_error("SKTB Hamiltonian onsite terms are missing for species " + symbol + ".");
        }
        table.atom_types[type].h_onsite =
            ExpandOnsiteDiagonal(h_onsite->second, table.atom_types[type], true, false);

        if (table.has_overlap) {
            auto s_onsite = overlap->onsite.find(symbol);
            if (s_onsite == overlap->onsite.end()) {
                // An unspecified overlap diagonal is the identity, which is the
                // normal convention for a normalized basis.
                table.atom_types[type].s_onsite =
                    ExpandOnsiteDiagonal({}, table.atom_types[type], false, true);
            } else {
                table.atom_types[type].s_onsite =
                    ExpandOnsiteDiagonal(s_onsite->second, table.atom_types[type], true, false);
            }
        }
    }

    auto resolve_pairs = [&](const SkParameters& parameters, bool for_overlap, const char* matrix_name) {
        for (const auto& pair_cfg : parameters.pairs) {
            const std::string row_symbol = vbcsr::atomic::CanonicalSymbol(pair_cfg.row_species);
            const std::string col_symbol = vbcsr::atomic::CanonicalSymbol(pair_cfg.col_species);

            auto row_it = symbol_to_type.find(row_symbol);
            auto col_it = symbol_to_type.find(col_symbol);
            if (row_it == symbol_to_type.end() || col_it == symbol_to_type.end()) {
                throw std::runtime_error(
                    std::string("SKTB ") + matrix_name + " pair (" + row_symbol + ", " + col_symbol +
                    ") names a species that is not in the declared basis.");
            }

            auto& pair = table.Pair(row_it->second, col_it->second);

            for (const auto& channel_entry : pair_cfg.channels) {
                SkChannelName channel_name;
                try {
                    channel_name = ParseChannelName(channel_entry.first);
                } catch (const std::runtime_error&) {
                    throw std::runtime_error(
                        std::string("SKTB ") + matrix_name + " pair (" + row_symbol + ", " + col_symbol +
                        ") uses unsupported channel '" + channel_entry.first + "'.");
                }
                channel_entry.second.Validate(
                    std::string("SKTB ") + matrix_name + " pair (" + row_symbol + ", " + col_symbol +
                    "), channel " + channel_name.name);

                auto radial = channel_entry.second;
                if (radial.NormalizedKind() == "spline" && radial.spline_second_derivatives.empty()) {
                    radial.spline_second_derivatives = radial.BuildSplineSecondDerivatives();
                }

                auto add_channel = [&](int row_l, int col_l) {
                    auto it = std::find_if(
                        pair.channels.begin(), pair.channels.end(),
                        [&](const SkPairChannel& channel) {
                            return channel.row_l == row_l &&
                                   channel.col_l == col_l &&
                                   channel.channel == channel_name.channel;
                        });
                    if (it == pair.channels.end()) {
                        SkPairChannel channel;
                        channel.row_l = row_l;
                        channel.col_l = col_l;
                        channel.channel = channel_name.channel;
                        pair.channels.push_back(std::move(channel));
                        it = pair.channels.end() - 1;
                    }

                    bool& has_value = for_overlap ? it->has_s : it->has_h;
                    if (has_value) {
                        throw std::runtime_error(
                            std::string("SKTB ") + matrix_name + " pair (" + row_symbol + ", " + col_symbol +
                            ") assigns channel '" + channel_name.name + "' more than once.");
                    }

                    if (for_overlap) {
                        it->s = radial;
                        pair.s_max_cutoff = std::max(pair.s_max_cutoff, radial.Cutoff());
                    } else {
                        it->h = radial;
                        pair.h_max_cutoff = std::max(pair.h_max_cutoff, radial.Cutoff());
                    }
                    has_value = true;
                };

                add_channel(channel_name.row_l, channel_name.col_l);
                if (channel_name.row_l != channel_name.col_l) {
                    add_channel(channel_name.col_l, channel_name.row_l);
                }
            }
        }
    };

    resolve_pairs(hamiltonian, false, "Hamiltonian");
    if (table.has_overlap) {
        resolve_pairs(*overlap, true, "overlap");
    }

    return table;
}

/// Derives the AtomicData basis requirements from a resolved table.
inline BasisSpec DeriveBasisSpec(const SkTable& table, const SpinSpec& spin) {
    BasisSpec basis;
    basis.type_symbols.reserve(table.atom_types.size());
    basis.type_norb.reserve(table.atom_types.size());
    for (const auto& atom_type : table.atom_types) {
        basis.type_symbols.push_back(atom_type.symbol);
        basis.type_norb.push_back(
            static_cast<int>(atom_type.orbitals.size()) * spin.Components());
    }
    basis.cutoff_radius = DeriveAtomwiseCutoffRadii(table);
    return basis;
}

}  // namespace sktb
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_SKTB_SK_TABLE_HPP
