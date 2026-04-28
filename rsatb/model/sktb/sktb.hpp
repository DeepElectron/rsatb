#ifndef RSATB_BACKEND_SKTB_HPP
#define RSATB_BACKEND_SKTB_HPP

#include "../io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rsatb {
namespace backend {

inline std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline std::string canonical_species_symbol(const std::string& raw_symbol) {
    const std::string trimmed = trim_copy(raw_symbol);
    if (trimmed.empty()) {
        throw std::runtime_error("SKTB species symbol must not be empty.");
    }
    // Normalize an accepted species token back to a canonical chemical symbol.

    const int z = io::get_atomic_number(trimmed);
    static const std::vector<std::string> symbols = {
        "",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
        "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
        "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
        "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
        "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm"
    };
    if (z <= 0 || z >= static_cast<int>(symbols.size()) || symbols[z].empty()) {
        throw std::runtime_error("Unsupported SKTB species symbol: " + raw_symbol);
    }
    return symbols[z];
}

inline std::vector<std::string> unique_type_symbols_from_atomic_numbers(const std::vector<int>& z) {
    std::vector<int> unique_z = z;
    std::sort(unique_z.begin(), unique_z.end());
    unique_z.erase(std::unique(unique_z.begin(), unique_z.end()), unique_z.end());

    std::vector<std::string> symbols;
    symbols.reserve(unique_z.size());
    for (int value : unique_z) {
        symbols.push_back(canonical_species_symbol(std::to_string(value)));
    }
    return symbols;
}

inline std::string canonical_orbital_label(const std::string& raw_label) {
    std::string label = lower_copy(trim_copy(raw_label));
    if (label == "dxz") {
        label = "dzx";
    }
    if (label == "dx2y2") {
        label = "dx2-y2";
    }
    return label;
}

inline bool is_supported_explicit_orbital(const std::string& label) {
    return label == "s" ||
           label == "px" || label == "py" || label == "pz" ||
           label == "dxy" || label == "dyz" || label == "dzx" ||
           label == "dx2-y2" || label == "dz2";
}

inline std::vector<std::string> expand_basis_label(const std::string& raw_label) {
    const std::string label = canonical_orbital_label(raw_label);
    if (label == "s") {
        return {"s"};
    }
    if (label == "p") {
        return {"py", "pz", "px"};
    }
    if (label == "d") {
        return {"dxy", "dyz", "dz2", "dzx", "dx2-y2"};
    }
    if (is_supported_explicit_orbital(label)) {
        return {label};
    }
    throw std::runtime_error("Unsupported SKTB orbital label: " + raw_label);
}

/// One expanded basis orbital in DeePTB magnetic order.
struct SKOrbital {
    std::string label;
    int l = 0;
    int m_index = 0;
};

inline SKOrbital make_orbital(const std::string& raw_label) {
    const std::string label = canonical_orbital_label(raw_label);
    if (label == "s") return {label, 0, 0};
    if (label == "py") return {label, 1, 0};
    if (label == "pz") return {label, 1, 1};
    if (label == "px") return {label, 1, 2};
    if (label == "dxy") return {label, 2, 0};
    if (label == "dyz") return {label, 2, 1};
    if (label == "dz2") return {label, 2, 2};
    if (label == "dzx") return {label, 2, 3};
    if (label == "dx2-y2") return {label, 2, 4};
    throw std::runtime_error("Unsupported SKTB orbital label: " + raw_label);
}

inline int shell_from_channel_char(char value) {
    if (value == 's') return 0;
    if (value == 'p') return 1;
    if (value == 'd') return 2;
    throw std::runtime_error("Unsupported SKTB channel shell.");
}

inline int sk_channel_index(const std::string& kind) {
    if (kind == "sigma") return 0;
    if (kind == "pi") return 1;
    if (kind == "delta") return 2;
    throw std::runtime_error("Unsupported SKTB channel kind: " + kind);
}

/// Parsed shell pair and SK channel from names like `pd_pi`.
struct SKChannelName {
    std::string name;
    int row_l = 0;
    int col_l = 0;
    int channel = 0;
};

/// Parses an SK channel name into shell-pair indices.
///
/// Args:
///   raw_name: Channel name such as `pp_sigma` or `dd_delta`.
/// Returns:
///   The shell pair and sigma/pi/delta channel index.
inline SKChannelName parse_channel_name(const std::string& raw_name) {
    const std::string name = lower_copy(trim_copy(raw_name));
    const size_t sep = name.find('_');
    if (sep == std::string::npos || sep != 2 || sep + 1 >= name.size()) {
        throw std::runtime_error("Unsupported SKTB channel name: " + raw_name);
    }

    SKChannelName parsed;
    parsed.name = name;
    parsed.row_l = shell_from_channel_char(name[0]);
    parsed.col_l = shell_from_channel_char(name[1]);
    parsed.channel = sk_channel_index(name.substr(sep + 1));

    if (parsed.row_l > parsed.col_l) {
        throw std::runtime_error(
            "SKTB channel name must use canonical shell order s, p, d: " + raw_name);
    }
    if (parsed.channel > std::min(parsed.row_l, parsed.col_l)) {
        throw std::runtime_error("SKTB channel is not valid for shell pair: " + raw_name);
    }
    return parsed;
}

inline std::string make_channel_name(const std::string& raw_shell_pair, const std::string& raw_channel) {
    std::string shell_pair = lower_copy(trim_copy(raw_shell_pair));
    shell_pair.erase(std::remove(shell_pair.begin(), shell_pair.end(), '-'), shell_pair.end());
    shell_pair.erase(std::remove(shell_pair.begin(), shell_pair.end(), '_'), shell_pair.end());
    shell_pair.erase(std::remove(shell_pair.begin(), shell_pair.end(), ' '), shell_pair.end());

    const std::string name = shell_pair + "_" + lower_copy(trim_copy(raw_channel));
    parse_channel_name(name);
    return name;
}

inline bool is_hamiltonian_matrix_label(const std::string& raw_label) {
    const std::string label = lower_copy(trim_copy(raw_label));
    return label == "h" || label == "hamiltonian";
}

inline bool is_overlap_matrix_label(const std::string& raw_label) {
    const std::string label = lower_copy(trim_copy(raw_label));
    return label == "s" || label == "overlap";
}

/// Distance-dependent value for one Slater-Koster channel.
struct SKRadialFunction {
    std::string kind;
    std::vector<double> r;
    std::vector<double> values;
    std::vector<double> spline_second_derivatives;
    std::vector<double> coeffs;
    double lambda_value = 0.0;
    double r_on = 0.0;
    double r_cut = 0.0;

    /// Returns true when the radial function has been configured.
    bool enabled() const {
        return !kind.empty();
    }

    /// Returns the normalized radial-function kind.
    std::string normalized_kind() const {
        return lower_copy(trim_copy(kind));
    }

    /// Validates the fields required by the selected radial-function kind.
    ///
    /// Args:
    ///   context: Prefix used in validation error messages.
    void validate(const std::string& context) const {
        const std::string mode = normalized_kind();
        if (mode == "table") {
            throw std::runtime_error(
                context + ": radial function kind 'table' has been removed; use 'spline' for tabulated data.");
        }
        if (mode == "spline") {
            if (r.empty() || values.empty() || r.size() != values.size()) {
                throw std::runtime_error(
                    context + ": spline radial function requires equally sized non-empty 'r' and 'values' arrays.");
            }
            for (size_t i = 1; i < r.size(); ++i) {
                if (std::abs(r[i] - r[i - 1]) <= 1e-12 || r[i] < r[i - 1]) {
                    throw std::runtime_error(context + ": spline radial distances must be strictly increasing.");
                }
            }
            if (r_cut <= 0.0) {
                throw std::runtime_error(context + ": spline radial function requires a positive r_cut.");
            }
            return;
        }
        if (mode == "exp_poly") {
            if (coeffs.empty()) {
                throw std::runtime_error(context + ": exp_poly radial function requires at least one coefficient.");
            }
            if (r_cut <= 0.0) {
                throw std::runtime_error(context + ": exp_poly radial function requires a positive r_cut.");
            }
            if (r_on < 0.0 || r_on > r_cut) {
                throw std::runtime_error(context + ": exp_poly radial function requires 0 <= r_on <= r_cut.");
            }
            return;
        }
        throw std::runtime_error(context + ": unsupported radial function kind '" + kind + "'.");
    }

    /// Returns the distance cutoff beyond which the value is zero.
    double cutoff() const {
        return r_cut;
    }

    /// Builds natural cubic-spline second derivatives for tabulated data.
    std::vector<double> build_spline_second_derivatives() const {
        if (r.size() < 2 || values.size() != r.size()) {
            throw std::runtime_error("Spline radial function requires at least two tabulated points.");
        }

        const size_t n = r.size();
        std::vector<double> y2(n, 0.0);
        std::vector<double> u(n - 1, 0.0);

        for (size_t i = 1; i + 1 < n; ++i) {
            const double h_prev = r[i] - r[i - 1];
            const double h_next = r[i + 1] - r[i];
            if (h_prev <= 1e-12 || h_next <= 1e-12) {
                throw std::runtime_error("Spline radial distances must be strictly increasing.");
            }

            const double sig = h_prev / (h_prev + h_next);
            const double p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;

            const double slope_next = (values[i + 1] - values[i]) / h_next;
            const double slope_prev = (values[i] - values[i - 1]) / h_prev;
            const double rhs = (6.0 * (slope_next - slope_prev) / (h_prev + h_next) - sig * u[i - 1]) / p;
            u[i] = rhs;
        }

        for (size_t k = n - 1; k-- > 0;) {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }

        return y2;
    }

    /// Evaluates the radial function at a pair distance.
    ///
    /// Args:
    ///   distance: Interatomic distance in the same units as `r` and `r_cut`.
    /// Returns:
    ///   Channel value, or zero outside the cutoff.
    double evaluate(double distance) const {
        const std::string mode = normalized_kind();
        if (distance < 0.0) {
            return 0.0;
        }

        if (mode == "table") {
            throw std::runtime_error("Radial function kind 'table' has been removed; use 'spline' for tabulated data.");
        }

        if (mode == "spline") {
            if (distance > r_cut) {
                return 0.0;
            }
            if (distance <= r.front()) {
                return values.front();
            }

            const auto upper = std::lower_bound(r.begin(), r.end(), distance);
            if (upper == r.end()) {
                return values.back();
            }

            const size_t i = static_cast<size_t>(std::distance(r.begin(), upper));
            const size_t lo = i - 1;
            const size_t hi = i;
            const double h = r[hi] - r[lo];
            if (h <= 1e-12) {
                return values[hi];
            }

            const std::vector<double>* y2 = &spline_second_derivatives;
            std::vector<double> local_y2;
            if (y2->empty()) {
                local_y2 = build_spline_second_derivatives();
                y2 = &local_y2;
            }
            const double a = (r[hi] - distance) / h;
            const double b = (distance - r[lo]) / h;
            return a * values[lo]
                 + b * values[hi]
                 + ((a * a * a - a) * (*y2)[lo] + (b * b * b - b) * (*y2)[hi]) * (h * h) / 6.0;
        }

        if (mode == "exp_poly") {
            if (distance > r_cut) {
                return 0.0;
            }
            double cutoff_weight = 1.0;
            if (distance > r_on) {
                const double span = r_cut - r_on;
                if (span <= 1e-12) {
                    cutoff_weight = 0.0;
                } else {
                    const double arg = M_PI * (distance - r_on) / span;
                    cutoff_weight = 0.5 * (1.0 + std::cos(arg));
                }
            }

            double poly = 0.0;
            double power = 1.0;
            for (double coeff : coeffs) {
                poly += coeff * power;
                power *= distance;
            }
            return cutoff_weight * std::exp(-lambda_value * distance) * poly;
        }

        throw std::runtime_error("Unsupported radial function kind: " + kind);
    }
};

/// Raw channel data for one ordered species pair from user input.
struct SKPairInput {
    std::array<std::string, 2> species = {"", ""};
    std::map<std::string, SKRadialFunction> channels;

    /// Adds one shell-pair SK channel using the public `pp`/`pi` vocabulary.
    SKPairInput& channel(
        const std::string& shell_pair,
        const std::string& sk_channel,
        const SKRadialFunction& radial
    ) {
        channels[make_channel_name(shell_pair, sk_channel)] = radial;
        return *this;
    }
};

/// Raw onsite and pair data for one matrix, either Hamiltonian or overlap.
struct SKMatrixInput {
    std::map<std::string, std::map<std::string, double>> onsite;
    std::vector<SKPairInput> pairs;

    /// Returns true when any onsite or pair data is present.
    bool enabled() const {
        return !onsite.empty() || !pairs.empty();
    }

    /// Sets one onsite value using species and shell/orbital labels.
    SKMatrixInput& set_onsite(
        const std::string& species,
        const std::string& orbital,
        double value
    ) {
        onsite[canonical_species_symbol(species)][canonical_orbital_label(orbital)] = value;
        return *this;
    }

    /// Returns the editable input entry for one ordered species pair.
    SKPairInput& pair(const std::string& row_species, const std::string& col_species) {
        const std::string row_symbol = canonical_species_symbol(row_species);
        const std::string col_symbol = canonical_species_symbol(col_species);
        for (auto& entry : pairs) {
            if (canonical_species_symbol(entry.species[0]) == row_symbol &&
                canonical_species_symbol(entry.species[1]) == col_symbol) {
                return entry;
            }
        }
        SKPairInput entry;
        entry.species = {row_symbol, col_symbol};
        pairs.push_back(std::move(entry));
        return pairs.back();
    }

    /// Adds one pair channel using the public shell-pair/channel vocabulary.
    SKMatrixInput& set_pair_channel(
        const std::string& row_species,
        const std::string& col_species,
        const std::string& shell_pair,
        const std::string& sk_channel,
        const SKRadialFunction& radial
    ) {
        pair(row_species, col_species).channel(shell_pair, sk_channel, radial);
        return *this;
    }
};

/// Raw SKTB parameter input before species and basis resolution.
struct SKTBInput {
    std::map<std::string, std::vector<std::string>> basis;
    SKMatrixInput hamiltonian;
    std::optional<SKMatrixInput> overlap;

    /// Returns true when the input has enough data to build a model.
    bool enabled() const {
        return !basis.empty() && hamiltonian.enabled();
    }

    /// Returns Hamiltonian or overlap input by the public `H`/`S` label.
    SKMatrixInput& matrix(const std::string& label) {
        if (is_hamiltonian_matrix_label(label)) {
            return hamiltonian;
        }
        if (is_overlap_matrix_label(label)) {
            if (!overlap.has_value()) {
                overlap = SKMatrixInput{};
            }
            return *overlap;
        }
        throw std::runtime_error("SKTB matrix label must be 'H' or 'S'.");
    }

    /// Sets one Hamiltonian or overlap onsite value.
    SKTBInput& set_onsite(
        const std::string& matrix_label,
        const std::string& species,
        const std::string& orbital,
        double value
    ) {
        matrix(matrix_label).set_onsite(species, orbital, value);
        return *this;
    }

    /// Adds one Hamiltonian or overlap pair channel.
    SKTBInput& set_pair_channel(
        const std::string& matrix_label,
        const std::string& row_species,
        const std::string& col_species,
        const std::string& shell_pair,
        const std::string& sk_channel,
        const SKRadialFunction& radial
    ) {
        matrix(matrix_label).set_pair_channel(row_species, col_species, shell_pair, sk_channel, radial);
        return *this;
    }
};

/// Basis and onsite data for one atom type.
struct SKAtomType {
    std::string symbol;
    std::vector<SKOrbital> orbitals;
    std::map<std::string, int> orbital_index;
    std::vector<double> h_onsite;
    std::vector<double> s_onsite;
};

/// One resolved sigma/pi/delta channel for an ordered shell pair.
struct SKPairChannel {
    int row_l = 0;
    int col_l = 0;
    int channel = 0;
    SKRadialFunction h;
    SKRadialFunction s;
    bool has_h = false;
    bool has_s = false;
};

/// All radial channels for one ordered atom-type pair.
struct SKAtomPair {
    int row_type = 0;
    int col_type = 0;
    std::vector<SKPairChannel> channels;
    double h_max_cutoff = 0.0;
    double s_max_cutoff = 0.0;

    double max_cutoff() const {
        return std::max(h_max_cutoff, s_max_cutoff);
    }
};

/// Resolved SKTB model ready for Hamiltonian and overlap construction.
struct SKTBModel {
    std::vector<SKAtomType> atom_types;
    std::vector<SKAtomPair> atom_pairs;
    std::map<std::pair<int, int>, size_t> pair_index_by_type;
    bool has_overlap = false;

    /// Returns true when the model contains a resolved basis.
    bool enabled() const {
        return !atom_types.empty();
    }

    /// Returns true when no explicit overlap matrix is parameterized.
    bool orthogonal() const {
        return !has_overlap;
    }

    const SKAtomPair* find_pair(int row_type, int col_type) const {
        auto it = pair_index_by_type.find(std::make_pair(row_type, col_type));
        return (it == pair_index_by_type.end()) ? nullptr : &atom_pairs[it->second];
    }

    SKAtomPair& pair(int row_type, int col_type) {
        const auto key = std::make_pair(row_type, col_type);
        auto it = pair_index_by_type.find(key);
        if (it != pair_index_by_type.end()) {
            return atom_pairs[it->second];
        }

        SKAtomPair pair_entry;
        pair_entry.row_type = row_type;
        pair_entry.col_type = col_type;
        atom_pairs.push_back(std::move(pair_entry));
        const size_t index = atom_pairs.size() - 1;
        pair_index_by_type.emplace(key, index);
        return atom_pairs.back();
    }
};

/// AtomicData requirements derived from an SKTB model.
struct SKTBAtomicPreparation {
    std::vector<std::string> type_symbols;
    std::vector<int> type_norb;
    std::vector<double> r_max_per_type;
};

/// Expands shell aliases such as `p` and `d` into an atom-type basis.
inline SKAtomType build_atom_type_basis(const std::vector<std::string>& raw_labels, const std::string& species_symbol) {
    SKAtomType atom_type;
    atom_type.symbol = species_symbol;
    for (const auto& raw_label : raw_labels) {
        const auto expanded = expand_basis_label(raw_label);
        for (const auto& label : expanded) {
            if (atom_type.orbital_index.count(label) != 0) {
                throw std::runtime_error(
                    "SKTB basis for species " + species_symbol + " contains a duplicate orbital: " + label);
            }
            const int index = static_cast<int>(atom_type.orbitals.size());
            atom_type.orbitals.push_back(make_orbital(label));
            atom_type.orbital_index.emplace(label, index);
        }
    }
    if (atom_type.orbitals.empty()) {
        throw std::runtime_error("SKTB basis for species " + species_symbol + " must not be empty.");
    }
    return atom_type;
}

/// Expands onsite shell/orbital entries into one diagonal value per basis orbital.
inline std::vector<double> expand_onsite_diagonal(
    const std::map<std::string, double>& raw_entries,
    const SKAtomType& atom_type,
    bool require_complete,
    bool default_identity
) {
    std::vector<double> diagonal(
        atom_type.orbitals.size(),
        default_identity ? 1.0 : 0.0);
    std::vector<bool> assigned(atom_type.orbitals.size(), default_identity);

    for (const auto& kv : raw_entries) {
        const auto labels = expand_basis_label(kv.first);
        for (const auto& label : labels) {
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
            diagonal[it->second] = kv.second;
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

/// Canonicalizes and validates constructor-side species order.
inline std::vector<std::string> resolve_type_symbols(const std::vector<std::string>& raw_symbols) {
    std::vector<std::string> symbols;
    symbols.reserve(raw_symbols.size());
    for (const auto& raw_symbol : raw_symbols) {
        symbols.push_back(canonical_species_symbol(raw_symbol));
    }

    if (symbols.empty()) {
        throw std::runtime_error(
            "SKTB setup requires constructor-side type symbols. "
            "Construct RSATB with type symbols before building SKTB operators.");
    }

    std::map<std::string, int> counts;
    for (const auto& symbol : symbols) {
        counts[symbol] += 1;
        if (counts[symbol] > 1) {
            throw std::runtime_error("SKTB constructor type symbols must not contain duplicates.");
        }
    }

    return symbols;
}

/// Builds a resolved SKTB model from raw input and constructor species order.
///
/// Args:
///   config: Raw SKTB basis, onsite, and pair data.
///   type_symbols: Constructor-side species order.
///   expected_type_norb: Optional orbital counts already stored in AtomicData.
/// Returns:
///   Resolved model keyed by atom type indices.
inline SKTBModel build_sktb_model(
    const SKTBInput& config,
    const std::vector<std::string>& type_symbols,
    const std::vector<int>& expected_type_norb = {}
) {
    if (!config.enabled()) {
        throw std::runtime_error("SKTB input must define a basis and Hamiltonian data.");
    }

    SKTBModel model;
    const auto resolved_symbols = resolve_type_symbols(type_symbols);
    const int n_types = static_cast<int>(resolved_symbols.size());
    model.atom_types.resize(n_types);
    model.has_overlap = config.overlap.has_value();

    std::map<std::string, int> symbol_to_type;
    for (int type = 0; type < n_types; ++type) {
        symbol_to_type.emplace(resolved_symbols[type], type);
    }

    for (int type = 0; type < n_types; ++type) {
        const std::string& symbol = resolved_symbols[type];
        auto basis_it = config.basis.find(symbol);
        if (basis_it == config.basis.end()) {
            throw std::runtime_error("SKTB basis is missing a definition for species " + symbol + ".");
        }

        model.atom_types[type] = build_atom_type_basis(basis_it->second, symbol);
        if (!expected_type_norb.empty() &&
            expected_type_norb[type] != static_cast<int>(model.atom_types[type].orbitals.size())) {
            throw std::runtime_error(
                "SKTB basis for species " + symbol +
                " expands to " + std::to_string(model.atom_types[type].orbitals.size()) +
                " orbitals, but AtomicData expects " + std::to_string(expected_type_norb[type]) + ".");
        }

        auto h_onsite_it = config.hamiltonian.onsite.find(symbol);
        if (h_onsite_it == config.hamiltonian.onsite.end()) {
            throw std::runtime_error("SKTB Hamiltonian onsite terms are missing for species " + symbol + ".");
        }
        model.atom_types[type].h_onsite =
            expand_onsite_diagonal(h_onsite_it->second, model.atom_types[type], true, false);

        if (model.has_overlap) {
            auto s_onsite_it = config.overlap->onsite.find(symbol);
            if (s_onsite_it == config.overlap->onsite.end()) {
                model.atom_types[type].s_onsite =
                    expand_onsite_diagonal({}, model.atom_types[type], false, true);
            } else {
                model.atom_types[type].s_onsite =
                    expand_onsite_diagonal(s_onsite_it->second, model.atom_types[type], true, false);
            }
        }
    }

    auto resolve_pairs = [&](const SKMatrixInput& input, bool for_overlap, const std::string& model_name) {
        for (const auto& pair_cfg : input.pairs) {
            const std::string row_symbol = canonical_species_symbol(pair_cfg.species[0]);
            const std::string col_symbol = canonical_species_symbol(pair_cfg.species[1]);

            auto row_it = symbol_to_type.find(row_symbol);
            auto col_it = symbol_to_type.find(col_symbol);
            if (row_it == symbol_to_type.end() || col_it == symbol_to_type.end()) {
                continue;
            }

            auto& pair = model.pair(row_it->second, col_it->second);

            for (const auto& channel_kv : pair_cfg.channels) {
                SKChannelName channel_name;
                try {
                    channel_name = parse_channel_name(channel_kv.first);
                } catch (const std::runtime_error&) {
                    throw std::runtime_error(
                        "SKTB " + model_name + " pair (" + row_symbol + ", " + col_symbol +
                        ") uses unsupported channel '" + channel_kv.first + "'.");
                }
                channel_kv.second.validate(
                    "SKTB " + model_name + " pair (" + row_symbol + ", " + col_symbol +
                    "), channel " + channel_name.name);
                auto radial = channel_kv.second;
                if (radial.normalized_kind() == "spline" && radial.spline_second_derivatives.empty()) {
                    radial.spline_second_derivatives = radial.build_spline_second_derivatives();
                }

                auto add_channel = [&](int row_l, int col_l) {
                    auto it = std::find_if(
                        pair.channels.begin(),
                        pair.channels.end(),
                        [&](const SKPairChannel& channel) {
                            return channel.row_l == row_l &&
                                   channel.col_l == col_l &&
                                   channel.channel == channel_name.channel;
                        });
                    if (it == pair.channels.end()) {
                        SKPairChannel channel;
                        channel.row_l = row_l;
                        channel.col_l = col_l;
                        channel.channel = channel_name.channel;
                        pair.channels.push_back(std::move(channel));
                        it = pair.channels.end() - 1;
                    }

                    bool& has_value = for_overlap ? it->has_s : it->has_h;
                    if (has_value) {
                        throw std::runtime_error(
                            "SKTB " + model_name + " pair (" + row_symbol + ", " + col_symbol +
                            ") assigns channel '" + channel_name.name + "' more than once.");
                    }

                    if (for_overlap) {
                        it->s = radial;
                        pair.s_max_cutoff = std::max(pair.s_max_cutoff, radial.cutoff());
                    } else {
                        it->h = radial;
                        pair.h_max_cutoff = std::max(pair.h_max_cutoff, radial.cutoff());
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

    resolve_pairs(config.hamiltonian, false, "Hamiltonian");
    if (model.has_overlap) {
        resolve_pairs(*config.overlap, true, "overlap");
    }

    return model;
}

/// Derives AtomicData basis sizes and graph cutoffs from a resolved model.
inline SKTBAtomicPreparation prepare_sktb_atomic_data(const SKTBModel& model) {
    SKTBAtomicPreparation prepared;
    prepared.type_symbols.reserve(model.atom_types.size());
    prepared.type_norb.reserve(model.atom_types.size());
    for (const auto& atom_type : model.atom_types) {
        prepared.type_symbols.push_back(atom_type.symbol);
        prepared.type_norb.push_back(static_cast<int>(atom_type.orbitals.size()));
    }

    prepared.r_max_per_type.assign(model.atom_types.size(), 0.0);
    for (const auto& pair : model.atom_pairs) {
        prepared.r_max_per_type[pair.row_type] =
            std::max(prepared.r_max_per_type[pair.row_type], pair.max_cutoff());
        prepared.r_max_per_type[pair.col_type] =
            std::max(prepared.r_max_per_type[pair.col_type], pair.max_cutoff());
    }

    return prepared;
}

inline SKTBAtomicPreparation prepare_sktb_atomic_data_from_atomic_numbers(
    const std::vector<int>& z,
    const SKTBInput& config
) {
    return prepare_sktb_atomic_data(build_sktb_model(config, unique_type_symbols_from_atomic_numbers(z)));
}

} // namespace backend
} // namespace rsatb

#endif // RSATB_BACKEND_SKTB_HPP
