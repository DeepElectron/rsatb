#ifndef RSATB_MODEL_SKTB_ORBITAL_LABELS_HPP
#define RSATB_MODEL_SKTB_ORBITAL_LABELS_HPP

// Orbital and Slater-Koster channel vocabulary.
//
// Two naming systems meet here. Basis orbitals are named by shell alias ("p")
// or explicitly ("px", "dz2"), and SK channels are named by shell pair plus
// bond type ("pp_pi", "dd_delta"). Both are parsed once, into indices, so the
// rest of the SKTB path works with integers rather than strings.

#include "vbcsr/core/atomic/periodic_table.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {
namespace sktb {

/// Normalizes accepted spellings of an orbital label.
inline std::string CanonicalOrbitalLabel(const std::string& raw_label) {
    std::string label = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_label));
    if (label == "dxz") {
        label = "dzx";
    }
    if (label == "dx2y2") {
        label = "dx2-y2";
    }
    return label;
}

inline bool IsSupportedExplicitOrbital(const std::string& label) {
    return label == "s" ||
           label == "px" || label == "py" || label == "pz" ||
           label == "dxy" || label == "dyz" || label == "dzx" ||
           label == "dx2-y2" || label == "dz2";
}

/// Expands a shell alias into its explicit orbitals, in DeePTB magnetic order.
///
/// Args:
///   raw_label: A shell alias ("s", "p", "d") or an explicit orbital ("pz").
/// Returns:
///   The explicit orbital labels the alias stands for.
inline std::vector<std::string> ExpandBasisLabel(const std::string& raw_label) {
    const std::string label = CanonicalOrbitalLabel(raw_label);
    if (label == "s") {
        return {"s"};
    }
    if (label == "p") {
        return {"py", "pz", "px"};
    }
    if (label == "d") {
        return {"dxy", "dyz", "dz2", "dzx", "dx2-y2"};
    }
    if (IsSupportedExplicitOrbital(label)) {
        return {label};
    }
    throw std::runtime_error("Unsupported SKTB orbital label: " + raw_label);
}

/// One expanded basis orbital in DeePTB magnetic order.
struct SkOrbital {
    std::string label;
    int l = 0;
    int m_index = 0;
};

inline SkOrbital MakeOrbital(const std::string& raw_label) {
    const std::string label = CanonicalOrbitalLabel(raw_label);
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

inline int ShellFromChannelChar(char value) {
    if (value == 's') return 0;
    if (value == 'p') return 1;
    if (value == 'd') return 2;
    throw std::runtime_error("Unsupported SKTB channel shell.");
}

inline int SkChannelIndex(const std::string& kind) {
    if (kind == "sigma") return 0;
    if (kind == "pi") return 1;
    if (kind == "delta") return 2;
    throw std::runtime_error("Unsupported SKTB channel kind: " + kind);
}

/// Parsed shell pair and SK channel from names like `pd_pi`.
struct SkChannelName {
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
inline SkChannelName ParseChannelName(const std::string& raw_name) {
    const std::string name = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_name));
    const size_t sep = name.find('_');
    if (sep == std::string::npos || sep != 2 || sep + 1 >= name.size()) {
        throw std::runtime_error("Unsupported SKTB channel name: " + raw_name);
    }

    SkChannelName parsed;
    parsed.name = name;
    parsed.row_l = ShellFromChannelChar(name[0]);
    parsed.col_l = ShellFromChannelChar(name[1]);
    parsed.channel = SkChannelIndex(name.substr(sep + 1));

    if (parsed.row_l > parsed.col_l) {
        throw std::runtime_error(
            "SKTB channel name must use canonical shell order s, p, d: " + raw_name);
    }
    if (parsed.channel > std::min(parsed.row_l, parsed.col_l)) {
        throw std::runtime_error("SKTB channel is not valid for shell pair: " + raw_name);
    }
    return parsed;
}

/// Assembles and validates a channel name from its two parts.
///
/// Args:
///   raw_shell_pair: Shell pair such as "pp" or "p-d"; separators are ignored.
///   raw_channel: Bond type, one of "sigma", "pi", "delta".
inline std::string MakeChannelName(const std::string& raw_shell_pair, const std::string& raw_channel) {
    std::string shell_pair = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_shell_pair));
    shell_pair.erase(std::remove(shell_pair.begin(), shell_pair.end(), '-'), shell_pair.end());
    shell_pair.erase(std::remove(shell_pair.begin(), shell_pair.end(), '_'), shell_pair.end());
    shell_pair.erase(std::remove(shell_pair.begin(), shell_pair.end(), ' '), shell_pair.end());

    const std::string name = shell_pair + "_" + vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_channel));
    ParseChannelName(name);
    return name;
}

}  // namespace sktb
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_SKTB_ORBITAL_LABELS_HPP
