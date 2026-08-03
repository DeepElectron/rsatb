#ifndef RSATB_MODEL_SKTB_SKF_TABLE_HPP
#define RSATB_MODEL_SKTB_SKF_TABLE_HPP

// Turns a directory of DFTB .skf files into an SkTable.
//
// The payoff of keeping the resolved table separate from its input dialect is
// here: this file only has to produce SkParameters. The Slater-Koster rotation
// machinery, the image construction, and every observable are reused unchanged.

#include "sk_table.hpp"
#include "../basis_spec.hpp"
#include "../../io/skf.hpp"

#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {
namespace sktb {

/// Highest shell each species declares, used to decide which .skf columns matter.
inline int MaxShellInBasis(const std::vector<std::string>& basis_labels) {
    int max_l = 0;
    for (const auto& raw_label : basis_labels) {
        for (const auto& label : ExpandBasisLabel(raw_label)) {
            max_l = std::max(max_l, MakeOrbital(label).l);
        }
    }
    return max_l;
}

/// Joins a directory and a file name with a single separator.
inline std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    return (dir.back() == '/') ? dir + name : dir + "/" + name;
}

/// Reads a .skf parameter set into Hamiltonian and overlap SkParameters.
///
/// One file is read per ORDERED species pair, so an A/B system needs A-A.skf,
/// A-B.skf, B-A.skf, and B-B.skf. Onsite energies come from the homonuclear
/// files.
///
/// Only channels the declared basis can use are kept: a set tabulating d
/// integrals is fine for an s/p basis, the extra columns are simply unused.
/// The basis is declared rather than inferred from which onsite energies are
/// non-zero, because "is E_d numerically zero" is a guess, not a statement.
///
/// What to do when a mixed-shell channel is genuinely ambiguous.
///
/// A .skf column such as `Hsp0` is a single number for the unordered pair
/// {s, p}. For an ordered species pair (A, B) it could mean the s on A with
/// the p on B, or the p on A with the s on B. Those are different integrals,
/// and the file does not say which.
///
/// Usually only one reading is possible, because the construction pairs
/// orbitals by shell: for C(s,p) with H(s), a p on H does not exist, so `Hsp0`
/// can only be the p on C with the s on H. The reading is forced by the basis,
/// not assumed. The ambiguity is real only when *both* species carry *both*
/// shells of the pair -- C(s,p) with N(s,p), where both readings exist.
enum class SkfHeteronuclearPolicy {
    /// Refuse a set only where the ambiguity is real. The default.
    kRejectAmbiguous,
    /// Resolve it by placing the lower shell on the first species of the file
    /// name. Set this only after validating against DFTB+ for the system in
    /// question.
    kAssumeLowerShellOnRowSpecies,
};

/// Returns true when a species declares the given shell.
inline bool BasisHasShell(int max_l, int shell_l) {
    return shell_l <= max_l;
}

/// Args:
///   skf_dir: Directory holding the .skf files.
///   basis: species symbol -> basis labels, e.g. {"C": {"s", "p"}}.
///   heteronuclear: What to do about multi-species sets.
///   hamiltonian: Receives the onsite energies and H channels.
///   overlap: Receives the S channels.
/// Throws:
///   std::runtime_error if a required file is missing or malformed, or if the
///   set is multi-species and `heteronuclear` is kReject.
inline void ReadSkfParameterSet(
    const std::string& skf_dir,
    const std::map<std::string, std::vector<std::string>>& basis,
    SkfHeteronuclearPolicy heteronuclear,
    SkParameters* hamiltonian,
    SkParameters* overlap
) {
    if (basis.empty()) {
        throw std::runtime_error("DFTB setup requires a basis definition.");
    }

    std::map<std::string, int> max_shell;
    for (const auto& entry : basis) {
        max_shell[vbcsr::atomic::CanonicalSymbol(entry.first)] = MaxShellInBasis(entry.second);
    }

    for (const auto& row_entry : max_shell) {
        for (const auto& col_entry : max_shell) {
            const std::string& row_symbol = row_entry.first;
            const std::string& col_symbol = col_entry.first;
            const std::string path = JoinPath(skf_dir, io::SkfFileName(row_symbol, col_symbol));
            const io::SkfTable table = io::ReadSkfFile(path, row_symbol, col_symbol);

            if (table.r.empty()) {
                throw std::runtime_error("Slater-Koster file tabulates no grid points: " + path);
            }
            const double r_cut = table.cutoff();

            if (row_symbol == col_symbol) {
                if (!table.has_onsite) {
                    throw std::runtime_error(
                        "Homonuclear Slater-Koster file is missing its onsite line: " + path);
                }
                // Assign one onsite energy per declared shell.
                static const char* kShellLabel[3] = {"s", "p", "d"};
                for (int l = 0; l <= row_entry.second; ++l) {
                    hamiltonian->SetOnsite(row_symbol, kShellLabel[l], table.onsite.energy[l]);
                }
            }

            const int row_max_l = row_entry.second;
            const int col_max_l = col_entry.second;

            for (const auto& spec : io::SkfChannelLayout()) {
                // A channel is usable when some assignment of its two shells to
                // the two species is compatible with both declared bases.
                const bool forward =
                    BasisHasShell(row_max_l, spec.lo_l) && BasisHasShell(col_max_l, spec.hi_l);
                const bool reverse =
                    BasisHasShell(row_max_l, spec.hi_l) && BasisHasShell(col_max_l, spec.lo_l);
                if (!forward && !reverse) {
                    continue;  // neither shell pair exists in this species pair
                }

                // Both readings possible AND the two species differ: the file
                // cannot say which of two distinct integrals its single column
                // holds. For a homonuclear pair the two readings describe the
                // same species on both sides, so they are the same integral up
                // to the Slater-Koster sign, which the construction already
                // applies geometrically.
                if (spec.mixed_shells() && forward && reverse && row_symbol != col_symbol &&
                    heteronuclear == SkfHeteronuclearPolicy::kRejectAmbiguous) {
                    throw std::runtime_error(
                        "Slater-Koster channel '" + std::string(spec.name) + "' is ambiguous for the pair (" +
                        row_symbol + ", " + col_symbol + "): both species declare both shells, so the "
                        "single column in " + path + " could be either integral, and the file does not "
                        "say which. This orientation has not been validated against a DFTB+ reference. "
                        "Restrict one of the bases, or pass "
                        "heteronuclear=assume_lower_shell_on_row_species after validating.");
                }

                const auto index = static_cast<size_t>(spec.channel);
                const auto& h_values = table.h[index];
                const auto& s_values = table.s[index];

                bool h_nonzero = false;
                for (double value : h_values) {
                    if (value != 0.0) { h_nonzero = true; break; }
                }
                bool s_nonzero = false;
                for (double value : s_values) {
                    if (value != 0.0) { s_nonzero = true; break; }
                }
                if (!h_nonzero && !s_nonzero) {
                    continue;  // channel not parameterized for this pair
                }

                hamiltonian->Pair(row_symbol, col_symbol).channels[spec.name] =
                    MakeSpline(table.r, h_values, r_cut);
                overlap->Pair(row_symbol, col_symbol).channels[spec.name] =
                    MakeSpline(table.r, s_values, r_cut);
            }
        }
    }
}

}  // namespace sktb
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_SKTB_SKF_TABLE_HPP
