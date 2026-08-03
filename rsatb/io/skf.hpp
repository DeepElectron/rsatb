#ifndef RSATB_IO_SKF_HPP
#define RSATB_IO_SKF_HPP

// DFTB Slater-Koster (.skf) parameter files.
//
// One file per ORDERED species pair: C-C.skf, C-H.skf, and H-C.skf are three
// separate files. The heteronuclear pair files are not transposes of each
// other -- each tabulates integrals with the first species as the row -- so
// both must be read.
//
// Layout:
//   line 1        gridDist nGridPoints          (Bohr, count)
//   line 2        Ed Ep Es SPE Ud Up Us fd fp fs   (homonuclear files only)
//   next line     mass c2..c9 rcut               (repulsive polynomial; unused)
//   nGridPoints x 20 values:
//                 Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0
//                 Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0
//   optional      Spline block (repulsive; unused here)
//
// Row r of the table is distance (r + 1) * gridDist; the file stores no r = 0
// entry.
//
// Units are Hartree and Bohr on disk, eV and Angstrom everywhere in rsatb.
// The conversion happens once, here, so no caller has to remember it.

#include "vbcsr/core/atomic/periodic_table.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

constexpr double kElectronVoltPerHartree = 27.211386245988;
constexpr double kAngstromPerBohrLength = 0.529177210903;

/// The ten Slater-Koster integrals a .skf row stores, in file order.
///
/// The names are the file's own: the digit is the bond type (0 = sigma,
/// 1 = pi, 2 = delta) and the letters are the shell pair.
enum class SkfChannel {
    kDdSigma = 0,
    kDdPi,
    kDdDelta,
    kPdSigma,
    kPdPi,
    kPpSigma,
    kPpPi,
    kSdSigma,
    kSpSigma,
    kSsSigma,
    kCount,
};

constexpr int kSkfChannelCount = static_cast<int>(SkfChannel::kCount);
constexpr int kSkfValuesPerRow = 2 * kSkfChannelCount;  // ten H then ten S

/// Maps a .skf column onto the shell pair and bond type it carries.
///
/// `lo_l` and `hi_l` are the two shells in ascending order, matching rsatb's
/// channel names. A .skf column is one integral for an unordered shell pair --
/// there is no separate "ps" column next to "sp" -- so a channel is directional
/// only through which species sits on which side of the pair.
struct SkfChannelSpec {
    SkfChannel channel;
    int lo_l;
    int hi_l;
    int bond_type;     // 0 = sigma, 1 = pi, 2 = delta
    const char* name;  // canonical rsatb channel name, e.g. "pp_pi"

    bool mixed_shells() const { return lo_l != hi_l; }
};

/// Returns the column layout of a .skf row.
///
/// This table is the whole mapping between the file format and rsatb's channel
/// vocabulary; everything downstream works from it.
inline const std::array<SkfChannelSpec, kSkfChannelCount>& SkfChannelLayout() {
    static const std::array<SkfChannelSpec, kSkfChannelCount> layout = {{
        {SkfChannel::kDdSigma, 2, 2, 0, "dd_sigma"},
        {SkfChannel::kDdPi,    2, 2, 1, "dd_pi"},
        {SkfChannel::kDdDelta, 2, 2, 2, "dd_delta"},
        {SkfChannel::kPdSigma, 1, 2, 0, "pd_sigma"},
        {SkfChannel::kPdPi,    1, 2, 1, "pd_pi"},
        {SkfChannel::kPpSigma, 1, 1, 0, "pp_sigma"},
        {SkfChannel::kPpPi,    1, 1, 1, "pp_pi"},
        {SkfChannel::kSdSigma, 0, 2, 0, "sd_sigma"},
        {SkfChannel::kSpSigma, 0, 1, 0, "sp_sigma"},
        {SkfChannel::kSsSigma, 0, 0, 0, "ss_sigma"},
    }};
    return layout;
}

/// Onsite data from a homonuclear .skf file.
struct SkfOnsite {
    /// Onsite energies by shell index (0 = s, 1 = p, 2 = d), in eV.
    std::array<double, 3> energy = {0.0, 0.0, 0.0};
    /// Hubbard U by shell index, in eV.
    std::array<double, 3> hubbard_u = {0.0, 0.0, 0.0};
    /// Shell occupations.
    std::array<double, 3> occupation = {0.0, 0.0, 0.0};
};

/// One parsed .skf file.
struct SkfTable {
    std::string row_species;
    std::string col_species;

    /// Distances in Angstrom, ascending, starting at one grid step.
    std::vector<double> r;
    /// Hamiltonian integrals in eV: [channel][grid point].
    std::array<std::vector<double>, kSkfChannelCount> h;
    /// Overlap integrals (dimensionless): [channel][grid point].
    std::array<std::vector<double>, kSkfChannelCount> s;

    /// Present only for a homonuclear file.
    bool has_onsite = false;
    SkfOnsite onsite;

    bool homonuclear() const { return row_species == col_species; }

    /// Returns the largest tabulated distance, in Angstrom.
    double cutoff() const { return r.empty() ? 0.0 : r.back(); }
};

namespace skf_detail {

inline std::vector<double> ParseDoubles(const std::string& line) {
    // .skf files use ',' as a filler separator and support a run-length form
    // "N*value" for repeated entries, both of which appear in released
    // parameter sets.
    std::string normalized = line;
    for (char& c : normalized) {
        if (c == ',') c = ' ';
    }

    std::vector<double> values;
    std::istringstream stream(normalized);
    std::string token;
    while (stream >> token) {
        const size_t star = token.find('*');
        if (star == std::string::npos) {
            values.push_back(std::stod(token));
            continue;
        }
        const int repeat = std::stoi(token.substr(0, star));
        const double value = std::stod(token.substr(star + 1));
        if (repeat < 0) {
            throw std::runtime_error("Negative repeat count in .skf line: " + line);
        }
        values.insert(values.end(), static_cast<size_t>(repeat), value);
    }
    return values;
}

/// Reads the next line that is neither blank nor a comment.
inline bool NextDataLine(std::istream& stream, std::string* line) {
    while (std::getline(stream, *line)) {
        const std::string trimmed = vbcsr::atomic::TrimCopy(*line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        *line = trimmed;
        return true;
    }
    return false;
}

}  // namespace skf_detail

/// Parses one .skf file.
///
/// Args:
///   path: Path to the file.
///   row_species: Element whose orbitals index the rows.
///   col_species: Element whose orbitals index the columns.
/// Returns:
///   The tabulated integrals, converted to eV and Angstrom.
/// Throws:
///   std::runtime_error if the file is missing, truncated, or has a row that
///   does not carry the expected 20 values.
inline SkfTable ReadSkfFile(
    const std::string& path,
    const std::string& row_species,
    const std::string& col_species
) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open Slater-Koster file: " + path);
    }

    SkfTable table;
    table.row_species = vbcsr::atomic::CanonicalSymbol(row_species);
    table.col_species = vbcsr::atomic::CanonicalSymbol(col_species);

    std::string line;
    if (!skf_detail::NextDataLine(file, &line)) {
        throw std::runtime_error("Empty Slater-Koster file: " + path);
    }

    // An "extended" file starts with '@' and carries f orbitals, which rsatb's
    // Slater-Koster construction does not implement.
    if (!line.empty() && line[0] == '@') {
        throw std::runtime_error(
            "Extended (f-orbital) Slater-Koster files are not supported: " + path);
    }

    const auto header = skf_detail::ParseDoubles(line);
    if (header.size() < 2) {
        throw std::runtime_error("Malformed .skf header, expected 'gridDist nGridPoints': " + path);
    }
    const double grid_dist_bohr = header[0];
    const int n_grid = static_cast<int>(header[1]);
    if (!(grid_dist_bohr > 0.0) || n_grid <= 0) {
        throw std::runtime_error("Malformed .skf grid specification in " + path);
    }

    if (table.homonuclear()) {
        if (!skf_detail::NextDataLine(file, &line)) {
            throw std::runtime_error("Truncated .skf: missing onsite line in " + path);
        }
        const auto values = skf_detail::ParseDoubles(line);
        if (values.size() < 10) {
            throw std::runtime_error(
                "Malformed .skf onsite line, expected 10 values (Ed Ep Es SPE Ud Up Us fd fp fs): " + path);
        }
        // The file orders shells d, p, s; rsatb indexes them s, p, d.
        table.onsite.energy = {values[2] * kElectronVoltPerHartree,
                               values[1] * kElectronVoltPerHartree,
                               values[0] * kElectronVoltPerHartree};
        table.onsite.hubbard_u = {values[6] * kElectronVoltPerHartree,
                                  values[5] * kElectronVoltPerHartree,
                                  values[4] * kElectronVoltPerHartree};
        table.onsite.occupation = {values[9], values[8], values[7]};
        table.has_onsite = true;
    }

    // Mass and repulsive polynomial; rsatb needs neither, but the line must be
    // consumed before the integral table starts.
    if (!skf_detail::NextDataLine(file, &line)) {
        throw std::runtime_error("Truncated .skf: missing mass/repulsive line in " + path);
    }

    table.r.reserve(static_cast<size_t>(n_grid));
    for (auto& channel : table.h) channel.reserve(static_cast<size_t>(n_grid));
    for (auto& channel : table.s) channel.reserve(static_cast<size_t>(n_grid));

    for (int row = 0; row < n_grid; ++row) {
        if (!skf_detail::NextDataLine(file, &line)) {
            throw std::runtime_error(
                "Truncated .skf: expected " + std::to_string(n_grid) + " integral rows, got " +
                std::to_string(row) + " in " + path);
        }
        // A "Spline" keyword ends the integral table early in some files that
        // over-declare nGridPoints.
        if (vbcsr::atomic::LowerCopy(line).rfind("spline", 0) == 0) {
            throw std::runtime_error(
                "Truncated .skf: the repulsive Spline block starts after only " +
                std::to_string(row) + " of " + std::to_string(n_grid) + " integral rows in " + path);
        }

        const auto values = skf_detail::ParseDoubles(line);
        if (values.size() < static_cast<size_t>(kSkfValuesPerRow)) {
            throw std::runtime_error(
                "Malformed .skf integral row " + std::to_string(row) + ": expected " +
                std::to_string(kSkfValuesPerRow) + " values, got " + std::to_string(values.size()) +
                " in " + path);
        }

        // Row r sits at distance (r + 1) * gridDist; there is no r = 0 entry.
        table.r.push_back(static_cast<double>(row + 1) * grid_dist_bohr * kAngstromPerBohrLength);
        for (int channel = 0; channel < kSkfChannelCount; ++channel) {
            table.h[static_cast<size_t>(channel)].push_back(
                values[static_cast<size_t>(channel)] * kElectronVoltPerHartree);
            // Overlap integrals are dimensionless and need no conversion.
            table.s[static_cast<size_t>(channel)].push_back(
                values[static_cast<size_t>(channel + kSkfChannelCount)]);
        }
    }

    return table;
}

/// Returns the conventional file name for an ordered species pair.
inline std::string SkfFileName(const std::string& row_species, const std::string& col_species) {
    return vbcsr::atomic::CanonicalSymbol(row_species) + "-" + vbcsr::atomic::CanonicalSymbol(col_species) + ".skf";
}

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_SKF_HPP
