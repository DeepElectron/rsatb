#ifndef RSATB_IO_HAMILTONIAN_READERS_HPP
#define RSATB_IO_HAMILTONIAN_READERS_HPP

// The one place that knows every HamiltonianReader implementation.
//
// To add a format: write the reader, include it here, and add one case to
// OpenHamiltonianFile. Nothing outside this header needs to change.

#include "abacus_csr.hpp"
#include "hamiltonian_file.hpp"
#include "hdf5_deeptb.hpp"
#include "nextham_hdf5.hpp"

#include <memory>

namespace rsatb {
namespace io {

/// Opens `file` with the reader registered for its format.
///
/// Args:
///   file: The path and its format tag.
///   layout: The model's basis over the file's atoms, for formats where
///     NeedsBasisLayout is true; ignored by the others.
inline std::unique_ptr<HamiltonianReader> OpenHamiltonianFile(
    const HamiltonianFile& file,
    const FileBasisLayout& layout
) {
    if (file.empty()) {
        throw std::runtime_error("Hamiltonian file path must not be empty.");
    }

    switch (file.format) {
        case HamiltonianFormat::kDeeptbHdf5:
            return std::make_unique<DeeptbHdf5Reader>(file.path);
        case HamiltonianFormat::kAbacusCsr:
            return std::make_unique<AbacusCsrReader>(file.path, layout);
        case HamiltonianFormat::kNextHamHdf5:
            return std::make_unique<NextHamHdf5Reader>(file.path, layout);
    }

    throw std::runtime_error(
        "No reader registered for Hamiltonian format '" + ToString(file.format) + "'.");
}

/// Opens `file` for a caller that has no basis layout to offer.
///
/// Throws:
///   std::runtime_error for a format that needs one, naming why.
inline std::unique_ptr<HamiltonianReader> OpenHamiltonianFile(const HamiltonianFile& file) {
    if (NeedsBasisLayout(file.format)) {
        throw std::runtime_error(
            "The '" + ToString(file.format) + "' Hamiltonian format is written in ABACUS's "
            "orbital order, which can only be converted knowing each atom's orbital shells; "
            "open it through a model that reads them from orbital files (LCAO).");
    }
    return OpenHamiltonianFile(file, FileBasisLayout{});
}

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_HAMILTONIAN_READERS_HPP
