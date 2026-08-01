#ifndef RSATB_IO_HAMILTONIAN_READERS_HPP
#define RSATB_IO_HAMILTONIAN_READERS_HPP

// The one place that knows every HamiltonianReader implementation.
//
// To add a format: write the reader, include it here, and add one case to
// OpenHamiltonianFile. Nothing outside this header needs to change.

#include "hamiltonian_file.hpp"
#include "hdf5_deeptb.hpp"

#include <memory>

namespace rsatb {
namespace io {

/// Opens `file` with the reader registered for its format.
inline std::unique_ptr<HamiltonianReader> OpenHamiltonianFile(const HamiltonianFile& file) {
    if (file.empty()) {
        throw std::runtime_error("Hamiltonian file path must not be empty.");
    }

    switch (file.format) {
        case HamiltonianFormat::kDeeptbHdf5:
            return std::make_unique<DeeptbHdf5Reader>(file.path);
    }

    throw std::runtime_error(
        "No reader registered for Hamiltonian format '" + ToString(file.format) + "'.");
}

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_HAMILTONIAN_READERS_HPP
