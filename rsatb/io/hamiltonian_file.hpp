#ifndef RSATB_IO_HAMILTONIAN_FILE_HPP
#define RSATB_IO_HAMILTONIAN_FILE_HPP

// Format-tagged real-space Hamiltonian input.
//
// A precomputed H(R) can come from several producers with incompatible on-disk
// layouts. The format is therefore an explicit tag on the input rather than an
// assumption baked into the reader: adding a producer means writing one
// HamiltonianReader and registering it in OpenHamiltonianFile, with no change
// anywhere else.

#include "periodic_table.hpp"

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

/// On-disk layout of a real-space Hamiltonian file.
enum class HamiltonianFormat {
    /// DeePTB HDF5: group "0", one dataset per block named "i_j_Rx_Ry_Rz",
    /// shaped [m, n] for real data or [m, n, 2] for complex.
    kDeeptbHdf5,
};

inline std::string ToString(HamiltonianFormat format) {
    switch (format) {
        case HamiltonianFormat::kDeeptbHdf5:
            return "deeptb";
    }
    throw std::runtime_error("Unhandled HamiltonianFormat.");
}

/// Parses a user-facing format name.
///
/// Args:
///   raw_name: Case-insensitive tag, e.g. "deeptb".
inline HamiltonianFormat ParseHamiltonianFormat(const std::string& raw_name) {
    const std::string name = LowerCopy(TrimCopy(raw_name));
    if (name == "deeptb" || name == "deeptb_hdf5") {
        return HamiltonianFormat::kDeeptbHdf5;
    }
    throw std::runtime_error(
        "Unsupported Hamiltonian file format '" + raw_name + "'. Supported formats: deeptb.");
}

/// A real-space Hamiltonian file and the layout it uses.
struct HamiltonianFile {
    std::string path;
    HamiltonianFormat format = HamiltonianFormat::kDeeptbHdf5;

    bool empty() const { return path.empty(); }
};

/// What a file states about itself.
///
/// Every field is optional because formats differ in what they record. A model
/// cross-checks the fields that are present against its own BasisSpec and
/// fails on disagreement, instead of trusting user-declared values.
struct FileMetadata {
    std::optional<int> n_orbital_total;
    std::optional<std::vector<int>> type_norb;
    std::optional<int> n_spin_component;
};

using CellShift = std::array<int, 3>;

/// A dense matrix block in row-major order.
struct DenseBlock {
    int rows = 0;
    int cols = 0;
    std::vector<double> values;

    void Resize(int n_rows, int n_cols) {
        rows = n_rows;
        cols = n_cols;
        values.assign(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols), 0.0);
    }
};

/// Reads real-space Hamiltonian blocks for one (i, j, R) triple at a time.
///
/// Implementations must support concurrent ReadBlock calls from multiple
/// threads, because the image-container build loops are OpenMP-parallel over
/// atoms and edges.
class HamiltonianReader {
public:
    virtual ~HamiltonianReader() = default;

    virtual HamiltonianFormat format() const = 0;
    virtual FileMetadata metadata() const = 0;

    /// Reads the block coupling atom `i` to atom `j` in image `shift`.
    ///
    /// Args:
    ///   i, j: Global atom indices.
    ///   shift: Lattice image of atom `j`.
    ///   out: Receives the block, row-major, when the return value is true.
    /// Returns:
    ///   False when the file holds no block for this triple, which is the
    ///   normal outcome for pairs outside the Hamiltonian's own cutoff.
    virtual bool ReadBlock(int i, int j, const CellShift& shift, DenseBlock* out) const = 0;
};

// The OpenHamiltonianFile factory lives in hamiltonian_readers.hpp, which is
// the single place that knows the full set of implementations. This header
// stays free of any concrete reader so a model can depend on the interface
// alone.

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_HAMILTONIAN_FILE_HPP
