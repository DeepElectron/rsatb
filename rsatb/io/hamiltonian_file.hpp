#ifndef RSATB_IO_HAMILTONIAN_FILE_HPP
#define RSATB_IO_HAMILTONIAN_FILE_HPP

// Format-tagged real-space Hamiltonian input.
//
// A precomputed H(R) can come from several producers with incompatible on-disk
// layouts. The format is therefore an explicit tag on the input rather than an
// assumption baked into the reader: adding a producer means writing one
// HamiltonianReader and registering it in OpenHamiltonianFile, with no change
// anywhere else.

#include "vbcsr/core/atomic/periodic_table.hpp"

#include <array>
#include <functional>
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
    /// ABACUS sparse H(R) (`out_mat_hs2`): one CSR matrix per lattice vector R
    /// over the whole basis, text or binary, in Rydberg and ABACUS orbital
    /// order. `hrs1_nao.csr` in current ABACUS, `data-HR-sparse_SPIN0.csr`
    /// before.
    kAbacusCsr,
    /// NextHam HDF5 (`abacus_atom_pair_operator_v1`, `HR_atom_pairs.h5`):
    /// sorted atom pairs, each with its R list and dense blocks, in ABACUS
    /// orbital order with the energy unit declared in the file.
    kNextHamHdf5,
};

inline std::string ToString(HamiltonianFormat format) {
    switch (format) {
        case HamiltonianFormat::kDeeptbHdf5:
            return "deeptb";
        case HamiltonianFormat::kAbacusCsr:
            return "abacus";
        case HamiltonianFormat::kNextHamHdf5:
            return "nextham";
    }
    throw std::runtime_error("Unhandled HamiltonianFormat.");
}

/// Parses a user-facing format name.
///
/// Args:
///   raw_name: Case-insensitive tag: "deeptb", "abacus" or "nextham".
inline HamiltonianFormat ParseHamiltonianFormat(const std::string& raw_name) {
    const std::string name = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_name));
    if (name == "deeptb" || name == "deeptb_hdf5") {
        return HamiltonianFormat::kDeeptbHdf5;
    }
    if (name == "abacus") {
        return HamiltonianFormat::kAbacusCsr;
    }
    if (name == "nextham") {
        return HamiltonianFormat::kNextHamHdf5;
    }
    throw std::runtime_error(
        "Unsupported Hamiltonian file format '" + raw_name +
        "'. Supported formats: deeptb, abacus, nextham.");
}

/// Whether a format's reader needs the model's FileBasisLayout.
///
/// DeePTB blocks are self-describing and already in rsatb's orbital order.
/// ABACUS and NextHam blocks are in ABACUS's order, which can only be undone
/// knowing each atom's shells, and ABACUS CSR does not even record where one
/// atom's orbitals end and the next one's begin.
inline bool NeedsBasisLayout(HamiltonianFormat format) {
    switch (format) {
        case HamiltonianFormat::kDeeptbHdf5:
            return false;
        case HamiltonianFormat::kAbacusCsr:
        case HamiltonianFormat::kNextHamHdf5:
            return true;
    }
    throw std::runtime_error("Unhandled HamiltonianFormat.");
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

/// The per-atom basis a file is indexed by, as the model knows it.
///
/// Atoms are numbered as the file numbers them: the structure's own order,
/// over the primitive cell when the structure is tiled. Within an atom the
/// orbitals run shell by shell in rsatb's order -- L outer, zeta inner, m
/// inner-most -- which ABACUS shares, so `type_shells` also locates every
/// shell inside an ABACUS block.
struct FileBasisLayout {
    /// Per type: the angular momentum of each radial shell, in block order.
    /// A 2s2p1d basis is {0, 0, 1, 1, 2}.
    std::vector<std::vector<int>> type_shells;
    /// Per type: the atomic number, to cross-check files that record species.
    std::vector<int> type_atomic_number;
    /// Per atom: its type.
    std::vector<int> atom_type;
    /// Basis-size multiplier for one spatial orbital; 2 for a spinor basis.
    int spin_components = 1;

    int n_atom() const { return static_cast<int>(atom_type.size()); }

    /// Spatial orbitals of one type, before the spin expansion.
    int TypeSpatialNorb(int type) const {
        int norb = 0;
        for (int l : type_shells.at(static_cast<size_t>(type))) norb += 2 * l + 1;
        return norb;
    }

    /// Orbitals of one atom, spin-expanded: the dimension of its blocks.
    int AtomNorb(int atom) const {
        return TypeSpatialNorb(atom_type.at(static_cast<size_t>(atom))) * spin_components;
    }

    /// First basis index of every atom, plus the total dimension at the end.
    std::vector<long long> AtomOffsets() const {
        std::vector<long long> offsets(atom_type.size() + 1, 0);
        for (size_t atom = 0; atom < atom_type.size(); ++atom) {
            offsets[atom + 1] = offsets[atom] + AtomNorb(static_cast<int>(atom));
        }
        return offsets;
    }
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

/// One block a rank asks a file for: atom `i` to atom `j` in image `shift`,
/// in the file's own atom numbering.
struct BlockKey {
    int i = 0;
    int j = 0;
    CellShift shift{{0, 0, 0}};

    std::array<int, 5> Tuple() const { return {{i, j, shift[0], shift[1], shift[2]}}; }
    bool operator<(const BlockKey& other) const { return Tuple() < other.Tuple(); }
    bool operator==(const BlockKey& other) const { return Tuple() == other.Tuple(); }
};

/// Receives each block found, with the index of the request it answers.
using BlockSink = std::function<void(size_t request, const DenseBlock& block)>;

/// Reads real-space Hamiltonian blocks, a whole rank's worth at a time.
///
/// The read is one bulk request rather than a call per block because only the
/// reader knows how its file is best read: sorted, coalesced slabs for an
/// indexed HDF5 file, positioned reads of just the needed rows for a binary
/// CSR, or one streaming pass on one rank for a text CSR that cannot be read
/// out of order. Each rank asks only for its own blocks, and nothing a rank
/// did not ask for has to stay resident there.
class HamiltonianReader {
public:
    virtual ~HamiltonianReader() = default;

    virtual HamiltonianFormat format() const = 0;
    virtual FileMetadata metadata() const = 0;

    /// Reads the blocks this rank asks for and hands each one found to `sink`.
    ///
    /// Collective over the communicator the reader was opened with: every
    /// rank calls it, once per read, with its own requests -- possibly none.
    /// A request the file holds no block for is skipped, which is the normal
    /// outcome for pairs outside the Hamiltonian's own cutoff. Blocks reach
    /// the sink on the calling thread, at most once per request, in no
    /// particular order.
    ///
    /// Args:
    ///   requests: Distinct blocks this rank wants.
    ///   sink: Called with (request index, block) for every block found.
    virtual void ReadBlocks(const std::vector<BlockKey>& requests, const BlockSink& sink) const = 0;
};

// The OpenHamiltonianFile factory lives in hamiltonian_readers.hpp, which is
// the single place that knows the full set of implementations. This header
// stays free of any concrete reader so a model can depend on the interface
// alone.

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_HAMILTONIAN_FILE_HPP
