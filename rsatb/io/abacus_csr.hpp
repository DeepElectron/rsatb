#ifndef RSATB_IO_ABACUS_CSR_HPP
#define RSATB_IO_ABACUS_CSR_HPP

// HamiltonianReader for ABACUS's sparse H(R) output (`out_mat_hs2 1`).
//
// Layout, text and binary alike: a header giving the basis dimension n and the
// number of R vectors, then for each R its integer components, its nonzero
// count nnz and -- when nnz > 0 -- one n x n CSR matrix: nnz values, nnz
// column indices, n + 1 row offsets. Rows and columns run over the whole
// basis, atom by atom in structure order, in Rydberg.
//
//   text:   "STEP: s" (ABACUS >= 3.0), "Matrix Dimension of H(R): n",
//           "Matrix number of H(R): nR", then per R the line "Rx Ry Rz nnz"
//           and, when nnz > 0, one line each of values, columns, row offsets.
//   binary: int32 step, n, nR; per R int32 Rx, Ry, Rz, nnz and, when nnz > 0,
//           nnz float64 values, nnz int32 columns, n + 1 int32 row offsets.
//
// An R with nnz = 0 is followed by nothing in current ABACUS; older versions
// wrote an all-zero row-offset array there. Both are accepted.
//
// The file has no atom boundaries, so the reader takes them from the model's
// FileBasisLayout, and it needs that layout's shells anyway to put each block
// into rsatb's orbital order (abacus_convention.hpp). Nothing can be looked up
// in a CSR file without scanning it, so the whole file is read once at open
// and held as dense blocks; ABACUS-sized systems fit comfortably.

#include "abacus_convention.hpp"
#include "hamiltonian_file.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

class AbacusCsrReader : public HamiltonianReader {
public:
    /// Reads `path` against the model's basis.
    ///
    /// Args:
    ///   path: An ABACUS H(R) file, e.g. OUT.<suffix>/hrs1_nao.csr.
    ///   layout: The model's basis over the file's atoms.
    /// Throws:
    ///   std::runtime_error if the file is malformed, holds complex (nspin=4)
    ///   or several MD steps of data, or its dimension disagrees with layout.
    AbacusCsrReader(const std::string& path, const FileBasisLayout& layout) : path_(path) {
        ValidateLayout(layout, "ABACUS");
        RequireSpinlessLayout(layout, "ABACUS");
        layout_ = layout;
        orders_ = MakeAbacusOrbitalOrders(layout_);
        offsets_ = layout_.AtomOffsets();

        std::ifstream probe(path, std::ios::binary);
        if (!probe) {
            throw std::runtime_error("Cannot open ABACUS Hamiltonian file " + path + ".");
        }
        char head[6] = {0, 0, 0, 0, 0, 0};
        probe.read(head, 6);
        const std::string start(head, static_cast<size_t>(probe.gcount()));
        probe.close();

        // Every text variant opens with one of these two words.
        try {
            if (start.compare(0, 4, "STEP") == 0 || start.compare(0, 6, "Matrix") == 0) {
                ReadText();
            } else {
                ReadBinary();
            }
        } catch (const std::logic_error& error) {  // std::stoi and friends
            throw Malformed(std::string("unreadable number (") + error.what() + ").");
        }
    }

    HamiltonianFormat format() const override { return HamiltonianFormat::kAbacusCsr; }

    FileMetadata metadata() const override {
        FileMetadata metadata;
        metadata.n_orbital_total = static_cast<int>(dimension_);
        return metadata;
    }

    bool ReadBlock(int i, int j, const CellShift& shift, DenseBlock* out) const override {
        if (out == nullptr) {
            throw std::runtime_error("AbacusCsrReader::ReadBlock requires a non-null output block.");
        }
        const auto found = blocks_.find(Key{{i, j, shift[0], shift[1], shift[2]}});
        if (found == blocks_.end()) {
            return false;
        }
        *out = found->second;
        return true;
    }

private:
    using Key = std::array<int, 5>;  // i, j, Rx, Ry, Rz

    // One R's CSR arrays as read, before they are cut into atom blocks.
    struct CsrMatrix {
        std::array<int, 3> shift{{0, 0, 0}};
        std::vector<double> values;
        std::vector<int> columns;
        std::vector<int> row_offsets;
    };

    std::runtime_error Malformed(const std::string& what) const {
        return std::runtime_error("Malformed ABACUS Hamiltonian file " + path_ + ": " + what);
    }

    void SetDimension(long long dimension) {
        if (dimension <= 0) {
            throw Malformed("non-positive matrix dimension " + std::to_string(dimension) + ".");
        }
        dimension_ = dimension;
        if (dimension_ != offsets_.back()) {
            throw std::runtime_error(
                "ABACUS Hamiltonian file " + path_ + " has dimension " +
                std::to_string(dimension_) + ", but the model's basis over the " +
                std::to_string(layout_.n_atom()) + " atoms of this structure has " +
                std::to_string(offsets_.back()) + " orbitals. Check that the structure lists "
                "the atoms in ABACUS's order, that the orbital files are the ones ABACUS used, "
                "and that the run was spin-degenerate (nspin=4 doubles the dimension).");
        }
        atom_of_index_.assign(static_cast<size_t>(dimension_), 0);
        for (int atom = 0; atom < layout_.n_atom(); ++atom) {
            for (long long index = offsets_[atom]; index < offsets_[atom + 1]; ++index) {
                atom_of_index_[static_cast<size_t>(index)] = atom;
            }
        }
    }

    // --- text ---------------------------------------------------------------

    static bool NextContentLine(std::istream& in, std::string* line) {
        while (std::getline(in, *line)) {
            if (line->find_first_not_of(" \t\r") != std::string::npos) return true;
        }
        return false;
    }

    static long long TrailingInteger(const std::string& line) {
        std::istringstream words(line);
        std::string word, last;
        while (words >> word) last = word;
        return std::stoll(last);
    }

    static std::vector<std::string> Words(const std::string& line) {
        std::istringstream words(line);
        std::vector<std::string> out;
        std::string word;
        while (words >> word) out.push_back(word);
        return out;
    }

    void ReadText() {
        std::ifstream in(path_);
        std::string line;
        if (!NextContentLine(in, &line)) throw Malformed("the file is empty.");
        if (line.compare(0, 4, "STEP") == 0 && !NextContentLine(in, &line)) {
            throw Malformed("the header ends after the STEP line.");
        }
        if (line.find("Dimension") == std::string::npos) {
            throw Malformed("expected the 'Matrix Dimension of H(R)' line, got '" + line + "'.");
        }
        SetDimension(TrailingInteger(line));
        if (!NextContentLine(in, &line) || line.find("number") == std::string::npos) {
            throw Malformed("expected the 'Matrix number of H(R)' line.");
        }
        const long long n_shift = TrailingInteger(line);

        bool have_line = NextContentLine(in, &line);
        for (long long r = 0; r < n_shift; ++r) {
            if (!have_line) throw Malformed("the file ends before all R records are read.");
            const auto header = Words(line);
            if (header.size() != 4) {
                throw Malformed("expected an 'Rx Ry Rz nnz' line, got '" + line + "'.");
            }
            CsrMatrix matrix;
            matrix.shift = {{std::stoi(header[0]), std::stoi(header[1]), std::stoi(header[2])}};
            const long long nnz = std::stoll(header[3]);
            if (nnz < 0) throw Malformed("negative nonzero count.");

            if (nnz == 0) {
                // Older ABACUS wrote an all-zero row-offset line here.
                have_line = NextContentLine(in, &line);
                if (have_line && Words(line).size() == static_cast<size_t>(dimension_ + 1)) {
                    have_line = NextContentLine(in, &line);
                }
                continue;
            }

            if (!NextContentLine(in, &line)) throw Malformed("missing values line.");
            if (line.find('(') != std::string::npos) {
                throw std::runtime_error(
                    "ABACUS Hamiltonian file " + path_ + " holds complex values, as an nspin=4 "
                    "run writes; rsatb's file-backed Hamiltonians are real-valued.");
            }
            matrix.values = ParseNumbers<double>(line, nnz, "values");
            if (!NextContentLine(in, &line)) throw Malformed("missing column-index line.");
            matrix.columns = ParseNumbers<int>(line, nnz, "column indices");
            if (!NextContentLine(in, &line)) throw Malformed("missing row-offset line.");
            matrix.row_offsets = ParseNumbers<int>(line, dimension_ + 1, "row offsets");
            AddMatrix(matrix);
            have_line = NextContentLine(in, &line);
        }

        if (have_line) {
            if (line.compare(0, 4, "STEP") == 0) {
                throw std::runtime_error(
                    "ABACUS Hamiltonian file " + path_ + " holds more than one STEP (an MD run "
                    "with out_app_flag); rsatb reads single-structure files. Split off the "
                    "step that matches the structure.");
            }
            throw Malformed("unexpected content after the last R record: '" + line + "'.");
        }
    }

    template <typename T>
    std::vector<T> ParseNumbers(const std::string& line, long long expected, const char* what) const {
        std::istringstream words(line);
        std::vector<T> out;
        out.reserve(static_cast<size_t>(expected));
        T value;
        while (words >> value) out.push_back(value);
        if (!words.eof() || static_cast<long long>(out.size()) != expected) {
            throw Malformed(std::string("expected ") + std::to_string(expected) + " " + what +
                            ", read " + std::to_string(out.size()) + ".");
        }
        return out;
    }

    // --- binary -------------------------------------------------------------

    void ReadBinary() {
        std::ifstream in(path_, std::ios::binary);
        in.seekg(0, std::ios::end);
        const auto size = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        size_t cursor = 0;

        auto remaining = [&]() { return size - cursor; };
        auto take = [&](void* out, size_t bytes, const char* what) {
            if (remaining() < bytes) {
                throw Malformed(std::string("the file ends inside ") + what +
                                " (a complex nspin=4 file does not parse as real data).");
            }
            in.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes));
            cursor += bytes;
        };
        auto take_int = [&](const char* what) {
            std::int32_t value = 0;
            take(&value, sizeof(value), what);
            return static_cast<long long>(value);
        };

        take_int("the header");  // step
        SetDimension(take_int("the header"));
        const long long n_shift = take_int("the header");
        if (n_shift < 0) throw Malformed("negative R count.");
        const size_t offsets_bytes = static_cast<size_t>(dimension_ + 1) * sizeof(std::int32_t);

        for (long long r = 0; r < n_shift; ++r) {
            CsrMatrix matrix;
            for (int axis = 0; axis < 3; ++axis) {
                matrix.shift[axis] = static_cast<int>(take_int("an R header"));
            }
            const long long nnz = take_int("an R header");
            if (nnz < 0) throw Malformed("negative nonzero count.");

            if (nnz == 0) {
                // Older ABACUS wrote an all-zero row-offset array here. A real
                // R header that follows is never all zero: its nnz is positive.
                if (remaining() >= offsets_bytes) {
                    std::vector<char> ahead(offsets_bytes);
                    in.read(ahead.data(), static_cast<std::streamsize>(offsets_bytes));
                    const bool all_zero =
                        std::all_of(ahead.begin(), ahead.end(), [](char b) { return b == 0; });
                    if (all_zero) {
                        cursor += offsets_bytes;
                    } else {
                        in.seekg(static_cast<std::streamoff>(cursor), std::ios::beg);
                    }
                }
                continue;
            }

            matrix.values.resize(static_cast<size_t>(nnz));
            take(matrix.values.data(), matrix.values.size() * sizeof(double), "values");
            std::vector<std::int32_t> raw(static_cast<size_t>(nnz));
            take(raw.data(), raw.size() * sizeof(std::int32_t), "column indices");
            matrix.columns.assign(raw.begin(), raw.end());
            raw.resize(static_cast<size_t>(dimension_ + 1));
            take(raw.data(), raw.size() * sizeof(std::int32_t), "row offsets");
            matrix.row_offsets.assign(raw.begin(), raw.end());
            AddMatrix(matrix);
        }

        if (remaining() != 0) {
            throw Malformed(std::to_string(remaining()) + " bytes follow the last R record (a "
                            "multi-step MD file, or complex nspin=4 data read as real).");
        }
    }

    // --- shared -------------------------------------------------------------

    /// Cuts one R's matrix into atom blocks and stores them in rsatb's order.
    void AddMatrix(const CsrMatrix& matrix) {
        const long long n = dimension_;
        const auto& offsets = matrix.row_offsets;
        if (offsets.front() != 0 ||
            offsets.back() != static_cast<long long>(matrix.values.size())) {
            throw Malformed("row offsets do not span the nonzeros (a complex nspin=4 file "
                            "does not parse as real data).");
        }

        std::map<Key, std::vector<double>> raw;  // ABACUS-order blocks of this R
        for (long long row = 0; row < n; ++row) {
            const int begin = offsets[static_cast<size_t>(row)];
            const int end = offsets[static_cast<size_t>(row + 1)];
            if (end < begin) throw Malformed("row offsets decrease.");
            const int atom_i = atom_of_index_[static_cast<size_t>(row)];
            const long long a = row - offsets_[atom_i];
            for (int k = begin; k < end; ++k) {
                const int column = matrix.columns[static_cast<size_t>(k)];
                if (column < 0 || column >= n) {
                    throw Malformed("column index " + std::to_string(column) + " is outside [0, " +
                                    std::to_string(n) + ").");
                }
                const int atom_j = atom_of_index_[static_cast<size_t>(column)];
                const Key key{{atom_i, atom_j, matrix.shift[0], matrix.shift[1], matrix.shift[2]}};
                auto& block = raw[key];
                const int ncol = layout_.AtomNorb(atom_j);
                if (block.empty()) {
                    block.assign(static_cast<size_t>(layout_.AtomNorb(atom_i)) * ncol, 0.0);
                }
                // CSR duplicates are additive; ABACUS never writes any.
                block[static_cast<size_t>(a) * ncol + (column - offsets_[atom_j])] +=
                    matrix.values[static_cast<size_t>(k)];
            }
        }

        for (const auto& entry : raw) {
            const Key& key = entry.first;
            if (blocks_.count(key) != 0) {
                throw Malformed("lattice vector (" + std::to_string(key[2]) + ", " +
                                std::to_string(key[3]) + ", " + std::to_string(key[4]) +
                                ") appears twice.");
            }
            DenseBlock block;
            AbacusBlockToRsatb(orders_[layout_.atom_type[key[0]]],
                               orders_[layout_.atom_type[key[1]]],
                               entry.second.data(), kAbacusEvPerRydberg, &block);
            blocks_.emplace(key, std::move(block));
        }
    }

    std::string path_;
    FileBasisLayout layout_;
    std::vector<AbacusOrbitalOrder> orders_;
    std::vector<long long> offsets_;
    std::vector<int> atom_of_index_;
    long long dimension_ = 0;
    std::map<Key, DenseBlock> blocks_;
};

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_ABACUS_CSR_HPP
