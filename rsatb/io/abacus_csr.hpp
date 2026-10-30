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
//           "Matrix number of H(R): nR", then per R "Rx Ry Rz nnz" and, when
//           nnz > 0, nnz values, nnz columns and n + 1 row offsets.
//   binary: int32 step, n, nR; per R int32 Rx, Ry, Rz, nnz and, when nnz > 0,
//           nnz float64 values, nnz int32 columns, n + 1 int32 row offsets.
//
// A binary R with nnz = 0 is followed by nothing in current ABACUS; older
// versions wrote an all-zero row-offset array there, and both are accepted.
// Text with nnz = 0 is followed by nothing, as current ABACUS writes it.
//
// The file has no atom boundaries, so they come from the model's
// FileBasisLayout, whose shells also put each block into rsatb's orbital
// order (abacus_convention.hpp).
//
// How a rank gets only its own rows:
//
//   binary  Random access. At open, rank 0 walks the record headers -- seeking
//           past the data -- and broadcasts where every record's values,
//           columns and row offsets start. Each rank then reads, per R, just
//           the row offsets, columns and values of its own atoms' rows, one
//           positioned read per run of consecutive atoms. Fully parallel;
//           memory is the rank's own rows.
//   text    No random access: a value's byte position depends on every value
//           before it. Rank 0 streams the file once with a buffered parser,
//           one R record at a time, and sends each rank the rows it asked for.
//           Rank 0 holds one R record; every other rank only its own rows.
//           The parse is serial, so for very large systems the binary output
//           is the one to use.

#include "abacus_convention.hpp"
#include "hamiltonian_file.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rsatb {
namespace io {

namespace abacus_detail {

/// Throws on every rank if any rank has an error message, with that message.
///
/// An error found on one rank inside a collective read must reach all of
/// them, or the others wait forever for a message that is never sent.
inline void AgreeOrThrow(MPI_Comm comm, const std::string& local_error) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const int mine = local_error.empty() ? -1 : rank;
    int first = -1;
    MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MAX, comm);
    if (first < 0) return;
    int length = rank == first ? static_cast<int>(local_error.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, first, comm);
    std::string message = rank == first ? local_error : std::string(static_cast<size_t>(length), ' ');
    MPI_Bcast(&message[0], length, MPI_CHAR, first, comm);
    throw std::runtime_error(message);
}

// Byte buffers of any size, sent in chunks MPI's int counts allow.
constexpr std::int64_t kChunk = 1LL << 30;

inline void SendBytes(const std::vector<char>& bytes, int to, MPI_Comm comm) {
    std::int64_t size = static_cast<std::int64_t>(bytes.size());
    MPI_Send(&size, 1, MPI_INT64_T, to, 0, comm);
    for (std::int64_t at = 0; at < size; at += kChunk) {
        MPI_Send(bytes.data() + at, static_cast<int>(std::min(kChunk, size - at)), MPI_BYTE, to, 1, comm);
    }
}

inline std::vector<char> RecvBytes(int from, MPI_Comm comm) {
    std::int64_t size = 0;
    MPI_Recv(&size, 1, MPI_INT64_T, from, 0, comm, MPI_STATUS_IGNORE);
    std::vector<char> bytes(static_cast<size_t>(size));
    for (std::int64_t at = 0; at < size; at += kChunk) {
        MPI_Recv(bytes.data() + at, static_cast<int>(std::min(kChunk, size - at)), MPI_BYTE, from, 1,
                 comm, MPI_STATUS_IGNORE);
    }
    return bytes;
}

template <typename T>
void Append(std::vector<char>* out, const T* values, size_t count) {
    const size_t at = out->size();
    out->resize(at + count * sizeof(T));
    if (count) std::memcpy(out->data() + at, values, count * sizeof(T));
}

// Copies out, since fields of mixed width leave later ones unaligned.
template <typename T>
std::vector<T> Take(const std::vector<char>& in, size_t* cursor, size_t count) {
    if (*cursor + count * sizeof(T) > in.size()) {
        throw std::runtime_error("ABACUS row message is shorter than its own counts.");
    }
    std::vector<T> values(count);
    if (count) std::memcpy(values.data(), in.data() + *cursor, count * sizeof(T));
    *cursor += count * sizeof(T);
    return values;
}

/// Whitespace-separated tokens of a text file, read through a large buffer.
class Tokens {
public:
    explicit Tokens(const std::string& path) : in_(path, std::ios::binary), buffer_(kBuffer) {
        if (!in_) throw std::runtime_error("Cannot open ABACUS Hamiltonian file " + path + ".");
    }

    /// Moves to the next token; false at end of file. The view lives until
    /// the next call.
    bool Next(std::string_view* token) {
        while (true) {
            while (pos_ < end_ && IsSpace(buffer_[pos_])) ++pos_;
            if (pos_ < end_) break;
            if (!Fill()) return false;
        }
        size_t stop = pos_;
        while (true) {
            while (stop < end_ && !IsSpace(buffer_[stop])) ++stop;
            if (stop < end_) break;
            const size_t length = stop - pos_;
            if (!Fill()) break;  // the token ends the file
            stop = pos_ + length;
        }
        *token = std::string_view(buffer_.data() + pos_, stop - pos_);
        pos_ = stop;
        return true;
    }

private:
    static constexpr size_t kBuffer = size_t{1} << 24;

    static bool IsSpace(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

    // Keeps [pos_, end_) and reads more after it.
    bool Fill() {
        if (eof_) return false;
        std::memmove(buffer_.data(), buffer_.data() + pos_, end_ - pos_);
        end_ -= pos_;
        pos_ = 0;
        in_.read(buffer_.data() + end_, static_cast<std::streamsize>(buffer_.size() - end_));
        const size_t got = static_cast<size_t>(in_.gcount());
        if (got == 0) eof_ = true;
        end_ += got;
        return got > 0;
    }

    std::ifstream in_;
    std::vector<char> buffer_;
    size_t pos_ = 0;
    size_t end_ = 0;
    bool eof_ = false;
};

}  // namespace abacus_detail

class AbacusCsrReader : public HamiltonianReader {
public:
    /// Opens `path` against the model's basis. Collective over `comm`.
    ///
    /// Args:
    ///   path: An ABACUS H(R) file, e.g. OUT.<suffix>/hrs1_nao.csr.
    ///   layout: The model's basis over the file's atoms.
    ///   comm: The ranks that will read it.
    /// Throws:
    ///   std::runtime_error, on every rank, if the file is malformed or its
    ///   dimension disagrees with layout.
    AbacusCsrReader(const std::string& path, const FileBasisLayout& layout, MPI_Comm comm)
        : path_(path), comm_(comm) {
        ValidateLayout(layout, "ABACUS");
        RequireSpinlessLayout(layout, "ABACUS");
        layout_ = layout;
        orders_ = MakeAbacusOrbitalOrders(layout_);
        offsets_ = layout_.AtomOffsets();
        MPI_Comm_rank(comm_, &rank_);

        std::string error;
        std::array<long long, 3> header{{0, 0, 0}};  // binary?, dimension, records
        if (rank_ == 0) {
            try {
                header = ReadHeaderOnRoot();
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        abacus_detail::AgreeOrThrow(comm_, error);
        MPI_Bcast(header.data(), 3, MPI_LONG_LONG, 0, comm_);
        binary_ = header[0] != 0;
        dimension_ = header[1];
        n_records_ = header[2];

        if (binary_) {
            long long n_entries = static_cast<long long>(records_.size());
            MPI_Bcast(&n_entries, 1, MPI_LONG_LONG, 0, comm_);
            records_.resize(static_cast<size_t>(n_entries));
            MPI_Bcast(records_.data(), static_cast<int>(n_entries * sizeof(Record)), MPI_BYTE, 0, comm_);
            for (size_t r = 0; r < records_.size(); ++r) record_of_[records_[r].shift] = r;
        }
    }

    HamiltonianFormat format() const override { return HamiltonianFormat::kAbacusCsr; }

    FileMetadata metadata() const override {
        FileMetadata metadata;
        metadata.n_orbital_total = static_cast<int>(dimension_);
        return metadata;
    }

    void ReadBlocks(const std::vector<BlockKey>& requests, const BlockSink& sink) const override {
        std::string error;
        for (const BlockKey& key : requests) {
            if (key.i < 0 || key.i >= layout_.n_atom() || key.j < 0 || key.j >= layout_.n_atom()) {
                error = "ABACUS Hamiltonian file " + path_ + " was asked for block (" +
                        std::to_string(key.i) + ", " + std::to_string(key.j) +
                        "), outside its " + std::to_string(layout_.n_atom()) + " atoms.";
                break;
            }
        }
        abacus_detail::AgreeOrThrow(comm_, error);

        const auto rows = GroupByRow(requests);
        if (binary_) {
            ReadBinaryRows(rows, sink);
        } else {
            ReadTextRows(rows, sink);
        }
    }

private:
    // Where one binary R record's arrays start; nnz = 0 records have none.
    struct Record {
        std::array<int, 3> shift;
        std::int64_t nnz;
        std::int64_t values, columns, row_offsets;  // byte positions
    };

    // This rank's requests for one (R, atom i): which j, and the request.
    struct RowRequest {
        std::array<int, 3> shift;
        int atom;
        std::vector<std::pair<int, size_t>> columns;  // (j, request), sorted by j
    };

    std::runtime_error Malformed(const std::string& what) const {
        return std::runtime_error("Malformed ABACUS Hamiltonian file " + path_ + ": " + what);
    }

    void CheckDimension(long long dimension) const {
        if (dimension <= 0) {
            throw Malformed("non-positive matrix dimension " + std::to_string(dimension) + ".");
        }
        if (dimension != offsets_.back()) {
            throw std::runtime_error(
                "ABACUS Hamiltonian file " + path_ + " has dimension " + std::to_string(dimension) +
                ", but the model's basis over the " + std::to_string(layout_.n_atom()) +
                " atoms of this structure has " + std::to_string(offsets_.back()) +
                " orbitals. Check that the structure lists the atoms in ABACUS's order, that "
                "the orbital files are the ones ABACUS used, and that the run was "
                "spin-degenerate (nspin=4 doubles the dimension).");
        }
    }

    /// Checks rows [0, n_rows) of a CSR slice: offsets non-decreasing, columns
    /// in range and strictly ascending within a row, which EmitRows relies on.
    template <typename Offset>
    void CheckRows(const Offset* row_ptr, std::int64_t n_rows, const std::int32_t* cols) const {
        for (std::int64_t a = 0; a < n_rows; ++a) {
            if (row_ptr[a + 1] < row_ptr[a]) throw Malformed("row offsets decrease.");
            for (std::int64_t k = row_ptr[a]; k < row_ptr[a + 1]; ++k) {
                if (cols[k] < 0 || cols[k] >= dimension_) {
                    throw Malformed("column index " + std::to_string(cols[k]) + " is outside [0, " +
                                    std::to_string(dimension_) + ").");
                }
                if (k > row_ptr[a] && cols[k] <= cols[k - 1]) {
                    throw Malformed("a row's column indices are not in ascending order.");
                }
            }
        }
    }

    // --- open (rank 0) ------------------------------------------------------

    std::array<long long, 3> ReadHeaderOnRoot() {
        std::ifstream probe(path_, std::ios::binary);
        if (!probe) {
            throw std::runtime_error("Cannot open ABACUS Hamiltonian file " + path_ + ".");
        }
        char head[6] = {0, 0, 0, 0, 0, 0};
        probe.read(head, 6);
        const std::string start(head, static_cast<size_t>(probe.gcount()));
        probe.close();

        // Every text variant opens with one of these two words.
        if (start.compare(0, 4, "STEP") == 0 || start.compare(0, 6, "Matrix") == 0) {
            abacus_detail::Tokens tokens(path_);
            long long dimension = 0, n_records = 0;
            ReadTextHeader(&tokens, &dimension, &n_records);
            return {{0, dimension, n_records}};
        }
        return ScanBinary();
    }

    // Walks the binary record headers, seeking past the data.
    std::array<long long, 3> ScanBinary() {
        std::ifstream in(path_, std::ios::binary);
        in.seekg(0, std::ios::end);
        const std::int64_t size = static_cast<std::int64_t>(in.tellg());
        std::int64_t at = 0;
        auto read_ints = [&](std::int32_t* out, int count, const char* what) {
            if (at + 4 * count > size) {
                throw Malformed(std::string("the file ends inside ") + what +
                                " (a complex nspin=4 file does not parse as real data).");
            }
            in.seekg(at);
            in.read(reinterpret_cast<char*>(out), 4 * count);
            at += 4 * count;
        };

        std::int32_t header[3];
        read_ints(header, 3, "the header");
        CheckDimension(header[1]);
        const std::int64_t n = header[1];
        if (header[2] < 0) throw Malformed("negative R count.");
        const std::int64_t offsets_bytes = 4 * (n + 1);

        std::map<std::array<int, 3>, int> seen;
        for (std::int32_t r = 0; r < header[2]; ++r) {
            std::int32_t h[4];
            read_ints(h, 4, "an R header");
            Record record{{{h[0], h[1], h[2]}}, h[3], 0, 0, 0};
            if (record.nnz < 0) throw Malformed("negative nonzero count.");
            if (seen[record.shift]++) {
                throw Malformed("lattice vector (" + std::to_string(h[0]) + ", " + std::to_string(h[1]) +
                                ", " + std::to_string(h[2]) + ") appears twice.");
            }
            if (record.nnz == 0) {
                // Older ABACUS wrote an all-zero row-offset array here. A real
                // R header that follows is never all zero: its nnz is positive.
                if (at + offsets_bytes <= size) {
                    std::vector<char> ahead(static_cast<size_t>(offsets_bytes));
                    in.seekg(at);
                    in.read(ahead.data(), offsets_bytes);
                    if (std::all_of(ahead.begin(), ahead.end(), [](char b) { return b == 0; })) {
                        at += offsets_bytes;
                    }
                }
                continue;
            }
            record.values = at;
            record.columns = at + 8 * record.nnz;
            record.row_offsets = record.columns + 4 * record.nnz;
            at = record.row_offsets + offsets_bytes;
            if (at > size) {
                throw Malformed("the file ends inside an R record (a complex nspin=4 file does "
                                "not parse as real data).");
            }
            std::int32_t first = -1, last = -1;
            in.seekg(record.row_offsets);
            in.read(reinterpret_cast<char*>(&first), 4);
            in.seekg(record.row_offsets + 4 * n);
            in.read(reinterpret_cast<char*>(&last), 4);
            if (first != 0 || last != record.nnz) {
                throw Malformed("row offsets do not span the nonzeros (a complex nspin=4 file "
                                "does not parse as real data).");
            }
            records_.push_back(record);
        }
        if (at != size) {
            throw Malformed(std::to_string(size - at) + " bytes follow the last R record (a "
                            "multi-step MD file, or complex nspin=4 data read as real).");
        }
        return {{1, n, header[2]}};
    }

    // --- shared -------------------------------------------------------------

    // Requests grouped by (R, atom i), in (R, i, j) order.
    std::vector<RowRequest> GroupByRow(const std::vector<BlockKey>& requests) const {
        std::vector<size_t> order(requests.size());
        std::iota(order.begin(), order.end(), size_t{0});
        auto key_of = [&](size_t k) {
            const BlockKey& key = requests[k];
            return std::array<int, 5>{{key.shift[0], key.shift[1], key.shift[2], key.i, key.j}};
        };
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key_of(a) < key_of(b); });
        std::vector<RowRequest> rows;
        for (size_t k : order) {
            const BlockKey& key = requests[k];
            if (rows.empty() || rows.back().shift != key.shift || rows.back().atom != key.i) {
                rows.push_back(RowRequest{key.shift, key.i, {}});
            }
            rows.back().columns.emplace_back(key.j, k);
        }
        return rows;
    }

    /// Cuts one atom's rows into the requested blocks and hands them on.
    ///
    /// Args:
    ///   row: The requests for this (R, atom).
    ///   row_ptr: na + 1 offsets into cols / values, one per row of the atom.
    template <typename Offset>
    void EmitRows(const RowRequest& row, const Offset* row_ptr, const std::int32_t* cols,
                  const double* values, const BlockSink& sink) const {
        const int ni = layout_.AtomNorb(row.atom);
        std::vector<double> abacus;
        DenseBlock block;
        for (const auto& wanted : row.columns) {
            const int j = wanted.first;
            const int nj = layout_.AtomNorb(j);
            const std::int32_t lo = static_cast<std::int32_t>(offsets_[j]);
            const std::int32_t hi = lo + nj;
            abacus.assign(static_cast<size_t>(ni) * nj, 0.0);
            bool found = false;
            for (int a = 0; a < ni; ++a) {
                const std::int32_t* begin = cols + row_ptr[a];
                const std::int32_t* end = cols + row_ptr[a + 1];
                for (const std::int32_t* c = std::lower_bound(begin, end, lo); c != end && *c < hi; ++c) {
                    abacus[static_cast<size_t>(a) * nj + (*c - lo)] = values[c - cols];
                    found = true;
                }
            }
            if (!found) continue;
            AbacusBlockToRsatb(orders_[layout_.atom_type[row.atom]], orders_[layout_.atom_type[j]],
                               abacus.data(), kAbacusEvPerRydberg, &block);
            sink(wanted.second, block);
        }
    }

    // --- binary: positioned reads on every rank -----------------------------

    void ReadBinaryRows(const std::vector<RowRequest>& rows, const BlockSink& sink) const {
        std::string error;
        try {
            std::ifstream in(path_, std::ios::binary);
            if (!in) throw std::runtime_error("Cannot open ABACUS Hamiltonian file " + path_ + ".");
            size_t at = 0;
            while (at < rows.size()) {
                // A run: consecutive atoms of one R, whose rows are contiguous.
                size_t end = at + 1;
                while (end < rows.size() && rows[end].shift == rows[at].shift &&
                       rows[end].atom == rows[end - 1].atom + 1) {
                    ++end;
                }
                const auto record = record_of_.find(rows[at].shift);
                if (record != record_of_.end()) {
                    ReadBinaryRun(&in, records_[record->second], rows, at, end, sink);
                }
                at = end;
            }
        } catch (const std::exception& e) {
            error = e.what();
        }
        abacus_detail::AgreeOrThrow(comm_, error);
    }

    void ReadBinaryRun(std::ifstream* in, const Record& record, const std::vector<RowRequest>& rows,
                       size_t begin, size_t end, const BlockSink& sink) const {
        const std::int64_t first_row = offsets_[rows[begin].atom];
        const std::int64_t n_rows = offsets_[rows[end - 1].atom + 1] - first_row;
        std::vector<std::int32_t> ptr(static_cast<size_t>(n_rows + 1));
        in->seekg(record.row_offsets + 4 * first_row);
        in->read(reinterpret_cast<char*>(ptr.data()), 4 * (n_rows + 1));
        const std::int64_t k0 = ptr.front();
        const std::int64_t count = ptr.back() - k0;
        if (k0 < 0 || count < 0 || k0 + count > record.nnz) {
            throw Malformed("row offsets are out of range.");
        }

        std::vector<std::int32_t> cols(static_cast<size_t>(count));
        std::vector<double> values(static_cast<size_t>(count));
        in->seekg(record.columns + 4 * k0);
        in->read(reinterpret_cast<char*>(cols.data()), 4 * count);
        in->seekg(record.values + 8 * k0);
        in->read(reinterpret_cast<char*>(values.data()), 8 * count);
        if (!*in) throw Malformed("a read ran past the end of the file.");

        std::vector<std::int64_t> row_ptr(ptr.size());
        for (size_t a = 0; a < ptr.size(); ++a) row_ptr[a] = ptr[a] - k0;
        CheckRows(row_ptr.data(), n_rows, cols.data());
        for (size_t r = begin; r < end; ++r) {
            EmitRows(rows[r], row_ptr.data() + (offsets_[rows[r].atom] - first_row), cols.data(),
                     values.data(), sink);
        }
    }

    // --- text: one streaming pass on rank 0 ---------------------------------

    template <typename T>
    T Parse(std::string_view token, const char* what) const {
        T value{};
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
        if (result.ec != std::errc() || result.ptr != token.data() + token.size()) {
            if (!token.empty() && token[0] == '(') {
                throw std::runtime_error(
                    "ABACUS Hamiltonian file " + path_ + " holds complex values, as an nspin=4 "
                    "run writes; rsatb's file-backed Hamiltonians are real-valued.");
            }
            throw Malformed(std::string("expected ") + what + ", read '" + std::string(token) + "'.");
        }
        return value;
    }

    std::string_view NextToken(abacus_detail::Tokens* tokens, const char* what) const {
        std::string_view token;
        if (!tokens->Next(&token)) throw Malformed(std::string("the file ends before ") + what + ".");
        return token;
    }

    // "[STEP: s] Matrix Dimension of H(R): n Matrix number of H(R): nR".
    void ReadTextHeader(abacus_detail::Tokens* tokens, long long* dimension, long long* n_records) const {
        std::string_view token = NextToken(tokens, "the header");
        if (token.substr(0, 4) == "STEP") {
            NextToken(tokens, "the header");  // the step number
            token = NextToken(tokens, "the header");
        }
        auto integer_after = [&](const char* word, std::string_view t) {
            bool seen = false;
            while (true) {
                if (t == word) seen = true;
                long long value = 0;
                const auto r = std::from_chars(t.data(), t.data() + t.size(), value);
                if (r.ec == std::errc() && r.ptr == t.data() + t.size()) {
                    if (!seen) throw Malformed(std::string("expected the '") + word + "' header line.");
                    return value;
                }
                t = NextToken(tokens, "the header");
            }
        };
        *dimension = integer_after("Dimension", token);
        CheckDimension(*dimension);
        *n_records = integer_after("number", NextToken(tokens, "the header"));
        if (*n_records < 0) throw Malformed("negative R count.");
    }

    void ReadTextRows(const std::vector<RowRequest>& rows, const BlockSink& sink) const {
        int n_rank = 1;
        MPI_Comm_size(comm_, &n_rank);

        // Every rank's wanted (R, atom) rows, gathered on rank 0.
        std::vector<std::int32_t> mine;
        for (const RowRequest& row : rows) {
            mine.insert(mine.end(), {row.shift[0], row.shift[1], row.shift[2], row.atom});
        }
        int count = static_cast<int>(mine.size());
        std::vector<int> counts(static_cast<size_t>(n_rank)), displs(static_cast<size_t>(n_rank) + 1, 0);
        MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);
        std::vector<std::int32_t> all;
        if (rank_ == 0) {
            for (int r = 0; r < n_rank; ++r) displs[r + 1] = displs[r] + counts[r];
            all.resize(static_cast<size_t>(displs[n_rank]));
        }
        MPI_Gatherv(mine.data(), count, MPI_INT32_T, all.data(), counts.data(), displs.data(),
                    MPI_INT32_T, 0, comm_);

        // Rank 0: R -> (rank, atom) rows wanted, by rank and then atom -- the
        // order each rank's own `rows` lists them in.
        std::map<std::array<int, 3>, std::vector<std::pair<int, int>>> wanted;
        for (int r = 0; rank_ == 0 && r < n_rank; ++r) {
            for (int k = displs[r]; k < displs[r + 1]; k += 4) {
                wanted[{{all[k], all[k + 1], all[k + 2]}}].emplace_back(r, all[k + 3]);
            }
        }
        std::vector<std::int32_t>().swap(all);
        // Every rank: R -> its own RowRequests.
        std::map<std::array<int, 3>, std::vector<const RowRequest*>> local;
        for (const RowRequest& row : rows) local[row.shift].push_back(&row);

        std::unique_ptr<abacus_detail::Tokens> tokens;
        std::string error;
        if (rank_ == 0) {
            try {
                tokens = std::make_unique<abacus_detail::Tokens>(path_);
                long long dimension = 0, n_records = 0;
                ReadTextHeader(tokens.get(), &dimension, &n_records);
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        abacus_detail::AgreeOrThrow(comm_, error);

        const std::int64_t n = dimension_;
        std::vector<double> values;
        std::vector<std::int32_t> cols, ptr;
        std::map<std::array<int, 3>, int> seen;
        for (long long record = 0; record < n_records_; ++record) {
            // Rank 0 parses and checks the record; every rank learns its R
            // and whether it holds data, or the error.
            std::array<int, 4> head{{0, 0, 0, 0}};  // Rx, Ry, Rz, has data
            if (rank_ == 0) {
                try {
                    for (int axis = 0; axis < 3; ++axis) {
                        head[axis] = Parse<int>(NextToken(tokens.get(), "an R header"), "an R component");
                    }
                    const long long nnz =
                        Parse<long long>(NextToken(tokens.get(), "an R header"), "a nonzero count");
                    if (nnz < 0) throw Malformed("negative nonzero count.");
                    if (seen[{{head[0], head[1], head[2]}}]++) {
                        throw Malformed("lattice vector (" + std::to_string(head[0]) + ", " +
                                        std::to_string(head[1]) + ", " + std::to_string(head[2]) +
                                        ") appears twice.");
                    }
                    head[3] = nnz > 0;
                    if (nnz > 0) {
                        values.resize(static_cast<size_t>(nnz));
                        cols.resize(static_cast<size_t>(nnz));
                        ptr.resize(static_cast<size_t>(n + 1));
                        for (auto& v : values) v = Parse<double>(NextToken(tokens.get(), "the values"), "a value");
                        for (auto& c : cols) c = Parse<std::int32_t>(NextToken(tokens.get(), "the columns"), "a column index");
                        for (auto& p : ptr) p = Parse<std::int32_t>(NextToken(tokens.get(), "the row offsets"), "a row offset");
                        if (ptr.front() != 0 || ptr.back() != nnz) {
                            throw Malformed("row offsets do not span the nonzeros.");
                        }
                        CheckRows(ptr.data(), n, cols.data());
                    }
                } catch (const std::exception& e) {
                    error = e.what();
                }
            }
            abacus_detail::AgreeOrThrow(comm_, error);
            MPI_Bcast(head.data(), 4, MPI_INT, 0, comm_);
            if (!head[3]) continue;
            const std::array<int, 3> shift{{head[0], head[1], head[2]}};

            const auto own = local.find(shift);
            if (rank_ == 0) {
                SendRows(shift, wanted, values, cols, ptr);
                if (own != local.end()) {
                    for (const RowRequest* row : own->second) {
                        EmitRows(*row, ptr.data() + offsets_[row->atom], cols.data(), values.data(), sink);
                    }
                }
            } else if (own != local.end()) {
                const auto payload = abacus_detail::RecvBytes(0, comm_);
                try {
                    EmitPayload(payload, own->second, sink);
                } catch (const std::exception& e) {
                    error = e.what();
                }
            }
            abacus_detail::AgreeOrThrow(comm_, error);
        }

        if (rank_ == 0) {
            std::string_view extra;
            if (tokens->Next(&extra)) {
                error = extra.substr(0, 4) == "STEP"
                    ? "ABACUS Hamiltonian file " + path_ + " holds more than one STEP (an MD run "
                      "with out_app_flag); rsatb reads single-structure files. Split off the "
                      "step that matches the structure."
                    : "Malformed ABACUS Hamiltonian file " + path_ +
                      ": unexpected content after the last R record: '" + std::string(extra) + "'.";
            }
        }
        abacus_detail::AgreeOrThrow(comm_, error);
    }

    // Rank 0: sends every other rank that wants rows of this checked record
    // exactly those rows, one message per rank.
    void SendRows(const std::array<int, 3>& shift,
                  const std::map<std::array<int, 3>, std::vector<std::pair<int, int>>>& wanted,
                  const std::vector<double>& values, const std::vector<std::int32_t>& cols,
                  const std::vector<std::int32_t>& ptr) const {
        const auto targets = wanted.find(shift);
        if (targets == wanted.end()) return;
        const auto& list = targets->second;
        std::vector<char> payload;
        std::vector<std::int64_t> rel;
        for (size_t at = 0; at < list.size();) {
            const int to = list[at].first;
            payload.clear();
            for (; at < list.size() && list[at].first == to; ++at) {
                if (to == 0) continue;
                const int atom = list[at].second;
                const std::int64_t first = offsets_[atom];
                const int na = layout_.AtomNorb(atom);
                const std::int64_t k0 = ptr[first], k1 = ptr[first + na];
                rel.resize(static_cast<size_t>(na) + 1);
                for (int a = 0; a <= na; ++a) rel[a] = ptr[first + a] - k0;
                abacus_detail::Append(&payload, rel.data(), rel.size());
                abacus_detail::Append(&payload, cols.data() + k0, static_cast<size_t>(k1 - k0));
                abacus_detail::Append(&payload, values.data() + k0, static_cast<size_t>(k1 - k0));
            }
            if (to != 0) abacus_detail::SendBytes(payload, to, comm_);
        }
    }

    // A rank's rows of one R, in the order it asked for them.
    void EmitPayload(const std::vector<char>& payload, const std::vector<const RowRequest*>& rows,
                     const BlockSink& sink) const {
        size_t cursor = 0;
        for (const RowRequest* row : rows) {
            const int na = layout_.AtomNorb(row->atom);
            const auto row_ptr = abacus_detail::Take<std::int64_t>(payload, &cursor, static_cast<size_t>(na) + 1);
            const size_t count = static_cast<size_t>(row_ptr[na]);
            const auto cols = abacus_detail::Take<std::int32_t>(payload, &cursor, count);
            const auto values = abacus_detail::Take<double>(payload, &cursor, count);
            EmitRows(*row, row_ptr.data(), cols.data(), values.data(), sink);
        }
    }

    std::string path_;
    MPI_Comm comm_;
    int rank_ = 0;
    FileBasisLayout layout_;
    std::vector<AbacusOrbitalOrder> orders_;
    std::vector<long long> offsets_;
    bool binary_ = false;
    long long dimension_ = 0;
    long long n_records_ = 0;
    std::vector<Record> records_;
    std::map<std::array<int, 3>, size_t> record_of_;
};

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_ABACUS_CSR_HPP
