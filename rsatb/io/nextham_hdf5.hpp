#ifndef RSATB_IO_NEXTHAM_HDF5_HPP
#define RSATB_IO_NEXTHAM_HDF5_HPP

// HamiltonianReader for NextHam's atom-pair HDF5 (`HR_atom_pairs.h5`).
//
// NextHam's post-processing writes H(R) and S(R) as two files of the format
// "abacus_atom_pair_operator_v1". Root attributes: format, operator ("HR" or
// "SR"), nspin, basis_num, atom_count, and the energy unit as HR_unit (or
// unit). Datasets, with P atom pairs sorted by (i, j) and NR (pair, R) entries:
//
//   atom_basis_count [n_atom]    orbitals per atom
//   atomic_numbers   [n_atom]    optional
//   atom_pair_offset [n_atom+1]  pairs of atom i are [off[i], off[i+1])
//   pair_indices     [P, 2]      (i, j)
//   pair_offset      [P+1]       R entries of pair p are [off[p], off[p+1])
//   R                [NR, 3]     sorted within each pair
//   data_offset      [P+1]       first value of pair p
//   data             [...]       complex128 (or float64); pair p's blocks
//                                follow its R list, each row-major
//
// Blocks are in ABACUS's orbital order, so the reader takes the model's
// FileBasisLayout to reorder them (abacus_convention.hpp).
//
// The layout is indexed by atom row, which is what makes it scale: a rank
// reads only the rows of the atoms it owns. Its rows are merged into runs of
// consecutive file atoms, each run's pair metadata is read in one slab, and
// its R lists and values in slabs of at most kSlabBytes -- every row once,
// nothing about other ranks' atoms, and no array over the whole system.

#include "abacus_convention.hpp"
#include "hamiltonian_file.hpp"
#include "hdf5.hpp"

#include <H5Cpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rsatb {
namespace io {

class NextHamHdf5Reader : public HamiltonianReader {
public:
    static constexpr const char* kFormatTag = "abacus_atom_pair_operator_v1";
    /// Upper bound on the values one slab read holds.
    static constexpr std::int64_t kSlabBytes = 64LL << 20;

    /// Opens `path` against the model's basis. Not collective: every rank
    /// opens the file independently and reads only what it asks for.
    ///
    /// Args:
    ///   path: A NextHam HR_atom_pairs.h5.
    ///   layout: The model's basis over the file's atoms.
    /// Throws:
    ///   std::runtime_error if the file is not a NextHam H(R) file, its nspin
    ///   disagrees with the layout's spin (nspin=4 is a spinor basis), or its
    ///   atom count or basis size disagrees with layout.
    NextHamHdf5Reader(const std::string& path, const FileBasisLayout& layout) : path_(path) {
        ValidateLayout(layout, "NextHam");
        layout_ = layout;
        spinor_ = IsSpinor(layout_);
        orders_ = MakeAbacusOrbitalOrders(layout_);
        try {
            H5::Exception::dontPrint();
            file_.openFile(path, H5F_ACC_RDONLY, ReadOnlyAccess());
            ReadHeader();
            atom_pair_offset_ = file_.openDataSet("atom_pair_offset");
            pairs_ = file_.openDataSet("pair_indices");
            pair_offset_ = file_.openDataSet("pair_offset");
            shifts_ = file_.openDataSet("R");
            data_offset_ = file_.openDataSet("data_offset");
            data_ = file_.openDataSet("data");
            complex_data_ = data_.getTypeClass() == H5T_COMPOUND;
            if (complex_data_) complex_type_ = ComplexMemoryType();
            if (file_.nameExists("atomic_numbers")) {
                atomic_numbers_ = file_.openDataSet("atomic_numbers");
                has_atomic_numbers_ = true;
            }
            if (Extent(file_.openDataSet("atom_basis_count")) != layout_.n_atom() ||
                Extent(atom_pair_offset_) != layout_.n_atom() + 1) {
                throw Mismatch("has " + std::to_string(Extent(file_.openDataSet("atom_basis_count"))) +
                               " atoms, but the structure has " +
                               std::to_string(layout_.n_atom()) + ".");
            }
        } catch (const H5::Exception& error) {
            throw std::runtime_error("Cannot read NextHam Hamiltonian file " + path + ": " +
                                     error.getDetailMsg());
        }
    }

    HamiltonianFormat format() const override { return HamiltonianFormat::kNextHamHdf5; }

    FileMetadata metadata() const override {
        FileMetadata metadata;
        metadata.n_orbital_total = basis_num_;
        metadata.n_spin_component = layout_.spin_components;
        return metadata;
    }

    bool complex_blocks() const override { return spinor_; }

    void ReadBlocks(const std::vector<BlockKey>& requests, const BlockSink& sink) const override {
        if (spinor_) throw Mismatch("is read as spinor (nspin=4): its blocks are complex.");
        Read<double>(requests, sink);
    }

    void ReadComplexBlocks(const std::vector<BlockKey>& requests,
                           const ComplexBlockSink& sink) const override {
        if (!spinor_) throw Mismatch("is read as spin-degenerate (nspin=1): its blocks are real.");
        Read<std::complex<double>>(requests, sink);
    }

private:
    template <typename V>
    void Read(const std::vector<BlockKey>& requests, const BlockSinkOf<V>& sink) const {
        // This rank's requests in (i, j, R) order, so a row's are adjacent.
        std::vector<size_t> order(requests.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return requests[a] < requests[b]; });
        for (size_t k : order) {
            const BlockKey& key = requests[k];
            if (key.i < 0 || key.i >= layout_.n_atom() || key.j < 0 || key.j >= layout_.n_atom()) {
                throw Mismatch("was asked for block (" + std::to_string(key.i) + ", " +
                               std::to_string(key.j) + "), outside its atoms.");
            }
        }

        try {
            size_t at = 0;
            while (at < order.size()) {
                // A run: consecutive file atoms that all have requests.
                size_t end = at;
                int last_atom = requests[order[at]].i;
                while (end < order.size() && requests[order[end]].i <= last_atom + 1) {
                    last_atom = requests[order[end]].i;
                    ++end;
                }
                ReadRun<V>(requests, order, at, end, sink);
                at = end;
            }
        } catch (const H5::Exception& error) {
            throw std::runtime_error("Cannot read NextHam Hamiltonian file " + path_ + ": " +
                                     error.getDetailMsg());
        }
    }

    std::runtime_error Mismatch(const std::string& what) const {
        return std::runtime_error("NextHam Hamiltonian file " + path_ + " " + what);
    }

    static long long Extent(const H5::DataSet& dataset) {
        hsize_t dims[2] = {0, 0};
        dataset.getSpace().getSimpleExtentDims(dims);
        return static_cast<long long>(dims[0]);
    }

    std::string StringAttribute(const std::string& name, const std::string& fallback) const {
        if (!file_.attrExists(name)) return fallback;
        H5::Attribute attribute = file_.openAttribute(name);
        std::string value;
        attribute.read(attribute.getStrType(), value);
        return value;
    }

    long long IntegerAttribute(const std::string& name) const {
        if (!file_.attrExists(name)) {
            throw Mismatch("has no '" + name + "' attribute.");
        }
        long long value = 0;
        file_.openAttribute(name).read(H5::PredType::NATIVE_LLONG, &value);
        return value;
    }

    // Reads rows [begin, begin + count) of a 1D or 2D dataset; each element of
    // `type` fills `per_element` entries of the result.
    template <typename T>
    std::vector<T> ReadSlab(const H5::DataSet& dataset, const H5::DataType& type,
                            hsize_t begin, hsize_t count, hsize_t width,
                            hsize_t per_element = 1) const {
        std::vector<T> values(static_cast<size_t>(count * width * per_element));
        if (count == 0) return values;
        H5::DataSpace space = dataset.getSpace();
        const int rank = space.getSimpleExtentNdims();
        hsize_t offset[2] = {begin, 0};
        hsize_t extent[2] = {count, width};
        space.selectHyperslab(H5S_SELECT_SET, extent, offset);
        H5::DataSpace memory(rank, extent);
        dataset.read(values.data(), type, memory, space);
        return values;
    }

    std::vector<std::int64_t> ReadInts(const H5::DataSet& dataset, std::int64_t begin,
                                       std::int64_t count, hsize_t width = 1) const {
        return ReadSlab<std::int64_t>(dataset, H5::PredType::NATIVE_INT64,
                                      static_cast<hsize_t>(begin), static_cast<hsize_t>(count), width);
    }

    void ReadHeader() {
        const std::string tag = StringAttribute("format", "");
        if (tag == "pair_predictions_v1") {
            throw Mismatch("is NextHam's raw prediction file (pair_predictions_v1), not its "
                           "post-processed H(R); point the path at HR_atom_pairs.h5.");
        }
        if (tag != kFormatTag) {
            throw Mismatch("has format '" + tag + "', expected '" + kFormatTag + "'.");
        }
        const std::string op = StringAttribute("operator", "");
        if (op == "SR") {
            throw Mismatch("holds the overlap S(R), not H(R); point the path at "
                           "HR_atom_pairs.h5. The LCAO model computes S from the orbital files.");
        }
        if (op != "HR") {
            throw Mismatch("has operator '" + op + "', expected 'HR'.");
        }
        const long long nspin = IntegerAttribute("nspin");
        if (nspin != (spinor_ ? 4 : 1)) {
            throw Mismatch("is an nspin=" + std::to_string(nspin) + " file, but the model's spin "
                           "setting makes it read as nspin=" + (spinor_ ? "4" : "1") + ". Use spin "
                           "'spinorbit' for an nspin=4 (spin-orbit) file and 'none' for nspin=1.");
        }
        if (spinor_) {
            const std::string order = StringAttribute("spin_order", "orbital_interleaved");
            if (order != "orbital_interleaved") {
                throw Mismatch("has spin_order '" + order + "'; rsatb's spinor basis is "
                               "orbital_interleaved.");
            }
        }

        const std::string unit = StringAttribute("HR_unit", StringAttribute("unit", "eV"));
        if (unit == "eV") {
            energy_factor_ = 1.0;
        } else if (unit == "Ry") {
            energy_factor_ = kAbacusEvPerRydberg;
        } else if (unit == "Ha") {
            energy_factor_ = 2.0 * kAbacusEvPerRydberg;
        } else {
            throw Mismatch("declares the energy unit '" + unit + "'; expected eV, Ry or Ha.");
        }

        basis_num_ = static_cast<int>(IntegerAttribute("basis_num"));
        const long long expected = layout_.AtomOffsets().back();
        if (basis_num_ != expected) {
            throw Mismatch("has " + std::to_string(basis_num_) + " orbitals, but the model's "
                           "basis over this structure has " + std::to_string(expected) +
                           ". Check the atom order and that the orbital files are the ones "
                           "NextHam's ABACUS data used.");
        }
    }

    // Reads the rows of requests order[begin, end), whose atoms are consecutive.
    template <typename V>
    void ReadRun(const std::vector<BlockKey>& requests, const std::vector<size_t>& order,
                 size_t begin, size_t end, const BlockSinkOf<V>& sink) const {
        const int first_atom = requests[order[begin]].i;
        const int n_atom = requests[order[end - 1]].i - first_atom + 1;

        // Pair metadata of the whole run: 40 bytes per pair.
        const auto apo = ReadInts(atom_pair_offset_, first_atom, n_atom + 1);
        const std::int64_t p0 = apo.front();
        const std::int64_t n_pair = apo.back() - p0;
        const auto pairs = ReadInts(pairs_, p0, n_pair, 2);
        const auto pair_offset = ReadInts(pair_offset_, p0, n_pair + 1);
        const auto data_offset = ReadInts(data_offset_, p0, n_pair + 1);
        if (has_atomic_numbers_) {
            const auto numbers = ReadInts(atomic_numbers_, first_atom, n_atom);
            for (int a = 0; a < n_atom; ++a) {
                const int atom = first_atom + a;
                const int expected = layout_.type_atomic_number.empty()
                    ? static_cast<int>(numbers[a])
                    : layout_.type_atomic_number[layout_.atom_type[atom]];
                if (numbers[a] != expected) {
                    throw Mismatch("has Z=" + std::to_string(numbers[a]) + " at atom " +
                                   std::to_string(atom) + ", but the structure has Z=" +
                                   std::to_string(expected) + " there; the atom orders differ.");
                }
            }
        }

        // Split the run's atoms into slabs of bounded value count.
        const std::int64_t budget = kSlabBytes / static_cast<std::int64_t>(complex_data_ ? 16 : 8);
        size_t request = begin;
        int slab_first = 0;
        while (slab_first < n_atom) {
            int slab_end = slab_first + 1;
            while (slab_end < n_atom &&
                   data_offset[apo[slab_end + 1] - p0] - data_offset[apo[slab_first] - p0] <= budget) {
                ++slab_end;
            }
            const std::int64_t q0 = apo[slab_first] - p0;   // first pair, run-relative
            const std::int64_t q1 = apo[slab_end] - p0;
            const std::int64_t r0 = pair_offset[q0];
            const auto shifts = ReadInts(shifts_, r0, pair_offset[q1] - r0, 3);
            const std::int64_t d0 = data_offset[q0];
            const auto values = ReadValues<V>(d0, data_offset[q1] - d0);

            for (; request < end && requests[order[request]].i < first_atom + slab_end; ++request) {
                const size_t k = order[request];
                const BlockKey& key = requests[k];
                const int a = key.i - first_atom;
                // The atom's pairs, sorted by j.
                std::int64_t lo = apo[a] - p0, hi = apo[a + 1] - p0;
                while (lo < hi) {
                    const std::int64_t mid = (lo + hi) / 2;
                    if (pairs[2 * mid + 1] < key.j) lo = mid + 1; else hi = mid;
                }
                if (lo == apo[a + 1] - p0 || pairs[2 * lo + 1] != key.j) continue;
                if (pairs[2 * lo] != key.i) throw Mismatch("lists a pair under the wrong atom row.");

                const int ni = layout_.AtomNorb(key.i);
                const int nj = layout_.AtomNorb(key.j);
                const std::int64_t n_shift = pair_offset[lo + 1] - pair_offset[lo];
                if (data_offset[lo + 1] - data_offset[lo] != n_shift * ni * nj) {
                    throw Mismatch("stores blocks of pair (" + std::to_string(key.i) + ", " +
                                   std::to_string(key.j) + ") that are not " + std::to_string(ni) +
                                   "x" + std::to_string(nj) + ", the model's basis for it. Check "
                                   "the atom order and the orbital files.");
                }
                for (std::int64_t q = 0; q < n_shift; ++q) {
                    const std::int64_t r = pair_offset[lo] + q - r0;
                    if (shifts[3 * r] != key.shift[0] || shifts[3 * r + 1] != key.shift[1] ||
                        shifts[3 * r + 2] != key.shift[2]) {
                        continue;
                    }
                    DenseBlockOf<V> block;
                    AbacusBlockToRsatb(orders_[layout_.atom_type[key.i]],
                                       orders_[layout_.atom_type[key.j]],
                                       values.data() + (data_offset[lo] - d0) + q * ni * nj,
                                       energy_factor_, &block);
                    sink(k, block);
                    break;
                }
            }
            slab_first = slab_end;
        }
    }

    // The memory type complex values are read into: {r, i} doubles at 0, 8,
    // which is how h5py and NextHam's writer store complex128.
    static H5::CompType ComplexMemoryType() {
        H5::CompType memory(2 * sizeof(double));
        memory.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
        memory.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
        return memory;
    }

    // Values [begin, begin + count) of `data`, as V.
    //
    // A spin-degenerate H(R) is real, and NextHam stores it complex with zero
    // imaginary parts; a nonzero one means the file is not what its nspin
    // says. A spinor H(R) is read as stored.
    template <typename V>
    std::vector<V> ReadValues(std::int64_t begin, std::int64_t count) const {
        const hsize_t first = static_cast<hsize_t>(begin), n = static_cast<hsize_t>(count);
        if constexpr (std::is_same_v<V, double>) {
            if (!complex_data_) return ReadSlab<double>(data_, H5::PredType::NATIVE_DOUBLE, first, n, 1);
            const auto raw = ReadSlab<double>(data_, complex_type_, first, n, 1, 2);
            std::vector<double> values(static_cast<size_t>(count));
            for (std::int64_t v = 0; v < count; ++v) {
                if (std::abs(raw[2 * v + 1]) > 1e-12) {
                    throw Mismatch("is nspin=1 but holds a complex H(R).");
                }
                values[v] = raw[2 * v];
            }
            return values;
        } else {
            if (!complex_data_) {
                throw Mismatch("is nspin=4 but stores real values; a spinor H(R) is complex.");
            }
            return ReadSlab<V>(data_, complex_type_, first, n, 1);
        }
    }

    std::string path_;
    FileBasisLayout layout_;
    bool spinor_ = false;
    std::vector<AbacusOrbitalOrder> orders_;
    H5::H5File file_;
    H5::DataSet atom_pair_offset_, pairs_, pair_offset_, shifts_, data_offset_, data_;
    H5::DataSet atomic_numbers_;
    bool has_atomic_numbers_ = false;
    bool complex_data_ = false;
    H5::CompType complex_type_;
    double energy_factor_ = 1.0;
    int basis_num_ = 0;
};

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_NEXTHAM_HDF5_HPP
