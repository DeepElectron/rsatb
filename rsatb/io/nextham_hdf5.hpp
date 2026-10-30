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
//   atom_pair_offset [n_atom+1]  pairs of atom i are [off[i], off[i+1])
//   pair_indices     [P, 2]      (i, j)
//   pair_offset      [P+1]       R entries of pair p are [off[p], off[p+1])
//   R                [NR, 3]     sorted within each pair
//   data_offset      [P+1]       first value of pair p
//   data             [...]       complex128 (or float64); pair p's blocks
//                                follow its R list, each row-major
//
// Blocks are in ABACUS's orbital order, so the reader takes the model's
// FileBasisLayout to reorder them (abacus_convention.hpp), and cross-checks it
// against the counts and atomic numbers the file records.
//
// Only one atom's pairs are resident at a time: the graph loops ask for one
// atom's blocks together, so they are read as one contiguous slab.

#include "abacus_convention.hpp"
#include "hamiltonian_file.hpp"

#include <H5Cpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

class NextHamHdf5Reader : public HamiltonianReader {
public:
    static constexpr const char* kFormatTag = "abacus_atom_pair_operator_v1";

    /// Opens `path` against the model's basis.
    ///
    /// Args:
    ///   path: A NextHam HR_atom_pairs.h5.
    ///   layout: The model's basis over the file's atoms.
    /// Throws:
    ///   std::runtime_error if the file is not a NextHam H(R) file, is spinor
    ///   (nspin=4), or its atoms and orbital counts disagree with layout.
    NextHamHdf5Reader(const std::string& path, const FileBasisLayout& layout) : path_(path) {
        ValidateLayout(layout, "NextHam");
        RequireSpinlessLayout(layout, "NextHam");
        layout_ = layout;
        orders_ = MakeAbacusOrbitalOrders(layout_);
        try {
            H5::Exception::dontPrint();
            file_.openFile(path, H5F_ACC_RDONLY);
            ReadHeader();
            pairs_ = file_.openDataSet("pair_indices");
            pair_offset_ = file_.openDataSet("pair_offset");
            shifts_ = file_.openDataSet("R");
            data_offset_ = file_.openDataSet("data_offset");
            data_ = file_.openDataSet("data");
            complex_data_ = data_.getTypeClass() == H5T_COMPOUND;
        } catch (const H5::Exception& error) {
            throw std::runtime_error("Cannot read NextHam Hamiltonian file " + path + ": " +
                                     error.getDetailMsg());
        }
    }

    HamiltonianFormat format() const override { return HamiltonianFormat::kNextHamHdf5; }

    FileMetadata metadata() const override {
        FileMetadata metadata;
        metadata.n_orbital_total = basis_num_;
        metadata.n_spin_component = 1;
        return metadata;
    }

    bool ReadBlock(int i, int j, const CellShift& shift, DenseBlock* out) const override {
        if (out == nullptr) {
            throw std::runtime_error("NextHamHdf5Reader::ReadBlock requires a non-null output block.");
        }
        if (i < 0 || i >= layout_.n_atom() || j < 0 || j >= layout_.n_atom()) {
            throw std::runtime_error("NextHam block (" + std::to_string(i) + ", " +
                                     std::to_string(j) + ") names an atom outside the file.");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        LoadRow(i);

        const auto pair = std::lower_bound(row_.columns.begin(), row_.columns.end(), j);
        if (pair == row_.columns.end() || *pair != j) {
            return false;
        }
        const size_t p = static_cast<size_t>(pair - row_.columns.begin());
        const std::array<std::int64_t, 3> wanted{{shift[0], shift[1], shift[2]}};
        for (std::int64_t q = row_.pair_offset[p]; q < row_.pair_offset[p + 1]; ++q) {
            const size_t r = static_cast<size_t>(q - row_.pair_offset.front());
            if (row_.shifts[r] != wanted) continue;

            const int ni = layout_.AtomNorb(i);
            const int nj = layout_.AtomNorb(j);
            const size_t start = static_cast<size_t>(row_.data_offset[p] - row_.data_offset.front()) +
                                 static_cast<size_t>(q - row_.pair_offset[p]) * ni * nj;
            AbacusBlockToRsatb(orders_[layout_.atom_type[i]], orders_[layout_.atom_type[j]],
                               row_.values.data() + start, energy_factor_, out);
            return true;
        }
        return false;
    }

private:
    // One atom's pairs: their columns, R lists and values.
    struct Row {
        int atom = -1;
        std::vector<int> columns;
        std::vector<std::int64_t> pair_offset;
        std::vector<std::int64_t> data_offset;
        std::vector<std::array<std::int64_t, 3>> shifts;
        std::vector<double> values;
    };

    std::runtime_error Mismatch(const std::string& what) const {
        return std::runtime_error("NextHam Hamiltonian file " + path_ + " " + what);
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

    std::vector<std::int64_t> ReadWhole(const std::string& name) const {
        H5::DataSet dataset = file_.openDataSet(name);
        std::vector<std::int64_t> values(
            static_cast<size_t>(dataset.getSpace().getSimpleExtentNpoints()));
        dataset.read(values.data(), H5::PredType::NATIVE_INT64);
        return values;
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
        if (IntegerAttribute("nspin") != 1) {
            throw Mismatch("is an nspin=" + std::to_string(IntegerAttribute("nspin")) +
                           " file. Spinor H(R) is complex, and rsatb's file-backed Hamiltonians "
                           "are real-valued; only nspin=1 is supported.");
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

        const auto counts = ReadWhole("atom_basis_count");
        if (static_cast<int>(counts.size()) != layout_.n_atom()) {
            throw Mismatch("has " + std::to_string(counts.size()) + " atoms, but the structure "
                           "has " + std::to_string(layout_.n_atom()) + ".");
        }
        for (int atom = 0; atom < layout_.n_atom(); ++atom) {
            if (counts[atom] != layout_.AtomNorb(atom)) {
                throw Mismatch("gives atom " + std::to_string(atom) + " " +
                               std::to_string(counts[atom]) + " orbitals, but the model's basis "
                               "gives it " + std::to_string(layout_.AtomNorb(atom)) +
                               ". Check the atom order and that the orbital files are the ones "
                               "NextHam's ABACUS data used.");
            }
        }
        if (file_.nameExists("atomic_numbers") && !layout_.type_atomic_number.empty()) {
            const auto numbers = ReadWhole("atomic_numbers");
            for (int atom = 0; atom < layout_.n_atom(); ++atom) {
                const int expected = layout_.type_atomic_number[layout_.atom_type[atom]];
                if (numbers[atom] != expected) {
                    throw Mismatch("has Z=" + std::to_string(numbers[atom]) + " at atom " +
                                   std::to_string(atom) + ", but the structure has Z=" +
                                   std::to_string(expected) + " there; the atom orders differ.");
                }
            }
        }
        basis_num_ = static_cast<int>(IntegerAttribute("basis_num"));

        atom_pair_offset_ = ReadWhole("atom_pair_offset");
        if (static_cast<int>(atom_pair_offset_.size()) != layout_.n_atom() + 1) {
            throw Mismatch("has an atom_pair_offset of the wrong length.");
        }
    }

    // Makes atom i's pairs resident. Called with mutex_ held.
    void LoadRow(int i) const {
        if (row_.atom == i) return;
        Row row;
        row.atom = i;
        try {
            const hsize_t begin = static_cast<hsize_t>(atom_pair_offset_[i]);
            const hsize_t count = static_cast<hsize_t>(atom_pair_offset_[i + 1]) - begin;
            const auto pairs = ReadSlab<std::int64_t>(pairs_, H5::PredType::NATIVE_INT64, begin, count, 2);
            row.columns.resize(count);
            for (hsize_t p = 0; p < count; ++p) {
                if (pairs[2 * p] != i) throw Mismatch("lists a pair under the wrong atom row.");
                row.columns[p] = static_cast<int>(pairs[2 * p + 1]);
            }
            row.pair_offset = ReadSlab<std::int64_t>(pair_offset_, H5::PredType::NATIVE_INT64, begin, count + 1, 1);
            row.data_offset = ReadSlab<std::int64_t>(data_offset_, H5::PredType::NATIVE_INT64, begin, count + 1, 1);

            const hsize_t r_begin = static_cast<hsize_t>(row.pair_offset.front());
            const hsize_t r_count = static_cast<hsize_t>(row.pair_offset.back()) - r_begin;
            const auto shifts = ReadSlab<std::int64_t>(shifts_, H5::PredType::NATIVE_INT64, r_begin, r_count, 3);
            row.shifts.resize(r_count);
            for (hsize_t r = 0; r < r_count; ++r) {
                row.shifts[r] = {{shifts[3 * r], shifts[3 * r + 1], shifts[3 * r + 2]}};
            }

            const hsize_t d_begin = static_cast<hsize_t>(row.data_offset.front());
            const hsize_t d_count = static_cast<hsize_t>(row.data_offset.back()) - d_begin;
            if (complex_data_) {
                H5::CompType complex_type(2 * sizeof(double));
                complex_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
                complex_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
                const auto raw = ReadSlab<double>(data_, complex_type, d_begin, d_count, 1, 2);
                row.values.resize(d_count);
                for (hsize_t v = 0; v < d_count; ++v) {
                    if (std::abs(raw[2 * v + 1]) > 1e-12) {
                        throw Mismatch("holds a complex H(R); rsatb's file-backed "
                                       "Hamiltonians are real-valued.");
                    }
                    row.values[v] = raw[2 * v];
                }
            } else {
                row.values = ReadSlab<double>(data_, H5::PredType::NATIVE_DOUBLE, d_begin, d_count, 1);
            }
        } catch (const H5::Exception& error) {
            throw std::runtime_error("Cannot read NextHam Hamiltonian file " + path_ + ": " +
                                     error.getDetailMsg());
        }
        row_ = std::move(row);
    }

    std::string path_;
    FileBasisLayout layout_;
    std::vector<AbacusOrbitalOrder> orders_;
    H5::H5File file_;
    H5::DataSet pairs_, pair_offset_, shifts_, data_offset_, data_;
    bool complex_data_ = false;
    double energy_factor_ = 1.0;
    int basis_num_ = 0;
    std::vector<std::int64_t> atom_pair_offset_;

    mutable std::mutex mutex_;
    mutable Row row_;
};

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_NEXTHAM_HDF5_HPP
