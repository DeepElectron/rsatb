#ifndef RSATB_IO_HDF5_DEEPTB_HPP
#define RSATB_IO_HDF5_DEEPTB_HPP

// HamiltonianReader for the DeePTB HDF5 layout.
//
// Layout: a single group "0" whose datasets are named "i_j_Rx_Ry_Rz", each
// shaped [m, n] for real data, or for complex data [m, n] of h5py's compound
// {r, i} (what DeePTB writes) or [m, n, 2]. Only the upper triple of a Hermitian
// pair need be stored; a missing (i, j, R) is retried as (j, i, -R) and conjugate
// transposed on read.
//
// A spinor H(R), from a spin-orbit calculation, is complex with blocks of twice
// the orbitals, spin-blocked per atom: [[uu, ud], [du, dd]], so index s * n + orb
// for an atom of n spatial orbitals (dptb/data/interfaces/abacus.py). rsatb's
// spinor basis interleaves the spins, 2 * orb + s, as ABACUS, NextHam and op2c do,
// so this reader reorders the blocks. The orbitals within a spin are already in
// rsatb's order: dftio and DeePTB convert ABACUS's the same way this code base does.
//
// The overlap DeePTB writes next to such an H(R) stays real over spatial orbitals,
// so it is read with spin_components = 1 and expanded by the model.

#include "hamiltonian_file.hpp"
#include "hdf5.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

class DeeptbHdf5Reader : public HamiltonianReader {
public:
    /// Args:
    ///   path: The HDF5 file.
    ///   spin_components: 2 to read a spinor H(R), complex and spin-blocked.
    explicit DeeptbHdf5Reader(const std::string& path, int spin_components = 1)
        : path_(path), spin_components_(spin_components),
          handler_(std::make_unique<HDF5Handler>(path)) {}

    HamiltonianFormat format() const override {
        return HamiltonianFormat::kDeeptbHdf5;
    }

    /// Returns an empty record.
    ///
    /// The DeePTB layout stores no header: orbital counts and spin treatment
    /// are implied by the block shapes alone. A model that wants to validate
    /// its basis against this file should read an onsite block and compare
    /// dimensions.
    FileMetadata metadata() const override {
        return FileMetadata{};
    }

    /// One dataset per block, so the requests are simply read in turn; there
    /// is no layout to coalesce and nothing collective.
    void ReadBlocks(const std::vector<BlockKey>& requests, const BlockSink& sink) const override {
        if (complex_blocks()) {
            throw std::runtime_error("The spinor H(R) in " + path_ + " has complex blocks.");
        }
        DenseBlock block;
        for (size_t request = 0; request < requests.size(); ++request) {
            const BlockKey& key = requests[request];
            if (ReadBlock(key.i, key.j, key.shift, &block)) {
                sink(request, block);
            }
        }
    }

    bool complex_blocks() const override { return spin_components_ == 2; }

    void ReadComplexBlocks(const std::vector<BlockKey>& requests,
                           const ComplexBlockSink& sink) const override {
        if (!complex_blocks()) {
            throw std::runtime_error("The spin-degenerate H(R) in " + path_ + " has real blocks.");
        }
        ComplexDenseBlock block;
        for (size_t request = 0; request < requests.size(); ++request) {
            const BlockKey& key = requests[request];
            if (ReadInterleavedBlock(key.i, key.j, key.shift, &block)) {
                sink(request, block);
            }
        }
    }

private:
    /// One spinor block, from the file's spin-blocked order to rsatb's interleaved.
    bool ReadInterleavedBlock(int i, int j, const CellShift& shift, ComplexDenseBlock* out) const {
        int rows = 0;
        int cols = 0;
        std::complex<double>* values = nullptr;
        bool found = false;
        const int status = handler_->read_from_ijR_dense_complex(
            i, j, shift[0], shift[1], shift[2], &rows, &cols, &values, &found);
        std::unique_ptr<std::complex<double>[]> owned(values);
        if (status != 0) {
            throw std::runtime_error(
                "Failed to read Hamiltonian block (" + std::to_string(i) + ", " +
                std::to_string(j) + ") from " + path_);
        }
        if (!found) {
            return false;
        }
        if (rows % 2 != 0 || cols % 2 != 0) {
            throw std::runtime_error(
                "Hamiltonian block (" + std::to_string(i) + ", " + std::to_string(j) + ") in " +
                path_ + " is " + std::to_string(rows) + "x" + std::to_string(cols) +
                ", which a spinor basis cannot hold: was the file written without spin?");
        }

        out->Resize(rows, cols);
        const int n_i = rows / 2;
        const int n_j = cols / 2;
        for (int si = 0; si < 2; ++si) {
            for (int oi = 0; oi < n_i; ++oi) {
                const size_t from_row = static_cast<size_t>(si * n_i + oi) * cols;
                const size_t to_row = static_cast<size_t>(2 * oi + si) * cols;
                for (int sj = 0; sj < 2; ++sj) {
                    for (int oj = 0; oj < n_j; ++oj) {
                        out->values[to_row + 2 * oj + sj] = values[from_row + sj * n_j + oj];
                    }
                }
            }
        }
        return true;
    }

    bool ReadBlock(int i, int j, const CellShift& shift, DenseBlock* out) const {
        int rows = 0;
        int cols = 0;
        double* values = nullptr;
        bool found = false;

        const int status = handler_->read_from_ijR_dense(
            i, j, shift[0], shift[1], shift[2], &rows, &cols, &values, &found);
        if (status != 0) {
            delete[] values;
            throw std::runtime_error(
                "Failed to read Hamiltonian block (" + std::to_string(i) + ", " +
                std::to_string(j) + ") from " + path_);
        }
        if (!found) {
            return false;
        }

        out->Resize(rows, cols);
        std::copy(values, values + out->values.size(), out->values.begin());
        delete[] values;
        return true;
    }

    std::string path_;
    int spin_components_ = 1;
    // HDF5Handler has no default constructor and opens eagerly, so it is held
    // behind a pointer to keep this class movable.
    std::unique_ptr<HDF5Handler> handler_;
};

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_HDF5_DEEPTB_HPP
