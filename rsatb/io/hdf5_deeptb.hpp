#ifndef RSATB_IO_HDF5_DEEPTB_HPP
#define RSATB_IO_HDF5_DEEPTB_HPP

// HamiltonianReader for the DeePTB HDF5 layout.
//
// Layout: a single group "0" whose datasets are named "i_j_Rx_Ry_Rz", each
// shaped [m, n] for real data or [m, n, 2] for complex. Only the upper triple
// of a Hermitian pair need be stored; a missing (i, j, R) is retried as
// (j, i, -R) and transposed on read.

#include "hamiltonian_file.hpp"
#include "hdf5.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

class DeeptbHdf5Reader : public HamiltonianReader {
public:
    explicit DeeptbHdf5Reader(const std::string& path)
        : path_(path), handler_(std::make_unique<HDF5Handler>(path)) {}

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
        DenseBlock block;
        for (size_t request = 0; request < requests.size(); ++request) {
            const BlockKey& key = requests[request];
            if (ReadBlock(key.i, key.j, key.shift, &block)) {
                sink(request, block);
            }
        }
    }

private:
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
    // HDF5Handler has no default constructor and opens eagerly, so it is held
    // behind a pointer to keep this class movable.
    std::unique_ptr<HDF5Handler> handler_;
};

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_HDF5_DEEPTB_HPP
