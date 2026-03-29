#ifndef hdf5_io
#define hdf5_io
/* learning materials: https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html */
/* Everything become so hard in c++. */

#include <H5Cpp.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rsatb::io {

class HDF5Handler {
private:
    H5::H5File file;
    std::string FILE_NAME;
    H5::Group group;

    bool open_dataset_with_fallback(
        int i, int j, int Rx, int Ry, int Rz,
        H5::DataSet& dataset,
        bool& transpose
    ) {
        transpose = false;

        std::string result = std::to_string(i) + "_" +
                             std::to_string(j) + "_" +
                             std::to_string(Rx) + "_" +
                             std::to_string(Ry) + "_" +
                             std::to_string(Rz);

        if (group.exists(result)) {
            dataset = group.openDataSet(result);
            return true;
        }

        result = std::to_string(j) + "_" +
                 std::to_string(i) + "_" +
                 std::to_string(-Rx) + "_" +
                 std::to_string(-Ry) + "_" +
                 std::to_string(-Rz);

        if (!group.exists(result)) {
            return false;
        }

        dataset = group.openDataSet(result);
        transpose = true;
        return true;
    }

    template <typename ScalarT>
    int read_from_ijR_dense_impl(
        int i, int j, int Rx, int Ry, int Rz,
        int* m,
        int* n,
        ScalarT** v,
        bool* found
    ) {
        *found = true;
        bool transpose = false;
        H5::DataSet dataset;

        try {
            if (!open_dataset_with_fallback(i, j, Rx, Ry, Rz, dataset, transpose)) {
                *found = false;
                return 0;
            }

            H5::DataSpace dataspace = dataset.getSpace();
            const int rank = dataspace.getSimpleExtentNdims();
            if (rank != 2 && rank != 3) {
                throw std::runtime_error("Unsupported ijR dataset rank");
            }

            hsize_t dims[3] = {0, 0, 0};
            dataspace.getSimpleExtentDims(dims, nullptr);

            const bool is_complex_dataset = (rank == 3);
            if (is_complex_dataset && dims[2] != 2) {
                throw std::runtime_error("Complex ijR datasets must use shape [m, n, 2]");
            }

            *m = transpose ? static_cast<int>(dims[1]) : static_cast<int>(dims[0]);
            *n = transpose ? static_cast<int>(dims[0]) : static_cast<int>(dims[1]);
            *v = new ScalarT[static_cast<size_t>(*m) * static_cast<size_t>(*n)];

            if (!is_complex_dataset) {
                std::vector<double> raw(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]));
                dataset.read(raw.data(), H5::PredType::NATIVE_DOUBLE);

                for (int row = 0; row < *m; ++row) {
                    for (int col = 0; col < *n; ++col) {
                        const int src_row = transpose ? col : row;
                        const int src_col = transpose ? row : col;
                        const double value = raw[static_cast<size_t>(src_row) * static_cast<size_t>(dims[1]) +
                                                 static_cast<size_t>(src_col)];

                        if constexpr (std::is_same_v<ScalarT, std::complex<double>>) {
                            (*v)[static_cast<size_t>(row) * static_cast<size_t>(*n) + static_cast<size_t>(col)] =
                                std::complex<double>(value, 0.0);
                        } else {
                            (*v)[static_cast<size_t>(row) * static_cast<size_t>(*n) + static_cast<size_t>(col)] = value;
                        }
                    }
                }
                return 0;
            }

            std::vector<double> raw(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * 2);
            dataset.read(raw.data(), H5::PredType::NATIVE_DOUBLE);

            for (int row = 0; row < *m; ++row) {
                for (int col = 0; col < *n; ++col) {
                    const int src_row = transpose ? col : row;
                    const int src_col = transpose ? row : col;
                    const size_t idx = (static_cast<size_t>(src_row) * static_cast<size_t>(dims[1]) +
                                        static_cast<size_t>(src_col)) * 2;
                    const std::complex<double> value(raw[idx], raw[idx + 1]);

                    if constexpr (std::is_same_v<ScalarT, std::complex<double>>) {
                        (*v)[static_cast<size_t>(row) * static_cast<size_t>(*n) + static_cast<size_t>(col)] = value;
                    } else {
                        if (std::abs(value.imag()) > 1e-12) {
                            throw std::runtime_error("Complex ijR dataset requested through the real-valued reader");
                        }
                        (*v)[static_cast<size_t>(row) * static_cast<size_t>(*n) + static_cast<size_t>(col)] = value.real();
                    }
                }
            }
        }
        catch (H5::Exception& error) {
            error.printErrorStack();
            return -1;
        }
        catch (std::exception& error) {
            std::cerr << "HDF5Handler error: " << error.what() << std::endl;
            return -1;
        }

        return 0;
    }

public:
    explicit HDF5Handler(const std::string& FILE_NAME)
        : FILE_NAME(FILE_NAME), file(FILE_NAME, H5F_ACC_RDONLY), group(file.openGroup("0")) {}

    int read_from_ijR_dense(
        int i, int j, int Rx, int Ry, int Rz,
        int* m,
        int* n,
        double** v,
        bool* found
    ) {
        return read_from_ijR_dense_impl<double>(i, j, Rx, Ry, Rz, m, n, v, found);
    }

    int read_from_ijR_dense_complex(
        int i, int j, int Rx, int Ry, int Rz,
        int* m,
        int* n,
        std::complex<double>** v,
        bool* found
    ) {
        return read_from_ijR_dense_impl<std::complex<double>>(i, j, Rx, Ry, Rz, m, n, v, found);
    }

    // void read_from_R_csr(
    //     int Rx, int Ry, int Rz,
    //     // output
    //     int *n, // nxn mat size
    //     int *indices,
    //     int *intptr,
    //     double *data
    //     );
};

} // namespace rsatb::io

#endif
