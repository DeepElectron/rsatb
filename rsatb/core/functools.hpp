#ifndef functools
#define functools

#include <mkl_spblas.h>
#include <mkl_cblas.h>
#include <optional>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cmath>

inline void mkl_check(sparse_status_t status, const char* where)
{
    if(status != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("[MKL] ") + where + " failed (code " + std::to_string(status) + ")");
}

inline void export_mkl_csr(sparse_matrix_t                 A,
                    int&                            n,
                    int&                            m,
                    int&                            nnz,
                    std::complex<double>*&          val,
                    int*&                           ia,
                    int*&                           ja,
                    int&                            indexBase)
{
    sparse_index_base_t indexing;      // 0- or 1-based?
    MKL_INT rows, cols;                // dimensions
    MKL_INT *row_start, *row_end;      // CSR row-pointer slices
    MKL_INT *columns;                  // JA
    MKL_Complex16  *values;                   // VAL

    sparse_status_t st =
        mkl_sparse_z_export_csr(A,
                                &indexing,
                                &rows, &cols,
                                &row_start, &row_end,
                                &columns,
                                reinterpret_cast<MKL_Complex16**>(&values));
    if (st != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("MKL export failed");

    // Cast MKL_INT (32/64) to plain int if you compiled MKL LP64.
    n         = static_cast<int>(rows);
    m         = static_cast<int>(cols);
    ia        = reinterpret_cast<int*>(row_start);
    ja        = reinterpret_cast<int*>(columns);
    val       = reinterpret_cast<std::complex<double>*>(values);
    indexBase = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;

    /* row_end[i] is IA[i+1]; hence nnz = IA[n] - indexBase */
    nnz = static_cast<int>(row_end[n-1] - indexBase);
}





// PYBIND11_MODULE(propagation, m) {
//     m.doc() = "pybind11 propogation_hpddm function"; // optional module docstring

//     m.def("propagation_cg", &propagation_cg, "A function that propagate the input wave function by e^{-iHt}",
//     py::args("H")
//     );
// }

#endif