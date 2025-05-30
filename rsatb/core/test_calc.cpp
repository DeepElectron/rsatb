#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#include <mkl_spblas.h>
#include <mkl_cblas.h>
#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include "calc.hpp"

using cd = std::complex<double>;

int main(int argc, char** argv){
    // MPI_Init(&argc, &argv);

    const std::size_t n    = 2;     // matrix dimension
    const std::size_t nrhs = 2;     // two RHS columns

    MKL_INT rows[3]    = {0, 1, 2};         // row pointers
    MKL_INT cols[2]    = {1, 0};      // column indices
    cd      vals[2];
    vals[0] = cd {1, 0};
    vals[1] = cd {1, 0};
    MKL_Complex16 *vals_ = reinterpret_cast<MKL_Complex16*>(vals);

    std::cout << "vals[0] " << vals_[0] << std::endl;
    std::cout << "vals[1] " << vals_[1] << std::endl;

    std::vector<double> ts = {0.1, 0.2, 0.3};

    
    MKL_INT rows_r[3]    = {0, 2, 3};         // row pointers
    MKL_INT *rows_rp = rows_r;
    MKL_INT cols_r[3]    = {0, 1, 1};      // column indices
    MKL_INT *cols_rp = cols_r;
    cd      vals_r[3];
    vals_r[0] = cd {1, 0};
    vals_r[1] = cd {0.1, 0};
    vals_r[2] = cd {1, 0};
    cd      *vals_rp = vals_r;

    std::vector<cd> Psi_in  = { 1.0, 0.0,   // ψ¹
                                0.0, 1.0 }; // ψ²
    std::vector<cd> Psi_out(n * nrhs);

    std::cout << "[DEBUG] step before dos " << std::endl;

    tdos(
        rows, cols, vals_, Psi_in.data(), n, nrhs, ts,
        // nullptr, nullptr, nullptr, true
        rows_rp, cols_rp, vals_rp, true
    );

    // MPI_Finalize();
    return 0;
}