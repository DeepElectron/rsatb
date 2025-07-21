// test_driver.cpp -----------------------------------------------------------
#include <mkl_spblas.h>
#include <mkl_cblas.h>
#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

using cd = std::complex<double>;
// ---------------------------------------------------------------------------
// Prototype of the routine from propagation.o

void propagate(const sparse_matrix_t&,
                  const matrix_descr&,
                  const std::vector<cd>&,
                  const std::vector<cd>&,
                  std::vector<cd>&,
                  std::size_t n,
                  std::size_t nrhs,
                  bool        forward = true);

std::vector<cd> dos(
                int& ih, 
                int& jh, 
                cd& h,
                std::vector<cd>& Psi_in,
                std::size_t n,
                std::size_t nrhs,
                std::vector<double>& ts,
                bool forward=true,
                double btol=1e-15
        );
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::cout << sizeof(MKL_INT) << std::endl;
    int provided;
    MPI_Init(&argc, &argv);

    const std::size_t n    = 2;     // matrix dimension
    const std::size_t nrhs = 2;     // two RHS columns
    const double      dt   = 0.3;   // physical Δt   (already scaled)

    // ----- 1) build H as CSR ----------------------------------------------
    MKL_INT rows[3]    = {0, 1, 2};         // row pointers
    MKL_INT cols[2]    = {1, 0};      // column indices
    cd      vals[2]    = {1.0, 1.0};

    sparse_matrix_t H_csr = nullptr;
    mkl_sparse_z_create_csr(&H_csr, SPARSE_INDEX_BASE_ZERO,
                            n, n, rows, rows + 1, cols,
                            reinterpret_cast<MKL_Complex16*>(vals));

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    // ----- 2) initial block of states --------------------------------------
    std::vector<cd> Psi_in  = { 1.0, 0.0,   // ψ¹
                                0.0, 1.0 }; // ψ²
    std::vector<cd> Psi_out(n * nrhs);

    // ----- 3) Bessel table --------------------------------------------------
    const std::size_t m_max = 5;
    std::vector<cd> B(m_max + 1);
    for(std::size_t k = 0; k <= m_max; ++k){
        B[k] = std::cyl_bessel_j(k, dt);   // J_k(dt)
        // std::cout << B[k] << ' ';
    }
    std::cout << "\n\n";

    // ----- 4) propagate -----------------------------------------------------
    propagate(H_csr, descr, B,
                 Psi_in, Psi_out,
                 n, nrhs,
                 /*forward=*/true);
    

    // ----- 5) dump results --------------------------------------------------
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial Ψ:\n";
    for(std::size_t j = 0; j < nrhs; ++j) {
        std::cout << "  column " << j << ": ";
        for(std::size_t i = 0; i < n; ++i)
            std::cout << Psi_in[i + j*n] << ' ';
        std::cout << '\n';
    }

    std::cout << "Propagated Ψ:\n";
    for(std::size_t j = 0; j < nrhs; ++j) {
        std::cout << "  column " << j << ": ";
        for(std::size_t i = 0; i < n; ++i)
            std::cout << Psi_out[i + j*n] << ' ';
        std::cout << '\n';
    }

    std::vector<double> ts(3);
    ts.data()[0] = 0.1;
    ts.data()[1] = 0.2;
    ts.data()[2] = 0.3;

    std::cout << "Sampled Time Steps: " << std::endl;
    for(int i = 0; i < 3; ++i) {
        std::cout << ts.data()[i] << ' ';
    }

    std::vector<cd> ddos = dos(*rows, *cols, *vals, Psi_in, 2, 2, ts);

    for(int i = 0; i < 3; ++i) {
        std::cout << ddos.data()[i] << ' ';
    }

    mkl_sparse_destroy(H_csr);

    MPI_Finalize();

    return 0;
}
