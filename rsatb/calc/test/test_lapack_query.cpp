#include "calc/submf.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int n = 10;
    std::vector<double> M(n * n);
    for(int i=0; i<n*n; ++i) M[i] = (double)i / (n*n);
    // Make symmetric
    for(int i=0; i<n; ++i) {
        for(int j=i+1; j<n; ++j) {
            M[j + i*n] = M[i + j*n];
        }
    }

    std::function<double(double)> func = [](double x) { return std::sqrt(std::abs(x)); };

    std::cout << "Running Double Precision Test..." << std::endl;
    try {
        rsatb::calc::dense_matrix_function(n, M, func);
        std::cout << "Double precision test passed." << std::endl;
    } catch (...) {
        std::cerr << "Double precision test failed." << std::endl;
    }

    // Complex test
    std::vector<std::complex<double>> Mc(n * n);
    for(int i=0; i<n*n; ++i) Mc[i] = std::complex<double>((double)i / (n*n), 0.0);
    // Make Hermitian
    for(int i=0; i<n; ++i) {
        for(int j=i+1; j<n; ++j) {
            Mc[j + i*n] = std::conj(Mc[i + j*n]);
        }
    }
    
    std::function<std::complex<double>(double)> func_c = [](double x) { return std::complex<double>(std::sqrt(std::abs(x)), 0.0); };

    std::cout << "Running Complex Precision Test..." << std::endl;
    try {
        rsatb::calc::dense_matrix_function(n, Mc, func_c);
        std::cout << "Complex precision test passed." << std::endl;
    } catch (...) {
        std::cerr << "Complex precision test failed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
