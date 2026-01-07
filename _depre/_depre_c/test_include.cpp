#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include "HPDDM.hpp"
#include <iostream>

int main() {
    MKLVersion ver;
    mkl_get_version(&ver);
    std::cout
        << "Intel MKL version "
        << ver.MajorVersion << "." << ver.MinorVersion << "." << ver.UpdateVersion
        << " (" << ver.ProductStatus << "), build " << ver.Build
        << "\nPlatform: " << ver.Platform
        << "\nOptimized for: " << ver.Processor
        << std::endl;
    return 0;
}