#ifndef RSATB_CALC_LAPACK_TYPES_HPP
#define RSATB_CALC_LAPACK_TYPES_HPP

#include "calc/config.hpp"
#include <complex>

namespace rsatb {
namespace calc {

#if defined(RSATB_USE_ILP64)
    typedef long long rsatb_lapack_int;
#else
    typedef int rsatb_lapack_int;
#endif

using rsatb_complex_double = std::complex<double>;

extern "C" {
    // Double precision symmetric eigendecomposition
    void dsyev_(const char* jobz, const char* uplo, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, double* w, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);
    
    // Complex double Hermitian eigendecomposition
    void zheev_(const char* jobz, const char* uplo, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, double* w, rsatb_complex_double* work, const rsatb_lapack_int* lwork, double* rwork, rsatb_lapack_int* info);

    // LU Factorization (Double)
    void dgetrf_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* ipiv, rsatb_lapack_int* info);
    
    // LU Inversion (Double)
    void dgetri_(const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, const rsatb_lapack_int* ipiv, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);

    // LU Factorization (Complex Double)
    void zgetrf_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* ipiv, rsatb_lapack_int* info);

    // LU Inversion (Complex Double)
    void zgetri_(const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, const rsatb_lapack_int* ipiv, rsatb_complex_double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);

    // MKL Threading Control (Fortran Interface)
    void mkl_set_num_threads_(int* num_threads);
}

} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_LAPACK_TYPES_HPP
