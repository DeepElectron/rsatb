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
    // BLAS Level 3
    void dgemm_(const char* transa, const char* transb, const rsatb_lapack_int* m, const rsatb_lapack_int* n, const rsatb_lapack_int* k, 
                const double* alpha, const double* a, const rsatb_lapack_int* lda, const double* b, const rsatb_lapack_int* ldb, 
                const double* beta, double* c, const rsatb_lapack_int* ldc);

    void zgemm_(const char* transa, const char* transb, const rsatb_lapack_int* m, const rsatb_lapack_int* n, const rsatb_lapack_int* k, 
                const rsatb_complex_double* alpha, const rsatb_complex_double* a, const rsatb_lapack_int* lda, 
                const rsatb_complex_double* b, const rsatb_lapack_int* ldb, 
                const rsatb_complex_double* beta, rsatb_complex_double* c, const rsatb_lapack_int* ldc);

    void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag, 
                const rsatb_lapack_int* m, const rsatb_lapack_int* n, const double* alpha, 
                const double* a, const rsatb_lapack_int* lda, double* b, const rsatb_lapack_int* ldb);

    void ztrsm_(const char* side, const char* uplo, const char* transa, const char* diag, 
                const rsatb_lapack_int* m, const rsatb_lapack_int* n, const rsatb_complex_double* alpha, 
                const rsatb_complex_double* a, const rsatb_lapack_int* lda, rsatb_complex_double* b, const rsatb_lapack_int* ldb);

    // Double precision symmetric eigendecomposition
    void dsyev_(const char* jobz, const char* uplo, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, double* w, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);
    void dsyevd_(const char* jobz, const char* uplo, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, double* w, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* iwork, const rsatb_lapack_int* liwork, rsatb_lapack_int* info);
    
    // Complex double Hermitian eigendecomposition
    void zheev_(const char* jobz, const char* uplo, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, double* w, rsatb_complex_double* work, const rsatb_lapack_int* lwork, double* rwork, rsatb_lapack_int* info);
    void zheevd_(const char* jobz, const char* uplo, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, double* w, rsatb_complex_double* work, const rsatb_lapack_int* lwork, double* rwork, const rsatb_lapack_int* lrwork, rsatb_lapack_int* iwork, const rsatb_lapack_int* liwork, rsatb_lapack_int* info);

    // LU Factorization (Double)
    void dgetrf_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* ipiv, rsatb_lapack_int* info);
    
    // LU Inversion (Double)
    void dgetri_(const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, const rsatb_lapack_int* ipiv, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);

    // LU Factorization (Complex Double)
    void zgetrf_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* ipiv, rsatb_lapack_int* info);

    // LU Inversion (Complex Double)
    void zgetri_(const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, const rsatb_lapack_int* ipiv, rsatb_complex_double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);

    // Cholesky Factorization
    void dpotrf_(const char* uplo, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* info);
    void zpotrf_(const char* uplo, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* info);

    // Rank-Revealing QR (GEQP3)
    void dgeqp3_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* jpvt, double* tau, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);
    void zgeqp3_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, rsatb_lapack_int* jpvt, rsatb_complex_double* tau, rsatb_complex_double* work, const rsatb_lapack_int* lwork, double* rwork, rsatb_lapack_int* info);

    // SVD (GESVD)
    void dgesvd_(const char* jobu, const char* jobvt, const rsatb_lapack_int* m, const rsatb_lapack_int* n, double* a, const rsatb_lapack_int* lda, double* s, double* u, const rsatb_lapack_int* ldu, double* vt, const rsatb_lapack_int* ldvt, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);
    void zgesvd_(const char* jobu, const char* jobvt, const rsatb_lapack_int* m, const rsatb_lapack_int* n, rsatb_complex_double* a, const rsatb_lapack_int* lda, double* s, rsatb_complex_double* u, const rsatb_lapack_int* ldu, rsatb_complex_double* vt, const rsatb_lapack_int* ldvt, rsatb_complex_double* work, const rsatb_lapack_int* lwork, double* rwork, rsatb_lapack_int* info);

    // Generate Q from QR (ORGQR/UNGQR)
    void dorgqr_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, const rsatb_lapack_int* k, double* a, const rsatb_lapack_int* lda, const double* tau, double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);
    void zungqr_(const rsatb_lapack_int* m, const rsatb_lapack_int* n, const rsatb_lapack_int* k, rsatb_complex_double* a, const rsatb_lapack_int* lda, const rsatb_complex_double* tau, rsatb_complex_double* work, const rsatb_lapack_int* lwork, rsatb_lapack_int* info);

    // MKL Threading Control (Fortran Interface)
    void mkl_set_num_threads_(int* num_threads);
}

} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_LAPACK_TYPES_HPP
