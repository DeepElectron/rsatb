#ifndef RSATB_CALC_LAPACK_ISQRT_HPP
#define RSATB_CALC_LAPACK_ISQRT_HPP

// LAPACK/BLAS prototypes for the dense inverse-square-root action.
//
// Declared here rather than borrowed from vbcsr because these are rsatb's
// needs, not vbcsr's: of the routines below only dgemm/zgemm/dsyevd/zheevd
// are used on both sides, and the rest -- band Cholesky, the two-stage
// dense-to-band reduction, blocked QR, condition estimation -- exist solely
// for the routes in isqrt_action.hpp. They used to live in vbcsr's lapack_api.hpp,
// which meant a linear-algebra library carried declarations for a
// chemistry-side kernel and rsatb pulled them in with a namespace-wide
// "using namespace vbcsr".
//
// The integer width is deliberately NOT redeclared: it encodes the ILP64
// build choice and must match whatever the linked library was built
// against, so vbcsr's single typedef stays the one source of truth.
// Repeating a C-linkage declaration in another namespace is well-defined
// and refers to the same function, provided the signature agrees -- which
// is why these are copies rather than paraphrases.

#include "vbcsr/core/detail/kernels/lapack_api.hpp"

#include <complex>

namespace rsatb {
namespace calc {
namespace isqrt {

using vbcsr::vbcsr_lapack_int;
using vbcsr::vbcsr_complex_double;

extern "C" {
    void dgemm_(const char* transa, const char* transb, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, 
                const double* alpha, const double* a, const vbcsr_lapack_int* lda, const double* b, const vbcsr_lapack_int* ldb, 
                const double* beta, double* c, const vbcsr_lapack_int* ldc);
    void dgeqrf_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, double* tau, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void dlarfb_(const char* side, const char* trans, const char* direct, const char* storev, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, const double* v, const vbcsr_lapack_int* ldv, const double* t, const vbcsr_lapack_int* ldt, double* c, const vbcsr_lapack_int* ldc, double* work, const vbcsr_lapack_int* ldwork);
    void dlarft_(const char* direct, const char* storev, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, const double* v, const vbcsr_lapack_int* ldv, const double* tau, double* t, const vbcsr_lapack_int* ldt);
    void dorgqr_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, double* a, const vbcsr_lapack_int* lda, const double* tau, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void dpbtrf_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* kd, double* ab, const vbcsr_lapack_int* ldab, vbcsr_lapack_int* info);
    void dpbtrs_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* kd, const vbcsr_lapack_int* nrhs, const double* ab, const vbcsr_lapack_int* ldab, double* b, const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
    void dpocon_(const char* uplo, const vbcsr_lapack_int* n, const double* a, const vbcsr_lapack_int* lda, const double* anorm, double* rcond, double* work, vbcsr_lapack_int* iwork, vbcsr_lapack_int* info);
    void dpotrf_(const char* uplo, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* info);
    void dpotrs_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* nrhs, const double* a, const vbcsr_lapack_int* lda, double* b, const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
    void dstevd_(const char* jobz, const vbcsr_lapack_int* n, double* d, double* e, double* z, const vbcsr_lapack_int* ldz, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* iwork, const vbcsr_lapack_int* liwork, vbcsr_lapack_int* info);
    void dsyevd_(const char* jobz, const char* uplo, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, double* w, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* iwork, const vbcsr_lapack_int* liwork, vbcsr_lapack_int* info);
    void zgemm_(const char* transa, const char* transb, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, 
                const vbcsr_complex_double* alpha, const vbcsr_complex_double* a, const vbcsr_lapack_int* lda, 
                const vbcsr_complex_double* b, const vbcsr_lapack_int* ldb, 
                const vbcsr_complex_double* beta, vbcsr_complex_double* c, const vbcsr_lapack_int* ldc);
    void zgeqrf_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_complex_double* tau, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void zheevd_(const char* jobz, const char* uplo, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, double* w, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, double* rwork, const vbcsr_lapack_int* lrwork, vbcsr_lapack_int* iwork, const vbcsr_lapack_int* liwork, vbcsr_lapack_int* info);
    void zlarfb_(const char* side, const char* trans, const char* direct, const char* storev, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, const vbcsr_complex_double* v, const vbcsr_lapack_int* ldv, const vbcsr_complex_double* t, const vbcsr_lapack_int* ldt, vbcsr_complex_double* c, const vbcsr_lapack_int* ldc, vbcsr_complex_double* work, const vbcsr_lapack_int* ldwork);
    void zlarft_(const char* direct, const char* storev, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, const vbcsr_complex_double* v, const vbcsr_lapack_int* ldv, const vbcsr_complex_double* tau, vbcsr_complex_double* t, const vbcsr_lapack_int* ldt);
    void zpbtrf_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* kd, vbcsr_complex_double* ab, const vbcsr_lapack_int* ldab, vbcsr_lapack_int* info);
    void zpbtrs_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* kd, const vbcsr_lapack_int* nrhs, const vbcsr_complex_double* ab, const vbcsr_lapack_int* ldab, vbcsr_complex_double* b, const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
    void zpocon_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_complex_double* a, const vbcsr_lapack_int* lda, const double* anorm, double* rcond, vbcsr_complex_double* work, double* rwork, vbcsr_lapack_int* info);
    void zpotrf_(const char* uplo, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* info);
    void zpotrs_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* nrhs, const vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_complex_double* b, const vbcsr_lapack_int* ldb, vbcsr_lapack_int* info);
    void zungqr_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, const vbcsr_complex_double* tau, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);

    // Two-stage tridiagonal reduction (the dense-to-band route).
    void dsytrd_sy2sb_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* kd, double* a, const vbcsr_lapack_int* lda, double* ab, const vbcsr_lapack_int* ldab, double* tau, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void zhetrd_he2hb_(const char* uplo, const vbcsr_lapack_int* n, const vbcsr_lapack_int* kd, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_complex_double* ab, const vbcsr_lapack_int* ldab, vbcsr_complex_double* tau, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);

}

}  // namespace isqrt
}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_LAPACK_ISQRT_HPP
