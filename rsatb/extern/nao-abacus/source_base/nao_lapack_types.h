#ifndef NAO_LAPACK_TYPES_H
#define NAO_LAPACK_TYPES_H

// Ensure MKL is included if needed to get MKL_ILP64 definition
#if defined(__MKL)
    #include <mkl.h>
#endif

#if defined(__MKL)
    // Avoid using MKL_INT directly as it might be a macro causing parser issues.
    // Instead, rely on the standard MKL macros to determine the integer type.
    #if defined(MKL_ILP64)
        typedef long long nao_lapack_int;
    #else
        typedef int nao_lapack_int;
    #endif
#else
    // For Generic BLAS/LAPACK (OpenBLAS, Netlib), the standard is 'int' (32-bit).
    typedef int nao_lapack_int;
#endif

#endif // NAO_LAPACK_TYPES_H
