#ifndef NAO_LAPACK_TYPES_H
#define NAO_LAPACK_TYPES_H

// Ensure MKL is included if needed to get MKL_ILP64 definition
#if defined(__MKL)
    #include <mkl.h>
#endif

#if defined(__MKL)
    // Use MKL_INT directly to ensure compatibility with the included mkl.h
    typedef MKL_INT nao_lapack_int;
#else
    // For Generic BLAS/LAPACK (OpenBLAS, Netlib), the standard is 'int' (32-bit).
    typedef int nao_lapack_int;
#endif

#endif // NAO_LAPACK_TYPES_H
