#ifndef RSATB_CALC_LAPACK_TYPES_HPP
#define RSATB_CALC_LAPACK_TYPES_HPP

// Ensure MKL is included if needed to get MKL_ILP64 definition
#if defined(VBCSR_USE_MKL) || defined(RSATB_USE_MKL)
    #include <mkl.h>
#endif

namespace rsatb {
namespace calc {

#if defined(VBCSR_USE_MKL) || defined(RSATB_USE_MKL)
    // Avoid using MKL_INT directly as it might be a macro causing parser issues.
    // Instead, rely on the standard MKL macros to determine the integer type.
    #if defined(MKL_ILP64)
        typedef long long rsatb_lapack_int;
    #else
        typedef int rsatb_lapack_int;
    #endif
#else
    // For Generic BLAS/LAPACK (OpenBLAS, Netlib), the standard is 'int' (32-bit).
    typedef int rsatb_lapack_int;
#endif

} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_LAPACK_TYPES_HPP
