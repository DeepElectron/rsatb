#ifndef RSATB_BACKEND_KERNELS_HPP
#define RSATB_BACKEND_KERNELS_HPP

#include <complex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer(omp_priv = std::complex<double>(0, 0))
#endif

#ifdef RSATB_USE_MKL
#include <mkl.h>
#elif defined(RSATB_USE_OPENBLAS)
#include <cblas.h>
extern "C" void openblas_set_num_threads(int num_threads);
#endif

// Forward declare CBLAS functions if needed or include header
// For now, we assume a standard CBLAS interface is available via linking
// or we provide a naive fallback.

#if !defined(RSATB_USE_MKL) && !defined(RSATB_USE_OPENBLAS)
extern "C" {
    // Basic BLAS signatures
    void cblas_dgemv(const int Order, const int TransA, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *X, const int incX, const double beta,
                     double *Y, const int incY);
                     
    void cblas_dgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb,
                     const double beta, double *C, const int ldc);
                     
    // Complex versions (zgemv, zgemm) usually take void* for alpha/beta/scalars in some implementations
    // or pass by value in others. Standard CBLAS uses void*.
    void cblas_zgemv(const int Order, const int TransA, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *X, const int incX, const void *beta,
                     void *Y, const int incY);
                     
    void cblas_zgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const void *alpha, const void *A, const int lda,
                     const void *B, const int ldb,
                     const void *beta, void *C, const int ldc);
}
#endif

namespace rsatb {
namespace backend {

#if !defined(RSATB_USE_MKL) && !defined(RSATB_USE_OPENBLAS)
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
#endif

// Naive Kernel (Fallback)
template <typename T>
struct NaiveKernel {
    static void gemv(int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy, CBLAS_TRANSPOSE trans = CblasNoTrans) {
        if (trans == CblasNoTrans) {
            for (int i = 0; i < m; ++i) {
                T sum = 0;
                for (int j = 0; j < n; ++j) {
                    sum += A[i + j*lda] * x[j*incx];
                }
                y[i*incy] = alpha * sum + beta * y[i*incy];
            }
        } else if (trans == CblasTrans) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int i = 0; i < m; ++i) {
                    sum += A[i + j*lda] * x[i*incx];
                }
                y[j*incy] = alpha * sum + beta * y[j*incy];
            }
        } else if (trans == CblasConjTrans) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int i = 0; i < m; ++i) {
                    sum += ScalarTraits<T>::conjugate(A[i + j*lda]) * x[i*incx];
                }
                y[j*incy] = alpha * sum + beta * y[j*incy];
            }
        }
    }

    static void gemm(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans) {
        // Only NoTrans/NoTrans implemented for naive fallback for now as it's the most common
        if (transA == CblasNoTrans && transB == CblasNoTrans) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    T sum = 0;
                    for (int l = 0; l < k; ++l) {
                        sum += A[i + l*lda] * B[l + j*ldb];
                    }
                    C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        } else {
            throw std::runtime_error("NaiveKernel::gemm: Only NoTrans/NoTrans implemented.");
        }
    }
};

// Template Kernel for Fixed Block Sizes
// M, N are compile-time constants
template <typename T, int M, int N>
struct FixedBlockKernel {
    static void gemv(const T* __restrict__ A, const T* __restrict__ x, T* __restrict__ y, T alpha, T beta) {
        // y = alpha * A * x + beta * y
        // A is ColMajor: A[i + j*M]
        
        // Scale y
        #pragma omp simd
        for (int i = 0; i < M; ++i) {
            y[i] *= beta;
        }
        
        // Accumulate
        for (int j = 0; j < N; ++j) {
            T x_val = alpha * x[j];
            #pragma omp simd
            for (int i = 0; i < M; ++i) {
                y[i] += A[i + j*M] * x_val;
            }
        }
    }

    static void gemm(int n, const T* __restrict__ A, int lda, const T* __restrict__ B, int ldb, T* __restrict__ C, int ldc, T alpha, T beta) {
        // C = alpha * A * B + beta * C
        // A: M x N (template N is K here) (lda), B: N x n (runtime n) (ldb), C: M x n (ldc)
        // Fixed M, K (template N). Runtime n (num_vecs).
        
        for (int j = 0; j < n; ++j) {
            T* C_col = &C[j*ldc];
            
            #pragma omp simd
            for (int i = 0; i < M; ++i) {
                C_col[i] *= beta;
            }
            
            for (int l = 0; l < N; ++l) { // N here is K (block col dim)
                T b_val = alpha * B[l + j*ldb];
                const T* A_col = &A[l*lda];
                
                #pragma omp simd
                for (int i = 0; i < M; ++i) {
                    C_col[i] += A_col[i] * b_val;
                }
            }
        }
    }
};

// MKL/BLAS Kernel
struct BLASKernel {
    // Double
    static void gemv(int m, int n, double alpha, const double* A, int lda, const double* x, int incx, double beta, double* y, int incy, CBLAS_TRANSPOSE trans = CblasNoTrans) {
        cblas_dgemv(CblasColMajor, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    static void gemm(int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans) {
        cblas_dgemm(CblasColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // Complex Double
    static void gemv(int m, int n, std::complex<double> alpha, const std::complex<double>* A, int lda, const std::complex<double>* x, int incx, std::complex<double> beta, std::complex<double>* y, int incy, CBLAS_TRANSPOSE trans = CblasNoTrans) {
        cblas_zgemv(CblasColMajor, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    }

    static void gemm(int m, int n, int k, std::complex<double> alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, std::complex<double> beta, std::complex<double>* C, int ldc, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans) {
        cblas_zgemm(CblasColMajor, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }

    static void init_threading() {
#ifdef RSATB_USE_MKL
        mkl_set_num_threads(1);
#elif defined(RSATB_USE_OPENBLAS)
        openblas_set_num_threads(1);
#else
        // Generic BLAS: Do nothing. 
        // We do NOT want to call omp_set_num_threads(1) here because it would disable
        // parallelism for the outer loops (Sparse MVP/MM).
#endif
    }

    static std::string name() {
#ifdef RSATB_USE_MKL
        return "Intel MKL";
#elif defined(RSATB_USE_OPENBLAS)
        return "OpenBLAS";
#else
        return "Generic BLAS";
#endif
    }
};

// Smart Kernel Dispatcher
// Smart Kernel Dispatcher
template <typename T>
struct SmartKernel {
    // Dispatch macros
    #define CASE_GEMV(R, C) case C: FixedBlockKernel<T, R, C>::gemv(A, x, y, alpha, beta); break;
    #define SWITCH_GEMV(R) \
        switch(n) { \
            CASE_GEMV(R, 1) CASE_GEMV(R, 2) CASE_GEMV(R, 3) CASE_GEMV(R, 4) CASE_GEMV(R, 5) \
            CASE_GEMV(R, 6) CASE_GEMV(R, 7) CASE_GEMV(R, 8) CASE_GEMV(R, 9) CASE_GEMV(R, 10) \
            CASE_GEMV(R, 11) CASE_GEMV(R, 12) CASE_GEMV(R, 13) CASE_GEMV(R, 14) CASE_GEMV(R, 15) \
            CASE_GEMV(R, 16) \
            default: BLASKernel::gemv(m, n, alpha, A, m, x, 1, beta, y, 1); break; \
        }

    #define CASE_ROW_GEMV(R) case R: SWITCH_GEMV(R); break;

    static void gemv(int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy) {
        // Only optimized for unit stride and packed A (lda=m)
        if (incx == 1 && incy == 1 && lda == m) {
            switch(m) {
                CASE_ROW_GEMV(1) CASE_ROW_GEMV(2) CASE_ROW_GEMV(3) CASE_ROW_GEMV(4) CASE_ROW_GEMV(5)
                CASE_ROW_GEMV(6) CASE_ROW_GEMV(7) CASE_ROW_GEMV(8) CASE_ROW_GEMV(9) CASE_ROW_GEMV(10)
                CASE_ROW_GEMV(11) CASE_ROW_GEMV(12) CASE_ROW_GEMV(13) CASE_ROW_GEMV(14) CASE_ROW_GEMV(15)
                CASE_ROW_GEMV(16)
                default: BLASKernel::gemv(m, n, alpha, A, lda, x, incx, beta, y, incy); break;
            }
        } else {
            BLASKernel::gemv(m, n, alpha, A, lda, x, incx, beta, y, incy);
        }
    }

    #define CASE_GEMM(R, C) case C: FixedBlockKernel<T, R, C>::gemm(n, A, lda, B, ldb, C_ptr, ldc, alpha, beta); break;
    #define SWITCH_GEMM(R) \
        switch(k) { \
            CASE_GEMM(R, 1) CASE_GEMM(R, 2) CASE_GEMM(R, 3) CASE_GEMM(R, 4) CASE_GEMM(R, 5) \
            CASE_GEMM(R, 6) CASE_GEMM(R, 7) CASE_GEMM(R, 8) CASE_GEMM(R, 9) CASE_GEMM(R, 10) \
            CASE_GEMM(R, 11) CASE_GEMM(R, 12) CASE_GEMM(R, 13) CASE_GEMM(R, 14) CASE_GEMM(R, 15) \
            CASE_GEMM(R, 16) CASE_GEMM(R, 17) CASE_GEMM(R, 18) CASE_GEMM(R, 19) CASE_GEMM(R, 20) \
            CASE_GEMM(R, 21) CASE_GEMM(R, 22) CASE_GEMM(R, 23) CASE_GEMM(R, 24) CASE_GEMM(R, 25) \
            CASE_GEMM(R, 26) CASE_GEMM(R, 27) CASE_GEMM(R, 28) CASE_GEMM(R, 29) CASE_GEMM(R, 30) \
            CASE_GEMM(R, 31) CASE_GEMM(R, 32) CASE_GEMM(R, 33) CASE_GEMM(R, 34) CASE_GEMM(R, 35) \
            CASE_GEMM(R, 36) CASE_GEMM(R, 37) CASE_GEMM(R, 38) CASE_GEMM(R, 39) CASE_GEMM(R, 40) \
            CASE_GEMM(R, 41) CASE_GEMM(R, 44) CASE_GEMM(R, 45) CASE_GEMM(R, 46) CASE_GEMM(R, 49) CASE_GEMM(R, 54) \
            default: BLASKernel::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C_ptr, ldc); break; \
        }

    #define CASE_ROW_GEMM(R) case R: SWITCH_GEMM(R); break;

    static void gemm(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C_ptr, int ldc) {
        // Only optimized for packed A (lda=m) and NoTrans
        if (lda == m) {
            switch(m) {
                CASE_ROW_GEMM(1) CASE_ROW_GEMM(2) CASE_ROW_GEMM(3) CASE_ROW_GEMM(4) CASE_ROW_GEMM(5)
                CASE_ROW_GEMM(6) CASE_ROW_GEMM(7) CASE_ROW_GEMM(8) CASE_ROW_GEMM(9) CASE_ROW_GEMM(10)
                default: BLASKernel::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C_ptr, ldc); break;
            }
        } else {
            BLASKernel::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C_ptr, ldc);
        }
    }
    
    // Cleanup macros
    #undef CASE_GEMV
    #undef SWITCH_GEMV
    #undef CASE_ROW_GEMV
    #undef CASE_GEMM
    #undef SWITCH_GEMM
    #undef CASE_ROW_GEMM
};




// Default alias
template <typename T>
using DefaultKernel = BLASKernel; 
// Or NaiveKernel<T> if BLAS not available. 
// Ideally controlled by macro.

} // namespace backend
} // namespace rsatb

#endif
