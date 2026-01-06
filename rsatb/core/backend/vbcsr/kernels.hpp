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

// Default alias
template <typename T>
using DefaultKernel = BLASKernel; 
// Or NaiveKernel<T> if BLAS not available. 
// Ideally controlled by macro.

} // namespace backend
} // namespace rsatb

#endif
