#ifndef RSATB_CALC_ISQRT_ACTION_HPP
#define RSATB_CALC_ISQRT_ACTION_HPP

// Thin action of the dense inverse square root:  X = A^{-1/2} B,
// with A real symmetric / complex Hermitian positive definite (n x n) and
// B a thin block of nrhs columns (nrhs ~ the orbital count of one atom).
//
// This is the per-subgraph kernel of the graph matrix function used to
// initialise S^{-1/2} (see isqrtm.hpp in this directory).  Only one block row of f(A) is ever
// needed, so a full eigendecomposition (the previous route, xSYEVD/xHEEVD) is
// wasteful: it costs a large multiple of n^3 to produce n^2 numbers of which
// only n*nrhs are used.
//
// The kernel is templated on the scalar type and computes in that type: a real
// matrix takes the (~3-4x cheaper) real LAPACK path, a complex matrix the
// complex one. Which arithmetic applies is the CALLER's choice, made by the
// type of the matrix handed in - there is no content inspection here.
//
// Three routes are implemented, dispatched automatically:
//
//   1. Fast path - "adaptive Cholesky + inverse block Lanczos".
//      One Cholesky factorization A = L L^H, then block Lanczos on
//      M = A^{-1} started from orth(B):  A^{-1/2} B = M^{1/2} B is
//      approximated by Q_m H_m^{1/2} E_1 R_0 with H_m = Q_m^H M Q_m.
//      Inverting maps the small overlap eigenvalues (which are what make
//      x^{-1/2} hard) to the *dominant* eigenvalues of M, exactly the modes a
//      Krylov space captures first, and only those coupling to B.  Dominant
//      cost is the single n^3/3 factorization.  Convergence is certified by
//      comparing nested projected approximations; a rejected certificate falls
//      through to route 2.
//
//   2. Robust path - tridiagonalize once (xSYTRD/xHETRD), apply the implicit Q
//      to the thin block only (xORMTR/xUNMTR), and evaluate T^{-1/2} C with the
//      sinc/trapezoidal rational form of  x^{-1/2} = (2/pi) int dt/(x+t^2).
//      Each pole is a shifted tridiagonal solve, O(n*nrhs).  Cost is the
//      4n^3/3 reduction, which is still far below a full eigensolve.
//
//   3. Last resort - full eigendecomposition with the clamped function
//      f(x) = (x > eigen_clamp) ? x^{-1/2} : 0, i.e. exactly the behaviour of
//      the previous implementation.  Used for tiny problems, for numerically
//      singular A (where the clamp actually matters), and whenever a LAPACK
//      call fails.
//
// The routine never throws: it is called from inside an OpenMP parallel region
// in the graph loop, where an escaping exception would terminate the process.
// Failures are reported through the returned Status.
//
// Storage convention matches vbcsr's dense_matrix_function: A is read as
// column-major with only the triangle named by Options::uplo referenced
// (default 'U', which for the row-major dense views produced by
// BlockSpMat::to_dense is the same triangle the old xSYEVD call used).

#include "vbcsr/core/detail/kernels/lapack_api.hpp"

#include "lapack_isqrt.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

namespace rsatb {
namespace calc {
namespace isqrt {

// The BLAS/LAPACK prototypes this kernel calls are declared in
// lapack_isqrt.hpp, in this namespace; the integer width comes from vbcsr, so
// there is one source of truth for the ILP64 choice.
using vbcsr::vbcsr_lapack_int;
using vbcsr::vbcsr_complex_double;

enum class Method {
    Auto,
    InverseBlockLanczos,  // fast path (Cholesky + inverse block Lanczos)
    BandZolotarev,        // robust path: dense-to-band + Zolotarev rational action
    Eigen                 // clamped eigendecomposition
};

enum class Status {
    Success,
    InvalidArgument,
    FastPathRejected,     // only returned when the fast path was forced
    RankBreakdown,
    NotPositiveDefinite,
    NearSingular,
    LapackFailure
};

struct Options {
    Method method = Method::Auto;
    char uplo = 'U';

    // Target relative Frobenius accuracy of X = A^{-1/2} B.
    double relative_tolerance = 1.0e-8;

    // The nested-difference certificate of the fast path is empirical, so it
    // aims below the requested tolerance to leave margin. <= 0 means
    // 0.25 * relative_tolerance.
    double fast_internal_tolerance = 0.0;

    int max_krylov_blocks = 28;
    int checkpoint_stride = 4;
    int minimum_checkpoints = 3;

    double accept_ratio = 0.65;   // geometric decay rate accepted as converging
    double reject_ratio = 0.90;   // decay rate deemed hopeless
    double rank_tolerance = 1.0e-13;

    // Scalar target of the rational approximations (Zolotarev and sinc);
    // <= 0 means 0.01 * relative_tolerance.
    double rational_scalar_tolerance = 0.0;

    // Half-bandwidth of the dense-to-band reduction; 0 picks a size-based
    // default (~n/20, clamped to [64, 256]). Autotune per machine if the
    // route dominates a production profile.
    int band_width = 0;
    // Stage-1 reflectors grouped per compact-WY (LARFT/LARFB) application.
    int reflector_block_size = 16;

    // f(x) = (x > eigen_clamp) ? x^{-1/2} : 0 on the eigen route.
    double eigen_clamp = 1.0e-12;

    // A smallest eigenvalue at or below this level is routed to the clamped
    // eigen path: the clamp becomes semantically relevant and no
    // approximation is trustworthy there anyway.
    double lambda_min_floor = 1.0e-10;

    // Problems this small are cheaper to diagonalize outright.
    int direct_eigen_max_n = 256;

    // The fast path costs about n^3/3 + 2 m nrhs n^2 (one Cholesky plus m
    // block solves), the robust path about 4n^3/3, so the fast path only pays
    // off while n >> 2 m nrhs.  With the measured m ~ 16 blocks that break-even
    // sits near n = 32 nrhs; below the ratio set here the Krylov space cannot
    // stay thin enough and the robust path is taken directly.
    int min_n_over_nrhs = 24;
    // Cap on the Krylov basis as a fraction 1/krylov_space_divisor of n: a
    // basis approaching the full space is never a win over a direct method.
    int krylov_space_divisor = 2;

    // Set only when B^H B == I to working precision (the selector case).
    bool rhs_is_orthonormal = false;
};

struct Stats {
    Method used_method = Method::Auto;
    Status status = Status::Success;

    bool fast_path_attempted = false;
    bool fast_path_accepted = false;

    int krylov_blocks = 0;
    int krylov_dimension = 0;
    int rational_poles = 0;
    int band_width = 0;

    double estimated_relative_error = 0.0;
    double last_nested_difference = 0.0;
    double convergence_ratio = 0.0;
    double lambda_min = 0.0;
    double lambda_max = 0.0;
    double rcond_estimate = 0.0;

};

// Per-run dispatch memory.  The graph loop solves one problem per atom and the
// spectra of neighbouring environments are alike, so a run whose fast path
// keeps getting rejected should stop paying for the trial.  Re-probes
// periodically so a heterogeneous system is not locked out.  Thread-safe.
//
// The counters are updated from several threads, so which route a given atom
// takes near the accept/reject boundary depends on the interleaving: results can
// differ between runs at the level of the requested tolerance.  Force a single
// route (Options::method, or RSATB_ISQRT_INIT) when run-to-run reproducibility
// matters more than speed.
class Router {
public:
    explicit Router(int probe_attempts = 8, int reject_percent = 60, int reprobe_period = 32)
        : probe_attempts_(probe_attempts),
          reject_percent_(reject_percent),
          reprobe_period_(reprobe_period) {}

    bool try_fast_path() {
        const int attempts = attempts_.load(std::memory_order_relaxed);
        const int rejects = rejects_.load(std::memory_order_relaxed);
        if (attempts < probe_attempts_) return true;
        if (rejects * 100 < attempts * reject_percent_) return true;
        const int skipped = skipped_.fetch_add(1, std::memory_order_relaxed);
        return reprobe_period_ > 0 && (skipped % reprobe_period_) == 0;
    }

    void record(bool accepted) {
        attempts_.fetch_add(1, std::memory_order_relaxed);
        if (!accepted) rejects_.fetch_add(1, std::memory_order_relaxed);
    }

    int attempts() const { return attempts_.load(std::memory_order_relaxed); }
    int rejects() const { return rejects_.load(std::memory_order_relaxed); }
    int skipped() const { return skipped_.load(std::memory_order_relaxed); }

private:
    std::atomic<int> attempts_{0};
    std::atomic<int> rejects_{0};
    std::atomic<int> skipped_{0};
    int probe_attempts_;
    int reject_percent_;
    int reprobe_period_;
};

namespace detail {

template <typename T>
constexpr bool is_complex_v = std::is_same<T, std::complex<double>>::value;

template <typename T>
inline double abs2(const T& x) {
    if constexpr (is_complex_v<T>) return std::norm(x);
    else return x * x;
}

template <typename T>
inline T conj_if(const T& x) {
    if constexpr (is_complex_v<T>) return std::conj(x);
    else return x;
}

inline const char* trans_str(char c, bool cplx) {
    if (c == 'N') return "N";
    if (c == 'T') return "T";
    return cplx ? "C" : "T";
}

// ---------------------------------------------------------------- BLAS/LAPACK

template <typename T>
void gemm(char ta, char tb, int m, int n, int k, T alpha, const T* a, int lda,
          const T* b, int ldb, T beta, T* c, int ldc) {
    vbcsr_lapack_int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    const char* sa = trans_str(ta, is_complex_v<T>);
    const char* sb = trans_str(tb, is_complex_v<T>);
    if constexpr (std::is_same<T, double>::value) {
        dgemm_(sa, sb, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta, c, &ldc_);
    } else {
        zgemm_(sa, sb, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta, c, &ldc_);
    }
}

template <typename T>
vbcsr_lapack_int potrf(char uplo, int n, T* a, int lda) {
    const char u[2] = {uplo, '\0'};
    vbcsr_lapack_int n_ = n, lda_ = lda, info = 0;
    if constexpr (std::is_same<T, double>::value) {
        dpotrf_(u, &n_, a, &lda_, &info);
    } else {
        zpotrf_(u, &n_, a, &lda_, &info);
    }
    return info;
}

template <typename T>
vbcsr_lapack_int potrs(char uplo, int n, int nrhs, const T* a, int lda, T* b, int ldb) {
    const char u[2] = {uplo, '\0'};
    vbcsr_lapack_int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb, info = 0;
    if constexpr (std::is_same<T, double>::value) {
        dpotrs_(u, &n_, &nrhs_, a, &lda_, b, &ldb_, &info);
    } else {
        zpotrs_(u, &n_, &nrhs_, a, &lda_, b, &ldb_, &info);
    }
    return info;
}

template <typename T>
vbcsr_lapack_int pocon(char uplo, int n, const T* factor, int lda, double anorm, double* rcond) {
    const char u[2] = {uplo, '\0'};
    vbcsr_lapack_int n_ = n, lda_ = lda, info = 0;
    if constexpr (std::is_same<T, double>::value) {
        std::vector<double> work(static_cast<std::size_t>(3) * n);
        std::vector<vbcsr_lapack_int> iwork(static_cast<std::size_t>(n));
        dpocon_(u, &n_, factor, &lda_, &anorm, rcond, work.data(), iwork.data(), &info);
    } else {
        std::vector<T> work(static_cast<std::size_t>(2) * n);
        std::vector<double> rwork(static_cast<std::size_t>(n));
        zpocon_(u, &n_, factor, &lda_, &anorm, rcond, work.data(), rwork.data(), &info);
    }
    return info;
}

// Full eigendecomposition; on exit a holds the eigenvectors (column-major).
template <typename T>
vbcsr_lapack_int heevd(char uplo, int n, T* a, int lda, double* w) {
    const char jobz[2] = {'V', '\0'};
    const char u[2] = {uplo, '\0'};
    vbcsr_lapack_int n_ = n, lda_ = lda, info = 0;
    vbcsr_lapack_int lwork = -1, lrwork = -1, liwork = -1;
    if constexpr (std::is_same<T, double>::value) {
        double wq = 0.0;
        vbcsr_lapack_int iwq = 0;
        dsyevd_(jobz, u, &n_, a, &lda_, w, &wq, &lwork, &iwq, &liwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, wq));
        liwork = std::max<vbcsr_lapack_int>(1, iwq);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        std::vector<vbcsr_lapack_int> iwork(static_cast<std::size_t>(liwork));
        dsyevd_(jobz, u, &n_, a, &lda_, w, work.data(), &lwork, iwork.data(), &liwork, &info);
    } else {
        T wq = T(0);
        double rwq = 0.0;
        vbcsr_lapack_int iwq = 0;
        zheevd_(jobz, u, &n_, a, &lda_, w, &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, wq.real()));
        lrwork = static_cast<vbcsr_lapack_int>(std::max(1.0, rwq));
        liwork = std::max<vbcsr_lapack_int>(1, iwq);
        std::vector<T> work(static_cast<std::size_t>(lwork));
        std::vector<double> rwork(static_cast<std::size_t>(lrwork));
        std::vector<vbcsr_lapack_int> iwork(static_cast<std::size_t>(liwork));
        zheevd_(jobz, u, &n_, a, &lda_, w, work.data(), &lwork, rwork.data(), &lrwork,
                       iwork.data(), &liwork, &info);
    }
    return info;
}



// Thin QR of the m x k block a (m >= k).  On exit a holds Q; r (k x k,
// column-major) holds the upper triangular factor.
template <typename T>
vbcsr_lapack_int thin_qr(int m, int k, T* a, int lda, std::vector<T>& r) {
    vbcsr_lapack_int m_ = m, k_ = k, lda_ = lda, lwork = -1, info = 0;
    std::vector<T> tau(static_cast<std::size_t>(k));
    T query = T(0);

    if constexpr (std::is_same<T, double>::value) {
        dgeqrf_(&m_, &k_, a, &lda_, tau.data(), &query, &lwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, query));
        std::vector<T> work(static_cast<std::size_t>(lwork));
        dgeqrf_(&m_, &k_, a, &lda_, tau.data(), work.data(), &lwork, &info);
    } else {
        zgeqrf_(&m_, &k_, a, &lda_, tau.data(), &query, &lwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, query.real()));
        std::vector<T> work(static_cast<std::size_t>(lwork));
        zgeqrf_(&m_, &k_, a, &lda_, tau.data(), work.data(), &lwork, &info);
    }
    if (info != 0) return info;

    r.assign(static_cast<std::size_t>(k) * k, T(0));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i <= j; ++i) {
            r[i + static_cast<std::size_t>(j) * k] = a[i + static_cast<std::size_t>(j) * lda];
        }
    }

    lwork = -1;
    query = T(0);
    if constexpr (std::is_same<T, double>::value) {
        dorgqr_(&m_, &k_, &k_, a, &lda_, tau.data(), &query, &lwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, query));
        std::vector<T> work(static_cast<std::size_t>(lwork));
        dorgqr_(&m_, &k_, &k_, a, &lda_, tau.data(), work.data(), &lwork, &info);
    } else {
        zungqr_(&m_, &k_, &k_, a, &lda_, tau.data(), &query, &lwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, query.real()));
        std::vector<T> work(static_cast<std::size_t>(lwork));
        zungqr_(&m_, &k_, &k_, a, &lda_, tau.data(), work.data(), &lwork, &info);
    }
    return info;
}

// ------------------------------------------------------------------- helpers

template <typename T>
double qr_rank_ratio(const std::vector<T>& r, int k) {
    double hi = 0.0;
    double lo = std::numeric_limits<double>::infinity();
    for (int i = 0; i < k; ++i) {
        const double v = std::abs(r[i + static_cast<std::size_t>(i) * k]);
        hi = std::max(hi, v);
        lo = std::min(lo, v);
    }
    if (!(hi > 0.0)) return 0.0;
    return lo / hi;
}

inline vbcsr_lapack_int stevd(int n, double* d, double* e, double* z, int ldz) {
    const char jobz[2] = {'V', '\0'};
    vbcsr_lapack_int n_ = n, ldz_ = ldz, lwork = -1, liwork = -1, info = 0;
    double wq = 0.0;
    vbcsr_lapack_int iwq = 0;
    dstevd_(jobz, &n_, d, e, z, &ldz_, &wq, &lwork, &iwq, &liwork, &info);
    if (info != 0) return info;
    lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, wq));
    liwork = std::max<vbcsr_lapack_int>(1, iwq);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    std::vector<vbcsr_lapack_int> iwork(static_cast<std::size_t>(liwork));
    dstevd_(jobz, &n_, d, e, z, &ldz_, work.data(), &lwork, iwork.data(), &liwork, &info);
    return info;
}

// Copy the referenced triangle of A into a fresh column-major n x n buffer,
// leaving the other triangle zero: every route reads one triangle only.
template <typename T>
std::vector<T> copy_triangle(const T* a, int lda, int n, char uplo) {
    std::vector<T> out(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j) {
        const int lo = (uplo == 'U') ? 0 : j;
        const int hi = (uplo == 'U') ? j : n - 1;
        for (int i = lo; i <= hi; ++i) {
            out[i + static_cast<std::size_t>(j) * n] = a[i + static_cast<std::size_t>(j) * lda];
        }
    }
    return out;
}

// 1-norm of the Hermitian matrix stored in one triangle.
template <typename T>
double hermitian_one_norm(const T* a, int lda, int n, char uplo) {
    std::vector<double> colsum(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        const int lo = (uplo == 'U') ? 0 : j;
        const int hi = (uplo == 'U') ? j : n - 1;
        for (int i = lo; i <= hi; ++i) {
            const double v = std::abs(a[i + static_cast<std::size_t>(j) * lda]);
            colsum[j] += v;
            if (i != j) colsum[i] += v;
        }
    }
    double out = 0.0;
    for (double v : colsum) out = std::max(out, v);
    return out;
}

template <typename T>
std::vector<T> copy_block(const T* b, int ldb, int rows, int cols) {
    std::vector<T> out(static_cast<std::size_t>(rows) * cols);
    for (int j = 0; j < cols; ++j) {
        std::copy(b + static_cast<std::size_t>(j) * ldb,
                  b + static_cast<std::size_t>(j) * ldb + rows,
                  out.begin() + static_cast<std::size_t>(j) * rows);
    }
    return out;
}

template <typename T>
void store_block(const std::vector<T>& x, int rows, int cols, T* out, int ldout) {
    for (int j = 0; j < cols; ++j) {
        std::copy(x.begin() + static_cast<std::size_t>(j) * rows,
                  x.begin() + static_cast<std::size_t>(j + 1) * rows,
                  out + static_cast<std::size_t>(j) * ldout);
    }
}

template <typename T>
double relative_difference(const std::vector<T>& x, const std::vector<T>& y) {
    long double num = 0.0L, den = 0.0L;
    const std::size_t m = std::min(x.size(), y.size());
    for (std::size_t i = 0; i < m; ++i) {
        num += static_cast<long double>(abs2<T>(x[i] - y[i]));
        den += static_cast<long double>(abs2<T>(x[i]));
    }
    if (!(den > 0.0L)) return 0.0;
    return static_cast<double>(std::sqrt(num / den));
}

// H^{1/2} E_1 for the compact Hermitian q x q matrix h; E_1 are the first
// nrhs coordinate columns.  h is consumed.
template <typename T>
bool projected_sqrt_first_columns(std::vector<T>& h, int q, int nrhs, std::vector<T>& out) {
    std::vector<double> w(static_cast<std::size_t>(q));
    if (heevd<T>('L', q, h.data(), q, w.data()) != 0) return false;

    const double scale = std::max(1.0, std::abs(w[q - 1]));
    if (w[0] < -1.0e-11 * scale) return false;  // projected A^{-1} lost SPD-ness

    std::vector<T> weighted(static_cast<std::size_t>(q) * nrhs);
    for (int j = 0; j < nrhs; ++j) {
        for (int k = 0; k < q; ++k) {
            const T vjk = h[j + static_cast<std::size_t>(k) * q];  // V(j,k)
            weighted[k + static_cast<std::size_t>(j) * q] =
                std::sqrt(std::max(0.0, w[k])) * conj_if(vjk);
        }
    }
    out.assign(static_cast<std::size_t>(q) * nrhs, T(0));
    gemm<T>('N', 'N', q, nrhs, q, T(1), h.data(), q, weighted.data(), q, T(0), out.data(), q);
    return true;
}

// Leading q x q block of the qcap-strided store, explicitly Hermitized.
template <typename T>
std::vector<T> compact_leading(const std::vector<T>& store, int stride, int q) {
    std::vector<T> out(static_cast<std::size_t>(q) * q);
    for (int j = 0; j < q; ++j) {
        std::copy(store.begin() + static_cast<std::size_t>(j) * stride,
                  store.begin() + static_cast<std::size_t>(j) * stride + q,
                  out.begin() + static_cast<std::size_t>(j) * q);
    }
    for (int j = 0; j < q; ++j) {
        for (int i = j + 1; i < q; ++i) {
            const T z = T(0.5) * (out[i + static_cast<std::size_t>(j) * q] +
                                  conj_if(out[j + static_cast<std::size_t>(i) * q]));
            out[i + static_cast<std::size_t>(j) * q] = z;
            out[j + static_cast<std::size_t>(i) * q] = conj_if(z);
        }
        out[j + static_cast<std::size_t>(j) * q] =
            T(std::real(out[j + static_cast<std::size_t>(j) * q]));
    }
    return out;
}



// -------------------------------------------------- Zolotarev on a band matrix

// Complete elliptic integral of the first kind K(k), by the arithmetic-
// geometric mean. Accurate to long double precision for 0 <= k < 1.
inline long double elliptic_K(long double k) {
    long double a = 1.0L, b = std::sqrt(std::max(0.0L, 1.0L - k * k));
    while (std::abs(a - b) > 1e-18L * a) {
        const long double an = 0.5L * (a + b);
        b = std::sqrt(a * b);
        a = an;
    }
    const long double pi = 3.14159265358979323846264338327950288L;
    return pi / (2.0L * a);
}

// Jacobi elliptic sn(u, k) by the descending Landen (AGM) transformation
// (Abramowitz & Stegun 16.12; the standard sncndn recurrence).
inline long double jacobi_sn(long double u, long double k) {
    const long double emmc = 1.0L - k * k;  // complementary parameter
    if (emmc == 0.0L) return std::tanh(u);

    constexpr int kMax = 16;
    long double em[kMax], en[kMax];
    long double a = 1.0L, dn = 1.0L, emc = emmc, c = 0.0L;
    int l = 0;
    for (int i = 0; i < kMax; ++i) {
        l = i;
        em[i] = a;
        emc = std::sqrt(emc);
        en[i] = emc;
        c = 0.5L * (a + emc);
        if (std::abs(a - emc) <= 1e-17L * a) break;
        emc *= a;
        a = c;
    }
    u *= c;
    long double sn = std::sin(u), cn = std::cos(u);
    if (sn != 0.0L) {
        a = cn / sn;
        c *= a;
        for (int i = l; i >= 0; --i) {
            const long double b = em[i];
            a *= c;
            c *= dn;
            dn = (en[i] + a) / (b + a);
            a = c / b;
        }
        a = 1.0L / std::sqrt(c * c + 1.0L);
        sn = (sn >= 0.0L) ? a : -a;
    }
    return sn;
}

struct ZolotarevCoefficients {
    double constant = 0.0;
    std::vector<double> shifts;
    std::vector<double> weights;
    double scalar_error = 0.0;
};

// Zolotarev minimax rational approximation of x^{-1/2} on [a, b] in
// partial-fraction form  r(x) = c0 + sum_j w_j / (x + s_j).  The error decays
// geometrically in the pole count with a rate set by log(b/a), so even a
// bounds overestimate of the condition number by 8x costs only ~1 extra pole.
// The final scaling and the reported error are measured by dense sampling, so
// the returned scalar_error is trustworthy independent of the construction.
inline ZolotarevCoefficients zolotarev_coefficients(double a, double b, int npoles) {
    const long double kappa = static_cast<long double>(b) / a;
    const long double bmax = std::sqrt(kappa);
    const long double modulus = std::sqrt(1.0L - 1.0L / (bmax * bmax));
    const long double complete_k = elliptic_K(modulus);

    std::vector<long double> c(static_cast<std::size_t>(2 * npoles));
    for (int l = 0; l < 2 * npoles; ++l) {
        const long double u = complete_k * (l + 1.0L) / (2.0L * npoles + 1.0L);
        const long double sn = jacobi_sn(u, modulus);
        c[l] = sn * sn / (1.0L - sn * sn);
    }

    long double d0 = 1.0L;
    for (int l = 0; l < npoles; ++l) d0 *= (1.0L + c[2 * l]) / (1.0L + c[2 * l + 1]);

    std::vector<long double> beta(static_cast<std::size_t>(npoles));
    for (int l = 0; l < npoles; ++l) {
        long double value = d0;
        for (int i = 0; i < npoles; ++i) {
            if (i < npoles - 1) value *= c[2 * i + 1] - c[2 * l];
            if (i != l) value /= c[2 * i] - c[2 * l];
        }
        beta[l] = value;
    }

    ZolotarevCoefficients out;
    const long double last = c[2 * npoles - 1];
    long double sum_beta = 0.0L;
    for (long double value : beta) sum_beta += value;
    out.constant = static_cast<double>(sum_beta / std::sqrt(static_cast<long double>(a)));
    out.shifts.reserve(static_cast<std::size_t>(npoles));
    out.weights.reserve(static_cast<std::size_t>(npoles));
    for (int l = 0; l < npoles; ++l) {
        out.shifts.push_back(static_cast<double>(static_cast<long double>(a) * c[2 * l]));
        out.weights.push_back(static_cast<double>(
            std::sqrt(static_cast<long double>(a)) * beta[l] * (last - c[2 * l])));
    }

    // Center the one-sided relative error by a scalar rescaling.
    constexpr int samples = 4096;
    const double log_a = std::log(a), log_b = std::log(b);
    long double lo = std::numeric_limits<long double>::infinity(), hi = 0.0L;
    for (int i = 0; i < samples; ++i) {
        const double q = static_cast<double>(i) / (samples - 1);
        const double x = std::exp(log_a + q * (log_b - log_a));
        long double value = out.constant;
        for (int l = 0; l < npoles; ++l) {
            value += static_cast<long double>(out.weights[l]) / (x + out.shifts[l]);
        }
        const long double ratio = value * std::sqrt(static_cast<long double>(x));
        lo = std::min(lo, ratio);
        hi = std::max(hi, ratio);
    }
    const long double scale = 2.0L / (lo + hi);
    out.constant = static_cast<double>(scale * out.constant);
    for (double& w : out.weights) w = static_cast<double>(scale * w);
    out.scalar_error = static_cast<double>((hi - lo) / (hi + lo));
    return out;
}

inline bool choose_zolotarev(double a, double b, double target, ZolotarevCoefficients& out) {
    if (!(a > 0.0) || !(b >= a)) return false;
    for (int poles = 2; poles <= 48; ++poles) {
        ZolotarevCoefficients c = zolotarev_coefficients(a, b, poles);
        if (c.scalar_error <= target) {
            out = std::move(c);
            return true;
        }
    }
    return false;
}

// ------------------------------------------------ band LAPACK wrappers

template <typename T>
vbcsr_lapack_int he2hb_lower(int n, int kd, T* a, int lda, T* ab, int ldab, T* tau) {
    const char uplo[2] = {'L', '\0'};
    vbcsr_lapack_int n_ = n, kd_ = kd, lda_ = lda, ldab_ = ldab, lwork = -1, info = 0;
    T query = T(0);
    if constexpr (std::is_same<T, double>::value) {
        dsytrd_sy2sb_(uplo, &n_, &kd_, a, &lda_, ab, &ldab_, tau, &query, &lwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, query));
        std::vector<T> work(static_cast<std::size_t>(lwork));
        dsytrd_sy2sb_(uplo, &n_, &kd_, a, &lda_, ab, &ldab_, tau, work.data(), &lwork, &info);
    } else {
        zhetrd_he2hb_(uplo, &n_, &kd_, a, &lda_, ab, &ldab_, tau, &query, &lwork, &info);
        if (info != 0) return info;
        lwork = static_cast<vbcsr_lapack_int>(std::max(1.0, std::real(query)));
        std::vector<T> work(static_cast<std::size_t>(lwork));
        zhetrd_he2hb_(uplo, &n_, &kd_, a, &lda_, ab, &ldab_, tau, work.data(), &lwork, &info);
    }
    return info;
}

template <typename T>
vbcsr_lapack_int pbtrf_lower(int n, int kd, T* ab, int ldab) {
    const char uplo[2] = {'L', '\0'};
    vbcsr_lapack_int n_ = n, kd_ = kd, ldab_ = ldab, info = 0;
    if constexpr (std::is_same<T, double>::value) {
        dpbtrf_(uplo, &n_, &kd_, ab, &ldab_, &info);
    } else {
        zpbtrf_(uplo, &n_, &kd_, ab, &ldab_, &info);
    }
    return info;
}

template <typename T>
vbcsr_lapack_int pbtrs_lower(int n, int kd, int nrhs, const T* ab, int ldab, T* b, int ldb) {
    const char uplo[2] = {'L', '\0'};
    vbcsr_lapack_int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = ldab, ldb_ = ldb, info = 0;
    if constexpr (std::is_same<T, double>::value) {
        dpbtrs_(uplo, &n_, &kd_, &nrhs_, ab, &ldab_, b, &ldb_, &info);
    } else {
        zpbtrs_(uplo, &n_, &kd_, &nrhs_, ab, &ldab_, b, &ldb_, &info);
    }
    return info;
}

// Stage-1 Q (or Q^H) applied to a thin block, reflectors grouped into
// compact-WY form so each group is one LARFT + one GEMM-rich LARFB instead of
// one rank-1 update per reflector (~7x on 13-40 columns).
//
// For UPLO='L', HE2HB stores Q = H_0 H_1 ... H_{n-kd-1}, where H_i acts on
// rows [i + kd, n) and its vector sits in column i of the overwritten dense
// matrix below the band. Consecutive reflectors form the same lower-
// trapezoidal V panel as a QR factorization.
template <typename T>
void apply_band_q(bool adjoint, int n, int kd, int nrhs, const T* reflectors, int lda,
                  const T* tau, T* x, int ldx, int block_size) {
    const int k = n - kd;
    if (k <= 0 || nrhs == 0) return;
    const int bs = std::max(1, std::min(block_size, k));
    const char side[2] = {'L', '\0'};
    const char direct[2] = {'F', '\0'};
    const char storev[2] = {'C', '\0'};
    const char* trans = trans_str(adjoint ? 'C' : 'N', is_complex_v<T>);

    std::vector<T> v, t, work;
    auto apply_block = [&](int first, int nb) {
        const int start = first + kd;
        const int rows = n - start;
        v.assign(static_cast<std::size_t>(rows) * nb, T(0));
        t.assign(static_cast<std::size_t>(nb) * nb, T(0));
        work.assign(static_cast<std::size_t>(nb) * nrhs, T(0));

        for (int j = 0; j < nb; ++j) {
            v[j + static_cast<std::size_t>(j) * rows] = T(1);
            const T* src = reflectors + static_cast<std::size_t>(first + j) * lda + start + j + 1;
            for (int r = j + 1; r < rows; ++r) {
                v[r + static_cast<std::size_t>(j) * rows] = src[r - j - 1];
            }
        }

        vbcsr_lapack_int rows_ = rows, nb_ = nb, nrhs_ = nrhs, ldx_ = ldx;
        if constexpr (std::is_same<T, double>::value) {
            dlarft_(direct, storev, &rows_, &nb_, v.data(), &rows_, tau + first, t.data(), &nb_);
            dlarfb_(side, trans, direct, storev, &rows_, &nrhs_, &nb_, v.data(), &rows_,
                    t.data(), &nb_, x + start, &ldx_, work.data(), &nrhs_);
        } else {
            zlarft_(direct, storev, &rows_, &nb_, v.data(), &rows_, tau + first, t.data(), &nb_);
            zlarfb_(side, trans, direct, storev, &rows_, &nrhs_, &nb_, v.data(), &rows_,
                    t.data(), &nb_, x + start, &ldx_, work.data(), &nrhs_);
        }
    };

    if (adjoint) {
        for (int first = 0; first < k; first += bs) apply_block(first, std::min(bs, k - first));
    } else {
        for (int first = ((k - 1) / bs) * bs; first >= 0; first -= bs) {
            apply_block(first, std::min(bs, k - first));
        }
    }
}

// --------------------------------------------------------------------- routes

// Route 4 (last resort): clamped eigendecomposition, X = V f(w) V^H B.
// Destructive: the eigenvectors overwrite `a`, so no n x n copy is made.
template <typename T>
Status eigen_action(T* a, int lda, int n, const T* b, int ldb, int nrhs, const Options& opt,
                    std::vector<T>& x, Stats& st) {
    std::vector<double> w(static_cast<std::size_t>(n));
    if (heevd<T>(opt.uplo, n, a, lda, w.data()) != 0) return Status::LapackFailure;

    std::vector<T> c = copy_block(b, ldb, n, nrhs);
    std::vector<T> scaled(static_cast<std::size_t>(n) * nrhs, T(0));
    // scaled = f(w) (V^H B)
    gemm<T>('C', 'N', n, nrhs, n, T(1), a, lda, c.data(), n, T(0), scaled.data(), n);
    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < n; ++i) {
            const double f = (w[i] > opt.eigen_clamp) ? std::pow(w[i], -0.5) : 0.0;
            scaled[i + static_cast<std::size_t>(j) * n] *= f;
        }
    }
    x.assign(static_cast<std::size_t>(n) * nrhs, T(0));
    gemm<T>('N', 'N', n, nrhs, n, T(1), a, lda, scaled.data(), n, T(0), x.data(), n);

    st.used_method = Method::Eigen;
    st.status = Status::Success;
    st.lambda_min = w.front();
    st.lambda_max = w.back();
    return Status::Success;
}

// Exact clamped action from an already-reduced tridiagonal: Y = Z f(w) Z^T C
// with T = Z diag(w) Z^T.  Used when A turns out numerically singular, where
// the clamp - not any approximation - defines the answer, and the reduction has
// already destroyed A.  The tridiagonal eigenvectors are real even for a complex
// Hermitian A, so a complex RHS is handled as 2*nrhs real columns.
template <typename T>
bool tridiagonal_clamped_action(std::vector<double>& d, std::vector<double>& e,
                                const std::vector<T>& c, int n, int nrhs, double clamp,
                                std::vector<T>& y) {
    std::vector<double> z(static_cast<std::size_t>(n) * n);
    if (stevd(n, d.data(), e.data(), z.data(), n) != 0) return false;

    const int rc = is_complex_v<T> ? 2 * nrhs : nrhs;
    std::vector<double> rhs(static_cast<std::size_t>(n) * rc, 0.0);
    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < n; ++i) {
            const T v = c[i + static_cast<std::size_t>(j) * n];
            rhs[i + static_cast<std::size_t>(j) * n] = std::real(v);
            if constexpr (is_complex_v<T>) {
                rhs[i + static_cast<std::size_t>(j + nrhs) * n] = std::imag(v);
            }
        }
    }
    std::vector<double> work(static_cast<std::size_t>(n) * rc, 0.0);
    gemm<double>('T', 'N', n, rc, n, 1.0, z.data(), n, rhs.data(), n, 0.0, work.data(), n);
    for (int j = 0; j < rc; ++j) {
        for (int i = 0; i < n; ++i) {
            const double f = (d[i] > clamp) ? std::pow(d[i], -0.5) : 0.0;
            work[i + static_cast<std::size_t>(j) * n] *= f;
        }
    }
    gemm<double>('N', 'N', n, rc, n, 1.0, z.data(), n, work.data(), n, 0.0, rhs.data(), n);

    y.assign(static_cast<std::size_t>(n) * nrhs, T(0));
    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < n; ++i) {
            const double re = rhs[i + static_cast<std::size_t>(j) * n];
            if constexpr (is_complex_v<T>) {
                y[i + static_cast<std::size_t>(j) * n] =
                    T(re, rhs[i + static_cast<std::size_t>(j + nrhs) * n]);
            } else {
                y[i + static_cast<std::size_t>(j) * n] = re;
            }
        }
    }
    return true;
}

// Route 2: dense-to-band reduction (stage one of the two-stage
// tridiagonalization, GEMM-rich) + Zolotarev rational action on the band.
//
//   A = Q B Q^H,   A^{-1/2} E = Q [ c0 C + sum_j w_j (B + s_j I)^{-1} C ],
//   C = Q^H E.
//
// This replaces the one-stage xSYTRD (memory-bound level-2, measured 5-9 GF/s
// against 30+ GF/s for blocked kernels) with SY2SB/HE2HB, and never performs
// the band-to-tridiagonal bulge chase: each Zolotarev pole is a banded
// Cholesky solve at O(n kd^2) + O(n kd nrhs), negligible next to stage one.
// (LAPACK provides no thin-Q application for the second stage anyway, which is
// what previously made the full tridiagonal route the only option.)
//
// Spectral bounds for the pole placement are computed on the band itself:
// lambda_max from Gershgorin, lambda_min by exponential search with banded
// Cholesky trials of B - sigma I. Both are conservative; Zolotarev's degree
// grows only logarithmically with the bounds ratio, so a factor-8 bracket
// costs ~1 extra pole.
//
// Destructive, and self-contained after the reduction: once HE2HB has
// overwritten `a` with its reflectors no external fallback can rerun on `a`,
// so st.used_method is set to BandZolotarev at that point (the dispatcher only
// reroutes while it is unset) and the numerically-singular case is handled
// here, by the clamped eigendecomposition of the band expanded back to dense
// - same semantics as the eigen route, wrapped in the stage-one Q.
template <typename T>
Status band_zolotarev_action(T* a, int lda, int n, const T* b, int ldb, int nrhs,
                             const Options& opt, std::vector<T>& x, Stats& st) {
    // Default half-bandwidth: measured flat within ~10% across [64, 256] on
    // dim-1500..3800 problems, with the optimum at the small end (the shifted
    // band factorizations scale as kd^2 while SY2SB gains little past ~n/40).
    // Autotune per machine via Options::band_width if this route dominates.
    int kd = opt.band_width;
    if (kd <= 0) {
        kd = std::max(16, ((std::max(64, std::min(n / 40, 192)) + 8) / 16) * 16);
    }
    kd = std::max(1, std::min(kd, n - 1));
    st.band_width = kd;

    // HE2HB is used here in its lower layout; mirror if the caller's valid
    // triangle is the upper one. Reads 'U', writes strictly below the
    // diagonal, so the 'U' triangle stays intact until HE2HB itself runs.
    if (opt.uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = j + 1; i < n; ++i) {
                a[i + static_cast<std::size_t>(j) * lda] =
                    conj_if(a[j + static_cast<std::size_t>(i) * lda]);
            }
        }
    }

    const int ldab = kd + 1;
    std::vector<T> band(static_cast<std::size_t>(ldab) * n, T(0));
    std::vector<T> tau(static_cast<std::size_t>(std::max(1, n - kd)), T(0));
    const vbcsr_lapack_int he2hb_info =
        he2hb_lower<T>(n, kd, a, lda, band.data(), ldab, tau.data());
    if (he2hb_info != 0) {
        // HE2HB reports only argument errors, raised before any data is
        // touched: `a`'s opt.uplo triangle is still valid and the dispatcher
        // may reroute.
        return Status::LapackFailure;
    }
    // From here on `a` holds reflectors, not A: this route owns the problem.
    st.used_method = Method::BandZolotarev;

    // C = Q^H E (needed by both the rational and the clamped path).
    std::vector<T> c = copy_block(b, ldb, n, nrhs);
    apply_band_q<T>(true, n, kd, nrhs, a, lda, tau.data(), c.data(), n,
                    opt.reflector_block_size);

    // lambda_max upper bound: Gershgorin row sums off the band storage.
    std::vector<double> row_sum(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r <= std::min(kd, n - 1 - j); ++r) {
            const double v = std::abs(band[r + static_cast<std::size_t>(j) * ldab]);
            row_sum[j + r] += v;
            if (r > 0) row_sum[j] += v;
        }
    }
    double lambda_hi = 0.0;
    for (double v : row_sum) lambda_hi = std::max(lambda_hi, v);

    // lambda_min lower bound: largest sigma on a x8 grid with B - sigma I
    // still positive definite. Each trial is one banded Cholesky, O(n kd^2).
    double lambda_lo = 0.0;
    std::vector<T> shifted(band.size());
    if (lambda_hi > 0.0) {
        for (double sigma = lambda_hi * 1e-12; sigma < lambda_hi; sigma *= 8.0) {
            std::copy(band.begin(), band.end(), shifted.begin());
            for (int j = 0; j < n; ++j) {
                shifted[static_cast<std::size_t>(j) * ldab] -= T(sigma);
            }
            if (pbtrf_lower<T>(n, kd, shifted.data(), ldab) != 0) break;
            lambda_lo = sigma;
        }
    }
    st.lambda_min = lambda_lo;  // conservative bracket, not the eigenvalue
    st.lambda_max = lambda_hi;

    ZolotarevCoefficients coeff;
    const double target = (opt.rational_scalar_tolerance > 0.0)
                              ? opt.rational_scalar_tolerance
                              : std::max(1.0e-14, 0.01 * opt.relative_tolerance);
    if (lambda_lo > opt.lambda_min_floor &&
        choose_zolotarev(lambda_lo, lambda_hi, target, coeff)) {
        st.rational_poles = static_cast<int>(coeff.shifts.size());
        st.estimated_relative_error = coeff.scalar_error;

        x.assign(c.begin(), c.end());
        for (T& v : x) v *= T(coeff.constant);

        std::vector<T> z(c.size());
        for (std::size_t pole = 0; pole < coeff.shifts.size(); ++pole) {
            std::copy(band.begin(), band.end(), shifted.begin());
            for (int j = 0; j < n; ++j) {
                shifted[static_cast<std::size_t>(j) * ldab] += T(coeff.shifts[pole]);
            }
            if (pbtrf_lower<T>(n, kd, shifted.data(), ldab) != 0) return Status::LapackFailure;
            std::copy(c.begin(), c.end(), z.begin());
            if (pbtrs_lower<T>(n, kd, nrhs, shifted.data(), ldab, z.data(), n) != 0) {
                return Status::LapackFailure;
            }
            const T w = T(coeff.weights[pole]);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] += w * z[i];
        }
    } else {
        // Numerically singular (or bounds unusable): the clamp defines the
        // answer. Expand the band back to a dense lower triangle and run the
        // clamped eigendecomposition on it - only this rare path pays the
        // extra n^2 buffer.
        std::vector<T> dense(static_cast<std::size_t>(n) * n, T(0));
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r <= std::min(kd, n - 1 - j); ++r) {
                dense[(j + r) + static_cast<std::size_t>(j) * n] =
                    band[r + static_cast<std::size_t>(j) * ldab];
            }
        }
        Options clamp_opt = opt;
        clamp_opt.uplo = 'L';
        Stats clamp_stats;
        const Status es = eigen_action<T>(dense.data(), n, n, c.data(), n, nrhs, clamp_opt,
                                          x, clamp_stats);
        if (es != Status::Success) return es;
        st.lambda_min = clamp_stats.lambda_min;
        st.lambda_max = clamp_stats.lambda_max;
        st.used_method = Method::Eigen;  // honest tally: the clamp defined the answer
    }

    // X = Q Y
    apply_band_q<T>(false, n, kd, nrhs, a, lda, tau.data(), x.data(), n,
                    opt.reflector_block_size);

    st.status = Status::Success;
    return Status::Success;
}

// Route 1: Cholesky + inverse block Lanczos with a nested-difference
// convergence certificate.
template <typename T>
Status inverse_block_lanczos(const T* a, int lda, int n, const T* b, int ldb, int nrhs,
                             const Options& opt, std::vector<T>& x, Stats& st) {
    st.fast_path_attempted = true;

    // potrf reads one triangle only, so no Hermitian mirror is needed here.
    std::vector<T> factor = copy_triangle(a, lda, n, opt.uplo);
    const vbcsr_lapack_int chol = potrf<T>(opt.uplo, n, factor.data(), n);
    if (chol > 0) return Status::NotPositiveDefinite;
    if (chol != 0) return Status::LapackFailure;

    // O(n^2) condition screen: a numerically singular A is sent to the clamped
    // eigen route, where the clamp is what defines the answer.
    {
        const double anorm = hermitian_one_norm(a, lda, n, opt.uplo);
        double rcond = 0.0;
        if (pocon<T>(opt.uplo, n, factor.data(), n, anorm, &rcond) == 0) {
            st.rcond_estimate = rcond;
            if (!(rcond * anorm > 0.01 * opt.lambda_min_floor)) return Status::NearSingular;
        }
    }

    const double internal_tol = (opt.fast_internal_tolerance > 0.0)
                                    ? opt.fast_internal_tolerance
                                    : 0.25 * opt.relative_tolerance;
    const int divisor = std::max(1, opt.krylov_space_divisor);
    const int max_blocks =
        std::max(1, std::min(opt.max_krylov_blocks, n / (divisor * nrhs)));
    const int qcap = max_blocks * nrhs;

    std::vector<T> q_basis(static_cast<std::size_t>(n) * qcap, T(0));
    for (int j = 0; j < nrhs; ++j) {
        std::copy(b + static_cast<std::size_t>(j) * ldb,
                  b + static_cast<std::size_t>(j) * ldb + n,
                  q_basis.begin() + static_cast<std::size_t>(j) * n);
    }
    std::vector<T> r0;
    if (!opt.rhs_is_orthonormal) {
        if (thin_qr<T>(n, nrhs, q_basis.data(), n, r0) != 0) return Status::LapackFailure;
        if (qr_rank_ratio(r0, nrhs) < opt.rank_tolerance) return Status::RankBreakdown;
    }

    std::vector<T> h_store(static_cast<std::size_t>(qcap) * qcap, T(0));
    std::vector<T> m_q(static_cast<std::size_t>(n) * nrhs);
    std::vector<T> candidate(static_cast<std::size_t>(n) * nrhs);
    std::vector<T> hcol(static_cast<std::size_t>(qcap) * nrhs);
    std::vector<T> coeff_buf(static_cast<std::size_t>(qcap) * nrhs);
    std::vector<T> projected;
    std::vector<T> current, previous;

    double previous_difference = std::numeric_limits<double>::infinity();
    int checkpoints = 0;
    Status status = Status::FastPathRejected;

    // X_k = Q_k (H_k^{1/2} E_1) R_0
    auto form_approximation = [&](int q_current, std::vector<T>& out) -> bool {
        std::vector<T> h = compact_leading(h_store, qcap, q_current);
        if (!projected_sqrt_first_columns<T>(h, q_current, nrhs, projected)) return false;
        out.assign(static_cast<std::size_t>(n) * nrhs, T(0));
        gemm<T>('N', 'N', n, nrhs, q_current, T(1), q_basis.data(), n, projected.data(),
                q_current, T(0), out.data(), n);
        if (!r0.empty()) {
            std::vector<T> tmp(out.size(), T(0));
            gemm<T>('N', 'N', n, nrhs, nrhs, T(1), out.data(), n, r0.data(), nrhs, T(0),
                    tmp.data(), n);
            out.swap(tmp);
        }
        return true;
    };

    for (int block = 0; block < max_blocks; ++block) {
        const int block_start = block * nrhs;
        const int q_current = block_start + nrhs;

        // m_q = A^{-1} Q_block, reusing the Cholesky factor.
        std::copy(q_basis.begin() + static_cast<std::size_t>(block_start) * n,
                  q_basis.begin() + static_cast<std::size_t>(q_current) * n,
                  m_q.begin());
        if (potrs<T>(opt.uplo, n, nrhs, factor.data(), n, m_q.data(), n) != 0) {
            return Status::LapackFailure;
        }

        // New block column of H = Q^H A^{-1} Q.  Hermitian symmetry fills the
        // matching block row, so previous A^{-1}Q blocks need not be stored.
        gemm<T>('C', 'N', q_current, nrhs, n, T(1), q_basis.data(), n, m_q.data(), n, T(0),
                hcol.data(), q_current);
        for (int j = 0; j < nrhs; ++j) {
            const int col = block_start + j;
            for (int i = 0; i < block_start; ++i) {
                const T v = hcol[i + static_cast<std::size_t>(j) * q_current];
                h_store[i + static_cast<std::size_t>(col) * qcap] = v;
                h_store[col + static_cast<std::size_t>(i) * qcap] = conj_if(v);
            }
            for (int i = 0; i < nrhs; ++i) {
                const T vij = hcol[block_start + i + static_cast<std::size_t>(j) * q_current];
                const T vji = hcol[block_start + j + static_cast<std::size_t>(i) * q_current];
                h_store[block_start + i + static_cast<std::size_t>(col) * qcap] =
                    T(0.5) * (vij + conj_if(vji));
            }
        }

        st.krylov_blocks = block + 1;
        st.krylov_dimension = q_current;

        if ((block + 1) % std::max(1, opt.checkpoint_stride) == 0 || block + 1 == max_blocks) {
            if (!form_approximation(q_current, current)) return Status::FastPathRejected;
            ++checkpoints;
            if (!previous.empty()) {
                const double diff = relative_difference(current, previous);
                st.last_nested_difference = diff;
                if (std::isfinite(previous_difference)) {
                    double rho = diff / std::max(previous_difference, 1.0e-300);
                    rho = std::max(0.0, std::min(rho, 0.999999));
                    st.convergence_ratio = rho;
                    const double tail = (rho < 0.95)
                                            ? diff * rho / std::max(1.0e-15, 1.0 - rho)
                                            : std::numeric_limits<double>::infinity();
                    st.estimated_relative_error = tail;

                    if (checkpoints >= opt.minimum_checkpoints && rho < opt.accept_ratio &&
                        tail < internal_tol) {
                        status = Status::Success;
                        x = std::move(current);
                        st.fast_path_accepted = true;
                        break;
                    }
                    if (checkpoints >= opt.minimum_checkpoints) {
                        const int remaining =
                            (max_blocks - (block + 1)) / std::max(1, opt.checkpoint_stride);
                        double needed = std::numeric_limits<double>::infinity();
                        if (rho > 0.0 && rho < 1.0 && diff > internal_tol) {
                            needed = std::log(internal_tol / diff) / std::log(rho);
                        }
                        if (rho > opt.reject_ratio || needed > remaining + 1.0) {
                            return Status::FastPathRejected;
                        }
                    }
                }
                previous_difference = diff;
            }
            previous = std::move(current);
        }

        if (block + 1 == max_blocks) break;

        // Next basis block: local block-Lanczos subtraction (current and
        // previous block) followed by one full reorthogonalization pass.  A
        // pure three-block recurrence loses orthogonality catastrophically on
        // these ill-conditioned matrices; a second full pass costs more
        // without measurable gain.
        candidate = m_q;
        const int first = std::max(0, block - 1);
        const int local_cols = (block - first + 1) * nrhs;
        gemm<T>('C', 'N', local_cols, nrhs, n, T(1),
                q_basis.data() + static_cast<std::size_t>(first * nrhs) * n, n, candidate.data(),
                n, T(0), coeff_buf.data(), local_cols);
        gemm<T>('N', 'N', n, nrhs, local_cols, T(-1),
                q_basis.data() + static_cast<std::size_t>(first * nrhs) * n, n, coeff_buf.data(),
                local_cols, T(1), candidate.data(), n);

        gemm<T>('C', 'N', q_current, nrhs, n, T(1), q_basis.data(), n, candidate.data(), n, T(0),
                coeff_buf.data(), q_current);
        gemm<T>('N', 'N', n, nrhs, q_current, T(-1), q_basis.data(), n, coeff_buf.data(),
                q_current, T(1), candidate.data(), n);

        std::vector<T> r;
        if (thin_qr<T>(n, nrhs, candidate.data(), n, r) != 0) return Status::LapackFailure;
        if (qr_rank_ratio(r, nrhs) < opt.rank_tolerance) return Status::RankBreakdown;
        std::copy(candidate.begin(), candidate.end(),
                  q_basis.begin() + static_cast<std::size_t>(q_current) * n);
    }

    if (status == Status::Success) {
        st.used_method = Method::InverseBlockLanczos;
        st.status = Status::Success;
    }
    return status;
}

}  // namespace detail

inline const char* method_name(Method m) {
    switch (m) {
        case Method::Auto: return "auto";
        case Method::InverseBlockLanczos: return "inverse_block_lanczos";
        case Method::BandZolotarev: return "band_zolotarev";
        case Method::Eigen: return "eigen";
    }
    return "unknown";
}

inline const char* status_name(Status s) {
    switch (s) {
        case Status::Success: return "success";
        case Status::InvalidArgument: return "invalid_argument";
        case Status::FastPathRejected: return "fast_path_rejected";
        case Status::RankBreakdown: return "rank_breakdown";
        case Status::NotPositiveDefinite: return "not_positive_definite";
        case Status::NearSingular: return "near_singular";
        case Status::LapackFailure: return "lapack_failure";
    }
    return "unknown";
}

// ---------------------------------------------------------------- entry points

// Core dispatcher.  `a` is the working buffer and is DESTROYED: the routes
// factorize in place, so a problem needs no n x n copy beyond the matrix the
// caller already holds.  Never throws (it runs inside the OpenMP graph loop).
template <typename T>
Status inverse_sqrt_action_destructive(T* a, int lda, int n, const T* b, int ldb, int nrhs,
                                      T* x, int ldx, const Options& opt = Options{},
                                      Router* router = nullptr, Stats* out_stats = nullptr) {
    Stats st;
    if (!a || !b || !x || n <= 0 || nrhs <= 0 || nrhs > n || lda < n || ldb < n || ldx < n ||
        !(opt.relative_tolerance > 0.0) || opt.max_krylov_blocks < 1 ||
        opt.checkpoint_stride < 1 || (opt.uplo != 'U' && opt.uplo != 'L')) {
        st.status = Status::InvalidArgument;
        if (out_stats) *out_stats = st;
        return Status::InvalidArgument;
    }

    std::vector<T> result;
    Status status = Status::Success;

    try {
        const bool tiny = n <= std::max(opt.direct_eigen_max_n, 4 * nrhs);
        const bool thick = n < opt.min_n_over_nrhs * nrhs;

        // Applicability guards are honoured for a forced fast path as well: a
        // subgraph of a few hundred rows cannot support a block Krylov space at
        // all, so "use the Krylov route" can only mean "wherever it applies".
        // What a forced route does disable is the convergence fallback below.
        Method route = opt.method;
        if (route == Method::Auto || route == Method::InverseBlockLanczos) {
            if (tiny) route = Method::Eigen;
            else if (thick) route = Method::BandZolotarev;
            else if (route == Method::Auto && router && !router->try_fast_path()) {
                route = Method::BandZolotarev;
            } else {
                route = Method::InverseBlockLanczos;
            }
        }

        if (route == Method::InverseBlockLanczos) {
            // The only route that needs its own n x n buffer: it factorizes a
            // copy so that `a` survives for the fallback below.
            status = detail::inverse_block_lanczos<T>(a, lda, n, b, ldb, nrhs, opt, result, st);
            if (router) router->record(status == Status::Success);
            if (status != Status::Success && opt.method != Method::InverseBlockLanczos) {
                route = Method::BandZolotarev;  // also covers the singular cases
                status = Status::Success;
            }
        }

        if (status == Status::Success && route == Method::BandZolotarev && result.empty()) {
            status = detail::band_zolotarev_action<T>(a, lda, n, b, ldb, nrhs, opt, result, st);
            // The route claims used_method the moment HE2HB has overwritten
            // `a`; before that point `a` is still valid and the clamped
            // eigendecomposition can take over. After it, the route resolves
            // everything internally, including the near-singular case.
            if (status != Status::Success && opt.method != Method::BandZolotarev &&
                st.used_method != Method::BandZolotarev) {
                route = Method::Eigen;
                status = Status::Success;
                result.clear();
            }
        }

        if (status == Status::Success && result.empty() && route != Method::BandZolotarev) {
            // Destructive and always applicable: the clamp is what defines the
            // answer when A turns out numerically singular.
            status = detail::eigen_action<T>(a, lda, n, b, ldb, nrhs, opt, result, st);
        }
    } catch (...) {
        status = Status::LapackFailure;
        result.clear();
    }

    if (status == Status::Success && !result.empty()) {
        detail::store_block(result, n, nrhs, x, ldx);
    } else {
        for (int j = 0; j < nrhs; ++j) {
            std::fill(x + static_cast<std::size_t>(j) * ldx,
                      x + static_cast<std::size_t>(j) * ldx + n, T(0));
        }
        if (status == Status::Success) status = Status::LapackFailure;
    }

    st.status = status;
    if (out_stats) *out_stats = st;
    return status;
}

// X (n x nrhs) = A^{-1/2} B, A preserved.  Column-major; only the Options::uplo
// triangle of A is referenced.  Costs one n x n copy over the destructive form.
template <typename T>
Status inverse_sqrt_action(const T* a, int lda, int n, const T* b, int ldb, int nrhs, T* x,
                           int ldx, const Options& opt = Options{}, Router* router = nullptr,
                           Stats* out_stats = nullptr) {
    if (!a || n <= 0) {
        if (out_stats) out_stats->status = Status::InvalidArgument;
        return Status::InvalidArgument;
    }
    try {
        std::vector<T> work = detail::copy_triangle(a, lda, n, opt.uplo);
        return inverse_sqrt_action_destructive<T>(work.data(), n, n, b, ldb, nrhs, x, ldx, opt,
                                                 router, out_stats);
    } catch (...) {
        if (out_stats) out_stats->status = Status::LapackFailure;
        return Status::LapackFailure;
    }
}

// X = A^{-1/2} E with E = [e_offset, ..., e_(offset+nrhs-1)]; the block row of
// A^{-1/2} at that offset is X^H.
template <typename T>
Status inverse_sqrt_selector(const T* a, int lda, int n, int offset, int nrhs, T* x, int ldx,
                             const Options& opt = Options{}, Router* router = nullptr,
                             Stats* stats = nullptr) {
    if (offset < 0 || nrhs <= 0 || offset + nrhs > n) {
        if (stats) stats->status = Status::InvalidArgument;
        return Status::InvalidArgument;
    }
    try {
        std::vector<T> selector(static_cast<std::size_t>(n) * nrhs, T(0));
        for (int j = 0; j < nrhs; ++j) {
            selector[offset + j + static_cast<std::size_t>(j) * n] = T(1);
        }
        Options o = opt;
        o.rhs_is_orthonormal = true;
        return inverse_sqrt_action<T>(a, lda, n, selector.data(), n, nrhs, x, ldx, o, router,
                                      stats);
    } catch (...) {  // no exception may reach the OpenMP graph loop
        if (stats) stats->status = Status::LapackFailure;
        return Status::LapackFailure;
    }
}

// The form the graph loop uses, and the frugal one: on entry `m` is the n x n
// Hermitian matrix (column-major, Options::uplo triangle), on exit it is the
// n x k_cols column-major block A^{-1/2}[:, col_start : col_start + k].
//
// `m`'s storage is the working buffer, so the input is consumed and the whole
// solve needs no second n x n allocation.
template <typename T>
Status inverse_sqrt_columns_inplace(int n, std::vector<T>& m, int k_cols, int col_start,
                                    const Options& opt = Options{}, Router* router = nullptr,
                                    Stats* stats = nullptr) {
    if (n <= 0) return Status::Success;
    const int k = (k_cols <= 0) ? n : k_cols;
    if (m.size() < static_cast<std::size_t>(n) * n || col_start < 0 || col_start + k > n) {
        if (stats) stats->status = Status::InvalidArgument;
        return Status::InvalidArgument;
    }
    try {
        // Real-arithmetic fast path.
        //
        // At a real k-point the Bloch phase is exactly 1, so the subgraph block
        // carries identically-zero imaginary parts. Gamma is the common case
        // for the large supercells this solver targets, and solving such a
        // block in complex arithmetic costs 2x the bytes and 4x the flops of
        // the equivalent real problem -- in the routine that is the O(n^3) hot
        // spot of the whole seed. Fewer bytes per problem is also what keeps a
        // full thread-count batch of concurrent problems off the memory system.
        //
        // The test is exact (== 0.0) rather than a tolerance: at a real k-point
        // the zeros come from multiplying by 1.0 + 0.0i and are bit-exact, and
        // anything genuinely complex must keep the complex path. It costs
        // O(n^2) to decide, against the O(n^3) it saves.
        //
        // The packing is in place. A complex element is two doubles, so the
        // real part of element i is written at offset i while element i is read
        // at offset 2i: the write cursor never overtakes the read cursor and no
        // second n x n buffer is needed -- which matters, because allocating
        // one would raise the very peak this path exists to lower.
        if constexpr (std::is_same<T, std::complex<double>>::value) {
            const std::size_t nn = static_cast<std::size_t>(n) * n;
            bool is_real = true;
            for (std::size_t i = 0; i < nn; ++i) {
                if (m[i].imag() != 0.0) {
                    is_real = false;
                    break;
                }
            }
            if (is_real) {
                double* real_view = reinterpret_cast<double*>(m.data());
                for (std::size_t i = 0; i < nn; ++i) real_view[i] = m[i].real();

                std::vector<double> selector(static_cast<std::size_t>(n) * k, 0.0);
                for (int j = 0; j < k; ++j) {
                    selector[col_start + j + static_cast<std::size_t>(j) * n] = 1.0;
                }
                std::vector<double> xr(static_cast<std::size_t>(n) * k, 0.0);
                Options o = opt;
                o.rhs_is_orthonormal = true;
                const Status s = inverse_sqrt_action_destructive<double>(
                    real_view, n, n, selector.data(), n, k, xr.data(), n, o, router, stats);
                std::vector<T> xc(xr.size());
                for (std::size_t i = 0; i < xc.size(); ++i) xc[i] = T(xr[i], 0.0);
                m = std::move(xc);
                return s;
            }
        }

        std::vector<T> selector(static_cast<std::size_t>(n) * k, T(0));
        for (int j = 0; j < k; ++j) {
            selector[col_start + j + static_cast<std::size_t>(j) * n] = T(1);
        }
        std::vector<T> x(static_cast<std::size_t>(n) * k, T(0));
        Options o = opt;
        o.rhs_is_orthonormal = true;
        const Status s = inverse_sqrt_action_destructive<T>(m.data(), n, n, selector.data(), n, k,
                                                           x.data(), n, o, router, stats);
        m = std::move(x);
        return s;
    } catch (...) {  // no exception may reach the OpenMP graph loop
        try {
            m.assign(static_cast<std::size_t>(n) * k, T(0));
        } catch (...) {
        }
        if (stats) stats->status = Status::LapackFailure;
        return Status::LapackFailure;
    }
}

}  // namespace isqrt
}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_ISQRT_ACTION_HPP
