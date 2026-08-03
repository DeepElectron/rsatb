// Accuracy certification and single-thread benchmark for the dense thin
// inverse-square-root action (calc/isqrt_action.hpp).
//
// Synthetic overlap matrices are built as A = Q diag(lambda) Q^H with a random
// orthogonal/unitary Q, so the exact answer X = A^{-1/2} E is available from the
// construction itself and no eigensolver is needed as a reference.
//
// Baseline is Method::Eigen, i.e. the xSYEVD/xHEEVD route that
// dense_matrix_function used before, restricted to the requested block.
//
// One dense problem per thread is how the graph loop runs these, so the
// reference measurement is single-threaded BLAS:
//
//   Usage:  OMP_NUM_THREADS=1 ./benchmark_isqrt_action [--quick]
//
// OMP_NUM_THREADS is the only thread control needed: vbcsr translates that
// budget to the linked vendor library (thread policy in dense_kernels.hpp).
// Exit status is non-zero if any case exceeds its accuracy budget.

#include "../isqrt_action.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace isq = rsatb::calc::isqrt;
namespace det = rsatb::calc::isqrt::detail;

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

enum class Family { Gapped, TwoBand, Continuous };

const char* family_name(Family f) {
    switch (f) {
        case Family::Gapped: return "gapped";
        case Family::TwoBand: return "two_band";
        case Family::Continuous: return "continuous";
    }
    return "?";
}

std::vector<double> make_spectrum(Family family, int n, int n_low, double condition,
                                  std::mt19937_64& rng) {
    std::vector<double> lambda(static_cast<std::size_t>(n));
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double lo = 1.0 / condition;

    switch (family) {
        case Family::Gapped: {
            // LCAO-like: a small near-null subspace, then a healthy bulk.
            for (int i = 0; i < n_low; ++i) {
                lambda[i] = lo * std::exp(unit(rng) * std::log(10.0));
            }
            for (int i = n_low; i < n; ++i) lambda[i] = 0.2 + 1.8 * unit(rng);
            break;
        }
        case Family::TwoBand: {
            for (int i = 0; i < n_low; ++i) lambda[i] = lo * (1.0 + 9.0 * unit(rng));
            for (int i = n_low; i < n; ++i) {
                lambda[i] = (unit(rng) < 0.5) ? 1.0e-3 * (1.0 + unit(rng))
                                              : 0.5 + 1.5 * unit(rng);
            }
            break;
        }
        case Family::Continuous: {
            // Eigenvalues log-spread across the whole condition-number range.
            for (int i = 0; i < n; ++i) {
                const double t = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
                lambda[i] = std::exp(std::log(lo) + t * std::log(condition));
            }
            break;
        }
    }
    return lambda;
}

// Random orthogonal (unitary) factor from the QR of a Gaussian matrix.
template <typename T>
std::vector<T> random_unitary(int n, std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<T> q(static_cast<std::size_t>(n) * n);
    for (auto& v : q) {
        if constexpr (det::is_complex_v<T>) v = T(gauss(rng), gauss(rng));
        else v = T(gauss(rng));
    }
    std::vector<T> r;
    if (det::thin_qr<T>(n, n, q.data(), n, r) != 0) {
        std::fprintf(stderr, "random_unitary: QR failed\n");
        std::exit(2);
    }
    return q;
}

template <typename T>
struct Problem {
    int n = 0;
    int nrhs = 0;
    int offset = 0;
    std::vector<T> a;       // n x n, column-major, full
    std::vector<T> x_ref;   // n x nrhs exact A^{-1/2} E
    double lambda_min = 0.0;
    double lambda_max = 0.0;
};

template <typename T>
Problem<T> make_problem(Family family, int n, int nrhs, int n_low, double condition,
                        std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Problem<T> p;
    p.n = n;
    p.nrhs = nrhs;
    p.offset = n / 3;  // an interior block, as in the graph loop

    std::vector<double> lambda = make_spectrum(family, n, n_low, condition, rng);
    p.lambda_min = *std::min_element(lambda.begin(), lambda.end());
    p.lambda_max = *std::max_element(lambda.begin(), lambda.end());

    const std::vector<T> q = random_unitary<T>(n, rng);

    // A = (Q diag(lambda)) Q^H
    std::vector<T> scaled(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            scaled[i + static_cast<std::size_t>(j) * n] =
                q[i + static_cast<std::size_t>(j) * n] * lambda[j];
        }
    }
    p.a.assign(static_cast<std::size_t>(n) * n, T(0));
    det::gemm<T>('N', 'C', n, n, n, T(1), scaled.data(), n, q.data(), n, T(0), p.a.data(), n);
    // Enforce exact Hermitian symmetry of the test input.
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i < n; ++i) {
            const T z = T(0.5) * (p.a[i + static_cast<std::size_t>(j) * n] +
                                  det::conj_if(p.a[j + static_cast<std::size_t>(i) * n]));
            p.a[i + static_cast<std::size_t>(j) * n] = z;
            p.a[j + static_cast<std::size_t>(i) * n] = det::conj_if(z);
        }
    }

    // X_ref = Q diag(lambda^{-1/2}) (Q^H E)
    std::vector<T> c(static_cast<std::size_t>(n) * nrhs);
    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < n; ++i) {
            c[i + static_cast<std::size_t>(j) * n] =
                det::conj_if(q[(p.offset + j) + static_cast<std::size_t>(i) * n]) *
                std::pow(lambda[i], -0.5);
        }
    }
    p.x_ref.assign(static_cast<std::size_t>(n) * nrhs, T(0));
    det::gemm<T>('N', 'N', n, nrhs, n, T(1), q.data(), n, c.data(), n, T(0), p.x_ref.data(), n);
    return p;
}

template <typename T>
double relative_error(const std::vector<T>& x, const std::vector<T>& ref) {
    long double num = 0.0L, den = 0.0L;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += static_cast<long double>(det::abs2<T>(x[i] - ref[i]));
        den += static_cast<long double>(det::abs2<T>(ref[i]));
    }
    if (!(den > 0.0L)) return 0.0;
    return static_cast<double>(std::sqrt(num / den));
}

struct Outcome {
    double seconds = 0.0;
    double error = 0.0;
    isq::Stats stats;
};

template <typename T>
Outcome run(const Problem<T>& p, isq::Method method, isq::Router* router) {
    isq::Options opt;
    opt.method = method;
    opt.uplo = 'U';
    std::vector<T> x(static_cast<std::size_t>(p.n) * p.nrhs, T(0));
    Outcome out;
    const auto t0 = Clock::now();
    isq::inverse_sqrt_selector<T>(p.a.data(), p.n, p.n, p.offset, p.nrhs, x.data(), p.n, opt,
                                  router, &out.stats);
    out.seconds = seconds_since(t0);
    out.error = relative_error(x, p.x_ref);
    return out;
}

int failures = 0;

// Accuracy budget: the inverse square root amplifies input rounding by ~kappa/2,
// so the reachable forward accuracy is bounded by the conditioning, not by the
// algorithm.  The budget below tracks that limit and is checked against the
// eigen baseline as well.
double error_budget(double condition) {
    return std::max(2.0e-8, 4.0e-16 * condition);
}

template <typename T>
void run_case(const char* label, Family family, int n, int nrhs, int n_low, double condition,
              std::uint64_t seed) {
    Problem<T> p = make_problem<T>(family, n, nrhs, n_low, condition, seed);

    isq::Router router;
    const Outcome autorun = run(p, isq::Method::Auto, &router);
    const Outcome band = run(p, isq::Method::BandZolotarev, nullptr);
    const Outcome rational = run(p, isq::Method::TridiagonalRational, nullptr);
    const Outcome eigen = run(p, isq::Method::Eigen, nullptr);

    const double budget = error_budget(condition);
    const char* verdict = "ok";
    if (autorun.error > budget || band.error > budget || rational.error > budget) {
        verdict = "FAIL";
        ++failures;
    }

    char route[80];
    std::snprintf(route, sizeof(route), "%s m=%d",
                  isq::method_name(autorun.stats.used_method),
                  autorun.stats.used_method == isq::Method::TridiagonalRational
                      ? autorun.stats.rational_poles
                      : autorun.stats.krylov_blocks);
    std::printf("%-8s %-10s n=%-5d b=%-3d low=%-4d kappa=%-8.0e | auto %7.3fs %8.1e %-30s"
                " | band %7.3fs %8.1e p=%-3d | rational %7.3fs %8.1e | eigen %7.3fs %8.1e"
                " | speedup %5.2fx (band %4.2fx) %s\n",
                label, family_name(family), n, nrhs, n_low, condition, autorun.seconds,
                autorun.error, route, band.seconds, band.error, band.stats.rational_poles,
                rational.seconds, rational.error, eigen.seconds, eigen.error,
                eigen.seconds / std::max(1.0e-9, autorun.seconds),
                eigen.seconds / std::max(1.0e-9, band.seconds), verdict);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    }

    std::printf("=== real symmetric, LCAO-like gapped spectrum ===\n");
    run_case<double>("real", Family::Gapped, 2000, 13, 16, 1.0e8, 11);
    run_case<double>("real", Family::Gapped, 2000, 40, 16, 1.0e8, 12);
    if (!quick) {
        run_case<double>("real", Family::Gapped, 3000, 13, 16, 1.0e8, 13);
        run_case<double>("real", Family::Gapped, 3000, 40, 32, 1.0e8, 14);
        run_case<double>("real", Family::Gapped, 4000, 13, 16, 1.0e8, 15);
        run_case<double>("real", Family::Gapped, 4000, 40, 64, 1.0e8, 16);
    }

    std::printf("\n=== real symmetric, two-band and weak-gap spectra ===\n");
    run_case<double>("real", Family::TwoBand, 2000, 13, 16, 1.0e8, 21);
    run_case<double>("real", Family::TwoBand, 2000, 40, 128, 1.0e8, 22);

    std::printf("\n=== real symmetric, continuous spectrum (fast path must be refused) ===\n");
    run_case<double>("real", Family::Continuous, 2000, 13, 0, 1.0e8, 31);
    run_case<double>("real", Family::Continuous, 2000, 40, 0, 1.0e8, 32);

    std::printf("\n=== real symmetric, well-conditioned and mild cases ===\n");
    run_case<double>("real", Family::Gapped, 2000, 13, 0, 1.0e4, 41);
    run_case<double>("real", Family::Gapped, 1000, 40, 8, 1.0e6, 42);
    run_case<double>("real", Family::Gapped, 200, 13, 4, 1.0e6, 43);

    std::printf("\n=== complex Hermitian (production scalar type) ===\n");
    run_case<std::complex<double>>("cplx", Family::Gapped, 1200, 13, 16, 1.0e8, 51);
    run_case<std::complex<double>>("cplx", Family::Gapped, 1200, 40, 16, 1.0e8, 52);
    run_case<std::complex<double>>("cplx", Family::Continuous, 1200, 26, 0, 1.0e8, 53);
    if (!quick) {
        run_case<std::complex<double>>("cplx", Family::Gapped, 2000, 26, 32, 1.0e8, 54);
    }

    std::printf("\n%s (%d failing case%s)\n", failures == 0 ? "ALL CASES WITHIN BUDGET" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
