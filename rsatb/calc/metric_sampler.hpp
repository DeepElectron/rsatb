#ifndef RSATB_CALC_METRIC_SAMPLER_HPP
#define RSATB_CALC_METRIC_SAMPLER_HPP

// Metric-correct random states for the stochastic LDOS estimator
// (doc/design_stochastic_ldos.md section 4.3).
//
// c0 = Zt q(W) xi with W = Zt^H S Zt and q ~ x^{-1/2}, so that
// E[c0 c0^H] = Zt W^{-1} Zt^H = S^{-1} exactly, for ANY invertible Zt
// (Proposition 4.7). Zt's quality sets the polynomial degree, never the
// answer: a purified Newton-Schulz factor gives W = I and one term, the raw
// subgraph seed gives kappa(W) ~ 2 and ~20 terms, no factor at all gives
// kappa(W) = kappa(S) and hundreds.
//
// Everything here is assembled from existing pieces:
//   Zt as an operator  ISQRTMPreconditioner(borrow)  preconditioner.hpp
//   W                  SymPrecondMat(S, Zt)          preconditioner.hpp
//   spectrum of W      PrimmeEigenSolver, Symmetric  eigen.hpp
//   coefficients of q  chebyshev_coefficients_...    kpm.hpp   (FFT, truncating)
//   sum_n c_n T_n(W)   Chebyshev::propagate          cheby.hpp (vec + multivec)
//
// Residual() is the deterministic bias certificate ||W q(W)^2 xi - xi|| /
// ||xi||: it bounds the covariance error without any sampling, and catches
// both a mis-scaled applicator and a spectrum estimate that missed part of W.

#include "cheby.hpp"
#include "eigen.hpp"
#include "kpm.hpp"
#include "preconditioner.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace calc {

struct MetricSamplerOptions {
    double eps = 1e-8;         ///< target accuracy of q^2 against W^{-1}
    double interval_pad = 0.05;  ///< pad as a fraction of the measured width of W
    int max_degree = 4000;     ///< cap on the Chebyshev degree
    double spectrum_tol = 1e-4;  ///< eigensolver tolerance for the interval ends
};

template <typename T>
class MetricSampler {
  public:
    using Matrix = vbcsr::BlockSpMat<T>;
    using Vector = vbcsr::DistVector<T>;
    using MultiVector = vbcsr::DistMultiVector<T>;

    /// Zt is borrowed, not copied: it is as large as S, and the caller
    /// usually shares one factor across several samplers. It must outlive this.
    MetricSampler(Matrix& S, Matrix& Zt, MetricSamplerOptions options = {})
        : options_(options), zt_(Zt, kBorrowFactor), w_(S, zt_) {
        MeasureSpectrum(S);
        // Pad by a fraction of the measured width, floored by the
        // eigensolver's own accuracy so the true spectrum cannot sit outside
        // the fitted interval. A relative pad would instead widen the
        // W = I case (zero width) to kappa = 1.1 and pay ~6 needless terms.
        const double pad = std::max(options_.interval_pad * (w_max_ - w_min_),
                                    2.0 * options_.spectrum_tol * w_max_);
        // The lower pad is clamped so it can never cross w_min/2: when
        // w_min << width the width-scaled pad would otherwise reach past zero,
        // where x^{-1/2} diverges.
        const double lo = w_min_ - std::min(pad, 0.5 * w_min_);
        const double hi = w_max_ + pad;
        q_lo_ = lo;
        q_hi_ = hi;
        const double center = 0.5 * (hi + lo);
        const double half = 0.5 * (hi - lo);

        // Coefficients of x^{-1/2} on [lo, hi], in the T_n((x - center)/half)
        // basis Chebyshev::propagate expands. Requested generously and
        // truncated at eps: for a purified Zt the function is constant on a
        // point interval and the series collapses to one term by itself, so
        // the small-||E|| case needs no separate branch.
        const double kappa = hi / lo;
        int n_moments = static_cast<int>(
            std::ceil(0.5 * std::sqrt(kappa) * std::log(2.0 / options_.eps))) + 4;
        n_moments = std::min(n_moments, options_.max_degree);
        coeffs_ = chebyshev_coefficients_from_function(
            n_moments,
            [center, half](double t) { return 1.0 / std::sqrt(center + half * t); },
            KpmKernel::None, options_.eps);

        cheby_ = std::make_unique<Chebyshev<T>>(S, zt_, PreconditionerType::Symmetric,
                                                lo, hi);
    }

    /// c0 = Zt q(W) xi, for xi with E[xi xi^H] = I (Rademacher or phases).
    void Sample(const Vector& xi, Vector& c0) {
        if (!tmp_) tmp_ = std::make_unique<Vector>(xi.graph);
        cheby_->propagate(xi, *tmp_, coeffs_);
        zt_.apply(*tmp_, c0);
    }

    /// Block form: one call per sample block, as the recursion wants.
    void Sample(const MultiVector& Xi, MultiVector& C0) {
        if (!tmp_mv_ || tmp_mv_->num_vectors != Xi.num_vectors) {
            tmp_mv_ = std::make_unique<MultiVector>(Xi.graph, Xi.num_vectors);
        }
        cheby_->propagate(Xi, *tmp_mv_, coeffs_);
        zt_.apply(*tmp_mv_, C0);
    }

    /// Deterministic bias certificate ||W q(W)^2 xi - xi|| / ||xi||.
    double Residual(const Vector& xi) {
        Vector q1(xi.graph), q2(xi.graph), r(xi.graph);
        cheby_->propagate(xi, q1, coeffs_);
        cheby_->propagate(q1, q2, coeffs_);
        w_.mult_optimized(q2, r);
        r.axpy(T(-1.0), xi);
        return std::sqrt(std::abs(r.dot(r))) / std::sqrt(std::abs(xi.dot(xi)));
    }

    double w_min() const { return w_min_; }
    double w_max() const { return w_max_; }
    /// ||W - I||, i.e. how far Zt is from an exact orthogonalizer.
    double residual_norm() const {
        return std::max(std::abs(w_min_ - 1.0), std::abs(w_max_ - 1.0));
    }
    /// Terms actually kept after truncation.
    int degree() const { return static_cast<int>(coeffs_.size()) - 1; }

    /// The coloring polynomial itself -- the coefficient table of q on
    /// [q_lo, q_hi] in the T_n((x - center)/half) basis -- for consumers
    /// that must reproduce THE SAME q elsewhere (the solve-form fragment
    /// surrogate compresses the actual chain Zt q(W), never an ideal
    /// x^{-1/2}: matching in covariance only, off by a rotation, kills the
    /// per-sample cancellation a control variate lives on).
    const std::vector<std::complex<double>>& q_coeffs() const { return coeffs_; }
    double q_lo() const { return q_lo_; }
    double q_hi() const { return q_hi_; }

  private:
    /// Extremal eigenvalues of W = Zt^H S Zt, via the composed-operator mode
    /// of the repo's eigensolver. Also certifies w_min > 0, i.e. that Zt is
    /// invertible and S positive definite -- never assumed.
    void MeasureSpectrum(Matrix& S) {
        PrimmeEigenSolver<T> solver;
        std::vector<typename vbcsr::ScalarTraits<T>::real_type> ev;
        std::vector<vbcsr::DistVector<T>> evec;

        solver.solve(S, nullptr, &zt_, 1, primme_largest, options_.spectrum_tol,
                     ev, evec, false, PreconditionerType::Symmetric);
        if (ev.empty()) throw std::runtime_error("MetricSampler: largest eigenvalue of W failed");
        w_max_ = static_cast<double>(ev.front());

        ev.clear();
        evec.clear();
        solver.solve(S, nullptr, &zt_, 1, primme_smallest, options_.spectrum_tol,
                     ev, evec, false, PreconditionerType::Symmetric);
        if (ev.empty()) throw std::runtime_error("MetricSampler: smallest eigenvalue of W failed");
        w_min_ = static_cast<double>(ev.front());

        if (w_min_ <= 0.0) {
            throw std::runtime_error(
                "MetricSampler: measured w_min <= 0 (" + std::to_string(w_min_) +
                "); Zt is numerically singular or S is not positive definite, "
                "so S^{-1/2} does not exist and no sampler can be built.");
        }
    }

    MetricSamplerOptions options_;
    ISQRTMPreconditioner<T> zt_;
    SymPrecondMat<Matrix, ISQRTMPreconditioner<T>, T> w_;
    double w_min_ = 0.0;
    double w_max_ = 0.0;
    double q_lo_ = 0.0;
    double q_hi_ = 0.0;
    std::vector<std::complex<double>> coeffs_;
    std::unique_ptr<Chebyshev<T>> cheby_;
    std::unique_ptr<Vector> tmp_;
    std::unique_ptr<MultiVector> tmp_mv_;
};

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_METRIC_SAMPLER_HPP
