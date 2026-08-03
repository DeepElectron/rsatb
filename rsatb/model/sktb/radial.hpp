#ifndef RSATB_MODEL_SKTB_RADIAL_HPP
#define RSATB_MODEL_SKTB_RADIAL_HPP

// Distance dependence of a single Slater-Koster channel.
//
// Two representations are supported: a cubic spline through tabulated values,
// and an exponential-times-polynomial form with a cosine taper. Both report a
// hard cutoff, which is what the atomic graph is sized from.

#include "vbcsr/core/atomic/periodic_table.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {
namespace sktb {

/// Distance-dependent value for one Slater-Koster channel.
struct SkRadialFunction {
    std::string kind;
    std::vector<double> r;
    std::vector<double> values;
    std::vector<double> spline_second_derivatives;
    std::vector<double> coeffs;
    double lambda_value = 0.0;
    double r_on = 0.0;
    double r_cut = 0.0;

    /// Returns true when the radial function has been configured.
    bool IsEnabled() const {
        return !kind.empty();
    }

    /// Returns the normalized radial-function kind.
    std::string NormalizedKind() const {
        return vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(kind));
    }

    /// Returns the distance cutoff beyond which the value is zero.
    double Cutoff() const {
        return r_cut;
    }

    /// Validates the fields required by the selected radial-function kind.
    ///
    /// Args:
    ///   context: Prefix used in validation error messages.
    void Validate(const std::string& context) const {
        const std::string mode = NormalizedKind();
        if (mode == "table") {
            throw std::runtime_error(
                context + ": radial function kind 'table' has been removed; use 'spline' for tabulated data.");
        }
        if (mode == "spline") {
            if (r.empty() || values.empty() || r.size() != values.size()) {
                throw std::runtime_error(
                    context + ": spline radial function requires equally sized non-empty 'r' and 'values' arrays.");
            }
            for (size_t i = 1; i < r.size(); ++i) {
                if (std::abs(r[i] - r[i - 1]) <= 1e-12 || r[i] < r[i - 1]) {
                    throw std::runtime_error(context + ": spline radial distances must be strictly increasing.");
                }
            }
            if (r_cut <= 0.0) {
                throw std::runtime_error(context + ": spline radial function requires a positive r_cut.");
            }
            return;
        }
        if (mode == "exp_poly") {
            if (coeffs.empty()) {
                throw std::runtime_error(context + ": exp_poly radial function requires at least one coefficient.");
            }
            if (r_cut <= 0.0) {
                throw std::runtime_error(context + ": exp_poly radial function requires a positive r_cut.");
            }
            if (r_on < 0.0 || r_on > r_cut) {
                throw std::runtime_error(context + ": exp_poly radial function requires 0 <= r_on <= r_cut.");
            }
            return;
        }
        throw std::runtime_error(context + ": unsupported radial function kind '" + kind + "'.");
    }

    /// Builds natural cubic-spline second derivatives for tabulated data.
    std::vector<double> BuildSplineSecondDerivatives() const {
        if (r.size() < 2 || values.size() != r.size()) {
            throw std::runtime_error("Spline radial function requires at least two tabulated points.");
        }

        const size_t n = r.size();
        std::vector<double> y2(n, 0.0);
        std::vector<double> u(n - 1, 0.0);

        for (size_t i = 1; i + 1 < n; ++i) {
            const double h_prev = r[i] - r[i - 1];
            const double h_next = r[i + 1] - r[i];
            if (h_prev <= 1e-12 || h_next <= 1e-12) {
                throw std::runtime_error("Spline radial distances must be strictly increasing.");
            }

            const double sig = h_prev / (h_prev + h_next);
            const double p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;

            const double slope_next = (values[i + 1] - values[i]) / h_next;
            const double slope_prev = (values[i] - values[i - 1]) / h_prev;
            const double rhs = (6.0 * (slope_next - slope_prev) / (h_prev + h_next) - sig * u[i - 1]) / p;
            u[i] = rhs;
        }

        for (size_t k = n - 1; k-- > 0;) {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }

        return y2;
    }

    /// Evaluates the radial function at a pair distance.
    ///
    /// Args:
    ///   distance: Interatomic distance in the same units as `r` and `r_cut`.
    /// Returns:
    ///   Channel value, or zero outside the cutoff.
    double Evaluate(double distance) const {
        const std::string mode = NormalizedKind();
        if (distance < 0.0) {
            return 0.0;
        }

        if (mode == "table") {
            throw std::runtime_error("Radial function kind 'table' has been removed; use 'spline' for tabulated data.");
        }

        if (mode == "spline") {
            if (distance > r_cut) {
                return 0.0;
            }
            if (distance <= r.front()) {
                return values.front();
            }

            const auto upper = std::lower_bound(r.begin(), r.end(), distance);
            if (upper == r.end()) {
                return values.back();
            }

            const size_t i = static_cast<size_t>(std::distance(r.begin(), upper));
            const size_t lo = i - 1;
            const size_t hi = i;
            const double h = r[hi] - r[lo];
            if (h <= 1e-12) {
                return values[hi];
            }

            const std::vector<double>* y2 = &spline_second_derivatives;
            std::vector<double> local_y2;
            if (y2->empty()) {
                local_y2 = BuildSplineSecondDerivatives();
                y2 = &local_y2;
            }
            const double a = (r[hi] - distance) / h;
            const double b = (distance - r[lo]) / h;
            return a * values[lo]
                 + b * values[hi]
                 + ((a * a * a - a) * (*y2)[lo] + (b * b * b - b) * (*y2)[hi]) * (h * h) / 6.0;
        }

        if (mode == "exp_poly") {
            if (distance > r_cut) {
                return 0.0;
            }
            double cutoff_weight = 1.0;
            if (distance > r_on) {
                const double span = r_cut - r_on;
                if (span <= 1e-12) {
                    cutoff_weight = 0.0;
                } else {
                    const double arg = M_PI * (distance - r_on) / span;
                    cutoff_weight = 0.5 * (1.0 + std::cos(arg));
                }
            }

            double poly = 0.0;
            double power = 1.0;
            for (double coeff : coeffs) {
                poly += coeff * power;
                power *= distance;
            }
            return cutoff_weight * std::exp(-lambda_value * distance) * poly;
        }

        throw std::runtime_error("Unsupported radial function kind: " + kind);
    }
};

/// Builds a cubic-spline radial function from tabulated values.
///
/// Args:
///   r: Strictly increasing distances.
///   values: Channel values at those distances.
///   r_cut: Hard cutoff; defaults to the largest tabulated distance.
inline SkRadialFunction MakeSpline(
    std::vector<double> r,
    std::vector<double> values,
    double r_cut = 0.0
) {
    if (r.empty()) {
        throw std::runtime_error("A spline radial function requires at least one distance.");
    }
    if (r.size() != values.size()) {
        throw std::runtime_error("A spline radial function requires equally sized r and values arrays.");
    }

    SkRadialFunction radial;
    radial.kind = "spline";
    radial.r_cut = (r_cut > 0.0) ? r_cut : *std::max_element(r.begin(), r.end());
    radial.r = std::move(r);
    radial.values = std::move(values);
    return radial;
}

/// Builds an exponential-times-polynomial radial function with a cosine taper.
inline SkRadialFunction MakeExpPoly(
    std::vector<double> coeffs,
    double r_cut,
    double lambda_value = 0.0,
    double r_on = 0.0
) {
    SkRadialFunction radial;
    radial.kind = "exp_poly";
    radial.coeffs = std::move(coeffs);
    radial.r_cut = r_cut;
    radial.lambda_value = lambda_value;
    radial.r_on = r_on;
    return radial;
}

}  // namespace sktb
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_SKTB_RADIAL_HPP
