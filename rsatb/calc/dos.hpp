#ifndef RSATB_CALC_DOS_HPP
#define RSATB_CALC_DOS_HPP

#include <algorithm>
#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include "cheby.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>

using namespace vbcsr;

namespace rsatb {
namespace calc {

// Correlation series from KPM moments (arXiv:2510.17432 Eq. 20): the
// propagated overlap is exactly a coefficient combination of the moments,
//   C(t_j) = <psi| e^{-i Op t_j} |psi> = Sum_n c_n(t_j) mu_n,
// so the full time series costs one moment recurrence instead of nt
// stepwise propagations of ~19 matvecs each. Identical to stepwise
// propagation to the 1e-14 expansion truncation: test_dos checks this
// against an inline stepwise propagation on diagonal and complex-hopping
// operators (<= 6e-14 measured). The stepwise corr_dos function itself was
// removed once that equivalence was established; git history has it.
//
// `mu` must carry at least as many moments as the expansion of the longest
// time e^{-i Op nt*pi} needs (timeprop_coeff(nt*pi) terms).
inline void corr_from_moments(const std::vector<std::complex<double>>& mu,
                              int nt,
                              std::vector<std::complex<double>>& cdos,
                              double eps = 1e-14) {
    cdos.assign(static_cast<size_t>(nt) + 1, std::complex<double>(0.0, 0.0));
    cdos[0] = mu[0];
    std::vector<std::complex<double>> coeffs;
    for (int j = 1; j <= nt; ++j) {
        timeprop_coeff(j * M_PI, eps, true, coeffs);
        const size_t n_terms = std::min(coeffs.size(), mu.size());
        std::complex<double> c(0.0, 0.0);
        for (size_t n = 0; n < n_terms; ++n) {
            c += coeffs[n] * mu[n];
        }
        cdos[static_cast<size_t>(j)] = c;
    }
}

// KPM trace moments mu_n = <psi|T_n(Op)|psi> for the moment-based DOS
// estimator, averaged over sample vectors in the multivector overload.
//
// When the scaled operator is Hermitian (cheby.op_is_hermitian()) the
// product identity 2 T_m T_n = T_{m+n} + T_{|m-n|} gives two moments per
// matvec from the states psi_n = T_n(Op) psi:
//   mu_{2n}   = 2 <psi_n|psi_n>     - mu_0
//   mu_{2n-1} = 2 <psi_n|psi_{n-1}> - mu_1
// halving the propagation cost against the plain recurrence, which remains
// the fallback for non-Hermitian applications (Left/Right preconditioning,
// solver mode). Both paths are exact per sample vector -- the identity is an
// operator identity, not a stochastic approximation -- so they agree to
// rounding, which the unit test checks.
template <typename T>
std::vector<std::complex<double>> kpm_dos_moments(Chebyshev<T>& cheby,
                                                  const DistVector<T>& psi_in,
                                                  int n_moments) {
    DistGraph* graph = psi_in.graph;
    const int rank = graph->rank;
    std::vector<std::complex<double>> mu(n_moments, std::complex<double>(0.0, 0.0));

    DistVector<T> t0(graph), t1(graph), t2(graph);
    t0.copy_from(psi_in);
    mu[0] = static_cast<std::complex<double>>(t0.dot(t0));
    if (n_moments == 1) return mu;

    cheby.apply_op(t0, t1, ChebyMode::NORMAL);
    mu[1] = static_cast<std::complex<double>>(t0.dot(t1));

    const bool doubled = cheby.op_is_hermitian();
    // Heartbeat stride: ~20 progress lines over the run (~n_moments/2
    // matvecs doubled, ~n_moments plain).
    const int stride = std::max(1, (doubled ? n_moments / 2 : n_moments) / 20);

    if (doubled) {
        for (int n = 1;; ++n) {
            ::rsatb::util::throw_if_interrupted(graph->comm);
            // t1 = psi_n, t0 = psi_{n-1}
            if (2 * n - 1 < n_moments && 2 * n - 1 > 1) {
                mu[2 * n - 1] =
                    2.0 * static_cast<std::complex<double>>(t1.dot(t0)) - mu[1];
            }
            if (2 * n < n_moments) {
                mu[2 * n] = 2.0 * static_cast<std::complex<double>>(t1.dot(t1)) - mu[0];
            }
            if (2 * n + 1 >= n_moments) break;
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
                if (::rsatb::util::every_nth(n - 1, stride))
                    std::cout << "dos: moment " << 2 * n << " / " << n_moments << std::endl;
            }
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(-1.0, t0, 2.0);
            t0.swap(t1);
            t1.swap(t2);
        }
    } else {
        for (int n = 2; n < n_moments; ++n) {
            ::rsatb::util::throw_if_interrupted(graph->comm);
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
                if (::rsatb::util::every_nth(n - 2, stride))
                    std::cout << "dos: moment " << n << " / " << n_moments << std::endl;
            }
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(-1.0, t0, 2.0);
            mu[n] = static_cast<std::complex<double>>(psi_in.dot(t2));
            t0.swap(t1);
            t1.swap(t2);
        }
    }
    return mu;
}

// Multivector version: independent random samples propagated as one block
// (the matrix stream is amortised over columns); moments are the average of
// the per-column estimates, which commutes with the doubling formulas since
// they are linear in the dot products.
template <typename T>
std::vector<std::complex<double>> kpm_dos_moments(Chebyshev<T>& cheby,
                                                  const DistMultiVector<T>& psi_in,
                                                  int n_moments) {
    DistGraph* graph = psi_in.graph;
    const int rank = graph->rank;
    const int num_vecs = psi_in.num_vectors;
    std::vector<std::complex<double>> mu(n_moments, std::complex<double>(0.0, 0.0));

    std::vector<T> dots(num_vecs);
    auto averaged = [&]() {
        std::complex<double> sum(0.0, 0.0);
        for (const auto& d : dots) sum += static_cast<std::complex<double>>(d);
        return sum / static_cast<double>(num_vecs);
    };

    DistMultiVector<T> t0(graph, num_vecs), t1(graph, num_vecs), t2(graph, num_vecs);
    t0.copy_from(psi_in);
    t0.dot(t0, dots);
    mu[0] = averaged();
    if (n_moments == 1) return mu;

    cheby.apply_op(t0, t1, ChebyMode::NORMAL);
    t0.dot(t1, dots);
    mu[1] = averaged();

    const bool doubled = cheby.op_is_hermitian();
    const int stride = std::max(1, (doubled ? n_moments / 2 : n_moments) / 20);

    if (doubled) {
        for (int n = 1;; ++n) {
            ::rsatb::util::throw_if_interrupted(graph->comm);
            if (2 * n - 1 < n_moments && 2 * n - 1 > 1) {
                t1.dot(t0, dots);
                mu[2 * n - 1] = 2.0 * averaged() - mu[1];
            }
            if (2 * n < n_moments) {
                t1.dot(t1, dots);
                mu[2 * n] = 2.0 * averaged() - mu[0];
            }
            if (2 * n + 1 >= n_moments) break;
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
                if (::rsatb::util::every_nth(n - 1, stride))
                    std::cout << "dos: moment " << 2 * n << " / " << n_moments << std::endl;
            }
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(-1.0, t0, 2.0);
            t0.swap(t1);
            t1.swap(t2);
        }
    } else {
        for (int n = 2; n < n_moments; ++n) {
            ::rsatb::util::throw_if_interrupted(graph->comm);
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
                if (::rsatb::util::every_nth(n - 2, stride))
                    std::cout << "dos: moment " << n << " / " << n_moments << std::endl;
            }
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(-1.0, t0, 2.0);
            psi_in.dot(t2, dots);
            mu[n] = averaged();
            t0.swap(t1);
            t1.swap(t2);
        }
    }
    return mu;
}

} // namespace calc
} // namespace rsatb

#endif
