#ifndef RSATB_CALC_VECTOR_UTILS_HPP
#define RSATB_CALC_VECTOR_UTILS_HPP

// Container-generic helpers for the Chebyshev recurrence drivers.
//
// The drivers (fermi_dirac.hpp, occupation.hpp, msd.hpp) are templated on
// VectorType so one implementation serves a single state and a block of them.
// DistVector and DistMultiVector differ in exactly two places, which is what
// these two functions paper over -- and nothing else.

#include "vbcsr/core/dist_multivector.hpp"
#include "vbcsr/core/dist_vector.hpp"

#include <complex>
#include <vector>

namespace rsatb {
namespace calc {

// A zeroed container shaped like `ref`.
//
// Not a copy: the recurrence temporaries and the accumulators start at zero.
// The constructors differ (DistVector takes a graph, DistMultiVector a graph
// and a column count), so generic code cannot spell this directly.
template <typename T>
vbcsr::DistVector<T> ZeroLike(const vbcsr::DistVector<T>& ref) {
    return vbcsr::DistVector<T>(ref.graph);
}

template <typename T>
vbcsr::DistMultiVector<T> ZeroLike(const vbcsr::DistMultiVector<T>& ref) {
    return vbcsr::DistMultiVector<T>(ref.graph, ref.num_vectors);
}

// Re<a|b> as one real number, averaged over columns for a block.
//
// DistVector::dot returns a scalar, DistMultiVector::dot one per column;
// this reduces both to the single number the degree schedules compare. The
// dots are collective, so every rank decides identically.
template <typename T>
double RealDot(const vbcsr::DistVector<T>& a, const vbcsr::DistVector<T>& b) {
    return std::real(static_cast<std::complex<double>>(a.dot(b)));
}

template <typename T>
double RealDot(const vbcsr::DistMultiVector<T>& a, const vbcsr::DistMultiVector<T>& b) {
    std::vector<T> dots;
    a.dot(b, dots);
    std::complex<double> acc(0.0, 0.0);
    for (const auto& d : dots) acc += static_cast<std::complex<double>>(d);
    return std::real(acc) / static_cast<double>(a.num_vectors);
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_VECTOR_UTILS_HPP
