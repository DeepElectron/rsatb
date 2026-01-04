#ifndef RSATB_BACKEND_SCALAR_TRAITS_HPP
#define RSATB_BACKEND_SCALAR_TRAITS_HPP

#include <complex>

namespace rsatb {
namespace backend {

template <typename T>
struct ScalarTraits {
    using real_type = T;
    static T conjugate(const T& v) { return v; }
};

template <typename T>
struct ScalarTraits<std::complex<T>> {
    using real_type = T;
    static std::complex<T> conjugate(const std::complex<T>& v) { return std::conj(v); }
};

} // namespace backend
} // namespace rsatb

#endif
