#ifndef RSATB_CALC_KPM_HPP
#define RSATB_CALC_KPM_HPP

#include <cmath>
#include <vector>

namespace rsatb {
namespace calc {

// Jackson Kernel for KPM smoothing
// g_n = ((N - n + 1) * cos(pi * n / (N + 1)) + sin(pi * n / (N + 1)) * cot(pi / (N + 1))) / (N + 1)
inline void jackson_kernel(int n, double *out) {
    auto denom = static_cast<double>(n + 1);
    double theta = M_PI / denom;
    double cot_theta = std::cos(theta) / std::sin(theta);
    out[0] = 1.0;
    for (int i = 1; i < n; ++i) {
        out[i] = ((denom - i) * std::cos(theta * i) + cot_theta * std::sin(theta * i)) / denom;
    }
}

// Vector version
inline std::vector<double> jackson_kernel(int n) {
    std::vector<double> out(n);
    jackson_kernel(n, out.data());
    return out;
}

} // namespace calc
} // namespace rsatb

#endif
