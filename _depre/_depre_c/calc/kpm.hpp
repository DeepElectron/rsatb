#ifndef KPM
#define KPM

#include <math.h>

namespace rsatb::calc{

inline void jackson_kernel(int n, double *out)
{
    auto denom = static_cast<double>(n + 1);
    double theta = M_PI / denom;
    double cot_theta = std::cos(theta) / std::sin(theta);
    out[0] = 1.0;
    for (int i = 1; i < n; ++i) {
        out[i] = ((denom - i) * std::cos(theta * i) + cot_theta * std::sin(theta * i)) / denom;
    }
}
}

#endif