#ifndef RSATB_CALC_PBC_GEOMETRY_HPP
#define RSATB_CALC_PBC_GEOMETRY_HPP

// Minimum-image geometry for real-space evaluation.
//
// Separated from any one observable because both the grid collocation and the
// orbital-set searches need it, and neither should depend on the other.
//
// Positions are used exactly as supplied -- the repository convention -- so an
// image is chosen by reducing the fractional offset into [-1/2, 1/2) along
// periodic directions rather than by moving either endpoint.

#include "vbcsr/core/atomic/atomic_data.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace rsatb {
namespace calc {
namespace detail {

/// Vector from an atom to a point, taken to the nearest periodic image.
///
/// This is what lets an evaluation point near a cell face see the orbitals
/// across the boundary, and it is also the orbital argument the collocation
/// needs.
inline void MinimumImageOffset(const double* point,
                               double ax, double ay, double az,
                               const std::vector<double>& cell,
                               const std::array<double, 9>& cell_inv,
                               const std::vector<bool>& pbc,
                               double* offset_out) {
    const double d[3] = {point[0] - ax, point[1] - ay, point[2] - az};

    double frac[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < 3; ++j) {
        frac[j] = d[0] * cell_inv[0 * 3 + j] + d[1] * cell_inv[1 * 3 + j] +
                  d[2] * cell_inv[2 * 3 + j];
    }
    for (int j = 0; j < 3; ++j) {
        if (pbc[j]) frac[j] -= std::round(frac[j]);
    }
    for (int x = 0; x < 3; ++x) {
        offset_out[x] = frac[0] * cell[0 * 3 + x] + frac[1] * cell[1 * 3 + x] +
                        frac[2] * cell[2 * 3 + x];
    }
}

/// Shortest distance between a point and an atom, over periodic images.
inline double MinimumImageDistance(const double* point,
                                   double ax, double ay, double az,
                                   const std::vector<double>& cell,
                                   const std::array<double, 9>& cell_inv,
                                   const std::vector<bool>& pbc) {
    double d[3] = {0.0, 0.0, 0.0};
    MinimumImageOffset(point, ax, ay, az, cell, cell_inv, pbc, d);
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

/// Perpendicular widths of the cell -- the distances that survive shear.
inline void CellWidths(const std::vector<double>& cell, double* widths) {
    const double volume = std::abs(
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
        cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
        cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
    for (int d = 0; d < 3; ++d) {
        const int e = (d + 1) % 3;
        const int f = (d + 2) % 3;
        const double cx = cell[3 * e + 1] * cell[3 * f + 2] - cell[3 * e + 2] * cell[3 * f + 1];
        const double cy = cell[3 * e + 2] * cell[3 * f + 0] - cell[3 * e + 0] * cell[3 * f + 2];
        const double cz = cell[3 * e + 0] * cell[3 * f + 1] - cell[3 * e + 1] * cell[3 * f + 0];
        const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
        widths[d] = (area > 0.0) ? volume / area : 0.0;
    }
}

}  // namespace detail
}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_PBC_GEOMETRY_HPP
