#ifndef RSATB_CALC_REAL_SPACE_GRID_HPP
#define RSATB_CALC_REAL_SPACE_GRID_HPP

// A uniform real-space grid over the cell, tiled into boxes.
//
// Real-space observables are wanted on a grid, not at scattered points, and
// the difference is structural rather than cosmetic:
//
//   - Points are implicit. A grid of 10^9 points is three integers and a cell,
//     not a 24 GB coordinate array.
//   - Boxes give the loop its unit of locality. Every point in a box has
//     essentially the same set of orbitals reaching it, so the active-atom
//     list is built once per box instead of once per point, and the
//     contraction over a box is one dense GEMM.
//   - A box has a centre and a radius, so "which atoms reach this box" is one
//     sphere-overlap test -- the same reach test rescu++'s grid projector uses
//     (grid_shell_mask), and far cheaper than testing every point.
//   - Boxes are what gets partitioned. Classifying box centres by the atoms'
//     own inertial cut binds grid ownership to atom ownership.
//
// Layout follows rescu++ (src/cpp/lcao_grid/orbital_grid.cpp,
// build_grid_boxes): boxes tile the index space in x-major order, and a box
// keeps its half-open index range so its points can be regenerated on demand.

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace calc {

/// Default box edge in grid points. Chosen so a box holds a few thousand
/// points -- enough for the contraction GEMM to run at BLAS3 rates, small
/// enough that the active-atom list stays short.
constexpr int kDefaultBoxEdge = 12;

/// One tile of the grid: a half-open index range plus its bounding sphere.
struct GridBox {
    int lo[3] = {0, 0, 0};
    int hi[3] = {0, 0, 0};
    double center[3] = {0.0, 0.0, 0.0};
    double radius = 0.0;

    int n_point() const {
        return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    }
};

/// A uniform grid over the cell, tiled into boxes.
///
/// Point (ix, iy, iz) sits at (ix/n0) a0 + (iy/n1) a1 + (iz/n2) a2, so the
/// grid is periodic-consistent by construction: index n_d is the same place as
/// index 0 in the neighbouring cell.
struct RealSpaceGrid {
    std::vector<double> cell;   ///< Row-major 3x3, Angstrom.
    int shape[3] = {0, 0, 0};   ///< Points along each lattice vector.
    int box_shape[3] = {0, 0, 0};
    std::vector<GridBox> boxes;

    size_t n_point() const {
        return static_cast<size_t>(shape[0]) * shape[1] * shape[2];
    }

    /// Row-major index of a grid point, x slowest.
    size_t LinearIndex(int ix, int iy, int iz) const {
        return (static_cast<size_t>(ix) * shape[1] + iy) * shape[2] + iz;
    }

    /// Cartesian position of a grid point, Angstrom.
    void PointAt(int ix, int iy, int iz, double* out) const {
        const double f0 = static_cast<double>(ix) / shape[0];
        const double f1 = static_cast<double>(iy) / shape[1];
        const double f2 = static_cast<double>(iz) / shape[2];
        for (int x = 0; x < 3; ++x) {
            out[x] = f0 * cell[0 * 3 + x] + f1 * cell[1 * 3 + x] + f2 * cell[2 * 3 + x];
        }
    }

    /// Fills `out` with the box's points, (n_point, 3), in the box's own order.
    void PointsOfBox(const GridBox& box, std::vector<double>* out) const {
        out->resize(static_cast<size_t>(box.n_point()) * 3);
        size_t at = 0;
        for (int ix = box.lo[0]; ix < box.hi[0]; ++ix) {
            for (int iy = box.lo[1]; iy < box.hi[1]; ++iy) {
                for (int iz = box.lo[2]; iz < box.hi[2]; ++iz) {
                    PointAt(ix, iy, iz, &(*out)[at]);
                    at += 3;
                }
            }
        }
    }

    /// Centres of every box, (n_box, 3) -- the array the partition classifies.
    std::vector<double> BoxCenters() const {
        std::vector<double> centers(boxes.size() * 3);
        for (size_t b = 0; b < boxes.size(); ++b) {
            for (int x = 0; x < 3; ++x) centers[3 * b + x] = boxes[b].center[x];
        }
        return centers;
    }

    double MaxBoxRadius() const {
        double radius = 0.0;
        for (const auto& box : boxes) radius = std::max(radius, box.radius);
        return radius;
    }
};

namespace detail {

/// Centre and bounding radius of a box, from its eight index corners.
///
/// The corners are exact grid points, so the sphere is tight for any cell
/// shape -- including a strongly sheared one, where an axis-aligned estimate
/// would be badly wrong.
inline void BoxCenterRadius(const RealSpaceGrid& grid, const GridBox& box,
                            double* center_out, double* radius_out) {
    double corners[8][3];
    int at = 0;
    for (int ix : {box.lo[0], box.hi[0] - 1}) {
        for (int iy : {box.lo[1], box.hi[1] - 1}) {
            for (int iz : {box.lo[2], box.hi[2] - 1}) {
                grid.PointAt(ix, iy, iz, corners[at]);
                ++at;
            }
        }
    }
    for (int x = 0; x < 3; ++x) {
        double sum = 0.0;
        for (int c = 0; c < 8; ++c) sum += corners[c][x];
        center_out[x] = sum / 8.0;
    }
    double radius = 0.0;
    for (int c = 0; c < 8; ++c) {
        double d2 = 0.0;
        for (int x = 0; x < 3; ++x) {
            const double d = corners[c][x] - center_out[x];
            d2 += d * d;
        }
        radius = std::max(radius, std::sqrt(d2));
    }
    *radius_out = radius;
}

}  // namespace detail

/// Builds a grid with at most `spacing` between points along each axis.
///
/// The shape is rounded up so the realised spacing is never coarser than
/// asked. Args:
///   cell: Row-major 3x3 lattice, Angstrom.
///   spacing: Target point separation, Angstrom.
///   box_edge: Box side in grid points.
inline RealSpaceGrid MakeRealSpaceGrid(const std::vector<double>& cell, double spacing,
                                       int box_edge = kDefaultBoxEdge) {
    if (cell.size() != 9) {
        throw std::runtime_error("Real-space grid needs a row-major 3x3 cell.");
    }
    if (!(spacing > 0.0)) {
        throw std::runtime_error("Real-space grid spacing must be positive.");
    }
    if (box_edge < 1) {
        throw std::runtime_error("Real-space grid box edge must be at least one point.");
    }

    RealSpaceGrid grid;
    grid.cell = cell;
    for (int d = 0; d < 3; ++d) {
        const double length = std::sqrt(cell[3 * d + 0] * cell[3 * d + 0] +
                                        cell[3 * d + 1] * cell[3 * d + 1] +
                                        cell[3 * d + 2] * cell[3 * d + 2]);
        if (!(length > 0.0)) {
            throw std::runtime_error("Real-space grid needs a non-degenerate cell.");
        }
        grid.shape[d] = std::max(1, static_cast<int>(std::ceil(length / spacing)));
        grid.box_shape[d] = std::min(box_edge, grid.shape[d]);
    }

    for (int ix = 0; ix < grid.shape[0]; ix += grid.box_shape[0]) {
        for (int iy = 0; iy < grid.shape[1]; iy += grid.box_shape[1]) {
            for (int iz = 0; iz < grid.shape[2]; iz += grid.box_shape[2]) {
                GridBox box;
                box.lo[0] = ix;
                box.lo[1] = iy;
                box.lo[2] = iz;
                box.hi[0] = std::min(ix + grid.box_shape[0], grid.shape[0]);
                box.hi[1] = std::min(iy + grid.box_shape[1], grid.shape[1]);
                box.hi[2] = std::min(iz + grid.box_shape[2], grid.shape[2]);
                detail::BoxCenterRadius(grid, box, box.center, &box.radius);
                grid.boxes.push_back(box);
            }
        }
    }
    return grid;
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_REAL_SPACE_GRID_HPP
