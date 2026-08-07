#ifndef RSATB_CALC_REAL_SPACE_GRID_HPP
#define RSATB_CALC_REAL_SPACE_GRID_HPP

// A uniform real-space grid over the cell, tiled into boxes, and the partition
// of those boxes across ranks.
//
// Real-space observables are wanted on a grid, not at scattered points, and
// the difference is structural rather than cosmetic:
//
//   - Points are implicit. A grid of 10^9 points is three integers and a cell,
//     not a 24 GB coordinate array.
//   - Boxes are implicit too, for the same reason. At 10^6 atoms and 0.2 A
//     spacing the grid holds ~7e8 points in ~4e5 boxes, and a stored box list
//     would be ~22 MB replicated on every rank, growing with the volume rather
//     than with the volume this rank owns. A box is derived from its index by
//     integer arithmetic instead (BoxAt), so the grid itself is O(1).
//   - Boxes give the loop its unit of locality. Every point in a box has
//     essentially the same set of orbitals reaching it, so the active-atom
//     list is built once per box instead of once per point, and the
//     contraction over a box is one dense GEMM.
//   - A box has a centre and a radius, so "which atoms reach this box" is one
//     sphere-overlap test -- the same reach test rescu++'s grid projector uses
//     (grid_shell_mask), and far cheaper than testing every point. A box is a
//     parallelepiped (the cell shape scaled down), not irregular, and the test
//     sphere circumscribes it, so the test is conservative: an atom can pass
//     yet reach no actual point of the box. Such an atom is kept, and is
//     harmless -- the evaluator returns exact 0.0 beyond its cutoff, so its
//     columns contribute nothing; the tight per-point truth is enforced by
//     the evaluation itself, never by this pre-filter.
//   - Boxes are what gets partitioned, and the partition is what makes the
//     whole thing scale (GridDomain below).
//
// Layout follows rescu++ (src/cpp/lcao_grid/orbital_grid.cpp,
// build_grid_boxes): boxes tile the index space in x-major order, and a box
// keeps its half-open index range so its points can be regenerated on demand.

#include "vbcsr/core/atomic/inertial_partition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace calc {

/// The physical box edge the default aims for, and the cap in grid points.
///
/// Fidelity is set by the SPACING alone -- every grid value is an exact
/// orbital evaluation, not an interpolation, so a coarse grid gives exact
/// values at fewer places. What the box edge controls is locality, and it
/// must be physical, not a point count: a property map on a 1 A grid with a
/// 12-point edge would make 12 A boxes whose bounding sphere pulls in
/// thousands of atoms, and the per-box buffers would grow a hundredfold. The
/// default therefore targets ~2.4 A of box whatever the spacing (12 points at
/// the 0.2 A integration spacing, 3 points at 1 A), capped so fine grids
/// keep a few thousand points per box -- enough for the GEMM to run at BLAS3
/// rates without the Phi buffer outgrowing the cache hierarchy.
constexpr double kTargetBoxEdgeAngstrom = 2.4;
constexpr int kMaxBoxEdge = 12;

/// One tile of the grid: a half-open index range plus its bounding sphere.
///
/// Returned by value from RealSpaceGrid::BoxAt. It is 56 bytes and derived by
/// arithmetic, so producing one per loop iteration costs nothing next to the
/// GEMM it introduces -- which is what lets the grid avoid storing them.
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
///
/// Holds no per-point and no per-box storage: everything below is computed
/// from `shape`, `box_edge` and `cell`.
struct RealSpaceGrid {
    std::vector<double> cell;    ///< Row-major 3x3, Angstrom.
    int shape[3] = {0, 0, 0};    ///< Points along each lattice vector.
    int box_edge[3] = {0, 0, 0}; ///< Box side in grid points, per axis.
    int box_count[3] = {0, 0, 0};///< Boxes along each axis.

    /// Total grid points. 64-bit: 10^6 atoms at 0.2 A already exceeds 2^31.
    int64_t n_point() const {
        return static_cast<int64_t>(shape[0]) * shape[1] * shape[2];
    }

    /// Total boxes. Comfortably within int at any cell size we target
    /// (10^9 points at the default edge is ~6e5 boxes).
    int n_box() const { return box_count[0] * box_count[1] * box_count[2]; }

    /// Row-major index of a grid point, x slowest.
    int64_t LinearIndex(int ix, int iy, int iz) const {
        return (static_cast<int64_t>(ix) * shape[1] + iy) * shape[2] + iz;
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

    /// The box with linear index `b` -- the box lattice flattened to one int
    /// the same way LinearIndex flattens points: b = (bx * box_count[1] + by)
    /// * box_count[2] + bz, x slowest. The divmods below invert it.
    ///
    /// Derived, not stored. The bounding sphere is fitted to the box's eight
    /// corner points, which are exact grid points, so it is tight for any cell
    /// shape -- including a strongly sheared one, where an axis-aligned
    /// estimate would be badly wrong.
    GridBox BoxAt(int b) const {
        if (b < 0 || b >= n_box()) {
            throw std::runtime_error("Grid box index out of range.");
        }
        const int bz = b % box_count[2];
        const int by = (b / box_count[2]) % box_count[1];
        const int bx = b / (box_count[2] * box_count[1]);

        GridBox box;
        const int origin[3] = {bx * box_edge[0], by * box_edge[1], bz * box_edge[2]};
        for (int d = 0; d < 3; ++d) {
            box.lo[d] = origin[d];
            box.hi[d] = std::min(origin[d] + box_edge[d], shape[d]);
        }

        double corners[8][3];
        int at = 0;
        for (int ix : {box.lo[0], box.hi[0] - 1}) {
            for (int iy : {box.lo[1], box.hi[1] - 1}) {
                for (int iz : {box.lo[2], box.hi[2] - 1}) {
                    PointAt(ix, iy, iz, corners[at]);
                    ++at;
                }
            }
        }
        for (int x = 0; x < 3; ++x) {
            double sum = 0.0;
            for (int c = 0; c < 8; ++c) sum += corners[c][x];
            box.center[x] = sum / 8.0;
        }
        for (int c = 0; c < 8; ++c) {
            double d2 = 0.0;
            for (int x = 0; x < 3; ++x) {
                const double d = corners[c][x] - box.center[x];
                d2 += d * d;
            }
            box.radius = std::max(box.radius, std::sqrt(d2));
        }
        return box;
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

    /// Largest bounding radius over all boxes.
    ///
    /// Every full box has the same shape, so this is BoxAt(0)'s radius unless
    /// the grid does not divide evenly, where the trailing boxes are smaller.
    /// Scanning is O(n_box) arithmetic and no memory, and it runs once.
    double MaxBoxRadius() const {
        double radius = 0.0;
        const int total = n_box();
        for (int b = 0; b < total; ++b) radius = std::max(radius, BoxAt(b).radius);
        return radius;
    }
};

/// Builds a grid with at most `spacing` between points along each axis.
///
/// The shape is rounded up so the realised spacing is never coarser than
/// asked. Args:
///   cell: Row-major 3x3 lattice, Angstrom.
///   spacing: Target point separation, Angstrom.
///   box_edge: Box side in grid points; 0 picks it per axis so a box spans
///     ~kTargetBoxEdgeAngstrom whatever the spacing is (see the constant for
///     why the default must be physical).
inline RealSpaceGrid MakeRealSpaceGrid(const std::vector<double>& cell, double spacing,
                                       int box_edge = 0) {
    if (cell.size() != 9) {
        throw std::runtime_error("Real-space grid needs a row-major 3x3 cell.");
    }
    if (!(spacing > 0.0)) {
        throw std::runtime_error("Real-space grid spacing must be positive.");
    }
    if (box_edge < 0) {
        throw std::runtime_error("Real-space grid box edge must be positive, or 0 for automatic.");
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
        int edge = box_edge;
        if (edge == 0) {
            const double realised = length / grid.shape[d];
            edge = std::min(kMaxBoxEdge, std::max(1, static_cast<int>(std::ceil(
                                             kTargetBoxEdgeAngstrom / realised))));
        }
        grid.box_edge[d] = std::min(edge, grid.shape[d]);
        grid.box_count[d] = (grid.shape[d] + grid.box_edge[d] - 1) / grid.box_edge[d];
    }
    return grid;
}

/// This rank's share of the grid: which boxes, and where their points sit in
/// the compact value buffer.
///
/// Values are stored in owned-box order, box by box, each box in its own point
/// order -- so a rank allocates and writes n_point() values, not the whole
/// grid. That is the difference between O(N/P) and O(N) per rank, and at 10^6
/// atoms it is the difference between megabytes and gigabytes.
///
/// `boxes` is the only per-rank storage and is O(n_box / P) ints. Point
/// indices are not stored: GlobalIndexOfOwnedPoint derives them, because an
/// explicit index array would be as large as the values themselves.
struct GridDomain {
    std::vector<int> boxes;      ///< Global box indices, ascending.
    std::vector<int64_t> offset; ///< size boxes.size() + 1; prefix sum of box point counts.

    int64_t n_point() const { return offset.empty() ? 0 : offset.back(); }
    int n_box() const { return static_cast<int>(boxes.size()); }
};

/// Builds the offset table for a set of owned boxes.
inline GridDomain MakeGridDomain(const RealSpaceGrid& grid, std::vector<int> owned_boxes) {
    std::sort(owned_boxes.begin(), owned_boxes.end());
    owned_boxes.erase(std::unique(owned_boxes.begin(), owned_boxes.end()), owned_boxes.end());

    GridDomain domain;
    domain.boxes = std::move(owned_boxes);
    domain.offset.assign(domain.boxes.size() + 1, 0);
    for (size_t i = 0; i < domain.boxes.size(); ++i) {
        domain.offset[i + 1] = domain.offset[i] + grid.BoxAt(domain.boxes[i]).n_point();
    }
    return domain;
}

/// The boxes `rank` owns, taken from the atoms' own inertial cut.
///
/// Ownership is decided by classifying the box centre through the same cut
/// planes the atoms were partitioned by, so
///
///     point_owner == atom_owner
///
/// by construction. That is what lets a rank contract its boxes against
/// coefficients it already holds: the orbitals reaching a box belong to atoms
/// in the same region, which this rank owns or already has as graph ghosts.
/// Partitioning the grid independently of the atoms would instead leave every
/// rank needing every orbital, and the work per rank would never fall.
///
/// Communication-free and deterministic: the tree is identical on every rank
/// (it is built by a collective in RedistributeByInertia and returned to all),
/// and classification is a pure function of it, so the ranks agree on a
/// disjoint cover without exchanging anything. rescu++ takes the same position
/// for the same reason -- see GridBoxPartition::set_owned_explicit, "the
/// caller derived box ownership from the atom partition".
///
/// The balance this gives is atom-balance, not point-balance: a rank owning a
/// dense region gets fewer grid points than one owning vacuum. That is the
/// right trade, because a box with no atoms near it costs nothing to contract
/// (its active-orbital list is empty and it is skipped), while a box whose
/// orbitals live on another rank would cost a halo exchange.
///
/// Boundary orbitals. A box is claimed whole, by where its centre falls, so a
/// box straddling a cut plane overhangs into the neighbour's region, and
/// atoms owned by other ranks reach into it -- across a boundary an orbital
/// always reaches boxes it does not own. The graph ghosts cover almost all
/// of that: a foreign atom reaching a point that any owned atom's orbital
/// also covers is, by the triangle inequality, within the neighbour-list
/// cutoff of that owned atom, hence already a ghost. The gap is the case
/// with no owned atom nearby -- an overhang next to a vacuum gap wider than
/// the neighbour-list cutoff, or a region thinner than a box -- where a
/// reaching foreign atom is NOT a ghost and a purely local orbital set would
/// silently drop its weight. That is why the production set is built from
/// the demand side (BuildOrbitalSetFromDemands in collocation.hpp): each
/// atom's owner enumerates the boxes the atom reaches and ships it to their
/// owners, so coverage is exact by construction for any geometry, and
/// ValidateLocalOrbitalCoverage exists to prove the cheaper local set
/// sufficient where it is used. What never degrades is exactness of the
/// cover itself: it is a partition at any ratio, which is what the compact
/// no-reduction buffers depend on.
inline std::vector<int> OwnedBoxesFromAtoms(const RealSpaceGrid& grid,
                                            const vbcsr::atomic::InertialCut& tree,
                                            int rank) {
    std::vector<int> owned;
    const int total = grid.n_box();
    for (int b = 0; b < total; ++b) {
        if (vbcsr::atomic::ClassifyPoint(tree, grid.BoxAt(b).center) == rank) {
            owned.push_back(b);
        }
    }
    return owned;
}

/// Global grid index of the k-th value in a domain's compact buffer.
///
/// Derived rather than tabulated: the alternative is an int64 array as long as
/// the values, which doubles the per-rank footprint of the thing this layout
/// exists to shrink.
inline int64_t GlobalIndexOfOwnedPoint(const RealSpaceGrid& grid, const GridDomain& domain,
                                       int64_t k) {
    if (k < 0 || k >= domain.n_point()) {
        throw std::runtime_error("Owned point index out of range.");
    }
    const auto it = std::upper_bound(domain.offset.begin(), domain.offset.end(), k);
    const size_t slot = static_cast<size_t>(it - domain.offset.begin()) - 1;
    const GridBox box = grid.BoxAt(domain.boxes[slot]);

    int64_t within = k - domain.offset[slot];
    const int ny = box.hi[1] - box.lo[1];
    const int nz = box.hi[2] - box.lo[2];
    const int ix = static_cast<int>(within / (static_cast<int64_t>(ny) * nz));
    within -= static_cast<int64_t>(ix) * ny * nz;
    const int iy = static_cast<int>(within / nz);
    const int iz = static_cast<int>(within - static_cast<int64_t>(iy) * nz);
    return grid.LinearIndex(box.lo[0] + ix, box.lo[1] + iy, box.lo[2] + iz);
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_REAL_SPACE_GRID_HPP
