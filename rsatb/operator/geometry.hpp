#ifndef RSATB_OPERATOR_GEOMETRY_HPP
#define RSATB_OPERATOR_GEOMETRY_HPP

// Geometric quantities every operator construction shares: where an image sits
// in Cartesian space, and where an atom sits.
//
// Small enough to be one header, and kept apart from the operators built from
// them so that "the Cartesian shift of image R" has one definition rather than
// one per construction path.

#include "vbcsr/core/atomic/atomic_data.hpp"

#include <vector>

namespace rsatb {
namespace op {

/// Returns the Cartesian component of an integer image label along one axis.
///
///   R_alpha = n1 * a1_alpha + n2 * a2_alpha + n3 * a3_alpha
///
/// Args:
///   atomic_data: Supplies the cell, row-major.
///   shift: Image label (n1, n2, n3).
///   axis: 0, 1 or 2 for x, y, z.
inline double CellShiftComponent(
    const vbcsr::atomic::AtomicData& atomic_data,
    const std::vector<int>& shift,
    int axis
) {
    const auto& cell = atomic_data.cell;
    return static_cast<double>(shift[0]) * cell[axis]
         + static_cast<double>(shift[1]) * cell[axis + 3]
         + static_cast<double>(shift[2]) * cell[axis + 6];
}

/// Returns true when any image carries a lattice translation along `axis`.
///
/// An axis with no translation contributes nothing to a lattice-moment sum, so
/// the corresponding operator is left absent rather than built as zero.
template <typename Images>
inline bool HasLatticeShiftAlong(
    const vbcsr::atomic::AtomicData& atomic_data,
    const Images& images,
    int axis
) {
    for (const auto& entry : images.image_blocks) {
        if (std::abs(CellShiftComponent(atomic_data, entry.first, axis)) > 1e-12) {
            return true;
        }
    }
    return false;
}

/// Rejects images that were not built from this atomic graph.
///
/// Every construction here indexes image blocks against the atomic data's own
/// graph, so a mismatch is a programming error rather than a physical one, and
/// it is caught once here instead of producing wrong blocks later.
template <typename Images>
inline void RequireImagesMatchGraph(
    const vbcsr::atomic::AtomicData& atomic_data,
    const Images& images,
    const std::string& context
) {
    if (images.atom_data != &atomic_data || images.base_graph != atomic_data.graph) {
        throw std::runtime_error(
            context + " requires images built from the same AtomicData graph.");
    }
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_GEOMETRY_HPP
