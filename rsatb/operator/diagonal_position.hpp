#ifndef RSATB_OPERATOR_DIAGONAL_POSITION_HPP
#define RSATB_OPERATOR_DIAGONAL_POSITION_HPP

// The position operator in an orthogonal basis.
//
// Orthogonality is what makes this simple: the basis functions are labelled by
// the atom they sit on, so r is diagonal and its entry is just the coordinate
// of that atom, replicated over the atom's orbitals:
//
//   d_alpha(i, mu) = r_i,alpha.
//
// A non-orthogonal basis has no such representation -- the overlap mixes sites
// and r needs the explicit matrix the model's own integrals produce -- so this
// route is valid exactly when there is no overlap. That is the same condition
// the commutator velocity needs, and both state it.

#include "types.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/dist_vector.hpp"

#include <complex>
#include <memory>

namespace rsatb {
namespace op {

/// Builds the diagonal of one Cartesian component of r.
inline vbcsr::DistVector<std::complex<double>> BuildDiagonalPositionAxis(
    vbcsr::atomic::AtomicData& atomic_data,
    int axis
) {
    auto* graph = atomic_data.graph;
    vbcsr::DistVector<std::complex<double>> diagonal(graph);
    diagonal.set_constant(std::complex<double>(0.0, 0.0));

    const int n_local_blocks = static_cast<int>(graph->block_sizes.size());
    for (int local_block = 0; local_block < n_local_blocks; ++local_block) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        atomic_data.get_pos(local_block, &x, &y, &z);
        const double coordinate = (axis == 0) ? x : ((axis == 1) ? y : z);

        const int norb = graph->block_sizes[local_block];
        const int offset = graph->block_offsets[local_block];
        for (int orbital = 0; orbital < norb; ++orbital) {
            diagonal[offset + orbital] = std::complex<double>(coordinate, 0.0);
        }
    }
    return diagonal;
}

/// Builds the complete orthogonal position operator: three diagonals, no matrix.
///
/// k-point independent -- r does not carry a Bloch phase in this representation
/// -- so one build serves every k-point.
///
/// Returns a triplet whose `matrix` members are null, which is what tells a
/// consumer it is looking at the orthogonal representation. Exactly one of
/// `matrix` and `orthogonal_diagonal` is ever populated.
inline std::unique_ptr<PositionTriplet> BuildDiagonalPosition(
    vbcsr::atomic::AtomicData& atomic_data
) {
    auto triplet = std::make_unique<PositionTriplet>();
    PositionAxisData* axes[3] = {&triplet->x, &triplet->y, &triplet->z};
    for (int axis = 0; axis < 3; ++axis) {
        axes[axis]->orthogonal_diagonal =
            std::make_unique<PositionTriplet::DiagonalVector>(
                BuildDiagonalPositionAxis(atomic_data, axis));
    }
    return triplet;
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_DIAGONAL_POSITION_HPP
