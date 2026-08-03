#ifndef RSATB_OPERATOR_LATTICE_MOMENT_HPP
#define RSATB_OPERATOR_LATTICE_MOMENT_HPP

// The first lattice moment of a real-space image decomposition:
//
//   M_alpha = sum_R R_alpha A(R),
//
// where A(R) is the operator image for lattice translation R and R_alpha is its
// Cartesian component along axis alpha.
//
// This is a property of the image decomposition alone -- it needs no model, no
// k-point and no basis -- which is why it lives here rather than beside any
// family. The mean-square displacement uses it for the periodic route, where
// the position operator is not representable but its commutator with H is.
//
// The diagonal orthogonal-basis position term d_alpha(i, mu) = r_i,alpha is a
// different object and lives in diagonal_position.hpp.

#include "geometry.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <complex>
#include <memory>
#include <stdexcept>
#include <string>

namespace rsatb {
namespace op {

struct LatticeMomentAxis {
    using Scalar = std::complex<double>;
    using Matrix = vbcsr::BlockSpMat<Scalar>;

    /// Absent when no image translates along this axis.
    std::unique_ptr<Matrix> first_moment;
};

struct LatticeMomentTriplet {
    LatticeMomentAxis x;
    LatticeMomentAxis y;
    LatticeMomentAxis z;
};

/// Builds M_alpha for one axis by weighting each image by its Cartesian shift.
inline std::unique_ptr<LatticeMomentAxis::Matrix> BuildLatticeMomentAxis(
    vbcsr::atomic::AtomicData& atomic_data,
    vbcsr::atomic::ImageContainer<double>& images,
    int axis
) {
    auto result = std::unique_ptr<LatticeMomentAxis::Matrix>(
        images.template accumulate_weighted_images<LatticeMomentAxis::Scalar>(
            atomic_data.graph,
            [&](const std::vector<int>& shift) {
                return CellShiftComponent(atomic_data, shift, axis);
            }));
    result->assemble();
    return result;
}

/// Builds the first lattice moment along x, y and z.
///
/// An axis is populated iff at least one image carries a lattice translation
/// along it; otherwise the sum is identically zero and the axis stays absent,
/// so a consumer that needs it fails by name rather than reading zeros.
inline std::unique_ptr<LatticeMomentTriplet> BuildLatticeMoment(
    vbcsr::atomic::AtomicData& atomic_data,
    vbcsr::atomic::ImageContainer<double>& hamiltonian_images
) {
    RequireImagesMatchGraph(atomic_data, hamiltonian_images, "First-lattice-moment construction");

    auto triplet = std::make_unique<LatticeMomentTriplet>();
    LatticeMomentAxis* axes[3] = {&triplet->x, &triplet->y, &triplet->z};

    for (int axis = 0; axis < 3; ++axis) {
        if (HasLatticeShiftAlong(atomic_data, hamiltonian_images, axis)) {
            axes[axis]->first_moment = BuildLatticeMomentAxis(atomic_data, hamiltonian_images, axis);
        }
    }
    return triplet;
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_LATTICE_MOMENT_HPP
