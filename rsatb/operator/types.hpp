#ifndef RSATB_OPERATOR_TYPES_HPP
#define RSATB_OPERATOR_TYPES_HPP

// The value types every operator is expressed in.
//
// They belong to the operator layer, not to any model: a model produces them,
// the setup layer caches them, and the observables consume them. Keeping them
// here is what lets those three depend on one vocabulary instead of on each
// other.

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"

#include <complex>
#include <functional>
#include <memory>
#include <vector>

namespace rsatb {
namespace op {

using KPoint = std::vector<double>;

/// Cartesian velocity operators at one k-point.
struct VelocityTriplet {
    using Scalar = std::complex<double>;
    using Matrix = vbcsr::BlockSpMat<Scalar>;

    std::unique_ptr<Matrix> x;
    std::unique_ptr<Matrix> y;
    std::unique_ptr<Matrix> z;
};

/// One Cartesian component of the position operator.
///
/// An orthogonal basis represents position by its diagonal alone, while a
/// non-orthogonal basis needs the explicit matrix. Exactly one of the two is
/// populated for a given model.
struct PositionAxisData {
    using Scalar = std::complex<double>;
    using Matrix = vbcsr::BlockSpMat<Scalar>;
    using DiagonalVector = vbcsr::DistVector<Scalar>;

    std::unique_ptr<Matrix> matrix;
    std::unique_ptr<DiagonalVector> orthogonal_diagonal;
};

struct PositionTriplet {
    using Scalar = PositionAxisData::Scalar;
    using Matrix = PositionAxisData::Matrix;
    using DiagonalVector = PositionAxisData::DiagonalVector;

    PositionAxisData x;
    PositionAxisData y;
    PositionAxisData z;
};

/// How a k-dependent operator is produced, on demand.
///
/// Velocity and position are built lazily, per k-point, because most runs
/// never ask for them -- so the operator set has to hold something callable
/// that outlives assembly. That is all a "builder" ever was: one captured
/// pointer and one call. A function is exactly that, so there is no interface
/// and no adapter class; `operator/builders.hpp` makes the generic ones and
/// `setup/operator_assembly.hpp` makes the model-backed ones inline.
using VelocityBuilder = std::function<
    std::unique_ptr<VelocityTriplet>(vbcsr::atomic::AtomicData&, const KPoint&)>;
using PositionBuilder = std::function<
    std::unique_ptr<PositionTriplet>(vbcsr::atomic::AtomicData&, const KPoint&)>;

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_TYPES_HPP
