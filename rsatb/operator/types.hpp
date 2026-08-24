#ifndef RSATB_OPERATOR_TYPES_HPP
#define RSATB_OPERATOR_TYPES_HPP

// The value types every operator is expressed in.
//
// They belong to the operator layer, not to any model: a model produces them,
// the setup layer caches them, and the observables consume them. Keeping them
// here is what lets those three depend on one vocabulary instead of on each
// other.

#include "velocity_op.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"

#include <complex>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rsatb {
namespace op {

using KPoint = std::vector<double>;

/// One Cartesian component of the velocity operator.
///
/// Orthogonal basis: the operator is a sparse matrix outright -- `matrix`
/// holds it and the composite pieces stay null. Non-orthogonal basis: the
/// section-4 commutator velocity (doc/position_commutator.md) is not sparse
/// -- its matrix contains S^-1 -- so it is carried as the three sparse pieces
/// of
///
///   V x = i * ( bond x  -  xbar S^-1 H x  +  H S^-1 delta x ),
///
///   bond  = sum_R (R + r_j - r_i) H(R) e^{i phase}     (bond moment of H)
///   xbar  = X0(k) - diag(r) S0(k)                      (bra-relative dipole)
///   delta = X0(k) - sum_R (R + r_j) S(R) e^{i phase}   (ket-centered dipole)
///
/// and applied through Chebyshev::apply_dual, which owns the metric solves.
/// Exactly one representation is populated per axis; view() hands whichever
/// it is to the engines as the non-owning VelocityOp (velocity_op.hpp).
struct VelocityAxis {
    using Scalar = std::complex<double>;
    using Matrix = vbcsr::BlockSpMat<Scalar>;

    std::unique_ptr<Matrix> matrix;
    std::unique_ptr<Matrix> bond;
    std::unique_ptr<Matrix> xbar;
    std::unique_ptr<Matrix> delta;

    bool composite() const { return static_cast<bool>(bond); }

    /// The axis as the calc engines consume it. The one conversion point
    /// between the owning storage and the borrowed view.
    VelocityOp<Scalar> view() const {
        if (composite()) {
            return {*bond, *xbar, *delta};
        }
        if (!matrix) {
            throw std::runtime_error(
                "Velocity axis holds neither a matrix nor the composite pieces; "
                "the operator set is inconsistent.");
        }
        return {*matrix};
    }
};

/// Cartesian velocity operators at one k-point.
struct VelocityTriplet {
    using Scalar = VelocityAxis::Scalar;
    using Matrix = VelocityAxis::Matrix;

    VelocityAxis x;
    VelocityAxis y;
    VelocityAxis z;
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
