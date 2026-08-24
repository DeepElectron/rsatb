#ifndef RSATB_OPERATOR_VELOCITY_OP_HPP
#define RSATB_OPERATOR_VELOCITY_OP_HPP

// The velocity operator as the Chebyshev engine consumes it: a non-owning,
// scalar-generic view over either a bare sparse matrix (orthogonal basis) or
// the three sparse pieces of the section-4 commutator velocity, whose full
// matrix contains S^-1 and cannot be materialized (types.hpp: VelocityAxis
// documents the pieces; doc/position_commutator.md derives them).
//
// This is the view half of an owner/view pair, string/string_view style:
//
//   op::VelocityAxis   owns the matrices, concrete complex<double> -- what a
//                      model builds and the operator set caches;
//   op::VelocityOp<T>  borrows them, templated over the scalar -- what the
//                      calc engines take, so real-T recursions instantiate.
//
// VelocityAxis::view() is the one conversion point between them. The metric
// solves the composite needs belong to the engine (CG against S in the Solve
// form, Z Z^H in the Sandwich form), which is why the composite is applied by
// Chebyshev::apply_dual rather than by this type. Implicit construction from
// a BlockSpMat keeps every orthogonal call site written against the matrix.
//
// V is Hermitian in either representation (it is the matrix of i[H, x]), so
// ADJOINT changes only the outer dressing, exactly as for a bare matrix.

#include "vbcsr/core/block_csr.hpp"

namespace rsatb {
namespace op {

template <typename T>
struct VelocityOp {
    vbcsr::BlockSpMat<T>* matrix = nullptr;
    vbcsr::BlockSpMat<T>* bond = nullptr;
    vbcsr::BlockSpMat<T>* xbar = nullptr;
    vbcsr::BlockSpMat<T>* delta = nullptr;

    VelocityOp(vbcsr::BlockSpMat<T>& m) : matrix(&m) {}
    VelocityOp(vbcsr::BlockSpMat<T>& b, vbcsr::BlockSpMat<T>& xb, vbcsr::BlockSpMat<T>& d)
        : bond(&b), xbar(&xb), delta(&d) {}

    bool composite() const { return bond != nullptr; }
};

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_VELOCITY_OP_HPP
