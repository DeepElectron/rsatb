#ifndef RSATB_OPERATOR_NONORTHOGONAL_VELOCITY_HPP
#define RSATB_OPERATOR_NONORTHOGONAL_VELOCITY_HPP

// The section-4 commutator velocity of a NON-orthogonal basis
// (doc/position_commutator.md): the same i[H, x] the orthogonal builder
// assembles, dressed by the overlap. Its matrix contains S^-1 and therefore
// cannot be materialized sparsely; what CAN be materialized are the three
// sparse pieces of
//
//   V x = i * ( bond x  -  xbar S^-1 H x  +  H S^-1 delta x ),
//
//   bond  = sum_R (R + r_j - r_i) H_ij(R) e^{i phase}      (bond moment of H)
//   xbar  = X0(k) - diag(r_bra) S0(k)                      (bra-relative dipole)
//   delta = X0(k) - sum_R (R + r_j) S_ij(R) e^{i phase}    (ket-centered dipole)
//
// which Chebyshev::apply_dual applies with the metric solves it already owns.
// In the orthogonal limit xbar and delta vanish identically and V reduces to
// the bare bond moment -- the same operator commutator_velocity.hpp builds.
//
// X0 is the one ingredient needing real orbital integrals (two-centre dipoles,
// no pseudopotentials); dropping it is NOT an approximation -- the measured
// f-sum inflates by three orders of magnitude without it (doc section 7) --
// which is why a model with an overlap but no position stays velocity-less.
//
// Every piece lives on the atomic graph, i.e. the H/S sparsity pattern:
// cheaper per application than an explicit-integral velocity on its denser
// two-hop projector pattern.

#include "commutator_velocity.hpp"
#include "types.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"

#include <complex>
#include <memory>
#include <stdexcept>

namespace rsatb {
namespace op {

/// Builds the composite velocity triplet at one k-point.
///
/// Args:
///   atomic_data: Geometry and the reference graph.
///   hamiltonian_images: H(R) on that graph.
///   overlap_images: S(R) on that graph.
///   position: The model's dipole matrices X0(k), on the same graph.
///   overlap_k: S0(k), on the same graph.
/// Returns:
///   A VelocityTriplet whose axes carry {bond, xbar, delta}.
/// Throws:
///   std::runtime_error if the position triplet lacks matrices (an orthogonal
///   diagonal cannot dress a non-orthogonal commutator) or graphs mismatch.
inline std::unique_ptr<VelocityTriplet> BuildNonorthogonalCommutatorVelocity(
    vbcsr::atomic::AtomicData& atomic_data,
    const vbcsr::atomic::ImageContainer<double>& hamiltonian_images,
    const vbcsr::atomic::ImageContainer<double>& overlap_images,
    const PositionTriplet& position,
    VelocityTriplet::Matrix& overlap_k,
    const KPoint& kpoint
) {
    using Matrix = VelocityTriplet::Matrix;
    using Scalar = VelocityTriplet::Scalar;

    const PositionAxisData* pos_axes[3] = {&position.x, &position.y, &position.z};
    for (const auto* axis : pos_axes) {
        if (!axis->matrix) {
            throw std::runtime_error(
                "The non-orthogonal commutator velocity needs the position as a full "
                "matrix; this model supplies only a diagonal, which is an orthogonal-"
                "basis representation.");
        }
    }
    vbcsr::DistGraph* graph = atomic_data.get_graph();
    if (position.x.matrix->graph != graph || overlap_k.graph != graph) {
        throw std::runtime_error(
            "Non-orthogonal velocity construction requires X0(k) and S0(k) on the "
            "AtomicData base graph.");
    }

    auto triplet = std::make_unique<VelocityTriplet>();
    VelocityAxis* axes[3] = {&triplet->x, &triplet->y, &triplet->z};

    // bond and the ket-position moment of S, one image sweep each.
    Matrix* bond[3];
    Matrix* ket_s[3];
    for (int a = 0; a < 3; ++a) {
        axes[a]->bond = std::make_unique<Matrix>(graph);
        bond[a] = axes[a]->bond.get();
    }
    AccumulateBondMoment(atomic_data, hamiltonian_images, kpoint, Scalar(1.0, 0.0),
                         MomentWeight::kDisplacement, bond);
    std::unique_ptr<Matrix> ket_s_store[3];
    for (int a = 0; a < 3; ++a) {
        ket_s_store[a] = std::make_unique<Matrix>(graph);
        ket_s[a] = ket_s_store[a].get();
    }
    AccumulateBondMoment(atomic_data, overlap_images, kpoint, Scalar(1.0, 0.0),
                         MomentWeight::kKetPosition, ket_s);

    // xbar = X0 - diag(r_bra) S0 and delta = X0 - ket_s, block by block. All
    // operands share the atomic graph, but not necessarily identical stored
    // patterns, so each combination is assembled by ADDing the operands into a
    // fresh matrix rather than by aligned in-place arithmetic.
    const int n_owned = static_cast<int>(graph->owned_global_indices.size());
    auto add_scaled_rows = [&](Matrix& out, Matrix& in, bool row_position_scale, int axis,
                               Scalar uniform) {
        const auto& row_ptr = in.row_ptr();
        const auto& col_ind = in.col_ind();
        for (int row = 0; row < n_owned; ++row) {
            double r[3] = {0.0, 0.0, 0.0};
            if (row_position_scale) atomic_data.get_pos(row, &r[0], &r[1], &r[2]);
            const Scalar factor = row_position_scale ? uniform * Scalar(r[axis], 0.0)
                                                     : uniform;
            const int row_dim = graph->block_sizes[row];
            for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
                const int col = col_ind[k];
                const int col_dim = graph->block_sizes[col];
                const Scalar* data = in.block_data(k);
                if (data == nullptr) continue;
                std::vector<Scalar> scaled(static_cast<size_t>(row_dim) * col_dim);
                for (size_t t = 0; t < scaled.size(); ++t) scaled[t] = factor * data[t];
                out.update_local_block(row, col, scaled.data(), row_dim, col_dim,
                                       vbcsr::AssemblyMode::ADD,
                                       vbcsr::kCanonicalBlockLayout);
            }
        }
    };

    for (int a = 0; a < 3; ++a) {
        axes[a]->xbar = std::make_unique<Matrix>(graph);
        add_scaled_rows(*axes[a]->xbar, *pos_axes[a]->matrix, false, a, Scalar(1.0, 0.0));
        add_scaled_rows(*axes[a]->xbar, overlap_k, true, a, Scalar(-1.0, 0.0));
        axes[a]->xbar->assemble();

        axes[a]->delta = std::make_unique<Matrix>(graph);
        add_scaled_rows(*axes[a]->delta, *pos_axes[a]->matrix, false, a, Scalar(1.0, 0.0));
        add_scaled_rows(*axes[a]->delta, *ket_s[a], false, a, Scalar(-1.0, 0.0));
        axes[a]->delta->assemble();
    }
    return triplet;
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_NONORTHOGONAL_VELOCITY_HPP
