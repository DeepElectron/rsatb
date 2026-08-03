#ifndef RSATB_OPERATOR_COMMUTATOR_VELOCITY_HPP
#define RSATB_OPERATOR_COMMUTATOR_VELOCITY_HPP

// The velocity operator of an orthogonal basis, from H(R) and the geometry.
//
//   v = (i / hbar) [H, r]
//
// In an orthogonal basis r is diagonal with entry r_i (diagonal_position.hpp),
// so the commutator has a closed form on each Hamiltonian block:
//
//   v_alpha(k)_{i,j} = (i / hbar) * (r_j,alpha + R_alpha - r_i,alpha)
//                      * e^{-2 pi i k . (R + s_ji)} * H(R)_{i,j},
//
// i.e. every hopping contributes its own displacement. The diagonal block of
// the home image has zero displacement and drops out, which is why a
// dispersionless model has no velocity.
//
// This is valid EXACTLY when the basis is orthogonal. With an overlap, r is not
// diagonal and this expression is not the velocity -- which is why a model that
// declares an overlap may not name this as its velocity source
// (model/operator_source.hpp).
//
// It needs no model: any family whose basis is orthogonal gets its velocity
// from here, whether H(R) came from Slater-Koster parameters or from a file.

#include "block_assembly.hpp"
#include "geometry.hpp"
#include "types.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rsatb {
namespace op {

/// Planck's constant over 2 pi, in eV fs -- the units H and the time
/// propagation are expressed in.
constexpr double kHbarEvFs = 0.6582119569;

/// Builds the Cartesian velocity operators at one k-point from H(R).
///
/// Args:
///   atomic_data: Supplies the atom coordinates and the reference graph.
///   hamiltonian_images: H(R), built from this same atomic graph.
///   kpoint: Fractional k-point, three components.
/// Returns:
///   The three Cartesian components, assembled.
/// Throws:
///   std::runtime_error if the k-point is malformed or the images do not
///   belong to this graph.
inline std::unique_ptr<VelocityTriplet> BuildCommutatorVelocity(
    vbcsr::atomic::AtomicData& atomic_data,
    const vbcsr::atomic::ImageContainer<double>& hamiltonian_images,
    const KPoint& kpoint
) {
    if (kpoint.size() != 3) {
        throw std::runtime_error("Commutator velocity construction requires 3-component k-points.");
    }
    RequireImagesMatchGraph(atomic_data, hamiltonian_images, "Commutator velocity construction");

    auto triplet = std::make_unique<VelocityTriplet>();
    triplet->x = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
    triplet->y = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
    triplet->z = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());

    vbcsr::DistGraph* reference_graph = atomic_data.get_graph();
    const int n_owned = static_cast<int>(reference_graph->owned_global_indices.size());
    const std::complex<double> imaginary_unit(0.0, 1.0);

    for (const auto& entry : hamiltonian_images.image_blocks) {
        const auto& shift = entry.first;
        auto* image_matrix = entry.second;
        auto* image_graph = image_matrix->graph;

        if (image_graph->owned_global_indices != reference_graph->owned_global_indices) {
            throw std::runtime_error(
                "Commutator velocity construction requires image graphs aligned with the "
                "AtomicData base graph.");
        }

        const std::complex<double> image_phase = BlochPhase(
            static_cast<double>(shift[0]), static_cast<double>(shift[1]),
            static_cast<double>(shift[2]), kpoint[0], kpoint[1], kpoint[2]);

        const double shift_x = CellShiftComponent(atomic_data, shift, 0);
        const double shift_y = CellShiftComponent(atomic_data, shift, 1);
        const double shift_z = CellShiftComponent(atomic_data, shift, 2);

        // The image graph's local column order need not match the base graph's,
        // so map it once per image rather than per block.
        const int n_image_columns = static_cast<int>(image_graph->block_sizes.size());
        std::vector<int> column_map(n_image_columns, -1);
        for (int local_column = 0; local_column < n_image_columns; ++local_column) {
            const int gid = image_graph->get_global_index(local_column);
            auto it = reference_graph->global_to_local.find(gid);
            if (it == reference_graph->global_to_local.end()) {
                throw std::runtime_error(
                    "Commutator velocity image column not found in the AtomicData base graph.");
            }
            column_map[local_column] = it->second;
        }

        for (int local_row = 0; local_row < n_owned; ++local_row) {
            double ri[3];
            atomic_data.get_pos(local_row, &ri[0], &ri[1], &ri[2]);

            const int start = image_matrix->row_ptr()[local_row];
            const int end = image_matrix->row_ptr()[local_row + 1];
            const int row_dim = image_graph->block_sizes[local_row];

            for (int k = start; k < end; ++k) {
                const int image_column = image_matrix->col_ind()[k];
                const int reference_column = column_map[image_column];
                const int col_dim = image_graph->block_sizes[image_column];

                double rj[3];
                atomic_data.get_pos(reference_column, &rj[0], &rj[1], &rj[2]);

                const double delta_x = rj[0] - ri[0];
                const double delta_y = rj[1] - ri[1];
                const double delta_z = rj[2] - ri[2];

                // The intra-cell part of the phase is carried in fractional
                // coordinates, matching the image part above.
                double frac_dx = delta_x;
                double frac_dy = delta_y;
                double frac_dz = delta_z;
                atomic_data.invert_cell(&frac_dx, &frac_dy, &frac_dz);
                const std::complex<double> phase = image_phase * BlochPhase(
                    frac_dx, frac_dy, frac_dz, kpoint[0], kpoint[1], kpoint[2]);

                const double displacement[3] = {
                    shift_x + delta_x, shift_y + delta_y, shift_z + delta_z};

                const int n_elem = static_cast<int>(image_matrix->block_size_elements(k));
                const double* block_data = image_matrix->block_data(k);

                VelocityTriplet::Matrix* axes[3] = {
                    triplet->x.get(), triplet->y.get(), triplet->z.get()};
                for (int axis = 0; axis < 3; ++axis) {
                    // A hopping with no displacement along this axis carries no
                    // velocity along it; skipping keeps the matrix sparse.
                    if (std::abs(displacement[axis]) <= 1e-12) {
                        continue;
                    }
                    const std::complex<double> factor =
                        phase * imaginary_unit * (displacement[axis] / kHbarEvFs);
                    // Elementwise scaling preserves the canonical row-major layout.
                    const auto block = ScaleBlock(block_data, n_elem, factor);
                    axes[axis]->update_local_block(
                        local_row, reference_column, block.data(), row_dim, col_dim,
                        vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
                }
            }
        }
    }

    triplet->x->assemble();
    triplet->y->assemble();
    triplet->z->assemble();
    return triplet;
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_COMMUTATOR_VELOCITY_HPP
