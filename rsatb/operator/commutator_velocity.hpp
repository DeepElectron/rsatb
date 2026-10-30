#ifndef RSATB_OPERATOR_COMMUTATOR_VELOCITY_HPP
#define RSATB_OPERATOR_COMMUTATOR_VELOCITY_HPP

// The commutator velocity of an orthogonal basis, assembled from H(R) and the
// geometry alone -- plus the bond-moment accumulation every velocity builder
// shares.
//
// DERIVATION. The velocity is v = i [H, x] (hbar = 1; see UNITS). Under
// periodic boundary conditions x itself has no matrix -- it grows without
// bound across cells -- but the commutator closes: in an orthogonal basis the
// position is diagonal (diagonal_position.hpp), the bra orbital of block
// (i, j, R) sits at r_i and its ket orbital at r_j + R, so on each image block
//
//   [H, x_alpha]_ij(R) = H_ij(R) (r_j + R)_alpha - r_i,alpha H_ij(R)
//                      = (R + r_j - r_i)_alpha H_ij(R).
//
// Every hopping is weighted by its own bond vector; the weight is bounded by
// the hopping range even though x is not, which is why this operator exists
// under periodic boundary conditions at all (doc/position_commutator.md,
// section 1). The on-site block has zero weight and drops out -- a
// dispersionless model has no velocity.
//
// Sampled at k this becomes the stored operator,
//
//   V_alpha(k)_ij = i * sum_R (R + r_j - r_i)_alpha H_ij(R)
//                     * e^{-2 pi i k.(R + s_ji)},
//
// with s_ji = f_j - f_i the fractional intra-cell displacement: the ATOMIC
// gauge, the same phase convention sample_k uses for H(k), so H and V
// transform identically under k -> k+G and under shifts of the intra-cell
// origin. Do not mix in lattice-gauge (R-only) moments; section 7 of the doc
// records the failure mode that convention mixing produces.
//
// SIGN. Differentiating the phase pulls down -i (R + r_j - r_i)_cart, so the
// bond moment is i dH/dk and the stored V = i * (bond moment) = -dH/dk: the
// eigenbasis diagonal is -dE_n/dk exactly, by Hellmann-Feynman. Every builder
// shares this minus; the quadratic consumers (AC, DC, MSD, metric) cannot see
// it, and test_velocity_units locks the diagonal against -dE_n/dk for both
// the orthogonal and the LCAO family.
//
// UNITS -- the canonical velocity convention of this codebase. The stored
// operator is hbar * velocity in eV*Angstrom, i.e. the dE/dk scale. This is
// the hbar = 1 unit system the rest of the pipeline lives in: the AC/DC
// correlators step time in 1/eV (timeprop_coeff), and the SI conversion
// constant in observables.hpp is exactly e^2/(hbar * Angstrom) for a sigma
// assembled from eV*Angstrom velocities. op2c's build_overlap_velocity stores
// the same convention (its erfactor is (hbar^2/m_e) x Bohr->Angstrom to seven
// digits). An earlier version of THIS builder divided by hbar, storing
// Angstrom/fs -- the one outlier -- which put SK-model conductivities on a
// scale hbar^2 = 0.433 apart from LCAO ones through the same unconverted
// consumers (doc/position_commutator.md section 7).
//
// SCOPE. The closed form above is exact ONLY while the basis is orthogonal,
// because only there is the diagonal the whole position operator. With an
// overlap, the intra-atomic dipole and the metric enter and the velocity
// stops being a sparse matrix altogether (it contains S^-1); that case is
// BuildNonorthogonalCommutatorVelocity (nonorthogonal_velocity.hpp), which
// reuses AccumulateBondMoment below for its bond-moment pieces. Which family
// gets which is the operator-source table (model/operator_source.hpp).
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

/// Planck's constant over 2 pi, in eV fs. Kept for consumers that convert
/// eV*Angstrom velocities to Angstrom/fs for presentation; the stored
/// operators themselves are in the hbar = 1 convention above and no longer
/// divide by this.
constexpr double kHbarEvFs = 0.6582119569;

/// Which position enters the weight of a bond moment.
///
/// kDisplacement weighs each block by (R + r_j - r_i) -- the bond vector, the
/// weight the DERIVATION above produces for [H, x]. kKetPosition weighs by
/// (R + r_j) -- the ket atom's absolute position, which the ket-centered
/// dipole of the non-orthogonal composite needs
/// (nonorthogonal_velocity.hpp: Delta = X0 - ket-position moment of S).
enum class MomentWeight { kDisplacement, kKetPosition };

/// The displacement-weighted Bloch accumulation every velocity builder shares:
///
///   axes_out[a] += factor * weight_a(R, i, j) * O_ij(R)
///                  * e^{-2 pi i k.(R + s_ji)}
///
/// with the weight chosen by `weighting` and the atomic-gauge phase of the
/// header note. Null entries of axes_out skip that axis.
///
/// Args:
///   atomic_data: Supplies the atom coordinates and the reference graph.
///   images: O(R), built from this same atomic graph (H or S in practice).
///   kpoint: Fractional k-point, three components.
///   factor: Uniform prefactor -- i for a velocity, 1 for a bare moment.
///   weighting: Which position weights each block (enum above).
///   axes_out: The three Cartesian accumulation targets; assembled on return.
/// Throws:
///   std::runtime_error if the k-point is malformed or the images do not
///   belong to this graph.
template <typename T>
void AccumulateBondMoment(
    vbcsr::atomic::AtomicData& atomic_data,
    const vbcsr::atomic::ImageContainer<T>& images,
    const KPoint& kpoint,
    std::complex<double> factor,
    MomentWeight weighting,
    VelocityTriplet::Matrix* axes_out[3]
) {
    if (kpoint.size() != 3) {
        throw std::runtime_error("Bond-moment construction requires 3-component k-points.");
    }
    RequireImagesMatchGraph(atomic_data, images, "Bond-moment construction");

    vbcsr::DistGraph* reference_graph = atomic_data.get_graph();
    const int n_owned = static_cast<int>(reference_graph->owned_global_indices.size());

    for (const auto& entry : images.image_blocks) {
        const auto& shift = entry.first;
        auto* image_matrix = entry.second;
        auto* image_graph = image_matrix->graph;

        if (image_graph->owned_global_indices != reference_graph->owned_global_indices) {
            throw std::runtime_error(
                "Bond-moment construction requires image graphs aligned with the "
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
                    "Bond-moment image column not found in the AtomicData base graph.");
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
                const T* block_data = image_matrix->block_data(k);

                for (int axis = 0; axis < 3; ++axis) {
                    if (axes_out[axis] == nullptr) continue;
                    // kDisplacement: R + r_j - r_i. kKetPosition: R + r_j,
                    // which is r_i + displacement -- the ket atom's absolute
                    // position, needed by the ket-centered dipole.
                    double weight = displacement[axis];
                    if (weighting == MomentWeight::kKetPosition) {
                        weight += ri[axis];
                    }
                    // A hopping with zero weight along this axis contributes
                    // nothing; skipping keeps the matrix sparse.
                    if (std::abs(weight) <= 1e-12) {
                        continue;
                    }
                    // Elementwise scaling preserves the canonical row-major layout.
                    const auto block = ScaleBlock(block_data, n_elem, phase * factor * weight);
                    axes_out[axis]->update_local_block(
                        local_row, reference_column, block.data(), row_dim, col_dim,
                        vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
                }
            }
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (axes_out[axis] != nullptr) axes_out[axis]->assemble();
    }
}

/// Builds the Cartesian velocity operators at one k-point from H(R):
/// V = i * bond moment of H -- exact for an orthogonal basis (SCOPE above),
/// stored in the canonical eV*Angstrom convention (UNITS above).
///
/// Args:
///   atomic_data: Supplies the atom coordinates and the reference graph.
///   hamiltonian_images: H(R), built from this same atomic graph.
///   kpoint: Fractional k-point, three components.
/// Returns:
///   The three Cartesian components, as bare matrices on the atomic graph.
/// Throws:
///   std::runtime_error if the k-point is malformed or the images do not
///   belong to this graph.
template <typename T>
std::unique_ptr<VelocityTriplet> BuildCommutatorVelocity(
    vbcsr::atomic::AtomicData& atomic_data,
    const vbcsr::atomic::ImageContainer<T>& hamiltonian_images,
    const KPoint& kpoint
) {
    auto triplet = std::make_unique<VelocityTriplet>();
    triplet->x.matrix = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
    triplet->y.matrix = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
    triplet->z.matrix = std::make_unique<VelocityTriplet::Matrix>(atomic_data.get_graph());
    VelocityTriplet::Matrix* axes[3] = {
        triplet->x.matrix.get(), triplet->y.matrix.get(), triplet->z.matrix.get()};
    AccumulateBondMoment(atomic_data, hamiltonian_images, kpoint,
                         std::complex<double>(0.0, 1.0), MomentWeight::kDisplacement, axes);
    return triplet;
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_COMMUTATOR_VELOCITY_HPP
