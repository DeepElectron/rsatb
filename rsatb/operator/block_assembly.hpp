#ifndef RSATB_OPERATOR_BLOCK_ASSEMBLY_HPP
#define RSATB_OPERATOR_BLOCK_ASSEMBLY_HPP

// Phased block assembly: the glue between a real-space operator image and a
// vbcsr matrix at one k-point.
//
// Nothing here knows about a model family, an operator, or op2c. It is the
// vocabulary every construction path shares: apply the Bloch phase for a
// lattice translation and add the result into a matrix block.
//
// These were four near-identical private overloads of OperatorConstruction,
// differing only in whether the shift arrived as int or double and whether the
// values were real or complex. Both are template parameters here, so the body
// is written once.

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <cmath>
#include <complex>
#include <vector>

namespace rsatb {
namespace op {

/// Returns the Bloch phase exp(-2 pi i k . R) of a lattice translation.
///
/// Args:
///   rx, ry, rz: Lattice translation, in the units k is expressed against --
///     fractional for an integer image shift, Cartesian for a displacement
///     paired with a Cartesian k.
///   kx, ky, kz: k-point.
inline std::complex<double> BlochPhase(
    double rx, double ry, double rz,
    double kx, double ky, double kz
) {
    return std::exp(std::complex<double>(
        0.0, -2.0 * M_PI * (rx * kx + ry * ky + rz * kz)));
}

/// Multiplies a block by a scalar, widening real input to complex.
///
/// Args:
///   values: Row-major block of `count` entries, real or complex.
///   factor: Scalar applied to every entry.
/// Returns:
///   The scaled block, row-major.
template <typename Value>
inline std::vector<std::complex<double>> ScaleBlock(
    const Value* values, int count, std::complex<double> factor
) {
    std::vector<std::complex<double>> scaled(static_cast<size_t>(count));
    for (int index = 0; index < count; ++index) {
        scaled[static_cast<size_t>(index)] = values[index] * factor;
    }
    return scaled;
}

/// Adds a phased block at local row `i`, local column `j`.
///
/// The block dimensions come from the atomic data, so a caller cannot pass a
/// block that disagrees with the graph it is adding into.
///
/// Args:
///   matrix: Destination, indexed by local block row and column.
///   atomic_data: Supplies the orbital counts of both atoms.
///   i, j: Local block indices.
///   rx, ry, rz: Lattice translation; int for an image shift, double for a
///     Cartesian displacement.
///   kx, ky, kz: k-point, in the units the translation is expressed against.
///   values: Row-major block, real or complex.
///   factor: Extra scalar, applied after the phase.
template <typename Matrix, typename Shift, typename Value>
inline void AddLocalBlock(
    Matrix& matrix,
    vbcsr::atomic::AtomicData& atomic_data,
    int i, int j,
    Shift rx, Shift ry, Shift rz,
    double kx, double ky, double kz,
    const Value* values,
    std::complex<double> factor = 1.0
) {
    int row_norb = 0;
    int col_norb = 0;
    atomic_data.get_atom_norb(i, &row_norb);
    atomic_data.get_atom_norb(j, &col_norb);

    const auto phase = BlochPhase(static_cast<double>(rx), static_cast<double>(ry),
                                  static_cast<double>(rz), kx, ky, kz) * factor;
    const auto block = ScaleBlock(values, row_norb * col_norb, phase);

    matrix.update_local_block(i, j, block.data(), row_norb, col_norb,
                              vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
}

/// Adds a phased block at global row `i_gid`, global column `j_gid`.
///
/// Unlike the local form, the block dimensions are passed in: the caller is
/// adding into a row it may not own, so the atomic data cannot supply them.
template <typename Matrix, typename Value>
inline void AddGlobalBlock(
    Matrix& matrix,
    int i_gid, int j_gid,
    int row_norb, int col_norb,
    double dx, double dy, double dz,
    double kx, double ky, double kz,
    const Value* values,
    std::complex<double> factor = 1.0
) {
    const auto phase = BlochPhase(dx, dy, dz, kx, ky, kz) * factor;
    const auto block = ScaleBlock(values, row_norb * col_norb, phase);

    matrix.add_block(i_gid, j_gid, block.data(), row_norb, col_norb,
                     vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::RowMajor);
}

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_BLOCK_ASSEMBLY_HPP
