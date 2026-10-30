#ifndef RSATB_IO_ABACUS_CONVENTION_HPP
#define RSATB_IO_ABACUS_CONVENTION_HPP

// ABACUS's orbital and unit conventions, and how to undo them.
//
// ABACUS and NextHam write blocks in ABACUS's own basis. It nests orbitals the
// way rsatb does (L outer, zeta inner, m inner-most) but differs inside a
// shell twice over:
//
//   - order: m runs 0, +1, -1, +2, -2, ... where rsatb runs -l, ..., +l;
//   - sign:  ABACUS's real harmonics carry the Condon-Shortley phase, so an
//            odd-m function is the negative of rsatb's, e.g. p = (z, -x, -y)
//            against rsatb's (y, z, x).
//
// So rsatb orbital m of a shell is (-1)^m times ABACUS orbital "position of m"
// of the same shell. Checked against op2c: ABACUS's own S(R) for 4s4p1d Al,
// transformed this way, matches op2c's overlap to the 1e-8 the file prints;
// untransformed it is off by 5e-2.
//
// Energies are in Rydberg, converted with ABACUS's own factor so that what
// rsatb reports agrees with what ABACUS printed in eV.

#include "hamiltonian_file.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace io {

/// ABACUS's Ry -> eV factor (source_base/constants.h, Ry_to_eV).
constexpr double kAbacusEvPerRydberg = 13.605698;

/// Where each rsatb orbital of one atom sits in the ABACUS block, and its sign.
struct AbacusOrbitalOrder {
    std::vector<int> source;    // rsatb index -> ABACUS index within the atom
    std::vector<double> sign;   // (-1)^m of that orbital
};

/// Builds the gather that turns one atom's ABACUS orbitals into rsatb's.
///
/// Args:
///   shells: The atom's shell angular momenta in block order.
inline AbacusOrbitalOrder MakeAbacusOrbitalOrder(const std::vector<int>& shells) {
    AbacusOrbitalOrder order;
    int offset = 0;
    for (int l : shells) {
        if (l < 0) {
            throw std::runtime_error("Negative angular momentum in a basis shell.");
        }
        for (int m = -l; m <= l; ++m) {
            const int am = m < 0 ? -m : m;
            // ABACUS position within the shell: 0 for m=0, 2m-1 for +m, 2m for -m.
            const int position = m == 0 ? 0 : (m > 0 ? 2 * m - 1 : 2 * am);
            order.source.push_back(offset + position);
            order.sign.push_back(am % 2 == 0 ? 1.0 : -1.0);
        }
        offset += 2 * l + 1;
    }
    return order;
}

/// One AbacusOrbitalOrder per type of a layout.
inline std::vector<AbacusOrbitalOrder> MakeAbacusOrbitalOrders(const FileBasisLayout& layout) {
    std::vector<AbacusOrbitalOrder> orders;
    orders.reserve(layout.type_shells.size());
    for (const auto& shells : layout.type_shells) {
        orders.push_back(MakeAbacusOrbitalOrder(shells));
    }
    return orders;
}

/// Writes rsatb's block from an ABACUS one: out(a, b) = s_a s_b factor in(src_a, src_b).
///
/// Args:
///   rows, cols: Orbital orders of the row and column atoms.
///   abacus: The ABACUS block, row-major, rows.source.size() x cols.source.size().
///   factor: Unit conversion applied on the way.
///   out: Receives the rsatb block.
inline void AbacusBlockToRsatb(
    const AbacusOrbitalOrder& rows,
    const AbacusOrbitalOrder& cols,
    const double* abacus,
    double factor,
    DenseBlock* out
) {
    const int n_row = static_cast<int>(rows.source.size());
    const int n_col = static_cast<int>(cols.source.size());
    out->Resize(n_row, n_col);
    for (int a = 0; a < n_row; ++a) {
        const double* in_row = abacus + static_cast<size_t>(rows.source[a]) * n_col;
        const double row_factor = factor * rows.sign[a];
        double* out_row = out->values.data() + static_cast<size_t>(a) * n_col;
        for (int b = 0; b < n_col; ++b) {
            out_row[b] = row_factor * cols.sign[b] * in_row[cols.source[b]];
        }
    }
}

/// Rejects a spinor layout for a format whose spinor H(R) is complex.
///
/// ABACUS nspin=4 and NextHam nspin=4 matrices are complex, and file-backed
/// Hamiltonians in rsatb are real-valued, so there is nothing correct to
/// return; failing names the reason instead of reading half of the data.
inline void RequireSpinlessLayout(const FileBasisLayout& layout, const std::string& format) {
    if (layout.spin_components != 1) {
        throw std::runtime_error(
            "The " + format + " reader supports spin-degenerate Hamiltonians only: a spinor "
            "(nspin=4) H(R) is complex, and rsatb's file-backed Hamiltonians are real-valued.");
    }
}

/// Checks that a layout is complete enough to index a file by.
inline void ValidateLayout(const FileBasisLayout& layout, const std::string& format) {
    if (layout.atom_type.empty() || layout.type_shells.empty()) {
        throw std::runtime_error(
            "The " + format + " reader needs the model's per-atom basis layout, which is empty.");
    }
    for (int type : layout.atom_type) {
        if (type < 0 || type >= static_cast<int>(layout.type_shells.size())) {
            throw std::runtime_error(
                "The " + format + " basis layout names atom type " + std::to_string(type) +
                ", but only " + std::to_string(layout.type_shells.size()) + " types exist.");
        }
    }
}

}  // namespace io
}  // namespace rsatb

#endif  // RSATB_IO_ABACUS_CONVENTION_HPP
