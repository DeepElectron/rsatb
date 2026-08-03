#ifndef RSATB_CALC_COLLOCATION_HPP
#define RSATB_CALC_COLLOCATION_HPP

// Contraction of a spectral or density block onto a real-space grid.
//
//     Q(r) = sum_{(mu,nu) in P} phi_mu(r) D_mu_nu phi_nu(r)
//
// This is equation (5.4) of doc/design_stochastic_ldos.md, and the same kernel
// serves the density path. The matrix of orbital values on the grid is never
// formed: only a few hundred orbitals are non-zero in any one box, so values
// are evaluated per box and discarded -- the strategy SIESTA and CP2K use to
// keep grid work O(N) in time and O(1) in storage beyond the output.
//
// Three structural choices carry the cost:
//
//   - The box is the unit of locality. Every point in a box sees essentially
//     the same orbitals, so the covering set is built once per box (one
//     sphere-overlap test per atom, as in rescu++'s grid_shell_mask) and the
//     box contracts in one dense GEMM.
//   - Only the real part of D is needed. D is Hermitian and the orbitals are
//     real, so the antisymmetric imaginary part cancels in the double sum.
//   - Samples are collapsed in orbital space before reaching the grid, never
//     averaged on it: assembling D first costs n_loc^2 per point against
//     S_smp * n_loc for the stochastic-on-grid route, and n_loc is much the
//     smaller of the two.
//
// Orbital values come from op2c's OrbitalEvaluator, which is also what built
// the Hamiltonian and overlap -- so the per-atom orbital ordering matches the
// blocks of H and S by construction, not by convention.

#include "pbc_geometry.hpp"
#include "real_space_grid.hpp"

#include "../model/hamiltonian_images.hpp"
#include "../model/lcao/units.hpp"
#include "nao/atomic_radials.h"
#include "nao/orbital_evaluator.h"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/detail/kernels/lapack_api.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace rsatb {
namespace calc {

/// The atoms a real-space contraction indexes its matrix block by.
///
/// Replicated across the communicator: the contraction runs on every rank over
/// its own grid boxes, and a rank generally does not own the atoms whose
/// orbitals reach those boxes.
struct OrbitalSet {
    std::vector<int> atoms;       ///< Global atom indices, ascending.
    std::vector<int> atom_type;   ///< Type of each atom.
    std::vector<int> atom_norb;   ///< Orbital count of each atom.
    std::vector<int> orb_offset;  ///< Prefix sum of atom_norb, size n_atom()+1.
    std::vector<double> positions;///< Cartesian Angstrom, (n_atom, 3).

    int n_atom() const { return static_cast<int>(atoms.size()); }
    int n_orb() const { return orb_offset.empty() ? 0 : orb_offset.back(); }
};

/// Atoms whose orbitals reach any of the given spheres.
///
/// Sphere-overlap rather than point tests: with a grid this is one test per
/// box instead of one per grid point, which is what makes selection affordable
/// at scale. Pass empty radii to test bare points.
///
/// Args:
///   atomic_data: Distributed geometry; supplies positions, types and cell.
///   centers: Sphere centres, Cartesian Angstrom, (n, 3).
///   radii: One radius per centre, or empty for points.
///   type_rcut: Orbital cutoff radius per atom type, Angstrom.
///   comm: Communicator the set must be agreed over.
inline OrbitalSet BuildOrbitalSetForSpheres(vbcsr::atomic::AtomicData& atomic_data,
                                            const std::vector<double>& centers,
                                            const std::vector<double>& radii,
                                            const std::vector<double>& type_rcut,
                                            MPI_Comm comm) {
    if (centers.empty() || centers.size() % 3 != 0) {
        throw std::runtime_error("Orbital-set selection needs a non-empty (n, 3) array.");
    }
    const int n_center = static_cast<int>(centers.size() / 3);
    if (!radii.empty() && static_cast<int>(radii.size()) != n_center) {
        throw std::runtime_error("Orbital-set selection needs one radius per centre.");
    }

    const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());

    std::vector<int> local_meta;    // (global index, type, norb) per atom
    std::vector<double> local_pos;  // (x, y, z) per atom
    for (int local = 0; local < atomic_data.n_atom; ++local) {
        const int type = atomic_data.atom_type[local];
        if (type < 0 || type >= static_cast<int>(type_rcut.size())) {
            throw std::runtime_error("Orbital-set selection was given no cutoff for an atom type.");
        }
        const double rcut = type_rcut[type];
        double ax = 0.0, ay = 0.0, az = 0.0;
        atomic_data.get_pos(local, &ax, &ay, &az);

        bool reaches = false;
        for (int p = 0; p < n_center && !reaches; ++p) {
            const double distance = detail::MinimumImageDistance(
                &centers[3 * static_cast<size_t>(p)], ax, ay, az,
                atomic_data.cell, cell_inv, atomic_data.pbc);
            reaches = (distance < rcut + (radii.empty() ? 0.0 : radii[p]));
        }
        if (reaches) {
            local_meta.push_back(atomic_data.get_global_index(local));
            local_meta.push_back(type);
            local_meta.push_back(atomic_data.type_norb[type]);
            local_pos.push_back(ax);
            local_pos.push_back(ay);
            local_pos.push_back(az);
        }
    }

    int comm_size = 1;
    if (comm != MPI_COMM_NULL) MPI_Comm_size(comm, &comm_size);

    std::vector<int> all_meta = local_meta;
    std::vector<double> all_pos = local_pos;
    if (comm != MPI_COMM_NULL && comm_size > 1) {
        const int local_count = static_cast<int>(local_meta.size() / 3);
        std::vector<int> counts(comm_size, 0);
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

        std::vector<int> displs(comm_size, 0);
        int total = 0;
        for (int r = 0; r < comm_size; ++r) {
            displs[r] = total * 3;
            total += counts[r];
            counts[r] *= 3;
        }
        all_meta.assign(static_cast<size_t>(total) * 3, 0);
        all_pos.assign(static_cast<size_t>(total) * 3, 0.0);
        MPI_Allgatherv(local_meta.data(), local_count * 3, MPI_INT,
                       all_meta.data(), counts.data(), displs.data(), MPI_INT, comm);
        MPI_Allgatherv(local_pos.data(), local_count * 3, MPI_DOUBLE,
                       all_pos.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
    }

    const size_t n_found = all_meta.size() / 3;
    std::vector<int> order(n_found);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return all_meta[3 * a] < all_meta[3 * b]; });

    OrbitalSet set;
    set.atoms.reserve(n_found);
    set.atom_type.reserve(n_found);
    set.atom_norb.reserve(n_found);
    set.positions.reserve(3 * n_found);
    for (int index : order) {
        set.atoms.push_back(all_meta[3 * index + 0]);
        set.atom_type.push_back(all_meta[3 * index + 1]);
        set.atom_norb.push_back(all_meta[3 * index + 2]);
        set.positions.push_back(all_pos[3 * index + 0]);
        set.positions.push_back(all_pos[3 * index + 1]);
        set.positions.push_back(all_pos[3 * index + 2]);
    }
    set.orb_offset.assign(set.atoms.size() + 1, 0);
    for (size_t i = 0; i < set.atoms.size(); ++i) {
        set.orb_offset[i + 1] = set.orb_offset[i] + set.atom_norb[i];
    }

    if (set.atoms.empty()) {
        throw std::runtime_error(
            "No orbital reaches any of the requested evaluation points. Check that the points "
            "lie inside or near the cell.");
    }
    return set;
}

/// Evaluates numerical atomic orbitals on points and contracts blocks onto them.
///
/// One OrbitalEvaluator is built per atom type, which splines every radial
/// once; point evaluation afterwards reuses those splines.
class OrbitalCollocation {
public:
    /// Builds evaluators from the same orbital files the model was given.
    explicit OrbitalCollocation(const std::vector<std::string>& orbital_paths,
                                MPI_Comm comm = MPI_COMM_WORLD) {
        if (orbital_paths.empty()) {
            throw std::runtime_error(
                "Real-space contraction needs orbital files; it is defined only for a model "
                "whose basis has radial functions (LCAO or DFTB with .orb).");
        }
        evaluators_.reserve(orbital_paths.size());
        for (size_t type = 0; type < orbital_paths.size(); ++type) {
            AtomicRadials radials;
            radials.build(orbital_paths[type], static_cast<int>(type), 0, 0, nullptr, comm);
            evaluators_.push_back(std::make_unique<OrbitalEvaluator>(radials));
        }
    }

    int n_type() const { return static_cast<int>(evaluators_.size()); }

    int nphi(int type) const { return Evaluator(type).nphi(); }

    /// Orbital cutoff of a type, Angstrom -- op2c tabulates radials in Bohr.
    double rcut_angstrom(int type) const {
        return Evaluator(type).rcut_max() * model::lcao::kAngstromPerBohr;
    }

    /// Contracts a matrix block onto the grid boxes this rank owns.
    ///
    /// The box is the unit of locality: its bounding sphere decides which atoms
    /// reach it once, and all of its points then share that covering set, one
    /// Phi buffer and one GEMM.
    ///
    /// Args:
    ///   atomic_data: Supplies cell and boundary conditions for imaging.
    ///   set: Atoms and orbital numbering the block is indexed by.
    ///   block: Row-major n_orb x n_orb Hermitian block over `set`.
    ///   grid: The grid whose boxes are being filled.
    ///   owned_boxes: Indices into `grid.boxes` this rank is responsible for.
    ///   values_out: grid.n_point() values; entries outside this rank's boxes
    ///     are left zero, ready to be summed across ranks.
    void ContractOntoBoxes(vbcsr::atomic::AtomicData& atomic_data,
                           const OrbitalSet& set,
                           const std::vector<std::complex<double>>& block,
                           const RealSpaceGrid& grid,
                           const std::vector<int>& owned_boxes,
                           std::vector<double>* values_out) const {
        if (values_out == nullptr) {
            throw std::runtime_error("ContractOntoBoxes needs an output pointer.");
        }
        const int n_orb = set.n_orb();
        if (block.size() != static_cast<size_t>(n_orb) * n_orb) {
            throw std::runtime_error("Block size does not match the orbital-set size.");
        }
        RequireUniqueMinimumImage(atomic_data, set);

        std::vector<double> block_real(block.size());
        for (size_t i = 0; i < block.size(); ++i) block_real[i] = block[i].real();

        values_out->assign(grid.n_point(), 0.0);
        const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());

        std::vector<double> points, phi, product, relative, local_block, box_values;
        std::vector<int> active;

        for (int b : owned_boxes) {
            const GridBox& box = grid.boxes[b];
            const int n_box_point = box.n_point();
            if (n_box_point == 0) continue;

            const LocalView local =
                ViewForSphere(atomic_data, set, box.center, box.radius, cell_inv);
            const int n_local_orb = static_cast<int>(local.orbitals.size());
            if (n_local_orb == 0) continue;

            local_block.assign(static_cast<size_t>(n_local_orb) * n_local_orb, 0.0);
            for (int i = 0; i < n_local_orb; ++i) {
                const size_t src_row = static_cast<size_t>(local.orbitals[i]) * n_orb;
                for (int j = 0; j < n_local_orb; ++j) {
                    local_block[static_cast<size_t>(i) * n_local_orb + j] =
                        block_real[src_row + local.orbitals[j]];
                }
            }

            grid.PointsOfBox(box, &points);
            phi.assign(static_cast<size_t>(n_box_point) * n_local_orb, 0.0);
            product.resize(static_cast<size_t>(n_box_point) * n_local_orb);
            box_values.assign(n_box_point, 0.0);

            EvaluatePhi(atomic_data, set, local, points, n_box_point, cell_inv, &relative,
                        &active, phi.data());
            ContractChunk(n_box_point, n_local_orb, phi.data(), local_block.data(),
                          product.data(), box_values.data());

            const double per_angstrom_cubed = model::lcao::kBohrPerAngstrom *
                                              model::lcao::kBohrPerAngstrom *
                                              model::lcao::kBohrPerAngstrom;
            int at = 0;
            for (int ix = box.lo[0]; ix < box.hi[0]; ++ix) {
                for (int iy = box.lo[1]; iy < box.hi[1]; ++iy) {
                    for (int iz = box.lo[2]; iz < box.hi[2]; ++iz) {
                        (*values_out)[grid.LinearIndex(ix, iy, iz)] =
                            box_values[at] * per_angstrom_cubed;
                        ++at;
                    }
                }
            }
        }
    }

    /// Contracts a block onto scattered points, serially.
    ///
    /// The grid path above is the production route; this exists because the
    /// orbital evaluation itself is validated against quadrature identities
    /// (normalization, angular orthogonality) that are naturally posed on a
    /// quadrature mesh rather than on the cell grid.
    void ContractOntoPoints(vbcsr::atomic::AtomicData& atomic_data,
                            const OrbitalSet& set,
                            const std::vector<std::complex<double>>& block,
                            const std::vector<double>& points,
                            std::vector<double>* values_out,
                            int point_chunk = 2048) const {
        if (values_out == nullptr) {
            throw std::runtime_error("ContractOntoPoints needs an output pointer.");
        }
        if (points.empty() || points.size() % 3 != 0) {
            throw std::runtime_error("Evaluation points must be a (n_point, 3) array.");
        }
        const int n_point = static_cast<int>(points.size() / 3);
        const int n_orb = set.n_orb();
        if (block.size() != static_cast<size_t>(n_orb) * n_orb) {
            throw std::runtime_error("Block size does not match the orbital-set size.");
        }
        RequireUniqueMinimumImage(atomic_data, set);

        std::vector<double> block_real(block.size());
        for (size_t i = 0; i < block.size(); ++i) block_real[i] = block[i].real();

        LocalView all;
        int offset = 0;
        for (int a = 0; a < set.n_atom(); ++a) {
            all.atoms.push_back(a);
            all.atom_orb_offset.push_back(offset);
            for (int o = 0; o < set.atom_norb[a]; ++o) all.orbitals.push_back(set.orb_offset[a] + o);
            offset += set.atom_norb[a];
        }

        values_out->assign(n_point, 0.0);
        const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());
        const int width = std::max(1, point_chunk);
        std::vector<double> phi(static_cast<size_t>(width) * n_orb);
        std::vector<double> product(static_cast<size_t>(width) * n_orb);
        std::vector<double> relative, chunk_points;
        std::vector<int> active;

        for (int begin = 0; begin < n_point; begin += width) {
            const int chunk = std::min(width, n_point - begin);
            chunk_points.assign(points.begin() + 3 * static_cast<size_t>(begin),
                                points.begin() + 3 * static_cast<size_t>(begin + chunk));
            std::fill(phi.begin(), phi.begin() + static_cast<size_t>(chunk) * n_orb, 0.0);
            EvaluatePhi(atomic_data, set, all, chunk_points, chunk, cell_inv, &relative,
                        &active, phi.data());
            ContractChunk(chunk, n_orb, phi.data(), block_real.data(), product.data(),
                          values_out->data() + begin);
        }

        const double per_angstrom_cubed = model::lcao::kBohrPerAngstrom *
                                          model::lcao::kBohrPerAngstrom *
                                          model::lcao::kBohrPerAngstrom;
        for (double& value : *values_out) value *= per_angstrom_cubed;
    }

private:
    /// The orbital-set restricted to what one box needs.
    struct LocalView {
        std::vector<int> atoms;            ///< Indices into the global set.
        std::vector<int> atom_orb_offset;  ///< Column offset in the local block.
        std::vector<int> orbitals;         ///< Global set orbital of each local column.
    };

    /// Atoms whose orbital sphere overlaps one bounding sphere.
    ///
    /// One distance test per atom instead of one per point -- what rescu++'s
    /// grid_shell_mask does, for the same reason.
    LocalView ViewForSphere(vbcsr::atomic::AtomicData& atomic_data, const OrbitalSet& set,
                            const double* center, double radius,
                            const std::array<double, 9>& cell_inv) const {
        LocalView view;
        int offset = 0;
        for (int a = 0; a < set.n_atom(); ++a) {
            const double reach = rcut_angstrom(set.atom_type[a]) + radius;
            const double distance = detail::MinimumImageDistance(
                center, set.positions[3 * a + 0], set.positions[3 * a + 1],
                set.positions[3 * a + 2], atomic_data.cell, cell_inv, atomic_data.pbc);
            if (distance >= reach) continue;

            view.atoms.push_back(a);
            view.atom_orb_offset.push_back(offset);
            for (int o = 0; o < set.atom_norb[a]; ++o) {
                view.orbitals.push_back(set.orb_offset[a] + o);
            }
            offset += set.atom_norb[a];
        }
        return view;
    }

    /// Fills the (n_point x n_local_orb) orbital-value block for one box.
    void EvaluatePhi(vbcsr::atomic::AtomicData& atomic_data, const OrbitalSet& set,
                     const LocalView& local, const std::vector<double>& points, int n_point,
                     const std::array<double, 9>& cell_inv, std::vector<double>* relative,
                     std::vector<int>* active, double* phi) const {
        const int stride = static_cast<int>(local.orbitals.size());
        relative->resize(static_cast<size_t>(n_point) * 3);

        for (size_t la = 0; la < local.atoms.size(); ++la) {
            const int a = local.atoms[la];
            const OrbitalEvaluator& evaluator = Evaluator(set.atom_type[a]);
            const int atom_nphi = evaluator.nphi();
            if (atom_nphi != set.atom_norb[a]) {
                throw std::runtime_error(
                    "Orbital file and Hamiltonian disagree on the orbital count of a type; "
                    "the contraction would mix unrelated orbitals.");
            }
            for (int c = 0; c < n_point; ++c) {
                detail::MinimumImageOffset(&points[3 * static_cast<size_t>(c)],
                                           set.positions[3 * a + 0], set.positions[3 * a + 1],
                                           set.positions[3 * a + 2],
                                           atomic_data.cell, cell_inv, atomic_data.pbc,
                                           &(*relative)[3 * static_cast<size_t>(c)]);
                for (int x = 0; x < 3; ++x) {
                    (*relative)[3 * static_cast<size_t>(c) + x] *= model::lcao::kBohrPerAngstrom;
                }
            }
            active->resize(atom_nphi);
            std::iota(active->begin(), active->end(), 0);
            evaluator.evaluate_active_batch(n_point, relative->data(), stride, *active,
                                            phi + local.atom_orb_offset[la]);
        }
    }

    /// Rejects a cell too narrow for one image of an orbital to be the answer.
    ///
    /// The contraction takes each orbital at its nearest image of the point.
    /// That is the whole Bloch sum only when no second image also reaches the
    /// point -- i.e. when every periodic width exceeds twice the cutoff. In a
    /// narrower cell the sum needs more terms, and taking one would quietly
    /// lose weight rather than fail.
    void RequireUniqueMinimumImage(vbcsr::atomic::AtomicData& atomic_data,
                                   const OrbitalSet& set) const {
        double rcut = 0.0;
        for (int type : set.atom_type) rcut = std::max(rcut, rcut_angstrom(type));

        double widths[3];
        detail::CellWidths(atomic_data.cell, widths);
        static const char* kAxis[3] = {"a", "b", "c"};
        for (int j = 0; j < 3; ++j) {
            if (!atomic_data.pbc[j] || widths[j] <= 0.0) continue;
            if (widths[j] <= 2.0 * rcut) {
                throw std::runtime_error(
                    "Real-space contraction needs every periodic cell width to exceed twice the "
                    "orbital cutoff, but width along " + std::string(kAxis[j]) + " is " +
                    std::to_string(widths[j]) + " A against a cutoff of " +
                    std::to_string(rcut) + " A. Use a supercell at least " +
                    std::to_string(2.0 * rcut) + " A across.");
            }
        }
    }

    const OrbitalEvaluator& Evaluator(int type) const {
        if (type < 0 || type >= static_cast<int>(evaluators_.size())) {
            throw std::runtime_error(
                "Real-space contraction has no orbital file for atom type " +
                std::to_string(type) + "; pass one file per type, in type order.");
        }
        return *evaluators_[type];
    }

    /// Q(r_p) = sum_ij phi[p,i] D[i,j] phi[p,j] for a block of points.
    ///
    /// One GEMM followed by a row-wise dot rather than a triple loop: with a
    /// few hundred orbitals per point the GEMM dominates and runs at BLAS3
    /// rates. Row-major phi is column-major phi^T and the block is symmetric,
    /// so a single 'N','N' call in Fortran layout produces the row-major
    /// product with no transposition.
    static void ContractChunk(int chunk, int n_orb, const double* phi, const double* block,
                              double* product, double* out) {
        using vbcsr::vbcsr_lapack_int;
        const vbcsr_lapack_int m = n_orb;
        const vbcsr_lapack_int n = chunk;
        const vbcsr_lapack_int k = n_orb;
        const double alpha = 1.0;
        const double beta = 0.0;
        vbcsr::dgemm_("N", "N", &m, &n, &k, &alpha, block, &m, phi, &k, &beta, product, &m);

        for (int p = 0; p < chunk; ++p) {
            const double* row_phi = phi + static_cast<size_t>(p) * n_orb;
            const double* row_product = product + static_cast<size_t>(p) * n_orb;
            double sum = 0.0;
            for (int j = 0; j < n_orb; ++j) sum += row_product[j] * row_phi[j];
            out[p] = sum;
        }
    }

    std::vector<std::unique_ptr<OrbitalEvaluator>> evaluators_;
};

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_COLLOCATION_HPP
