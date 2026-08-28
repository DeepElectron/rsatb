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
#include "vbcsr/core/dist_multivector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

namespace rsatb {
namespace calc {

namespace detail {

/// Drops rows of Phi that are identically zero, in place.
///
/// A box is a cube; orbital support is a union of spheres. Points outside
/// every active cutoff get exactly 0.0 from the evaluator -- not a small
/// number, the exact value -- so removing them changes no result at all
/// while shrinking the GEMM that follows. This is rescu++'s pass-1 trim
/// (orbital_grid.cpp, "threshold 0 drops [rows] ... EXACTLY zero"); its
/// optional >0 ONETEP-style variant is approximate and deliberately not
/// taken here.
///
/// The saving is geometry-dependent and honest about it: near zero for a
/// box deep inside a dense solid, where every point is covered, and large
/// at a surface, in vacuum, or wherever the active atoms only clip a
/// corner of the box. It is never a cost beyond the scan itself.
///
/// Returns the kept row count; `kept[q]` is the original row of output q.
inline int TrimZeroRows(int n_point, int n_orb, std::vector<double>* phi,
                        std::vector<int>* kept) {
    kept->clear();
    kept->reserve(n_point);
    double* data = phi->data();
    int n_kept = 0;
    for (int p = 0; p < n_point; ++p) {
        const double* row = data + static_cast<size_t>(p) * n_orb;
        bool nonzero = false;
        for (int i = 0; i < n_orb && !nonzero; ++i) nonzero = (row[i] != 0.0);
        if (!nonzero) continue;
        if (n_kept != p) {
            std::copy(row, row + n_orb, data + static_cast<size_t>(n_kept) * n_orb);
        }
        kept->push_back(p);
        ++n_kept;
    }
    return n_kept;
}

}  // namespace detail

/// The atoms a real-space contraction indexes its matrix block by.
///
/// There are two ways to build one, and the choice is a scaling decision
/// rather than a convenience.
///
/// BuildOrbitalSetForSpheres is REPLICATED: it gathers the reaching atoms from
/// every rank so all of them share one numbering. That is right for a probe of
/// a small region -- a few spheres, a handful of atoms, and the block indexed
/// by it is small -- and it is what the deterministic region LDOS wants.
///
/// BuildOrbitalSetFromDemands is per-rank and is the production form: each
/// atom's owner ships it to the ranks whose boxes it reaches. That is the only
/// form that works for a full-cell map, because the replicated set is O(N)
/// atoms on every rank and the block indexed by it is O(N^2) -- at 10^6 atoms
/// the block alone would be 10^14 entries.
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

namespace detail {

/// Calls `emit(local_atom, box, distance)` for every (owned atom, box) pair
/// whose orbitals reach the box; `distance` runs from the atom to the box
/// centre, minimum-image.
///
/// The single traversal the whole real-space path is built on: box ownership
/// and the coefficient demands are two readings of it, so they cannot drift
/// apart. Cost is O(n_owned_atom) -- the box lattice is walked only in a
/// window around each atom, sized by the cutoff against the cell's
/// perpendicular widths (so the bound survives shear), with the exact sphere
/// test inside. At production spacings that is a few dozen boxes per atom,
/// and it never touches a box far from this rank's atoms. That is what keeps
/// the path O(N/P) instead of O(cell volume) per rank.
template <class EmitFn>
void ForEachBoxOfOwnedAtoms(vbcsr::atomic::AtomicData& atomic_data,
                            const RealSpaceGrid& grid,
                            const std::vector<double>& type_rcut, EmitFn&& emit) {
    const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());
    double widths[3];
    CellWidths(atomic_data.cell, widths);
    const double max_box_radius = grid.MaxBoxRadius();

    for (int local = 0; local < atomic_data.n_atom; ++local) {
        const int type = atomic_data.atom_type[local];
        if (type < 0 || type >= static_cast<int>(type_rcut.size())) {
            throw std::runtime_error("Orbital-set selection was given no cutoff for an atom type.");
        }
        const double rcut = type_rcut[type];
        double ax = 0.0, ay = 0.0, az = 0.0;
        atomic_data.get_pos(local, &ax, &ay, &az);

        double frac[3];
        for (int j = 0; j < 3; ++j) {
            frac[j] = ax * cell_inv[0 * 3 + j] + ay * cell_inv[1 * 3 + j] +
                      az * cell_inv[2 * 3 + j];
        }

        int b_lo[3], b_hi[3];
        for (int d = 0; d < 3; ++d) {
            const double margin = (widths[d] > 0.0) ? (rcut + max_box_radius) / widths[d] : 0.0;
            const double scale = static_cast<double>(grid.shape[d]) / grid.box_edge[d];
            b_lo[d] = static_cast<int>(std::floor((frac[d] - margin) * scale));
            b_hi[d] = static_cast<int>(std::floor((frac[d] + margin) * scale));
            if (!atomic_data.pbc[d]) {
                b_lo[d] = std::max(b_lo[d], 0);
                b_hi[d] = std::min(b_hi[d], grid.box_count[d] - 1);
            } else if (b_hi[d] - b_lo[d] + 1 >= grid.box_count[d]) {
                b_lo[d] = 0;  // window covers the whole axis; visit each box once
                b_hi[d] = grid.box_count[d] - 1;
            }
        }

        for (int bx = b_lo[0]; bx <= b_hi[0]; ++bx) {
            const int wx = ((bx % grid.box_count[0]) + grid.box_count[0]) % grid.box_count[0];
            for (int by = b_lo[1]; by <= b_hi[1]; ++by) {
                const int wy = ((by % grid.box_count[1]) + grid.box_count[1]) % grid.box_count[1];
                for (int bz = b_lo[2]; bz <= b_hi[2]; ++bz) {
                    const int wz =
                        ((bz % grid.box_count[2]) + grid.box_count[2]) % grid.box_count[2];
                    const int index = (wx * grid.box_count[1] + wy) * grid.box_count[2] + wz;
                    const GridBox box = grid.BoxAt(index);
                    const double distance = MinimumImageDistance(
                        box.center, ax, ay, az, atomic_data.cell, cell_inv, atomic_data.pbc);
                    if (distance >= rcut + box.radius) continue;
                    emit(local, index, distance);
                }
            }
        }
    }
}

/// One rank's claim on one box, and the rule that settles competing claims.
///
/// Whoever owns a box must fetch every orbital reaching it, so handing the box
/// to the rank that already owns most of them minimises exactly what the
/// coefficient halo ships. Distance, then rank, break ties -- making the rule
/// a strict total order, so the winner does not depend on the order claims
/// arrive in and every rank derives the same answer from the same set.
struct BoxClaim {
    int orbitals = 0;      ///< orbitals this rank owns that reach the box
    double distance = 0.0; ///< nearest such atom to the box centre
    int rank = -1;         ///< negative means "no claim at all"

    bool Beats(const BoxClaim& other) const {
        if (rank < 0) return false;
        if (other.rank < 0) return true;
        if (orbitals != other.orbitals) return orbitals > other.orbitals;
        if (distance != other.distance) return distance < other.distance;
        return rank < other.rank;
    }
};

}  // namespace detail

/// Which grid boxes this rank owns, and who owns the boxes its atoms reach.
///
/// Box ownership is DERIVED FROM THE ATOM PARTITION ITSELF, whatever produced
/// that partition -- inertial bisection, a ParMETIS refinement of one, or a
/// decomposition the caller chose. Nothing here re-derives or assumes a
/// partitioning scheme, which is what makes
///
///     point_owner == atom_owner
///
/// exact rather than approximate. An earlier design classified box centres
/// through the atoms' inertial cut tree: correct only while the atom partition
/// still IS that tree, which stops being true the moment ParMETIS refines it
/// (from_points) or the caller supplies its own (from_distributed).
///
/// `boxes` is this rank's share. `reached_box` / `reached_owner` answer "who
/// owns box b" for exactly the boxes this rank's atoms reach, which is all the
/// demand side needs -- so no rank ever holds an owner entry for the whole
/// grid.
struct GridOwnership {
    std::vector<int> boxes;          ///< Owned boxes, ascending.
    std::vector<int> reached_box;    ///< Boxes this rank's atoms reach, ascending.
    std::vector<int> reached_owner;  ///< Owner of reached_box[i].

    int n_box() const { return static_cast<int>(boxes.size()); }

    /// Owner of a box this rank's atoms reach; -1 if they do not reach it.
    int OwnerOf(int box) const {
        const auto it = std::lower_bound(reached_box.begin(), reached_box.end(), box);
        if (it == reached_box.end() || *it != box) return -1;
        return reached_owner[static_cast<size_t>(it - reached_box.begin())];
    }
};

/// Settles box ownership across the communicator, in one exchange out and back.
///
/// Every box has exactly one referee -- `box % size`, a pure function of the
/// index -- so the cover is a partition BY CONSTRUCTION rather than by a
/// geometric argument about cut planes. Each rank claims the boxes its own
/// atoms reach; the referee applies detail::BoxClaim's total order and reports
/// the winner back along the same routing. Boxes nobody claims stay with their
/// referee: no orbital reaches them, so they contract to exactly zero and
/// their owner is irrelevant beyond keeping the cover complete.
///
/// Cost: one local walk (O(owned atoms) -- see ForEachBoxOfOwnedAtoms), one
/// alltoallv of claims out, one of winners back. Nothing anywhere is O(n_box)
/// per rank.
inline GridOwnership ResolveGridOwnership(vbcsr::atomic::AtomicData& atomic_data,
                                          const RealSpaceGrid& grid,
                                          const std::vector<double>& type_rcut,
                                          MPI_Comm comm) {
    int rank = 0, size = 1;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    // 1. This rank's claim on every box its atoms reach: how many orbitals it
    //    contributes, and its nearest contributing atom.
    struct Reach { int box; int orbitals; double distance; };
    std::vector<Reach> reaches;
    detail::ForEachBoxOfOwnedAtoms(
        atomic_data, grid, type_rcut, [&](int local, int box, double distance) {
            reaches.push_back({box, atomic_data.type_norb[atomic_data.atom_type[local]],
                               distance});
        });
    std::sort(reaches.begin(), reaches.end(),
              [](const Reach& a, const Reach& b) { return a.box < b.box; });

    std::vector<int> my_box, my_orb;
    std::vector<double> my_dist;
    for (size_t i = 0; i < reaches.size();) {
        size_t j = i;
        int orbitals = 0;
        double nearest = reaches[i].distance;
        while (j < reaches.size() && reaches[j].box == reaches[i].box) {
            orbitals += reaches[j].orbitals;
            nearest = std::min(nearest, reaches[j].distance);
            ++j;
        }
        my_box.push_back(reaches[i].box);
        my_orb.push_back(orbitals);
        my_dist.push_back(nearest);
        i = j;
    }

    // 2. Route each claim to its box's referee, grouped by destination.
    const bool exchange = (comm != MPI_COMM_NULL && size > 1);
    std::vector<int> send_counts(size, 0), recv_counts(size, 0);
    for (int box : my_box) ++send_counts[box % size];
    std::vector<int> sdis(size + 1, 0);
    for (int r = 0; r < size; ++r) sdis[r + 1] = sdis[r] + send_counts[r];

    const size_t n_send = my_box.size();
    std::vector<int> s_box(n_send), s_orb(n_send);
    std::vector<double> s_dist(n_send);
    {
        std::vector<int> cursor(sdis.begin(), sdis.end() - 1);
        for (size_t i = 0; i < n_send; ++i) {
            const int at = cursor[my_box[i] % size]++;
            s_box[at] = my_box[i];
            s_orb[at] = my_orb[i];
            s_dist[at] = my_dist[i];
        }
    }

    std::vector<int> r_box, r_orb, r_src;
    std::vector<double> r_dist;
    std::vector<int> rdis(size + 1, 0);
    if (exchange) {
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
        for (int r = 0; r < size; ++r) rdis[r + 1] = rdis[r] + recv_counts[r];
        r_box.assign(rdis[size], 0);
        r_orb.assign(rdis[size], 0);
        r_dist.assign(rdis[size], 0.0);
        MPI_Alltoallv(s_box.data(), send_counts.data(), sdis.data(), MPI_INT,
                      r_box.data(), recv_counts.data(), rdis.data(), MPI_INT, comm);
        MPI_Alltoallv(s_orb.data(), send_counts.data(), sdis.data(), MPI_INT,
                      r_orb.data(), recv_counts.data(), rdis.data(), MPI_INT, comm);
        MPI_Alltoallv(s_dist.data(), send_counts.data(), sdis.data(), MPI_DOUBLE,
                      r_dist.data(), recv_counts.data(), rdis.data(), MPI_DOUBLE, comm);
    } else {
        recv_counts[0] = static_cast<int>(n_send);
        rdis[size] = static_cast<int>(n_send);
        r_box = s_box;
        r_orb = s_orb;
        r_dist = s_dist;
    }
    r_src.assign(r_box.size(), 0);
    for (int r = 0; r < size; ++r) {
        for (int i = rdis[r]; i < rdis[r + 1]; ++i) r_src[i] = r;
    }

    // 3. Referee: the best claim wins each of its boxes.
    std::unordered_map<int, detail::BoxClaim> best;
    best.reserve(r_box.size() * 2 + 1);
    for (size_t i = 0; i < r_box.size(); ++i) {
        const detail::BoxClaim claim{r_orb[i], r_dist[i], r_src[i]};
        auto slot = best.find(r_box[i]);
        if (slot == best.end()) {
            best.emplace(r_box[i], claim);
        } else if (claim.Beats(slot->second)) {
            slot->second = claim;
        }
    }
    std::vector<int> r_winner(r_box.size(), -1);
    for (size_t i = 0; i < r_box.size(); ++i) r_winner[i] = best[r_box[i]].rank;

    // 4. The winners travel back along the reversed routing, so they land in
    //    the order the claims left in and pair off with s_box.
    std::vector<int> s_winner(n_send, -1);
    if (exchange) {
        MPI_Alltoallv(r_winner.data(), recv_counts.data(), rdis.data(), MPI_INT,
                      s_winner.data(), send_counts.data(), sdis.data(), MPI_INT, comm);
    } else {
        s_winner = r_winner;
    }

    GridOwnership ownership;
    std::vector<int> order(n_send);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return s_box[a] < s_box[b]; });
    ownership.reached_box.reserve(n_send);
    ownership.reached_owner.reserve(n_send);
    for (int i : order) {
        if (s_winner[i] < 0 || s_winner[i] >= size) {
            throw std::runtime_error("Grid box ownership resolved to a rank outside the "
                                     "communicator; the claim exchange is inconsistent.");
        }
        ownership.reached_box.push_back(s_box[i]);
        ownership.reached_owner.push_back(s_winner[i]);
        if (s_winner[i] == rank) ownership.boxes.push_back(s_box[i]);
    }

    // 5. Boxes this rank referees that nobody claimed. Walking them is
    //    O(n_box / size) -- this rank's share of the index space, not the
    //    whole grid.
    for (int box = rank; box < grid.n_box(); box += size) {
        if (best.find(box) == best.end()) ownership.boxes.push_back(box);
    }
    std::sort(ownership.boxes.begin(), ownership.boxes.end());
    ownership.boxes.erase(std::unique(ownership.boxes.begin(), ownership.boxes.end()),
                          ownership.boxes.end());
    return ownership;
}

/// The ownership rule stated independently of how it is communicated: given
/// every atom and who owns it, which rank owns each box.
///
/// This is the specification ResolveGridOwnership must reproduce, written the
/// obvious way -- every box against every atom -- so the two share the rule
/// (detail::BoxClaim) and nothing else. O(n_box * n_atom), for tests and small
/// systems, never for production.
inline std::vector<int> ResolveGridOwnershipReference(
    const RealSpaceGrid& grid, const std::vector<double>& positions,
    const std::vector<int>& types, const std::vector<int>& owners,
    const std::vector<int>& type_norb, const std::vector<double>& type_rcut,
    const std::vector<double>& cell, const std::vector<bool>& pbc, int n_rank) {
    const int n_atom = static_cast<int>(owners.size());
    const auto cell_inv = vbcsr::atomic::InvertCell3x3(cell.data());
    std::vector<int> owner_of(grid.n_box(), -1);

    for (int box_index = 0; box_index < grid.n_box(); ++box_index) {
        const GridBox box = grid.BoxAt(box_index);
        std::vector<detail::BoxClaim> claims(n_rank);
        for (int r = 0; r < n_rank; ++r) claims[r].rank = -1;
        for (int a = 0; a < n_atom; ++a) {
            const double rcut = type_rcut[types[a]];
            const double distance = detail::MinimumImageDistance(
                box.center, positions[3 * a], positions[3 * a + 1], positions[3 * a + 2],
                cell, cell_inv, pbc);
            if (distance >= rcut + box.radius) continue;
            detail::BoxClaim& claim = claims[owners[a]];
            if (claim.rank < 0) {
                claim.rank = owners[a];
                claim.orbitals = 0;
                claim.distance = distance;
            }
            claim.orbitals += type_norb[types[a]];
            claim.distance = std::min(claim.distance, distance);
        }
        detail::BoxClaim winner;
        for (const detail::BoxClaim& claim : claims) {
            if (claim.Beats(winner)) winner = claim;
        }
        owner_of[box_index] = (winner.rank >= 0) ? winner.rank : (box_index % n_rank);
    }
    return owner_of;
}

namespace detail {

/// Calls `emit(owner_rank, local_atom)` once per (rank whose boxes this atom
/// reaches, owned atom) pair -- the demand side, read off the ownership the
/// same traversal produced.
///
/// Deduplication is free: ForEachBoxOfOwnedAtoms groups its emissions by atom,
/// so the owners of one atom's boxes arrive consecutively and a short running
/// list collapses them.
template <class EmitFn>
void ForEachBoxDemand(vbcsr::atomic::AtomicData& atomic_data, const RealSpaceGrid& grid,
                      const GridOwnership& ownership, const std::vector<double>& type_rcut,
                      EmitFn&& emit) {
    std::vector<int> owners;
    int current = -1;
    const auto flush = [&]() {
        if (current < 0) return;
        for (int owner : owners) emit(owner, current);
        owners.clear();
    };
    ForEachBoxOfOwnedAtoms(atomic_data, grid, type_rcut,
                           [&](int local, int box, double) {
        if (local != current) {
            flush();
            current = local;
        }
        const int owner = ownership.OwnerOf(box);
        if (owner < 0) {
            throw std::runtime_error(
                "A box this rank's atoms reach has no resolved owner; the ownership was "
                "built from a different geometry than the one being contracted.");
        }
        if (std::find(owners.begin(), owners.end(), owner) == owners.end()) {
            owners.push_back(owner);
        }
    });
    flush();
}

}  // namespace detail

/// How the demand-built set's atoms travel: the routing of the one alltoallv
/// that built the set, kept so the SAME routing can later ship data that
/// lives on the atoms -- coefficient rows of a Chebyshev-filtered state,
/// which the owner holds and the box-owning rank needs to contract.
///
/// Records are rank-major on both sides. send_local[i] is the sender's local
/// atom index of its i-th outgoing record; recv_slot[i] is the position in
/// the sorted OrbitalSet where the i-th incoming record landed. The plan is
/// geometry-fixed: build it once with the set, reuse it for every gather.
struct OrbitalSetExchangePlan {
    std::vector<int> send_counts;  ///< records per destination rank
    std::vector<int> send_local;   ///< local atom index per outgoing record
    std::vector<int> recv_counts;  ///< records per source rank
    std::vector<int> recv_slot;    ///< set slot per incoming record
};

/// The orbital set for this rank's boxes, built from the DEMAND side: exact
/// for any geometry, at the cost of one alltoall.
///
/// Reading the atoms a rank already holds -- owned plus graph ghosts -- would
/// be cheaper, but is sufficient only while those ghosts cover every box
/// overhang, and there are honest geometries where they do not: a box
/// straddling a cut plane next to a vacuum gap wider than the neighbour-list
/// cutoff is reached by foreign atoms that are nobody's ghosts. This builder
/// closes that hole by inverting the direction of knowledge: each atom's
/// owner enumerates the boxes the atom reaches (the same windowed walk and
/// the same overlap predicate as everywhere else) and ships the atom -- index,
/// type and position -- to those boxes' owners. Nothing needs to be near
/// anything for that to work, so coverage is exact by construction.
///
/// `rank` selects whose boxes the set is for; with a real communicator it
/// must be this process's rank in it. With MPI_COMM_NULL or one rank the
/// exchange degenerates to keeping own demands.
inline OrbitalSet BuildOrbitalSetFromDemands(vbcsr::atomic::AtomicData& atomic_data,
                                             const RealSpaceGrid& grid,
                                             const GridOwnership& ownership,
                                             const std::vector<double>& type_rcut,
                                             int rank, MPI_Comm comm,
                                             OrbitalSetExchangePlan* plan = nullptr) {
    int size = 1;
    if (comm != MPI_COMM_NULL) MPI_Comm_size(comm, &size);

    // One record per demand: (gid, type), (x, y, z) and the sender-local atom
    // index, same order throughout.
    std::vector<int> meta;
    std::vector<double> pos;
    std::vector<int> local_kept;
    const bool exchange = (comm != MPI_COMM_NULL && size > 1);
    std::vector<std::vector<int>> meta_out(exchange ? size : 0);
    std::vector<std::vector<double>> pos_out(exchange ? size : 0);
    std::vector<std::vector<int>> local_out(exchange ? size : 0);
    detail::ForEachBoxDemand(
        atomic_data, grid, ownership, type_rcut, [&](int owner, int local) {
            double ax = 0.0, ay = 0.0, az = 0.0;
            atomic_data.get_pos(local, &ax, &ay, &az);
            const int gid = atomic_data.get_global_index(local);
            const int type = atomic_data.atom_type[local];
            if (!exchange) {
                if (owner != rank) return;
                meta.insert(meta.end(), {gid, type});
                pos.insert(pos.end(), {ax, ay, az});
                local_kept.push_back(local);
                return;
            }
            if (owner < 0 || owner >= size) {
                throw std::runtime_error(
                    "A grid box classified to a rank outside the communicator; the cut tree "
                    "and the communicator disagree on the rank count.");
            }
            meta_out[owner].insert(meta_out[owner].end(), {gid, type});
            pos_out[owner].insert(pos_out[owner].end(), {ax, ay, az});
            local_out[owner].push_back(local);
        });

    std::vector<int> send_counts(size, 0), recv_counts(size, 0);
    if (exchange) {
        for (int r = 0; r < size; ++r) {
            send_counts[r] = static_cast<int>(meta_out[r].size() / 2);
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

        std::vector<int> sdis(size + 1, 0), rdis(size + 1, 0);
        for (int r = 0; r < size; ++r) {
            sdis[r + 1] = sdis[r] + send_counts[r];
            rdis[r + 1] = rdis[r] + recv_counts[r];
        }
        std::vector<int> send_meta(static_cast<size_t>(sdis[size]) * 2);
        std::vector<double> send_pos(static_cast<size_t>(sdis[size]) * 3);
        for (int r = 0; r < size; ++r) {
            std::copy(meta_out[r].begin(), meta_out[r].end(),
                      send_meta.begin() + static_cast<size_t>(sdis[r]) * 2);
            std::copy(pos_out[r].begin(), pos_out[r].end(),
                      send_pos.begin() + static_cast<size_t>(sdis[r]) * 3);
        }
        meta.assign(static_cast<size_t>(rdis[size]) * 2, 0);
        pos.assign(static_cast<size_t>(rdis[size]) * 3, 0.0);

        std::vector<int> sc(size), sd(size), rc(size), rd(size);
        for (int r = 0; r < size; ++r) {
            sc[r] = send_counts[r] * 2; sd[r] = sdis[r] * 2;
            rc[r] = recv_counts[r] * 2; rd[r] = rdis[r] * 2;
        }
        MPI_Alltoallv(send_meta.data(), sc.data(), sd.data(), MPI_INT,
                      meta.data(), rc.data(), rd.data(), MPI_INT, comm);
        for (int r = 0; r < size; ++r) {
            sc[r] = send_counts[r] * 3; sd[r] = sdis[r] * 3;
            rc[r] = recv_counts[r] * 3; rd[r] = rdis[r] * 3;
        }
        MPI_Alltoallv(send_pos.data(), sc.data(), sd.data(), MPI_DOUBLE,
                      pos.data(), rc.data(), rd.data(), MPI_DOUBLE, comm);

        if (plan != nullptr) {
            plan->send_local.clear();
            plan->send_local.reserve(sdis[size]);
            for (int r = 0; r < size; ++r) {
                plan->send_local.insert(plan->send_local.end(), local_out[r].begin(),
                                        local_out[r].end());
            }
        }
    } else {
        send_counts[0] = static_cast<int>(local_kept.size());
        recv_counts[0] = send_counts[0];
        if (plan != nullptr) plan->send_local = std::move(local_kept);
    }

    const size_t n_found = meta.size() / 2;
    std::vector<int> order(n_found);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return meta[2 * a] < meta[2 * b]; });

    if (plan != nullptr) {
        plan->send_counts = std::move(send_counts);
        plan->recv_counts = std::move(recv_counts);
        plan->recv_slot.assign(n_found, 0);
        for (size_t slot = 0; slot < n_found; ++slot) {
            plan->recv_slot[order[slot]] = static_cast<int>(slot);
        }
    }

    OrbitalSet set;
    set.atoms.reserve(n_found);
    set.atom_type.reserve(n_found);
    set.atom_norb.reserve(n_found);
    set.positions.reserve(3 * n_found);
    for (int index : order) {
        set.atoms.push_back(meta[2 * index + 0]);
        set.atom_type.push_back(meta[2 * index + 1]);
        set.atom_norb.push_back(atomic_data.type_norb[meta[2 * index + 1]]);
        set.positions.insert(set.positions.end(),
                             {pos[3 * index + 0], pos[3 * index + 1], pos[3 * index + 2]});
    }
    set.orb_offset.assign(set.atoms.size() + 1, 0);
    for (size_t i = 0; i < set.atoms.size(); ++i) {
        set.orb_offset[i + 1] = set.orb_offset[i] + set.atom_norb[i];
    }
    // Empty is legitimate for the same reason as the local builder: a vacuum
    // region's boxes are reached by nothing.
    return set;
}

/// Ships the set's coefficient rows to this rank through the plan's routing.
///
/// X holds each atom's orbital rows on the atom's OWNER, in the matrix
/// distribution; the contraction needs them on the BOX owner, in the set's
/// compact numbering. The plan already knows both endpoints of every record
/// -- it is the routing that built the set -- so the gather is one packing
/// pass, one alltoallv, one unpacking pass, with nothing recomputed. This is
/// the O(boundary) halo the whole real-space path budgets for; call it once
/// per filtered block, after the Chebyshev sweep.
///
/// coeffs_out is (set.n_orb(), n_vec) row-major over the set's numbering --
/// exactly what SquareVectorsOntoBoxes takes.
inline void GatherSetRows(vbcsr::atomic::AtomicData& atomic_data,
                          const vbcsr::DistMultiVector<double>& X,
                          const OrbitalSet& set,
                          const OrbitalSetExchangePlan& plan,
                          MPI_Comm comm,
                          std::vector<double>* coeffs_out) {
    if (coeffs_out == nullptr) {
        throw std::runtime_error("GatherSetRows needs an output pointer.");
    }
    int size = 1;
    if (comm != MPI_COMM_NULL) MPI_Comm_size(comm, &size);
    const bool exchange = (comm != MPI_COMM_NULL && size > 1);
    const int n_vec = X.num_vectors;

    // Scalar-row offset of each owned atom in the matrix distribution: the
    // graph's rows are the owned atoms' orbital blocks in local atom order.
    std::vector<int> row_offset(atomic_data.n_atom + 1, 0);
    for (int a = 0; a < atomic_data.n_atom; ++a) {
        row_offset[a + 1] = row_offset[a] + atomic_data.type_norb[atomic_data.atom_type[a]];
    }

    // Pack outgoing records: each record is its atom's (norb, n_vec) rows.
    std::vector<double> send_buf;
    std::vector<int> send_elems(size, 0);
    {
        size_t total = 0;
        size_t record = 0;
        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < plan.send_counts[r]; ++i, ++record) {
                const int local = plan.send_local[record];
                const int norb = row_offset[local + 1] - row_offset[local];
                send_elems[r] += norb * n_vec;
                total += static_cast<size_t>(norb) * n_vec;
            }
        }
        send_buf.resize(total);
        size_t at = 0;
        for (const int local : plan.send_local) {
            for (int row = row_offset[local]; row < row_offset[local + 1]; ++row) {
                const double* src = X.row_data(row);
                std::copy(src, src + n_vec, send_buf.begin() + at);
                at += n_vec;
            }
        }
    }

    std::vector<double> recv_buf;
    if (exchange) {
        std::vector<int> recv_elems(size, 0);
        // Receive sizes follow from the plan and the set: records are
        // rank-major, and each carries its set slot's orbital count.
        {
            size_t record = 0;
            for (int r = 0; r < size; ++r) {
                for (int i = 0; i < plan.recv_counts[r]; ++i, ++record) {
                    recv_elems[r] += set.atom_norb[plan.recv_slot[record]] * n_vec;
                }
            }
        }
        std::vector<int> sdis(size + 1, 0), rdis(size + 1, 0);
        for (int r = 0; r < size; ++r) {
            sdis[r + 1] = sdis[r] + send_elems[r];
            rdis[r + 1] = rdis[r] + recv_elems[r];
        }
        recv_buf.resize(rdis[size]);
        MPI_Alltoallv(send_buf.data(), send_elems.data(), sdis.data(), MPI_DOUBLE,
                      recv_buf.data(), recv_elems.data(), rdis.data(), MPI_DOUBLE, comm);
    } else {
        recv_buf = std::move(send_buf);
    }

    coeffs_out->assign(static_cast<size_t>(set.n_orb()) * n_vec, 0.0);
    size_t at = 0;
    for (const int slot : plan.recv_slot) {
        const int norb = set.atom_norb[slot];
        std::copy(recv_buf.begin() + at,
                  recv_buf.begin() + at + static_cast<size_t>(norb) * n_vec,
                  coeffs_out->begin() + static_cast<size_t>(set.orb_offset[slot]) * n_vec);
        at += static_cast<size_t>(norb) * n_vec;
    }
}

/// What the box loop actually did, for diagnosing scaling.
///
/// Linear scaling rests on a claim about per-box work: the orbital count
/// reaching a box is set by the cutoff and the density, so it must stay flat
/// as the cell grows. That is not visible in a timing -- a run that slowly
/// drifts to O(N) per box looks like a run that is merely slow -- so the loop
/// reports the sizes it saw and a scaling study can watch them directly.
struct GridWorkStats {
    int64_t boxes_contracted = 0;  ///< Boxes with a non-empty active set.
    int64_t boxes_empty = 0;       ///< Boxes no orbital reached; skipped.
    int max_box_orbitals = 0;      ///< Largest local orbital count over boxes.
    int64_t sum_box_orbitals = 0;  ///< Total, for the mean per contracted box.
    int64_t points_kept = 0;       ///< Rows surviving the exact-zero trim.
    int64_t points_total = 0;      ///< Rows before it.

    void Merge(const GridWorkStats& other) {
        boxes_contracted += other.boxes_contracted;
        boxes_empty += other.boxes_empty;
        max_box_orbitals = std::max(max_box_orbitals, other.max_box_orbitals);
        sum_box_orbitals += other.sum_box_orbitals;
        points_kept += other.points_kept;
        points_total += other.points_total;
    }
};

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
                "whose basis has radial functions, which today means LCAO.");
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
    /// The output is COMPACT -- one value per owned point, in the domain's box
    /// order -- not a full grid with holes. Owned boxes are disjoint (one
    /// referee settles each box, see ResolveGridOwnership), so the global result is the
    /// concatenation of the ranks' buffers and no reduction is needed at all.
    /// A full-grid buffer would instead cost every rank the whole result: ~5.6
    /// GB per rank at 10^6 atoms and 0.2 A, plus a grid-sized allreduce, to
    /// hold something each rank owns 1/P of. rescu++ reaches the same layout
    /// from the same argument (accumulate_density_packed returns
    /// (nspin, n_owned_points), "no reduce (owned points are disjoint)").
    ///
    /// Phi is evaluated per box and discarded rather than cached. That is
    /// rescu++'s streaming mode, and it is the right default here: it holds
    /// flat O(box) memory instead of O(N), which is what a 10^6-atom cell
    /// needs, and the op2c evaluator underneath is the same one.
    ///
    /// Args:
    ///   atomic_data: Supplies cell and boundary conditions for imaging.
    ///   set: Atoms and orbital numbering the block is indexed by.
    ///   block: Row-major n_orb x n_orb Hermitian block over `set`.
    ///   grid: The grid whose boxes are being filled.
    ///   domain: This rank's owned boxes and their offsets.
    ///   values_out: domain.n_point() values, in owned-box order.
    void ContractOntoBoxes(vbcsr::atomic::AtomicData& atomic_data,
                           const OrbitalSet& set,
                           const std::vector<std::complex<double>>& block,
                           const RealSpaceGrid& grid,
                           const GridDomain& domain,
                           std::vector<double>* values_out) const {
        if (values_out == nullptr) {
            throw std::runtime_error("ContractOntoBoxes needs an output pointer.");
        }
        const int n_orb = set.n_orb();
        if (block.size() != static_cast<size_t>(n_orb) * n_orb) {
            throw std::runtime_error("Block size does not match the orbital-set size.");
        }
        RequireUniqueMinimumImage(atomic_data, set);
        ValidateSetAgainstEvaluators(set);

        std::vector<double> block_real(block.size());
        for (size_t i = 0; i < block.size(); ++i) block_real[i] = block[i].real();

        values_out->assign(static_cast<size_t>(domain.n_point()), 0.0);
        const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());

        // One thread per box, dynamic because the active count varies wildly
        // (a vacuum box is near-free). Boxes write disjoint slices of the
        // compact buffer, so there is nothing to reduce. BLAS goes sequential
        // on its own inside the region, which is the one-GEMM-per-thread shape
        // rescu++'s threaded box loop uses. Scratch is per-thread and O(box).
#pragma omp parallel
        {
            std::vector<double> points, phi, product, relative, local_block, box_values;
            std::vector<int> active, kept;

#pragma omp for schedule(dynamic)
            for (int slot = 0; slot < domain.n_box(); ++slot) {
                const GridBox box = grid.BoxAt(domain.boxes[slot]);
                const int n_box_point = box.n_point();
                if (n_box_point == 0) continue;
                double* out = values_out->data() + domain.offset[slot];

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
                const int n_kept = detail::TrimZeroRows(n_box_point, n_local_orb, &phi, &kept);
                if (n_kept == 0) continue;  // no orbital reaches any point of this box
                product.resize(static_cast<size_t>(n_kept) * n_local_orb);
                box_values.assign(n_kept, 0.0);
                ContractChunk(n_kept, n_local_orb, phi.data(), local_block.data(),
                              product.data(), box_values.data());

                // Trimmed rows are exactly zero and values_out was zeroed, so
                // only the kept ones are written back.
                const double per_angstrom_cubed = model::lcao::kBohrPerAngstrom *
                                                  model::lcao::kBohrPerAngstrom *
                                                  model::lcao::kBohrPerAngstrom;
                for (int q = 0; q < n_kept; ++q) {
                    out[kept[q]] = box_values[q] * per_angstrom_cubed;
                }
            }
        }
    }

    /// Accumulates mean_v |sum_i c_i^(v) phi_i(r)|^2 on this rank's boxes.
    ///
    /// Route (5.3b), "square-on-grid", of doc/design_stochastic_ldos.md 5.3:
    /// the real-space state of each sample is evaluated and squared pointwise,
    /// and the n_orb x n_orb object is never formed. The two routes agree
    /// exactly rather than approximately -- for one vector c,
    ///
    ///     sum_ij (c_i c_j) phi_i(r) phi_j(r) = ( sum_i c_i phi_i(r) )^2
    ///
    /// is an algebraic identity, so this equals ContractOntoBoxes fed the outer
    /// product c c^T with no sampling error to hide behind. The test asserts
    /// exactly that.
    ///
    /// Why it is the route for the stochastic LDOS. Per box the work is one
    /// GEMM of (n_point x n_local_orb) against (n_local_orb x n_vec), i.e.
    /// linear in the sample count where the two-index route is quadratic in
    /// the orbital count; (5.3b) wins on flops whenever n_vec is below
    /// n_local_orb, which for a few hundred samples against a few hundred
    /// local orbitals is the usual case. On memory it wins unconditionally: it
    /// holds n_local_orb x n_vec coefficients instead of an n_orb^2 block,
    /// which at 10^6 atoms is the difference between kilobytes and 10^14
    /// entries that could not be stored at all.
    ///
    /// Args:
    ///   coeffs: Row-major (n_orb, n_vec) coefficients over `set`'s numbering.
    ///   n_vec: Number of sample vectors; the mean is taken over them.
    ///   values_out: domain.n_point() values, in owned-box order.
    void SquareVectorsOntoBoxes(vbcsr::atomic::AtomicData& atomic_data,
                                const OrbitalSet& set,
                                const std::vector<double>& coeffs,
                                int n_vec,
                                const RealSpaceGrid& grid,
                                const GridDomain& domain,
                                std::vector<double>* values_out,
                                GridWorkStats* stats = nullptr) const {
        if (values_out == nullptr) {
            throw std::runtime_error("SquareVectorsOntoBoxes needs an output pointer.");
        }
        if (n_vec <= 0) {
            throw std::runtime_error("SquareVectorsOntoBoxes needs at least one vector.");
        }
        const int n_orb = set.n_orb();
        if (coeffs.size() != static_cast<size_t>(n_orb) * n_vec) {
            throw std::runtime_error("Coefficient array does not match (n_orb, n_vec).");
        }
        RequireUniqueMinimumImage(atomic_data, set);
        ValidateSetAgainstEvaluators(set);

        values_out->assign(static_cast<size_t>(domain.n_point()), 0.0);
        const auto cell_inv = vbcsr::atomic::InvertCell3x3(atomic_data.cell.data());
        const double per_angstrom_cubed = model::lcao::kBohrPerAngstrom *
                                          model::lcao::kBohrPerAngstrom *
                                          model::lcao::kBohrPerAngstrom;
        const double inv_n_vec = 1.0 / static_cast<double>(n_vec);

        // Threaded exactly like ContractOntoBoxes: disjoint output slices,
        // per-thread O(box) scratch, dynamic schedule over uneven boxes.
#pragma omp parallel
        {
            std::vector<double> points, phi, local_coeff, psi, relative;
            std::vector<int> active, kept, col_ids;
            std::vector<char> col_active;
            GridWorkStats mine;

#pragma omp for schedule(dynamic)
            for (int slot = 0; slot < domain.n_box(); ++slot) {
                const GridBox box = grid.BoxAt(domain.boxes[slot]);
                const int n_box_point = box.n_point();
                if (n_box_point == 0) continue;
                double* out = values_out->data() + domain.offset[slot];

                const LocalView local =
                    ViewForSphere(atomic_data, set, box.center, box.radius, cell_inv);
                const int n_local_orb = static_cast<int>(local.orbitals.size());
                if (n_local_orb == 0) {  // vacuum box: stays zero
                    ++mine.boxes_empty;
                    continue;
                }

                // Columns of the block with any weight on this box's rows.
                // Random sample blocks are dense and settle the scan after a
                // row or two; spatially local columns (the identity-streamed
                // fragment baseline of rs_ldos) leave most boxes with a
                // fraction of the block or none of it, and the GEMM -- and
                // for untouched boxes the orbital evaluation itself -- must
                // not be paid for exact zeros. The skip is what holds the
                // baseline stream at O(N): active (block, box) pairs are set
                // by the overlap geometry, not by the basis size.
                col_active.assign(n_vec, 0);
                int n_active = 0;
                for (int i = 0; i < n_local_orb && n_active < n_vec; ++i) {
                    const double* src =
                        coeffs.data() + static_cast<size_t>(local.orbitals[i]) * n_vec;
                    for (int v = 0; v < n_vec; ++v) {
                        if (!col_active[v] && src[v] != 0.0) {
                            col_active[v] = 1;
                            ++n_active;
                        }
                    }
                }
                if (n_active == 0) {  // no column reaches this box: stays zero
                    ++mine.boxes_empty;
                    continue;
                }
                ++mine.boxes_contracted;
                mine.max_box_orbitals = std::max(mine.max_box_orbitals, n_local_orb);
                mine.sum_box_orbitals += n_local_orb;
                mine.points_total += n_box_point;

                // Gather this box's rows of the coefficient array, compacted
                // to the active columns when some are dead.
                const bool dense_cols = (n_active == n_vec);
                if (!dense_cols) {
                    col_ids.clear();
                    for (int v = 0; v < n_vec; ++v) {
                        if (col_active[v]) col_ids.push_back(v);
                    }
                }
                local_coeff.resize(static_cast<size_t>(n_local_orb) * n_active);
                for (int i = 0; i < n_local_orb; ++i) {
                    const double* src =
                        coeffs.data() + static_cast<size_t>(local.orbitals[i]) * n_vec;
                    double* dst = local_coeff.data() + static_cast<size_t>(i) * n_active;
                    if (dense_cols) {
                        std::copy(src, src + n_vec, dst);
                    } else {
                        for (int a = 0; a < n_active; ++a) dst[a] = src[col_ids[a]];
                    }
                }

                grid.PointsOfBox(box, &points);
                phi.assign(static_cast<size_t>(n_box_point) * n_local_orb, 0.0);

                EvaluatePhi(atomic_data, set, local, points, n_box_point, cell_inv, &relative,
                            &active, phi.data());
                const int n_kept = detail::TrimZeroRows(n_box_point, n_local_orb, &phi, &kept);
                mine.points_kept += n_kept;
                if (n_kept == 0) continue;
                psi.resize(static_cast<size_t>(n_kept) * n_active);

                // psi = phi * local_coeff, both row-major. In Fortran layout
                // that is local_coeff^T * phi^T with no transposition, the
                // same trick ContractChunk uses.
                {
                    using vbcsr::vbcsr_lapack_int;
                    const vbcsr_lapack_int m = n_active;
                    const vbcsr_lapack_int n = n_kept;
                    const vbcsr_lapack_int k = n_local_orb;
                    const double alpha = 1.0, beta = 0.0;
                    vbcsr::dgemm_("N", "N", &m, &n, &k, &alpha, local_coeff.data(), &m,
                                  phi.data(), &k, &beta, psi.data(), &m);
                }

                // The dead columns contribute exactly zero to the square sum,
                // so the mean keeps its full n_vec normalization.
                for (int q = 0; q < n_kept; ++q) {
                    const double* row = psi.data() + static_cast<size_t>(q) * n_active;
                    double sum = 0.0;
                    for (int v = 0; v < n_active; ++v) sum += row[v] * row[v];
                    out[kept[q]] = sum * inv_n_vec * per_angstrom_cubed;
                }
            }
            if (stats != nullptr) {
#pragma omp critical
                stats->Merge(mine);
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
    /// Per-box internals, exposed so a test can measure what the box loop does
    /// rather than only what it returns -- how much of a box the orbitals
    /// actually cover is not visible in the contracted output.
    LocalView ViewForSphereForTest(vbcsr::atomic::AtomicData& atomic_data, const OrbitalSet& set,
                                   const double* center, double radius,
                                   const std::array<double, 9>& cell_inv) const {
        return ViewForSphere(atomic_data, set, center, radius, cell_inv);
    }
    void EvaluatePhiForTest(vbcsr::atomic::AtomicData& atomic_data, const OrbitalSet& set,
                            const LocalView& local, const std::vector<double>& points,
                            int n_point, const std::array<double, 9>& cell_inv,
                            std::vector<double>* relative, std::vector<int>* active,
                            double* phi) const {
        EvaluatePhi(atomic_data, set, local, points, n_point, cell_inv, relative, active, phi);
    }

private:
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

    /// Raises, BEFORE a parallel box loop starts, everything the per-box work
    /// could otherwise raise from inside it -- a type with no orbital file, or
    /// an orbital count that disagrees with the Hamiltonian's. An exception
    /// escaping an OpenMP region terminates the program instead of
    /// propagating, so the loops must be entered exception-free.
    void ValidateSetAgainstEvaluators(const OrbitalSet& set) const {
        for (int a = 0; a < set.n_atom(); ++a) {
            if (Evaluator(set.atom_type[a]).nphi() != set.atom_norb[a]) {
                throw std::runtime_error(
                    "Orbital file and Hamiltonian disagree on the orbital count of a type; "
                    "the contraction would mix unrelated orbitals.");
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
