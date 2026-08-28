#ifndef RSATB_CALC_PDOS_HPP
#define RSATB_CALC_PDOS_HPP

// Projected DOS moments: the KPM trace moments of dos.hpp, resolved onto
// orbital groups by a segmented reduction of the same recursion vectors.
//
//     mu_m^g = E[ psi^H P_g T_m(Op) psi ],   P_g = sum_{i in g} |i><i|
//
// P_g is diagonal, so the projection is a pointwise conj(psi) (.) t_m followed
// by a per-group row reduction -- O(rows x samples) per moment against the
// O(nnz x samples) matvec that produced t_m. Every grouping of the same run
// shares the recursion; adding groups adds reductions only.
//
// WHICH population the cheap masking yields depends on the operator form, and
// the two are different, standard, conventions -- not two routes to one number:
//
//   Sandwich (Z^H H Z, Hermitian):  mu_m^g sums the LOWDIN populations
//       sum_{i in g} |(S^{1/2} C)_{in}|^2. The orthogonalized orbital i IS the
//       Lowdin orbital, so diagonal masking is the Lowdin projection.
//   Solve (S^-1 H, plain Rademacher seed):  [T_m(A)]_{ii} pairs C_{in} with
//       (S C)_{in}^*, so mu_m^g sums the MULLIKEN populations; take the real
//       part (Mulliken weights are Re[C (SC)^*]; the imaginary parts cancel
//       only in the total). Mulliken populations can legitimately be negative.
//
// Crossing the conventions (Lowdin from the solve form, Mulliken from the
// sandwich) costs extra S/Z applications per moment and is deliberately not
// offered here.
//
// Moment doubling: in the sandwich form the masked doubling
//     mu_{2n}^g = 2 <t_n| P_g |t_n> - mu_0^g
// is exact for the same reason the unmasked one is -- 2 T_m T_n =
// T_{m+n} + T_{|m-n|} is an operator identity and polynomials of one operator
// commute, so the mask inserts cleanly under the trace. In the solve form it
// would NOT be valid (T_m(A)^H = S T_m(A) S^-1 != T_m(A)), and the loop below
// follows cheby.op_is_hermitian() exactly as kpm_dos_moments does, so the
// solve form always takes the direct pairing.
//
// The sum rule sum_g mu_m^g = mu_m^total holds moment by moment for any
// partition of the rows, and holds here to rounding because the total is
// accumulated by the same segmented kernel as an implicit all-rows group.

#include "cheby.hpp"
#include "../utils/interrupt.hpp"
#include "../utils/profile.hpp"
#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/scalar_traits.hpp"

#include <complex>
#include <stdexcept>
#include <vector>

#include <mpi.h>

namespace rsatb {
namespace calc {

/// Which rows belong to which group. `group_of_row` is indexed by LOCAL
/// scalar row and holds a group id in [0, n_groups) or -1 for "no group".
/// Group ids are global: the same physical group must carry the same id on
/// every rank, since the reduction sums buckets across ranks.
struct PdosGroups {
    int n_groups = 0;
    std::vector<int> group_of_row;
    /// Per-group descriptors, size n_groups when the grouping resolves atom
    /// types and empty otherwise (the by-atom grouping's id IS its label):
    /// the atom type and the orbital offset within that type's atom block.
    /// Without them a multi-type run cannot say which curve is which species.
    std::vector<int> group_type;
    std::vector<int> group_slot;
};

/// One group per atom, id = the atom's ORIGINAL index from the input
/// (AtomicData::atom_index) -- stable across repartitioning, the same choice
/// observables.hpp makes for the sample seeds.
inline PdosGroups PdosGroupsByAtom(const vbcsr::atomic::AtomicData& ad, MPI_Comm comm) {
    PdosGroups g;
    int local_max = -1;
    for (int a = 0; a < ad.n_atom; ++a) local_max = std::max(local_max, ad.atom_index[a]);
    int global_max = local_max;
    if (comm != MPI_COMM_NULL) {
        MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, comm);
    }
    g.n_groups = global_max + 1;
    for (int a = 0; a < ad.n_atom; ++a) {
        const int norb = ad.type_norb[ad.atom_type[a]];
        g.group_of_row.insert(g.group_of_row.end(), norb, ad.atom_index[a]);
    }
    return g;
}

/// One group per (atom TYPE, orbital slot) pair: rows at offset o of every atom
/// of type t share a group, and different types never share one.
///
/// Slot alone is wrong as soon as two types are present -- slot 3 of a 2s2p1d
/// carbon is a p orbital while slot 3 of a smaller hydrogen basis is something
/// else, so merging them projects onto nothing physical. With ONE type this is
/// exactly the old slot grouping (offset[0] = 0, n_groups = type_norb[0], group
/// id == slot), which is why single-species results are unaffected.
///
/// No communication: type_norb is global metadata replicated on every rank when
/// the AtomicData is built, so every rank derives the same offsets from it.
///
/// `group_type` indexes the ascending list of distinct atomic numbers, the same
/// ordering the per-type input tables use, so the caller recovers the element
/// from its own structure. Mapping slot to a SHELL stays with the caller too:
/// the atomic metadata carries orbital counts, not angular momenta.
inline PdosGroups PdosGroupsByOrbitalSlot(const vbcsr::atomic::AtomicData& ad) {
    const int n_type = ad.n_type();
    std::vector<int> offset(static_cast<size_t>(n_type) + 1, 0);
    for (int t = 0; t < n_type; ++t) offset[t + 1] = offset[t] + ad.type_norb[t];

    PdosGroups g;
    g.n_groups = offset[n_type];
    g.group_type.resize(static_cast<size_t>(g.n_groups));
    g.group_slot.resize(static_cast<size_t>(g.n_groups));
    for (int t = 0; t < n_type; ++t) {
        for (int o = 0; o < ad.type_norb[t]; ++o) {
            g.group_type[static_cast<size_t>(offset[t] + o)] = t;
            g.group_slot[static_cast<size_t>(offset[t] + o)] = o;
        }
    }
    for (int a = 0; a < ad.n_atom; ++a) {
        const int t = ad.atom_type[a];
        for (int o = 0; o < ad.type_norb[t]; ++o) g.group_of_row.push_back(offset[t] + o);
    }
    return g;
}

/// Trace moments and their group-projected split from one recursion.
/// `group` is row-major [g * n_moments + m]. `total` is the all-rows channel,
/// accumulated by the same kernel -- NOT the sum over groups, so a partition
/// can be checked against it.
struct PdosMoments {
    int n_groups = 0;
    int n_moments = 0;
    std::vector<std::complex<double>> total;
    std::vector<std::complex<double>> group;
    std::complex<double> at(int g, int m) const {
        return group[static_cast<size_t>(g) * n_moments + m];
    }
};

namespace detail {

/// raw[slot] += sum_cols conj(a(row, c)) b(row, c), slot = 1 + group_of_row
/// (slot 0 is the all-rows channel). LOCAL accumulation only; the caller does
/// one deferred Allreduce over all moments at the end, because the doubling
/// transforms are affine in the reduced dots and commute with the sum.
template <typename T>
void pdos_segmented_accumulate(const vbcsr::DistMultiVector<T>& a,
                               const vbcsr::DistMultiVector<T>& b,
                               const PdosGroups& groups,
                               std::complex<double>* raw) {
    const int nv = a.num_vectors;
    for (int row = 0; row < a.local_rows; ++row) {
        const T* ra = a.row_data(row);
        const T* rb = b.row_data(row);
        std::complex<double> acc(0.0, 0.0);
        for (int c = 0; c < nv; ++c) {
            acc += static_cast<std::complex<double>>(
                vbcsr::ScalarTraits<T>::conjugate(ra[c]) * rb[c]);
        }
        raw[0] += acc;
        const int g = groups.group_of_row[row];
        if (g >= 0) raw[1 + g] += acc;
    }
}

}  // namespace detail

/// KPM moments with per-group projection. Mirrors kpm_dos_moments exactly --
/// the same recursion, the same Hermiticity-gated doubling, the same
/// column-averaging convention (divide by num_vectors; a complete-basis
/// caller multiplies back by the state count to convert mean to trace).
/// Collective on the graph communicator (one Allreduce at the end).
template <typename T>
PdosMoments kpm_pdos_moments(Chebyshev<T>& cheby,
                             const vbcsr::DistMultiVector<T>& psi_in,
                             int n_moments, const PdosGroups& groups) {
    using vbcsr::DistMultiVector;
    vbcsr::DistGraph* graph = psi_in.graph;
    const int rank = graph->rank;
    if (static_cast<int>(groups.group_of_row.size()) != psi_in.local_rows) {
        throw std::runtime_error(
            "pdos: group_of_row must map every local scalar row (got " +
            std::to_string(groups.group_of_row.size()) + " for " +
            std::to_string(psi_in.local_rows) + " rows)");
    }
    for (const int g : groups.group_of_row) {
        if (g >= groups.n_groups) {
            throw std::runtime_error("pdos: group id exceeds n_groups");
        }
    }

    const int width = 1 + groups.n_groups;  // slot 0: all rows
    // raw[m * width + slot]: local segment dots, reduced once at the end.
    std::vector<std::complex<double>> raw(
        static_cast<size_t>(n_moments) * width, std::complex<double>(0.0, 0.0));
    auto raw_m = [&](int m) { return raw.data() + static_cast<size_t>(m) * width; };

    DistMultiVector<T> t0(graph, psi_in.num_vectors), t1(graph, psi_in.num_vectors),
        t2(graph, psi_in.num_vectors);
    t0.copy_from(psi_in);
    detail::pdos_segmented_accumulate(t0, t0, groups, raw_m(0));

    if (n_moments > 1) {
        cheby.apply_op(t0, t1, ChebyMode::NORMAL);
        detail::pdos_segmented_accumulate(t0, t1, groups, raw_m(1));

        const bool doubled = cheby.op_is_hermitian();
        const int stride =
            std::max(1, (doubled ? n_moments / 2 : n_moments) / 20);

        if (doubled) {
            for (int n = 1;; ++n) {
                ::rsatb::util::throw_if_interrupted(graph->comm);
                if (2 * n - 1 < n_moments && 2 * n - 1 > 1) {
                    detail::pdos_segmented_accumulate(t1, t0, groups, raw_m(2 * n - 1));
                }
                if (2 * n < n_moments) {
                    detail::pdos_segmented_accumulate(t1, t1, groups, raw_m(2 * n));
                }
                if (2 * n + 1 >= n_moments) break;
                if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
                    if (::rsatb::util::every_nth(n - 1, stride))
                        std::cout << "pdos: moment " << 2 * n << " / " << n_moments
                                  << std::endl;
                }
                cheby.apply_op(t1, t2, ChebyMode::NORMAL);
                t2.axpby(-1.0, t0, 2.0);
                t0.swap(t1);
                t1.swap(t2);
            }
        } else {
            for (int n = 2; n < n_moments; ++n) {
                ::rsatb::util::throw_if_interrupted(graph->comm);
                if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
                    if (::rsatb::util::every_nth(n - 2, stride))
                        std::cout << "pdos: moment " << n << " / " << n_moments
                                  << std::endl;
                }
                cheby.apply_op(t1, t2, ChebyMode::NORMAL);
                t2.axpby(-1.0, t0, 2.0);
                detail::pdos_segmented_accumulate(psi_in, t2, groups, raw_m(n));
                t0.swap(t1);
                t1.swap(t2);
            }
        }
    }

    // ONE reduction for the whole run. Safe to defer past the doubling
    // because the doubling transform below is affine in the reduced dots.
    if (graph->comm != MPI_COMM_NULL && graph->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, raw.data(), static_cast<int>(raw.size()),
                      MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, graph->comm);
    }

    const double inv_nv = 1.0 / static_cast<double>(psi_in.num_vectors);
    for (auto& c : raw) c *= inv_nv;

    PdosMoments out;
    out.n_groups = groups.n_groups;
    out.n_moments = n_moments;
    out.total.assign(n_moments, std::complex<double>(0.0, 0.0));
    out.group.assign(static_cast<size_t>(groups.n_groups) * n_moments,
                     std::complex<double>(0.0, 0.0));

    const bool doubled = (n_moments > 1) && cheby.op_is_hermitian();
    auto transform = [&](auto get_raw, auto put) {
        // mu[0] = raw[0]; mu[1] = raw[1]; doubled m >= 2:
        // mu[m] = 2 raw[m] - mu[m & 1]  (even pairs with mu[0], odd with mu[1])
        std::complex<double> mu0 = get_raw(0), mu1(0.0, 0.0);
        put(0, mu0);
        if (n_moments > 1) {
            mu1 = get_raw(1);
            put(1, mu1);
        }
        for (int m = 2; m < n_moments; ++m) {
            const std::complex<double> r = get_raw(m);
            put(m, doubled ? 2.0 * r - ((m % 2 == 0) ? mu0 : mu1) : r);
        }
    };

    transform([&](int m) { return raw_m(m)[0]; },
              [&](int m, std::complex<double> v) { out.total[m] = v; });
    for (int g = 0; g < groups.n_groups; ++g) {
        transform([&](int m) { return raw_m(m)[1 + g]; },
                  [&](int m, std::complex<double> v) {
                      out.group[static_cast<size_t>(g) * n_moments + m] = v;
                  });
    }
    return out;
}

/// Single-vector overload: copies into a one-column multivector. The copy is
/// O(rows) against a recursion of O(nnz) per moment.
template <typename T>
PdosMoments kpm_pdos_moments(Chebyshev<T>& cheby, const vbcsr::DistVector<T>& psi_in,
                             int n_moments, const PdosGroups& groups) {
    vbcsr::DistMultiVector<T> mv(psi_in.graph, 1);
    for (int i = 0; i < mv.local_rows; ++i) mv(i, 0) = psi_in[i];
    return kpm_pdos_moments(cheby, mv, n_moments, groups);
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_PDOS_HPP
