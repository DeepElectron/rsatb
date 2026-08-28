// Projected DOS moments (calc/pdos.hpp) against a dense reference, both
// operator forms.
//
// The complete-basis sweep makes the stochastic estimator exact, so the
// comparison against the dense Chebyshev recursion is a correctness test of
// the projection and the doubling, not a statistical one. The dense side
// shares only Z with the driver; the recursion, the masking and the doubling
// are all independent code.
#undef NDEBUG

#include "../pdos.hpp"
#include "../isqrtm.hpp"
#include "../preconditioner.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using namespace vbcsr;

extern "C" {
void dposv_(const char* uplo, const int* n, const int* nrhs, double* a, const int* lda,
            double* b, const int* ldb, int* info);
}

namespace {

constexpr int kAtoms = 16;
constexpr int kNorb = 3;
constexpr double kLo = -12.0, kHi = 12.0;  // generous pencil bounds

double SymEntry(int p, int q) {
    if (p > q) std::swap(p, q);
    double v = 0.7 * std::sin(0.31 * p + 0.17 * q) * std::cos(0.05 * p * q);
    if (p == q) v += 0.4 + 0.03 * p;
    return v;
}

/// Two species in one cell: the (species, slot) grouping must not merge them.
///
/// A grouping keyed on the orbital offset ALONE puts slot 1 of a 2-orbital
/// species and slot 1 of a 4-orbital species in one bucket, producing a curve
/// that is not a projection onto anything. With a single species the two
/// groupings coincide exactly -- which is why the rest of this file, a
/// one-species chain, cannot tell them apart.
void CheckMultiSpeciesGrouping(int rank) {
    constexpr int kNa = 12;  // alternating C and H
    std::vector<double> pos;
    std::vector<int> z;
    for (int i = 0; i < kNa; ++i) {
        pos.insert(pos.end(), {2.0 * i, 0.0, 0.0});
        z.push_back(i % 2 == 0 ? 6 : 1);
    }
    const std::vector<double> cell = {2.0 * kNa + 10.0, 0, 0, 0, 10, 0, 0, 0, 10};
    // Types are the distinct atomic numbers ASCENDING, so type 0 is H and
    // type 1 is C; the per-type tables below follow that order.
    const std::vector<int> type_norb = {2, 4};
    std::unique_ptr<atomic::AtomicData> ad(atomic::AtomicData::from_points(
        pos, z, cell, {false, false, false}, {1.1, 1.1}, type_norb, MPI_COMM_WORLD));

    const auto by_slot = rsatb::calc::PdosGroupsByOrbitalSlot(*ad);
    // 2 + 4, NOT max(2, 4): every (species, slot) pair is its own group.
    assert(by_slot.n_groups == 6);
    assert(static_cast<int>(by_slot.group_type.size()) == 6);
    assert(static_cast<int>(by_slot.group_slot.size()) == 6);

    // Types are the distinct atomic numbers ascending, so type 0 is H and
    // type 1 is C; groups run type by type in that order.
    const std::vector<int> want_type = {0, 0, 1, 1, 1, 1};
    const std::vector<int> want_slot = {0, 1, 0, 1, 2, 3};
    for (int g = 0; g < 6; ++g) {
        assert(by_slot.group_type[g] == want_type[g]);
        assert(by_slot.group_slot[g] == want_slot[g]);
    }

    // Every row must land in a group whose species is the row's own, and whose
    // slot is the row's offset inside its atom block.
    size_t row = 0;
    for (int a = 0; a < ad->n_atom; ++a) {
        const int t = ad->atom_type[a];
        const int norb = ad->type_norb[t];
        for (int o = 0; o < norb; ++o, ++row) {
            const int gid = by_slot.group_of_row[row];
            assert(by_slot.group_type[gid] == t);
            assert(by_slot.group_slot[gid] == o);
        }
    }
    assert(row == by_slot.group_of_row.size());

    // The grouping derives its ids from type_norb WITHOUT communicating, which
    // is only sound because that table is global. Assert the property here,
    // where it is free, rather than paying for a reduction on every call: all
    // ranks must agree on the group count and on every label.
    int lo = by_slot.n_groups, hi = by_slot.n_groups;
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    assert(lo == hi && lo == by_slot.n_groups);
    std::vector<int> lab = by_slot.group_type, lab_max = by_slot.group_type;
    MPI_Allreduce(MPI_IN_PLACE, lab_max.data(), 6, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    assert(lab == lab_max);

    if (rank == 0) {
        std::printf("multi-species grouping: 6 groups (H:2 + C:4), no cross-species merge\n");
    }
}

struct Chain {
    std::unique_ptr<atomic::AtomicData> ad;
    std::unique_ptr<BlockSpMat<double>> S, H, Zt;

    Chain() {
        std::vector<double> pos;
        for (int i = 0; i < kAtoms; ++i) pos.insert(pos.end(), {2.0 * i, 0.0, 0.0});
        const std::vector<int> z(kAtoms, 6);
        const std::vector<double> cell = {2.0 * kAtoms + 10.0, 0, 0, 0, 10, 0, 0, 0, 10};
        // rcut 1.1: neighbours within 2.2 A -> nearest-neighbour chain.
        ad.reset(atomic::AtomicData::from_points(pos, z, cell, {false, false, false},
                                                 {1.1}, {kNorb}, MPI_COMM_WORLD));
        DistGraph* g = ad->get_graph();

        S.reset(new BlockSpMat<double>(g));
        H.reset(new BlockSpMat<double>(g));
        const auto& row_ptr = S->row_ptr();
        const auto& col_ind = S->col_ind();
        const int n_rows = static_cast<int>(row_ptr.size()) - 1;
        std::vector<double> s_buf(kNorb * kNorb), h_buf(kNorb * kNorb);
        for (int i = 0; i < n_rows; ++i) {
            const int ga = g->get_global_index(i);
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int gb = g->get_global_index(col_ind[k]);
                for (int a = 0; a < kNorb; ++a) {
                    for (int b = 0; b < kNorb; ++b) {
                        const int p = ga * kNorb + a, q = gb * kNorb + b;
                        s_buf[a * kNorb + b] = (p == q) ? 1.0 : (ga == gb ? 0.0 : 0.15 * (a == b));
                        h_buf[a * kNorb + b] = SymEntry(p, q);
                    }
                }
                S->add_block(ga, gb, s_buf.data(), kNorb, kNorb, AssemblyMode::INSERT,
                             MatrixLayout::RowMajor);
                H->add_block(ga, gb, h_buf.data(), kNorb, kNorb, AssemblyMode::INSERT,
                             MatrixLayout::RowMajor);
            }
        }
        S->assemble();
        H->assemble();

        Zt.reset(new BlockSpMat<double>(S->duplicate(true)));
        rsatb::calc::inverse_sqrt(*S, Zt.get(), {1e-10, 1e-11, 50});
    }

    static int n() { return kAtoms * kNorb; }

    /// Dense M in global scalar indexing, summed over ranks (each holds
    /// disjoint owned rows).
    static std::vector<double> Densify(const BlockSpMat<double>& M) {
        const int N = n();
        std::vector<double> out(static_cast<size_t>(N) * N, 0.0);
        const DistGraph* g = M.graph;
        const auto& row_ptr = M.row_ptr();
        const auto& col_ind = M.col_ind();
        const int n_owned = static_cast<int>(g->owned_global_indices.size());
        for (int i = 0; i < n_owned; ++i) {
            const int ga = g->get_global_index(i);
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int gb = g->get_global_index(col_ind[k]);
                const double* blk = M.block_data(k);
                for (int a = 0; a < kNorb; ++a)
                    for (int b = 0; b < kNorb; ++b)
                        out[static_cast<size_t>(ga * kNorb + a) * N + gb * kNorb + b] =
                            blk[a * kNorb + b];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, out.data(), static_cast<int>(out.size()), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        return out;
    }
};

std::vector<double> MatMul(const std::vector<double>& A, const std::vector<double>& B, int N) {
    std::vector<double> C(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            const double a = A[static_cast<size_t>(i) * N + k];
            if (a == 0.0) continue;
            for (int j = 0; j < N; ++j) C[static_cast<size_t>(i) * N + j] += a * B[static_cast<size_t>(k) * N + j];
        }
    return C;
}

/// Group-resolved trace moments of the dense scaled operator: for each m,
/// ref[g][m] = sum_{i in g} [T_m(A_scaled)]_{ii}. Plain three-term recursion
/// on dense matrices -- shares nothing with the driver's doubled masked path.
std::vector<std::vector<double>> DenseGroupMoments(const std::vector<double>& A_raw,
                                                   const std::vector<int>& group_of_global,
                                                   int n_groups, int n_moments) {
    const int N = static_cast<int>(group_of_global.size());
    std::vector<double> A(A_raw);
    const double shift = 0.5 * (kHi + kLo), scale = 0.5 * (kHi - kLo);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[static_cast<size_t>(i) * N + j] /= scale;
            if (i == j) A[static_cast<size_t>(i) * N + j] -= shift / scale;
        }
    std::vector<double> T0(static_cast<size_t>(N) * N, 0.0), T1(A), T2;
    for (int i = 0; i < N; ++i) T0[static_cast<size_t>(i) * N + i] = 1.0;

    std::vector<std::vector<double>> ref(n_groups + 1, std::vector<double>(n_moments, 0.0));
    auto record = [&](int m, const std::vector<double>& Tm) {
        for (int i = 0; i < N; ++i) {
            const double d = Tm[static_cast<size_t>(i) * N + i];
            ref[n_groups][m] += d;  // all-rows channel
            ref[group_of_global[i]][m] += d;
        }
    };
    record(0, T0);
    if (n_moments > 1) record(1, T1);
    for (int m = 2; m < n_moments; ++m) {
        T2 = MatMul(A, T1, N);
        for (size_t t = 0; t < T2.size(); ++t) T2[t] = 2.0 * T2[t] - T0[t];
        record(m, T2);
        T0.swap(T1);
        T1.swap(T2);
    }
    return ref;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    Chain sys;
    const int N = Chain::n();
    const int n_moments = 40;
    DistGraph* g = sys.S->graph;

    rsatb::calc::ISQRTMPreconditioner<double> pre(*sys.Zt, rsatb::calc::kBorrowFactor);

    // Groups: per atom through the builder, and the global map the dense side
    // uses. from_points assigns atom_index = original order, so global group
    // of scalar row p is p / kNorb.
    const auto by_atom = rsatb::calc::PdosGroupsByAtom(*sys.ad, MPI_COMM_WORLD);
    assert(by_atom.n_groups == kAtoms);
    const auto by_slot = rsatb::calc::PdosGroupsByOrbitalSlot(*sys.ad);
    // One species, so the (species, slot) grouping degenerates to the slots
    // themselves, in order -- the property that keeps single-species results
    // unchanged by the multi-species fix.
    assert(by_slot.n_groups == kNorb);
    for (int o = 0; o < kNorb; ++o) {
        assert(by_slot.group_slot[o] == o);
        assert(by_slot.group_type[o] == 0);
    }
    std::vector<int> atom_of_global(N), slot_of_global(N);
    for (int p = 0; p < N; ++p) {
        atom_of_global[p] = p / kNorb;
        slot_of_global[p] = p % kNorb;
    }

    // Complete basis: the identity as one multivector, so the estimator is
    // exact and mean * N = trace.
    DistMultiVector<double> eye(g, N);
    for (int row = 0; row < eye.local_rows; ++row) {
        const int atom = row / kNorb;  // local rows are atom-major, kNorb each
        const int p = g->get_global_index(atom) * kNorb + row % kNorb;
        eye(row, p) = 1.0;
    }

    // ---- Sandwich form: masked moments are the Lowdin projection. ---------
    {
        rsatb::calc::Chebyshev<double> cheby(*sys.H, pre,
                                             rsatb::calc::PreconditionerType::Symmetric,
                                             kLo, kHi);
        assert(cheby.op_is_hermitian());
        const auto mom = rsatb::calc::kpm_pdos_moments(cheby, eye, n_moments, by_atom);

        // Dense reference on the SAME Z: Hp = Z^T H Z.
        const auto Zd = Chain::Densify(*sys.Zt);
        const auto Hd = Chain::Densify(*sys.H);
        std::vector<double> Zt_d(static_cast<size_t>(N) * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                Zt_d[static_cast<size_t>(i) * N + j] = Zd[static_cast<size_t>(j) * N + i];
        const auto Hp = MatMul(Zt_d, MatMul(Hd, Zd, N), N);
        const auto ref = DenseGroupMoments(Hp, atom_of_global, kAtoms, n_moments);

        double worst = 0.0, sum_rule = 0.0;
        for (int m = 0; m < n_moments; ++m) {
            std::complex<double> sum(0.0, 0.0);
            for (int a = 0; a < kAtoms; ++a) {
                sum += mom.at(a, m);
                worst = std::max(worst,
                                 std::abs(N * mom.at(a, m).real() - ref[a][m]));
            }
            sum_rule = std::max(sum_rule, std::abs(sum - mom.total[m]));
            worst = std::max(worst, std::abs(N * mom.total[m].real() - ref[kAtoms][m]));
        }
        if (rank == 0)
            std::printf("sandwich/Lowdin: worst |moment - dense| %.3e, sum rule %.3e\n",
                        worst, sum_rule);
        assert(worst < 1e-8);      // exact estimator; rounding + Z consistency only
        assert(sum_rule < 1e-12);  // partition => exact by construction
    }

    // ---- Solve form: masked moments are the Mulliken projection. ----------
    {
        rsatb::calc::Chebyshev<double> cheby(*sys.H, *sys.S, pre,
                                             rsatb::calc::PreconditionerType::Symmetric,
                                             kLo, kHi);
        assert(!cheby.op_is_hermitian());  // must take the direct pairing
        const auto mom = rsatb::calc::kpm_pdos_moments(cheby, eye, n_moments, by_slot);

        // Dense A = S^-1 H by an exact Cholesky solve, not through Z.
        auto Sd = Chain::Densify(*sys.S);
        auto A = Chain::Densify(*sys.H);  // dposv solves S X = H in place (col-major
        int info = 0;                     // of a symmetric pair: layout cancels)
        dposv_("U", &N, &N, Sd.data(), &N, A.data(), &N, &info);
        assert(info == 0);
        const auto ref = DenseGroupMoments(A, slot_of_global, kNorb, n_moments);

        double worst = 0.0, sum_rule = 0.0;
        for (int m = 0; m < n_moments; ++m) {
            std::complex<double> sum(0.0, 0.0);
            for (int s = 0; s < kNorb; ++s) {
                sum += mom.at(s, m);
                worst = std::max(worst, std::abs(N * mom.at(s, m).real() - ref[s][m]));
            }
            sum_rule = std::max(sum_rule, std::abs(sum - mom.total[m]));
            worst = std::max(worst, std::abs(N * mom.total[m].real() - ref[kNorb][m]));
        }
        if (rank == 0)
            std::printf("solve/Mulliken:  worst |moment - dense| %.3e, sum rule %.3e\n",
                        worst, sum_rule);
        assert(worst < 1e-5);      // limited by the CG solve inside the recursion
        assert(sum_rule < 1e-12);
    }

    // ---- Stochastic sanity: Rademacher masks reduce exactly at m = 0. -----
    {
        rsatb::calc::Chebyshev<double> cheby(*sys.H, pre,
                                             rsatb::calc::PreconditionerType::Symmetric,
                                             kLo, kHi);
        DistMultiVector<double> v(g, 8);
        std::mt19937_64 gen(7 + 977ull * rank);
        std::bernoulli_distribution coin(0.5);
        for (int row = 0; row < v.local_rows; ++row)
            for (int c = 0; c < 8; ++c) v(row, c) = coin(gen) ? 1.0 : -1.0;
        const auto mom = rsatb::calc::kpm_pdos_moments(cheby, v, 4, by_atom);
        // (+-1)^2 = 1: mu_0^g = |g| exactly, sample by sample.
        for (int a = 0; a < kAtoms; ++a) {
            assert(std::abs(mom.at(a, 0).real() - kNorb) < 1e-12);
        }
        assert(std::abs(mom.total[0].real() - N) < 1e-12);
    }

    CheckMultiSpeciesGrouping(rank);

    if (rank == 0) std::printf("test_pdos PASSED (np=%d)\n", world);
    MPI_Finalize();
    return 0;
}
