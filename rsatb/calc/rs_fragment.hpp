#ifndef RSATB_CALC_RS_FRAGMENT_HPP
#define RSATB_CALC_RS_FRAGMENT_HPP

// Fragment control variate for the filtered-square real-space density
// (rs_density.hpp, through the engine of rs_ldos.hpp).
//
// The stochastic map estimator n(r) = E[|Phi T xi|^2] carries an irreducible
// chi^2 sampling floor: each map value is a SQUARED AMPLITUDE of one random
// state, so the pointwise relative error is sqrt(2/n_samples) -- 17.7% at 64
// samples -- no matter how good the filter is (measured against dense
// diagonalization on diamond sc444/sc888: 17.0% and 8.5% at 64 and 256
// samples against the predicted 17.7% / 8.8%). Maps have no trace
// self-averaging to hide behind.
//
// The fix here is a CONTROL VARIATE, not more samples. Build a cheap local
// approximation T0 of the filtered operator T from dense fragment
// diagonalization, then
//
//     n(r) = n0(r) + E[ |Phi T xi|^2 - |Phi T0 xi|^2 ],
//
// where n0(r) is the EXACT map of T0 (computed by streaming the columns of T0
// through the same contraction as identity blocks). The identity holds for
// ANY T0 -- the estimator stays unbiased for the map of T -- while the
// correction's variance is governed by the residual (T - T0):
//
//     pointwise rel error ~ sqrt(2/n) * sqrt(2 * ||row residual||^2 / ||row||^2),
//
// exponentially small in the fragment buffer radius by nearsightedness
// (Benzi-Boito-Razouk-type decay). The buffer is the single systematic knob;
// cost stays O(N) (O(N / core volume) fragments of O(1) size) and the
// Chebyshev sweeps are unchanged.
//
// WHAT THIS IS, in the language of the graph seed (isqrtm.hpp,
// vbcsr graph_function.hpp): the SUBMATRIX METHOD [Lass, Mohr, Wiebeler,
// Kuehne, Plessl 2018; Lass, Schade, Kuehne, Plessl 2020 (CP2K); Schade et
// al. 2022 (NOLSM)] applied to the sandwich operator M = Z^T H Z. Because
// the fragment's variational space is spanned by FULL-HEIGHT columns of Z,
// the Rayleigh compression below IS the principal submatrix:
//
//     A_FF = (Z[:, F])^T H (Z[:, F]) = M[F, F],
//
// so each fragment is one submatrix-method subproblem with a multi-atom
// anchor (their "merged submatrices", our cores), evaluated for an arbitrary
// Chebyshev filter table instead of sign/isqrt, and prolonged back through
// the TRUE global Z columns:
//
//     T0 = Z[:, F]  V diag(g_j) V^T ,   M[F, F] = V diag(a_j) V^T.
//
// Two things distinguish it from NOLSM-class use of submatrices. First, the
// gauge: NOLSM recomputes a local S^{-1/2} inside every submatrix, which is
// fine when the submatrix result IS the answer, but fatal for a control
// variate -- the isqrtm Z equals the symmetric S^{-1/2} only up to a
// near-orthogonal gauge factor, maps are gauge-invariant but per-sample
// states are not, and a surrogate built from (H, S) floors at the gauge
// level (~1e-3 measured) no matter the buffer. Compressing the actual global
// Z removes the floor entirely (machine precision in the full-fragment
// limit). Second, the error channel: the submatrix method ACCEPTS its
// truncation error; here the stochastic correction removes it in
// expectation, so fragment quality only sets the VARIANCE.
//
// SPECTRAL WEIGHTS ARE POINT EVALUATIONS, and that was measured, not
// assumed. The Ritz pairs (a_j, v_j) are not eigenpairs of M; each state's
// energy variance under M, sigma_j^2 = ||M[:, F] v_j||^2 - a_j^2, is exactly
// computable inside the fragment window, and evaluating the filter smeared
// by that width (Gauss-Hermite over N(a_j, sigma_j)) is the L2-optimal
// diagonal weight IF the state's spectral measure were Gaussian. It is not:
// the measure of a truncated state is a dominant peak plus low-weight far
// tails, sigma^2 is tail-dominated (Heisenberg broadening of the spatial
// truncation, eV-scale), and the Gaussian moves the PEAK's weight by a
// width the peak does not have. Measured on diamond sc888/sc444 (2026-08):
// density correction 20x worse (0.0053 -> 0.106), LDOS gains erased at
// every eta (e.g. eta=1.0: 14.5% back to 16.3% of the 17.7% plain floor).
// The regression coefficient kept it no worse than plain -- the designed
// safety -- but the point evaluation p(a_j) is the better weight
// everywhere measured, so it is the only weight. A peak-plus-tail model
// (second-order embedding weights from the window diagonal) is the recorded
// next attempt if sharp-eta surrogate quality is ever worth another shot.
//
// FRAGMENTS. Owned atoms are partitioned into spatial CORES (fractional-
// coordinate bins of target edge `core_edge`); each core is dressed with
// every atom within `buffer` (inclusive; vbcsr NeighborList, so images and
// sheared cells are handled by the library), giving the fragment orbital set
// F. Only the CORE rows of T0 are stored -- P_k = Z[core, F] V diag(g_k) V^T
// -- so cores partition the rows of T0 exactly once and the per-fragment
// storage is |core| x |F| per table.
//
// THE SOLVE FORM compresses the full chain. Its estimator applies
// T = p(S^-1 H) Z q(W) to raw white seeds, so the same CV identity holds with
// T0 truncating every factor of THAT chain: the generalized pencil
// (Z_F^T H Z_F, Z_F^T S Z_F) -- both principal submatrices, W[F, F] and
// M[F, F], through the same full-height Z columns -- and the sampler's own
// polynomial q on its own fitted interval (SolveChainSpec; design doc
// section 9.3). Purified Z gives W = I, q = 1, and the chain
// collapses to the sandwich construction. The gauge lesson above extends
// one factor: a local coloring that merely matches the covariance is off by
// a rotation and kills the per-sample cancellation.
//
// RANK INDEPENDENCE. Fragments are geometric objects over the GLOBAL atom
// list -- fractional-coordinate bins and minimum-image balls know nothing of
// the partition. What must be globally agreed is the fragment DEFINITION:
// atom positions and Z's sparsity pattern with per-block weights (the row
// trim), ~20 bytes per block, replicated once. The block PAYLOADS -- the
// actual O(nnz) data -- are fetched per fragment window through the same
// batched exchange the graph seed uses (fetch_batched_block_payloads /
// construct_submatrix), so a core spanning ranks is built by every rank
// owning part of it from identical fetched numbers: duplicated dense work at
// rank boundaries, identical answers, no global payload replication.

#include "cheby.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_multivector.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda,
             double* w, double* work, const int* lwork, int* iwork, const int* liwork,
             int* info);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

namespace rsatb {
namespace calc {
namespace detail {

/// Relative row-weight floor for the Z-column support of a fragment. Rows of
/// Z whose largest block against the fragment's columns falls below this
/// fraction of Z's largest block anywhere are dropped from the compression.
/// The dropped tail enters the SURROGATE only -- the control variate absorbs
/// it as ~kZrowTrim of residual, invisible until the buffer truncation error
/// falls to the same level -- while the compression cost falls from
/// nf^2 * (full Z pattern) to ~nf^3: without the trim, the sc888 build was
/// ~1e17 flops (days); with it, minutes.
inline constexpr double kZrowTrim = 1e-3;

/// Clenshaw evaluation of a Chebyshev coefficient table at one point of
/// [-1, 1] -- the same series the recursion accumulates, term for term.
inline double ClenshawEval(const std::vector<std::complex<double>>& table, double x) {
    const int m = static_cast<int>(table.size());
    double b1 = 0.0, b2 = 0.0;
    for (int j = m - 1; j >= 1; --j) {
        const double b0 = 2.0 * x * b1 - b2 + table[j].real();
        b2 = b1;
        b1 = b0;
    }
    return x * b1 - b2 + table[0].real();
}

/// The task-independent part of the fragment surrogate: geometry, the
/// per-fragment eigenbasis of the principal submatrix M[F, F], and the
/// gauge-exact core factors. Everything here depends only on (Z, H,
/// core_edge, buffer) -- never on a filter table -- so ONE build serves
/// every density task of a run (all its energy windows); the per-table row
/// blocks are assembled from it by TablesToCv at a cost of one small GEMM
/// per fragment per table.
struct FragmentBasis {
    int n_frag = 0;
    int64_t sum_dom_orb = 0;   ///< Sum of fragment domain sizes (profile).
    int64_t max_dom_orb = 0;
    int64_t n_global_rows = 0;  ///< Global scalar rows (the seed gather size).
    /// Per fragment: GLOBAL scalar rows of the fragment orbital set F (the
    /// seed side, indexing the gathered seed block; ascending, ApplyIdentity
    /// binary-searches them) and LOCAL scalar rows of this rank's share of
    /// the core orbitals (the output scatter).
    std::vector<std::vector<int64_t>> dom_rows;
    std::vector<std::vector<int>> core_rows;
    struct Frag {
        std::vector<double> a_eig;  ///< Compression eigenvalues, raw eV
                                    ///< (solve chain: nk kept pencil values).
    };
    std::vector<Frag> frags;
    /// The per-table operator rows P_k = Z[core, F] V g_k(eps) V^T (solve
    /// chain: ZcV diag(g_k) K), [table][frag], column-major |core| x |F|.
    /// Assembled INSIDE the build, fragment by fragment, so the nf x nf
    /// eigenvector matrix never outlives its own eigensolve: retaining it
    /// for later tables cost n_frag x nf^2 doubles -- 216 MB per fragment
    /// at an 8 A buffer, 225 GB for the 12^3 silicon cell, the OOM that
    /// killed the first 12^3 density runs -- where P is |core| x |F| per
    /// table, 115x smaller. The price is that a task with different tables
    /// rebuilds (one eigensolve per fragment), which the cache key records.
    std::vector<std::vector<std::vector<double>>> P;
    std::vector<std::vector<std::complex<double>>> tables;  ///< P's tables.
    double center = 0.0, half = 0.0;                        ///< Their interval.
};

/// The solve-form surrogate chain: the estimator applies
/// T = p(S^-1 H) Zt q(W) to RAW white seeds, so the fragment surrogate must
/// compress the whole chain -- the pencil (Zb^T H Zb, Zb^T S Zb) AND the
/// sampler's own coloring polynomial q on its own fitted interval
/// (design_subspace_deflation.md section 9.3). Default-constructed =
/// sandwich chain (no S, no q; today's construction, to which the solve
/// chain collapses when W = I and q = 1).
struct SolveChainSpec {
    vbcsr::BlockSpMat<double>* S = nullptr;   ///< Overlap; null = sandwich.
    std::vector<std::complex<double>> q;      ///< Sampler polynomial table.
    double lo = 0.0, hi = 0.0;                ///< Its fitted interval on W.
    bool active() const { return S != nullptr; }
};

/// One-slot cache of the basis, held by the caller across task calls (the
/// RSATB object holds one per setup). The key is the pair of geometric
/// knobs; the (Z, H) pair is fixed by the setup generation, and whoever
/// owns the cache invalidates it when that changes.
struct FragmentBasisCache {
    FragmentBasis basis;
    bool valid = false;
    double core_edge = 0.0;
    double buffer = 0.0;
    bool solve_chain = false;  ///< Which chain the cached basis compresses.
    void invalidate() {
        valid = false;
        basis = FragmentBasis();
    }
};

/// The per-table fragment operator rows P_k and the bookkeeping to apply
/// them; the geometry lives in the referenced basis, which must outlive
/// this object (it does: either the caller's cache or a local in the same
/// scope).
struct FragmentCv {
    const FragmentBasis* basis = nullptr;
    /// The basis' per-table rows, [table][frag], column-major |core| x |F|;
    /// owned by the basis (the cache), never copied.
    const std::vector<std::vector<std::vector<double>>>* P = nullptr;
    /// Inverted domain index for the identity stream: every (global column,
    /// fragment, position-in-domain) triple of this rank's fragments, sorted
    /// by column. ApplyIdentity binary-searches a column block here instead
    /// of scanning every fragment -- the difference between O(touched) and
    /// the O(n_frag) screening that made the baseline stream quadratic in N
    /// (blocks x fragments both grow with the cell). Built lazily once.
    struct ColRef {
        int64_t col;
        int32_t frag;
        int32_t pos;
    };
    mutable std::vector<ColRef> col_index;

    bool enabled() const { return basis != nullptr && basis->n_frag > 0; }
    int n_frag() const { return basis ? basis->n_frag : 0; }
    int64_t n_global_rows() const { return basis ? basis->n_global_rows : 0; }

    /// chi0 = T0(table k) xi, from the GLOBALLY GATHERED seed block
    /// (n_global_rows x nv, row-major) -- the same numbers on every rank, so
    /// the fragment products are rank-count invariant. Cores partition the
    /// rows globally and this rank holds only its owned share, so every
    /// owned row is written exactly once; chi0 is zeroed first regardless.
    void Apply(int k, const std::vector<double>& xi_global, int nv,
               vbcsr::DistMultiVector<double>* chi0) const {
        chi0->scale(0.0);
        // Cores partition the output rows, so fragments write disjoint rows
        // and the loop threads without synchronization; the BLAS inside runs
        // sequential under the active parallel region.
#pragma omp parallel
        {
            std::vector<double> x_dom, y_core;
#pragma omp for schedule(dynamic)
            for (int f = 0; f < basis->n_frag; ++f) {
                const int nd = static_cast<int>(basis->dom_rows[f].size());
                const int nc = static_cast<int>(basis->core_rows[f].size());
                if (nc == 0) continue;
                x_dom.resize(static_cast<size_t>(nd) * nv);
                y_core.assign(static_cast<size_t>(nc) * nv, 0.0);
                for (int j = 0; j < nd; ++j) {
                    const double* src = xi_global.data() + basis->dom_rows[f][j] * nv;
                    for (int v = 0; v < nv; ++v) {
                        x_dom[static_cast<size_t>(v) * nd + j] = src[v];
                    }
                }
                const double one = 1.0, zero = 0.0;
                dgemm_("N", "N", &nc, &nv, &nd, &one, (*P)[k][f].data(), &nc,
                       x_dom.data(), &nd, &zero, y_core.data(), &nc);
                for (int i = 0; i < nc; ++i) {
                    double* dst = chi0->row_data(basis->core_rows[f][i]);
                    for (int v = 0; v < nv; ++v) {
                        dst[v] += y_core[static_cast<size_t>(v) * nc + i];
                    }
                }
            }
        }
    }

    /// chi0 = T0 [e_start .. e_(start+nv-1)], the identity-block special case
    /// the exact baseline streams. A column of T0 IS a stored column of P --
    /// T0 e_s reads column dom_pos(s) of every fragment holding s in its
    /// domain -- so the block is a scatter of already-computed numbers; the
    /// dense product Apply would run against these seeds is all zeros but for
    /// one entry per column.
    ///
    /// Bounded-work contract: `rows` carries, on entry, the rows the PREVIOUS
    /// call wrote (they are zeroed here -- never the whole multivector, whose
    /// blanket scale(0) was itself an O(N) touch per block); on return it
    /// holds the rows this call wrote, sorted and unique. Pass it empty on
    /// the first call over a freshly zeroed chi0. Fragments are found through
    /// the inverted column index, so the work is O(touched), independent of
    /// the fragment count.
    void ApplyIdentity(int k, int64_t start, int nv,
                       vbcsr::DistMultiVector<double>* chi0,
                       std::vector<int>* rows) const {
        for (const int r : *rows) {
            double* d = chi0->row_data(r);
            std::fill(d, d + nv, 0.0);
        }
        rows->clear();
        if (col_index.empty()) {
            size_t total = 0;
            for (int f = 0; f < basis->n_frag; ++f) total += basis->dom_rows[f].size();
            col_index.reserve(total);
            for (int f = 0; f < basis->n_frag; ++f) {
                if (basis->core_rows[f].empty()) continue;
                const auto& dr = basis->dom_rows[f];
                for (size_t j = 0; j < dr.size(); ++j) {
                    col_index.push_back({dr[j], f, static_cast<int32_t>(j)});
                }
            }
            std::sort(col_index.begin(), col_index.end(),
                      [](const ColRef& a, const ColRef& b) { return a.col < b.col; });
        }
        const auto lo = std::lower_bound(
            col_index.begin(), col_index.end(), start,
            [](const ColRef& a, int64_t c) { return a.col < c; });
        const auto hi = std::lower_bound(
            lo, col_index.end(), start + nv,
            [](const ColRef& a, int64_t c) { return a.col < c; });
        for (auto it = lo; it != hi; ++it) {
            const int f = it->frag;
            const int nc = static_cast<int>(basis->core_rows[f].size());
            const int v = static_cast<int>(it->col - start);
            const double* col =
                (*P)[k][f].data() + static_cast<size_t>(it->pos) * nc;
            for (int i = 0; i < nc; ++i) {
                chi0->row_data(basis->core_rows[f][i])[v] += col[i];
            }
            rows->insert(rows->end(), basis->core_rows[f].begin(),
                         basis->core_rows[f].end());
        }
        std::sort(rows->begin(), rows->end());
        rows->erase(std::unique(rows->begin(), rows->end()), rows->end());
    }
};

/// Builds the fragment BASIS: spatial cores of target edge `core_edge`,
/// dressed by `buffer` (both Angstrom, inclusive), one dense diagonalization
/// of the (Z, H) Rayleigh compression per fragment. Fragment definitions
/// come from globally replicated geometry and Z-pattern weights, so the
/// surrogate operator is identical at any rank count; the block payloads are
/// fetched per fragment window. This rank keeps only fragments holding at
/// least one of its owned core atoms and stores factors for those atoms
/// only. The filter `tables` (on the scaled interval `center`, `half`) enter
/// here: the per-table rows are assembled from each fragment's eigensolve
/// on the spot and the eigenvectors are dropped (see FragmentBasis::P).
/// `workspace_mb` bounds the fragments in flight: each holds its Z panel
/// Z[nr, F] (|Z row support| x |F| doubles -- with a dense Z that is N x
/// |F|), the compression panel, and the nf x nf workspace; 0 = one per
/// thread, the same cap convention as the spectrum tasks' max_workspace_mb.
inline FragmentBasis BuildFragmentBasis(
    const vbcsr::atomic::AtomicData& ad, vbcsr::BlockSpMat<double>& H,
    vbcsr::BlockSpMat<double>& Z, double core_edge, double buffer,
    const std::vector<std::vector<std::complex<double>>>& tables, double center,
    double half, MPI_Comm comm, const SolveChainSpec& chain = {},
    int workspace_mb = 0) {
    if (!(core_edge > 0.0) || !(buffer > 0.0)) {
        throw std::runtime_error("Fragment control variate needs positive core and buffer.");
    }
    if (tables.empty() || !(half > 0.0)) {
        throw std::runtime_error("Fragment basis needs the task's filter tables.");
    }
    if (chain.active() && (chain.q.empty() || !(chain.hi > chain.lo))) {
        throw std::runtime_error(
            "Fragment solve chain needs the sampler's q table and its interval.");
    }
    int size = 1;
    if (comm != MPI_COMM_NULL) MPI_Comm_size(comm, &size);
    const bool exchange = (comm != MPI_COMM_NULL && size > 1);

    auto allgather_i = [&](std::vector<int>& v) {
        if (!exchange) return;
        const int mine = static_cast<int>(v.size());
        std::vector<int> counts(size, 0), displs(size + 1, 0);
        MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        for (int r = 0; r < size; ++r) displs[r + 1] = displs[r] + counts[r];
        std::vector<int> all(displs[size]);
        MPI_Allgatherv(v.data(), mine, MPI_INT, all.data(), counts.data(),
                       displs.data(), MPI_INT, comm);
        v.swap(all);
    };
    auto allgather_d = [&](std::vector<double>& v) {
        if (!exchange) return;
        const int mine = static_cast<int>(v.size());
        std::vector<int> counts(size, 0), displs(size + 1, 0);
        MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        for (int r = 0; r < size; ++r) displs[r + 1] = displs[r] + counts[r];
        std::vector<double> all(displs[size]);
        MPI_Allgatherv(v.data(), mine, MPI_DOUBLE, all.data(), counts.data(),
                       displs.data(), MPI_DOUBLE, comm);
        v.swap(all);
    };

    // ---- Global atom table: (gid -> type, position), one allgather. -------
    // Atom global ids are assumed dense 0..n_glob-1, the same assumption the
    // complete-basis enumeration makes; checked below through the row count.
    std::vector<int> meta;      // (gid, type) pairs, owned atoms
    std::vector<double> pos;    // (x, y, z), owned atoms
    for (int a = 0; a < ad.n_atom; ++a) {
        meta.push_back(ad.get_global_index(a));
        meta.push_back(ad.atom_type[a]);
        pos.insert(pos.end(), {ad.x[a], ad.y[a], ad.z[a]});
    }
    allgather_i(meta);
    allgather_d(pos);
    const int n_glob = static_cast<int>(meta.size() / 2);
    std::vector<int> gtype(n_glob, -1);
    std::vector<double> gx(n_glob), gy(n_glob), gz(n_glob);
    for (int i = 0; i < n_glob; ++i) {
        // Records arrive rank-major; scatter them into gid order.
        const int gid = meta[2 * i];
        if (gid < 0 || gid >= n_glob || gtype[gid] >= 0) {
            throw std::runtime_error(
                "Fragment build: atom global indices are not the dense "
                "0..N-1 enumeration this path assumes.");
        }
        gtype[gid] = meta[2 * i + 1];
        gx[gid] = pos[3 * i];
        gy[gid] = pos[3 * i + 1];
        gz[gid] = pos[3 * i + 2];
    }
    std::vector<int64_t> grow_off(n_glob + 1, 0);
    for (int g = 0; g < n_glob; ++g) grow_off[g + 1] = grow_off[g] + ad.type_norb[gtype[g]];

    // Owned gid -> local atom, for the core filter and the output scatter.
    std::vector<int> local_of_gid(n_glob, -1);
    std::vector<int> row_off(ad.n_atom + 1, 0);
    for (int a = 0; a < ad.n_atom; ++a) {
        local_of_gid[ad.get_global_index(a)] = a;
        row_off[a + 1] = row_off[a] + ad.type_norb[ad.atom_type[a]];
    }

    // ---- Z's pattern with block weights, globally replicated. -------------
    // The row support and its trim must be the SAME set on every rank for the
    // surrogate to be partition-free, and the pattern is all that takes:
    // (row, col, |block|_F) is ~20 bytes against the payload's kilobyte-plus.
    // The payloads themselves are fetched per fragment window below.
    std::vector<int> z_head;     // (gi, gj) per block
    std::vector<double> z_fro;   // Frobenius norm per block
    {
        const auto& row_ptr = Z.row_ptr();
        const auto& col_ind = Z.col_ind();
        for (int i = 0; i < ad.n_atom; ++i) {
            const int r = ad.type_norb[ad.atom_type[i]];
            const int gi = ad.get_global_index(i);
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int j = col_ind[k];
                const int c = Z.graph->block_sizes[j];
                const double* blk = Z.block_data(k);
                double f2 = 0.0;
                for (int64_t t = 0; t < static_cast<int64_t>(r) * c; ++t) f2 += blk[t] * blk[t];
                z_head.push_back(gi);
                z_head.push_back(Z.graph->get_global_index(j));
                z_fro.push_back(std::sqrt(f2));
            }
        }
    }
    allgather_i(z_head);
    allgather_d(z_fro);
    double z_max_block = 0.0;
    std::vector<std::vector<std::pair<int, double>>> z_rows_of_col(n_glob);
    for (size_t b = 0; b < z_fro.size(); ++b) {
        z_max_block = std::max(z_max_block, z_fro[b]);
        z_rows_of_col[z_head[2 * b + 1]].emplace_back(z_head[2 * b], z_fro[b]);
    }

    // ---- Geometry on the global table, on the library's own machinery. ----
    // Cores: fractional bins with target edge core_edge along each axis --
    // partition-free by construction. The bin assignment reuses the library
    // cell inverse; the buffer search below is the library NeighborList
    // (cell-listed, multi-image, correct for sheared cells where a bare
    // minimum-image round is not), built ONCE on the replicated global
    // positions and shared by every fragment -- identical on every rank.
    const std::vector<double>& c = ad.cell;  // rows are the lattice vectors
    const auto inv = vbcsr::atomic::InvertCell3x3(c.data());
    const double len[3] = {std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]),
                           std::sqrt(c[3] * c[3] + c[4] * c[4] + c[5] * c[5]),
                           std::sqrt(c[6] * c[6] + c[7] * c[7] + c[8] * c[8])};
    int nb[3];
    for (int d = 0; d < 3; ++d) {
        nb[d] = std::max(1, static_cast<int>(std::floor(len[d] / core_edge)));
    }
    std::vector<std::vector<int>> core_atoms(static_cast<size_t>(nb[0]) * nb[1] * nb[2]);
    for (int g = 0; g < n_glob; ++g) {
        const double r[3] = {gx[g], gy[g], gz[g]};
        int ib[3];
        for (int d = 0; d < 3; ++d) {
            const double f =
                r[0] * inv[0 + d] + r[1] * inv[3 + d] + r[2] * inv[6 + d];
            double w = f - std::floor(f);  // wrap into [0, 1)
            ib[d] = std::min(nb[d] - 1, static_cast<int>(w * nb[d]));
        }
        core_atoms[(static_cast<size_t>(ib[0]) * nb[1] + ib[1]) * nb[2] + ib[2]].push_back(g);
    }

    vbcsr::atomic::NeighborList nlist;
    {
        std::vector<double> gpos(3 * static_cast<size_t>(n_glob));
        for (int g = 0; g < n_glob; ++g) {
            gpos[3 * static_cast<size_t>(g)] = gx[g];
            gpos[3 * static_cast<size_t>(g) + 1] = gy[g];
            gpos[3 * static_cast<size_t>(g) + 2] = gz[g];
        }
        // NeighborList compares strictly (r < cutoff); the buffer is
        // inclusive here, and the difference is not measure-zero in practice
        // -- a round buffer on a lattice with round spacings (the chain
        // fixture: 2.0 A atoms, 6.0 A buffer) sits atoms EXACTLY on the
        // boundary, and losing that shell tripled the fixture's correction.
        // The 1e-9 A inclusive tolerance is the SAME one AtomicData's own
        // graph construction applies to its cutoff test.
        nlist.build(gpos, ad.cell, ad.pbc, buffer + 1e-9);
    }

    // Every rank walks every fragment (the geometry is global), but only
    // ranks owning part of the core pay for the dense solve. The paying set
    // is fixed here so the work below can thread over fragments with each
    // slot writing its own preallocated entry -- the output order is the
    // geometric bin order regardless of scheduling.
    struct FragTask {
        const std::vector<int>* core;
    };
    std::vector<FragTask> tasks;
    for (int i0 = 0; i0 < nb[0]; ++i0) {
        for (int i1 = 0; i1 < nb[1]; ++i1) {
            for (int i2 = 0; i2 < nb[2]; ++i2) {
                const auto& core =
                    core_atoms[(static_cast<size_t>(i0) * nb[1] + i1) * nb[2] + i2];
                if (core.empty()) continue;
                bool mine = false;
                for (int a : core) {
                    if (local_of_gid[a] >= 0) {
                        mine = true;
                        break;
                    }
                }
                if (mine) tasks.push_back({&core});
            }
        }
    }
    const int n_mine = static_cast<int>(tasks.size());

    // ---- Fragment supports: dom (columns F) and nr (Z row support). -------
    // Cheap geometry + pattern walks, threaded; the sets are needed up front
    // because the payload fetch below is batched and collective.
    std::vector<std::vector<int>> frag_dom(n_mine), frag_nr(n_mine);
#pragma omp parallel
    {
        std::vector<char> seen(n_glob, 0);
#pragma omp for schedule(dynamic)
        for (int fi = 0; fi < n_mine; ++fi) {
            const std::vector<int>& core = *tasks[fi].core;
            // dom = core dressed with every within-buffer neighbour; multiple
            // images of the same atom count once (dom is a SET of columns).
            std::vector<int> dom;
            for (int a : core) {
                if (!seen[a]) {
                    seen[a] = 1;
                    dom.push_back(a);
                }
                for (const auto& nbr : nlist.neighbors[a]) {
                    if (!seen[nbr.index]) {
                        seen[nbr.index] = 1;
                        dom.push_back(nbr.index);
                    }
                }
            }
            for (int a : dom) seen[a] = 0;
            std::sort(dom.begin(), dom.end());

            // Row support: atoms whose Z row carries weight against an F
            // column, trimmed at kZrowTrim of Z's largest block; F itself
            // always kept.
            const double w_floor = kZrowTrim * z_max_block;
            std::vector<int> nr;
            for (int b : dom) {
                if (!seen[b]) {
                    seen[b] = 1;
                    nr.push_back(b);
                }
                for (const auto& [i, w] : z_rows_of_col[b]) {
                    if (w >= w_floor && !seen[i]) {
                        seen[i] = 1;
                        nr.push_back(i);
                    }
                }
            }
            for (int a : nr) seen[a] = 0;
            std::sort(nr.begin(), nr.end());

            frag_dom[fi] = std::move(dom);
            frag_nr[fi] = std::move(nr);
        }
    }

    FragmentBasis basis;
    basis.n_global_rows = grow_off[n_glob];
    basis.n_frag = n_mine;
    basis.dom_rows.resize(n_mine);
    basis.core_rows.resize(n_mine);
    basis.frags.resize(n_mine);
    basis.P.assign(tables.size(), std::vector<std::vector<double>>(n_mine));
    basis.tables = tables;
    basis.center = center;
    basis.half = half;
    std::vector<int> nf_of(n_mine, 0);
    int eig_fail = 0;  // dsyevd info of the first failure; thrown after the loop

    // ---- Dense solves, batched: fetch payloads for one batch of fragment
    // windows (collective -- every rank runs the same batch count, short
    // ranks padding with empty requests), then one fragment per thread with
    // the BLAS sequential inside. Same structure as the graph seed's driver.
    // The batch is bounded by the workspace cap against the LARGEST
    // fragment's working set (Z panel + compression panel + nf x nf
    // matrices), uniformly across ranks so the collectives stay aligned.
    int batch_size = std::max(1, omp_get_max_threads());
    if (workspace_mb > 0) {
        double max_bytes = 0.0;
        for (int fi = 0; fi < n_mine; ++fi) {
            double nrr = 0.0, nf = 0.0;
            for (int a : frag_nr[fi]) nrr += ad.type_norb[gtype[a]];
            for (int a : frag_dom[fi]) nf += ad.type_norb[gtype[a]];
            const double bytes =
                8.0 * (nrr * nf + 512.0 * (nrr + nf) + 3.0 * nf * nf);
            max_bytes = std::max(max_bytes, bytes);
        }
        const double cap = 1048576.0 * static_cast<double>(workspace_mb);
        const int fit = (max_bytes > 0.0) ? static_cast<int>(cap / max_bytes) : batch_size;
        batch_size = std::max(1, std::min(batch_size, fit));
    }
    int n_mine_max = n_mine;
    if (exchange) {
        int tmp_i = batch_size;
        MPI_Allreduce(&tmp_i, &batch_size, 1, MPI_INT, MPI_MIN, comm);
        tmp_i = n_mine;
        MPI_Allreduce(&tmp_i, &n_mine_max, 1, MPI_INT, MPI_MAX, comm);
    }
    const int n_batches = (n_mine_max + batch_size - 1) / batch_size;
    std::vector<std::vector<int>> batch(batch_size);
    // Fetched blocks grouped by global row, rebuilt per batch (only touched
    // rows are cleared): a fragment walks just its own rows and filters
    // columns against its window by binary search -- the whole "submatrix"
    // is this loop, so no per-fragment graph or matrix is ever built.
    std::vector<std::vector<int>> h_of_row(n_glob), z_of_row(n_glob),
        s_of_row(n_glob);
    std::vector<int> touched;

    for (int b = 0; b < n_batches; ++b) {
        for (int s = 0; s < batch_size; ++s) {
            const int idx = b * batch_size + s;
            batch[s] = (idx < n_mine) ? frag_nr[idx] : std::vector<int>{};
        }
        const auto ctxH = vbcsr::detail::fetch_batched_block_payloads(H, batch);
        const auto ctxZ = vbcsr::detail::fetch_batched_block_payloads(Z, batch);
        // The solve chain also compresses S (collective like the others;
        // chain.active() is uniform across ranks -- it is the operator form).
        vbcsr::detail::FetchedBlockContext<double> ctxS;
        if (chain.active()) {
            ctxS = vbcsr::detail::fetch_batched_block_payloads(*chain.S, batch);
        }
        for (int g : touched) {
            h_of_row[g].clear();
            z_of_row[g].clear();
            s_of_row[g].clear();
        }
        touched.clear();
        auto index_ctx = [&](const vbcsr::detail::FetchedBlockContext<double>& ctx,
                             std::vector<std::vector<int>>& of_row) {
            for (size_t k = 0; k < ctx.blocks.size(); ++k) {
                const int g = ctx.blocks[k].global_row;
                if (h_of_row[g].empty() && z_of_row[g].empty() &&
                    s_of_row[g].empty()) {
                    touched.push_back(g);
                }
                of_row[g].push_back(static_cast<int>(k));
            }
        };
        index_ctx(ctxH, h_of_row);
        index_ctx(ctxZ, z_of_row);
        if (chain.active()) index_ctx(ctxS, s_of_row);

#pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < batch_size; ++s) {
            const int fi = b * batch_size + s;
            if (fi >= n_mine) continue;
            const std::vector<int>& core = *tasks[fi].core;
            const std::vector<int>& dom = frag_dom[fi];
            const std::vector<int>& nr = frag_nr[fi];

            const int na = static_cast<int>(nr.size());
            std::vector<int> r_off(na + 1, 0);
            for (int p = 0; p < na; ++p) r_off[p + 1] = r_off[p] + ad.type_norb[gtype[nr[p]]];
            const int nrr = r_off.back();

            // F = positions of dom inside nr (dom is a subset by construction).
            const int nd = static_cast<int>(dom.size());
            std::vector<int> fpos(nd);
            std::vector<int> pos2f(na, -1);
            std::vector<int> f_off(nd + 1, 0);
            for (int q = 0; q < nd; ++q) {
                fpos[q] = static_cast<int>(
                    std::lower_bound(nr.begin(), nr.end(), dom[q]) - nr.begin());
                pos2f[fpos[q]] = q;
                f_off[q + 1] = f_off[q] + ad.type_norb[gtype[dom[q]]];
            }
            const int nf = f_off.back();

            auto pos_in_nr = [&](int gid) {
                const auto it = std::lower_bound(nr.begin(), nr.end(), gid);
                return (it != nr.end() && *it == gid) ? static_cast<int>(it - nr.begin())
                                                      : -1;
            };

            // Zb = Z[nr, F], column-major, straight from the fetched blocks.
            // The batch fetch returns the union of the batch's windows per
            // row, so columns outside THIS window are filtered here.
            std::vector<double> Zb(static_cast<size_t>(nrr) * nf, 0.0);
            for (int i = 0; i < na; ++i) {
                for (const int k : z_of_row[nr[i]]) {
                    const auto& blk = ctxZ.blocks[k];
                    const int j = pos_in_nr(blk.global_col);
                    if (j < 0) continue;
                    const int q = pos2f[j];
                    if (q < 0) continue;
                    for (int a = 0; a < blk.r_dim; ++a) {
                        for (int b2 = 0; b2 < blk.c_dim; ++b2) {
                            Zb[static_cast<size_t>(f_off[q] + b2) * nrr + (r_off[i] + a)] =
                                blk.data[static_cast<size_t>(a) * blk.c_dim + b2];
                        }
                    }
                }
            }

            // tmp = H Zb as CHUNKED dense GEMMs: a group of block rows is
            // gathered into one dense panel (~kPanelRows scalar rows x nrr)
            // and multiplied in one k = nrr GEMM. The chunk width is the
            // whole game, both ways. Per-block 13x13 GEMMs measured 910 s at
            // sc888 (the 1e-3-trimmed window spans ~850 of 1024 atoms --
            // nearly dense -- and tiny GEMMs with strided output run at a
            // tenth of BLAS3); one-block-row panels (m = 13) measured 3710 s,
            // because each GEMM re-streams the full nrr x nf right-hand side
            // for 13 rows of output -- 1.6 flops/byte across the whole team
            // is a bandwidth wall, not a compute kernel. A ~512-row chunk
            // re-streams it nrr/512 times instead of nrr/13, is
            // compute-bound, and holds O(chunk x nrr) -- never the nrr^2
            // dense window whose full-team peak measured 123 GB.
            // Compression Zb^T (Op Zb) as CHUNKED dense GEMMs: a group of
            // block rows is gathered into one dense panel (~kPanelRows scalar
            // rows x nrr) and multiplied in one k = nrr GEMM. The chunk width
            // is the whole game, both ways. Per-block 13x13 GEMMs measured
            // 910 s at sc888 (the 1e-3-trimmed window spans ~850 of 1024
            // atoms -- nearly dense -- and tiny GEMMs with strided output run
            // at a tenth of BLAS3); one-block-row panels (m = 13) measured
            // 3710 s, because each GEMM re-streams the full nrr x nf
            // right-hand side for 13 rows of output -- 1.6 flops/byte across
            // the whole team is a bandwidth wall, not a compute kernel. A
            // ~512-row chunk re-streams it nrr/512 times instead of nrr/13,
            // is compute-bound, and holds O(chunk x nrr) -- never the nrr^2
            // dense window whose full-team peak measured 123 GB. Shared by
            // the H compression (A_FF = M[F, F], the principal submatrix of
            // the sandwich operator; nothing beyond the F rows is ever
            // needed -- the smeared-weights experiment that consumed the
            // full window was measured out) and, on the solve chain, the S
            // compression W_FF = W[F, F]. Symmetrized against accumulation
            // rounding.
            // Per row panel R: out += Zb[R, :]^T (Op[R, nr] Zb) -- the
            // accumulation runs on the nf x nf output directly, so the
            // nrr x nf product Op Zb is never materialized (it doubled the
            // per-fragment working set: 1.9 GB more per in-flight fragment
            // at the 12^3 cell's dense Z and an 8 A buffer).
            constexpr int kPanelRows = 512;
            std::vector<double> panel, prod;
            auto compress = [&](const std::vector<std::vector<int>>& op_of_row,
                                const vbcsr::detail::FetchedBlockContext<double>& ctx,
                                std::vector<double>& out) {
                out.assign(static_cast<size_t>(nf) * nf, 0.0);
                const double one = 1.0, zero = 0.0;
                for (int a0 = 0; a0 < na;) {
                    int a1 = a0;
                    while (a1 < na && r_off[a1 + 1] - r_off[a0] <= kPanelRows) ++a1;
                    if (a1 == a0) ++a1;  // one oversized block row still goes alone
                    const int m = r_off[a1] - r_off[a0];
                    panel.assign(static_cast<size_t>(m) * nrr, 0.0);
                    bool any = false;
                    for (int i = a0; i < a1; ++i) {
                        const int base = r_off[i] - r_off[a0];
                        for (const int k : op_of_row[nr[i]]) {
                            const auto& blk = ctx.blocks[k];
                            const int j = pos_in_nr(blk.global_col);
                            if (j < 0) continue;
                            any = true;
                            for (int a = 0; a < blk.r_dim; ++a) {
                                for (int c2 = 0; c2 < blk.c_dim; ++c2) {
                                    panel[static_cast<size_t>(r_off[j] + c2) * m +
                                          (base + a)] =
                                        blk.data[static_cast<size_t>(a) * blk.c_dim + c2];
                                }
                            }
                        }
                    }
                    if (any) {
                        prod.assign(static_cast<size_t>(m) * nf, 0.0);
                        dgemm_("N", "N", &m, &nf, &nrr, &one, panel.data(), &m,
                               Zb.data(), &nrr, &zero, prod.data(), &m);
                        dgemm_("T", "N", &nf, &nf, &m, &one, Zb.data() + r_off[a0],
                               &nrr, prod.data(), &m, &one, out.data(), &nf);
                    }
                    a0 = a1;
                }
                for (int i = 0; i < nf; ++i) {
                    for (int j = 0; j < i; ++j) {
                        const double v = 0.5 * (out[static_cast<size_t>(j) * nf + i] +
                                                out[static_cast<size_t>(i) * nf + j]);
                        out[static_cast<size_t>(j) * nf + i] = v;
                        out[static_cast<size_t>(i) * nf + j] = v;
                    }
                }
            };
            // dsyevd in place (ascending; a -> eigenvectors), info returned
            // rather than thrown: throwing across the parallel region would
            // terminate without a message, so failures are recorded and
            // raised after the join.
            auto sym_eig = [](int n2, double* a, double* w) {
                int info = 0, lwork = -1, liwork = -1, iwq = 0;
                double wq = 0.0;
                dsyevd_("V", "U", &n2, a, &n2, w, &wq, &lwork, &iwq, &liwork, &info);
                lwork = static_cast<int>(wq);
                liwork = iwq;
                std::vector<double> work(static_cast<size_t>(lwork));
                std::vector<int> iwork(static_cast<size_t>(liwork));
                dsyevd_("V", "U", &n2, a, &n2, w, work.data(), &lwork, iwork.data(),
                        &liwork, &info);
                return info;
            };

            std::vector<double> A;
            compress(h_of_row, ctxH, A);
            std::vector<double> a_eig(nf, 0.0);
            std::vector<double> Kfac;  // solve chain right factor, nk x nf
            int nk = nf;
            if (!chain.active()) {
                const int info = sym_eig(nf, A.data(), a_eig.data());
                if (info != 0) {
#pragma omp atomic write
                    eig_fail = info;
                    continue;
                }
            } else {
                // Solve chain: the generalized pencil (A_FF, W_FF) plus the
                // sampler's coloring, T0's factors per design section 9.3:
                //   A_FF v = W_FF v eps,  V^T W_FF V = I,
                //   K = V^T W q(W) = Vt^T diag(sqrt(w) q(w)) Vw_kept^T,
                // through W's own eigenbasis (which the q factor needs
                // anyway); directions of W below the rank floor drop out of
                // K analytically (V is orthogonal to them through X).
                std::vector<double> Wf;
                compress(s_of_row, ctxS, Wf);
                std::vector<double> w(nf, 0.0);
                int info = sym_eig(nf, Wf.data(), w.data());
                if (info == 0) {
                    const double wmax = w[nf - 1];
                    int first = nf;
                    for (int i = 0; i < nf; ++i) {
                        if (w[i] > 1e-12 * wmax && w[i] > 0.0) {
                            first = i;
                            break;
                        }
                    }
                    nk = nf - first;
                    if (nk == 0) info = -1;  // W_FF numerically zero: no basis
                    if (info == 0) {
                        // X = Vw_kept diag(w^{-1/2}); At = X^T A X.
                        std::vector<double> X(static_cast<size_t>(nf) * nk);
                        for (int j = 0; j < nk; ++j) {
                            const double s2 = 1.0 / std::sqrt(w[first + j]);
                            const double* src =
                                Wf.data() + static_cast<size_t>(first + j) * nf;
                            double* dst = X.data() + static_cast<size_t>(j) * nf;
                            for (int i = 0; i < nf; ++i) dst[i] = src[i] * s2;
                        }
                        std::vector<double> AX(static_cast<size_t>(nf) * nk);
                        std::vector<double> At(static_cast<size_t>(nk) * nk);
                        const double one = 1.0, zero = 0.0;
                        dgemm_("N", "N", &nf, &nk, &nf, &one, A.data(), &nf,
                               X.data(), &nf, &zero, AX.data(), &nf);
                        dgemm_("T", "N", &nk, &nk, &nf, &one, X.data(), &nf,
                               AX.data(), &nf, &zero, At.data(), &nk);
                        a_eig.assign(nk, 0.0);
                        info = sym_eig(nk, At.data(), a_eig.data());
                        if (info == 0) {
                            // V = X Vt replaces A's role as the left factor.
                            std::vector<double> V(static_cast<size_t>(nf) * nk);
                            dgemm_("N", "N", &nf, &nk, &nk, &one, X.data(), &nf,
                                   At.data(), &nk, &zero, V.data(), &nf);
                            // K = Vt^T diag(sqrt(w) q(w)) Vw_kept^T, with q
                            // evaluated as the ACTUAL sampler polynomial on
                            // its own fitted interval (gauge matching).
                            const double qc = 0.5 * (chain.hi + chain.lo);
                            const double qh = 0.5 * (chain.hi - chain.lo);
                            std::vector<double> Dvw(static_cast<size_t>(nk) * nf);
                            for (int t = 0; t < nk; ++t) {
                                const double wt = w[first + t];
                                double x = (wt - qc) / qh;
                                x = std::max(-1.0, std::min(1.0, x));
                                const double d =
                                    std::sqrt(wt) * ClenshawEval(chain.q, x);
                                const double* vw =
                                    Wf.data() + static_cast<size_t>(first + t) * nf;
                                for (int c2 = 0; c2 < nf; ++c2) {
                                    Dvw[static_cast<size_t>(c2) * nk + t] =
                                        d * vw[c2];
                                }
                            }
                            Kfac.assign(static_cast<size_t>(nk) * nf, 0.0);
                            dgemm_("T", "N", &nk, &nf, &nk, &one, At.data(), &nk,
                                   Dvw.data(), &nk, &zero, Kfac.data(), &nk);
                            A.swap(V);  // A now holds the pencil vectors
                        }
                    }
                }
                if (info != 0) {
#pragma omp atomic write
                    eig_fail = info;
                    continue;
                }
            }

            // This rank's core rows: owned atoms only. P_k = Z[core, F] V g V^T.
            std::vector<int> core_r;
            std::vector<int> core_scalar;
            for (int a : core) {
                const int loc = local_of_gid[a];
                if (loc < 0) continue;  // another rank's share of this core
                const int p = static_cast<int>(
                    std::lower_bound(nr.begin(), nr.end(), a) - nr.begin());
                const int norb = ad.type_norb[gtype[a]];
                for (int o = 0; o < norb; ++o) {
                    core_r.push_back(r_off[p] + o);
                    core_scalar.push_back(row_off[loc] + o);
                }
            }
            std::vector<int64_t> dom_scalar;
            for (int q = 0; q < nd; ++q) {
                const int norb = ad.type_norb[gtype[dom[q]]];
                for (int o = 0; o < norb; ++o) dom_scalar.push_back(grow_off[dom[q]] + o);
            }
            const int nc = static_cast<int>(core_r.size());

            std::vector<double> Zc(static_cast<size_t>(nc) * nf);
            for (int j = 0; j < nf; ++j) {
                for (int i = 0; i < nc; ++i) {
                    Zc[static_cast<size_t>(j) * nc + i] =
                        Zb[static_cast<size_t>(j) * nrr + core_r[i]];
                }
            }
            std::vector<double> ZcV(static_cast<size_t>(nc) * nk, 0.0);
            {
                const double one = 1.0, zero = 0.0;
                dgemm_("N", "N", &nc, &nk, &nf, &one, Zc.data(), &nc, A.data(), &nf,
                       &zero, ZcV.data(), &nc);
            }
            // The per-table rows, assembled now: P_k = (ZcV diag(g_k)) R with
            // R = V^T (sandwich) or K (solve chain), g_k the SAME Chebyshev
            // series the recursion accumulates, evaluated at the compression
            // eigenvalues on the task's scaled interval. V, K and ZcV die
            // with this iteration.
            {
                const int nd_rows = chain.active() ? nf : nf;  // |F| scalar columns
                std::vector<double> ZVg(static_cast<size_t>(nc) * nk);
                for (size_t k = 0; k < tables.size(); ++k) {
                    for (int j = 0; j < nk; ++j) {
                        double x = (a_eig[j] - center) / half;
                        x = std::max(-1.0, std::min(1.0, x));
                        const double g = ClenshawEval(tables[k], x);
                        for (int i = 0; i < nc; ++i) {
                            ZVg[static_cast<size_t>(j) * nc + i] =
                                ZcV[static_cast<size_t>(j) * nc + i] * g;
                        }
                    }
                    std::vector<double> Pk(static_cast<size_t>(nc) * nd_rows, 0.0);
                    const double one = 1.0, zero = 0.0;
                    if (nc > 0 && nk > 0) {
                        if (chain.active()) {
                            dgemm_("N", "N", &nc, &nd_rows, &nk, &one, ZVg.data(), &nc,
                                   Kfac.data(), &nk, &zero, Pk.data(), &nc);
                        } else {
                            dgemm_("N", "T", &nc, &nd_rows, &nk, &one, ZVg.data(), &nc,
                                   A.data(), &nd_rows, &zero, Pk.data(), &nc);
                        }
                    }
                    basis.P[k][fi] = std::move(Pk);
                }
            }
            basis.frags[fi].a_eig = std::move(a_eig);
            basis.dom_rows[fi] = std::move(dom_scalar);
            basis.core_rows[fi] = std::move(core_scalar);
            nf_of[fi] = nf;
        }
    }
    // Collectively agreed, like the graph seed's failure accounting: every
    // rank finished the batched collectives above, and every rank must throw
    // together or none, or the survivors hang in the next collective.
    if (exchange) {
        int tmp_i = eig_fail;
        MPI_Allreduce(&tmp_i, &eig_fail, 1, MPI_INT, MPI_MAX, comm);
    }
    if (eig_fail != 0) {
        throw std::runtime_error("Fragment dsyevd failed, info=" + std::to_string(eig_fail));
    }
    for (int fi = 0; fi < n_mine; ++fi) {
        basis.sum_dom_orb += nf_of[fi];
        basis.max_dom_orb = std::max<int64_t>(basis.max_dom_orb, nf_of[fi]);
    }
    return basis;
}

/// The per-table operator as a view of the basis, which must have been
/// built for exactly these tables on this interval (the build assembles P
/// itself; see FragmentBasis::P). A mismatch is a caller error, not a
/// silent recomputation.
inline FragmentCv TablesToCv(const FragmentBasis& basis,
                             const std::vector<std::vector<std::complex<double>>>& tables,
                             double center, double half) {
    if (basis.tables != tables || basis.center != center || basis.half != half) {
        throw std::runtime_error(
            "Fragment basis was built for different filter tables; rebuild it.");
    }
    FragmentCv cv;
    cv.basis = &basis;
    cv.P = &basis.P;
    return cv;
}

}  // namespace detail
}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_RS_FRAGMENT_HPP
