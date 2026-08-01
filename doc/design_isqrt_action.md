# Accelerating the S^{-1/2} graph matrix function

Ownership: vbcsr provides only the GENERIC graph matrix-function machinery
(`vbcsr/core/detail/ops/spmf/graph_function.hpp`: `graph_function_apply` and
`graph_matrix_function`). Everything specific to the inverse square root — the
dense thin-action kernel, the route dispatch, and the Newton-Schulz
purification — is rsatb's, in `rsatb/calc/isqrt_action.hpp` and
`rsatb/calc/isqrtm.hpp` (namespace `rsatb::calc`, kernel in
`rsatb::calc::isqrt`).

## API

```cpp
// vbcsr - the generic graph driver:
#include "vbcsr/core/detail/ops/spmf/graph_function.hpp"

// f(A), one dense subgraph problem per block row.
vbcsr::graph_matrix_function(A, &out, std::function<T(double)> f, verbose);

// Same traversal, caller-supplied dense kernel:
//   action(total_dim, M, k_cols, col_start) replaces M with
//   f(M)[:, col_start : col_start+k_cols]
vbcsr::graph_function_apply(A, &out, action, verbose);

// rsatb - the inverse square root built on it:
#include "rsatb/calc/isqrtm.hpp"    // namespace rsatb::calc

// A^{-1/2}: the specialised thin-action kernel (isqrt::Options selects/tunes the
// route, GraphInverseSqrtReport says which routes ran).
graph_inverse_sqrt(S, &out, isqrt::Options{}, verbose, &report);

// A^{-1/2} refined by Newton-Schulz on the sparse matrix.
inverse_sqrt(S, &out, InverseSqrtOptions{...}, verbose);

// The dense kernel on its own (rsatb/calc/isqrt_action.hpp): A^{-1/2} B,
// A^{-1/2} E, or the in-place column-block form the graph loop uses.
isqrt::inverse_sqrt_action(a, lda, n, b, ldb, nrhs, x, ldx, kernel_options);
isqrt::inverse_sqrt_selector(a, lda, n, offset, nrhs, x, ldx, kernel_options);
isqrt::inverse_sqrt_columns_inplace(n, m, k_cols, col_start, kernel_options);
```

Scalar type: everything is templated on the matrix's scalar type and computes
in that type. A real matrix takes the (~3-4x cheaper) real LAPACK path, a
complex matrix the complex one — the choice is the CALLER's, made by the type
of the matrix handed in; the kernel never inspects matrix content to decide.

`rsatb::calc::ISQRTMPreconditioner` (calc/preconditioner.hpp) is the production
entry.

## The problem

`graph_inverse_sqrt` builds `S^{-1/2}` one block row at a time: for every atom `i` it
assembles the dense overlap matrix `A = S[C_i, C_i]` of the atom's whole
neighbourhood `C_i` and keeps only the block row belonging to `i`. In production
that means `n = dim(C_i) ~ 2000-5000` and a block width `b = 13-40` (one atom's
orbitals), one problem per OpenMP thread, dense SPD input, target relative
accuracy `1e-8`.

The previous route diagonalized every subgraph (`xSYEVD`/`xHEEVD`) — it produces
`n^2` numbers of which `n*b` are used. What is actually needed is the thin
action

```
X = A^{-1/2} E,      E = [e_offset, ..., e_(offset+b-1)],
```

and the wanted block row is `X^H`. `rsatb/calc/isqrt_action.hpp` computes
exactly that.

## Routes

(Real vs complex arithmetic is not a route: the kernel computes in the
matrix's own scalar type. A Gamma-point overlap is worth ~3-4x if the caller
builds it as a real matrix — measured `max|Im S| / max|Re S| = 0.00e+00` on
diamond sc888 — but that is the caller's decision, not a detection inside the
kernel.)

**1. Fast path — Cholesky + inverse block Lanczos.**
Factor `A = L L^H` once, then run block Lanczos on `M = A^{-1}` (applied with
`xPOTRS`, reusing the factor) starting from `orth(E)`:

```
A^{-1/2} E = M^{1/2} E  ~  Q_m H_m^{1/2} E_1 R_0,     H_m = Q_m^H M Q_m.
```

Inverting is the point: a troublesome overlap eigenvalue `lambda << 1` becomes a
*dominant* eigenvalue `1/lambda >> 1` of `M`, and a Krylov space resolves
dominant modes first — and only those modes that couple to the requested `b`
columns. The remaining `M`-spectrum is comparatively clustered. Cost is one
`n^3/3` factorization plus `m` block solves, `m ~ 16` in practice.

The recurrence uses the local block-Lanczos subtraction plus one full
reorthogonalization pass; a pure three-block recurrence loses orthogonality
catastrophically at these condition numbers, and a second full pass costs more
without measurable gain.

Convergence is certified empirically, not proved: every `checkpoint_stride`
blocks the nested approximations are compared,

```
d_k = ||X_k - X_{k-4}||_F / ||X_k||_F,     rho_k = d_k / d_{k-4},
```

and the route is accepted only when the estimated geometric tail
`d_k rho_k / (1 - rho_k)` falls below `0.25 * relative_tolerance`. A refused
certificate falls through to route 2.

**2. Robust path — dense-to-band + Zolotarev.**
Stage one of the two-stage tridiagonalization only (`xSYTRD_SY2SB`/`xHETRD_HE2HB`,
GEMM-rich), then the Zolotarev minimax rational form of `x^{-1/2}` evaluated
directly on the band: each pole is one banded Cholesky solve
(`xPBTRF`/`xPBTRS`, `O(n kd^2)`), and the stage-one Q is applied to the thin
block via compact-WY groups (`xLARFT`/`xLARFB`). The band-to-tridiagonal bulge
chase is never performed — LAPACK offers no thin-Q application for it anyway,
which is what previously forced the full one-stage `xSYTRD` (measured 5-9 GF/s,
96% of the old robust route's time). Spectral bounds come from the band itself
(Gershgorin above, exponential banded-Cholesky search below); Zolotarev needs
only ~15-17 poles at the production conditioning and its degree grows
logarithmically in the bounds ratio, so the factor-8 bracket costs ~1 pole.
Elliptic functions are computed by AGM/Landen — no external dependency.
Numerically singular subgraphs are handled inside the route (clamped
eigendecomposition of the band expanded back to dense, wrapped in Q).

**2b. Fallback robust path — tridiagonal + sinc rational.**
Reduce once (`xSYTRD`/`xHETRD`), apply the implicit `Q^H` to the thin block only
(`xORMTR`/`xUNMTR`), and evaluate `T^{-1/2} C` from

```
x^{-1/2} = (2/pi) \int_0^\infty dt/(x + t^2),   t = sqrt(c) e^s,  c = sqrt(ab),
```

trapezoidal in `s`: `sum_k w_k (T + s_k I)^{-1} C` with
`s_k = c e^{2kh}`, `w_k = (2h/pi) sqrt(c) e^{kh}`, `h = 0.39`. The pole count is
chosen by checking the scalar error on `[lambda_min, lambda_max]` (~145 poles at
`kappa = 1e8`); each shifted system is tridiagonal, `O(n b)`, so the `4n^3/3`
reduction dominates. Kept as the fallback for a LAPACK too old for `SY2SB`.

**3. Last resort — clamped eigendecomposition.** `f(x) = (x > 1e-12) ? x^{-1/2} : 0`,
i.e. exactly the previous behaviour. Used for subgraphs below
`direct_eigen_max_n`, for numerically singular `A` (where the clamp is what
defines the answer — screened with an `O(n^2)` `xPOCON` estimate before the
Krylov work), and whenever a LAPACK call fails.

## Dispatch

`Method::Auto`:

| condition | route |
| --- | --- |
| `n <= max(256, 4b)` | eigen (overhead-dominated) |
| `n < 24 b` | band-Zolotarev |
| router says the fast path keeps failing | band-Zolotarev |
| otherwise | fast path, falling back to band-Zolotarev |

The `24 b` threshold is the measured break-even: the fast path costs
`n^3/3 + 2 m b n^2` against the robust path's `4n^3/3`, so it only pays while
`n >> 2 m b`, and `m ~ 16`.

`isqrt::Router` caches the outcome across the atoms of one run: after 8 probes,
if 60% or more were rejected it stops attempting the trial and re-probes every
32nd problem. This matters because a rejected trial is pure overhead (~20-40% of
one problem) and neighbouring atoms have similar spectra.

Forced routes (`isqrt::Options::method`, or `RSATB_ISQRT_INIT=krylov|rational|eigen`)
still honour the *applicability* guards — a 200-row subgraph cannot support a
block Krylov space, and a singular `A` has no approximation — but disable the
convergence fallback, so a forced route that cannot converge is an error rather
than a silent downgrade. `RSATB_ISQRT_INIT=eigen` restores the plain
diagonalization exactly.

## Measured

Where the checks live: `vbcsr/core/test/benchmark_isqrt_action.cpp` (dense
kernel, synthetic spectra), `vbcsr/core/test/test_isqrt_graph.cpp` (graph driver
against the plain diagonalization), and - because it needs the two-center
integral engine - `rsatb/calc/test/benchmark_isqrt_diamond.cpp` for the real
system.

Single thread (one dense problem per thread is how the graph loop schedules
them), MKL LP64, `OMP_NUM_THREADS=1`, `kappa = 1e8`, error against the exact
spectral construction. `benchmark_isqrt_action`:

| type | spectrum | n | b | auto | route | rational | eigen | speedup | error (auto) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| real | gapped, 16 low | 2000 | 13 | 0.26 s | krylov, m=16 | 1.10 s | 1.87 s | **7.3x** | 4.1e-09 |
| real | gapped, 16 low | 2000 | 40 | 0.54 s | krylov, m=16 | 1.14 s | 1.87 s | **3.5x** | 4.0e-09 |
| real | gapped, 16 low | 3000 | 13 | 0.67 s | krylov, m=16 | 4.44 s | 6.76 s | **10.0x** | 4.1e-09 |
| real | gapped, 64 low | 4000 | 40 | 1.92 s | krylov, m=16 | 10.5 s | 15.7 s | **8.2x** | 6.8e-09 |
| real | gapped, 16 low | 4000 | 13 | 1.35 s | krylov, m=16 | 10.3 s | 15.7 s | **11.6x** | 3.7e-09 |
| real | continuous | 2000 | 13 | 1.30 s | rational (trial refused) | 1.09 s | 1.71 s | 1.3x | 1.7e-09 |
| real | two-band, 128 low | 2000 | 40 | 1.65 s | rational (trial refused) | 1.13 s | 1.80 s | 1.1x | 9.7e-09 |
| complex | gapped, 16 low | 1200 | 13 | 0.28 s | krylov, m=16 | 0.88 s | 1.38 s | **5.0x** | 1.3e-09 |
| complex | gapped, 32 low | 2000 | 26 | 1.30 s | krylov, m=16 | 4.15 s | 6.42 s | **5.0x** | 1.6e-09 |

The fast path is also *more* accurate than the eigensolver baseline here
(4e-09 against 1.3e-08): at `kappa = 1e8` the forward accuracy of `A^{-1/2}` is
limited by `cond(A^{-1/2}) ~ kappa/2`, and the thin action avoids the full
eigendecomposition's amplification. Beyond `kappa ~ 1e10` no route reaches 1e-8
and the honest remedies are physical (prune near-dependent basis combinations,
regularize the overlap), not algorithmic.

## Measured on the real system (diamond sc888)

`benchmark_isqrt_diamond` builds the actual diamond overlap at Gamma through the
op2c integral engine (1024 atoms, 13 orbitals each, overlap cutoff
`2 r_cut = 7.41 Ang`) and runs the very subgraph problems the graph loop feeds
the kernel. Single thread, MKL, per atom, `dim = 3809`, `b = 13`:

| route | time | deviation vs legacy | speedup |
| --- | ---: | ---: | ---: |
| legacy `zheevd` (production before this work) | 43.8 s | — | 1.00x |
| eigen, real input matrix | 13.2 s | 2.2e-13 | 3.3x |
| tridiagonal rational, real input matrix | 9.0 s | 3.7e-11 | 4.9x |
| **band-Zolotarev** (kd 96, 17 poles, real input) | **2.6 s** | 3.4e-11 | **16.9x** |
| auto (Krylov trial refused, then band) | 3.8 s | 3.4e-11 | 11.4x |

The "real input" rows apply when the caller hands the kernel a REAL matrix
(the Gamma-point overlap is exactly real); the production driver currently
samples S(k) into a complex matrix even at Gamma, so it pays the complex-path
cost until it builds a real S for Gamma-only runs — see "Next". Measured
end-to-end on sc888 at 64 threads: seed 415 s complex versus 147 s when the
same problems ran in real arithmetic (2.8x), all 1024 subgraphs on the band
route, 0 failures.

The band route's kd profile is flat (within ~10% over kd 64-256 here, optimum at
the small end), so the default `~n/40` needs no per-run tuning; `benchmark
--route band --kd N` sweeps it when a new machine matters. On the magic-angle
moire cell the same route gives 0.23 s/atom against 2.9 s legacy (**12.8x**),
same caveat.

Three things this settles that the synthetic sweep could not:

1. **The real spectrum is not the assumed one.** `lambda in [2.48e-03, 1.54e+01]`,
   so `kappa = 6.2e3` — six orders milder than the 1e8 design target — but the
   low end is a *broad tail*, not an isolated near-null subspace: 825 of 3809
   eigenvalues lie below `1e-2 lambda_max` (none below `1e-4 lambda_max`). A
   block Krylov space of at most `28 x 13 = 364` vectors cannot represent 825
   dominant modes of `A^{-1}`, so the certificate refuses the fast path — and it
   refuses early, at 12 of 28 blocks, for about 1.1 s of the 10.3 s. This is the
   regime where the robust route is the answer, and it is why the rational
   fallback had to be implemented rather than assumed away.
2. **`isqrt::Router` is load-bearing here, not a nicety.** Every atom in a
   homogeneous system produces the same verdict, so caching it across the run
   turns the 4.32x into the 4.83x rational figure. (The benchmark deliberately
   uses no router, to show the per-problem worst case.)
3. **The subgraph has no exploitable sparsity.** Block density 0.47 with a
   half-bandwidth equal to the full block count in atom-index order: a 47%-dense
   pattern cannot be made narrow by reordering, so sparse or banded Cholesky
   (the "remaining opportunity" of the original design note) is not worth
   pursuing for this system.

For the whole sc888 preconditioner that is 1024 x 44.4 s = 12.6 core-hours of
subgraph work today against 2.6 core-hours now.

The full graph-level path was run end-to-end on sc888 as well
(the production python route down to
`graph_inverse_sqrt`, all 1024 atoms, 64 threads, 8 GiB budget): route tally
krylov 0 / rational 1024 / eigen 0 / failed 0, peak RSS 8.0 GiB at 12
concurrent problems (~0.45 GB each, measured under the since-removed budget
cap), and ~22 min wall for the whole seed versus ~108 min projected for the
legacy per-subgraph diagonalization at the same thread count.

The same benchmark on the magic-angle twisted-bilayer cell
(`test/data/moire/conf_d6.vasp`, 11164 C atoms, same 2s2p1d basis) lands in the
same regime with smaller problems: 2D bilayer neighbourhoods give
`dim ~ 1534-1573`, `kappa ~ 7.3e3`, again a broad shallow tail (~235 of ~1550
eigenvalues below `1e-2 lambda_max`, none below `1e-4`), block density 0.58, no
usable band. Per atom (real input): legacy 2.8-3.2 s, rational 0.50-0.58 s
(**5.2-6.3x**, dev 4e-11), Krylov trial refused early at m = 12. Full-cell
subgraph work: ~9.3 core-hours legacy vs ~1.7 accelerated. The three sampled
moire atoms are *not* congruent (dims 1534/1573/1547 across the
AA/AB/domain-wall regions) - the per-problem kernel is what carries the moire
case.

`test_isqrt_graph` (rsatb/calc/test) runs the assembled thing: on block-sparse
overlap matrices (120 atoms, b=13, subgraph dim 403, 4 ranks x 8 threads) every
route reproduces the legacy diagonalization to a relative deviation of 5e-11,
for a real scalar type, a complex one, and a genuinely complex Hermitian input.
That last case carries a non-zero imaginary part on purpose - it is the only
configuration in which a transpose and a conjugate transpose differ, so it pins
the layout conventions (`to_dense` is row-major, the kernels read column-major,
and the graph loop extracts the transposed block; the two flips cancel).

## Memory

The graph loop runs one dense problem per thread, so the per-problem working set
times the thread count is what a rank holds - at `dim ~ 3800` and 96 threads the
difference between 2 and 4 `dim^2` is tens of GB. Measured on the diamond sc888
subgraphs (`--route` mode of `benchmark_isqrt_diamond`, peak RSS added on top of
the assembled overlap, in units of `dim^2` complex scalars):

| route | peak / problem | |
| --- | ---: | --- |
| legacy `zheevd` | 902 MiB = 4.07 dim^2 | dense matrix + eigensolver workspace + payload |
| eigen (this kernel) | 682 MiB = 3.08 dim^2 | in-place eigendecomposition |
| rational | **449 MiB = 2.03 dim^2** | no n x n workspace at all |
| auto (Krylov trial + rational) | 579 MiB = 2.62 dim^2 | one factorization copy + thin basis |

What buys it:

- **The routes are destructive.** `inverse_sqrt_action_destructive` hands the
  caller's buffer straight to LAPACK: the Cholesky, the tridiagonal reduction and
  the eigendecomposition all factorize in place, so no route copies the matrix.
  Only the Krylov route needs a copy, because it must leave the matrix intact for
  the fallback - which is also why it is the one route that costs extra memory.
- **The near-singular fallback reuses the reduction.** When `lambda_min` turns out
  to be below the floor, the exact clamped action is computed from the
  tridiagonal already in hand rather than by re-forming and re-diagonalizing the
  dense matrix.
- **The sparse submatrix is dropped before the solve.** The graph loop keeps
  `sub_mat` (~0.5 `dim^2`) only while `to_dense` runs, and reads the result block
  straight out of the dense buffer instead of staging it through a multivector.
- **No separate memory cap.** With the working set halved, the explicit
  per-rank budget (`VBCSR_GMF_MEM_BUDGET_GB` and its plumbing) was removed: one
  problem runs per OpenMP thread, so peak memory is OMP_NUM_THREADS x ~2 dim^2
  and the thread budget is also the memory lever - the same single knob that
  governs threading.

## Notes

- The kernel never throws: it runs inside the OpenMP subgraph loop, where an
  escaping exception terminates the process. Failures come back as a `Status`,
  are counted per rank, and `graph_inverse_sqrt` raises after the parallel region.
- The route taken near the certificate boundary depends on thread interleaving,
  so results can differ between runs at the level of the requested tolerance.
  Force a route if reproducibility matters more than speed.
- `RSATB_ISQRT_INIT=auto|krylov|band|rational|eigen` overrides the route without
  a rebuild; `eigen` reproduces the plain per-subgraph diagonalization exactly.
- Thread policy (dense_kernels.hpp): `OMP_NUM_THREADS` is the only knob. The
  vendor pool rests at that budget; the graph loop holds a `ScopedSerialBLAS`
  while its OpenMP threads each solve one dense problem, and restores the
  budget on exit. Nothing in `rsatb` sets BLAS thread counts.
- Newton-Schulz refinement densifies at scale: on sc888 one step with
  `sptol = 1e-5` takes the seed from block density 0.286 to 1.0 (a 2.8 GB
  dense preconditioner) for a residual gain of 1.9e-2 -> 9.5e-4, at ~200 s of
  SpGEMM. If the preconditioner must stay sparse, use the seed alone
  (`maxit = 0`, the production default) or a coarser `sptol`.
  `benchmark_isqrt_diamond --ns N` runs the production `inverse_sqrt` chain
  natively (with a SIGBUS/SIGSEGV forensic handler) for measuring this.

## Next

The structural opportunity (sparse or banded Cholesky on the subgraph) is now
measured and closed for this system: 47% block density, no usable band. What is
left is either physical or architectural, not kernel tuning:

- **A real overlap for Gamma-only runs.** The kernel computes in the matrix's
  scalar type, and the Gamma-point overlap is exactly real - but the driver
  currently samples it into a complex matrix, paying ~2.5-3x on every dense
  solve. Building S as a real matrix for Gamma-only runs recovers the real-input
  figures in the table above; this is a driver/typing change, not kernel work.
- **Cheaper problems, not faster solves.** The subgraph dimension, `3809` for a
  `7.41 Ang` overlap cutoff, sets the `n^3` cost. Any justified reduction of the
  neighbourhood (a tighter overlap cutoff, or dropping the far tail of the
  pattern before the graph function rather than after) buys more than any
  remaining kernel work: cost scales as `dim^3`.
- **Krylov for a milder spectrum.** At `kappa = 6e3` the near-null subspace is
  broad but shallow. A route that deflates the low tail explicitly (rather than
  representing it in a Krylov basis) may beat the rational path; the record's
  earlier tests rejected deflation at `kappa = 1e8`, where the tail was deeper,
  so that conclusion should be re-tested at the conditioning production actually
  has.
