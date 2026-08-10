# S^{-1/2} by Newton-Schulz: memory and cost design

Status: implemented, tested, merged to `main`. Written for external audit.

The target workload is a 700k-atom system. At that size the binding constraint
is peak resident memory, not flops: the iteration below holds several sparse
matrices simultaneously, and the largest of them is produced by an SpGEMM whose
intermediate is denser than its own answer. Everything here is either about not
forming that intermediate, or about not making a result resident before it holds
a value.

Two numbers frame the problem. On a 4096-atom diamond supercell the iteration
peaked at **101.4 GB** while its steady state was **31.7 GB** — a peak/steady
ratio of **3.2x**. The work below brings that to **85.6 GB / 33.3 GB = 2.57x**
with the iteration's arithmetic unchanged (identical residual trajectory, digit
for digit).

---

## 1. The iteration

`rsatb::calc::inverse_sqrt` (`rsatb/calc/isqrtm.hpp`) computes `Z = S^{-1/2}`
for a sparse Hermitian overlap matrix `S`, as a preconditioner.

A seed `Z_0` comes from `graph_inverse_sqrt`: one thin dense inverse-square-root
action per block row, over that row's neighbourhood subgraph. Newton-Schulz
purification then refines it:

```
X = Z^H S Z                                  (the residual operand; X -> I)
P = 1.875 I - 1.25 X + 0.375 X^2             (the NS polynomial)
Z <- Z P
```

Invariants the implementation relies on, and which an auditor should check are
not assumed anywhere else:

- **`Z` is NOT Hermitian.** Only `Z^H S Z = I` holds at convergence. Nothing may
  treat `Z` as symmetric.
- **`X` is exactly Hermitian**, because every route that forms it mirrors, and
  `X^2` of a Hermitian matrix is Hermitian.
- **`P` is exactly Hermitian**, because `X` is and the polynomial's coefficients
  are real. The propagate route below depends on this (`P^H = P`, so no
  transpose is materialised).
- All products are dropped at `sptol`. The iteration **cannot converge below
  sptol**; a run with `tol < sptol` plateaus, and the code says so explicitly
  rather than reporting success.

### The residual metric

`residual_norm` reports **max over blocks of `||(X - I)[i,j]||_F`**, computed
without mutating `X`.

It is deliberately not called "maxabs". It is a max block *Frobenius* norm, which
upper-bounds the largest element (`||B||_F >= max|b_ij|`), so as a convergence
test it is conservative -- it stops later, never earlier. It is block-size
dependent, and **it does not bound `||I - X||_2`**, so it is not a
convergence-basin certificate. A row-sum bound or a few Lanczos steps would be;
neither is implemented.

Every local row is accounted for whether or not it stores a diagonal block. The
earlier implementation did `shift(-1)` / scan / `shift(+1)`, and `shift` only
touches a diagonal block that is already **stored** -- so a row whose diagonal
had been dropped, or an entirely empty `X`, contributed 0 instead of `||-I||_F`.
With a loose enough `sptol` that is an immediate false "converged" on an invalid
result.

### Never stopping on a propagated residual

`X <- P X P` is not the residual of the `Z` being returned. `Z` is
`filter(Z P)`, and whatever the threshold dropped from that product -- call it
`D` -- never enters `P X P`, so the true residual operand is
`(ZP - D)^H S (ZP - D)`. The difference is **truncation, not rounding**, and the
five-step drift refresh does not bound it in the common three-step run where it
never fires at all. Measured on 4096 atoms the gap is a factor of 2.8.

So after the loop, if the last `X` came from propagation, it reforms once and
re-measures purely to REPORT. Measured on 4096 atoms the reported 1.98e-4 was
really 9.95e-4 -- a factor of 5.

That result deliberately does NOT feed the termination test, and the first
attempt did, which is worth recording as a trap. The exact residual is bounded
below by `sptol`, so "still above tol" is permanently true on any run whose
`tol` is under `sptol`; and re-entering compares an exact norm against a
propagated `norm_pre`, so the plateau detector -- which assumes a single metric
-- never fires either. A 4096-atom run that had been exiting cleanly in three
steps instead ground on for **eight hours**, its exact residual getting WORSE at
each extra termination (9.95e-4, then 1.49e-3), because at the `sptol` floor the
extra propagation only adds truncation. Report honestly; stop where the
iteration actually stopped improving.

After each step `X` must be advanced. There are two ways, and choosing between
them is the main design decision:

| route | formula | cost shape | effect on X |
|---|---|---|---|
| **reform** | `X <- Z^H (S Z)` | two products, one with the sparse `S` | X inherits Z's density |
| **propagate** | `X <- P X P` | one fused triple product, no `S` | X contracts toward I |

---

## 2. Route selection

This is the part most worth auditing, because it is a heuristic and it was
wrong in a measurable way until recently.

The decision is **collective**: rank 0's choice is broadcast. Every other input
already agrees across ranks (the step counter, and the Allreduced residual and
neighbour counts), but `seconds_per_unit` is timed locally, so `cost_reform` and
`cost_propagate` could differ between ranks. The two branches issue *different*
collective sequences, so a split decision is a hang, not a slightly worse route.
Measured durations are also `MPI_Allreduce`d with `MAX` before they update the
calibration -- a route is a collective sequence, so its cost is the slowest
rank's, and per-rank constants are what let the decision drift apart.

```cpp
drift_refresh          = ((count + 1) % 5 == 0);
x_saturated            = !(x_nbr < kSaturatedFraction * row_length);   // 0.98
cost_favours_propagate = x_saturated ? (cost_propagate < cost_reform)
                                     : (cost_propagate < kTrajectoryPremium * cost_reform);
propagate = !drift_refresh && norm < 1e-1 && cost_favours_propagate;
```

Three guards, in priority order:

**(a) Drift refresh, every 5th step.** `P X P` accumulates rounding that only
`Z^H (S Z)` clears. This is a stability bound fixed by measurement, and cost
never overrides it.

**(b) Residual guard, `norm < 1e-1`.** The model ranks cost, not stability. A
seed poor enough to leave the residual above 1e-1 is where propagation has least
margin. It can only ever force *more* reforms.

**(c) Cost, with a structural correction.** The cost model predicts each route in
seconds: block-pair `units` times a per-route measured `seconds_per_unit`. The
unit estimate uses a fill model where sparsity *radii* add:

```
nbr_product(a, b) = min(row_length, (a^(1/d) + b^(1/d))^d)
```

`d` (`fill_dim`) is calibrated at runtime by solving
`a^(1/d) + b^(1/d) = (ab)^(1/d)` against a measured `S·Z`. It defaults to 3 and
is **not** learned from a saturated product (a full row is a structural limit,
not a measurement) — that guard shares `kSaturatedFraction` with the route rule.
The `min(row_length, ...)` cap exists because radii add without bound but
neighbour counts do not; uncapped, a dense system was predicted to fill past
full, which biased the two routes unequally.

### Why cost alone is not the objective

The two routes do not leave the same `X` behind, and the cost model scores only
the current step. Measured on 4096 atoms:

- Propagation sets `X <- P X P`; `P` collapses toward the identity under `sptol`
  as the iteration converges, so **X contracts**: 2080 -> 1514 -> 933 neighbours
  per row.
- Reformation sets `X <- Z^H S Z`, which inherits `Z`'s density — and `Z` only
  ever fills in. Routed on step cost alone, **X went 2080 -> 3092 in one step.**

The cost model was *correct about the step* (predicted reform 1206 s, measured
1168 s) and wrong about the run: 3279 s / 101 GB over three propagating steps
became >3500 s in step 2 alone, past 122 GB.

What decides whether that trade is real is whether `X` has room to contract.
Saturated — a full block row, as on sc888 where `X` sits at 1024 of 1024 —
`P X P` is dense whatever the polynomial does, reform costs nothing in
trajectory terms, and routing on pure cost is a large win. With headroom,
contraction is worth buying, up to `kTrajectoryPremium` (2.0) times the reform's
cost, which one contracting step repays by shrinking every step after it.

**Audit questions.** Is `0.98` of a full row the right saturation test? Is a
fixed premium of 2.0 defensible, or should it scale with how many steps remain?
Should the model instead predict the *resulting* `X` density for each route and
compare over a two-step horizon? The current rule is a structural criterion
bolted onto a greedy cost model, chosen because it is cheap and explainable, not
because it is optimal.

### Known limitation of the cost calibration

`seconds_per_unit` falls back to the *other* route's constant when a route has
not yet run, deliberately, so the comparison stays a pure count ratio rather
than mixing a calibrated side with an uncalibrated one. The consequence: until
each route has run once, the model cannot distinguish per-route cost at all. On
a 3-step sc888 run, propagate never runs before the last step, so **every
decisive comparison used identical constants for both routes** and the model
degenerated to a units ratio.

Where both constants were learned (4096), they differ meaningfully: reform
2.35e-4 s/unit vs propagate 2.72e-4 s/unit.

As a *cost estimate* the model is unreliable — step 2 propagate predicted 535 s
against 214 s measured (2.5x over), step 3 predicted 0.39 s against 3.4 s. As a
*ranking*, it chose correctly at all six steps across both systems. Ranking is
all the router consumes, but an auditor should know the magnitudes are not
trustworthy.

---

## 3. Fused kernels

Three kernels exist to avoid materialising an intermediate that is denser than
the answer it produces.

### 3.1 `square_polynomial_upper` — `vbcsr/core/detail/ops/spmm/square_polynomial.hpp` (new)

```cpp
C = c2 A^2 + c1 A + c0 I      // upper triangle, A Hermitian, A^2 never formed
```

Replaces `spmm_hermitian` + `axpby` + `shift` + `filter`.

**Why.** `A^2` is denser than the answer whenever the polynomial contracts.
Squaring doubles the sparsity radius, but the NS polynomial collapses toward the
identity under `sptol` while the square does not. Measured on a 4096-atom step:
`X` carries 2080 block neighbours per row, the `P` that comes out carries 1628,
and the `X^2` in between reaches the full row — **39 GB materialised and then
filtered away to produce 17 GB**. The kernel holds one row of the square in
per-thread scratch instead, and drops once on the finished `P`.

**Pruning gate.** Per row, a product pair is skipped when
`||A[i,k]|| * ||A[k,j]|| < eps` with

```
eps = threshold / n_A
```

Derivation: a skipped pair contributes `c2*A[i,k]*A[k,j]`, at most `|c2|*eps`,
and there are at most `n_A` of them, so the block is off by at most
`n_A*|c2|*eps`. The two-call route gates the square at `threshold/n_A`, so its
`A^2` is off by at most `threshold` and the `c2 A^2` it forms by at most
`|c2|*threshold`. Equating leaves `eps = threshold/n_A`, with `c2` cancelling on
both sides. This is deliberately set to **parity with the route it replaces**, not
tighter: an earlier factor-of-2 safety margin bought accuracy the other route
never had, and bought it where it is most expensive — as `X` approaches the
identity the pair norms pile up just under the gate, so a gate half this wide
stops skipping almost anything and the cheap near-converged steps stay dense.
Widening it to parity measured **31% faster** on the step where it bites
(16.78 s -> 11.64 s, sc888 step 2, matched input) with **no measurable accuracy
change**.

**Accuracy is comparable, NOT uniformly better.** The header originally claimed
"strictly less error" because this route drops once where the two-call route
drops twice. That claim is false and has been corrected. The routes drop
*different* blocks — two-call decides on `|A^2|`, this one on the finished
`|c2 A^2 + c1 A + c0 I|` — so a block the polynomial cancels is kept there and
dropped here, and vice versa. On a scattered pattern this route is ~16% worse at
matched nnz, at every gate width tried. The test asserts same-order, not better.

### 3.2 `rarh_upper` — `vbcsr/core/detail/ops/spmm/rarh.hpp` (new)

```cpp
C = A B A^H                   // upper triangle, one pass
```

Named after PETSc's `MatRARt` / hypre's `RAP`. Used for `P X P`. `P X` is denser
than `P X P` (radius `r_P + r_X` while the answer collapses), so forming it was
both the largest allocation in the loop and where the threshold did most damage.
The kernel holds one row of the intermediate per thread and drops once. `P^H = P`
exactly, so neither half needs a transpose.

Stage 2 gates on an incremental norm bound accumulated from the stage-1 gate
products; an exact `block_norm` sweep was measured as a 5-7% *regression* and
was replaced.

**The stage-1 gate carries both fan-outs**, and originally did not. A skipped
pair `(k,l)` contributes `A[i,k]B[k,l]` to `(AB)[i,l]`, which reaches `C[i,j]`
multiplied by `A[j,l]^H` -- so at most `eps1 * a_max_norm`. The number of skipped
pairs reaching ONE output block is not `n_A` but the `(k,l)` grid: `n_A` rows of
`B`, each up to `|B row k|` wide. The old bound divided only by `n_A` and was
unsound. Concretely (now the `coherent` test case): 6x6 scalar blocks, `A` all
ones, `B` all 0.08, `threshold` 1. The old `eps1` is `1/12 = 0.0833` and every
pair norm is 0.08, so every pair was skipped and the kernel returned **zero**,
against an exact answer of `36 x 0.08 = 2.88` -- an error of 2.88 against a
claimed bound of 1. The gate is now scaled per `k` by `|B row k|`, which keeps
the sum telescoping to `threshold/2`. The previous random decaying test matrices
could not expose this: they never accumulate coherently.

### 3.3 `spmm_inplace` — `vbcsr/core/block_csr.hpp:1573`

```cpp
A.spmm_inplace(B, threshold);   // A <- A B, consuming A
```

Modelled on BLAS `trmm` and PyTorch's `mul_`: a named consuming operation, not a
boolean flag. Used for `Z <- Z P`, the phase with the largest transient after
`P X P`.

The numeric loop is chunked over rows (16 chunks). Row `i` of `A` is read only
while accumulating result row `i`, so after each chunk's barrier `A`'s value
pages below that row boundary are provably dead and are handed back with
`madvise(MADV_DONTNEED)`. The ghost exchange completes before the numeric loop,
so nothing another rank still needs is released.

Self-aliasing (`A.spmm_inplace(A, ...)`) is rejected: it would release rows
another row still has to read, and distributed it would release rows other ranks
fetch.

**Coverage: BSR and CSR. VBCSR falls through to the plain product** — see §5.

---

## 4. Allocation and first touch

This is where the measured peak actually was, and it took three wrong hypotheses
to find (a suspected double-copy in `filter_blocks`, then page granularity, then
the real cause).

**The cause.** Every backend built its result through a first-touch pass that
zero-filled the whole allocation on the thread partition the later apply would
use. Good for NUMA placement — and it makes the entire result resident before a
single value exists. On a 4096-atom step this single allocation inside
`filter_blocks` was the run's peak: **20+ GB resident before a row had been
copied.**

**The fix.** `BlockSpMat::make_overwritten_result` (`block_csr.hpp:1614`)
allocates a result whose every block the caller promises to overwrite, skipping
the zero pass. The copy that follows *is* the first touch, on the same row
partition the pass would have used, so placement is unchanged. It is used by
`filter_blocks`, by the CSR SpGEMM, and by `square_polynomial`'s assemble.

Per backend:

- **BSR / CSR** — `initialize_structure_deferred_touch` skips the pass outright.
  Safe because the consuming loop writes every block; both were verified to do
  so (the CSR copy-out builds its own graph adjacency from the accumulated row
  entries, so an empty row has no slots, and it throws if row widths disagree).
- **VBCSR** — `build_first_touch_structure(defer_zero = true)` skips the pass
  outright too, on the same contract. An earlier version kept per-shape,
  per-domain ranges so a caller could run the pass later via `zero_domain()`;
  nothing ever called it (every user writes every block), so the state and its
  accessors were dead and have been removed.

Placement is preserved by the WRITER, not by the allocation: `copy_kept_blocks`
and both fused assembles walk the thread-domain partition. See section 8.

**Page release.** `PagedArray::release_pages_before` returns whole OS pages with
`madvise(MADV_DONTNEED)`. Sub-page release is impossible, and BSR often puts the
whole matrix in one page — which is why an earlier whole-page-only release never
fired.

---

## 5. Known gaps

| gap | state |
|---|---|
| VBCSR `release_value_blocks_before` | **not implemented.** Shape-packed store: "blocks below row R" is a prefix of *each* shape's buffer. Those prefixes do exist — `build_first_touch_structure` fills `shape_blocks` walking domains in order with rows ascending, and domains are contiguous ascending row ranges, so every shape's buffer is row-ordered. What is missing is the per-shape count below R, which `shape_domain_offsets` gives exactly when R is a domain edge. Closing it = a per-shape release plus a consuming path chunking on domain edges. |
| VBCSR `spmm_inplace` | falls through to the plain product. Correct, no page release. Blocked on the row above. |
| VBCSR SpGEMM eager zero pass | **deliberately kept.** Redundant for correctness (the numeric loop `std::fill`s every destination block), but it is what makes placement deterministic by thread domain, which the apply plan reuses — and that loop is `schedule(dynamic, 4)`. Closing it means a domain-aligned static schedule, trading the load balance dynamic was chosen for. A measurement, not a cleanup. |

---

## 6. Files changed

### vbcsr (submodule) — 929 insertions, 129 deletions

| file | change | brings |
|---|---|---|
| `detail/storage/numa_buffer.hpp` | trait `is_trivially_default_constructible_v` -> `is_trivially_copyable_v && is_trivially_destructible_v` | `std::complex` was excluded, so `new T[]` ran a **single-threaded constructor pass over every allocation**. The largest single speedup found. |
| `detail/storage/paged_array.hpp` | `release_pages_before` via `madvise(MADV_DONTNEED)`; `resize()` zeroes only the genuinely new range; `append_page()` skips fill for mmap pages; `released_prefix_` guard | sub-page-accurate release; removes redundant zeroing |
| `detail/backend/bsr_backend.hpp` | `initialize_structure_deferred_touch`, `zero_row_range`, `release_blocks_before` | deferred first touch + page release for BSR |
| `detail/backend/csr_backend.hpp` | same three (1 value per block, so block index == element index) | same for CSR |
| `detail/backend/vbcsr_backend.hpp` | `defer_zero` param, `zero_domain()`, `has_deferred_zero()`; move ctor/assignment fixed to carry the new member | deferred (not skipped) first touch for VBCSR |
| `detail/ops/spmm/common.hpp` | extracted `FusedRowAccumulator`, `FusedProductGroup`, `block_squared_magnitude`, `fused_gemm_accumulate`, `grouped_block_gemm_batch`, `vendor_batch_profitable` | shared primitives so fused kernels use the *same* block kernels as spmm (vendor batch above the per-type crossover, else `rm_gemm`) |
| `detail/ops/spmm/rarh.hpp` **(new)** | fused `A B A^H` | replaces two-call `P X P` |
| `detail/ops/spmm/square_polynomial.hpp` **(new)** | fused `c2 A^2 + c1 A + c0 I` | never forms `X^2` |
| `detail/ops/spmm/bsr.hpp` | `run_consuming`, chunked `run_generic(consume_A)`, `make_result_matrix_deferred_touch` | `spmm_inplace` for BSR |
| `detail/ops/spmm/csr.hpp` | `run_consuming`, chunked `run_generic(consume_A)`, deferred-touch result | `spmm_inplace` + deferred touch for CSR |
| `detail/ops/spmm/vbcsr.hpp` | documentation of the retained eager pass | records a deliberate trade-off |
| `core/block_csr.hpp` | `spmm_inplace`, `square_polynomial_upper`, `rarh_upper`, `make_overwritten_result`, `release_value_blocks_before`, `bsr_thread_domains`, `bsr_zero_row_range` | public surface for all of the above |

### rsatb — 415 insertions, 56 deletions

| file | change | brings |
|---|---|---|
| `rsatb/calc/isqrtm.hpp` | cost-model route selection + saturation/headroom criterion; fused polynomial and `spmm_inplace` wiring; per-step timing; `fill_dim` calibration with saturation guard; peak/steady reporting | the algorithm in §1-2 |
| `rsatb/utils/profile.hpp` | `RSATB_PHASE_CALLS` per-call entry/peak/exit/transient reporting | pinpointed the real peak in one run after three failed hypotheses |
| `rsatb/CMakeLists.txt` | registers the three new tests | — |
| `rsatb/calc/test/test_rarh.cpp` **(new)** | dense reference, exact Hermiticity, accuracy at matched nnz, empty rows | — |
| `rsatb/calc/test/test_square_polynomial.cpp` **(new)** | as above, plus a **scattered-pattern** case | catches an ordering coupling a dense test pattern structurally cannot (see §7) |
| `rsatb/calc/test/test_spmm_inplace.cpp` **(new)** | pattern + value equivalence to `spmm`, self-alias rejection | — |

---

## 7. A coupling worth knowing about

`square_polynomial`'s `assemble` sorts each row's staged columns. That sort looks
like the optional output ordering `spgemm_sorted_output_enabled()` governs
elsewhere. **It is not, and it must stay unconditional.**
`construct_result_graph` sorts every row's columns, while the copy-out walks the
staging order writing consecutive slots from `graph_c->adj_ptr[i]`. The two
orders must agree. Putting it behind the flag silently transposes values onto the
wrong columns — it reaches the caller as a diverged iteration, not as anything
the kernel can detect.

Every unit test passed with the sort removed, because the test graph was fully
populated with ascending columns, so the staging order was *already* sorted. The
`scatter` / `scatvar` cases exist to make a later contraction reach a column
smaller than one already touched.

---

## 8. Measured results

Diamond, `sptol` 1e-4, `tol` 1e-7, 1 rank, 48 threads. Sizes come from
`diamond_sc888.vasp` through the library's own `Structure::BuildSupercell`
(`--supercell 1 1 1` / `2 1 1` / `2 2 1` = 1024 / 2048 / 4096 atoms), not
from pre-tiled POSCARs -- one base structure, and the supercell path gets
exercised as a side effect.

**Against the real baseline.** 2048 atoms (`--supercell 2 1 1`), HEAD `6b41b8d`
with vbcsr `ebd5d4e` built from a worktree and run on the same machine, same
day, same driver source -- only the library differs:

| | HEAD | current | |
|---|---|---|---|
| NS loop | 1076.9 s | 969.3 s | -10.0% |
| peak | 46.60 GB | 34.75 GB | -25.4% |
| NS steps | 4 | 3 | |
| residual | 1.9975e-4 | 1.9726e-4 | comparable |

An earlier draft of this document reported a "+16.6% time regression" measured
against `diag.log`. That was wrong: `diag.log` is an intermediate build from
partway through development, not the starting point, and its build state is not
reproducible. Comparisons against it should be disregarded. The table above is
the sound one.

**Verified in the final runs:**

| | 1024 atoms | 4096 atoms |
|---|---|---|
| `inverse_sqrt` | 367.9 s (seed 159.2 + NS 208.7) | 3747.9 s (seed 612.5 + NS 3135.4) |
| NS steps / routes | 3 — reform, reform, propagate | 3 — propagate x3 |
| peak / steady / ratio | 19.10 / 5.86 GB = **3.26x** | 85.64 / 33.27 GB = **2.57x** |
| peak phase | **the seed** (17.15 GB transient) | `P X P` |
| final residual | 1.929e-04 | 5.536e-04 |

4096 against the pre-fused baseline: peak **101.42 -> 85.64 GB (-15.6%)**, ratio
**3.20 -> 2.57x**, with the residual trajectory identical digit for digit
(`0.0191689 -> 0.00080307 -> 0.00019822`). Per-phase transients: the polynomial
phase went **27.90 -> 0.00 -> 0.00 GB** across steps; `Z·P` **32.28 -> 4.50 ->
1.49 GB**.

**Measured during development, not re-verified in the final runs:** the fused
`rarh` kernel replacing two-call `P X P` (413/408 s -> ~71 s on the step
measured); cost-model routing on sc888 (955.8 s -> 223.2 s); the `numa_buffer`
trait fix (~20% on NS time).

**Partly explained, and one cause found and fixed.** Deferring the first touch
also deferred the PLACEMENT it used to establish. The eager zero pass ran over
the thread-domain partition; the copy that replaced it ran under
`schedule(dynamic)`, so pages landed on whichever thread won the row while the
forward apply still split the matrix along `thread_domains`. Three sites had
this, all introduced here -- `copy_kept_blocks` in `filter_blocks`, and the
assembles in `square_polynomial` and `rarh`. `bsr.hpp`'s SpGEMM never did: it
runs a domain-aligned `schedule(static)` zeroing pass per chunk before its
dynamic numeric loop, which is the pattern the others now follow.

This is measured, not inferred. `vbcsr/core/test/test_numa_locality.cpp` was
extended to cover `filter_blocks` results for all three kinds; against the
dynamic copy it reports **51.1% of sampled pages node-local** and trips its
assertion, against the domain-aligned copy **100%**. Roughly half of every
filtered matrix was on the wrong node, and every phase of the loop reads
filtered matrices.

**Still unconfirmed:** whether this accounts for the whole regression. NS time on
4096 rose **2590.9 -> 3135.4 s (+21%)** against that
same baseline, systematically across phases — `P X P` 1506.9 -> 1829.8 s,
`Z·P` 150.9 -> 257.7 s, `X^2` 433.8 -> 534.0 s at matched step-1 state. `X^2`
getting slower directly contradicts the gate widening, which measured 31% faster
in isolation, so this is not kernel-level. CPU utilisation was healthy (43.6 of
48 cores). The two candidates are the deferred first touch degrading NUMA
placement (the hazard documented for VBCSR applies to BSR too) and a fragmented
machine — a 122 GB run had been killed eight minutes earlier. **One run cannot
separate them.** This is the first thing an auditor should ask about.

---

## 9. Measurement caveats

- **`/usr/bin/time` on this machine is GNU time 1.7 and is unreliable.** It
  over-reports max RSS by exactly **4x** (verified against the internal profiler:
  19.1035 vs 19.1021 GiB) and reported "317% CPU" where the true figure is 4362%.
  Divide RSS by 4; compute CPU% from user+sys.
- **The benchmark driver's verification is not production work.** It computes
  `||Z^H S Z - I||` with both products at **threshold 0.0**, deliberately
  untruncated. At 4096 atoms that costs **44 minutes and 123 GiB** — more than
  the solve it checks, and it, not the solve, is the process peak.
- The internal and external residuals differed (1.98e-04 vs 5.54e-04 at 4096)
  because the reported number was of `P X P` rather than of the returned `Z`.
  The termination reform described in section 1 closes this; the measurements in
  section 8 predate it, so their reported internal residual is still the
  propagated one.

---

## 10. Test coverage

`test_rarh`, `test_square_polynomial`, `test_spmm_inplace`, `test_spmm_hermitian`
— all pass at **1, 2 and 4 ranks**, each covering `bsr`, `vbcsr` and `csr`
matrix kinds. Python suite: 81 passed, 1 skipped. vbcsr adds
`test_vendor_batch_gemm` (§13).

Multi-rank matters specifically for `spmm_inplace`: the 1-rank result passes
whether or not the mid-product page release is correct, because there are no
ghosts. Only 2/4 ranks exercise it.

---

## 11. External audit, round 1

An external reviewer raised eight issues. Disposition:

**Fixed.**

| finding | fix |
|---|---|
| Ranks could choose different routes (rank-local timing constants driving a branch with different collective sequences -- a hang, not a bad route) | Decision broadcast from rank 0; durations `MPI_Allreduce`d with `MAX` before updating calibration |
| A missing diagonal block made the residual falsely zero | `residual_norm` rewritten: no mutation, every row accounted for, absent diagonal contributes `||-I||_F` |
| Metric named "maxabs" but computed Frobenius | Renamed throughout to "max block residual"; the semantics are documented above |
| Propagated residual is not the residual of the returned `Z` | Reform-and-re-measure before any termination that would report a propagated residual |
| `rarh` stage-one pruning bound omitted the second fan-out | Gate scaled per `k` by `|B row k|`. The reviewer's 6x6 counterexample is now the `coherent` test case; it fails against the old gate (error 2.88 vs bound 1.0) and is exact against the new one |
| `rarh::assemble` eagerly zero-touched the full result while staging was live | Uses `make_overwritten_result`, as `square_polynomial` already did |
| Peak and steady RSS reduced with independent `MPI_MAX`, so they could come from different ranks | `MPI_MAXLOC` on the peak, then the steady value is broadcast from that rank |
| `mean_krylov_blocks` used a local numerator over a local denominator | Both globally reduced |

**Accepted, not implemented.** Each is a design change that needs measurement,
not a defect with an obvious fix:

- **Adaptive drift refresh** driven by accumulated dropped-block error rather
  than a fixed period of 5. The termination reform above removes the *reporting*
  hazard; the fixed period remains a heuristic.
- **Memory-budget constraint in route selection.** The stated objective is
  peak-memory survival, but the router optimises time only. Rejecting routes
  whose predicted live set exceeds a per-rank budget, then minimising time among
  the feasible ones, is the right shape.
- **`rarh` halo scaling.** It fetches the union of all remote rows reached by all
  locally owned rows; with poor partitioning that approaches replicating `A` and
  `B` per rank. Tiling local rows with bounded per-tile halo exchange is the fix.
- **Cost model uses products of global mean neighbour counts** -- product-of-means
  is not mean-of-products, and it ignores degree skew, variable block dimensions,
  and communication entirely. Weighted local degree sums or sampled symbolic
  counts, fitted as `time = fixed + pair_cost * weighted_pairs + comm_cost`,
  would be better founded.
- **`complete_hermitian_consuming`.** The upper triangle and the completed matrix
  currently coexist; mirroring with page release would remove that overlap.

**Noted, not a defect introduced here.** The repo has **no** `add_test` or
`enable_testing()` anywhere, so the new tests follow the existing convention of
build targets run manually. That is a real CI gap -- these tests can compile and
never run -- but adopting ctest is a project-wide decision. Separately,
`rsatb/calc/test/test_isqrtm.cpp` is stale (references the removed
`options.factor`, and returns success while printing failure); it is not a build
target, so it is dead code rather than a broken test.

---

## 12. External audit, round 2

**Fixed.**

| finding | note |
|---|---|
| `rarh_upper` validated only the backend kind, while the implementation indexes one operand's global block space with the other's indices | `ensure_product_operands_compatible` added (communicator via `MPI_Comm_compare`, rank ordering, global block count, owned distribution) and applied to `rarh_upper` and `spmm_inplace`. The unchecked case was an out-of-bounds **write**, not a wrong answer |
| VBCSR `zero_domain()` / `has_deferred_zero()` were dead state, and this document described them as a pass that runs later | Removed; `defer_zero` now skips outright like CSR/BSR, and the description is corrected |
| `global_max_norm()` did an `MPI_Allreduce` whose result was used only as a nonzero test | Direct consequence of the round-1 gate fix, which removed the norm from the denominator. Now a local `has_nonzero_block` |
| `vendor_batch_profitable()` called `getenv` from the grouped dispatch path | Resolved once into a function-local static |
| `profile_rss_gb()` hardcoded 4096-byte pages | `sysconf(_SC_PAGESIZE)` |
| `release_pages_before` re-advised the whole prefix on every call | Tracks `released_through_`; a consuming product's 16 chunk calls no longer re-advise quadratically |
| `RARhExecutor::gemm_accumulate` unused wrapper; `t_poly` permanently zero and printed | Removed |
| `PagedBuffer` comments described `munmap` / null pages; `spmm_inplace` comment said BSR-only | Corrected -- the mapping survives `MADV_DONTNEED`, which is *why* the grow guard is needed, and CSR consumes pages now too |

**Accepted, not implemented** -- these are structural and each needs its own
measurement:

- **RARH halo replication** -- characterised, NOT closed, and the tiling attempt
  was reverted. The distributed path fetches the union of every remote row its
  local rows reach, which at large rank counts approaches a copy of both
  operands per rank.

  A tiled payload fetch was implemented and measured (4 ranks, 2048 atoms,
  against the same run with tiling disabled). It was a clear regression:
  `P X P` took **1796.7s against 532.6s on identical step-1 inputs, 3.4x
  slower**, and the peak went **UP**, 31.49 GB against 30.50 GB. The trajectory
  then diverged because the cost model learned the inflated propagate constant
  and switched routes -- a consequence of the slowdown, not a correctness break
  (tiled and untiled results agree exactly; that test held).

  Two things the measurement settled. The slowdown was the per-tile support
  derivation rebuilding a `std::set` over the tile's rows x their columns x
  those rows' columns -- work the untiled path does ONCE into a flat array, and
  a tree instead of an array on top of that. That is fixable. The absence of a
  memory win is not: at this scale the halo is a few GB against a 30 GB peak,
  so it is simply not what sets the peak, and no implementation of tiling can
  produce a saving that is not there.

  Reopening it needs a configuration where the halo actually dominates -- far
  more ranks than 4, or a partition bad enough that a rank's reach approaches
  the whole matrix. Until such a case exists to measure against, tiling costs a
  large measured amount for a benefit that is currently hypothetical.

- **The kernels stage the whole answer.** Both executors keep one dense
  intermediate row per thread, which is what the design claims, but they also
  retain every finished result row in `row_values` before building the matrix.
  So the sparse answer does exist twice, briefly, plus one heap allocation per
  row. An `UpperResultBuilder` owning pattern staging, ordered values, graph
  construction, first touch, coverage validation and backend attachment would
  remove both the duplication and the per-row allocation.
- **Storage internals are public.** `make_overwritten_result`,
  `bsr_thread_domains`, `bsr_zero_row_range` and `release_value_blocks_before`
  are all reachable from user code; the first rests on an unenforced promise that
  every block gets written, and the last can invalidate live contents. They
  belong behind a builder or friend access, with the builder asserting every
  graph slot received exactly one value.
- **Duplication between the two fused kernels** (row directories, flattened
  remote rows, request conversion, visitation, staged sorting, result-graph
  construction, deferred-touch copy) should be shared -- while keeping the two
  contraction loops separate, since a callback-driven universal kernel would
  recreate the complexity somewhere harder to debug.
- **`common.hpp` is a 689-line grab bag** and should be split
  (`block_gemm_dispatch`, `fused_row_accumulator`, `remote_rows`,
  `row_pattern_exchange`, `upper_result_builder`, `symbolic_spmm`), with the
  fused operations moved out of the path every `BlockSpMat` translation unit
  pulls in.

---

## 13. The vendor-batch gate

Carried out of the round-2 audit as the one item left undecided. `spmm` and both
fused kernels contract a group of block products that share one left operand and
have distinct destinations, which is exactly the shape MKL's `?gemm_batch`
wants. Whether to use it was gated on block size alone: `bs >= 16`.

Re-measured end-to-end (`spmm` on identical operands, batch forced off vs on via
`VBCSR_SPGEMM_BATCH`, min of 5-9 runs depending on block size — 9 through the
decision band, fewer where the gap is large enough not to need them; ratio =
vendor / `rm_gemm`, below 1 the vendor wins):

| bs | 13 | 16 | 18 | 20 | 21 | 22 | 26 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| `complex<double>` | 1.11 | 1.02 | 1.005 | 1.01 | **0.92** | 0.93 | 0.90 | 0.84 | 0.77 |
| `double` | 1.89 | 1.57 | — | 1.23 | — | 1.13 | 1.00 | 1.13 | 1.00 |

Two findings, and the second is the real one:

1. **Complex crosses over at bs 21, not 16.** At 16 the vendor is 2% slower, and
   16-20 is a wash. Small, but the gate was on the wrong side of it.
2. **Real `double` never crosses over at all** — measured out to bs 96 (1.05).
   At bs 16, where the gate switched it on, the vendor batch was **56% slower**.

That is what the amortisation argument predicts and nobody had checked: the
vendor's fixed per-call cost is repaid by the *flops* in the call, and a complex
block product carries ~4x the flops of a real one at the same block size. One
block-size threshold cannot serve both scalar types, so `vendor_batch_profitable`
is now a template on `T` — bs >= 21 for `complex<double>`, never for `double`.

rsatb itself is unaffected: carbon is 13 orbitals, below both thresholds and
below the old one. The exposure was for any other vbcsr consumer with a heavier
basis, which is why it was worth fixing in a generic library.

**Err high when re-deriving this elsewhere.** The two directions are not
symmetric: below the crossover the penalty reaches 2.2x, above it the gain tops
out near 1.2x. A threshold set too low costs far more than one set too high.

### What the measurement does not cover

One machine, one BLAS (MKL 2024.2.2, 48 threads). The crossover is a property of
the vendor's call overhead and will move elsewhere. The constants are therefore
documented with the method rather than presented as universal, and
`VBCSR_SPGEMM_BATCH=1/0` reproduces the table on a new machine.

### Test

`vbcsr/core/test/test_vendor_batch_gemm.cpp` — the vendor path had **no**
coverage. Every other vbcsr test runs block sizes 1-3, so the batch was never
reached from a matrix-level test.

It drives `grouped_block_gemm_batch` directly rather than through `spmm`,
because routing to it depends on rank count, page count and block size at once —
a matrix-level test that *looked* like it covered this mostly would not. That is
not hypothetical: the first attempt here added a bs-21 profile to
`test_numeric_reference` and it never reached the batch, because
`can_use_mkl_serial` sent single-rank products to MKL's sparse BSR routine
instead. The test now asserts `grouped_block_gemm_batch` returned true, so it
cannot silently degrade into comparing the fallback against itself.

Destinations are seeded non-zero (the batch runs with `beta = 1` and must
accumulate, not overwrite), both scalar types run, and the sub-`kMinBatch` case
is pinned to decline *without* having written anything. Verified against two
deliberate mutants — `beta` 1 -> 0, and `CblasNoTrans` -> `CblasTrans` — each of
which the test catches.
