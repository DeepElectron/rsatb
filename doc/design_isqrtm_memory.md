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

So after the loop, every run reports `||(Z^H S Z - I) v||_2` for a fixed
random unit probe `v` -- three sparse matrix-vector products, WITHOUT forming
`Z^H S Z`. An earlier version instead reformed the product once purely to
re-measure (on 4096 atoms the propagated 1.98e-4 was really 9.95e-4, a factor
of 5), but that re-ran the largest phase of the solve to produce a log line:
1389 s, and the 4096-atom peak went 85.6 -> 107.0 GB, undoing the memory
saving the whole design exists for. The probe is not a norm and does not claim
to be one -- the preconditioner is APPLIED to vectors, so the drift of
`Z^H S Z v` from `v` is the error a caller sees; one probe can miss the worst
direction, which is the honest price of three products. It is cheap enough to
run unconditionally, so every run reports the same quantity whichever route
the last step took.

That number deliberately does NOT feed the termination test, and the first
attempt at honest reporting -- the reform-based one -- did, which is worth
recording as a trap. The exact residual is bounded below by `sptol`, so "still
above tol" is permanently true on any run whose `tol` is under `sptol`; and
re-entering compares an exact metric against a propagated `norm_pre`, so the
plateau detector -- which assumes a single metric -- never fires either. A
4096-atom run that had been exiting cleanly in three steps instead ground on
for **eight hours**, its exact residual getting WORSE at each extra
termination (9.95e-4, then 1.49e-3), because at the `sptol` floor the extra
propagation only adds truncation. Report honestly; stop where the iteration
actually stopped improving.

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

**Coverage: all three kinds.** VBCSR was the last (see §5): its executor
defers GEMMs into per-thread batches holding raw pointers into `A`, so its
chunk boundary also flushes the batches before releasing; and its result is
built deferred with a per-chunk domain-aligned placing touch, because the
eager zero pass would have made the whole result resident beside the whole
operand — exactly the peak this method exists to avoid.

### 3.4 `rhar_upper` — `vbcsr/core/detail/ops/spmm/rhar.hpp` (new)

```cpp
C = A^H B A                   // upper triangle, one pass, A NOT assumed Hermitian
```

The sibling of `rarh_upper`, as PETSc pairs `MatRARt` with `MatPtAP` — and a
separate kernel rather than a `transA` flag, because the two rest on different
contracts. `rarh` reaches its second contraction through `A[j,l]^H == A[l,j]`
and so *requires* `A` Hermitian; `rhar` requires nothing of `A`. That is the
contract the reform `X = Z^H S Z` actually presents: `Z` is a thresholded
polynomial of `S`, Hermitian in neither values nor — after per-block drops —
pattern, which is why the reform could not use `rarh` and previously ran as
`S Z` (spmm) followed by `spmm_hermitian_upper(..., transA=true)`. That
`transA` path is transpose-then-multiply: it materialised a **full copy of
Z^H**, on top of the **S Z intermediate** — two Z-sized matrices for the
densest, never-contracting matrix in the loop.

Row `i` of `C` is `(column i of A)^H B A`:

- Column `i` of `A` is never materialised. Local blocks — the overwhelming
  majority on a locality partition — are read in place through a transpose
  *index* and conjugate-transposed block by block into thread scratch, an
  O(dim²) copy against the O(dim³) products each one feeds across `B`'s row.
  Boundary blocks (rows owned elsewhere, column owned here) are **pushed** by
  their row owners, already conjugated — the receiver cannot request them,
  because the transpose pattern is exactly what it does not know. One
  surface-sized exchange.
- The second contraction reads rows of `A` in natural layout through the same
  remote-row fetch `rarh` uses. `B` must be Hermitian so that `C` is; `S` is,
  exactly, by construction.

Same per-row error budget as `rarh` (a stage-one neglect is multiplied by one
more factor of `A`, so the gate divides by the global max block norm), one drop
on finished values, one staged row per thread, deferred-first-touch assembly.
Flop count is the same pair count as the two products it replaces. The fill
calibration that used to read the `S Z` intermediate now solves
`nbr(X)^(1/d) = s^(1/d) + 2 z^(1/d)` from the reform's own result — same
additive-radius model, same post-threshold bias, measured on the triple product
instead of its half (`isqrtm.hpp`, `calibrate_fill_dim`).

Measured, one reform at the 2048-atom diamond shape (synthetic operands, 2048
block rows, bs 13, S 0.74 GB / Z 3.82 GB / X 7.22 GB, threshold 0 so both
routes keep the identical 2,866,048 blocks; 48 threads, `VmHWM` scoped to the
route with `clear_refs`):

| route | peak | over resident operands | time |
|---|---|---|---|
| `S Z` then `transA` product | 16.51 GB | 11.91 GB | 51.2 s |
| `rhar_upper` | **12.62 GB** | **8.01 GB** | **38.8 s** |

−3.9 GB of peak — one `Z` copy plus part of the intermediate — landing within
0.85 GB of the `S + Z + X` floor, and −24 % time. Note the *whole-run* peak of
the standard benchmark does not move at these sizes: at 1024 atoms the run
peaks in the seed's dense batches, at 2048 in the `P X P` step, and both runs
route around the loop reform (2048 routes propagate throughout; the drift
refresh only fires at step 5). The reform-phase saving matters exactly where
the reform runs at scale: the unconditional seed reform, every-fifth-step
drift refreshes on long runs, and saturated systems whose router picks reform
on cost.

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

### 4.1 The staged rows, and what the profile hooks could not see

Both fused kernels stage the answer row by row and then copy it into the result,
so the obvious question is whether staging and result are live at once. The
`VBCSR_PROFILE_RARH` hooks read RSS either side of `assemble` and said mostly
no — assembled ≈ staged at 512–2048 blocks, +21% at 4096.

**That reading was an artifact.** Two point samples cannot see a peak that rises
and falls between them. Measured instead with `VmHWM`, reset per call by writing
`5` to `/proc/self/clear_refs` so the high water mark belongs to the call, the
transient was 0.5–0.6× the result *at every size*, not zero below 4096.

Reading `VmHWM` at each stage boundary — it only rises, so a step in it names the
stage that raised the peak — split that transient into two causes, neither of
them the co-residency the hooks were built to look for:

| stage | Δ peak | Δ resident | what |
|---|---|---|---|
| `fused_rows` | +5.98 GB | +5.98 GB | the staged answer itself (5.72 GB) |
| ordering pass in `assemble` | **+2.20 GB** | +1.90 GB | a second buffer per row, to permute |
| result graph, `make_overwritten_result` | +0.02 GB | +0.01 GB | deferred touch working as intended |
| copy-out | **+3.33 GB** | +1.96 GB | freed staging not returned to the OS |

*(4096 blocks × 200 neighbours, block size 13, complex, 48 threads.)*

1. **The ordering pass permuted out of place.** The row accumulator hands columns
   back in first-touch order; the result graph is ordered. `assemble` reconciled
   them by building a sorted copy of every row and swapping. Both kernels now
   stage in column order directly — the surviving accumulator slots are sorted as
   *indices* (a few hundred ints) before the values are copied out — and the pass
   is gone. Rows are also `reserve`d to their exact final size, which removes the
   geometric growth that left staging holding 9.70 GB of capacity for 5.72 GB of
   answer.
2. **`std::vector<T>().swap(v)` frees to the allocator, not to the kernel.** glibc
   returns a large block by `munmap` only while it is above the mmap threshold,
   and that threshold is *dynamic*: freeing one mmapped chunk raises it to that
   chunk's size, so the first staged row returned is also the last. The copy pass
   wrote a 5.72 GB result while freeing 5.72 GB of staging and RSS still rose
   3.9 GB. `release_and_drop` (`detail/ops/spmm/common.hpp`) advises the buffer's
   interior pages `MADV_DONTNEED` before dropping it — the same instrument
   `PagedBuffer::release_pages_before` uses, and a statement to the kernel rather
   than to malloc: `free()` writes its own metadata into the chunk afterwards, a
   page or two faulted straight back, and never reads what was there.

Peak resident, 48 threads, complex, block size 13, 200 neighbours per row:

| kernel | n | result | before | after | floor (operands + result) |
|---|---|---|---|---|---|
| `rarh_upper` | 1024 | 1.09 GB | 2.87 GB | 2.29 GB | 2.15 GB |
| | 2048 | 2.63 GB | 6.58 GB | 4.87 GB | 4.72 GB |
| | 4096 | 5.72 GB | 15.95 GB | **10.12 GB** (−37%) | 9.88 GB |
| `square_polynomial_upper` | 1024 | 0.83 GB | 2.38 GB | 1.95 GB | 1.89 GB |
| | 2048 | 1.86 GB | 5.16 GB | 4.06 GB | 3.92 GB |
| | 4096 | 3.92 GB | 11.26 GB | **8.20 GB** (−27%) | 8.08 GB |

Both now land within 0.25 GB of holding nothing but their operands and their
answer. Runtime is a wash — at 4096, `rarh_upper` min-of-4 went 16.20 s → 15.80 s
— which is the expected trade: a full permutation copy of the answer removed, a
few hundred index sorts and one `madvise` per row added.

### 4.2 The completion, and why it is the mutating one

Both fused kernels produce only the upper triangle; `complete_hermitian` then
builds the full matrix from it. Measured across a whole update at 4096 atoms
(block size 13, 200 neighbours, 48 threads), the completion — not the kernel —
was where the step peaked:

| | before §4.1 | after §4.1 | after §4.2 | floor |
|---|---|---|---|---|
| step peak | 24.26 GB | 21.50 GB | **16.88 GB** | 15.55 GB |

The floor is the two operands (4.13) plus the answer (11.43): what the step would
cost holding nothing else. The 5.95 GB the step sat above it was **exactly the
upper triangle** (5.72 GB), standing whole beside the completed matrix it was no
longer needed for. `complete_hermitian` returns a new matrix, so the triangle had
to stay readable until its last block was mirrored — and then stayed anyway,
until the enclosing scope ended.

`complete_hermitian_inplace()` releases each row as it is mirrored. What is left
over the floor is 1.32 GB, which is one chunk of the source (5.72/4): the release
runs on 4 chunks, so one chunk's worth is always still held. That is the design's
own bound, not a leak.

**Why in place rather than a consuming overload.** `complete_hermitian` is
`const`. An implementation that frees its own object under that signature would
be lying, and there is no honest way to spell "returns a new matrix, and also
destroys the one you called it on". `spmm`/`spmm_inplace` already split exactly
this way, for exactly this reason — the in-place form exists so the operand's
pages can be released mid-product. "In place" means the VARIABLE, not the
allocation: the result has a different graph and about twice the blocks, so it
cannot be written into the old storage.

**The trade this forces.** `conjugate_transpose` built its result through the
plain constructor, which runs the first-touch zero pass and so places the
result's pages by thread domain. Consuming cannot keep that: the pass makes the
result fully resident *before* a single source row has been released, which is
the peak being removed. The two are mutually exclusive, so the consuming path
skips it and the plain path keeps it.

**Measured, and it costs nothing downstream.** Alternating runs on an idle
machine, 2048 atoms, timing the NEXT product with the completed matrix as its
operand -- `square_polynomial_upper`, exactly what the loop does with it:

| | copying | in place |
|---|---|---|
| step peak | 10.03 GB (3 runs, identical) | **7.83 GB** (3 runs, identical) |
| the completion itself | 0.60 / 0.73 / 0.80 s | 1.35 / 1.70 / 1.80 s |
| next product on the result | 43.46 / 44.45 / 44.17 s | 44.41 / 38.40 / 40.42 s |

No downstream penalty -- the in-place mean is lower (41.1 s against 44.0), which
is within the spread rather than a win, but it is certainly not a loss. An
earlier reading that showed in place 2-7% slower was taken while another job held
the machine at load 100 and absolute times drifted 44 -> 58 s mid-comparison; it
measured the contention.

The completion itself does get about 0.8 s slower, consistently, and that is
real: the writes now fault their own pages in rather than finding them faulted by
the zero pass, and each chunk boundary costs an madvise and its TLB shootdown
across every thread. It is ~2% of a step whose other stages run 8 s and 40 s.

**The placement itself is genuinely lost, though, and that is worth stating
separately from the timing.** `test_numa_locality` samples value pages and asks
whether each sits on the node of the thread whose domain owns its row:

| | node-local |
|---|---|
| copying completion | 100.0% |
| in-place completion | 27.3 / 44.0 / 54.9 / 66.8 / 72.7 / 72.7% |

Mean ~53% on a two-node machine, where chance is 50%. So it is not *partially*
aligned — it is **uncorrelated** with the domain partition, and no assertion can
separate success from luck on that path, which is why the case is reported rather
than asserted. Placement follows whichever thread wins each dynamically scheduled
row, so the figure moves run to run.

A per-chunk placing touch does not rescue it, and the reason is worth recording
so it is not re-attempted: the local writes are a scatter (source row i lands in
result row j > i), so by the time a chunk is reached its rows may already hold
values written from earlier chunks, and zeroing them to place them would destroy
those. Touching in DESCENDING chunk order would be safe — the only writers into a
chunk's rows are chunks at or below it — but then the dead rows are a suffix and
the release takes a prefix. The two orders are incompatible, and the release is
worth more.

That this costs no time is structural rather than lucky: nothing downstream reads
this matrix by domain. Both fused kernels take their operands with `schedule(dynamic, 8)` — the
thread that reads a row is whichever picks it up, not the row's domain owner — so
the domain partition of an *operand* is not exploited. It is exploited when a
matrix is being WRITTEN (the first touch) and by `copy_from`; the completed X and
P are neither.

The value ordering also had to change. The source is read by two passes — the
remote pack and the local writes — so a row is dead only once both have seen it.
The remote rounds now run first, which makes "read by everything" a row prefix,
which is what `release_values_below_row` takes. For the plain path the order is
immaterial: the two write disjoint slots of an already-placed result.

---

**The lesson for the next measurement.** `VBCSR_PROFILE_RARH` and
`VBCSR_PROFILE_BSR_SPGEMM` sample RSS at phase boundaries. That is the right
instrument for *which phase allocates* and the wrong one for *what the peak is*;
here it reported 0% where the true answer was 20%. Use `VmHWM` with
`clear_refs` for peak questions.

---

## 5. Known gaps

| gap | state |
|---|---|
| VBCSR value release | **closed.** The contract was stated as a block index into the value array — a CSR/BSR concept, since only those hold one array in `adj_ind` order. It is now `release_values_below_row(row)`, the unit all three callers were already converting into, and each backend converts in its own layout; VBCSR releases a prefix of every shape's buffer, rounded down to a thread-domain edge (releasing fewer pages than permitted is always sound, and it keeps the alignment out of shared caller code). Measured on `filter_blocks`, which calls it and is kind-agnostic — 4096 rows, 400 neighbours, mixed block sizes, 2.95 GB payload, 48 threads: VBCSR peak **5.69 → 3.73 GB**, overhang 1.01 → 0.34× payload. An overhang of 1.01× is a full duplicate: VBCSR had been holding the whole source alongside the whole filtered copy, every time. BSR unchanged at 3.78, as it already had the release. |
| VBCSR `spmm_inplace` | **closed.** `VBCSRSpMMExecutor::run_consuming` chunks the numeric loop (4 chunks, bsr.hpp's measured constant), flushes each thread's queued product batches before the chunk release — the tasks hold raw pointers into `A` — and builds its result deferred with a per-chunk domain-aligned placing touch (100% node-local in `test_numa_locality`; the eager pass would have stood the whole result beside the whole operand). Measured on a mixed-block `Z <- Z P` (Z 2.73 GB, P 2.37 GB, result 4.58 GB, 48 threads): peak 10.83 → **8.72 GB** against the plain product, at +12% time — the same memory-for-barriers trade bsr.hpp prices. |
| VBCSR SpGEMM task queues | **found and capped while closing the row above.** The executor deferred every block pair's GEMM into per-thread batches flushed once per pass — 24 bytes per PAIR, the fastest-growing quantity in the product. Measured at 2048 rows × ~820 × ~700 neighbours: **~29 GB of task queues beside a 4.6 GB result** (peak 38.94 GB), and the churn cost time too. Queues now flush at a row boundary past 2^18 tasks; grouping saturates far below that because execution chunks every batch to 1 MB scratch anyway. Plain `spmm` on the same product: 38.94 → **10.83 GB and 46.0 → 28.7 s (−38%)**. Every VBCSR-kind product benefits, not just the consuming path. |
| Reform's `transA` transpose copy | **closed** (§3.4). The reform's second product went through `spmm_hermitian_upper(..., transA=true)`, whose implementation is transpose-then-multiply — a full materialised `Z^H` on top of the `S Z` intermediate. `rhar_upper` computes `Z^H S Z` in one pass from row-distributed `Z`; measured −3.9 GB and −24% time on the reform phase at the 2048-atom shape. The `transA` flag itself still materialises the transpose for any *other* caller of `spmm_hermitian_upper`; only the reform stopped paying it. |
| Transpose exchange unbounded on the receive side | **closed.** The chunked value exchange bounded what each rank *packs* per round (`VBCSR_TRANSPOSE_CHUNK_MB`, default 256 MB) but not what it takes in: a rank fed by every other rank at once saw one budget *per sender* per round — unbounded in the rank count, tens of GB on a hot receiver at cluster scale. The metadata exchange already tells each receiver exactly how much every source holds for it, so it now cuts its budget into per-source shares proportional to those holdings and grants them up front (one extra `MPI_Alltoall`, 8 bytes per rank); senders close round boundaries against both their own budget and every destination's allowance, priced at segment granularity through `plan_remote_groups`' prefix table. Proportional shares keep the round count at max over receivers of (incoming/budget) — the minimum any receive-bounded exchange needs. Measured on the worst case, a column strip funnelling 8 ranks' blocks into rank 0 (0.5 GB/rank source, 256 MB budget): rank 0's transient **5.79 → 4.28 GB** (the removed 1.5 GB is exactly (P−2)×budget; what remains is the result itself), time 2.1 → 1.6 s. The balanced band control is unchanged-to-slightly-better on both counts. Boundaries stay a row prefix, so the consuming release ordering is untouched. |
| Spectral window not decomposition-invariant | **known, not closed.** `emax` = 271.1921 / 271.2007 / 271.2051 at 1/2/4 ranks on diamond sc444, each reproducible to 12 digits at a fixed rank count. PRIMME seeds its own initial basis from the rank (`iseed[0] = procID % 4096`) and each rank fills only its local rows, so the start differs per partition; the 1e-3 tolerance is loose enough for that to survive into the answer. Inside spec (0.013 eV spread against a 0.27 eV residual allowance, and `bound_pad` gives ~2.8 eV of margin), but it is now the largest source of run-to-run DOS variation — 7.9e-4, against 1.1e-10 with the bounds pinned. Closing it = hand PRIMME an initial basis from the counter-based sampler. See `rsatb/setup/spectral_bounds.hpp`. |
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

**Re-measured on real diamond after the staging (§4.1) and completion
(§4.2) work.** 2048 atoms, `sptol` 1e-4, 48 threads, idle machine, peak polled
from `VmHWM` externally (the internal sampler does not collect in this
benchmark, and GNU time 1.7 here over-reports RSS 4x — §9):

| | §8 table, pre-work | re-measured |
|---|---|---|
| peak RSS | 34.75 GB | **31.21 GB** (-10.2%) |
| steady | — | 11.78 GB |
| residual trajectory | 0.0191689 / 0.00080307 / 0.00019822 | 0.0191865 / 0.000875553 / 0.000197259 |

Read this as **-10% on a whole run**, not the -21% §4.2 measures on a step: the
seed phase is 300 s of the 1500 s here and none of the work above touches it.
Two limits on the comparison — the pre-work figure came from the internal
sampler and this one from an external poll, which may not be scoped identically;
and this is measured against a published number rather than a paired A/B, since
reverting the change set across both repos to re-run a baseline was not done.

**What the run says about which matrix sizes the problem**, which matters more
than the peak itself for extrapolating:

| step | Z nbr/row | X nbr/row | rss |
|---|---|---|---|
| 1 | 293 | 1590 | 13.08 GB |
| 2 | 1686.5 | 1401.4 | 26.52 GB |
| 3 | 1683.5 | 657.4 | 19.25 GB |

**Z fills 5.8x in one step and then plateaus; X contracts.** So Z sets the
memory, and an extrapolation keyed on X's neighbour count is keyed on the matrix
that collapses. Per-atom at 2048: 15.2 MB peak, 5.75 MB steady; the 4096 row
above gives 20.9 and 8.12, so the per-atom cost is still RISING between them --
the interaction sphere is not yet contained. A 14 A decay sphere in diamond
holds ~2030 atoms, which is about where X lands at 4096 (2080) and where the
count should stop climbing. Taking 4096 as the saturation point puts a million
atoms near **8 TB steady and 16-19 TB peak** -- 130-150 nodes at 125 GB. That
extrapolation rests on saturation happening at 4096, which is one measured
point for X and none for Z, so it is the number most worth checking next.

**Re-measured again after the fused reform (§3.4) and the dead-P releases.**
Same 2048-atom configuration, idle machine, external `VmHWM` poll:

| | previous re-measure | after §3.4 |
|---|---|---|
| peak RSS | 31.21 GB | **30.93 GB** |
| total | 1500.0 s | **1306.8 s** (-12.9%) |
| NS loop | 1199.5 s | 1007.4 s |
| seed | 300.6 s | 299.4 s |
| residual trajectory | 0.0191865 / 0.000875553 / 0.000197259 | 0.0191865 / 0.000874253 / 0.00019782 |

The step-1 residual is IDENTICAL to six digits — both runs form the same
`Z^H S Z` after the seed, by different routes — and step 2 pairs within 2%
across all three unchanged phases, bounding machine noise. The peak barely
moves because this run routes propagate at every step (both runs do), so the
in-loop reform never executes and the whole-run peak stays `P X P`-bound; the
reform-phase saving is the §3.4 kernel A/B, invisible from here. The -13%
time is real and downstream: the fused reform's single-drop X is slightly
tighter (P 1405.6 -> 1367 nbr/row out of the same X), and step 1's polynomial
and `P X P` both inherit it (808.4 -> 621.2 s). The seed — one full reform —
is a wash in time, confirming the fused kernel costs what the two products
cost while holding two fewer Z-sized matrices.

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

**Resolved: there was no loop-time regression to explain.** The question as
first written: NS time on 4096 appeared to rise **2590.9 -> 3135.4 s (+21%)**,
systematically across phases — `P X P` 1506.9 -> 1829.8 s, `Z·P`
150.9 -> 257.7 s, `X^2` 433.8 -> 534.0 s — with NUMA placement and a
fragmented machine as the candidate causes. Two findings closed it:

- **The 2590.9 endpoint is `diag.log`** — it is the only log that contains
  that number — which is the same non-reproducible intermediate build this
  section already disqualifies above ("comparisons against it should be
  disregarded"). The per-phase "systematic" deltas came from the same
  comparison. The +21% was never measured against the real baseline.
- **The controlled A/B says the library got faster.** Same driver source, the
  true baseline (`6b41b8d` + vbcsr `ebd5d4e`) against the round-2 HEAD
  (`3732334` + vbcsr `edbe797`, clean worktrees), back to back on a quiet
  machine, 2048 atoms at the 4096 regime (sptol 1e-4, tol 1e-7): the baseline
  binary reproduces its own historical number (1066.3 s vs 1076.9, within 1%
  — so machine state contributes nothing), and HEAD runs the loop in
  **1013.1 s against 1066.3** while taking 3 steps to the baseline's 4 and
  peaking at **31.34 GB against 46.60**. Phase totals: fused `P X P` 654.8 s
  over 3 steps vs two-call 577.5 over 4; fused `X^2+P` 158.2 vs 231.0;
  `Z·P` 170.5 vs 215.7; reform 27.0 (one fused kernel) vs 41.6 (two products
  + congruence machinery).

What remains true and unalarming: 4096 loop times are self-consistent across
sound builds (3226.8 s intermediate, 3195.3 s HEAD, quiet machine both), and
nothing sound ever put a comparable build at 2590 s. The lesson is the one
this section already carries: a number whose build state cannot be reproduced
is not a baseline, and it cost a round of ghost-hunting here.

**4096-atom validation at the round-2 HEAD** (rsatb `0904713`, vbcsr
`edbe797`; fused `rhar` reform, consuming updates, in-place completion,
task-queue cap, receive-bounded transpose all in). Same regime and machine as
the run above (`--supercell 2 2 1 --ns 6 --sptol 1e-4 --tol 1e-7`, 48
threads), idle machine, log in `validation_4096/`:

| | pre-round-2 (`final4096e`) | this validation |
|---|---|---|
| step 1 / 2 / final residual | 0.0191689 / 8.0307e-4 / 1.98072e-4 | **0.0191689** / 8.02084e-4 / 1.99403e-4 |
| steps to plateau | 3 | 3 |
| whole-run peak | 83.75 GB | **82.06 GB** |
| reform phase (seed -> X) peak | 42.60 GB, two products | **31.72 GB**, one fused kernel |
| step-2 resident | 65.09 GB | **43.67 GB** |
| steady after release | 33.47 GB | **27.76 GB** |
| NS loop / total | 3226.8 / 3823.3 s | 3195.3 / 3797.1 s |

Reading it: the trajectory is unchanged (identical step-1 residual; the
3rd-digit drift downstream is the fused reform's single-drop X, whose pattern
is slightly tighter — final density 0.549 -> 0.526). The whole-run peak moves
only 1.7 GB because it is set by step 1's `Z·P`/`P X P` sequence, whose
operands all necessarily coexist — the structural floor §5 describes; the
round's savings land in every phase that no longer sets the peak. The
applied-residual probe reports 1.4565e-3 against the propagated loop metric
1.994e-4 — the factor-7 honesty gap, now visible on every run. On the open
question above: this run had a quiet, unfragmented machine and still spent
3195 s in the loop, which was the first hint that the "2590.9 s" endpoint —
not the machine — was the problem; the resolution above settles it. GNU-time
cross-check: max RSS / 4 = 82.06 GB, matching the internal profiler exactly.

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
  The applied-residual probe described in section 1 closes the honesty gap, but
  note it is a different QUANTITY: a probe `||(Z^H S Z - I) v||_2` against the
  external check's untruncated norm, so the two numbers are related, not equal.
  The measurements in section 8 predate the probe, so their reported internal
  residual is still the propagated max-block norm.

---

## 10. Test coverage

`test_rarh`, `test_square_polynomial`, `test_spmm_inplace`, `test_spmm_hermitian`
— all pass at **1, 2 and 4 ranks**, each covering `bsr`, `vbcsr` and `csr`
matrix kinds. Python suite: 81 passed, 1 skipped. vbcsr adds
`test_vendor_batch_gemm` (§13) and `test_fused_upper` (§4.1).

**A blind spot found while adding §4.1.** `test_rarh` uses 6 uniform blocks. At 4
ranks that is at most 2 rows per rank — few enough that the accumulator's
first-touch order coincides with column order, so a column-ordering fault is
invisible. With the ordering deliberately removed it fails at 1 and 2 ranks and
*passes* at 4. `test_fused_upper` (in vbcsr, where the kernels are) uses 9 blocks
of five different sizes with the input adjacency listed descending, and fails at
1, 2, 4 and 5. Both are kept: the rsatb pair also checks the thresholded
accuracy claim against the two-call route, which the vbcsr one does not.

**A rank owning no blocks deadlocked the job**, found by running
`test_fused_upper` with more ranks than block rows. Nothing to do with the fused
kernels or with the exchange rounds — `BlockSpMat(DistGraph*)` is *collective*,
because `detect_matrix_kind` Allreduces the block-size extremes (a rank owning
nothing cannot decide CSR/BSR/VBCSR alone), and nothing said so.
`make_overwritten_result` called `detect_matrix_kind` and then, in the branch it
takes when the result graph is empty, constructed through that constructor —
so exactly the starved ranks entered two Allreduce nobody else did. It hung
`test_rarh` and `test_square_polynomial` too, at 10 ranks over their 6 blocks.
Fixed by constructing with the kind already in hand; `test_fused_upper` now runs
every check twice, once over a partition that starves half the ranks.

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
| Propagated residual is not the residual of the returned `Z` | Reform-and-re-measure before any termination that would report a propagated residual (later replaced by the §1 applied-residual probe, which reports without re-running the reform) |
| `rarh` stage-one pruning bound omitted the second fan-out | Gate scaled per `k` by `|B row k|`. The reviewer's 6x6 counterexample is now the `coherent` test case; it fails against the old gate (error 2.88 vs bound 1.0) and is exact against the new one |
| `rarh::assemble` eagerly zero-touched the full result while staging was live | Uses `make_overwritten_result`, as `square_polynomial` already did |
| Peak and steady RSS reduced with independent `MPI_MAX`, so they could come from different ranks | `MPI_MAXLOC` on the peak, then the steady value is broadcast from that rank |
| `mean_krylov_blocks` used a local numerator over a local denominator | Both globally reduced |

**Accepted, not implemented.** Each is a design change that needs measurement,
not a defect with an obvious fix:

- **Adaptive drift refresh** driven by accumulated dropped-block error rather
  than a fixed period of 5. The applied-residual probe (§1) removes the
  *reporting* hazard; the fixed period remains a heuristic.
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
- **`complete_hermitian_consuming`.** Since implemented as
  `complete_hermitian_inplace` (§4.2): the upper triangle is released as it is
  mirrored. Kept here so the audit trail reads in order.

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
