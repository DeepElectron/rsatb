# Spectral bounds: the 11x that wasn't in the arithmetic

Status: diagnosed, fixed (rsatb `e0f10a1`), verified at 1/2/4 ranks. Written
for the record, because the failure mode — a solver whose *cost and answer*
both depend on the wall clock — is the kind that gets rediscovered expensively.

## 1. The incident

Two production runs of the same job — 178,624-atom 4x4 moire, LCAO, 6 ranks,
one k-point — with bit-identical operators (every Newton-Schulz residual
matches to the last printed digit), the same partition, and the same PRIMME
seed, put wildly different prices on `setup/spectral_bounds`:

| run | spectral_bounds | share of run | emax found |
|---|---|---|---|
| first | 142.3 s | 3.5% | 232.495902 eV |
| second | 1633.3 s | 31.6% | 232.501211 eV |

The phase computes two extreme eigenpairs of the pencil `(H, S)` at `eps =
1e-3`. "Two eigenvalues, loose tolerance" reads as cheap; here it silently
became the second-largest phase of a 5154 s run, and the profile table was the
only witness — the phase printed nothing.

## 2. The diagnosis, by elimination

Everything reproducible was checked and found identical between the runs:

- **Operators**: the Newton-Schulz iteration that consumes them downstream
  reproduces every residual digit-for-digit. The inputs are the same bits.
- **Partition**: same structure file, same rank count, same RIB + ParMETIS
  path.
- **PRIMME's seed**: `iseed[0] = procID % 4096`
  (`extern/primme/src/eigs/primme_c.c:341`) — a function of rank alone, and
  both runs are 6 ranks. Same seed, same rank-local rows: the same starting
  basis.

What remained was the clock. `PrimmeEigenSolver::solve` never called
`primme_set_method`, and PRIMME's documented default is `PRIMME_DYNAMIC`
(`primme_interface.c:546`): it starts with GD+k, measures its own wall-clock
per matvec / per inner step as it runs (`main_iter.c`, `primme_CostModel`,
driven by `primme_wTimer()`), and switches to JDQMR when the measured ratios
favour it. **The method choice — and with it, the entire iteration path — is a
function of machine noise.** Two bit-identical runs on a shared cluster take
different branches.

The switch multiplies the worst possible term here. On the generalized path
the "preconditioner" PRIMME applies is itself an iterative solve, CG on
`S y = x` to `1e-4` (`calc/eigen.hpp`, `inverse_s_wrapper`). GD+k applies it
once per outer iteration; JDQMR applies it once per **inner** QMR step. A
timing flip toward JDQMR therefore doesn't cost a constant — it multiplies the
count of the single most expensive operation in the phase. Five-ish serialised
inner solves per outer step is the 142 → 1633 s.

The same branch explains the answer moving. For a Hermitian problem Ritz
values approach the largest eigenvalue **from below**, and `eps = 1e-3` stops
at a residual of ~0.23 eV — anywhere inside the top cluster qualifies. The
slow path took more steps and stopped *closer to the true top*: 232.5012 over
232.4959. Neither value is wrong; both are inside the tolerance band, and
`bound_pad` covers the spread. One nondeterministic branch, two symptoms.

This also retires a false claim previously in the code: *"reproducible to 12
digits at a fixed rank count."* That was measured on diamond sc444 — a solve
fast enough that the timing model never fired. The claim did not survive its
first noisy cluster.

## 3. Why the phase is expensive at all

"We only need the maximum and minimum to converge" is true of `H`. This is the
pencil `(H, S)`, and its top edge is not physics:

- The moire run finds `emax ≈ 232.5 eV` against carbon bands that end near
  35 eV. Pencil eigenvalues scale like `h / s_min`, and a 13-orbital LCAO
  basis at 3.7 Å cutoff on a 2.5° moire has near-linearly-dependent functions:
  `s_min` is tiny, and the top of the pencil spectrum is a **dense cluster of
  basis-conditioning artifacts**.
- Davidson-type convergence to an extreme pair is governed by the *relative
  gap*. A cluster of artifacts is flat; flat tops converge slowly. That is
  inherent, not a solver defect.
- The cost compounds through the same number twice: each preconditioner
  application is CG on the very `S` whose conditioning made the top flat.

So the phase's baseline price is real and stays. What could not stand was the
11x scatter *around* that price, and the silence.

## 4. The fix (rsatb `e0f10a1`)

Three changes in `calc/eigen.hpp` / `setup/spectral_bounds.hpp` /
`setup/operator_set.hpp`:

1. **Method pinned** to `PRIMME_DEFAULT_MIN_MATVECS` (GD+k with Olsen). The
   right side of the trade wherever the matvec is a distributed SpMV and the
   preconditioner is an inner solve — and it removes the timing branch, which
   was the entire run-to-run cost variance. Pinned **after** the operator
   wiring: `primme_set_method` reads `applyPreconditioner` to decide
   `correctionParams.precondition`, and calling it early silently disabled
   preconditioning — caught as `preconds=0` on the stats line of a probe, the
   kind of regression the stats line exists to catch.
2. **Start pinned**: an explicit initial basis (`primme.initSize = 1`) drawn
   by the counter-based sampler keyed on **atom input indices**
   (`BoundsKPointView::stable_block_ids`), so the same system starts the same
   solve at any rank count. PRIMME's own draw is rank-seeded over rank-local
   rows, which baked the decomposition into the answer: diamond sc444 gave
   emax = 271.1921 / 271.2007 / 271.2051 at 1/2/4 ranks. This was the
   long-standing "spectral window not decomposition-invariant" gap
   (design_isqrtm_memory.md §5), now closed.
3. **Price on the record**: each solve prints matvecs, preconditioner
   applications, and seconds under verbose. A phase that can eat a third of a
   run must say so next to the window it bought.

## 5. What is now guaranteed, and what is not

Verified on a 384-block, bs=3 pencil probe through the real
`ExtremalEigenvalues` path:

- **Fixed rank count: bit-reproducible.** Two 4-rank runs agree to all twelve
  printed digits, matvec counts included (93/136 matvecs, 16/27 preconds,
  both runs). No timing branch, no entropy — a fixed computation.
- **Across rank counts: agreement to ~1e-4**, not bitwise. Floating-point
  summation order differs per partition (local partial sums, Allreduce), a
  Krylov iteration amplifies it, and `eps = 1e-3` is loose enough for a
  one-iteration difference in the stopping decision to show. Irreducible
  without tightening eps, which costs matvecs in the most expensive phase of
  setup. The spread is inside the tolerance the caller asked for.

Tests: full rebuild clean; setup_stages / solver / isqrtm / rarh at 1/2/4
ranks; metric_sampler fast subtests 67.6 s against 68.8 s baseline (the pin
does not regress the composed-operator path). `test_operator_api` fails at
np≥2 identically at baseline — a pre-existing HDF5 file-lock issue on this
filesystem, unrelated.

## 6. The open consequence: the artifact top taxes every KPM task

Since `emax ≈ 232 eV` is basis conditioning rather than physics, the Chebyshev
window is ~7x wider than the spectrum that matters. The moire DOS run pays it
directly: `dE 0.05 eV over span 254.65 eV -> 18,839 moments`, where a ~40 eV
physical span would need ~3,000. Every KPM task — DOS, AC, LDOS — pays this
factor on every moment, forever.

Candidate closures, in rising order of invasiveness: an explicit user `emax`
for KPM tasks (already supported via `energy_bounds`); filtering the overlap's
near-null space out of the pencil; a basis-conditioning diagnostic at setup
that reports `s_min` and the implied span tax. Deliberately not done here — it
changes answers (a truncated window clips whatever weight sits in the
artifact states, which must be shown to be negligible), so it deserves its own
design and measurement.
