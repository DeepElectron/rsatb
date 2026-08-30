# The spectral diffusion-spectrum route

This note derives and specifies `corr_msd_spectral` (rsatb/calc/msd_spectral.hpp):
the energy-resolved mean-square displacement $\mathrm{MSD}(E,t)$ computed at
*resolution-plus-evolution* cost, linear in the evolution window through
segmented anchoring, with every output time obtained by scalar
post-processing. For the expensive-apply operator forms it is at least on
par with the stepwise route in every regime measured and up to an order of
magnitude faster; the stepwise route (msd.hpp) remains the tool for cheap
orthogonal tight-binding matvecs, where the spectral dot table dominates.

Throughout, $\tilde A$ is the scaled operator the Chebyshev engine iterates
(`cheby.apply_op`; $\tilde A = (S^{-1}H - c)/\lambda$ in the Solve form, with
$c$ the window centre and $\lambda$ the half-width), $T_p$ are Chebyshev
polynomials, $U(\tilde t) = e^{-i\tilde A \tilde t}$ is the propagator in the
scaled time $\tilde t = t\,\lambda/\hbar$, $D$ is the form-dressed velocity
(`apply_dual`: $D = S^{-1}V$ under Solve), $W$ is the form's metric ($W = S$
under Solve, the identity otherwise), and $|r\rangle$ is a stochastic sample
with metric-correct covariance $\mathbb{E}\,|r\rangle\langle r| = W^{-1}$
(MetricSampler). The self-adjointness $W\,T_p(\tilde A) = T_p(\tilde A)^\dagger W$
is used repeatedly; it is the same identity behind the doubled moments in
msd.hpp.

## 1. The cost structure being replaced

Every published linear-scaling implementation of the energy-resolved MSD — the
LSQT review [1] (Eqs. 67–70), GPUQT [2], and rsatb's own `corr_msd` — evaluates

$$
\rho(E)\,\mathrm{MSD}(E,t) \;=\; \big\langle \varphi(t) \big|\, \delta(E-\hat H) \,\big|\varphi(t)\big\rangle,
\qquad
|\varphi(t)\rangle = [\hat X, U(t)]\,|r\rangle ,
$$

by expanding $\delta(E-\hat H)$ in a KPM series *between two time-evolved
states* and re-expanding it at every output time: $n_t$ independent
$n_{\mathrm{kpm}}$-term recurrences, i.e. $n_t \cdot n_{\mathrm{kpm}}$ operator
applies ($n_t \cdot n_{\mathrm{kpm}}/2$ after the metric-weighted moment
doubling in msd.hpp). The propagation itself is the small term.

The concurrent shared-state framework of Fu, Li, Zhou and Yuan [3], which
rsatb implements for the AC correlator and the DOS (`corr_ac_concurrent`,
`corr_from_moments`), does not touch this term. Its gain is the Bessel tail of
$e^{-i\tilde A\,\Delta\tilde t}$ amortised over a block, and the moment
recurrence has no tail — all $n_{\mathrm{kpm}}$ terms are signal. The
obstruction is structural: $\delta(E-\hat H)$ sits between two states that
both move, and the moments are bilinear in the moving state, so every
block-shared expansion of $\varphi(t)$ costs a Chebyshev chain per stored
state where the AC correlator pays one dot per sample.

## 2. The reformulation: move the resolution onto the static state

Write the displacement as the time integral of the Heisenberg velocity,
$\hat X(t) - \hat X = \int_0^t \hat v(s)\,ds$. Stationarity of the velocity
autocorrelation and $[T_p(\tilde A), U] = 0$ give the exact identity

$$
\mathbb{E}\,\big\langle \varphi(t)\big| T_p(\tilde A) \big|\varphi(t)\big\rangle
= \int_0^t\!\!\int_0^t \mathrm{Tr}\big[\hat v(s)\, T_p(\tilde A)\, \hat v(s')\big]\, ds\, ds'
= 2\,\mathrm{Re} \int_0^t (t-u)\, C_p(u)\, du ,
$$

$$
C_p(u) \;=\; \mathrm{Tr}\big[\, T_p(\tilde A)\; v(u)\, v \,\big],
\qquad v(u) = U(u)^{-1}\, v\, U(u),
$$

which moves all time dependence into a single Heisenberg factor. Expanding
that factor's left evolution in Chebyshev,
$U(u)^{-1} = \sum_j c_j(u)^{*}\, T_j(\tilde A)$ — a scalar-coefficient
identity, valid for any $\tilde A$ — and folding the product onto the *same*
chain with $T_p T_j = \tfrac12 (T_{p+j} + T_{|p-j|})$ yields

$$
C_p(u) = \sum_j c_j(u)^{*}\; \tfrac{1}{2}\big[\, d_{p+j}(u) + d_{|p-j|}(u) \,\big],
\qquad
d_q(u) = \big\langle W\, T_q(\tilde A)\, r \,\big|\, D\, g(u) \big\rangle,
\quad g(u) = U(u)\, D\, r .
$$

Every object now belongs to one of two *one-time* computations:

- **one chain** $T_q(\tilde A)\,r$ for $q < n_{\mathrm{kpm}} + K$, where
  $K = T\lambda/\hbar$ plus the Bessel tail is the Chebyshev content of the
  full window $[0, T]$;
- **one evolution** of $g(u)$ across the window, block-concurrent in the
  sense of [3], costing $\sim K$ applies.

The dot table $d_q(u)$ couples them, and the number of *output times* no
longer appears in the operator-apply count at all:

$$
\text{stepwise:}\ \ \tfrac12\, n_t\, n_{\mathrm{kpm}}
\qquad\text{vs.}\qquad
\text{spectral:}\ \ (n_{\mathrm{kpm}} + K) + K \ \ \text{applies (+ GEMM)} .
$$

This is the information-theoretic floor shape: evolving over $[0,T]$ cannot
cost less than $\sim K$ applies (no fast-forwarding), and resolving the
$\delta$ to $n_{\mathrm{kpm}}$ moments cannot cost less than
$\sim n_{\mathrm{kpm}}$; the spectral route pays their *sum*, the stepwise
route a product. A symmetric split of the Heisenberg factor,
$v(u/2)\,v(-u/2)$, was checked and trades the halved chain exactly into a
second half-evolution — the arrangement above is optimal within its family.

**Segmented anchoring.** Expanding $U(u)^{-1}$ from the time origin spreads
it over $\sim u$ coefficients, which makes the dot table quadratic in the
window — the flaw that confined the original formulation to short windows.
The cure is to re-anchor the expansion every segment of scaled length
$\tilde L$: for $u = \tilde u_s + \delta$ inside segment $s$,
metric-unitarity moves $U^{-1}(\tilde u_s)$ onto the bra, where — because
$U$ commutes with polynomials of $\tilde A$ — the evolved chain family *is*
the chain of the evolved seed, $U(\tilde u_s)\, T_q(\tilde A)\, r =
T_q(\tilde A)\, r_s$ with $r_s = U(\tilde u_s)\, r$. Hence

$$
C_p(\tilde u_s + \delta) = \sum_j c_j(\delta)^{*}\; \tfrac12
\big[ d^{s}_{p+j}(\delta) + d^{s}_{|p-j|}(\delta) \big],
\qquad
d^{s}_q(\delta) = \big\langle W\, T_q(\tilde A)\, r_s \,\big|\, D\, g(\tilde u_s + \delta) \big\rangle,
$$

with $\delta \le \tilde L$: coefficients never spread beyond a segment.
The cost becomes two evolved trajectories ($g$ and the anchor $r_s$, together
$\sim 2$ applies per unit scaled time), a chain of $n_{\mathrm{kpm}} +
M(\tilde L)$ per segment (amortised to $\sim n_{\mathrm{kpm}}/\tilde L$
per unit time), and dots linear in the window; memory is bounded by the
segment, $O(\tilde L\, N_{\mathrm{local}})$, not the window. Every
rearrangement is an operator identity on the same quadratic form, so the
segmented estimator equals the single-segment one per sample to rounding
(unit-tested at $2\times 10^{-12}$ relative), and a window short enough for
one segment reproduces the original formulation exactly. `segment_scaled`
sets $\tilde L$ (auto: $\max(192, \min(768, n_{\mathrm{kpm}}))$).

**Metric–velocity cancellation.** In the Solve form with a non-composite
velocity the dressing of the second velocity cancels against the metric,

$$
\big\langle S\, T_q(\tilde A) r \,\big|\, S^{-1} V g \big\rangle
= \big\langle T_q(\tilde A) r \,\big|\, V g \big\rangle ,
$$

so the per-state velocity is a bare SpMV and the chain needs no $S$
products — the $\sim 1.3K$ per-point CG solves of the naive placement vanish.
Composite velocities keep their inner metric solves (only the outer dressing
cancels); the Sandwich form has no such pairing. The seed $w = D r$ is
physical and stays dressed in every form.

## 3. Engine organisation

The implementation makes every axis either one sequential recurrence or a
scalar contraction; no per-grid-point vector work exists at all.

**Evolution.** $g$ advances through Chebyshev blocks. Within a block seeded
at $g_b$, only the *velocity-applied* states $\mathrm{VX}_m = D_v\,
T_m(\tilde A)\, g_b$ are stored ($\approx 0.8$ columns per grid point across
the window); the raw states are never kept — the block-end seed accumulates
during the recurrence as $g_{b+1} = \sum_m c_m(\ell_b)\, T_m(\tilde A) g_b$.
Block length trades only Bessel-tail amortisation (default 256 scaled units).

**Chain and dot table.** The chain runs once, dotted against the stored
columns in tiled ZGEMMs to form the scalar table
$G_{qm} = \langle W T_q r | \mathrm{VX}_m\rangle$. The table is trapezoidal —
a block's states are read only up to the chain order its last grid point
reaches, $q \le n_{\mathrm{kpm}} - 1 + M(u)$ with $M(u)$ the expansion length
of $e^{-i\tilde A u}$ — so high-$q$ tiles restrict to their late-block column
suffix.

**Grid points.** $d_q(u) = \sum_m c_m(u - u_b)\, G_{qm}$ and the $j$-fold to
$C_p(u)$ are cache-contiguous scalar loops, OpenMP-parallel over grid points
with thread-local coefficient builds. Because the dot table is identical on
every rank after the tile reductions, the grid points are additionally dealt
round-robin across MPI ranks and the correlator samples reduced once at the
end — the scalar floor then scales with ranks like the vector work, instead
of becoming the Amdahl term of a large run.

**Memory.** Per rank, $\approx 1.3\, \min(\tilde L, \tilde T)\,
N_{\mathrm{local}} \times 16\,\mathrm{B}$ of *segment-local* stored
columns plus scalar tables — bounded by the segment length regardless of
the window.

## 4. The time integral: cardinal quadrature in closed form

$C_p(u)$ is band-limited: its frequencies are eigenvalue differences,
$|\omega| \le 2$ in scaled units (the bound behind ac.hpp's
`kAcOversample`), with content at the band edge — a trapezoid on the sampling
grid errs at the percent level. The grid samples at $h \le \pi/4$ (twice
Nyquist), and the integral is taken through the cardinal interpolant

$$
C(u) = \sum_{e} C[|e|]\; \kappa\!\big(u/h - e\big),
$$

with $\kappa$ a Kaiser-windowed sinc of half-width $W_\kappa$ samples and the
even mirror $C[-e] = C[e]$ (exact: $C_p(-u) = C_p(u)^{*}$). For a sample
whose kernel support lies inside $(0, t)$ the weight is *closed form*,

$$
\int_0^{t} (t-u)\,\kappa\!\big(u/h - c\big)\, du
= h^2\,\mu_0\,\big(t/h - c\big),
\qquad \mu_0 = \int \kappa(s)\, ds,
$$

because the kernel's first moment vanishes by evenness. Only the
$O(W_\kappa)$ samples within a kernel half-width of $u = 0$ and $u = t$ carry
corrections, and those depend on the on-grid offset alone — three
shift-invariant tables, integrated numerically once. With per-moment prefix
sums every output time costs $O(W_\kappa)$, so output grids of any density
(including log-spaced times for $d\,\mathrm{MSD}/dt$ extraction) are
genuinely free. The evolution runs one kernel half-width past the window for
the right edge; output times shorter than the two-edge overlap use directly
integrated dense weights.

## 5. Estimator equivalence and validation

The stepwise and spectral estimators share the physical trace as their
expectation but differ per sample: the identity of §2 uses trace cyclicity,
which a single quadratic form $\langle r|\cdot|r\rangle$ obeys only on
average. Summed over a complete metric basis
($\sum_i r_i^{\vphantom{\dagger}} r_i^{\dagger} = W^{-1}$, e.g. the columns
of a Cholesky factor of $S^{-1}$) both become the exact trace, which is what
the dense unit tests compare — against dense references built independently
from each route's own definition, and against each other:

- orthogonal form, complete basis vs. dense reference: $7\times 10^{-11}$;
- Solve form: $2.5\times 10^{-8}$ (set by the CG tolerance, matching the
  stepwise route's own accuracy);
- long-window regression ($\tilde T = 291$, 33 output steps):
  $5\times 10^{-10}$ — quadrature and truncations do not accumulate;
- stepwise vs. spectral on the complete-basis Solve fixture: 8 digits.

At few stochastic samples the two routes scatter against each other as
different unbiased estimators of one trace; the benchmark prints the
stepwise seed-to-seed noise floor next to the cross-route deviation to keep
that legible. Trace self-averaging makes one or few samples sufficient at
production system sizes.

## 5.1 Blocked samples (2026-08-27)

The route is blocked over stochastic samples on the APPLY axis: chunks
of `kSpectralSampleChunk = 16` columns advance through every recursion
(trajectory, chain, anchor) as one multivector, so each SpMM feeds the
whole chunk -- the width-1 loop measured 4x slower end to end on a
1664-orbital sandwich block of 64 purely from per-apply overhead. The
dot structure is SAMPLE-DIAGONAL ($G_s$ pairs a sample's chain with the
same sample's trajectory), so stores are sample-major slabs, the tile
GEMMs run one slice per sample (identical total dot work to the loop),
and one Allreduce serves all slabs per tile. Memory scales with the
CHUNK, never the sample count: segment store $\times$ chunk, dot table
chunk $\times q_{seg} \times$ seg\_cols $\times 16$ B; 16 sits on
the SpMM efficiency plateau and keeps diamond-sized systems in budget.
Any chunking reproduces the width-1 spectrum to summation rounding
(`BlockedSampleChunksMatchWidthOne`); `sample_chunk` overrides the cap.
The auto dispatch (`spectral_msd_cheaper`) is width-aware: counts are
derated by the saturating per-apply efficiency ratio
$\mathrm{eff}(w) = w/(w+3)$ between the routes' widths -- exactly 1 at
equal widths, and the machine-independence of the comparison survives
as a ratio of widths rather than a flop model. (Pre-blocking, the
count-only estimate misrouted a 64-sample production block to the
4x-slower path; measured 2026-08-27.)

Measured clean (2026-08-28, silicon sc444, 1664 orbitals, sandwich, 64
samples, n\_steps 80, dt 0.5 fs, n\_kpm 1000, machine idle): width-1
spectral $\approx 60$ min task; blocked (4 chunk passes of 16)
**18m02s** -- a 3.4x speedup from blocking alone. The stepwise route
with moment prediction (`n_kpm_compute` 320) runs the same task in
13m12s. The control run resolved the comparison (2026-08-28,
`si_stepwise_full`): stepwise at FULL moments costs 18m16s -- equal to
blocked spectral -- so the prediction trick was the only source of the
stepwise advantage, and it is UNSAFE on discrete spectra: with
`n_kpm_compute` 320 the silicon MSD grid is 29% negative (worst
$-1.2\times10^5$ against a $\sim10^4$ scale) and $D(E,t)$ deviates
$\sim24\%$ median from either full-resolution route, because the
Chebyshev moments of a sparse discrete spectrum are undamped cosines
that linear prediction cannot extrapolate. At full moments the
stepwise route is exactly positivity-preserving (zero negatives on the
whole grid, as the diagonal Jackson density guarantees) and the two
routes cross-validate: 5.1% median pointwise deviation, correlation
0.9991, residual = kernel difference. Use prediction only on dense
spectra (the 1024-atom diamond shows 2 band-edge fuzz points); on
small cells run full moments, where the route choice is a cost wash
and section 6's window scaling decides.

## 5.2 Trace-first fold (2026-08-29)

The silicon tie above had a precise cause: the apply-count model says the
blocked spectral route runs $\sim 1.9\times$ fewer operator applies than
full-moment stepwise on that task, so roughly half of its 18 minutes was
*not* applies — it was the per-sample scalar fold (the
sample–time–energy–Bessel loops) plus the per-sample dot tables and their
reductions. Everything downstream of the dot table $G_s$ is *linear* in it,
and every coefficient depends only on grid geometry, never on the sample —
so the tables are summed the moment they form (a $\beta$-chained GEMM
accumulation into one aggregate table per segment) and the fold runs once
per chunk instead of once per sample. Exactly the same estimator, to
summation order; table memory and Allreduce volume drop by the chunk width.

The fold itself is then restructured from loops into dense primitives:

- **Bessel tables hoisted.** The forward offset-$k$ coefficients depend
  only on $kh$ (shared by every block and chunk) and are tabled once per
  call; the backward tables ride along the existing per-segment $M(\delta)$
  probe, since the backward coefficients are the forward ones' conjugates.
- **$d_q(u)$ by GEMM.** $d_q(u) = \sum_m c_m(k)\, \bar G_{q,\mathrm{col}_0+m}$
  is one ZGEMM per block over its grid points.
- **The Bessel fold by FFT.** $C_p = \tfrac12 \sum_j b_j (d_{p+j} +
  d_{|p-j|})$ is a Toeplitz-plus-Hankel product; with the even extension
  $\tilde D_t = d_{|t|}$ and the symmetric kernel $\kappa_0 = b_0$,
  $\kappa_{\pm j} = b_j/2$, it collapses to the single linear convolution
  $C_p = (\kappa * \tilde D)_p$, evaluated with zero-padded FFTs (FFTW,
  one plan pair per segment, thread-safe new-array execution) —
  $O(L \log L)$ per grid point instead of $O(n_{\mathrm{kpm}} M(\delta))$,
  which on the silicon task was $\sim 3\times 10^{12}$ multiplies even
  after chunk aggregation.
- **MPI deal preserved.** Each rank folds a contiguous share of every
  block's points, computing its own D-GEMM slice, so the (now small) scalar
  work still scales with ranks; one correlator reduction at the end, as
  before.

This is the organizing principle of FastCheb-style methods — transform the
moment sequences rather than materialize their full coupling — applied to
the segmented fold. The estimator is untouched: all dense complete-basis
tests, the chunking-equality test, and the segment-equality test pass at
their prior tolerances.

Measured (2026-08-29, same silicon task, machine ~idle): **15m33s**
against the blocked baseline's 18m02s, and the output reproduces the
campaign's spectrum to $3.8\times10^{-8}$ relative over the full
$81\times1000$ grid. The remaining wall is the route's own inner
products — the operator applies and the chain-dot GEMMs, which
trace-first deliberately does not reduce (they *are* the estimator) —
plus the chunk-width-16 apply efficiency gap against the 64-wide
stepwise sweep. The fold no longer registers in the profile. The
apply-count dispatch model over-predicts the spectral margin here
($1.8\times$ predicted, $1.18\times$ measured) precisely because it
ignores the dot GEMMs; it still ranks the routes correctly, which is
all the dispatch asks of it.

## 6. When each route wins

With segmented anchoring the spectral cost is linear in the window and
independent of the output grid: per unit scaled time, roughly
$2 + n_{\mathrm{kpm}}/\tilde L$ applies plus $\sim 10\, n_{\mathrm{kpm}}$
dot-flops per matrix row-element. The stepwise route pays
$\tfrac12 n_{\mathrm{kpm}}$ applies *per output time* plus its propagation.
Measured on the Solve lattice fixture (1728 sites, CG per apply,
$n_{\mathrm{kpm}} = 300$, $\lambda = 13$ eV):

| window | output grid | stepwise | spectral (segmented) | ratio |
|---|---|---|---|---|
| 12 fs | 48 steps @ 0.25 fs | 34.9 s | 4.7 s | 7.5× |
| 12 fs | 12 steps @ 1.0 fs | 12.4 s | 5.5 s | 2.2× |
| 12 fs | 3 steps @ 4.0 fs | 5.5 s | 5.1 s | 1.1× |
| 192 fs | 48 steps @ 4.0 fs | 81.8 s | 74.3 s | 1.1× |
| 192 fs | 192 steps @ 1.0 fs | 212.4 s | 76.3 s | 2.8× |

Two structural facts show in the table: the spectral time depends only on
the window (74.3 s vs 76.3 s for the same 192 fs at 4× different sampling),
and it never falls below parity — the long-coarse case that lost under the
origin-anchored formulation (0.96×) now wins. For the expensive-apply forms
(Solve, Sandwich — rsatb's LCAO/DFTB targets) the segmented spectral route
is therefore the *default*: parity at worst, up to an order of magnitude
otherwise, with fine and log-spaced output grids free.

The one remaining regime distinction is *structural, not parametric*: for
cheap orthogonal tight-binding matvecs the $\sim 10\,n_{\mathrm{kpm}}$
dot-flops per unit time dominate the few-flop applies, and the stepwise
route (with doubling and the LP knob) stays ahead. That dispatch is decided
by the operator form at setup, not tuned per input. For full diffusion
tensors the chain anchor is axis-independent: $x/y/z$ share the anchor
trajectory and chains against three $g$-trajectories.

One corner of the expensive-apply regime does admit a stepwise win: *very
sparse output sampling of very long windows*. Per unit scaled time the
stepwise route pays $2$ applies of propagation plus $\tfrac12
n_{\mathrm{kpm}}$ per output row, while the spectral route pays $\approx 3 +
n_{\mathrm{kpm}}/\tilde L$; when the output step exceeds $\sim 2\tilde
L\,\hbar/\lambda$ (with $\tilde L$ memory-capped at 768), the moment runs
amortise away and the chain term does not — e.g. a 10 ps window sampled at
50 points loses $\approx 1.9\times$. Because both routes' dominant cost
there is applies of the *same* operator, the comparison is
machine-independent — an apply *count*, not a flop model — and
`spectral_msd_cheaper` evaluates it exactly (Bessel expansion lengths
included). Checked against the table above, it predicts $0.9\times$ and
$2.7\times$ where $1.10\times$ and $2.78\times$ were measured.

**Measured scaling (2026-08-29, idle 96-core node; `benchmark_msd_moments
run` mode).** On the Solve-form lattice (CG per apply) the spectral kernel
wins at every size and widens with $N$ ($1.6\times$ at 512 sites,
$3.1\times$ at 8000); on the orthogonal lattice stepwise is $3$–$6\times$
faster — the structural boundary the dispatch encodes. Both kernels are
measured linear in the window (exponents 1.01 stepwise / 0.97 spectral,
12–192 steps) with the spectral margin holding at 2.3–2.6$\times$; both are
sub-linear in $n_{\mathrm{kpm}}$ ($8\times$ finer resolution costs
$2.2\times$ / $3.2\times$). Spectral memory is segment-local as designed: a
$16\times$ longer window costs $1.19\times$ the peak RSS (165→196 MB at
N=4096). Sixty-four samples cost the spectral kernel $7.9\times$
(trace-first: only applies scale) against $10.2\times$ stepwise. MPI strong
scaling at N=32768 (4 threads/rank): spectral $5.6\times$ at 8 ranks (69%
efficiency, per-rank memory 505→126 MB) vs stepwise $3.5\times$ (43%).
OpenMP on real 13-orbital LCAO blocks (silicon, 6→24 threads): stepwise 94%
incremental efficiency, spectral 69%; the single-orbital synthetic lattice
saturates much earlier (pointer-bound SpMV, worst case), and low-sample
spectral runs are additionally bounded by their chunk width — a 2-sample
run applies width-2 operators while the stepwise readout tiles itself to
width 16 regardless.

The task layer exposes exactly this dispatch: `compute_msd_spectrum` takes
`method = "auto" | "stepwise" | "spectral"`, where `"auto"` (the default)
selects the spectral route whenever the operator form is expensive-apply
(Solve or Sandwich), the commutator velocity exists, *and* the apply-count
estimate favours it — which is everywhere outside the sparse-sampling
corner. The orthogonal side stays a structural rule: there the applies are
cheap and the decision would compare bandwidth-bound SpMVs against
compute-bound GEMM dots, which no machine-independent count settles.
The spectral estimator integrates velocity correlations, so an explicit
`method = "spectral"` requires a periodic axis with velocity data and
refuses anything else by name. The two methods are different unbiased
estimators of one trace (§5), so a parameter sweep that must not switch
estimators mid-sweep should pin the method explicitly.

## 7. Moment extrapolation (the `n_kpm_compute` knob)

A Chebyshev moment sequence $\mu_p = \sum_\alpha w_\alpha \cos(p\,
\theta_\alpha)$ is a mixture of sampled sinusoids — exactly the signal class
autoregressive models represent. Following the DMRG literature [4, 5, 6],
both routes accept `n_kpm_compute`: the recurrence runs to $n_c$ moments and
linear prediction extends the sequence to $n_{\mathrm{kpm}}$ at zero operator
cost, before the shared Jackson reconstruction. The fit is covariance least
squares (Tikhonov-regularised normal equations) iterated through the
companion-matrix eigendecomposition with poles outside the unit circle
renormalised onto it — deliberately *not* Burg's lattice method, whose
greedy stagewise fit biases undamped-sinusoid poles to $\approx 0.999$ and
decays a 5× extrapolation to $O(1)$ error where least squares holds
$10^{-8}$ over 50×.

Measured: quarter-order runs recover full-order runs to $3\times 10^{-11}$
on dense fixtures; on a semicircle-plus-sharp-Lorentzian test measure, 300
moments extrapolated to 1200 reconstruct with 0.1% error where plain
truncation errs 12.8%, robust to $10^{-3}$ relative moment noise. The
practical criterion for $n_c$: moments are worth computing while their
band-wise signal-to-noise — readable from the seed-to-seed difference of two
sample sets — stays above $\sim 1$; beyond that the computed tail is noise
and the model tail is at least as physical. On the small benchmark fixture
that crossover sits near $p \approx 60$ of 300; at production
$N \times n_{\mathrm{samples}}$ it moves out by orders of magnitude. With
the knob the fine-grid Solve benchmark reaches 9.2× over stepwise.

## 8. Numerical safeguards fixed along the way

- **Batched CG/PCG column guards** (solver.hpp): a column with an all-zero
  right-hand side — how the stepwise kernel's commutator state $\phi(0)$
  arrives — produced $\alpha = 0/0$ and silently poisoned the whole block
  with NaN (NaN survives neither `max()` nor the density gate downstream).
  Columns below tolerance now freeze.
- **Scale-relative breakdown tests**: on an SPD operator
  $p^{\dagger}\!A p \ge \lambda_{\min} \|r\|^2$, so breakdown is judged
  against the residual; the previous absolute $10^{-16}$ threshold fired
  spuriously right at convergence, where $\|r\|^2 \sim \mathrm{tol}^2$ makes
  $p^{\dagger}\!A p \sim 10^{-16}$ legitimate. The report states iteration
  and magnitudes.

## 9. Directions surveyed and set aside

- *Sampling axis* (Hutch++/XTrace [7, 8]; Krylov-aware trace estimation [9],
  whose deflation basis could reuse the chain states at no extra applies):
  set aside on the owner's assessment — MSD is a trace observable and
  self-averaging makes one or few samples sufficient at production sizes.
- *Precision axis* (FP32 chains, as KITE and GPUQT run): declined for this
  codebase.
- *Few-energy-window regime*: since $\mathrm{Tr}[\delta\, C^{\dagger} C] =
  \mathbb{E}\,\|C\,\delta^{1/2} r\|^2$, applying the half-filter once and
  evolving the filtered state gives a per-sample positive estimator at
  $\sim n_{\mathrm{kpm}} + 3K$ per window — the right tool when the question
  is $D(E, t)$ near $E_F$ only (same family as polynomial spectrum slicing,
  EVSL [10]).
- *GEMM subsampling* on the $u$-axis with importance weights: only relevant
  in the cheap-matvec regimes the stepwise route serves anyway.

Natural next steps: task-layer wiring (a `method` knob for the msd task),
$\mathrm{VAC}(E,t)$ and $D(E,t)$ as derivative-kernel outputs from the same
table (Green–Kubo integration then cross-checks the fftcheb dc_spectrum),
and multivector batching of the sample loop.

## References

1. Z. Fan, J. H. Garcia, A. W. Cummings, J. E. Barrios-Vargas, M. Panhans,
   A. Harju, F. Ortmann, S. Roche, *Linear scaling quantum transport
   methodologies*, Phys. Rep. **903**, 1 (2021) — Eqs. 67–70 define the
   stepwise state of practice.
2. Z. Fan, V. Vierimaa, A. Harju, *GPUQT*, Comput. Phys. Commun. **230**,
   113 (2018).
3. Z. Fu, Y. Li, W. Zhou, S. Yuan, *Large-scale stochastic propagation
   method beyond the sequential approach*, arXiv:2510.17432 — the concurrent
   framework whose moment-based variant this route extends from the
   one-operator DOS correlator to the two-operator diffusion correlator.
4. M. Ganahl, P. Thunström, F. Verstraete, K. Held, H. G. Evertz, Phys.
   Rev. B **90**, 045144 (2014) — linear prediction on Chebyshev moments.
5. F. A. Wolf, J. A. Justiniano, I. P. McCulloch, U. Schollwöck, Phys. Rev.
   B **91**, 115144 (2015).
6. S. R. White, I. Affleck, Phys. Rev. B **77**, 134437 (2008).
7. R. A. Meyer, C. Musco, C. Musco, D. P. Woodruff, *Hutch++*, SOSA (2021).
8. E. N. Epperly, J. A. Tropp, R. J. Webber, *XTrace*, SIAM J. Matrix Anal.
   Appl. **45** (2024).
9. T. Chen, E. Hallman, *Krylov-aware stochastic trace estimation*, SIAM J.
   Matrix Anal. Appl. **44**, 1218 (2023).
10. R. Li, Y. Xi, L. Erlandson, Y. Saad, *The EVSL package*, SIAM J. Sci.
    Comput. **41** (2019).
11. A. Weiße, G. Wellein, A. Alvermann, H. Fehske, *The kernel polynomial
    method*, Rev. Mod. Phys. **78**, 275 (2006).
12. S. M. João et al., *KITE*, R. Soc. Open Sci. **7**, 191809 (2020);
    A. Ferreira, E. R. Mucciolo, Phys. Rev. Lett. **115**, 106601 (2015) —
    the energy-domain siblings of the dot table.

## One formalism (2026-08-29)

MSD is computed from exactly one estimator family: the commutator velocity
correlator, $\phi(t) = [X, U(t)]\,\psi$ built from $V$ alone, valid for
every basis, operator form, and boundary condition (the bond moment's
$R + r_j - r_i$ weight degenerates to the open-boundary $[H, X]$ at a
single $R = 0$ image). The stepwise position-representation routes the code
used to carry — explicit position matrix, diagonal position, the
Gamma-affine split — computed the same estimator from redundant data and
were removed. What remains is two *evaluation kernels* of the one
formalism: the time-batched stepwise kernel (`msd.hpp`, the positivity
oracle — every full-moment row is a diagonal Jackson density) and the
trace-first spectral kernel (`msd_spectral.hpp`), behind the `method` knob.
No third estimator.

## Implementation map

- `rsatb/calc/msd.hpp` — `corr_msd(...)`: THE stepwise kernel (commutator
  pair recursion, grouped time-tiled moment readout); knobs
  `n_kpm_compute`, `time_tile`.
- `rsatb/calc/msd_spectral.hpp` — `corr_msd_spectral(...)`: blocks, chain,
  trace-first aggregate tables, GEMM d-rows, FFT Bessel fold, cardinal
  quadrature; knobs `time_oversample`, `block_substeps`, `n_kpm_compute`,
  `segment_scaled`, `sample_chunk`. Output contract identical to
  `corr_msd` ($[n_t{+}1,\, n_{\mathrm{kpm}}]$ density-normalised rows on
  the ascending Chebyshev grid).
- `rsatb/calc/moment_extrapolation.hpp` — `extrapolate_moments_lp`.
- `rsatb/calc/observables.hpp` — `ComputeMsdSpectrum(..., method,
  n_kpm_compute)`: requires the velocity, dispatches
  `"auto" | "stepwise" | "spectral"` as above, surfaced through
  `rsatb.hpp`, the pybind layer, and `driver.py::compute_msd_spectrum` /
  `compute_diffusion_spectrum`.
- Validation: `test_msd.cpp` (dense complete-basis fixtures for all three
  operator forms — Orthogonal, Solve, and Sandwich, the last against a
  dense reference written in its own $Z^H H Z$ variables — plus
  multi-segment long-window regression, segment/single-segment equality,
  the apply-count dispatch model pinned to the measured sweep, and LP
  recovery); `test_operator_api.cpp` and `test/test_binding.py` (method-knob
  dispatch, refusals, and the spectral route end-to-end on the LCAO
  fixture); benchmark:
  `benchmark_msd_moments.cpp` (route timings, dt sweep, noise-floor and
  moment-SNR diagnostics).
