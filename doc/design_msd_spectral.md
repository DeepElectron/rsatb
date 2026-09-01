# Energy-resolved diffusion: one formalism, two kernels

**What this computes.** The energy-resolved mean-square displacement
$\mathrm{MSD}(E,t)$ — how far a carrier of energy $E$ spreads in time $t$ —
and with it the diffusion coefficient $D(E,t) = \tfrac12\,\partial_t
\mathrm{MSD}(E,t)$, for systems too large to diagonalise. It is the
real-space transport observable: from its short-time slope one reads the
band velocity, from its long-time plateau the mean free path and the
Kubo conductivity.

**Why it is hard.** The textbook estimator resolves energy *inside* the time
loop, spending a full kernel-polynomial expansion at every output time. Cost
is the product $n_t \times n_{\mathrm{kpm}}$ of output times and energy
resolution, and it is the reason a converged $\mathrm{MSD}(E,t)$ on a
few-thousand-orbital cell costs hours where a density of states costs
minutes.

**What is done here.** Two results. First, a *single* physical formalism —
the commutator–velocity correlator — replaces the three
position-representation routes the code used to carry; they were the same
estimator built from redundant data (§2). Second, that one formalism is
evaluated by two interchangeable kernels: a time-domain **stepwise** kernel
that is exactly positivity-preserving and serves as the reference oracle
(§3), and a **spectral** kernel that decouples resolution from evolution and
so pays their *sum* rather than their product (§4–§5). On the
expensive-apply operator forms that rsatb targets, the spectral kernel is
parity-at-worst and up to an order of magnitude faster, with cost linear in
the evolution window, memory bounded by a segment rather than the window,
and output grids of any density free. A dispatch rule picks the kernel from
the operator form and an apply count (§7); both kernels agree with dense
references to the solver tolerance (§6).

Notation, fixed throughout: $\tilde A$ is the scaled operator the Chebyshev
engine iterates (`cheby.apply_op`; $\tilde A = (S^{-1}H - c)/\lambda$ in the
Solve form, $c$ the window centre, $\lambda$ the half-width); $T_p$ are
Chebyshev polynomials; $U(\tilde t) = e^{-i\tilde A\tilde t}$ propagates in
scaled time $\tilde t = t\lambda/\hbar$; $D$ is the form-dressed velocity
(`apply_dual`; $D = S^{-1}V$ under Solve); $W$ is the form's metric ($W = S$
under Solve, the identity otherwise); $|r\rangle$ is a stochastic sample with
metric-correct covariance $\mathbb{E}\,|r\rangle\langle r| = W^{-1}$
(MetricSampler). The self-adjointness $W\,T_p(\tilde A) = T_p(\tilde
A)^{\dagger} W$ is used throughout; it is also what makes the doubled moment
recurrence valid.

---

## 1. The cost structure being replaced

Every published linear-scaling implementation of the energy-resolved MSD —
the LSQT review [1] (Eqs. 67–70), GPUQT [2], and rsatb's own original
`corr_msd` — evaluates

$$
\rho(E)\,\mathrm{MSD}(E,t) \;=\; \big\langle \varphi(t) \big|\, \delta(E-\hat H) \,\big|\varphi(t)\big\rangle,
\qquad
|\varphi(t)\rangle = [\hat X, U(t)]\,|r\rangle ,
$$

by expanding $\delta(E-\hat H)$ in a kernel-polynomial series *between two
time-evolved states*, and re-expanding it at every output time: $n_t$
independent $n_{\mathrm{kpm}}$-term recurrences, i.e. $n_t\,
n_{\mathrm{kpm}}$ operator applies, halved to $n_t\, n_{\mathrm{kpm}}/2$ by
the metric-weighted doubling below. The propagation itself is the small term.

The concurrent shared-state framework of Fu, Li, Zhou and Yuan [3] — which
rsatb implements for the AC correlator and the DOS — does not reduce this
term, and the obstruction is structural rather than incidental. Its gain is
the Bessel tail of $e^{-i\tilde A\Delta\tilde t}$ amortised over a block,
and the moment recurrence has no tail: all $n_{\mathrm{kpm}}$ terms are
signal. Worse, $\delta(E-\hat H)$ sits between two states that *both* move
and the moments are bilinear in the moving state, so any block-shared
expansion of $\varphi(t)$ costs a Chebyshev chain per stored state where the
AC correlator pays a single dot per sample. Whenever the per-step scaled
time exceeds unity — every physical MSD run — stepping in time *is* the
compressed representation.

## 2. One formalism

MSD is computed from exactly one estimator family: the commutator–velocity
correlator, with the state

$$
|\varphi(t)\rangle = [\hat X,\, U(t)]\,|r\rangle
$$

built from the velocity $V$ alone, never from a position operator. The
pair $(\varphi, \psi)$ propagates under the block-upper-triangular
generator with diagonal blocks $\tilde A$ and coupling $\tilde C =
[\hat X, \tilde A] = (i/\lambda)\,D$ — exactly what `apply_dual` computes
for the operator set's velocity, bare or composite — starting from
$\varphi(0) = 0$.

This is valid for every basis, every operator form ($H$, $S^{-1}H$,
$Z^{\dagger}HZ$), and *either boundary condition* — the last point is the one
that consolidated the code. The periodic bond-moment velocity carries the
weight $R + r_j - r_i$; at a single $R = 0$ image that weight *is* the
open-boundary $[H, \hat X]$. The three position-representation routes the
code used to carry (explicit position matrix, diagonal position, and the
Gamma-affine translation-covariant split) therefore computed the same
estimator from redundant data, as the dense fixtures confirm per sample.
They were removed; what remains is one formalism and two evaluation kernels
behind a `method` knob. No third estimator.

The single capability a model must supply is the velocity: orthogonal models
always have it (the bond moment of $H$), non-orthogonal models need the
position/dipole integrals — the same requirement the explicit-position route
carried, so no capability was lost.

## 3. The stepwise kernel — reference and oracle

The time-domain kernel (`msd.hpp`) propagates the commutator pair and reads
out the moments $\mu_p = \langle\varphi(t)|W\,T_p(\tilde A)|\varphi(t)\rangle$
at each output time. Two structural optimisations make it the *fast*
reference rather than merely the honest one.

**Doubled moments.** $\tilde A$ is self-adjoint in $W$, so the product
identity $2T_mT_n = T_{m+n} + T_{|m-n|}$ folds onto a single chain
$t_n = T_n(\tilde A)\varphi$:

$$
\mu_{2n} = 2\langle t_n|W|t_n\rangle - \mu_0, \qquad
\mu_{2n-1} = 2\langle t_n|W|t_{n-1}\rangle - \mu_1 ,
$$

two moments per operator apply instead of one. The Solve form pays one extra
$S$ SpMV per pair — noise beside the CG solve inside its apply.

**Time-batched readout.** A separate recurrence per output time means
width-1 recurrences for few-sample runs, which are overhead- rather than
flop-bound (measured $\approx 4\times$), and one collective reduction per
doubled iteration per step. Instead, commutator states for `time_tile`
output times are stashed and read out through *one* grouped recurrence at
combined $(\text{time} \times \text{sample})$ width: per-column dots come
from a single batched reduction serving the whole tile, then average within
each sample group. Per-column arithmetic is untouched, so tiling changes no
value (unit-tested at $10^{-10}$ across tile depths, partial trailing tile
included, on both orthogonal and Solve forms). The auto tile fills the
combined width to 16 columns — the SpMM efficiency plateau.

**Why it stays.** At full moments every output row is a diagonal Jackson
density of a physical state, hence non-negative bin by bin: the stepwise
kernel is *exactly* positivity-preserving, which makes it the oracle against
which the spectral kernel and any approximation are judged. And where
operator applies are cheap — orthogonal tight-binding matvecs — its
few-flop applies beat the spectral kernel's dot floor outright (§7).

## 4. The spectral kernel — resolution plus evolution

Write the displacement as the time integral of the Heisenberg velocity,
$\hat X(t) - \hat X = \int_0^t \hat v(s)\,ds$. Stationarity of the velocity
autocorrelation and $[T_p(\tilde A), U] = 0$ give the exact identity

$$
\mathbb{E}\,\big\langle \varphi(t)\big| T_p(\tilde A) \big|\varphi(t)\big\rangle
= 2\,\mathrm{Re} \int_0^t (t-u)\, C_p(u)\, du ,
\qquad
C_p(u) = \mathrm{Tr}\big[\, T_p(\tilde A)\; v(u)\, v \,\big],
$$

with $v(u) = U(u)^{-1} v\, U(u)$ — all time dependence now sits in a single
Heisenberg factor. Expanding that factor's left evolution in Chebyshev,
$U(u)^{-1} = \sum_j c_j(u)^{*} T_j(\tilde A)$ (a scalar-coefficient identity,
valid for any $\tilde A$), and folding the product onto the *same* chain
with $T_pT_j = \tfrac12(T_{p+j} + T_{|p-j|})$:

$$
C_p(u) = \sum_j c_j(u)^{*}\, \tfrac12\big[\, d_{p+j}(u) + d_{|p-j|}(u) \,\big],
\qquad
d_q(u) = \big\langle W\, T_q(\tilde A)\, r \,\big|\, D\, g(u) \big\rangle,
\quad g(u) = U(u)\, D\, r .
$$

Every object now belongs to one of two *one-time* computations: **one chain**
$T_q(\tilde A) r$ resolving the energy axis, and **one evolution** of $g(u)$
across the window, block-concurrent in the sense of [3]. The number of
output times has left the operator-apply count entirely:

$$
\text{stepwise: } \tfrac12 n_t\, n_{\mathrm{kpm}}
\qquad\text{versus}\qquad
\text{spectral: } (n_{\mathrm{kpm}} + K) + K , \qquad K = T\lambda/\hbar .
$$

This is the floor shape one expects on information grounds: evolving over
$[0,T]$ cannot cost less than $\sim K$ applies (no fast-forwarding), and
resolving $\delta(E - \hat H)$ to $n_{\mathrm{kpm}}$ moments cannot cost less
than $\sim n_{\mathrm{kpm}}$. The spectral kernel pays their sum where the
stepwise kernel pays their product. (A symmetric split $v(u/2)v(-u/2)$ was
checked: it trades the halved chain exactly into a second half-evolution, so
the arrangement above is optimal within its family.)

**Segmented anchoring — what makes it hold at long times.** Expanding
$U(u)^{-1}$ from the time origin spreads it over $\sim u$ coefficients,
making the dot table quadratic in the window; this confined the original
formulation to short windows. The cure is to re-anchor every segment of
scaled length $\tilde L$. For $u = \tilde u_s + \delta$ inside segment $s$,
metric-unitarity moves $U^{-1}(\tilde u_s)$ onto the bra, where — because
$U$ commutes with polynomials of $\tilde A$ — the evolved chain family *is*
the chain of the evolved seed:

$$
U(\tilde u_s)\, T_q(\tilde A)\, r = T_q(\tilde A)\, r_s,
\qquad r_s = U(\tilde u_s)\, r ,
$$

so that $d^{s}_q(\delta) = \langle W T_q(\tilde A) r_s | D\, g(\tilde u_s +
\delta)\rangle$ with $\delta \le \tilde L$: coefficients never spread beyond
a segment. Cost becomes two evolved trajectories ($g$ and the anchor $r_s$,
$\sim 2$ applies per unit scaled time), a chain of $n_{\mathrm{kpm}} +
M(\tilde L)$ per segment (amortised to $\sim n_{\mathrm{kpm}}/\tilde L$ per
unit time), and dots linear in the window; memory is bounded by the segment,
$O(\tilde L\,N_{\mathrm{local}})$, not by the window. Every rearrangement is
an operator identity on the same quadratic form, so the segmented estimator
equals the single-segment one per sample to rounding (unit-tested at
$2\times10^{-12}$), and a window short enough for one segment reproduces the
unsegmented formulation exactly. `segment_scaled` sets $\tilde L$
(auto: $\max(192, \min(768, n_{\mathrm{kpm}}))$).

**Metric–velocity cancellation.** In the Solve form with a non-composite
velocity, the dressing of the second velocity cancels against the metric,

$$
\big\langle S\, T_q(\tilde A) r \,\big|\, S^{-1} V g \big\rangle
= \big\langle T_q(\tilde A) r \,\big|\, V g \big\rangle ,
$$

so the per-state velocity is a bare SpMV and the chain needs no $S$
products — the $\sim 1.3K$ per-point CG solves of the naive placement
vanish. Composite velocities keep their inner metric solves (only the outer
dressing cancels); the Sandwich form has no such pairing. The seed
$w = Dr$ is physical and stays dressed in every form.

## 5. Evaluating it: every axis a recurrence or a contraction

The implementation is organised so that no per-grid-point vector work exists
at all.

**Evolution.** $g$ advances through Chebyshev blocks. Within a block seeded
at $g_b$, only the *velocity-applied* states $\mathrm{VX}_m = D_v T_m(\tilde
A) g_b$ are stored ($\approx 0.8$ columns per grid point); raw states are
never kept, since the block-end seed accumulates during the recurrence as
$g_{b+1} = \sum_m c_m(\ell_b) T_m(\tilde A) g_b$. Block length trades only
Bessel-tail amortisation (default 256 scaled units).

**Samples on the apply axis.** Chunks of `sample_chunk` (default 16) columns
advance through every recursion — trajectory, chain, anchor — as one
multivector, so each SpMM feeds the whole chunk; the width-1 loop measured
$4\times$ slower end to end on a 64-sample production block purely from
per-apply overhead. Memory scales with the chunk, never the sample count.

**Trace-first dot table.** Everything downstream of the dot table is *linear*
in it, and every coefficient depends on grid geometry alone, never on the
sample. The per-sample GEMM slices are therefore accumulated ($\beta$-chained)
into **one aggregate table** per segment as they form. The dot flops are
unchanged — they *are* the estimator's inner products — but the table, its
reduction volume, and the entire scalar fold below become independent of the
chunk width. The table is trapezoidal: a block's states are read only up to
the chain order its last grid point reaches, $q \le n_{\mathrm{kpm}} - 1 +
M(u)$.

**The fold, as dense primitives.** With the aggregate table in hand:

- *Coefficient tables are hoisted.* Forward offset-$k$ coefficients depend
  only on $kh$, shared by every block and chunk; backward coefficients are
  their conjugates and ride the existing per-segment $M(\delta)$ probe.
- *$d_q(u)$ by GEMM.* $d_q(u) = \sum_m c_m(k)\,\bar G_{q,\mathrm{col}_0+m}$
  is one ZGEMM per block over its grid points.
- *The Bessel fold by FFT.* $C_p = \tfrac12\sum_j b_j(d_{p+j} + d_{|p-j|})$
  is a Toeplitz-plus-Hankel product. With the even extension $\tilde D_t =
  d_{|t|}$ and the symmetric kernel $\kappa_0 = b_0$, $\kappa_{\pm j} =
  b_j/2$, it collapses to the single linear convolution $C_p = (\kappa *
  \tilde D)_p$, evaluated by zero-padded FFT (FFTW; one plan pair per
  segment, thread-safe new-array execution): $O(L\log L)$ per grid point in
  place of $O(n_{\mathrm{kpm}} M(\delta))$, which on the production silicon
  task was $\sim 3\times10^{12}$ multiplies even after aggregation.

This is the organising principle of FastCheb-style methods — transform the
moment sequences rather than materialise their full coupling.

**Distribution.** Each rank folds a contiguous share of every block's grid
points from its own GEMM slice, with one correlator reduction at the end, so
the scalar work scales with ranks instead of becoming the Amdahl term of a
large run.

**Memory.** Per rank, $\approx 1.3\,\min(\tilde L, \tilde T)\,
N_{\mathrm{local}} \times 16\,\mathrm{B}$ of segment-local columns
(× chunk) plus scalar tables — bounded by the segment length regardless of
window, and by the chunk regardless of sample count.

### The time integral: cardinal quadrature in closed form

$C_p(u)$ is band-limited — its frequencies are eigenvalue differences,
$|\omega| \le 2$ in scaled units, with content at the band edge, so a
trapezoid on the sampling grid errs at the percent level. The grid samples at
$h \le \pi/4$ (twice Nyquist) and the integral is taken through the cardinal
interpolant $C(u) = \sum_e C[|e|]\,\kappa(u/h - e)$, with $\kappa$ a
Kaiser-windowed sinc of half-width $W_\kappa$ and the exact even mirror
$C_p(-u) = C_p(u)^{*}$. For a sample whose kernel support lies inside
$(0,t)$ the weight is closed form,

$$
\int_0^{t} (t-u)\,\kappa\!\big(u/h - c\big)\,du = h^2\mu_0\big(t/h - c\big),
\qquad \mu_0 = \int\kappa(s)\,ds ,
$$

because the kernel's first moment vanishes by evenness. Only the
$O(W_\kappa)$ samples within a kernel half-width of $u = 0$ and $u = t$ carry
corrections, and those depend on the on-grid offset alone — three
shift-invariant tables, integrated numerically once. With per-moment prefix
sums each output time costs $O(W_\kappa)$: **output grids of any density,
including log-spaced times for $D(E,t)$ extraction, are free.** The evolution
runs one kernel half-width past the window for the right edge; output times
shorter than the two-edge overlap use directly integrated dense weights.

## 6. Correctness

**The two kernels are different unbiased estimators of one trace.** The
identity of §4 uses trace cyclicity, which a single quadratic form
$\langle r|\cdot|r\rangle$ obeys only on average. They therefore agree in
expectation but scatter per sample — a fact with a testing consequence:
per-sample cross-route comparison is meaningless, while summing over a
*complete metric basis* ($\sum_i r_i r_i^{\dagger} = W^{-1}$, e.g. the
columns of a Cholesky factor of $S^{-1}$) makes both the exact trace. That
is what the dense unit tests compare, against references built independently
from each route's own definition:

| check | agreement |
|---|---|
| Orthogonal form, complete basis vs. dense reference | $7\times10^{-11}$ |
| Solve form ($S^{-1}H$), complete basis vs. dense reference | $2.5\times10^{-8}$ (CG tolerance) |
| Sandwich form ($Z^{\dagger}HZ$), vs. dense reference in its own variables | $2.5\times10^{-8}$ |
| Long-window regression ($\tilde T = 291$, multi-segment) | $5\times10^{-10}$ |
| Segmented vs. single-segment, per sample | $2\times10^{-12}$ |
| Sample chunking vs. width-1, per sample | summation rounding |
| Time-tiled vs. per-step stepwise readout | $10^{-10}$ |
| Production silicon grid, trace-first vs. campaign baseline ($81\times1000$) | $3.8\times10^{-8}$ |

All three operator forms carry a dense complete-basis fixture. The suite
runs on 1 and 2 MPI ranks; the spectral and chunking subset also runs on 3,
which exercises an uneven rank deal of the fold. Independently, the two
kernels cross-validate on the production silicon cell at 5.1% median
pointwise deviation with correlation 0.9991 — the residual is the kernel
difference between two estimators, not an error in either.

## 7. Performance, and which kernel runs

**Cost model.** Per unit scaled time the spectral kernel pays $\approx 2 +
n_{\mathrm{kpm}}/\tilde L$ applies plus $\sim 10\,n_{\mathrm{kpm}}$ dot-flops
per matrix row-element; the stepwise kernel pays $\tfrac12 n_{\mathrm{kpm}}$
applies *per output time* plus its propagation. On expensive-apply forms
both kernels' dominant cost is applies of the same operator, so the
comparison reduces to an apply **count** — machine-independent, no flop
rates — derated by the saturating per-apply efficiency ratio
$\mathrm{eff}(w) = w/(w+3)$ between the two kernels' SpMM widths (exactly 1
at equal widths). `spectral_msd_cheaper` evaluates it exactly, Bessel
expansion lengths included.

**Dispatch.** `compute_msd_spectrum(..., method)` takes
`"auto" | "stepwise" | "spectral"`. `"auto"` selects spectral when the
operator form is expensive-apply (Solve or Sandwich) *and* the apply count
favours it; orthogonal forms stay stepwise as a structural rule, because
there the decision would weigh bandwidth-bound SpMVs against compute-bound
GEMM dots, which no machine-independent count settles. Since the kernels are
different unbiased estimators (§6), a parameter sweep that must not switch
estimators mid-sweep should pin `method` explicitly.

**Output-grid independence.** Measured on a Solve lattice fixture (1728
sites, $n_{\mathrm{kpm}} = 300$, $\lambda = 13$ eV), the spectral kernel's
cost tracks the window and ignores the sampling density:

| window | output grid | stepwise | spectral | ratio |
|---|---|---|---|---|
| 12 fs | 48 steps @ 0.25 fs | 34.9 s | 4.7 s | 7.5× |
| 12 fs | 12 steps @ 1.0 fs | 12.4 s | 5.5 s | 2.2× |
| 12 fs | 3 steps @ 4.0 fs | 5.5 s | 5.1 s | 1.1× |
| 192 fs | 48 steps @ 4.0 fs | 81.8 s | 74.3 s | 1.1× |
| 192 fs | 192 steps @ 1.0 fs | 212.4 s | 76.3 s | 2.8× |

The same 192 fs window costs 74.3 s and 76.3 s at 4× different sampling
densities, and the route never falls below parity.

**Scaling** (idle 96-core node; `benchmark_msd_moments run` mode, §9):

| axis | stepwise | spectral |
|---|---|---|
| System size $N$, Solve form (512 → 8000 sites) | baseline | **1.6× → 3.1×** faster, widening with $N$ |
| System size $N$, orthogonal form (1000 → 17576) | **3–6× faster** | dot floor dominates |
| Window (12 → 192 steps) | exponent 1.01 | exponent **0.97** — linear |
| Resolution $n_{\mathrm{kpm}}$ (8× finer) | 2.2× cost | 3.2× cost |
| Samples (1 → 64) | 10.2× cost | **7.9×** cost (trace-first) |
| Peak memory vs window (16× longer) | flat | **1.19×** (165 → 196 MB) |
| MPI strong scaling, $N = 32768$, 8 ranks | 3.5× (43%) | **5.6× (69%)**, 505 → 126 MB/rank |
| OpenMP on 13-orbital LCAO blocks (6 → 24 threads) | **94%** incremental | 69% incremental |

Three honest caveats sit in that table. Cheap orthogonal tight-binding
belongs to the stepwise kernel, by 3–6×, and the dispatch sends it there.
Low-sample spectral runs thread at their chunk width — a 2-sample run applies
width-2 operators where the stepwise readout tiles itself to width 16 — so
threading recovers with more samples or more ranks; the single-orbital
synthetic lattice used above is deliberately adversarial for threading
(pointer-bound SpMV) and saturates far earlier than real blocked operators.
And the dispatch model ignores dot GEMMs, so it over-predicts the spectral
margin (1.8× predicted against 1.18× measured on silicon); it has ranked the
kernels correctly in every measured configuration, but the *margin* is
optimistic.

One corner of the expensive-apply regime admits a stepwise win: very sparse
output sampling of very long windows. When the output step exceeds $\sim
2\tilde L\hbar/\lambda$ ($\tilde L$ memory-capped at 768) the stepwise
moment runs amortise away while the spectral chain term does not — a 10 ps
window sampled at 50 points loses $\approx 1.9\times$. The apply-count model
predicts this corner and the dispatch honours it.

**Production anchor.** LCAO silicon sc444 (1664 orbitals, Sandwich form, 64
samples, 80 × 0.5 fs, $n_{\mathrm{kpm}} = 1000$, 24 threads): full-moment
stepwise 18m16s; spectral 18m02s before the trace-first fold and **15m33s**
after, reproducing the campaign spectrum to $3.8\times10^{-8}$ relative over
the full grid. The profile then shows the fold gone entirely: what remains is
the estimator's own inner products — operator applies and chain-dot GEMMs,
which trace-first deliberately does not touch — so the kernel is compute-bound
on irreducible work.

## 8. Accuracy policy

**Positivity.** At full moments the stepwise kernel is exactly
positivity-preserving (every row a diagonal Jackson density). The spectral
kernel reconstructs a numerator and a density separately and divides; where
the density is Jackson-tiny — outside the spectrum, at thin band edges — the
ratio is noise-dominated and can go negative. Negatives are passed through
rather than clipped, since silently clipping would hide exactly the
diagnostic that says a grid point carries no spectral weight.

**Moment extrapolation is opt-in and off by default.** A Chebyshev moment
sequence $\mu_p = \sum_\alpha w_\alpha \cos(p\theta_\alpha)$ is a mixture of
sampled sinusoids — the signal class autoregressive models represent — so
following the DMRG literature [4, 5, 6] both kernels accept `n_kpm_compute`:
run the recurrence to $n_c$ moments, then extend to $n_{\mathrm{kpm}}$ by
linear prediction at zero operator cost. The fit is covariance least squares
(Tikhonov-regularised normal equations) iterated through the companion-matrix
eigendecomposition with poles renormalised onto the unit circle —
deliberately *not* Burg's lattice method, whose greedy stagewise fit biases
undamped-sinusoid poles to $\approx 0.999$ and decays a 5× extrapolation to
$O(1)$ error where least squares holds $10^{-8}$ over 50×.

On dense spectra it is excellent: quarter-order runs recover full-order runs
to $3\times10^{-11}$; on a semicircle-plus-Lorentzian measure, 300 moments
extended to 1200 reconstruct with 0.1% error where plain truncation errs
12.8%. **On discrete or sparse spectra it fails, measured**: with $n_c = 320$
of 1000 the silicon MSD grid goes 29% negative (worst $-1.2\times10^5$
against a $\sim 10^4$ scale) and $D(E,t)$ deviates $\sim 24\%$ median from
either full-resolution kernel — the moments of a sparse discrete spectrum are
undamped cosines that no finite AR model extrapolates. Aluminium reached 56%
negative points; diamond, dense enough, showed 12.7% falling to 0.4% band-edge
fuzz at full resolution. The knob therefore prints a warning naming this
measurement whenever it is set. The practical criterion for $n_c$: moments are
worth computing while their band-wise signal-to-noise — readable from the
seed-to-seed difference of two sample sets — stays above $\sim 1$.

**Precision.** FP64 throughout; the FP32 chains that KITE and GPUQT run were
declined for this codebase.

## 9. Reproducing the measurements

```bash
# unit tests: dense references for all three operator forms, equalities,
# dispatch model (serial, then multi-rank)
./test_msd
mpirun -n 2 ./test_msd

# one benchmark configuration -> one RESULT line (wall s, peak VmHWM, checksum)
#   run <stepwise|spectral> <orthogonal|solve> <L> <n_steps> <dt_fs> <n_kpm> <nsamples>
OMP_NUM_THREADS=16 ./benchmark_msd_moments run spectral solve 12 12 1.0 300 2

# the default no-argument suite: route timings, dt sweep, noise-floor and
# moment-SNR diagnostics
OMP_NUM_THREADS=16 ./benchmark_msd_moments
```

Scaling sweeps are shell loops over that single-configuration mode, run
sequentially on an otherwise idle machine (a co-running job invalidates the
numbers; peak memory is `VmHWM`, max over ranks, since point RSS readings
miss the transients). The synthetic fixture is a 3D open-boundary
tight-binding lattice in two operator regimes: Solve ($S = 1 + 0.08\,A$,
every apply a batched CG solve) and Orthogonal (bare SpMM). Its disorder is
seeded by global row index, so timings are comparable across rank counts
while checksums differ by realisation. The production fixture and its inputs
live in `pub_exp/transport/` (`si_spectral_bench.json`).

## 10. Surveyed and set aside

- *Sampling axis* (Hutch++/XTrace [7, 8]; Krylov-aware trace estimation [9],
  whose deflation basis could reuse the chain states at no extra applies):
  set aside on the owner's assessment — MSD is a trace observable and
  self-averaging makes one or few samples sufficient at production sizes.
- *Anchor elimination* via $\delta(E-\tilde A)\,U^{-1} = e^{iEt}\,
  \delta(E-\tilde A)$: exact for a true delta, but not for the finite Jackson
  kernel, whose phase error grows like $T/M$. Segmented anchoring stays.
- *Prefix-measured routing* (choosing the kernel from a short timed prefix):
  rejected for determinism — the estimator choice would depend on machine
  load, making runs irreproducible.
- *Few-energy-window regime*: since $\mathrm{Tr}[\delta\,C^{\dagger}C] =
  \mathbb{E}\,\|C\,\delta^{1/2}r\|^2$, applying the half-filter once and
  evolving the filtered state gives a per-sample *positive* estimator at
  $\sim n_{\mathrm{kpm}} + 3K$ per window — the right tool when the question
  is $D(E,t)$ near $E_F$ only (the family of polynomial spectrum slicing,
  EVSL [10]).
- *GEMM subsampling* on the $u$-axis with importance weights: relevant only
  in the cheap-matvec regime the stepwise kernel already serves.

Open directions: $\mathrm{VAC}(E,t)$ and $D(E,t)$ as derivative-kernel
outputs from the same dot table (Green–Kubo integration would then
cross-check the fftcheb `dc_spectrum`); batchwise standard errors and an
$h$ versus $h/2$ quadrature self-check as reported diagnostics; and for full
diffusion tensors, sharing one anchor trajectory across $x/y/z$ against three
$g$-trajectories, which the algebra already permits.

## Implementation map

- `rsatb/calc/msd.hpp` — `corr_msd(...)`: the stepwise kernel (commutator
  pair recursion, doubled moments, grouped time-tiled readout). Knobs
  `n_kpm_compute`, `time_tile`.
- `rsatb/calc/msd_spectral.hpp` — `corr_msd_spectral(...)`: blocks, chain,
  trace-first aggregate table, GEMM $d$-rows, FFT Bessel fold, cardinal
  quadrature. Knobs `time_oversample`, `block_substeps`, `n_kpm_compute`,
  `segment_scaled`, `sample_chunk`. Output contract identical to `corr_msd`
  ($[n_t{+}1,\,n_{\mathrm{kpm}}]$ density-normalised rows on the ascending
  Chebyshev grid). Also `spectral_msd_cheaper(...)`, the dispatch model.
- `rsatb/calc/moment_extrapolation.hpp` — `extrapolate_moments_lp`.
- `rsatb/calc/observables.hpp` — `ComputeMsdSpectrum(..., method,
  n_kpm_compute)`: requires the velocity, dispatches the `method` knob,
  surfaced through `rsatb.hpp`, the pybind layer, `driver.py`
  (`compute_msd_spectrum` / `compute_diffusion_spectrum`), and the JSON
  input schema.
- Validation: `rsatb/calc/test/test_msd.cpp` (the table of §6);
  `rsatb/test_operator_api.cpp` and `test/test_binding.py` (dispatch,
  refusals, and the spectral kernel end-to-end on the LCAO fixture);
  `rsatb/calc/test/benchmark_msd_moments.cpp` (timings and diagnostics).

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
