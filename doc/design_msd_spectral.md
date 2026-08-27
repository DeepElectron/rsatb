# The spectral diffusion-spectrum route

This note derives and specifies `corr_msd_spectral` (rsatb/calc/msd_spectral.hpp):
the energy-resolved mean-square displacement $\mathrm{MSD}(E,t)$ computed at
*resolution-plus-evolution* cost, with every output time obtained by scalar
post-processing. It replaces, wherever its regime applies, the stepwise route
whose cost is *resolution × number of time steps*.

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
with thread-local coefficient builds.

**Memory.** Per rank, $\approx 0.8\, n_{\mathrm{cols}}\, N_{\mathrm{local}}
\times 16\,\mathrm{B}$ of stored columns, $n_{\mathrm{cols}}$ being the
window at $\le \pi/4$ scaled spacing. `memory_tiles` $> 1$ splits the grid
into groups: the evolution still runs once, the chain re-runs per group —
applies scale with the tile count, memory divides by it. Tiled and untiled
runs agree to $2\times 10^{-12}$ (unit test).

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

## 6. When each route wins

The choice hinges on two dimensionless numbers: the output sampling in
scaled time, $\tau_{\mathrm{step}} = \lambda\,\Delta t/\hbar$, and the
computed resolution degree $n_c$. Per sample and axis,

$$
\text{stepwise: } \tfrac12 n_t n_c + 2 n_t (\tau_{\mathrm{step}} + \text{tail}),
\qquad
\text{spectral: } n_c + 2.3\,K \ (+\ \text{GEMM}),
\qquad K = n_t \tau_{\mathrm{step}} + \text{tail},
$$

so the spectral route wins on applies iff roughly
$n_c \gtrsim 5\,\tau_{\mathrm{step}}$, i.e.

$$
\Delta t \;\lesssim\; \Delta t^{*} \;\sim\; \frac{n_c\, \hbar}{5\, \lambda}
$$

— output sampling finer than about a fifth of the resolution's Heisenberg
time. Crucially the spectral cost does not depend on $n_t$ at fixed physical
window. Measured on the Solve lattice fixture (1728 sites, CG per apply,
fixed 12 fs window, $n_{\mathrm{kpm}} = 300$, $\lambda = 13$ eV, so
$\Delta t^{*} \approx 3$ fs):

| output grid | stepwise | spectral | ratio |
|---|---|---|---|
| 48 steps @ 0.25 fs | 37.2 s | 5.3 s | 7.0× |
| 12 steps @ 1.0 fs | 13.3 s | 5.5 s | 2.4× |
| 3 steps @ 4.0 fs | 5.4 s | 5.3 s | 1.0× |

The spectral column is constant across the sweep, as the model requires; at
the crossover the routes cost the same, so choosing spectral whenever
$\Delta t < \Delta t^{*}$ never loses. Historical inputs chose coarse
$\Delta t$ partly *because* the stepwise route priced every extra sample at
$n_{\mathrm{kpm}}$ applies; with sampling free, the fine grids that
$d\,\mathrm{MSD}/dt$ extraction actually wants stop being a luxury.
Cheap-matvec systems (orthogonal tight binding) add the GEMM flops
($8 (n_c + K)\, n_{\mathrm{cols}}$ per matrix row-element) to the spectral
side of the ledger and keep the stepwise route longer. For full diffusion
tensors the chain is axis-independent: $x/y/z$ share one chain against three
evolutions, where the stepwise route triples everything.

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
  right-hand side — how the stepwise commutator route's affine component
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

## Implementation map

- `rsatb/calc/msd_spectral.hpp` — `corr_msd_spectral(...)`: blocks, chain,
  tiled GEMMs, scalar assembly, cardinal quadrature; knobs
  `time_oversample`, `block_substeps`, `n_kpm_compute`, `memory_tiles`.
  Output contract identical to `corr_msd` ($[n_t{+}1,\, n_{\mathrm{kpm}}]$
  density-normalised rows on the ascending Chebyshev grid).
- `rsatb/calc/moment_extrapolation.hpp` — `extrapolate_moments_lp`.
- Validation: `test_msd.cpp` (dense complete-basis fixtures for both forms,
  long-window regression, memory-tiles equality, LP recovery); benchmark:
  `benchmark_msd_moments.cpp` (route timings, dt sweep, noise-floor and
  moment-SNR diagnostics).
