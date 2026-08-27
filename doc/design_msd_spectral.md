# Spectral (single-chain) diffusion spectrum: MSD(E, t) without per-step KPM

## The cost structure being replaced

Every published linear-scaling implementation of the energy-resolved MSD --
the LSQT review (Fan, Vila, Roche et al., Phys. Rep. 903, 1 (2021), Eqs.
67-70), GPUQT (Fan, Zheng, Harju, Comput. Phys. Commun. 230, 113 (2018)), and
rsatb's own `corr_msd` -- evaluates

    MSD(E, t) * rho(E) = < phi(t) | delta(E - H) | phi(t) >,
    phi(t) = [X, U(t)] |r>

by placing the KPM expansion of delta(E - H) BETWEEN two time-evolved states
and re-expanding it at every output time: n_t independent n_kpm-term
recurrences, cost n_t * n_kpm operator applies (n_t * n_kpm / 2 after the
metric-weighted moment doubling in msd.hpp). The propagation itself is the
small term. The concurrent shared-state framework of Fu, Li, Zhou, Yuan
(arXiv:2510.17432), which rsatb implements for the AC correlator and the DOS
(`corr_ac_concurrent`, `corr_from_moments`), does not touch this term: its
gain is the Bessel tail of e^{-iA dt} amortised over a block, and the moment
recurrence has no tail. The obstruction is structural: delta sits between two
states that both move.

## The reformulation: move the resolution onto the static state

The exact identity (stationarity of the velocity autocorrelation plus
[T_p(A), U] = 0)

    <phi(t)| T_p(A~) |phi(t)>
      = int_0^t ds int_0^t ds' Tr[ v(s) T_p(A~) v(s') ]
      = 2 Re int_0^t (t - u) C_p(u) du,
    C_p(u) = Tr[ T_p(A~) v(u) v ],      v(u) = U(u)^{-1} v U(u)

moves all the time dependence into ONE Heisenberg factor. Expanding the
left evolution in Chebyshev, U(u)^{-1} = sum_j c_j(u)^* T_j(A~) (the scalar
coefficient identity holds for any A, Hermitian or not), and folding
T_p T_j = (T_{p+j} + T_{|p-j|}) / 2 onto the SAME chain:

    C_p(u) = sum_j c_j(u)^* [ d_{p+j}(u) + d_{|p-j|}(u) ] / 2,
    d_q(u) = < W T_q(A~) r | D g(u) >,     g(u) = U(u) D r

where D is the form-dressed velocity (`apply_dual`), W the form's metric (S
under Solve, identity otherwise; W T_q(A~) = T_q(A~)^H W is the same
self-adjointness the doubled moments in msd.hpp lean on), and r the
covariance-W^{-1} stochastic sample (MetricSampler). Every object now falls
into one of two one-time computations:

  - ONE left chain  T_q(A~) r,  q < n_kpm + K   (K = T_max * half_width / hbar
    plus the Bessel tail: the Chebyshev content of the full evolution window),
  - ONE evolution   g(u) = U(u) D r  on a Nyquist-oversampled grid, done with
    the arXiv:2510.17432 block-concurrent scheme so its cost is ~K applies,
    not (samples x per-step tail).

The dot table d_q(u) couples them: a (n_kpm + K) x n_samples ZGEMM. The
number of OUTPUT times no longer appears in the operator-apply count at all;
MSD(E, t) at any t <= T_max is scalar post-processing.

    stepwise (doubled):   n_t * n_kpm / 2                  applies
    spectral:             (n_kpm + K)  +  K  + n_u         applies
                          (chain)     (evolve) (D per grid point)
                          + (n_kpm + K) x n_u dot table (GEMM)

This is the information-theoretic floor shape: evolving over [0, T] cannot
cost less than ~K applies (no fast-forwarding), resolving delta to n_kpm
cannot cost less than ~n_kpm -- the spectral route pays their SUM, the
stepwise route their PRODUCT-like combination. Asymptotically in T the win is
n_kpm / (4 tau_step), tau_step = half_width * dt / hbar.

## Where each route wins

The spectral route trades operator applies for GEMM flops
(8 * (n_kpm + K) * n_u per matrix element-row). It wins when applies are
expensive -- the Solve form (one CG per apply) and Sandwich form (three
SpMVs), i.e. exactly rsatb's non-orthogonal LCAO/DFTB targets -- and when
K is not enormous compared to n_kpm. For cheap orthogonal tight-binding
matvecs with very long windows (graphene-scale K >> n_kpm), the stepwise
route with doubling remains the right tool. Route selection stays an explicit
caller choice; estimate_matvecs.py can carry the crossover model.

Silicon-like fixture (n_t = 81, n_kpm = 1000, tau_step ~ 15): stepwise
40.5k CG-applies vs spectral ~2.3k chain + ~1.3k evolve applies -- an order
of magnitude on the apply count.

## Time quadrature: windowed-sinc, not trapezoid

C_p(u) is band-limited: its frequencies are eigenvalue differences, |omega|
<= 2 in scaled units (the same bound behind ac.hpp's kAcOversample). The
grid samples at dsub <= pi/4 (2x Nyquist margin). The triangular-kernel
integral int_0^t (t - u) C_p(u) du is then evaluated with cardinal
(Kaiser-windowed sinc) quadrature: C_p is reconstructed exactly (to window
error ~1e-9) from its samples, and the weights

    W_k[u] = int_0^{t_k} (t_k - u') kappa((u' - u_u)/h) du'

are p-independent scalars integrated once. A naive trapezoid at this spacing
would err at the percent level for band-edge frequencies; the cardinal
weights make the quadrature spectrally accurate. Re C_p(u) is even in u
(C_p(-u) = C_p(u)^*), so the grid mirrors exactly through u = 0; the right
edge is covered by evolving one kernel-half-width past T_max.

## Estimator equivalence

Stepwise and spectral estimators have the SAME expectation (the physical
trace) but differ per sample: the identity above uses trace cyclicity, which
a single quadratic form <r|...|r> obeys only on average. Summed over a
complete metric basis (sum_i r_i r_i^H = W^{-1}, e.g. the columns of
S^{-1/2}) both become the exact trace, which is what the unit tests compare
-- against dense references and against each other. Variance at finite
samples is checked empirically; the usual self-averaging of large systems
applies to both.

## What this buys beyond speed

- No position operator anywhere: only the form-dressed velocity. PBC needs
  no affine split and no pair generator; one implementation covers every
  basis, boundary condition, and operator form.
- Output times are free: MSD(E, t) on an arbitrarily fine t-grid from the
  same table. VAC(E, t) and D(E, t) = d(MSD)/2dt come from the same d-table
  with derivative kernels (future work), and Green-Kubo integration of
  VAC(E, t->inf) cross-checks the fftcheb dc_spectrum.
- The d-table is the time-domain sibling of the Kubo-Bastin two-index object
  fftcheb_transport already streams; the tiling/memory story is the same.

## References

- Fan, Vila, ..., Roche, "Linear scaling quantum transport methodologies",
  Phys. Rep. 903, 1 (2021) -- Eqs. 67-70 define the stepwise state of
  practice this design replaces.
- Fan, Zheng, Harju, GPUQT, Comput. Phys. Commun. 230, 113 (2018).
- Fu, Li, Zhou, Yuan, arXiv:2510.17432 -- the concurrent propagation
  framework; this design extends its moment-based variant from the
  one-operator DOS correlator to the two-operator diffusion correlator.
- Weisse, Wellein, Alvermann, Fehske, Rev. Mod. Phys. 78, 275 (2006) --
  KPM, kernels, and the product-identity moment doubling.
- Joao et al., KITE, R. Soc. Open Sci. 7, 191809 (2020); Ferreira &
  Mucciolo, PRL 115, 106601 (2015) -- large-scale two-index Chebyshev
  tables for Kubo formulas (the energy-domain siblings of the d-table).

## Measured (2026-08, single node, benchmark_msd_moments)

- Dense complete-basis validation (test_msd): spectral route vs exact dense
  reference, orthogonal form 2.0e-8, Solve form 4.3e-8 absolute on O(1)
  spectra; long-window regression (T~ = 291 scaled time, 33 output steps)
  3.7e-8 -- the cardinal quadrature and coefficient truncations do not
  accumulate. Stepwise and spectral agree to 8 digits on the complete-basis
  Solve fixture.
- Solve-form lattice fixture (1728 sites, CG per apply, n_kpm = 300,
  48 output steps at dt = 0.25 fs): stepwise 36 s, spectral 6-8 s --
  5-6x measured, matching the apply-count model (~9.8k vs ~1.3k CG applies
  per sample). The asymptotic model is n_kpm / (4 tau_step) with output
  times free.
- At few stochastic samples the two routes scatter per-sample as different
  unbiased estimators of one trace; the benchmark prints the stepwise
  seed-to-seed noise floor next to the cross-route deviation to keep that
  legible (0.62 vs 1.49 max-over-bins on the 1728-site fixture with 2
  samples and ~6 states per energy bin; both shrink with system size and
  samples in the usual self-averaging way).
- Exposed and fixed on the way: the batched CG/PCG in solver.hpp had no
  per-column breakdown guard, so any all-zero column (the commutator MSD's
  affine seed) poisoned the whole block with NaN.

## Implementation map

- `rsatb/calc/msd_spectral.hpp` -- `corr_msd_spectral(...)`: the left chain,
  block-concurrent evolution, d-table GEMM, and post-processing. Same output
  contract as `corr_msd` ([n_steps+1, n_kpm] rows over the Chebyshev grid,
  density-normalised via the shared `write_msd_row`).
- Validation: `test_msd.cpp` dense complete-basis fixtures, orthogonal and
  Solve forms, plus the dense identity check num_from_VAC == num_from_[X,U].
- Benchmark: `benchmark_msd_moments.cpp` gains a spectral-route timing on the
  Solve fixture (the regime it targets).
