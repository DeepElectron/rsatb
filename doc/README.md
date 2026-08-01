# Temperature-independent Fermi–Dirac operator expansion (benchmark)

Goal: replace the Jackson-KPM evaluation of Fermi–Dirac observables in rsatb
(`rsatb/calc/fermi_dirac.hpp`), whose Chebyshev degree must grow like
`N ~ pi*W/kT ~ beta*W` at low temperature / for metals, with a method whose
cost is (nearly) **independent of temperature**, using only matrix–vector
products (no solves, no matrix–matrix products).

## The idea (weak-form / observable-targeted expansion)

Everything rsatb computes from `f_beta(H)` — electron densities, particle
number, band energy, forces, conductivities — has the form

    q = Tr[A f_beta(H)] = ∫ f_beta(E) rho_A(E) dE,

an integral of the Fermi function against a (weighted, local) density of
states. Two consequences:

1. A degree-N truncation `f_N` only has to be accurate *in the weak sense*,
   integrated against `rho_A`. The pointwise (sup-norm) requirement that
   forces `N = Omega(beta*W)` (a provable lower bound for polynomials) does
   not apply.
2. The truncation residual `f - f_N` of a **kernel-free** Chebyshev expansion
   is an oscillatory Gibbs ringing with wavelength `~ pi*W/N` around mu.
   Integrated against an LDOS that is smooth on that scale, it self-cancels:
   the error is governed by the **local smoothness of the DOS near mu**, not
   by `beta*W`. For extended systems the LDOS is smooth on ~0.1–1 eV scales
   regardless of temperature, so the required degree saturates at a
   T-independent value (a few hundred to a few thousand).

The Jackson kernel destroys exactly this cancellation: it converts the
oscillatory residual into a one-signed Gaussian smearing of width
`sigma ~ pi*W/N`, whose bias only decays like `sigma^2`, and pointwise
convergence of the Fermi edge then demands `sigma <~ kT`, i.e. the full
`beta*W` degree. In other words: **the kernel is the reason low-T KPM is
expensive, not the Fermi function itself.**

## The method as it would ship ("ACE-FD")

1. Expand the *exact* `f_beta` (any T, even T=0) in Chebyshev polynomials —
   no Jackson damping.
2. Sum the series with a smooth high-order taper (raised-cosine / de la
   Vallée Poussin window over the last half of the coefficients). The taper
   reproduces polynomials up to degree N/2 (no smearing bias, unlike
   Jackson) and suppresses the residual ringing far from mu.
3. Adaptive degree with a posteriori error control: double N until
   `|q(N) - q(N/2)| < tol` (moments are a stream; nothing is recomputed).
4. Optional cross-check/alternative: Jackson sums at N, N/2, N/4 from the
   same moments, Richardson-extrapolated in the resolution `sigma ∝ 1/N`
   (removes the sigma^2 and sigma^4 bias analytically).
5. Moments are reused for every (mu, T): the chemical-potential search in an
   SCF/canonical loop costs zero extra MVPs.

Cost per random/deterministic vector: N/2 MVPs (moment doubling trick),
with N set by DOS smoothness, not by beta*W.

## Relation to the literature (short version)

- The weak-form principle is implicit in DFT smearing practice
  (Methfessel–Paxton 1989, Gillan 1989, Marzari–Vanderbilt 1999) and was
  recently made rigorous for spectral sums: undamped Chebyshev moment
  matching is Wasserstein-optimal (Braverman–Krishnan–Musco, STOC 2022;
  Chen–Trogdon–Ubaru, SISC 2025). High-order kernels with smoothness-limited
  convergence for the *delta function/LDOS* appeared in Yi et al.,
  arXiv:2512.03149 (2025).
- Sub-linear-in-beta alternatives all need operations rsatb cannot afford:
  pole/rational expansions need solves (Ozaki 2007; Lin–Lu–Ying–E 2009;
  PEXSI; Moussa 2016), telescopic/recursive FOE needs matrix–matrix products
  (Ceriotti–Kühne–Parrinello 2008; Niklasson-lineage).
- Uniform (pointwise) approximation of `f_beta(H)v` by MVPs alone provably
  needs degree `Omega(beta*W)`; the trace/observable setting is the loophole,
  and it is exactly rsatb's setting.

## Files

- `chebfd.py` — coefficients, summation rules (Jackson / kernel-free / VP /
  cos tapers / Richardson), exact-spectrum and stochastic (doubling) moments,
  Lanczos/SLQ baseline, adaptive driver.
- `bench_bias.py` — pure truncation-bias study on exact spectra (2D/3D
  metals, t-t' van Hove case, doped + near-neutral graphene, gapped
  insulator, vacancy LDOS on a finite lattice), T = 300/100/10/1 K.
- `bench_stochastic.py` — real sparse-MVP stochastic-trace benchmark on a
  256x256 lattice metal incl. SLQ comparison.
- `analyze.py` — figures + tables (`fig_*.png`, `tables.md`).
- `RESULTS.md` — findings and speedup summary.

Run (conda env `rsatb`):

    conda run -n rsatb python bench_bias.py
    conda run -n rsatb python bench_stochastic.py
    conda run -n rsatb python analyze.py

## Scope / caveats

- Applies to spectrally *integrated* quantities (density, N_e, E_band,
  forces, finite-broadening response). Not to sharply energy-resolved
  spectral functions — there the resolution/degree tradeoff is fundamental.
- For a *finite* system whose level spacing near mu exceeds kT (small
  clusters at very low T), the advantage shrinks; the a posteriori estimator
  detects this and the method degrades gracefully to Jackson-like cost.
- DOS singularities exactly at mu (van Hove, band edge, Dirac point) reduce
  the convergence order; still far cheaper than sigma=kT practice, see
  benchmark.
