# Variance reduction for stochastic real-space maps: subspace deflation and block-local capture

> **Superseded (2026-09-05).** The code this note designed -- block-local
> capture, packed image sweeps, sign replicas, the distance audit, the
> certified routing, and the cross-fitted global deflation with its residual
> estimator -- was removed in the production cleanup. The LDOS map has one
> estimator, the survey-sampling estimator of
> `design_survey_sampling_ldos.md`; the density has one, the fragment
> surrogate. This note is kept as the record of what was tried, measured and
> why it lost (sections 16-17).


Method notes. This document defines and derives, from scratch, the
variance-reduction stack of `rsatb/calc/rs_ldos.hpp`:

* **global subspace deflation** (`subspace_cv`) with its **frequency-comb
  enrichment** (`subspace_filters`),
* **block-local capture** (`local_capture`, `local_pack`),
* and the **operator-form geometry** (sandwich vs. solve) under which both
  are valid, including the solve-form fragment surrogate chain.

It assumes linear algebra and basic probability, nothing about the code
base; every symbol is defined before use. Companions:
`design_realspace_density_ldos.md` (grid pipeline, fragment control
variate machinery), `design_stochastic_ldos.md` (plain estimator and
metric sampler). Measurement history, per-campaign numbers, and
abandoned alternatives live in `pub_exp/spectrum_realspace/RESULTS.md`;
here only the results that calibrate a constant or certify a claim are
kept (section 11).

---

## 1. Setting and notation

**Basis and pencil.** A localized (LCAO) basis of $N$ real orbitals
$\{\phi_s(\mathbf r)\}$, generally non-orthogonal, with sparse

$$
H_{st} = \langle \phi_s|\hat H|\phi_t\rangle, \qquad
S_{st} = \langle \phi_s|\phi_t\rangle .
$$

The generalized eigenproblem $H C = S C\,\mathrm{diag}(\varepsilon_n)$
has eigenpairs $(\varepsilon_n, \mathbf c_n)$ normalized $C^T S C = I$;
the eigenstate in real space is
$\psi_n(\mathbf r) = \sum_s \phi_s(\mathbf r)(\mathbf c_n)_s$.

**Target.** The LDOS map at probe energies $E_k$, broadening $\eta$:

$$
\rho(\mathbf r, E) = \sum_n |\psi_n(\mathbf r)|^2\, g_\eta(E-\varepsilon_n),
\qquad g_\eta = \text{Gaussian of std } \sqrt2\,\eta ,
$$

evaluated at every point of a real-space grid. The evaluation matrix
$\Phi \in \mathbb R^{P\times N}$, $\Phi_{ps} = \phi_s(\mathbf r_p)$, is
never formed; the code applies it box by box. The electron density is
the same machinery with $g_\eta \to$ Fermi–Dirac occupation.

**Filters.** With the spectrum scaled into $[-1,1]$
($x = (\varepsilon - c)/h$, $c, h$ from padded bounds), a *filter* is a
Chebyshev table $p(x) = \sum_{m<d} a_m T_m(x)$; applying $p(\mathrm{Op})$
to a vector costs $d$ operator applications ("matvecs") via the
three-term recursion, and one recursion serves every table
simultaneously (the energy dependence is only in the scalar $a_m$). The
map uses the **square root of the broadening kernel**,
$p_k^2(x) = g_\eta(E_k - (c + hx))$, of degree $d \sim h/\eta$.

**Factor.** $\tilde Z \approx S^{-1/2}$ (the `isqrtm` module), with
$W := \tilde Z^T S \tilde Z$ measuring its quality ($W = I$ when
purified). $q$ denotes the metric sampler's polynomial,
$q(W) \approx W^{-1/2}$ on $W$'s measured interval, so that
$L := \tilde Z q(W)$ satisfies $L L^T = S^{-1}$ **exactly up to the
sampler's residual certificate, for any invertible $\tilde Z$**.

**Seeds.** $\boldsymbol\xi \in \{\pm1\}^N$ Rademacher,
$\mathbb E[\boldsymbol\xi\boldsymbol\xi^T] = I$; the sign of entry $s$
of sample $i$ is a counter hash of (seed, global row $s$, $i$), so the
draw is identical at any MPI rank count.

### 1.1 The two operator forms, unified

Both forms compute, for each energy and sample, a state $\chi$ whose
grid square estimates the map; both are the same white-noise estimator
$\chi = \mathcal T_k\,\boldsymbol\xi$ in different factorizations of the
chain $\mathcal T_k$:

| | **sandwich** | **solve** |
|---|---|---|
| iterated operator | $M = (\tilde Z^T H \tilde Z - c)/h$, symmetric | $B = (S^{-1}H - c)/h$, self-adjoint in $\langle u,v\rangle_S = u^T S v$ |
| recursion seed | $\boldsymbol\xi$ (white) | $\mathbf c_0 = L\boldsymbol\xi$, $\;\mathbb E[\mathbf c_0\mathbf c_0^T] = S^{-1}$ |
| stored state $\mathbf y_i$ | $p_k(M)\,\boldsymbol\xi_i$ | $p_k(B)\,\mathbf c_0^{(i)}$ |
| grid map of a state | $\Phi\,\tilde Z\,\mathbf y$ | $\Phi\,\mathbf y$ |
| whole chain $\mathcal T_k$ | $\tilde Z\, p_k(M)$ | $p_k(B)\,\tilde Z\, q(W)$ |
| whitening inner product | Euclidean | $S$ inner product |
| exact answer of | the $\tilde Z$-pencil ($\tilde Z$ is part of the operator) | the true $(H,S)$ pencil, any invertible $\tilde Z$ |
| cost per matvec | 3 SpMMs | one CG solve with $S$ |

The **whitening inner product** column is the entire dictionary between
the forms: every construction below whitens, projects, and contracts in
that inner product, and is otherwise identical. The reason is:

**Proposition 1 (gauge equivalence).** *Let $Z = S^{-1/2}$ exactly and
$A = ZHZ$ (scaled). Then $B = Z A Z^{-1}$, hence
$p(B) = Z\,p(A)\,Z^{-1}$ for every polynomial, and with
$O := Z^{-1}L$ (orthogonal, since $L L^T = S^{-1} = ZZ^T$),*

$$
\chi_{\text{solve}} = p(B)\,L\,\boldsymbol\xi
= Z\,p(A)\,(O\boldsymbol\xi).
$$

*The solve form is the sandwich estimator with the exact factor and a
rotated white seed: same law, and for a purified factor ($O = I$)
literally the same sample paths.* ∎

Consequently every statistical statement in this document — the noise
floor, capacity laws, comb stride, Nyquist bound, capture intensivity,
packing noise — is a statement about $p(A)$ acting on white noise and
holds for both forms. What differs is *where inexactness lands*
(sandwich: in the operator; solve: in certified iteration counts) and
the cost column. The sandwich is strictly cheaper; the solve form is the
tool for an unpurifiable $\tilde Z$.

Throughout, statements are written in the sandwich form's Euclidean
notation; section 9 gives the finite-$\tilde Z$ solve-form derivations
and the implementation dictionary.

---

## 2. The plain estimator and its noise floor

For any matrix $T$ and white $\boldsymbol\xi$,
$\mathbb E\,|(T\boldsymbol\xi)_p|^2 = (TT^T)_{pp}$. With
$T = \Phi\,\mathcal T_k$ and either form's chain,

$$
TT^T = \Phi\, Z\, p_k^2(A)\, Z\,\Phi^T
\quad\Longrightarrow\quad
(TT^T)_{pp} = \sum_n |\psi_n(\mathbf r_p)|^2\, g_\eta(E_k - \varepsilon_n)
= \rho(\mathbf r_p, E_k),
$$

using $p_k^2 = g_\eta$ and $\mathbf c_n = Z\mathbf v_n$. Hence the
**plain estimator**

$$
\hat\rho_{\text{plain}}(\mathbf r_p, E_k)
= \frac1n \sum_{i=1}^n |(\Phi\,\mathcal T_k\,\boldsymbol\xi_i)_p|^2
$$

is unbiased, and non-negative pointwise by construction (a mean of
squares — no Jackson kernel needed for positivity, so the full $\sqrt2$
of resolution is kept).

**Proposition 2 (the $\chi^2$ floor).** *Expanding a sample amplitude in
the eigenbasis, $(T\boldsymbol\xi)_p = \sum_n \alpha_{pn}\gamma_n$ with
$\gamma_n = \mathbf v_n^T\boldsymbol\xi$ approximately i.i.d. standard
normal, gives
$\mathrm{Var}\,|(T\boldsymbol\xi)_p|^2 \approx 2\rho(\mathbf r_p,E_k)^2$,
so the pointwise relative standard error is*

$$
\sigma_{\text{rel}} = \sqrt{2/n},
$$

*independent of $N$, $E$, $\eta$, and the grid.* (17.7% at $n = 64$;
measured 17.0% against dense diagonalization.) Unlike traces, a point
value has no self-averaging; this floor is what the rest of the
document attacks. Conversely it gives the unconditional,
$N$-independent budget $n = 2/\epsilon^2$ for any fixed target.

**Effective window rank.** Define

$$
m_{\text{eff}}(E_k) = \#\{\,n : g_\eta(E_k-\varepsilon_n) >
10^{-12}\max g_\eta\,\}
$$

— the numerical rank of $p_k(A)$, i.e. the states within
$|E_k - \varepsilon_n| \lesssim 5\eta$. In a crystal at an in-band
energy $m_{\text{eff}} \propto N$; at gaps, band edges, and defect
levels it is small. Every mechanism below is governed by
$m_{\text{eff}}$ against a *capacity*.

---

## 3. The window sketch

**Proposition 3 (the samples span the window).** *Each stored state has
the exact spectral form*

$$
\mathbf y_i = p_k(A)\,\boldsymbol\xi_i
= \sum_n \mathbf v_n\, p_k(x_n)\,\gamma_{n,i},
\qquad \gamma_{n,i} = \mathbf v_n^T \boldsymbol\xi_i
$$

*(tables act on eigenvectors by scalar evaluation, by induction on the
recursion). The $\gamma_{n,i}$ are uncorrelated with unit variance, so
$Y = [\mathbf y_1 \cdots \mathbf y_q]$ is a randomized range sketch of
$p_k(A)$: for $q \ge m_{\text{eff}}$, $\mathrm{span}(Y)$ equals the
window span with probability one; for $q < m_{\text{eff}}$ it is a
random $q$-dimensional slice.* ∎

The estimator computes $Y$ anyway; the only change is storing it
($n_E\, n\, N_{\text{loc}}$ doubles per family) instead of discarding it
after the squaring. Equivalently: one step of Chebyshev-filtered
subspace iteration whose filter is the production filter itself.

**Whitening (Gram route).** An orthonormal basis of the sketch span, in
the form's whitening inner product $\langle\cdot,\cdot\rangle$:

$$
G = \big[\langle \mathbf y_i, \mathbf y_j\rangle\big]_{ij}
\;\;(\text{one Allreduce}), \qquad
G = V\,\mathrm{diag}(s_j)\,V^T, \qquad
B = V_r\,\mathrm{diag}(s_j^{-1/2}), \qquad
Q = YB,
$$

with $r$ the count of retained eigenvalues. Three properties are
load-bearing:

1. *The rank filter is the physics.* When $m_{\text{eff}} < q$ the
   columns of $Y$ are dependent and $r$ **is** $m_{\text{eff}}$ —
   near-dependence is the signal, not a nuisance. Relative cutoff:
   $s_j > 10^{-12} s_{\max}$.
2. *The absolute floor.* The relative cutoff assumes the window holds at
   least one state. At an **empty** window every $\mathbf y_i$ is
   far-tail response plus sweep roundoff, the Gram spectrum is nearly
   flat, and every direction passes the relative test; the whitening
   $s^{-1/2}$ then amplifies stored-image roundoff into an $O(1)$
   artifact. The floor $s_j > 10^{-10}\big(\sum_m |a_m|\big)^2$
   is set against the filter's own scale ($\sum|a_m|$ bounds
   $\max|p_k|$): a discarded direction responds below $10^{-5}$ of the
   filter peak and contributes nothing the plain estimator cannot
   absorb. The constant is pinched from both sides by measurement:
   empty-window noise directions sit near
   $4\times10^{-12}(\sum|a_m|)^2$ (must drop), while in-gap maps of a
   semiconductor are legitimately carried by band-edge tails with $s_j$
   down to $10^{-8}(\sum|a_m|)^2$ (must keep).
3. *Determinism across MPI layouts.* $G$ is Allreduced identically
   everywhere and the small eigensolve is replicated, so $Q$ is a
   deterministic function of $Y$, itself layout-invariant by the
   counter seeding.

---

## 4. Exact filtered images and the comb-enriched family

Using $Q$ requires its filtered image $p_k(A)\,Q$. Naively that is $r$
new sweeps; it is instead free:

**Proposition 4 (image identity).** *From
$T_iT_j = \tfrac12(T_{i+j} + T_{|i-j|})$, the square of a table is a
table ($O(d^2)$ scalar coefficient work, `SquaredTable` — host-side,
no matvecs). Storing $\mathbf z_i = p_k^2(A)\boldsymbol\xi_i$
alongside $\mathbf y_i$, any sketch-span vector
$\mathbf q = Y\mathbf b$ has*

$$
p_k(A)\,\mathbf q = Z\mathbf b, \qquad\text{in particular}\qquad
U := p_k(A)\,Q = ZB
$$

*— exact to the table truncation, no new sweeps, pure polynomial
algebra (valid in every operator form).* ∎

**Degree of the squared table (the robustness question).** The images
are not literally free: the recursion runs to the longest truncated
table, and $p_k^2$'s length must be controlled — *guaranteed*, not
observed. Three nested guarantees, strongest last:

1. *Unconditional cap.* The fold is exact at degree $2d-1$, so for ANY
   filter the sweep can never more than double. Algebra, no analysis.
2. *The Gaussian pair: a uniform closed form.* Under $x = \cos\theta$,
   Chebyshev coefficients are Fourier cosine coefficients; a Gaussian
   of $x$-width $\sigma$ at $x_0$ has $\theta$-width
   $\sigma_\theta = \sigma/\sqrt{1-x_0^2}$, coefficients
   $\sim e^{-m^2\sigma_\theta^2/2}$, and $\varepsilon$-truncated
   length $m_\varepsilon = \sqrt{2\ln(1/\varepsilon)}\,
   \sqrt{1-x_0^2}/\sigma$. The sqrt-filter has $\sigma = \sqrt2\,
   \bar\eta$, its square $\sigma = \bar\eta$; every $\eta$, span, and
   tolerance dependence cancels in the ratio:
   $\mathrm{len}(p_k^2)/d = \sqrt2\,\sqrt{1-x_0^2} \le \sqrt2$,
   with $d$ the center-worst-case base degree the code sizes all
   tables to. Verified numerically: ratio 1.407–1.411 at band center
   across a 2.4× range of $\eta$ ($\to\sqrt2$, $\eta$-independent as
   claimed); 0.615 at $x_0 = 0.9$ vs the predicted 0.617. So the
   guaranteed worst case is a 41% sweep extension — at the scaled band
   center only — and for $|x_0| > 1/\sqrt2$ the squared table is
   *shorter* than the base and costs nothing. Physical probe energies
   in an LCAO basis sit far down the scaled spectrum (the span runs to
   hundreds of eV), typically $|x_0| \sim 0.8$–$0.95$, which is why
   measured deflation overheads are percent-level, not 41%.
3. *Analyticity-limited filters (the density's sqrt-Fermi).* Both $p$
   and $p^2$ are singularity-limited by the same complex points (the
   Fermi factor's Matsubara structure), so both decay geometrically
   with the same Bernstein-ellipse rate and
   $\mathrm{len}(p^2)/\mathrm{len}(p) \to 1$: measured 1.08–1.09
   across a 16× range of $h/k_BT$. The entire-function Gaussian is
   the worst case of the family, and it is capped by (2).

With the comb, the deepest table is $T_{(J-1)s}\cdot p_k^2$, so
$\mathrm{degree\_all} \approx d\cdot\max\big(1,\;
\sqrt{2(1-x_0^2)}\big) + (J-1)s$ — the two extensions compose
additively and both are bounded a priori.

### 4.1 The block-Krylov family and the frequency comb (`subspace_filters`)

The sketch rank need not stop at the sample count: the recursion passes
through every $T_m(A)\boldsymbol\xi$, so the *modulated* tables
$T_j\cdot p_k$ (coefficient shift-folds, degree $+j$) yield extra
families $T_j(A)\mathbf y_i$ — still inside the window span — whose
exact images $T_j\cdot p_k^2$ ride the same sweep. $J$ families give up
to $Jq$ sketch dimensions from $q$ samples.

**Which $j$: the comb.** $T_j(\cos\theta) = \cos j\theta$ varies on the
angular scale $1/j$, while the $\pm5\eta$ window spans
$\delta\theta = (10\eta/h)/\sqrt{1-x_0^2}$. For small $j$ the modulated
states are nearly collinear inside the window (measured: rank pinned at
exactly the bare capacity for $J=4$ consecutive on a metallic window).
Two modulations decorrelate over the window once
$|j-j'|\,\delta\theta \gtrsim \pi$, so the family is the comb

$$
j \in \{0, s, 2s, \dots, (J-1)s\}, \qquad
s = \left\lceil \frac{\pi\sqrt{1-x_0^2}}{10\eta/h}\right\rceil
\approx \frac d{10},
$$

derived per energy (no option). Each family is then an independent
shape on the window — which supports $\sim 10$ independent degree-$d$
polynomials, a system-independent count — and **capacity multiplies by
$J$** at sweep cost $\times(1 + J/10)$ (the deepest image table has
degree $d + (J-1)s$). This is the only superlinear knob in the stack;
past $J \sim 10$ (the window's polynomial capacity) it turns linear, at
$10\times$ better rate than samples.

---

## 5. The deflation estimator

Let $\Pi$ be the projector onto $\mathrm{span}(Q)$, orthogonal in the
whitening inner product, and split every seed:
$\boldsymbol\xi = \Pi\boldsymbol\xi + (I-\Pi)\boldsymbol\xi$.

**Proposition 5 (exact orthogonal split).** *For a projector orthogonal
in the metric in which the seed is white, the cross term of the expanded
square vanishes exactly,
$\mathbb E[\Pi\boldsymbol\xi\,\boldsymbol\xi^T(I-\Pi)^T] = 0$, and the
captured part is deterministic,
$\mathbb E[\Pi\boldsymbol\xi\boldsymbol\xi^T\Pi^T] = \sum_j q_jq_j^T$.
Hence*

$$
\boxed{\;
\hat\rho(\mathbf r_p, E_k)
= \underbrace{\sum_{j=1}^r |(\Phi Z\, U_{:,j})_p|^2}_{\text{captured (deterministic)}} +
\underbrace{\frac1n \sum_i |(\Phi Z\,(\mathbf y_i - U C_{:,i}))_p|^2}_{\text{deflated (sampled)}}
\;}
$$

*with $C_{:,i} = Q^T\boldsymbol\xi_i \in \mathbb R^r$ the projection
coefficients of the seed (small reduced products; the seeds regenerate
from their counters).* ∎

**Proposition 6 (cross-fitting: unbiased at any sketch quality).**
*$\Pi$ is built from samples; if the same samples build it and are
deflated by it, the expectations above no longer factor and the
estimator is biased. The remedy is the exchangeability discipline of
XTrace: block $b$'s seeds are deflated against the span of all OTHER
samples' families (leave-one-block-out; the full Gram is built once,
each left-out basis is a principal-submatrix eigensolve). Conditional
on the other blocks, $\Pi$ is a fixed matrix and each block's two-term
sum is exactly unbiased — whatever the capture quality. Capacity
reaches $J(n - w)$ for block width $w$.* ∎

**Proposition 7 (never worse, pointwise, no fitted coefficient).** *Row
$p$ of the deflated operator is the orthogonal projection of the plain
row, so its norm — and by Proposition 2 its variance — can only shrink,
at every grid point, deterministically. At captured mass fraction
$\varphi_p$,*

$$
\frac{\mathrm{Var}_{\text{deflated}}}{\mathrm{Var}_{\text{plain}}}
= (1-\varphi_p)^2 .
$$

*Remark (why not a control variate).* A regression control on the
squares removes only the captured part's own fluctuation, not the
interference between captured and rest: writing a sample amplitude as
$A + B$, the control leaves $2AB$ and achieves $1 - \varphi^2$ against
the split's $(1-\varphi)^2$ — at $\varphi = 1/2$ a 1.3× reduction
against 4×. Deflating the *seeds*, not correcting the *squares*, is
what kills $2AB$. (Measured before this design: the control fit
$\beta = 1.00$ and moved nothing at partial capture.)

### 5.1 Algorithm (per energy $E_k$; both forms)

```
sweep stores per family:  Y (N_loc x nJ)  y = p_k(Op) seed,  and their
                          exact images Z = (T_js * p_k^2)(Op) seed
solve form additionally:  SY = S*Y (one SpMM per energy at assembly),
                          c0_store (the colored seeds)
1  G = Allreduce(Y^T Y)          sandwich   |   Allreduce(Y^T SY)  solve
2  for each left-out block b: principal submatrix of G ->
     eig, keep s_j > max(1e-12 s_max, 1e-10 (sum|a_m|)^2) -> rank r
     B = V_r s^{-1/2}
3  Q = Y_sub B  (sandwich)  |  Q = SY_sub B  (solve)   ;   U = Z_sub B
4  captured map += (w_b/n) * grid-square of U's columns
5  seeds of block b:  xi from counters  |  c0 from c0_store
6  C = Allreduce(Q^T seed)                                  (r x w)
7  deflated states Yt = Y[:,b] - U C ; sampled map += grid-square
grid-square = [sandwich: apply Zt] -> halo -> per-box square
```

Communication: one $(nJ)^2$ and one $r\times w$ Allreduce per
(energy, block) — no collective grows with $N$. All heavy work is thin
dense algebra of width $\le nJ$ against $N_{\text{loc}}$ rows plus one
extra pass of the existing grid pipeline.

---

## 6. Capacity geometry: regimes, ladder, and the matvec floor

Everything is the single ratio $m_{\text{eff}}(E_k)$ vs. capacity
$J(n-w)$:

* **Exact regime, $m_{\text{eff}} \le J(n-w)$:** the sketch spans the
  window, $\varphi \to 1$, the deflated states vanish to the filter
  cutoff; the map is exact to filter leakage. The capture cliff sits at
  capacity $\approx 1.3\,m_{\text{eff}}$ (sketch oversampling: measured
  capacity$/m_{\text{eff}} = 1.19 \to \varphi = 0.967$,
  $1.05 \to 0.717$).
* **Partial regime:** average capture
  $\varphi \approx \text{capacity}/m_{\text{eff}}$, error
  $(1-\varphi)\sqrt{2/n}$ — rising toward, but never crossing, the
  plain floor as $N$ grows at fixed knobs (Proposition 7). Holding
  exactness at an in-band energy as $N$ grows requires capacity
  $\propto N\rho(E_k)$ — unavoidable in *any* method: an extensive
  invariant subspace has no sub-extensive representation. Section 7
  is the way through that wall.
* **Sparse windows are the payoff:** gaps, band edges, defect and
  disorder-localized levels have small, often $N$-independent
  $m_{\text{eff}}$; there the deflation is exact-class at zero extra
  sweeps, permanently.

### 6.1 Equal-budget guarantees: deflation vs. spending the same matvecs on samples

The images and stores are not free (section 4's degree law), so the
fair question is: at equal matvec budget, when is deflation guaranteed
to beat simply adding samples? Define the **win**
$G = (\epsilon^2 C)_{\text{samples}} / (\epsilon^2 C)_{\text{defl}}$;
the sample route has $\epsilon^2 C = 2$ identically (it lives ON the
frontier), and deflation has cost factor
$\kappa = \max(1, \sqrt{2(1-x_0^2)}) + (J-1)s/d$ and error
$(1-\varphi)\sqrt{2/n}$, so

$$
G \;=\; \frac{1}{\kappa\,(1-\varphi)^2},
\qquad \varphi = \text{captured mass fraction}.
$$

Validated against the recorded runs: Al 216 atoms $J{=}12$
($\kappa = 1.77$, $\varphi = 0.967$) predicts 444× vs measured 444×;
Al 729 atoms $J{=}12$ ($\rho = 576/1644$) predicts 1.34 vs measured
1.28; $J{=}36$ predicts 5.5 vs measured 5.2.

**Proposition 10 (never-lose floor).** *Always
$G \ge 1/\kappa \ge 1/(\sqrt2 + (J{-}1)/10)$: at $J = 1$ the worst
possible efficiency loss is 29% (band-center probe, zero capture), and
$\kappa = 1$ — hence $G \ge 1$, deflation strictly free — for probes at
$|x_0| > 1/\sqrt2$, where physical LCAO energies sit. Independently of
$G$, at a FIXED run the accuracy itself can never degrade (Proposition
7), and the run reports where it stands (`subspace_capture`): the fade
is observable, never silent.*

**Proposition 11 (a-priori upper bound on the win, partial regime).**
*No rank-$R$ projector of any construction can capture more than the
$R$ largest filter weights:
$\varphi \le \varphi_{\text{oracle}}(R) =
\sum_{\text{top-}R} p_k^2(x_n) / \sum_n p_k^2(x_n)$, computable from
the DOS alone. Hence
$G \le 1/\big(\kappa(1 - \varphi_{\text{oracle}}(J(n{-}w)))^2\big)$ —
an upper bound BEFORE any run, from $(N, \rho(E), \eta, n, J)$.*

**The capture law and the variables.** The random sketch achieves, for
dense windows, the uniform law $\varphi \approx \rho := R/m_{\text{eff}}$
(measured exactly: $0.397 \approx 576/1644$), with the soft shoulder
near $\rho = 1$ ($1.05 \to 0.717$, $1.19 \to 0.967$). Substituting
$m_{\text{eff}} \approx 10\,\eta\,N\rho(E)$ makes every dependence
explicit:

$$
G \approx \frac{1}{\kappa}\,
\Big(1 - \frac{J(n-w)}{10\,\eta\,N\rho(E)}\Big)^{-2} ,
$$

so at fixed knobs the win **fades with system size** toward the
Proposition-10 floor as $\rho \sim 1/N$; grows as filters **sharpen**
($m_{\text{eff}} \propto \eta$ — the win is largest exactly where the
$\chi^2$ floor hurts most); is largest where the **local DOS is small**
(gaps, edges, defects: $m_{\text{eff}}$ small or $N$-independent, the
permanent-win regime, $G$ capped only by the accuracy floors —
leakage, certificates — i.e. $G \sim (2/n)/\epsilon_{\text{floor}}^2
\sim 10^6$–$10^{10}$); and is *held* against growing $N$ by
$J \propto m_{\text{eff}}$ at the comb's $10\times$ discount over the
sample route ($\kappa$ grows as $m_{\text{eff}}/(10(n-w))$ instead of
linearly). Past the point where even that is unattractive
($\rho \ll 0.16/J$-ish), the stack's answer is not more global
capacity but a different mechanism: local capture (section 7), the
$\eta$-ladder, or teeth — which is what route-by-outcome automates.

**The $\eta$-ladder (exactness at density-proportional cost).** The
Gaussian factors as a convolution,
$\eta^2 = \eta'^2 + \eta''^2$, so the target is a small Gauss–Hermite
sum of finer maps,
$\rho_\eta(E) = \sum_l w_l\,\rho_{\eta'}(E + \sqrt2\eta'' t_l)$, all
$L \sim 5$–$9$ probe energies riding one sweep of degree
$\times\,\eta/\eta'$. Each sub-window holds
$m_{\text{eff}}\,\eta'/\eta$ states; refining until that is under
capacity puts every rung in the exact regime. Pure input choice — no
code.

**Proposition 8 (the matvec floor for window probes).** *The seed
budget $n = 2/\epsilon^2$ is Cramér–Rao-optimal given $n$ scalar
projections per point, but a sweep holds more: since
$T_j^2 = \tfrac12(1 + T_{2j})$, each comb tooth is a near-unbiased
resample of the map whose error decorrelates on the filter's core, and
the number of variance-independent probes a sweep of total degree $D$
can yield is the Nyquist count $D\sigma_\theta/\pi$,
$\sigma_\theta = (\eta/h)/\sqrt{1-x_0^2}$ — i.e. one independent probe
of an $\eta$-window costs $\pi h\sqrt{1-x_0^2}/\eta$ matvecs,
regardless of scheme.* (Verified to three digits: predicted 4.51
effective probes, correlation-sum 4.52.) Teeth averaging and deflation
attack different error terms (measured correlation 0.18) and compose to
$3.85\times$ beyond the $\epsilon^2\times\text{sweeps} \approx 2$
frontier; the production composition is recorded in RESULTS.md and not
implemented.

---

## 7. Block-local capture: through the extensive-rank wall

Global sketches fail at extensive $m_{\text{eff}}$; local ones do not,
because the window's **local rank is intensive**:

**Participation law.** Restricted to a fragment $F$ (a set of orbital
rows), the $m_{\text{eff}}$ window states span only
$r_F \approx m_{\text{eff}}\,\dim_F/N$ dimensions (measured: 557 states
→ rank 40/64 at 90/99% weight on a 567-orbital fragment). Per-fragment
capture bases therefore have total capacity $\sum_F r_F$ — unbounded by
the sketch width.

**Quantitative estimate (two regimes and a crossover).** Phase-space
counting sharpens the participation law. The window occupies the
$k$-shell of radius $k_E$, thickness $\Delta k = \eta/\hbar v_E$; a
fragment of linear size $L$ blurs every wavevector by $\sim 1/L$, so

$$ r \;\approx\; \rho(E)\,\dim_F\,\max(\eta,\ E_{\mathrm{Th}}),
\qquad E_{\mathrm{Th}} = \hbar v_E/L
\quad\Longrightarrow\quad
\frac{r}{m_{\text{eff}}} \approx \frac{\dim_F}{N}
\max\!\Big(1, \frac{E_{\mathrm{Th}}}{\eta}\Big). $$

For $\eta > E_{\mathrm{Th}}$ (fragment Thouless energy; $\hbar D/L^2$
diffusive) the ratio is the pure volume fraction — the participation
law above (predicts $r \approx 70$ for the measured 567/4536-orbital,
557-state point; measured 40 at 90% weight, below the full-support
count as an $\varepsilon$-rank should be, the Landau–Widom boundary
correction being only $\sim \log(k_E L)\log(1/\varepsilon)$ modes).
For $\eta < E_{\mathrm{Th}}$ — the dense region under a sharp window —
$r$ FLOORS at the channel count $\sim (k_E L)^{d-1}$, independent of
both $\eta$ and $N$: at fixed sharp $\eta$ the compression
$m_{\text{eff}}/r$ then grows $\propto N$ unboundedly. The crossover
fixes the fragment size: $L^\* \approx \hbar v_E/\eta = \xi$, the
window's coherence length. Larger fragments grow $r$ with volume for
no accuracy; much smaller ones store redundant shared boundary
channels ($\sum_F r_F = \rho N \max(\eta, E_{\mathrm{Th}})$ exceeds
$m_{\text{eff}}$ by $E_{\mathrm{Th}}/\eta$ — the storage price of
locality for over-sharp windows). Homogeneous-band estimate, not a
bound: a flat band ($v_E \to 0$) never enters the channel regime and
its local rank is honestly extensive over the flat-band weight.

**Why locality survives spectral sharpness.** A sharp window is
spectrally local, so its KERNEL
$P(\mathbf r,\mathbf r') = \sum_{\text{win}}\psi_n(\mathbf r)
\psi_n(\mathbf r')$ is long-ranged (Friedel power-law tails in a metal)
— which is why operator-truncation methods (the fragment CV, buffered
images) genuinely fail here. But the map is the DIAGONAL, and the
diagonal at $\mathbf r$ depends only on how the window states *look
near* $\mathbf r$. That local appearance is low-dimensional even at
$m_{\text{eff}} \propto N$: metallic window states are waves with
$|k| \approx k_F$ in all directions, and a fragment of linear size $L$
resolves only angular differences $\gtrsim 1/(k_F L)$ — the thousands
of globally orthogonal states collapse onto $\sim (k_F L)^2$
distinguishable local patterns, a count set by the fragment and the
energy, independent of crystal size (the participation law in physical
terms; QCD's "local coherence" of low Dirac modes). The kernel's
non-locality does not vanish — it moves into the COST, exactly twice:
the images of restricted basis vectors spread over the full kernel
range, so they are computed by dedicated GLOBAL sweeps of the true
operator (long-range physics carried exactly, by matvecs, never by
truncation); and packed fragments talk through the kernel at their
separation, the packing-noise law below. Correctness (Proposition 9)
never touches a locality assumption.

**Proposition 9 (state-space split — exact for any block projector).**
*Let $\Pi = \mathrm{blockdiag}(U_F U_F^T)$ with each $U_F$ orthonormal
on its fragment's rows, acting on the STATES $\mathbf y$, and let
$C_p := \mathbb E[\mathbf y\mathbf y^T]$ ($= p_k^2(M)$ sandwich,
$= p_k^2(B)\,S^{-1}$ solve). Then, identically in $\Pi$,*

$$
(I-\Pi)C_p(I-\Pi) + \big[\Pi C_p + C_p\Pi - \Pi C_p\Pi\big] = C_p ,
$$

*so the map splits into a sampled deflated term
$\mathbb E|\Phi Z(I-\Pi)\mathbf y|^2$ plus a deterministic complement —
with no orthogonality condition in any metric. This is what makes
fragment restriction legal: the projector need only be a symmetric
block projector.* ∎

### 7.1 Objects and the estimator

Fix the notation for one operator form (sandwich; the dictionary of
section 1.1 gives the other). Let $G_{\text{map}}$ denote the grid map
of a state ($\Phi\tilde Z\,\cdot$, or $\Phi\,\cdot$ for solve), and
$C_p = \mathbb E[\mathbf y\mathbf y^T]$ the state covariance
($p_k^2(M)$, or $p_k^2(B)S^{-1}$). The target is
$\rho = \mathrm{diag}\,G_{\text{map}}\, C_p\, G_{\text{map}}^T$.

**Regions** $F$: the distributed sparse operator fixes atom ownership
before the LDOS calculation begins (weighted RIB followed by the
ParMETIS refinement).  Local capture never changes that partition.  Each
rank instead subdivides only its owned atoms by a serial, orbital-weighted
recursive inertial bisection.  All orbital rows of an atom stay together,
and every owned row belongs to exactly one region.  The target size is
$\max(256,6r)$ scalar orbital rows: as large as the local algebra affords,
because fewer regions mean fewer co-packed in one image column.  The
resulting $P_F$ is therefore applied entirely to local rows and introduces
no projector communication.  These regions need not coincide with the
independent ownership of real-space grid boxes.  Their Cartesian centers
are converted to global fractional coordinates only for the packing color
below; that coloring does not define the regions.

**Capture bases** $U_F \in \mathbb R^{\dim_F\times r}$, orthonormal
columns: the top-$r$ eigenvectors of the fragment-restricted sketch
Gram, pooled over the energy scan,

$$
G_F = \sum_k \frac{1}{\mathrm{tr}_k}\, Y_F^{(k)T}\,Y_F^{(k)} ,
$$

where $Y_F^{(k)}$ is the FIRST-half sample blocks' stored families
(all comb teeth) restricted to $F$'s rows. The $1/\mathrm{tr}_k$
**trace balance** gives every window an equal vote (a dense window
otherwise crowds sparse windows out of the $r$ slots), and energies at
the sweep's noise floor are excluded by the section-3 absolute scale
($\mathrm{tr}_k < 10^{-10}(\sum|a_m|)^2 q$) — their states are
roundoff and must not enter a basis.

**The block projector** $\Pi = \mathrm{blockdiag}(U_FU_F^T)$, acting
on states. Proposition 9 splits the target for ANY such $\Pi$:

$$
\rho \;=\;
\underbrace{\mathrm{diag}\,G_{\text{map}}\big[\Pi C_p + C_p\Pi -
\Pi C_p\Pi\big]G_{\text{map}}^T}_{\text{deterministic complement }D}
\;+\;
\underbrace{\mathbb E\,\big|G_{\text{map}}(I-\Pi)\mathbf y\big|^2}_{\text{sampled residual}} .
$$

The residual term is trivial to compute: per fragment,
$\mathbf y|_F \mathrel{-}= U_F(U_F^T\mathbf y|_F)$, then the ordinary
grid square — contracted over the SECOND-half sample blocks only
(split-sample cross-fitting: $\Pi$ is independent of those seeds, so
the term is exactly unbiased for its expectation, Proposition 6 in
split form). Everything nontrivial is in evaluating $D$.

### 7.2 The deterministic complement: images and polarization

$D$ contains $C_p$ applied to fragment-restricted vectors, and a
restricted vector's filtered image is NOT spanned by the stored
families (restriction breaks the spectral structure — this cost is
information-theoretically forced, section 14's wall in its honest
form). So the images are computed by **dedicated sweeps**: with basis
columns assembled into $\hat X$ (next subsection),

$$
W_k \;=\; C_p\,\hat X
$$

rides one recursion carrying every energy's squared table (one image
sweep per RUN, not per energy); in the solve form the sweep is seeded
with $S^{-1}\hat X$ so the images carry the actual covariance.

$D$ is then read off grid products, never matrices. Define
$P = 2W - \Pi W$ (each fragment deflates its own row-block of EVERY
image column — the FULL block projector: the cross-fragment blocks of
$\Pi C_p \Pi$ vanish on the orbital diagonal, but the collocation
smears across fragment boundaries through orbital tails and
$\tilde Z$'s range, and omitting them measured a 21%
integral-preserving shape error). Using $W^T = \hat X^T C_p$
($C_p$ symmetric) and $\mathbb E_\sigma[\hat X\hat X^T] = \Pi$
(the packing signs, below):

$$
\mathrm{diag}\,G_{\text{map}}\,\hat X P^T G_{\text{map}}^T
= \mathrm{diag}\,G_{\text{map}}\big[2\Pi C_p - \Pi C_p\Pi\big]
G_{\text{map}}^T
= D
$$

(the symmetrized diagonal: $\mathrm{diag}\,M = \mathrm{diag}\,M^T$
turns $2\Pi C_p$ into $\Pi C_p + C_p\Pi$). The cross-diagonal of two
different multivectors is evaluated by **polarization** — pointwise,
per column, no $P\times N$ object:

$$
D(\mathbf r_p) = \tfrac14\Big(
\big|G_{\text{map}}(\hat X + P)\big|^2_p -
\big|G_{\text{map}}(\hat X - P)\big|^2_p \Big)
\quad\text{summed over columns} ,
$$

two extra passes of the standard grid pipeline.

### 7.3 Packing: the cost dial and its noise

Sweeping one column per basis vector per fragment ("unpacked") is
exact-class but EXTENSIVE: $r\cdot n_{\text{frag}} \propto N$ columns.
The scalable mode packs fragments into $G$ groups by a global spatial
coloring of their centers (nearby regions preferentially receive different
colors), one shared column per $(g, j)$:

$$
\hat X_{:,(g,j)} = \sum_{F\in g} \sigma_{F,j}\, U_F\,\mathbf e_j,
\qquad \sigma_{F,j} = \pm1 \;\;\text{counter-hashed.}
$$

Own-fragment products see $\sigma^2 = 1$ and are untouched;
cross-fragment crosstalk — the kernel evaluated at the packing
distance — is made ZERO-MEAN by the signs, converting bias into noise

$$
\epsilon_{\text{pack}} \sim
\text{kernel}(D_{\min})\,\sqrt{\text{coordination}}\,/\sqrt r ,
$$

the packing-noise law. Reaching a crosstalk target $\tau$ costs
columns $\propto \tau^{-3/2}$, intensive in $N$, with the cost
invariant $\text{columns} \approx 2(m_{\text{eff}}/N)\,
N_{\text{orb}}(D\text{-ball})$. Sharp metallic kernels (Friedel
power law) set the practical operating point near $\sim$1% at
$\sim$10× the baseline sweeps; small boxes cannot pack at all (the
partners are adjacent). The recorded buffered-local-image-sweep lever
removes this noise entirely by making each column's sweep subdomain
private.

### 7.4 Algorithm and routing

```
setup, once per run (first-half sample blocks):
1  regions <- weighted RIB subdivisions of each rank's fixed owned atoms,
              target max(256,6r) orbital rows; centers only set packing color
2  G_F = sum_k Y_F^T Y_F / tr_k   (all comb families; noise-floor
                                    energies excluded)
3  U_F = top-r eigenvectors of G_F        (rank floor as section 3)
4  X̂  = lattice-colored, sign-packed U columns      (G*r columns)
5  ONE image sweep: W_k = C_p X̂ for every energy
                                    (solve form: seed = S^{-1} X̂)
per energy, only where global capture < 0.9:
6  P  = 2 W_k - Pi W_k     (every fragment deflates every column)
7  D  = 1/4 [ |G(X̂+P)|^2 - |G(X̂-P)|^2 ]  summed over columns
8  residual: second-half blocks, y|_F -= U_F U_F^T y|_F per fragment,
   grid-square, average;  estimate = D + residual mean
```

**Route by outcome** (step "only where"): global deflation runs first
(it is ~free) and its measured `subspace_capture` decides; local
capture replaces it only below 0.9. Rank heuristics on the sketch Gram
are fragile (comb collinearity thins the spectrum) and were measured
to misroute. Insulator/fixed-$m_{\text{eff}}$ windows never route
here; local capture is specifically the metallic
$m_{\text{eff}}\propto N$ tool, and its cost (steps 5–7) is paid once
per run, not per energy.

**Error model.** Three intensive terms: the residual sampling
$(1-\varphi_{\text{loc}})\sqrt{2/(n/2)}$ with $\varphi_{\text{loc}}$
set by $r$ against the fragment's local rank (measured fall:
16.97% $\to$ 4.91/0.71/0.08% at $r = 40/64/96$); the packing noise of
7.3; and the captured part, exact (global sweeps, no truncation).
None depends on $N$ — the flatness is measured (0.20–0.25% at 216
atoms, 0.277% at 729 unpacked, while global deflation faded 0.61%
$\to$ 11.3% over the same range).

---

## 8. Composition with the fragment control variate

The fragment CV (`fragment_buffer`; machinery in `rs_fragment.hpp` and
the density design doc) is the third, complementary mechanism: a
deterministic *local* surrogate $T_0 \approx \mathcal T_k$ from dense
fragment diagonalization, with

$$
\hat\rho = n_0 + \mathbb E\big[|\Phi\,\mathcal T\boldsymbol\xi|^2 -
|\Phi\, T_0\boldsymbol\xi|^2\big],
\qquad n_0 = \text{exact map of } T_0,
$$

unbiased for ANY $T_0$ (white seeds), variance governed by the residual
$\mathcal T - T_0$ — exponentially small in the fragment buffer. Its
gain grows with $\eta$ (broad filters are local); deflation's with
sharpness (sharp windows are low-rank). Under deflation the fragment
control is regressed against the deflated remainder (per-point-centered
covariance, $[0,1]$ clamp). The **gauge-matching requirement**: $T_0$
must compress the *actual* chain — the isqrtm $\tilde Z$, not an ideal
local $S^{-1/2}$; maps are gauge-invariant but per-sample states are
not, and a surrogate matching only in law is off by a rotation that
destroys the per-sample cancellation (measured floor $\sim10^{-3}$
however large the buffer). The density task uses fragments and energy
windows (its "window" is the whole occupied manifold — extensive rank
always — so `subspace_cv` does not apply to it).

---

## 9. The solve form in detail

Proposition 1 fixes the law; this section is the finite-$\tilde Z$
algebra and the implementation dictionary (all landed in
rs_ldos.hpp / rs_fragment.hpp / metric_sampler.hpp).

### 9.1 The S-metric split (global deflation)

$B$ is self-adjoint in $\langle u,v\rangle_S$, and the colored seed is
**white in the $S$ metric**:
$\mathbb E[\langle u,\mathbf c_0\rangle_S\langle v,\mathbf c_0\rangle_S]
= \langle u, v\rangle_S$. With the $S$-Gram $G = Y^T S Y$, whitened
columns $Q$ ($Q^TSQ = I$), and the $S$-orthogonal projector
$\Pi_S = QQ^TS$:

* cross term: $\mathbb E[\Pi_S\mathbf c_0((I-\Pi_S)\mathbf c_0)^T]
  = \Pi_S S^{-1}(I - \Pi_S^T) = QQ^T - QQ^T = 0$;
* captured part: $\mathbb E[\Pi_S\mathbf c_0\mathbf c_0^T\Pi_S^T]
  = QQ^T = \sum_j q_jq_j^T$, so it is again grid squares of the exact
  images $U = Z_{\text{store}}B$ (Proposition 4 is metric-free);
* deflation coefficients: $C = Q^T S\,\mathbf c_0$.

Propositions 5–7 then hold verbatim (conjugate by $Z$: $Z^{-1}\Pi_S Z$
is Euclidean-orthogonal on the white seeds $O\boldsymbol\xi$).
Implementation deltas: the gate admits both forms; `fill_sy` applies
$S$ to the stored family once per energy (an SpMM of $(2J-1)n$ columns
— sub-percent against the sweep); the contraction rows are built from
$SY$; the colored seeds are kept in `c0_store`. The empty-window floor
is unchanged — by gauge conjugation the $S$-Gram spectrum has the same
scale as the sandwich Gram.

### 9.2 Local capture

Proposition 9 needs no metric, so the Euclidean fragment SVDs, packing,
and contraction all stand. The single change: the image sweep's seed
becomes $S^{-1}\hat X$ (block CG), so $W = C_p\hat X$ with the solve
form's $C_p = p^2(B)S^{-1}$; the polarization pairing
$\hat X$ vs. $P = 2W - \Pi W$ then reproduces the Proposition-9
complement because $\hat X\hat X^T \approx \Pi$ and
$p^2(B)S^{-1}$ is symmetric. The grid stage consumes the states
directly (no $\tilde Z$ apply) in every solve-form path.

### 9.3 The fragment surrogate chain

At the white-seed level the solve estimator applies
$\mathcal T = p(B)\,\tilde Z\,q(W)$ to raw $\boldsymbol\xi$, so the CV
identity of section 8 holds with $T_0$ truncating **that whole chain**.
Using the exact similarity
$p(B)\tilde Z = \tilde Z\,p(W^{-1}\hat M)$ with
$\hat M = \tilde Z^T H\tilde Z$, $W = \tilde Z^T S\tilde Z$:

$$
\mathcal T = \tilde Z\; p(W^{-1}\hat M)\; q(W),
\qquad
T_0 = \tilde Z[\text{core},F]\;p(W_{FF}^{-1}\hat M_{FF})\;q(W_{FF}),
$$

both middle factors compressions by the same full-height $\tilde Z$
columns the sandwich CV already fetches
($\hat M_{FF} = Z_b^THZ_b = \hat M[F,F]$,
$W_{FF} = Z_b^TSZ_b = W[F,F]$). With generalized pencil eigenpairs
$\hat M_{FF}V = W_{FF}V\,\mathrm{diag}(\epsilon)$, $V^TW_{FF}V = I$
(computed through $W_{FF}$'s own eigenbasis $V_w, w$, which the $q$
factor needs anyway; directions under the rank floor drop out of $K$
analytically):

$$
T_0 = \big(Z_{\text{core}}V\big)\;\mathrm{diag}\big(p(\epsilon_j)\big)\;K,
\qquad
K = V^TW\,q(W)
= \tilde V^T\,\mathrm{diag}\big(\sqrt{w}\;q(w)\big)\,V_{w,\text{kept}}^T ,
$$

with $q$ evaluated as the ACTUAL sampler polynomial on its own fitted
interval — the gauge-matching requirement of section 8 extended one
factor. Purified $\tilde Z$ gives $W = I$, $q = 1$, and the chain
collapses to the sandwich construction. Seeds, counters, the $n_0$
streaming, and the regression are untouched (white-seed identity).
Buffer sizing must cover range(filter) $+$ range($\tilde Z q$); for the
unpurifiable-$\tilde Z$ case that motivates the solve form, $q(W)$'s
extra range (overlap range × its degree) is the price, absorbed by the
existing buffer knob.

### 9.4 When to use which form

With a purifiable $\tilde Z$ the forms are gauge copies and the
sandwich is strictly cheaper (3 SpMMs vs. a CG per matvec) — there is
no accuracy reason to run solve. The solve form is for
ill-conditioned/unpurifiable $S$: it answers the true pencil for any
invertible $\tilde Z$, with the sampler residual and CG tolerance as
its only bias terms, and the entire variance-reduction stack available
at the same statistical performance.

---

## 10. Cost, memory, scalability

Per added mechanism, at $n$ samples, $J$ families, $n_E$ energies,
$N/P$ local rows:

| item | cost | scaling |
|---|---|---|
| deflation matvecs | none beyond the degree law of section 4: $\times\max(1,\sqrt{2(1-x_0^2)})$ for the squared tables ($\le\sqrt2$, usually 1 — physical probes sit near the spectrum edge), $+(J{-}1)s$ for the comb | — |
| stored states | $(2J{-}1)\,n_E\,n\,(N/P)$ doubles ($+\,n$ for solve `c0_store`) | $O(N/P)$ |
| Grams + whitening | $O((Jn)^2N/P)$ GEMM, $(Jn)^2$ Allreduce, $(Jn)^3$ replicated eigensolves | $O(N/P)$, constant collectives |
| deflation + captured grid passes | thin GEMMs and one width-$r$/-$w$ grid pass per (energy, block) | $O(N/P)$ |
| solve-form deltas | one $S$ SpMM per energy per family; one CG per image column | noise vs. the CG the form pays per matvec |
| local-capture image sweeps | $G\cdot r$ columns × full degree — the dominant cost when on (~88% of a full-stack run) | intensive in $N$ (packing invariant) |

Memory law per rank: states dominate,
$n_E(2J{-}1)n(N/P)\cdot 8\,\mathrm B$; at $N = 2\times10^7$, $P = 64$,
$n = 64$, $J = 4$, $n_E = 3$: $\approx 4$ GB/rank. No $O(N)$ object is
replicated in the core path and no collective grows with $N$ (the two
$O(N)$-replicated tables live in the optional fragment-CV machinery,
with distributed replacements sketched). Recorded levers, in value
order: fp32 image sweeps (~2× on the dominant cost; needs a float
operator path), LOCAL image sweeps on buffered subdomains
($N$-independent column cost, kills packing noise — the million-atom
design), fp32 stores (memory /2).

---

## 11. Validation summary

Chain fixture = the 12-atom op2c carbon chain with synthetic $H$
(test_rs_ldos / test_rs_density, exact dense reference); production =
64 samples, $J = 4$, vs. k-space diagonalization references, masked
pointwise rel-std. Full history: RESULTS.md.

| claim | measurement |
|---|---|
| $\chi^2$ floor $\sqrt{2/n}$ | 17.0% measured vs 17.7% predicted ($n=64$); $N$-independent (216 → 729 atoms: 17.1 → 17.0%) |
| exact regime | diamond sc444 all energies $10^{-9}..5\times10^{-7}$ rel (plain 7–15%); silicon ×3 energies 0.0001% |
| comb capacity | Al $E_F$ 485 states: consecutive $J{=}4$ rank-pinned; comb $J{=}4$ 5.0%, $J{=}12$ 0.61% at predicted $1{+}J/10$ cost; diamond 1024-atom 552-state column 0.019% at $J{=}12$ |
| capacity fade + restore | 729 atoms: capture 0.397 $\approx 576/1644$ as predicted; $J{=}36$ → 0.717 |
| Nyquist bound | predicted 4.51 independent probes vs measured 4.52; teeth+deflation compose 3.85× past $\epsilon^2$·sweeps ≈ 2 |
| $\eta$-ladder | sc888 dense column 0.52% → 0.0095% at one rung (1.9× sweep) |
| local capture | Al666 $E_F$ 0.20–0.25% vs 17.1% plain; disorder ($W{=}3$ eV Anderson) 0.086%; sandbox intensivity 210× through the global wall |
| packing law | diamond $G{=}2$ 42.7%, Al999 $G{=}8$ 5.0%, unpacked 0.28% — kernel(D)·√coord/√r as predicted |
| empty windows | absolute floor: deflated map $8.7\times10^{-12}$ abs where truth is 0 (was $O(1)$ garbage without it) |
| fragment CV | full-fragment limit exact to $2.3\times10^{-14}$ (sandwich) |
| **solve form, fixture** | deflation 16.3/25.8% → 0.00% (capture 1.000); local capture 25.8% → 0.00%; full-fragment CV with crude $\tilde Z$ (graph seed, real $q$): $2.6\times10^{-12}$ rel; multi-fragment CV 36× |
| **solve form, production** | silicon 128-atom full stack 0.0000/0.0000/0.0001% masked dev; aluminum 216-atom dense metal: Fermi window 0.181% (sandwich 0.20–0.25%), empty window $1.2\times10^{-10}$ abs, band edge 0.0067%; rank-count invariance np1 vs np2 at $10^{-8}..3\times10^{-6}$ rel |

Run diagnostics: `subspace_rank` ($r$ per energy; saturated at capacity
means more samples/comb would capture more), `subspace_capture`
(deterministic fraction of the integrated map; the route-by-outcome
signal), `correction_rms` (rms of the sampled remainder),
`sampler_residual` (solve-form covariance certificate).

**Measured dead ends** (details and reasons in RESULTS.md and the
project memory; do not revisit without new structure):
$\eta$-telescoping MLMC, local-PCA covariance denoising, probe-pair
U-statistics, zero-sweep buffered images, Gram-rank routing,
smeared (Gauss–Hermite) fragment spectral weights.

---

## 12. Relation to published methods

The global deflation is the exact-sketch-plus-stochastic-residual
principle of Hutch++ [Meyer, Musco, Musco, Woodruff 2021] with the
exchangeable cross-fitting of XTrace/XDiag [Epperly, Tropp, Webber,
SIMAX 2024], transplanted from traces and matrix diagonals to
real-space maps, and made exact-in-the-captured-part by the
squared-filter identity — the Krylov-aware idea [Chen, Hallman 2023]
of reusing information the function application already generated, in
Chebyshev rather than Lanczos form. Block-local capture is the QCD
local-coherence/low-mode-averaging transfer [arXiv:2412.06347]: local
restriction converts extensive capture into intensive per-fragment
work. The fragment surrogate is the submatrix method [Lass et al.
2018/2020; NOLSM 2022] repurposed as a control variate (truncation
error removed in expectation rather than accepted), with the gauge
lesson that distinguishes a CV from a standalone approximation.
Stochastic-DFT levers (embedded fragments, energy windows ew-sDFT,
MLMC, probing) attack the density's variance through locality or
occupation splitting; none addresses the sharp-$\eta$ pointwise floor
this stack removes.

---

## 13. Implementation notes

* Sweep accumulation is **blocked moments**: recursion iterates stage
  into a ring buffer (depth 8) and fold into all filter tables with one
  GEMM per flush — an order of magnitude less accumulator traffic than
  per-step axpys at deep combs/multi-energy scans.
* `DistMultiVector` rows have stride `ld` padded to 64 bytes: any
  contiguous copy across rows is wrong whenever $8\cdot$blk is not a
  multiple of 64 — always copy per row (`row_data`). Found as a 17%
  complete-basis error at blk = 20; an env-gated redundant sweep
  self-check (`RSATB_SWEEP_CHECK`) is kept permanently.
* One `SymEig` (dsyevd wrapper) serves every Gram eigensolve; one
  `grid_square` serves every metric-apply → halo → box-square chain,
  form-aware (sandwich applies $\tilde Z$, solve does not).
* The base sample signs are counter-hashed from global orbital rows, so the
  plain and global-projector estimates do not depend on the MPI partition.
  Local-capture regions deliberately inherit the already-fixed rank-local
  atom domains; their packing signs are keyed by rank and local RIB leaf.
  Changing the atom partition therefore changes this particular unbiased
  realization, but never requires atoms or projector coefficients to move
  between ranks.
