# Overlap deflation for ill-conditioned bases: a localized basis of the null band

Status: **future plan, not implemented** (decision of 2026-09-15). The
design below was validated numerically on dense proofs of concept (a
343-atom Al cell and a 729-atom fill measurement; k-mesh checks on the
primitive cells) but has not been studied in the sparse C++ path, where the
first open question is whether the truncated re-projected iteration
reproduces the measured fill of the exact deflated factor. Until that study
exists, production runs on ill-conditioned bases use the current code as is
and pay the full spectral span (aluminium: ~4x the moment counts of the
silicon and diamond cells, see `examples/publication/transport/`). Scripts
and results of the proofs of concept are outside the repository
(`pub_exp/overlap_deflation/`).

Rich numerical-atomic-orbital bases are near-linearly dependent. The
redundant combinations produce spurious eigenvalues hundreds of eV above
the physical band and make $S^{-1/2}$ dense, so the Chebyshev machinery
pays for a span it does not need and for $O(N^2)$ applies. The fix below
removes exactly the offending subspace — the few smallest eigenvalues of
$S$ — by representing it with a thin matrix of exponentially localized
functions, 4 per atom, obtained by Chebyshev-filtered subspace iteration
from a free seed. It is exact to $10^{-8}$ eV and $O(N)$ to build. What it
buys is the spectrum: the span returns to the physical window, 3.7× fewer
Chebyshev moments, and a well-conditioned overlap whose inverse square
root the existing iteration handles without diverging. What it does
**not** buy is a sparse $S^{-1/2}$: at $\kappa = 48$ that factor is as
dense as diamond's at practical cell sizes, so the cost per Chebyshev
step is unchanged. The net gain is ~3× on the aluminium transport, and
the per-step cost remains the general dense-$Z$ problem, which is a
separate design.

---

## 1. The problem

The electronic structure is the generalized eigenproblem

$$
H\,c = \varepsilon\,S\,c ,
$$

with $H$, $S$ block-sparse over atoms. For the Al `4s4p1d` basis
(21 orbitals/atom) the spectrum of $S$ has, across the whole Brillouin zone,
a **gap**: a null band of exactly 4 eigenvalues per atom in
$[2\times10^{-4},\,0.083]$, nothing until $0.181$, then the rest up to
$8.9$. A near-null combination has

$$
\varepsilon = \frac{\langle c|H|c\rangle}{\langle c|S|c\rangle}
$$

with a vanishing denominator and appears at $+300\ldots+550$ eV while the
physical band ends at $+80$ eV. Two costs follow.

- **Span.** A Chebyshev expansion with $M$ moments resolves
  $\Delta E \approx \pi\,(E_{\max}-E_{\min})/M$; 638 eV instead of ~180 eV
  costs 3.5× the moments.
- **Orthogonalization.** The elements of $Z = S^{-1/2}$ decay with graph
  distance $d$ as $r^{d}$, $r = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$; at
  $\kappa(S) = 3\times10^{4}$, $r = 0.989$: $Z$ is dense at every practical
  size and every apply is $O(N^2)$.

Both are the null band's doing; the physics near $E_F$ is correct
throughout.

## 2. The right object is the small subspace, not its complement

Exact removal means the spectral projector $P$ onto $\{\lambda_S > \mu\}$
with $\mu$ in the gap. Computed as a square matrix (matrix-sign
purification, or canonical orthogonalization), $P$ is $17/21$ of the
space and inherits the gap's localization length — 2–4 hops of 86 atoms
each — so it is effectively dense on cells up to $12^3$ and costs an hour
of setup. A fixed per-atom map fails in the opposite way: the on-site
block of the null projector has spectrum
$0.82, 0.72, 0.72, 0.72 \mid 0.38, 0.18, 0.18, 0.18 \mid \le 0.03$, so the
null band's local directions are 18–28 % physical and dropping them costs
7 eV.

What *is* small is the null band itself: $4N_{\rm atom}$ states out of
$21N_{\rm atom}$, and an isolated band, so it admits exponentially
localized Wannier-like functions. Write

$$
Q = V V^{\dagger},\qquad P = I - V V^{\dagger},\qquad
V \in \mathbb{R}^{N \times 4N_{\rm atom}},\quad V^{\dagger}V = I ,
$$

with each column of $V$ exponentially localized. Then $P$ is never a
matrix: $P v = v - V G^{-1} V^{\dagger} v$ with the sparse Gram matrix
$G = V^{\dagger}V$ (the identity if $V$ is orthonormal), thin products
only. The localization length is that of the band's projector — measured
20× per overlap-graph hop, an e-fold of 2.5 Å — but a hop is 7.4 Å and
86 atoms, so a $10^{-3}$ tail still spans ~17 Å, about 1250 atoms per
function, and $\mathrm{nnz}(V) \approx 3\,\mathrm{nnz}(S)$. That extent is
a property of the basis, not of the method; it is what every cheaper
object (per-atom, one-hop, scalar shifts) approximates at first order,
with eV-level errors measured.

## 3. The method

### 3.1 Locate the gap and count the band

Compute the density of states of $S$ itself with the stochastic Chebyshev
trace already used for $H$,

$$
\rho_S(\lambda) = \frac{1}{N}\,\mathrm{Tr}\,\delta(\lambda - S),
$$

Rademacher probes, Jackson kernel, mapping
$[-0.05\lambda_{\max},\,\lambda_{\max}]$ so the null band is not at the
Chebyshev edge singularity. The gap is the widest stretch on which
$\rho_S < 0.1\,\bar\rho$ ($\bar\rho = N_{\rm orb}/(N_{\rm atom}\lambda_{\max})$),
wider than the kernel resolution $\pi\lambda_{\max}/M$, above the spectral
onset; $\mu$ is its geometric centre and
$n_0 = N_{\rm atom}\int_{-\infty}^{\mu}\rho_S$ the number of null states,
rounded. Measured: gap $[0.082, 0.178]$, $\mu = 0.121$, $3.97$ states/atom
(exact 4), for $2000\times12$ matvecs. No qualifying stretch means no gap:
the run says so and does not deflate.

### 3.2 A localized basis of the null band

Take the Chebyshev polynomial $T_m$ mapped so that $[\mu, \lambda_{\max}]$
goes to $[-1, 1]$,

$$
\tilde S = \frac{2S - (\lambda_{\max}+\mu)\,I}{\lambda_{\max}-\mu},\qquad
F_m = T_m(\tilde S).
$$

On the physical band $|F_m| \le 1$; on the null band, at
$x = \tilde\lambda > 1$,

$$
F_m = \cosh\!\big(m\,\mathrm{arccosh}\,x\big) \approx
\tfrac12\exp\!\Big(m\sqrt{\tfrac{4(\mu-\lambda)}{\lambda_{\max}-\mu}}\Big),
$$

which is largest for the smallest $\lambda$ and smallest at the gap edge:
for $m = 50$, $\approx 300$ at $\lambda = 0.083$ and $10^{4}$–$10^{5}$
deep in the band. Subspace iteration with this filter,

$$
V \leftarrow \mathrm{orth}\big(F_m\,V\big),
$$

converges to the null band at that rate per pass. The seed is free: the
$n_0/N_{\rm atom}$ highest-zeta orbitals on each atom as unit vectors
(here $s_3, p_{3x}, p_{3y}, p_{3z}$; in general, random on-site
combinations, which cost at most one extra pass). $\mathrm{orth}$ is a
Löwdin orthonormalization, $V \leftarrow V\,(V^{\dagger}V)^{-1/2}$, of the
sparse $4N_{\rm atom}\times4N_{\rm atom}$ Gram matrix, well conditioned
after the first pass; it is needed during the iteration to keep the
columns independent.

Once converged, do **not** keep the orthonormal columns: measured, they
decay only ~3× per hop (0.48, 0.21, 0.06 at 0, 1, 2 hops), because
$(V^{\dagger}V)^{-1/2}$ spreads each function over its neighbours. Re-express
the subspace in the non-orthogonal **projected basis**
$v_j = Q\,e_j = V(V^{\dagger}e_j)$, the converged projector applied to the
on-site seeds: these inherit the projector's own decay (20× per hop,
measured directly on its blocks), and $P$ is applied with the Gram
inverse, $Pv = v - VG^{-1}V^{\dagger}v$, $G_{jk} = e_j^{\dagger}Qe_k$ sparse
and well conditioned (its diagonal is the seeds' null-band weight,
0.5–0.8). Truncate $V$ at a tail $\epsilon$; the deflation error is second
order, $\sim\epsilon^{2}\Delta$ with $\Delta \approx 400$ eV.

Measured (7³ cell, 343 atoms): residual of $V$ outside the exact null band
$1.9\times10^{-3}$, $4.8\times10^{-7}$, $5.6\times10^{-8}$ after 1, 2, 3
passes from the zero-cost seed, and the same rates on 5³ and 6³ (the rate
is set by the gap, not by $N$); $1.2\times10^{-5}$, $5.7\times10^{-8}$
after 1, 2 passes from a cluster-projected seed. Each pass is $m = 50$
sparse products with $S$ on the thin block. Truncating the converged
basis at $10^{-3}$, $10^{-4}$, $10^{-5}$ and re-orthonormalizing gives
$1.5\times10^{-2}$, $2.8\times10^{-4}$, $1.0\times10^{-5}$ eV max near
$E_F$ — the second-order law. These periodic cells (≤ 20 Å) are smaller
than a function's $10^{-3}$ support, so the fill at scale follows from the
decay rate, not from the cell.

The degree follows from the gap: $m \approx 3\sqrt{(\lambda_{\max}-\mu)/(\mu-\lambda_{\rm edge})}$
gives ~300× per pass at the edge; two passes reach $10^{-6}$, three
$10^{-8}$.

### 3.3 The deflated pencil

With $E_{\rm park}$ = physical band top + 10 eV, define, never assembled,

$$
\tilde H = P H P + E_{\rm park}\,Q,\qquad \tilde S = P S P + Q,
$$

applied as $\tilde H v = P\,(H\,(P v)) + E_{\rm park}\,V(V^{\dagger}v)$ and
likewise $\tilde S$. On $\mathrm{range}(P)$ this is the original problem
restricted to the well-conditioned subspace — canonical orthogonalization,
exact to the accuracy of $V$ — and on $\mathrm{range}(Q)$ it is
$E_{\rm park}\,I$: every null mode sits at $E_{\rm park}$, above $E_F$,
unoccupied, outside every window. The Chebyshev expansion runs on the
operator $\tilde S^{-1}\tilde H$: each step applies $\tilde H$ once and
solves $\tilde S x = y$ by preconditioned conjugate gradient. With the null
band removed,

$$
\kappa(\tilde S) = \lambda_{\max}/\mu \approx 48 ,
$$

and a 1-hop cluster inverse-square-root seed with modes below $\mu$
dropped preconditions it to ~3 iterations. The velocity and position
operators enter through the same $P$ applies, so transport needs no
separate treatment. DOS and PDOS normalize over all $N$ states and show a
delta of $n_0$ states at $E_{\rm park}$, which the run prints.

Measured with the converged $V$: eigenvalues within 5 eV of $E_F$ agree
with exact canonical orthogonalization to **$1.7\times10^{-8}$ eV max,
$3\times10^{-9}$ median**; the parked states have spread $2\times10^{-12}$;
$\kappa(\tilde S) = 48.3$.

## 4. Cost

**Setup.** Locating the gap: $M\times R \approx 2000\times12$ matvecs with
$S$. The null basis: 2–3 passes of $m \approx 50$ sparse products on the
thin block plus sparse Löwdin steps, linear in $N$, of order a minute at
$12^3$. Then the deflated factor. Today's `isqrtm` factor is not $S^{-1/2}$: it
is an inverse factor, $Z^{\dagger}SZ = I$, non-symmetric and not commuting
with $S$ (the seed fixes an arbitrary unitary). Write it as
$Z = S^{-1/2}W$ with $W = S^{1/2}Z$ unitary — the only thing
$Z^{\dagger}SZ = I$ says. $V$ spans the null *eigenspace* of $S$, so
$P = I - VG^{-1}V^{\dagger}$ is a spectral projector of $S$ and commutes
with $S^{\pm1/2}$; that is the only commutation used. Take the one-sided
projection

$$
\tilde Z = P\,Z .
$$

Then

$$
\tilde Z^{\dagger} H \tilde Z = Z^{\dagger}\,PHP\,Z
= W^{\dagger}\Big[P\,\big(S^{-1/2}HS^{-1/2}\big)\,P\Big]W ,
$$

the full orthogonalized Hamiltonian compressed to the good subspace —
canonical orthogonalization with the threshold in the gap — under a
unitary similarity, so its spectrum is exactly the deflated one whatever
$W$ is; the null band becomes a zero block on $W^{\dagger}\mathrm{range}(Q)$
(for such $v$, $Zv \in \mathrm{range}(Q)$ and $PZv = 0$). The metric on the
retained subspace is exact, $\tilde Z^{\dagger}S\tilde Z = W^{\dagger}PW =: P'$,
a projector, so there is no first-order metric error — only today's
`isqrtm` tolerance and $V$'s residual. Verified with a random unitary $W$
standing in for the seed (`poc_pz_nonsym.py`, 6³ cell): $P'$ a projector
to $2\times10^{-14}$ with trace equal to the good count, a zero block of
exactly the null count, eigenvalues within $5\times10^{-13}$ eV of exact
canonical orthogonalization near $E_F$, undeflated span $[-92, 493]$ →
$[-92, 81]$.

Cost: today's `isqrtm` unchanged, then $PZ = Z - VG^{-1}(V^{\dagger}Z)$,
one thin projection of the converged factor, about $2\,\mathrm{nnz}(V)\,N$
flops (~10 s at $12^3$). No new iteration. Probes are projected once by
$P'v = v - (Z^{\dagger}SV)\,G^{-1}\,(V^{\dagger}(Zv))$, with $Z^{\dagger}SV$
precomputed ($4N_{\rm atom}$ columns): one $Z$ apply plus thin products
per probe. If on some basis the raw iteration cannot converge, the
fallback is the re-projected iteration $Z \leftarrow P[Z(3I - Z^{\dagger}SZ)/2]P$
from the thresholded subgraph seed (measured: three steps to
$3.7\times10^{-6}$; the unprojected iteration diverges), judged by the
applied residual on a probe as `isqrtm` does.

**Per Chebyshev step.** The cost is $2\,\mathrm{fill}(\tilde Z) + 1$ in
$S$-matvec units, so the question is whether the deflated factor is any
denser than today's. Measured on the exact factors of the 9³ cell
(729 atoms, 3 hops, dense $= 9.5\,\mathrm{nnz}(S)$;
`poc_ztilde_decay.py`), block magnitude max / median per hop and fill at
absolute entry thresholds:

| | on-site | hop 1 | hop 2 | hop 3 | fill at $10^{-3}$ / $10^{-4}$ / $10^{-5}$ |
|---|---|---|---|---|---|
| raw $S^{-1/2}$, $\kappa = 28\,900$ | 15 | 2.3 / 0.33 | 0.34 / 0.055 | 0.060 / 0.015 | 4.6× / 7.6× / 8.8× |
| deflated $(PSP)^{-1/2}$, $\kappa = 49$ | 1.3 | 0.14 / 0.027 | 0.0087 / 0.0013 | 0.0016 / 0.00017 | 1.1× / 4.4× / 8.1× |

The deflated factor is the *more* local one: ~10× decay per hop against
~5×, a tail at three hops ~100× smaller, and at a $10^{-3}$ threshold as
sparse as $S$ itself. The raw factor's on-site magnitude of 15 is the
$\lambda^{-1/2}\sim 70$ null-band content — the same part that produces the
+546 eV states — and its slow tail keeps filling as the cell grows, while
the deflated factor is near its asymptote at three hops. Caveats: exact
factors, not the truncated Newton–Schulz iterates; absolute thresholds
(a relative one favours the deflated factor further); the truncated
re-projected iteration must be shown to reproduce this fill, which is the
first check of the sparse implementation. Within that, at a $10^{-4}$
threshold the step goes from ~16 to ~10 units, at $10^{-3}$ from ~10 to
~3, improving with $N$. The sandwich is $\tilde Z^{\dagger}H\tilde Z$ and
nothing else — $S$ never appears, $P$ is folded into the stored factor,
$\tilde Z Q = 0$ so probes are projected once by $P'$ and no parking term
exists. The solve form is not needed at any size.

**Net.** Setup is today's iteration with the threshold in the seed and
the per-step re-projection, plus a minute of filtering; 3.7× fewer
moments; and a per-step cost that is lower, not higher, by the fill
ratio above (1.6–3× at the 9³ cell, growing with $N$). The whole run
gains **the spectrum factor times the fill factor — of order 6–10× at
$9^3$** and more beyond, with $O(N)$ memory throughout, plus robustness
(the iteration is well-conditioned instead of resolving fake modes at
the `sptol` floor). The per-step
cost is the general dense-$Z$ problem shared by every non-orthogonal
basis in the code; the null band only made it worse. Solving it — a
truncated inverse square root with controlled error, or a preconditioner
good enough for a 2–3-iteration solve — is a separate design, orthogonal
to this one.

## 5. Validation (Al, 7³ cell, 343 atoms, $N = 7203$)

| quantity | full basis | deflated |
|---|---|---|
| null states found | — | 1372 = 4.00 / atom |
| residual of $V$ outside the null band | — | $5.6\times10^{-8}$ (3 passes, free seed; same rate on 5³, 6³) |
| decay of the null functions | — | projected basis: 20× / hop (e-fold 2.5 Å); Löwdin basis: 3× / hop |
| error vs truncation of $V$ at $10^{-3}$ / $10^{-4}$ / $10^{-5}$ | — | $1.5\times10^{-2}$ / $2.8\times10^{-4}$ / $1.0\times10^{-5}$ eV |
| span | 603 eV | 182 eV |
| $\kappa$ of the solved system | $3.15\times10^{4}$ | 48.3 |
| error near $E_F$ vs exact canonical orthogonalization | — | $1.7\times10^{-8}$ eV max |
| parked states | — | all at $E_{\rm park}$, spread $2\times10^{-12}$ |
| exact factor fill at $10^{-4}$, 9³ cell | 7.6 nnz(S) | 4.4 nnz(S) |
| exact factor decay per hop, 9³ cell | ~5× | ~10× |
| $\tilde Z$ by re-projected Newton–Schulz on the sparse $S$ | diverges | 8 steps (3 from the cluster seed) |

Sources: `poc_pz_nonsym.py` (one-sided deflation of a non-symmetric
factor), `poc_null_wannier.py`, `poc_null_wannier_scaling.py`,
`poc_null_wannier_trunc.py`, `mu_by_kpm_dos_of_S.py`,
`contraction_atom_blocked.py` (the per-atom result of section 2) and their
`*_results.txt`.

## 6. What this does not fix

- The $-92$ eV floor is a genuine eigenvalue of $H$ on the well-conditioned
  subspace — two contracted $s$ functions with on-site energies $-47.6$ and
  $-41.1$ eV form a bonding state. No overlap-based method touches it; it is
  a defect of the Hamiltonian data and the remaining ~3× of span.
- Without a gap in $\rho_S$ the null band is not defined and the filter
  converges slowly; the run reports it and does not deflate. This is not
  hypothetical: on an 8³ k-mesh (`overlap_spectra_diamond_silicon_results.txt`)
  **diamond** ($\kappa(S) = 8\,700$) has a gapped null band of 2 states per
  atom in $[1.8\times10^{-3}, 1.4\times10^{-2}]$ with a gap to
  $3.2\times10^{-2}$ (ratio 2.3) — deflatable by the same construction, and
  the origin of its 273 eV transport span — whereas **silicon**
  ($\kappa(S) = 10\,600$) has no gap: its low modes run continuously from
  $5.7\times10^{-4}$ into the next band (ratio 1.03 at the widest break).
  All three bases in `test/data` are ill-conditioned; the method applies to
  two of them.
- The code's present factor is sparse for diamond and silicon by
  measurement (a bounded multiple of an $S$-apply), which at
  $\kappa \sim 10^{4}$ means the truncated Newton–Schulz iteration does not
  fully resolve the near-null modes — an implicit deflation by `sptol`.
  The deflated factor is built by the same truncated iteration, so its
  fill is to be measured against today's, not assumed; the exact-factor
  decay comparison (§4) is the available proxy and says it is sparser.

## 7. Implementation plan

1. **Options.** `deflation = {enabled, mu (auto|value), passes, degree
   (auto|value), sptol, e_park (auto|value)}` in `SetupOptions`; JSON
   schema and `doc/json_input.md` follow.
2. **Where.** `OperatorSet::build_preconditioner`
   (`rsatb/setup/operator_set.hpp`), which holds `state.overlap` and
   `state.hamiltonian` per k-point: run §3.1, build $V$ by §3.2, build the
   thresholded 1-hop cluster seed $\tilde Z$ from the existing dense
   subgraph kernel with modes below $\mu$ dropped; re-express $V$ in the
   projected basis, truncate at the configured tail, and factor the sparse
   Gram matrix $G$ by the existing Newton–Schulz routine; store $V$,
   $G^{-1}$ (local) and $\tilde Z$;
   `H`, `S` untouched, `state.overlap` kept; the ill-conditioned
   Newton–Schulz is not run.
3. **Kernels**, in `rsatb/calc/` beside `isqrtm.hpp`: `overlap_dos` (thin
   wrapper over the Chebyshev trace); `null_band_basis` — block Chebyshev
   recursion on a distributed thin matrix with localized columns (the
   intermediate $T_k V$ carry a bounded physical component and an
   exponentially growing null component, so a radius cap of 3–4 hops and
   a relative drop tolerance during the recursion are safe once the null
   component dominates; validate this in the first sparse test), and a
   sparse Löwdin step on the Gram matrix via the existing Newton–Schulz
   routine. Phase markers `setup/deflate_gap`, `setup/deflate_filter`,
   `setup/deflate_orth`. A `DeflationReport {mu, gap, null_states, passes,
   residual, support_hops, nnz_ratio, e_park}` that the summary prints.
4. **Operator forms.** At practical sizes, `ChebyOpForm::Sandwich` with
   the deflated $\tilde Z = PZ$ (today's factor projected once on the
   left through $V$ after `isqrtm`; the re-projected iteration only as
   fallback) and each stochastic probe projected once by
   $P' = I - (Z^{\dagger}SV)G^{-1}(V^{\dagger}Z)$ where the samplers draw it; no parking
   term, no per-step projection; velocity/position operators need nothing
   extra since $\tilde Z Q = 0$. Parking ($+E_{\rm park}Q$) is only the
   pencil form's device. For $\gtrsim 10^{4}$ atoms, a
   `ChebyOpForm::DeflatedSolve`: $\tilde S^{-1}\tilde H$ with $P$ applied as
   $v - VG^{-1}(V^{\dagger}v)$ and $\tilde S$ as an action inside the existing
   PCG preconditioned by the cluster seed.
5. **Bounds.** `resolve_spectral_bounds` runs on the deflated operator
   unchanged; $E_{\rm park}$ auto = band top + 10 eV from one bounds pass.
6. **Tests** (`test/`): 2-atom diamond — $\rho_S$ shows no gap, deflation
   is a no-op and operators are bit-identical; Al primitive on a small
   k-mesh — 4 null states/atom, residual $< 10^{-6}$ in 3 passes, span
   $< 200$ eV, eigenvalues within $10^{-6}$ eV of `scipy` canonical
   orthogonalization near $E_F$, parked states at $E_{\rm park}$; the
   transport-scale test with deflation on — dc spectrum unchanged near
   $E_F$ to the stochastic noise.

## 8. References

- Zhou, Saad, Tiago, Chelikowsky, "Self-consistent-field calculations using
  Chebyshev-filtered subspace iteration", J. Comput. Phys. 219, 172 (2006)
  — the filter of §3.2.
- Damle, Lin, Ying, "SCDM-k: localized orbitals for solids via selected
  columns of the density matrix", J. Comput. Phys. 334, 1 (2017) — localized
  bases of spectral subspaces.
- Brouder, Panati, Calandra, Mourougane, Marzari, Phys. Rev. Lett. 98,
  046402 (2007); Benzi, Boito, Razouk, SIAM Rev. 55 (2013) — exponential
  localization of Wannier functions and spectral projectors of isolated
  bands, which sets the 2-hop support.
- Fix, Heiberger, SIAM J. Numer. Anal. 9 (1972) — deflation of the
  near-singular subspace with a parking value (§3.3).
- Röder, Silver, Drabold, Dong, Phys. Rev. B 55, 15382 (1997) — KPM on the
  generalized problem with $S^{-1}$ by preconditioned CG.
- Nakata, Bowler, Miyazaki, Phys. Chem. Chem. Phys. 17, 31427 (2015);
  Lara et al., J. Chem. Theory Comput. (2026), arXiv:2512.00118 — why a
  per-atom contraction cannot do this (§2).
