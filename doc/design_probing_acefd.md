# Probing × ACE-FD: the composed error budget

*Design note closing validation item V2 of `design_realspace_density_ldos.md`
(§7.1): can the deterministic distance-d probing of the density matrix use the
short tapered (ACE-FD) expansion, and with what error control? Measured
evidence: `benchmarks/probing_acefd/`.*

**Verdict: yes, with a co-adaptation rule for (N, d) and one repaired
indicator.** In metals the composition is *better* than either analysis
predicted separately — the short taper acts as a beneficial range cutoff. One
flaw in the original probing design was found and fixed: the symmetry
indicator Δ_ij is exactly blind on (near-)periodic systems.

---

## 1. What was unproven

Two analyses each left a gap at their interface:

* Probing (`design_realspace_density_ldos.md` §5.3, Eq. 14) bounds the
  contamination by the far field of the **exact** kernel,
  $|K_{ik}| \le C e^{-r/\xi_K}$. But the object actually probed is the
  degree-N polynomial $f_N(H)$, whose far field is *not* the exact one: the
  truncation residual has spatial range set by the resolution, not by
  $\xi_K$.
* ACE-FD (`benchmarks/fermi_dirac/METHOD.md` §7) certifies weak-form safety
  for traces; individual entries $D_{ij}$ are contractions against
  cross-LDOS measures $\nu_{ij}$ — smooth-envelope expected but unverified
  (this was V2).

## 2. Composed identity

Probing recovers, for entry $(i,j)$ with class $c \ni j$:

$$
\hat K_{ij}
 = \underbrace{[f_N(H)]_{ij}}_{\text{wanted}}
 + \underbrace{\sum_{k \in c,\,k \ne j} [f_N(H)]_{ik}}_{\text{contamination},\ r_{ik} > d - 2r_c}
$$

so the total per-entry error splits into two *separately measurable* terms:

$$
\hat K_{ij} - K_{ij}
 = \varepsilon_{\mathrm{trunc}}(N;\nu_{ij})
 + \varepsilon_{\mathrm{contam}}(N, d),
$$

* $\varepsilon_{\mathrm{trunc}} = -\sum_{n} (1-g_n) a_n \mu_n^{(ij)} - \sum_{n \ge N} a_n \mu_n^{(ij)}$
  — a weak-form product series against the near-diagonal cross-LDOS
  $\mu_n^{(ij)} = (t_n)_i$ (free during the recursion). Same structure as the
  trace case; ACE-FD's indicator applies **entry-wise** from checkpoint
  partial sums at $N/2, 3N/4, N$ plus the Richardson–Jackson value.
* $\varepsilon_{\mathrm{contam}}$ = far field of the **effective** kernel =
  exact far field ($e^{-r/\xi_K}$, or algebraic in $T\to 0$ metals) **plus**
  the residual far field, whose range grows with $N$: the tapered $f_N$
  behaves spatially like a Fermi kernel at effective resolution
  $\sigma_{\mathrm{eff}} \approx 2\pi a/N$ (effective degree $N/2$), i.e.
  range $\sim \hbar \bar v/\sigma_{\mathrm{eff}} \propto N$, with an
  additional weak algebraic ringing tail.

**Consequence — the N–d coupling.** $\varepsilon_{\mathrm{trunc}}$ falls with
$N$; $\varepsilon_{\mathrm{contam}}$ *rises* with $N$ at fixed $d$ (longer
effective range). At fixed $d$ there is an interior optimum $N^\*(d)$;
pushing $N$ beyond it is not merely wasted MVPs but actively harmful. This
is invisible in the trace analysis and specific to the probing composition.

## 3. Measured facts (60×60 tight binding, kT = 0.01t, βW ≈ 800)

Script: `benchmarks/probing_acefd/bench_probing_taper.py`; raw output in
`benchmarks/probing_acefd/results.txt`. Errors are max over the wanted
pattern (r ≤ 2); "exact" rows probe the true $f(H)$ (pure contamination
baseline). Pointwise degree Eq. (7) would be ≈ 1270.

### 3.1 Metal (μ = −0.3t, clean, periodic colouring)

| d | best kernel | total | contamination | exact-kernel baseline |
|---|---|---|---|---|
| 20 | taper-64 | 1.0e-3 | 7.7e-4 | 1.1e-3 |
| 30 | taper-64 | 1.1e-3 | 1.1e-4 | 8.1e-3 |
| 30 | jack-2538 (converged) | 8.2e-3 | 8.2e-3 | — |

* Contamination at d = 30: taper-64 → 1.1e-4, taper-128 → 7.9e-3,
  taper-256 → 8.5e-3 ≈ exact. **Range grows with N as predicted; the short
  taper is 70× shorter-ranged than the exact T≈0 kernel here.**
* taper-64 *beats the fully converged kernel* at equal d (1.1e-3 vs
  8.2e-3): its truncation on wanted entries is weak-form small (max
  1.2e-3, rms 6e-4 — at this K = 3600 the granularity floor
  $\sim (NK)^{-1/2}$ dominates; improves as $K^{-1/2}$), while its
  contamination is cut by the range effect. The effective-temperature
  picture: the composed method evaluates the *cold* observable through an
  object with the *warm* system's spatial structure — exactly what probing
  needs.
* Jackson at equal N has 3–5× larger entry truncation (one-signed smearing
  bias; e.g. jack-64 diagonal density error 3.2e-3 at every d).

### 3.2 Gapped (Δ = 1, μ mid-gap)

Everything converges fast and $d$ controls the error alone: at d = 12 all
kernels sit on the exact contamination floor 1.2e-4; at d = 20, taper-128
reaches 2.7e-5. The taper's ringing leaves a far-field floor (median
8e-7 at r ∈ [22,30) vs 1.4e-8 for Jackson-128, 6.7e-10 exact) — relevant
only below ~1e-5 tolerances; if needed, raise N (floor drops as ringing
$1/(N\Delta E)^3$) or use Jackson there (gapped N* is small for both, ACE-FD
was never needed for gaps).

### 3.3 Indicator blindness on periodic systems (new finding, with fix)

On the clean lattice with superlattice colouring, the free symmetry
indicator $\Delta_{ij} = |\hat K_{ij} - \hat K_{ji}|$ of
`design_realspace_density_ldos.md` §5.5 is **identically zero
(≈ 1e-15) while the true contamination is 6e-2**: translation invariance
maps the contaminant set of $\hat K_{ij}$ onto that of $\hat K_{ji}$, so the
two contaminations are equal, not independent. The independence argument
fails exactly in the (near-)periodic limit. With disorder + randomized
colouring the indicator tracks contamination within ~30–50% (measured), but
random colouring costs ~1.5× more classes and worse worst-case geometry, so
it is not the fix of choice.

## 4. The design

1. **Colouring**: keep the periodic superlattice (optimal class count).
2. **Truncation control (knob N)**: per-entry ACE-FD indicator from
   checkpoint partial sums at N/2, 3N/4, N + Richardson–Jackson, all from
   the already-recorded $(t_n)_i$. Memory: ~4 copies of the wanted-entry
   set.
3. **Contamination control (knob d)** — two independent, cheap measurements
   replacing the blind Δ_ij:
   * **Shifted-colouring cross-check**: re-probe a small random subset of
     classes with the superlattice shifted by an offset outside the
     superlattice. The two recoveries of the same entry have different
     contaminant sets (translation maps one colouring onto itself, not onto
     the shifted one); their mismatch measures contamination. Cost = subset
     fraction × recursion, amortizable across SCF rounds.
   * **Canary columns**: a handful of isolated single seeds (contamination-
     free by construction) compared against their probed recovery. Also
     validates the truncation indicator against a full-degree Jackson canary.
4. **Co-adaptation**: raise N while the truncation indicator dominates the
   budget; raise d while the contamination indicator dominates. In metals
   expect and accept an interior optimum (N*, d*); never raise N past the
   contamination floor at the current d. Fallbacks: N capped at
   $N_{\mathrm{pointwise}}$ (Jackson parity), d capped by the class budget
   $\chi(d) \propto d^3$; if both saturate, the tolerance is unreachable —
   the honest algebraic-decay limit of the T→0 metal, which no
   linear-scaling method escapes.
5. **Downstream discipline**: all uses of $\hat K$ stay linear (fused
   contraction §5.4); symmetrize $\hat K \to (\hat K + \hat K^T)/2$.
6. **SCF closure**: errors are deterministic and smooth in $H$ — no
   stochastic noise rectification. Independently audit global $N_e$ and
   $E_{\mathrm{band}}$ each SCF round with trace-mode ACE-FD (shared
   moments, ~free) as a global consistency check on the probed density.

## 5. Status of V2

Closed for the frozen-H model level: per-entry weak-form truncation
confirmed (taper beats Jackson per entry; diagonal = LDOS measures behave as
traces), N–d coupling confirmed and quantified, indicator repaired. Still
open at production level: DFTB-scale system, non-orthogonal S, SCF loop
(checklist in `design_realspace_density_ldos.md` §10, using the audits of
§4.6 above).
