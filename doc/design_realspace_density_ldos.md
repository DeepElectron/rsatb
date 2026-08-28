# Real-space LDOS and electron density for LCAO rsatb

*Design of record: one Chebyshev engine, three seed policies. Deterministic
distance-d probing for the density; region-block seeding for the LDOS;
the existing random seeding retained only as a cross-check.*

**Status: proposed.** Agreed constraints, from design review: the methods must
be deterministic (no stochastic estimator in the production path), must not
use density-matrix purification (tried; slow, and carries a non-ignorable
error floor — §8.4 explains why it is structural), must not perform sparse
matrix–matrix products at runtime (memory growth and cost), must never store
the grid-orbital matrix $\Phi$, and must scale to $10^7$ atoms in a
non-orthogonal LCAO basis with hard-cutoff orbitals. Positions and atom order
follow the repository convention: exactly as the user supplied them.

---

## 1. Setting, targets, and the one structural lemma

LCAO basis $\{\varphi_i(\mathbf r)\}$, real orbitals with a hard cutoff:
$\varphi_i(\mathbf r) = 0$ for $|\mathbf r - \boldsymbol\tau_i| > r_c^{(i)}$,
with $r_c = \max_i r_c^{(i)}$ (diamond test basis: $r_c = 3.70$ Å). Sparse
Hermitian $H$, $S$ with $S_{ij} = \langle\varphi_i|\varphi_j\rangle$,
generalized problem $Hc_k = \varepsilon_k S c_k$, normalization
$c_k^\dagger S c_l = \delta_{kl}$.

The two targets, from the eigenvector expansion
$\psi_k(\mathbf r) = \sum_i c_{ki}\varphi_i(\mathbf r)$:

**Density** (per SCF round, whole cell):

$$
n(\mathbf r) = \sum_k f_\beta(\varepsilon_k - \mu)\,|\psi_k(\mathbf r)|^2
= \sum_{ij} \varphi_i(\mathbf r)\,\varphi_j(\mathbf r)\, D_{ij},
\qquad
D = \sum_k f_\beta(\varepsilon_k-\mu)\, c_k c_k^\dagger .
\tag{1}
$$

**LDOS** (selected energies, selected region — STM via Tersoff–Hamann):

$$
\rho(\mathbf r, E) = \sum_k \delta_\eta(E - \varepsilon_k)\,|\psi_k(\mathbf r)|^2
= \sum_{ij} \varphi_i(\mathbf r)\,\varphi_j(\mathbf r)\, A_{ij}(E),
\qquad
A(E) = \sum_k \delta_\eta(E-\varepsilon_k)\, c_k c_k^\dagger .
\tag{2}
$$

Both are the same contraction $Q(\mathbf r) = \varphi(\mathbf r)^T K
\varphi(\mathbf r)$ against a kernel-filtered matrix $K \in \{D, A(E)\}$. What
separates the two problems is only *which entries of $K$ are needed*, and
that is fixed by the following.

**Lemma 1 (needed-entry set).** The contraction (1)–(2) involves $K_{ij}$
only for pairs whose orbitals overlap somewhere:
$|\boldsymbol\tau_i - \boldsymbol\tau_j| < 2r_c$. This set is exactly the
sparsity pattern of $S$ — finite, known in advance, and *exact* because the
LCAO cutoff is hard, not asymptotic.

For the LDOS on an STM patch, additionally only orbitals reaching the
evaluation plane contribute: the needed block is $A_{RR}(E)$ with $R$ the
surface-region orbital set, $|R| \sim 10^3\!-\!10^4$ per patch. For the
density, the needed set is the full S-pattern:
$N_\mathrm{orb} \times$ (pattern row size) entries — at the diamond basis,
$1.3\times10^8 \times 3809$. The whole density problem is how those columns
share a small set of seeds (§5).

---

## 2. Operator algebra: the recursion that never forms $S^{-1/2}$

Let $H' = (H - c\,S)/w$ with $c = (E_\mathrm{max}+E_\mathrm{min})/2$,
$w = (1+\epsilon)(E_\mathrm{max}-E_\mathrm{min})/2$, so the pencil spectrum
maps into $(-1,1)$. Define the Löwdin-orthogonalized operator
$\tilde H = S^{-1/2} H' S^{-1/2}$ and the *contravariant Chebyshev matrices*

$$
X_n \;=\; S^{-1/2}\, T_n(\tilde H)\, S^{-1/2} .
\tag{3}
$$

Any spectral kernel then takes the form needed by (1)–(2). Writing
$g(\tilde H) = \sum_n \gamma_n T_n(\tilde H)$ and using
$c_k = S^{-1/2} u_k$ with $u_k$ the orthonormal eigenvectors of $\tilde H$:

$$
K = S^{-1/2} g(\tilde H) S^{-1/2} = \sum_n \gamma_n X_n ,
\qquad
\begin{cases}
\gamma_n = \gamma_n^{f}, & K = D \ \text{(Fermi coefficients, §7.1)}\\[2pt]
\gamma_n = g_n^{J}\,\tfrac{2-\delta_{n0}}{\pi}\tfrac{T_n(\tilde E)}{\sqrt{1-\tilde E^2}}, & K = A(E) \ \text{(Jackson-damped delta)} .
\end{cases}
\tag{4}
$$

**Derivation of the $S^{-1/2}$-free recursion.** From
$T_{n+1}(\tilde H) = 2\tilde H\, T_n(\tilde H) - T_{n-1}(\tilde H)$, multiply
by $S^{-1/2}$ on both sides:

$$
X_{n+1}
= 2\, S^{-1/2}\tilde H\, T_n(\tilde H) S^{-1/2} - X_{n-1}
= 2\, (S^{-1}H')\, \underbrace{S^{-1/2} T_n(\tilde H) S^{-1/2}}_{X_n} - X_{n-1},
$$

because $S^{-1/2}\tilde H = S^{-1}H' S^{-1/2}$. Hence

$$
\boxed{\;X_0 = S^{-1}, \qquad X_1 = S^{-1}H'\,S^{-1},\qquad
X_{n+1} = 2\,S^{-1}H'\,X_n - X_{n-1}\;}
\tag{5}
$$

Equivalently $X_n = T_n(S^{-1}H')\,S^{-1}$ (similarity transform of the
Chebyshev identity). Each step costs **one $H'$ application and one
$S$-solve** — no $S^{-1/2}$ is ever formed, so the Newton–Schulz filter
tolerance (`sptol` in `rsatb/calc/isqrtm.hpp`) no longer caps accuracy
anywhere. $S^{-1/2}$ survives only as one *option* for applying $S^{-1}$
(§7.3), where being approximate costs iterations, not correctness.

$X_n$ is symmetric: $X_n^T = S^{-1}T_n(H'S^{-1}) = T_n(S^{-1}H')S^{-1} = X_n$.
This symmetry is used twice below (moment doubling §7.2, error indicator
§5.5).

**Vector mode.** Seed $s$, iterate $t_n = X_n s$:

$$
t_0 = S^{-1}s,\qquad t_1 = S^{-1}H'\,t_0,\qquad
t_{n+1} = 2\,S^{-1}H'\,t_n - t_{n-1},
\tag{6}
$$

and accumulate on the fly, $y = \sum_n \gamma_n t_n$, keeping three state
vectors plus one accumulator. For a unit seed $s = e_j$ this yields **column
$j$ of $K$ exactly** (to polynomial degree): $y = K e_j$. That exactness is
the primitive both methods are built on. All seeds are processed as blocks
(`DistMultiVector`, `mult_dense`) for BLAS3 efficiency.

This is the existing non-orthogonal branch of
`rsatb/calc/cheby.hpp` (`Chebyshev::apply_op_internal`, `S != nullptr` path),
reused unchanged in block form.

---

## 3. LDOS / STM: deterministic region-block seeding

### 3.1 Method

1. **Region.** $R$ = orbitals whose cutoff spheres reach the evaluation
   points (for an STM plane above a surface: top layers only; the plane
   itself sits where only orbital tails reach, so the covering set per
   point is small).
2. **Seeds.** The $|R|$ unit vectors, one block; chunked if large.
3. **Recursion.** (6) on a truncated domain (§3.3); record the block rows
   $R$ of each $t_n$: $(X_n)_{RR} = E_R^T t_n$ — an $|R|\times|R|$ dense
   block per moment.
4. **Accumulate per requested energy** (bias voltages are few):
   $A_{RR}(E) = \sum_n \gamma_n(E)\,(X_n)_{RR}$, with **Jackson** damping —
   the positive kernel; an STM signal must be non-negative, so the
   ACE-FD-style tapers are excluded here by design. All energies come from
   the same moments; additional biases are free.
5. **Contract once per energy** with the collocation engine (§6):
   $\rho(\mathbf r,E) = \Phi_R(\mathbf r)^T A_{RR}(E)\, \Phi_R(\mathbf r)$.

No randomness, no $D$, no matrix products, no pattern truncation. The
resolution $\sigma \approx \pi w/N$ is a *chosen* broadening (a known
positive kernel), not an uncontrolled error; degree $N \approx \pi w/\eta$
(≈ 600 at $\eta = 50$ meV, $W = 20$ eV).

### 3.2 Moment doubling for the block (halves the step count)

From $T_{2n} = 2T_n^2 - I$ and $T_{2n+1} = 2T_nT_{n+1} - T_1$, conjugating
as in (3) and inserting $S^{1/2}S^{-1/2}$ between the factors:

$$
X_{2n} = 2\,X_n S X_n - X_0, \qquad
X_{2n+1} = 2\,X_n S X_{n+1} - X_1 .
\tag{7}
$$

With $t_n = X_n E_R$ and symmetry of $X_n$:

$$
(X_{2n})_{RR} = 2\,t_n^T (S\,t_n) - (X_0)_{RR}, \qquad
(X_{2n+1})_{RR} = 2\,t_n^T (S\,t_{n+1}) - (X_1)_{RR} :
\tag{8}
$$

$2N$ moments from $N$ recursion steps, at the cost of one extra $S$·block
product per step. (Vector-mode *readout* for the density cannot use this —
consistent with `benchmarks/fermi_dirac/METHOD.md` §6: doubling is for
quadratic forms, and the density readout needs $y$ itself.)

### 3.2b Window-integrated maps and point probes

*Constant-current* STM needs $\int_{E_F}^{E_F+eV}\rho(\mathbf r,E)\,dE$, not
the fixed-energy LDOS. Since the energy dependence lives entirely in the
scalar coefficients, the window integral is just another coefficient vector,
$\gamma_n^{\mathrm{win}} = \int_{E_F}^{E_F+eV}\gamma_n(E)\,dE$ (a 1-D
quadrature over Jackson coefficients), applied to the same moments in the
same accumulation loop. Both STM modes and all bias windows come from one
recursion.

Because Jackson is a positive kernel, $A_{RR}(E)$ is PSD and
$\rho(\mathbf r,E) = \Phi_R^T A_{RR} \Phi_R \ge 0$ pointwise — interference
($i \neq j$) terms included. No negative-LDOS artifacts are possible.

*Point probes* (the minimal case): for a single evaluation point, $R$ = the
$\sim$470 orbitals covering it, domain $\sim10^4$ atoms, $\sim$300 doubled
steps: $\sim10^{13}$ flops — about a second, full spectrum at that point,
deterministic. Note that no probing/colouring appears anywhere in the LDOS
path: every column of the seed block is its own seed, so the column
superposition of §5 never occurs and there is no separation condition.

### 3.2c As implemented

Shipped in `rsatb/calc/ldos.hpp` (moments) and `rsatb/calc/collocation.hpp`
(contraction), exposed as `RSATB::compute_ldos` / `driver.compute_ldos`.
Three details the implementation pinned down that the derivation left open:

**The seed must be dressed for whichever operator `Chebyshev::apply_op`
implements.** `X_n = S^{-1/2}T_n(\tilde H)S^{-1/2}` is recovered differently in
each of the three configurations the class supports:

| `Chebyshev` state | `apply_op` | seed | read-out |
|---|---|---|---|
| `S`, `M` both null | $H'$ | $V$ | rows $R$ of $t_n$ |
| `S` set | $S^{-1}H'$ | $S^{-1}V$ | rows $R$ of $t_n$ |
| `S` released, `M` set | $MH'M$ | $MV$ | rows $R$ of $Mt_n$ |

The middle row is the design's prescription (§2); the third arises only when
`operator_form="sandwich"` discarded the overlap.

**Output units are states/eV/Å³.** op2c normalizes its radials in Bohr, so a
raw contraction lands in Bohr⁻³ — a factor 0.529177³ = 0.1482 low. Everything
else in rsatb (positions, cell, cutoffs, the caller's points) is Ångström, so
the conversion is applied once at the end of the contraction. This was caught
by the orbital-normalization test, not by inspection.

**Minimum-image collocation carries a validity condition.** Taking each
orbital at its nearest image of the evaluation point is the whole Bloch sum
only when no second image also reaches the point, i.e. when every periodic
cell width exceeds $2r_c$. Narrower cells are rejected with the measured width
rather than silently losing weight. Large supercells at Γ — the regime this
path targets — satisfy it comfortably; a primitive cell generally does not.

### 3.2d The evaluation set is a grid, not a point list

`rsatb/calc/real_space_grid.hpp` defines the sole evaluation geometry: a
uniform grid over the cell at a requested spacing $dr$, tiled into boxes.
Point $(i_x,i_y,i_z)$ sits at $(i_x/n_0)\mathbf a_0 + (i_y/n_1)\mathbf a_1 +
(i_z/n_2)\mathbf a_2$, so the grid is periodic-consistent by construction.
Following rescu++ (`src/cpp/lcao_grid/orbital_grid.cpp`, `build_grid_boxes`).

Four things follow from the grid being a grid, none of which an arbitrary
point list can give:

- **Points are implicit.** $10^9$ points are three integers and a cell, not a
  24 GB coordinate array.
- **The box is the unit of locality.** Every point in a box sees essentially
  the same orbitals, so the active-atom list is built once per box rather than
  once per point, and the box contracts in one dense GEMM.
- **Reach is one sphere test.** A box carries a centre and a bounding radius
  computed from its eight index corners (tight for sheared cells), so "which
  atoms reach this box" is a single overlap test per atom — rescu++'s
  `grid_shell_mask`. Region selection drops from atoms x points to
  atoms x boxes.
- **Boxes are what gets partitioned.** Classifying box centres by the atoms'
  own inertial cut (§3.2d below) binds grid ownership to atom ownership.

### 3.2e Binding grid points to atoms: inertial decomposition

The contraction's scalability rests on *where* a rank's evaluation points are,
not how many it has. Splitting points by index leaves every rank needing every
orbital — the full $n_{\mathrm{orb}}$-wide $\Phi$ and the full block — so the
work per rank never falls.

vbcsr `core/atomic/inertial_partition.hpp` (ported from rescu++
`rescupp/operators/grid_partition.py`) fixes this with **Recursive Inertial
Bisection**: recursively split the weighted atoms perpendicular to their
principal axis of inertia at the weighted median. Cut planes tilt to follow the
geometry, so regions are compact — small cut surface, small halo — without
being box-shaped. The tree records each plane, so *the same planes classify
arbitrary points*, giving

$$
\texttt{point\_owner} = \texttt{atom\_owner}
$$

by construction (rescu++ calls this the seam-collapse precondition). It is
deterministic — a $3\times3$ symmetric eigenproblem plus a stable sort — so
every rank derives the identical partition from the replicated region with no
communication.

Consequences for the contraction, all measured by the tests: a rank's points
are spatially compact, so the orbitals reaching them are a compact subset
$R_r \subset R$; the orbital evaluation drops from $|R|$ to $|R_r|$ (linear)
and the GEMM from $|R|^2$ to $|R_r|^2$ (quadratic). Points are then scattered
by geometry rather than by index, so results are placed back by index and
summed — ownership being a partition is what makes that sound, and is tested
directly (`AssignsEveryPointExactlyOnce`).

Weighting atoms by orbital count balances collocation work rather than atom
count. Verified in `calc/test/test_collocation.cpp`:
`AssignsEveryPointExactlyOnce` and `BindsPointsToTheAtomsTheyLieOn` at 2, 3, 4,
8 and 16 parts, `IsDeterministicAndIndependentOfRankOrder`, and
`SurvivesMoreRanksThanAtoms` at 64 parts over 32 atoms. The partition is
communication-free, so one process deriving every rank's share is the same
computation the ranks would each do.

One condition is worth stating because it is a resolution limit rather than a
bug. A box is claimed whole, by where its centre falls, so the binding is only
as sharp as the boxes are small: a region finer than a box may contain no box
centre, and its points are then served by a neighbouring rank which does need
orbitals it does not own. The target regime is comfortably the other way -- at
$10^6$ atoms on $10^3$ ranks a region spans $\sim18$ A against a 2.4 A box at
0.2 A spacing -- and the overhang is what the ghost layer covers, so it
degrades rather than breaks. Exactness does not degrade at any ratio, which is
what the compact no-reduction buffers rely on:
`CoverStaysExactWhenBoxesAreCoarserThanRegions` asserts it in the inverted
regime.

The same tree is what a future subdomain recursion (§3.3) needs: it already
defines each rank's compact cell, so adding a buffer of $\xi(\eta)$ around it
turns the global recursion into independent local ones.

### 3.3 Domain truncation — exact locality, then physical locality

$T_n(S^{-1}H')$ applied to a seed has support within exactly $n$ graph hops
(each application of $S^{-1}H'$ extends support by one hop of the $H/S$
pattern plus the short, exponentially-bounded range of $S^{-1}$ — $S$ is
strongly diagonally dominant in an AO basis). So a degree-$N$ run touches
$\le N$ hops: truncating the domain to that ball is *exact*. The useful
radius is smaller: the Jackson kernel at resolution $\sigma$ suppresses
propagation beyond the physical correlation length

$$
\xi(\sigma) \sim \hbar v / \sigma
\quad(\text{Combes–Thomas-type resolvent locality}),
\tag{9}
$$

$\approx 65$ Å at 50 meV in diamond-like bands. Truncate at a few $\xi$,
verify by growing the radius (same protocol as the shift-convention tests).
Cost is then **independent of total system size**; lateral STM patches are
independent and map onto the existing `npar` sub-communicator machinery
unchanged.

### 3.3b Full-cell LDOS maps

Region mode restricts *space*, never energy (all energies come from one run's
moments). For $\rho(\mathbf r, E)$ over the whole cell, the probing machinery
of §5 applies verbatim — Lemmas 1–2 are kernel-agnostic — with the kernel's
decay length replacing $\xi_D$ in the choice of $d$: for $\delta_\eta$,
$\xi(\eta) = \hbar v/\eta$. Since $\chi \propto d^3 \propto \eta^{-3}$ and
$N \propto \eta^{-1}$, the cost law is

$$
\text{cost}_{\text{full-cell LDOS}} \;\propto\; \eta^{-4} \quad (3\text{D}):
\tag{9b}
$$

energy resolution buys delocalization, so full-cell maps are priced by the
requested broadening. Regimes at $10^7$ atoms: $\eta \approx 0.5$ eV —
$\sim\!10^{20}$ flops, a legitimate **one-shot** (LDOS is post-processing,
not per-SCF; ~100 density rounds ≈ one SCF trajectory); $\eta = 50$ meV
full-cell — ruled out by its own output volume ($10^{11}$ points × $10^2$
energies ≈ 100 TB) before its flops; fine-$\eta$ anywhere specific — region
mode on slices/planes/point sets at $\sim\!10^{16}$ flops per patch.

*Window-integrated maps* ("occupied LDOS in $[E_1,E_2]$ everywhere") are the
density machinery verbatim: $D_{\mathrm{win}} = f(H;\mu{=}E_2) -
f(H;\mu{=}E_1)$, with $\xi$ set by the **edge smearing** $\sigma$
($\xi = \hbar v/\sigma$), not the window width. A partition-of-unity stack of
soft-edged ~eV windows tiles the full spectrum, all windows accumulated from
the same probe recursions ($d$ set by the narrowest edge) — the practical
meaning of "full-range LDOS like the density", at density-grade cost.

### 3.4 Cost (formula first, then the working example)

Per patch: $N/2$ steps (doubling) × block width $|R|$ × $2\,\mathrm{nnz}
(H_\mathrm{dom})$ flops, ×$\kappa_S$ (the $S$-tax, §7.3). Example:
$10\times10$ nm patch, $|R| = 5\times10^3$, domain $10^5$ atoms
($\mathrm{nnz} \approx 5\times10^9$), $N = 600$, $\kappa_S = 3$:
$\approx 4.5\times10^{16}$ flops — **minutes on one modern node, regardless
of the $10^7$-atom cell**. Memory: chunked block states
$3 \times 1.3\times10^6 \times 10^3$ cols $\times 16$ B ≈ 60 GB per 1000-col
chunk, distributed.

---

## 4. Why the density cannot reuse the LDOS trick, and what is excluded

The density needs every S-pattern entry: $1.3\times10^8$ columns. Direct
seeding is $1.3\times10^8$ recursions — impossible. The excluded routes, with
reasons (details §8):

| route | excluded because |
|---|---|
| stochastic $D\varphi$ estimators | requirement: deterministic (SCF noise propagates nonlinearly) |
| purification (McWeeny/TC2/TRS4) | error amplification at the Fermi level is structural (§8.4); user-observed error floor |
| Chebyshev matrix accumulation, masked spgemm | runtime spgemm vetoed; 24 TB working set at target |
| divide & conquer, dense fragments | buffer³ arithmetic — the same pattern that cost 78 GB in `isqrtm_init` |
| PEXSI selected inversion | exact and spgemm-free, but $O(N^2)$ in 3D; viable only quasi-2D (kept as fallback) |

What remains is the observation that **columns of $K$ can share a seed if
they are far enough apart** — deterministic probing.

---

## 5. Density: deterministic distance-d probing

### 5.1 Class construction

Partition atoms into classes such that same-class atoms are separated by at
least $d$ (a periodic superlattice colouring achieves this with

$$
\chi_\mathrm{at}(d) \approx d^3 / v_\mathrm{atom}
\tag{10}
$$

classes — near the information-theoretic optimum, §8.1). Orbitals on one atom
never share a class (they are at distance 0), so
$\chi(d) = n_\mathrm{orb/atom}\,\chi_\mathrm{at}(d)$: **independent of system
size.** Diamond basis, $v_\mathrm{atom} = 5.67$ Å³:
$\chi(14.8\ \text{Å}) \approx 7.4\times10^3$,
$\chi(20\ \text{Å}) \approx 1.8\times10^4$.

Seed per class: $s_c = \sum_{k\in c} e_k$. By linearity the recursion output
is a **superposition of exact columns**:

$$
y_c = K s_c = \sum_{k \in c} K e_k .
\tag{11}
$$

### 5.2 Recovery: the uniqueness lemma

**Lemma 2.** Let the needed pattern be $P(j) = \{i :
|\boldsymbol\tau_i - \boldsymbol\tau_j| < 2r_c\}$ (Lemma 1). If the class
separation satisfies $d > 4 r_c$, then for every row $i$ at most one class
member $j \in c$ has $i \in P(j)$.

*Proof.* If $i \in P(j) \cap P(k)$ for $j,k \in c$, then
$|\boldsymbol\tau_j - \boldsymbol\tau_k| \le
|\boldsymbol\tau_j - \boldsymbol\tau_i| + |\boldsymbol\tau_i -
\boldsymbol\tau_k| < 4r_c < d$, contradicting the class separation. ∎

So the read-out rule is purely geometric, decided before any computation:

$$
\hat K_{ij} := (y_c)_i
\quad\text{for the unique } j \in c \text{ with } i \in P(j).
\tag{12}
$$

At the diamond basis $4r_c = 14.8$ Å — hence $d \ge 14.8$ Å and the class
counts above.

### 5.3 The single error term

Substituting (11) into (12):

$$
\hat K_{ij} = K_{ij} + \sum_{k\in c,\;k\neq j} K_{ik},
\qquad
|\boldsymbol\tau_i - \boldsymbol\tau_k| \;>\; d - 2r_c \;\;\forall k\neq j .
\tag{13}
$$

Every contaminant is an entry of the **exact** $K$ at distance $> d - 2r_c$
from $i$. With the standard off-diagonal decay of spectral-projector-like
functions (Kohn nearsightedness; Benzi–Boito–Razouk):
$|K_{ik}| \le C e^{-|\boldsymbol\tau_i-\boldsymbol\tau_k|/\xi_K}$, summing
shells of the superlattice gives

$$
|\hat K_{ij} - K_{ij}| \;\le\; C' \, e^{-(d-2r_c)/\xi_K},
\tag{14}
$$

deterministic, with $d$ the single knob and $d \to$ system size recovering
exactness continuously. $\xi_K$: gapped systems, $\xi_K \sim \hbar v /
E_g$-type, short; finite-$T$ metals, $\xi_K \propto \hbar v_F \beta$; $T\to0$
metals, algebraic decay — the method degrades gracefully ($d$ must grow) and
honestly (no linear-scaling method escapes this regime).

Note the contrast with the excluded routes: the polynomial $g_N$ is applied
to the seed **exactly** — nothing is truncated inside the recursion and
nothing iterates on an approximation. The only error is a *read-off of the
exact far field*, Eq. (14). Masked spgemm, by comparison, truncates
intermediates inside the recursion (legitimate 2-hop weight that should fold
back into pattern entries is silently dropped), and purification amplifies
its intermediate truncations (§8.4).

### 5.4 Fused contraction — the density without storing $\hat D$

Per chunk of classes, after accumulating $y_c = \sum_n \gamma_n^f t_n$:

```
for each class c in chunk:                   # y_c = one multivector column
    for each j in c:                         # private read-out window (Lemma 2)
        for each grid block b touched by sphere(j):
            C = orbital images covering b    # §6; ~500 orbitals
            n(r in b) += phi_j(r) * sum_{i in C} phi_i(r) * (y_c)_i
discard y_c
```

The only persistent object is the grid array $n(\mathbf r)$. $\hat D$ never
exists as a matrix. This realizes the "accumulate $D\varphi(\mathbf r)$ on
the fly" structure with a deterministic inner object.

### 5.5 Free a-posteriori error indicator

$K$ is symmetric, but $\hat K_{ij}$ and $\hat K_{ji}$ come from two
*different* seeds (column $j$'s class and column $i$'s class), with
independent contamination. Therefore

$$
\Delta_{ij} = |\hat K_{ij} - \hat K_{ji}|
\tag{15}
$$

is a computed, per-entry measurement of the contamination level, at zero
extra cost. Use: auto-tune $d$ on the first SCF iteration (raise $d$ until
$\max\Delta$ meets tolerance); symmetrize
$\hat K \leftarrow (\hat K + \hat K^T)/2$ for the contraction (averages the
two independent contaminations).

---

## 6. k-points, periodic images, and the R-space contraction

### 6.1 Ordering: k-sum in the matrix domain, one real contraction

Per k-point the recursion runs on the complex $H(k), S(k)$ built by the
existing `ImageContainer::sample_k` under the `R_AND_POSITION` convention
$H(k) = \sum_{\mathbf R} e^{i\mathbf k\cdot(\mathbf R + \boldsymbol\tau_j -
\boldsymbol\tau_i)} H_{ij}(\mathbf R)$. The read-out step accumulates the
real-space density matrix directly:

$$
D_{ij}(\mathbf R) = \sum_k w_k\,
e^{-i\mathbf k\cdot(\mathbf R + \boldsymbol\tau_j - \boldsymbol\tau_i)}
\,(y^k_c)_i ,
\tag{16}
$$

real by time-reversal ($k \leftrightarrow -k$), supported on exactly the
$S(\mathbf R)$ image pattern — the same layout the `ImageContainer` already
stores for $H(\mathbf R)$/$S(\mathbf R)$. Then **one** real-space
contraction:

$$
n(\mathbf r) = \sum_i \sum_{j,\mathbf R}
\varphi_i(\mathbf r)\,\varphi_j(\mathbf r - \mathbf R)\, D_{ij}(\mathbf R),
\tag{17}
$$

(translational collapse of the double lattice sum: fix the home cell on
index $i$, sum images on $j$). The k-sum costs one phase-multiply per entry
per k — negligible; k-points parallelize on the existing `npar` split.

### 6.2 Aliasing identity, and the equivalence with probing

Discrete k-sum over an $N_1\times N_2\times N_3$ mesh, applied to the exact
Bloch transform, gives (using
$\frac{1}{N_k}\sum_k e^{i\mathbf k\cdot(\mathbf R'-\mathbf R)} =
\sum_{\mathbf L} \delta_{\mathbf R',\,\mathbf R + \mathsf N\mathbf L}$ with
$\mathsf N = \mathrm{diag}(N_1,N_2,N_3)$):

$$
\tilde D_{ij}(\mathbf R) = D_{ij}(\mathbf R)
+ \sum_{\mathbf L \neq 0} D_{ij}(\mathbf R + \mathsf N \mathbf L) .
\tag{18}
$$

The k-sampled result equals the exact entry **plus the exact matrix's far
field beyond the Born–von-Kármán supercell** — formally identical to the
probing contamination (13). This is not an analogy but the same mechanism:
a Bloch seed $e_j$ at all k seeds orbital $j$ *in every BvK image
simultaneously* — a symmetry-enforced d-separated class with $d = N_i a_i$ —
and the Fourier phases perform the read-out separation that geometry performs
in (12). One physical criterion governs both knobs:

$$
\min_i N_i a_i \gtrsim \text{a few } \xi_K
\qquad\text{and}\qquad
d \gtrsim \text{a few } \xi_K .
\tag{19}
$$

Regimes: small cell / many k — every unit-cell orbital gets its own seed, the
k-machinery does all sharing; $10^7$-atom $\Gamma$-only — colouring does all
sharing, images survive only as boundary wraps; intermediate — colour within
the cell *and* k-sum across images, the two contamination bounds adding.
Same code path throughout. The LDOS block transforms identically
($A_{RR}(E;\mathbf R)$ before its single contraction).

### 6.3 Collocation engine (shared by both targets)

Never store $\Phi$. Per grid block: gather the **orbital images** $(j,
\mathbf R)$ whose spheres cover the block (a `NeighborList`-style search over
positions-as-given, shifts reported in the caller's frame — the repository
convention); evaluate $\Phi_\mathrm{blk}$ on the fly (cubic-spline radials
from op2c tables × real spherical harmonics — the one genuinely new numerical
layer); contract as two dense GEMMs:
$n_\mathrm{blk} = \mathrm{rowsum}\big(\Phi_\mathrm{blk} K_\mathrm{loc}
\circ \Phi_\mathrm{blk}\big)$. Image-awareness is built in from day one —
cheap now, painful to retrofit. Cost at $10^7$ atoms, standard grids:
$\sim10^{16}$ flops, a few percent of the recursion, **once per kernel**
(per SCF round for $n$; per energy for STM) — never per moment.

---

## 7. The acceleration stack

Ordered by leverage; each independent of the others.

### 7.1 ACE-FD degree adaptation (density only)

The Fermi coefficients $\gamma_n^f$ and stopping rule come from ACE-FD
(`benchmarks/fermi_dirac/METHOD.md`): degree set a posteriori by spectral
regularity, not by $\beta W$ — measured $N^* \approx 50\!-\!100$ where
Jackson-grade pointwise resolution would need $10^3\!-\!10^5$; worst case
capped at Jackson parity by the §5.2 fallback. **Open validation, flagged
honestly:** ACE-FD's safe class is verified for traces; individual $D_{ij}$
are contractions against oscillatory off-diagonal spectral measures, and the
weak-form argument must be *measured* for them (validation V2, §10). LDOS
does not use ACE-FD (positivity requires Jackson; $N \approx \pi w/\eta$ is
irreducible and affordable).

### 7.2 Moment doubling (LDOS only)

Eq. (8): $2N$ moments from $N$ steps, one extra $S$·block product per step.

### 7.3 The $S^{-1}$ application — three interchangeable options

| option | cost/step ($\kappa_S$) | notes |
|---|---|---|
| CG on $S$, block-Jacobi precond | $\sim$8–12 $S$-mults | exact to CG tol; zero setup; the default |
| Chebyshev polynomial $p_m(S) \approx S^{-1}$ | $2m{+}1$, $m\!\approx\!8$–16 | solver-free, deterministic step count |
| sparse $Z \approx S^{-1/2}$, apply $Z(\cdot)Z$ | $\sim$3 | one-time Newton–Schulz per **geometry** (`init_method="simple"` — the 78 GB subgraph-diagonalization path never runs); the only spgemm anywhere, amortized over all SCF rounds |

$S$ is strongly diagonally dominant in AO bases; all three are
well-conditioned. Accuracy is never capped by `sptol`: with options 1–2 the
recursion is exact; with option 3 residual $Z$-error is a preconditioner-like
perturbation, checkable against option 1.

### 7.4 Chunked BLAS3 execution

Classes processed as $\sim10^3$-column multivectors through `mult_dense`:
dense-block spmm at BLAS3 efficiency vs. spgemm's few percent of peak.
Memory per chunk at $10^7$ atoms: $4 \times 1.3\times10^8 \times 10^3 \times
16\,\mathrm B \approx 8$ TB distributed → **~8 GB/rank on $10^3$ ranks**;
grid array aside, nothing else persists.

### 7.5 SCF-adaptive separation

$\delta D$ between SCF iterations is small in amplitude, so the *same
absolute* tolerance admits a much smaller $d$ after the first iteration —
$\chi \propto d^3$: halving $d$ is $8\times$ fewer seeds. First iteration at
full $d$ (auto-tuned by the indicator (15)); later iterations compute
$\delta n$ from $f(H_\mathrm{new}) - f(H_\mathrm{old})$ columns at reduced
$d$. Classes depend only on geometry: built once, reused every round (and
across MD steps until atoms move materially).

### 7.6 Domain truncation and patch parallelism (LDOS)

§3.3: cost independent of system size; patches ride the `npar` machinery.

### 7.7 Optimality position (why this is not leaving a big factor)

*Lower bound.* Exact recovery of a pattern from matvec outputs requires the
probe count to be at least the chromatic number of the conflict graph
(pairs of columns sharing a pattern row), which is at least its max degree —
the orbitals within $4r_c$: $\approx 2.3\times10^3$ atoms ×13 $\approx
3\times10^4$ at the diamond basis (cf. the fixed-sparsity recovery bound of
Amsel et al., necessity and sufficiency of $\chi$ matvecs). The lattice
colouring at $d = 4r_c$ uses $\chi \approx 7.4\times10^3$ — *below* that
exact-recovery bound, because (14) relaxes exactness to the decaying tail;
within its own family (decay-tolerant probing) the superlattice construction
is within a packing constant ($\pi/6$-type) of minimal. No deterministic
matvec-based method does asymptotically better at fixed tolerance.

*Against the spgemm alternative* (per SCF round, $10^7$ atoms, diamond
basis, $N^* = 100$):

| | flops | working memory | error mechanism |
|---|---|---|---|
| probing, $d=14.8$ Å, $\kappa_S=3$ | $2.2\times10^{18}$ | ~8 GB/rank | exact-$K$ tail beyond $d$, Eq. (14) + indicator (15) |
| masked spgemm Chebyshev | $\sim10^{18}$ (incl. $S$-handling) | **24 TB** (6 pattern matrices) | intermediate truncation, no per-entry indicator |

Nominal flop parity to ~2×, but probing runs at BLAS3 efficiency
(5–10× the sustained rate of irregular spgemm), needs $10^3\times$ less
memory, and carries the cleaner error term. With §7.5, later SCF iterations
drop well below the first.

---

## 8. Error budget and exclusions

### 8.1 Complete error budget

| source | bound / character | knob | indicator |
|---|---|---|---|
| polynomial degree | chosen resolution (LDOS) / ACE-FD indicator (density) | $N$ | METHOD.md Eq. (15) triple check |
| probing contamination | $C'e^{-(d-2r_c)/\xi_K}$, Eq. (14) | $d$ | symmetric read-out $\Delta_{ij}$, Eq. (15) |
| BvK aliasing | $\sum_{\mathbf L\neq0} D(\mathbf R + \mathsf N\mathbf L)$, Eq. (18) | k-mesh | same physics as above, Eq. (19) |
| LDOS domain truncation | exponential beyond $\xi(\sigma)$ | radius | grow-and-compare |
| collocation | spline/quadrature order | mesh, spline order | validate $\varphi$ against op2c overlaps |
| $S^{-1}$ application | CG tol / $p_m$ residual / $Z$ residual | tol, $m$ | cross-option comparison |

Every term has a knob and a computable indicator. Nothing is stochastic.

### 8.2 What is exact

Class construction and read-out assignment (hard LCAO cutoff, Lemma 2);
the polynomial applied to every seed; the LDOS at its declared resolution
on its declared domain.

### 8.3 Honest limits

$T\to0$ metals: $\xi_K$ algebraic, $d$ grows, the advantage shrinks — shared
by every linear-scaling method in existence. ACE-FD off-diagonal validity is
*unproven* until V2 runs. Exact untruncated $D$ at $10^7$ atoms is not a
meaningful target for any method.

### 8.4 Why purification was excluded (the structural argument)

The McWeeny map $f(x) = 3x^2 - 2x^3$ has $f'(1/2) = 3/2$: the fixed points
0, 1 attract but the mid-spectrum — the Fermi level — is repelling with gain
1.5 per iteration. A truncation error injected at iteration $k$ of $m$ is
amplified by up to $1.5^{\,m-k}$ (up to $\sim2\times10^5$ over 30
iterations), concentrated at the energies that decide occupations. The
observed non-ignorable error floor is therefore structural, not tunable. The
linear accumulation (5) has no such mechanism: a step-$n$ perturbation
$\delta$ enters the result as $\sum_m \gamma_m U_{m-n}\delta$ with
$|U_k| \le k+1$, $|\gamma_m| \le 2/(\pi m)$ — errors add, bounded.

---

## 9. Prior art (searched 2026-07) and what is new here

**Exists, and this design builds on it directly:**

- Deterministic probing via graph colouring for *diagonals of inverses*:
  [Tang & Saad 2012](https://onlinelibrary.wiley.com/doi/10.1002/nla.779).
- Rigorous decay-based error bounds for probing approximations of matrix
  functions on patterns — the theory behind Eq. (14):
  [Frommer, Rinelli, Schweitzer, SIMAX 2021](https://arxiv.org/pdf/2009.01589).
- $\chi$-matvec necessity/sufficiency for exact fixed-pattern recovery:
  [Amsel et al. 2024](https://arxiv.org/pdf/2402.09379).
- Hierarchical/stochastic probing on lattices (QCD):
  [Stathopoulos et al.](https://dx.doi.org/10.1137/120881452),
  [extension](https://www.cs.wm.edu/~andreas/publications/extended_HP.pdf),
  [recent lattice-QCD practice](https://arxiv.org/pdf/2606.00394).
- Chebyshev/FOE and KPM machinery: CheSS
  ([arXiv:1704.00512](https://arxiv.org/abs/1704.00512)), KPM review
  (Weiße et al., arXiv:cond-mat/0504627), TBPLaS LDOS at scale
  ([arXiv:2509.26309](https://arxiv.org/html/2509.26309v1)).
- R-space pair-loop density contraction without storing $\Phi$: SIESTA
  ([arXiv:cond-mat/0111138](https://arxiv.org/pdf/cond-mat/0111138)), CP2K.
- Density-matrix decay theory:
  [Benzi–Boito–Razouk](https://arxiv.org/abs/1203.3953); BvK aliasing of
  $D$ from finite k-meshes is standard lore in periodic quantum chemistry.
- Alternatives kept as named fallbacks: PEXSI
  ([docs](https://pexsi.readthedocs.io/en/latest/introduction.html),
  quasi-2D), CONQUEST-scale deterministic DFT
  ([arXiv:2002.07704](https://arxiv.org/abs/2002.07704)), stochastic-DFT
  fragment/window noise reduction
  ([arXiv:2101.05906](https://arxiv.org/abs/2101.05906),
  [energy windows](https://pubs.aip.org/aip/jcp/article/151/11/114116/198880/Energy-window-stochastic-density-functional-theory),
  [metals 2024](https://pubs.aip.org/aip/jcp/article-abstract/160/21/214125/3296407/Noise-reduction-of-stochastic-density-functional))
  — retained only as a cross-check seed policy.

**Not found in the literature (the new assembly):** deterministic distance-d
probing applied to the *full off-diagonal S-pattern* of the LCAO density
matrix for *real-space grid contraction* (prior probing work targets
diagonals and traces); the k-sampling ≡ probing identification with BvK
aliasing as contamination, Eqs. (18)–(19), used as one unified error
criterion; the symmetric-read-out per-entry indicator (15); the composition
with ACE-FD degree adaptation; SCF-adaptive separation $d$. Components
published, assembly and the four italicized elements apparently novel — a
methods-paper opportunity alongside the implementation.

---

## 10. Validation plan and build order

Status: **V1 and V3 done** (LDOS path shipped and green); V2 and V4 pending
with the density.

Gates (frozen $H$, before any SCF):

- **V1 — collocation** *(done, `rsatb/calc/test/test_collocation.cpp`)*:
  orbital normalization $\int|\varphi_i|^2 = 1$ by quadrature through the
  contraction path, $s$-$p$ angular orthogonality, compact support past the
  cutoff, and the Angstrom cutoff read back from the file. The normalization
  check caught a real Bohr/Angstrom slip (factor $0.529177^3$) that no other
  test would have shown — the density was simply 6.75x too small.
- **V2 — probing error law**: diamond sc444, exact $D$ by diagonalization;
  measure $|\hat D - D|$ and $n(\mathbf r)$ error vs. $(d, N)$; confirm
  Eq. (14) slope and indicator (15) correlation; decide production ACE-FD
  applicability (§7.1) by direct measurement.
- **V3 — LDOS** *(done, `rsatb/calc/test/test_ldos.cpp` and
  `test/test_ldos_sum_rule.py`)*: every entry of $A_{RR}(E)$ against dense
  `zheev` at $10^{-9}$, driven with the same Jackson weights on both sides so
  a mismatch can only be the recursion, the seeding or the read-out; run at
  one and three orbitals per atom, since with one the region-orbital to
  scalar-row map is the identity and would hide a block-offset error.
  Also positive semi-definiteness of the block, chunking invariance, region
  selection, and $\int A(E)\,dE = S^{-1}$ (identity for an orthogonal model,
  to $5\times10^{-3}$) — which verifies the Jackson weights, the
  $1/\mathrm{half\_width}$ Jacobian and the reconstruction with nothing
  spatial involved. End to end on diamond sc444 via the parameter-free sum
  rule $\int\!\!\int \rho\,dE\,d^3r = N_{\mathrm{orb}}$: **1622.4 against
  1664, 2.5% low**, with the residual attributable to the test's sampling
  (energy spacing 2.9 eV against a 2.2 eV Jackson width, a 0.38 A spatial
  grid, and 2% of window clipped at each end) rather than to the method,
  since the energy axis is verified exactly by the identity check above.
  Still open: vacancy in sc444 (localized state), domain-truncation
  convergence (the truncation itself is not yet implemented).
- **V4 — k/Γ consistency**: same physical system as (cell, $N_k$ mesh) vs.
  BvK supercell at $\Gamma$: results must agree to the aliasing bound (18)
  — a direct test of §6.2.

Build order:

1. ~~**Block recursion engine** (Eq. 6; chunked `mult_dense`) +
   **collocation engine** (splines × $Y_{lm}$)~~ — done,
   `rsatb/calc/ldos.hpp`, `rsatb/calc/collocation.hpp`.
2. ~~**LDOS** (region seeds, Jackson, patches)~~ — done, exposed as
   `RSATB::compute_ldos` / `driver.compute_ldos`.
3. **Colouring + read-out + fused density** → V2, V4.
4. ACE-FD coefficients + SCF-adaptive $d$ (needs an SCF loop or a
   frozen-$\delta H$ mock) — last.

Deferred from step 2, in priority order:

- **Moment doubling** (Eq. 8). Not implemented: it needs a Gram matrix
  $t_n^\dagger S t_n$ per step, costing $|R|^2 N_{\mathrm{local}}$ against the
  recursion's $|R|\cdot\mathrm{nnz}$, so it wins only for
  $|R| \lesssim$ nnz-per-row (about 3800 at the diamond basis) — a clear win
  for point probes, roughly break-even for large patches. Worth adding behind
  that crossover once profiled, not before.
- **Domain truncation** (§3.3). Without it each matvec is $O(N)$ rather than
  $O(\xi^3)$; results are identical, only the cost differs. This is what makes
  the $10^7$-atom patch case affordable and is the main remaining work.
- **k-points** (§6.1). Currently Gamma-only, enforced with an explicit error;
  the generalization is the same read-out phase the density needs.
- The three $S^{-1}$ options (§7.3): only the CG solve path is wired, which is
  the accurate default. The polynomial and $Z$-apply variants are cost
  optimizations.

Risks: ACE-FD off-diagonal validity (V2 decides; fallback cap bounds cost,
not correctness); metallic $\xi_K$ growth (physics, disclosed); spline
$Y_{lm}$ conventions vs. op2c (V1 catches on day one).

---

## Appendix: colour-masked (probing) seeds — tried and rejected

Supporting each seed on one class of a colouring, so same-class cross-terms
cancel instead of being averaged, is unbiased for any colouring and keeps the
estimator a sum of squares (so positivity survives). In a cubic tight-binding
model it looked good: 1.9x at $\chi = 64$ and 2.8x at $\chi = 216$ for
$\eta/W = 0.046$, rising to 240x as the operator localises.

**On real diamond sc444 it lost**, 0.64-1.02x across valence, gap and
conduction at $\chi = 104$ and $351$ (six seeds each, matched `n_states`,
bias/SE 0.59-0.76 so the implementation was correct and unbiased).

Not class imbalance: at $\chi_{\text{spatial}} = 8$ the classes held exactly
16 atoms each with none empty. The colouring separates same-colour atoms by
5.06 Å, while the orbital cutoff alone is 3.70 Å and $\delta_\eta(E-H)$ at
$\eta/W \approx 0.045$ was measured not to decay over that range at all — so
it separates nothing it needs to and only unbalances the per-class weights. The
model misled because a cubic lattice has 6 neighbours per site where diamond has
~176 atoms inside the cutoff, and because the $S^{-1/2}$ metric sampler spreads
the seed before the filter runs.

Probing needs a local operator: large $\eta$, an in-gap energy, or strong
disorder — not the in-band narrow-$\eta$ maps this path exists to produce.
The lesson generalises: **locality of $\delta_\eta(E-H)$, not sampler design,
governs variance reduction for this estimator.**

---

## 10. Addendum (2026-08): the fragment control variate for sampled maps

Post-processing maps (rs_ldos.hpp / rs_density.hpp, route 5.3b) do not need
SCF-grade exactness, and §5's full probing costs χ(14.8 Å) ≈ 7.4×10³ seeds at
the diamond basis — ~100× the sampling sweep budget. What they suffered from
instead was the χ² sampling floor: each map point is a squared amplitude of a
random state, pointwise relative error √(2/n) — measured 17.0% / 8.5% at
64 / 256 samples on diamond sc444/sc888 against dense diagonalization,
against the predicted 17.7% / 8.8%.

`rs_fragment.hpp` closes that gap with a control variate (options
`fragment_core`, `fragment_buffer` on both map tasks):

$$ n(r) = n_0(r) \;+\; \mathbb E\big[\,|\Phi T\xi|^2 - |\Phi T_0\xi|^2\,\big], $$

with $T_0$ a dense-fragment surrogate of the filtered operator and $n_0$ its
exact map (streamed through the same contraction as identity blocks).
Unbiased for any $T_0$; the correction's variance carries only the residual
$T - T_0$, exponentially small in the buffer. This is the
overlapped-embedded-fragment scheme of stochastic DFT [Neuhauser–Baer–Rabani
JCP 141, 041102 (2014); Chen et al. JCP 150, 034106 (2019)] carried to the
LCAO pencil, and it composes with §5: probing spends locality on exactness,
the control variate spends the same locality on variance.

### 10.1 As implemented: the surrogate, the baseline, and the split

**$T_0$ approximates a square-root FACTOR of the density matrix, never
$D^{1/2}$.** The estimator does not iterate $D$; it iterates the factor

$$ T \;=\; Z\,p\big(Z^H H Z\big), \qquad p = \sqrt{f}, \qquad D = T\,T^H, $$

with $f$ the occupation for the density and $\delta_\eta(E_k - \cdot)$ for
each requested LDOS energy. $T$ is *a* factor, not the symmetric square root:
$TQ$ factors the same $D$ for every orthogonal $Q$, and that non-uniqueness is
exactly the gauge §10.2 is about. $T_0$ is therefore built as a
factor directly — the fragment eigendecomposition emits it in one step.
Building a surrogate $D_0$ and rooting it would be a sparse matrix product
(excluded by the design constraints) and would throw away precisely what the
map needs, which is the factor's columns.

**Three index sets, two of them physical.** Fragments are geometric objects,
and `BuildFragmentCv` carries three distinct sets:

| set | code | built from | role |
|---|---|---|---|
| core | `core_rows` | one fractional bin of edge `fragment_core`; the bins PARTITION the atoms | the **rows** of $T_0$ this fragment owns |
| $F$ (domain) | `dom_rows`, size $n_f$ | every atom within `fragment_buffer` of a core atom, minimum image; $\supset$ core | the **columns** of $T_0$ — the subspace the operator is compressed onto |
| row support | `nr`, size $n_r$ | atoms whose $Z$ rows carry weight against an $F$ column, trimmed at `kZrowTrim` = 1e-3 | scratch only: the rows of $Z[:,F]$ and $H$ needed to FORM the compression |

$F$ is an orbital set, not an atom set — the atom list is expanded through
`type_norb`. The row support is not a fragment in any physical sense; it
exists because $(Z[:,F])^H H\,Z[:,F]$ cannot be formed without knowing which
rows of $Z[:,F]$ are nonzero, and that support is strictly wider than $F$.
Untrimmed it is $Z$'s full row pattern, which is what made the first sc888
build ~1e17 flops.

Because the cores partition the atoms, every row of $T_0$ comes from exactly
one fragment: $T_0$ is a single global operator assembled from disjoint row
blocks, not a set of overlapping patches to be stitched. Rows owned by
different fragments still couple in the map — $(T_0T_0^H)_{ij}$ pairs them
through any column their domains share — so unlike the embedded-fragment
schemes it descends from, nothing is patched together on the grid.

**One fragment, once per run.** With $c, h$ the padded interval's centre and
half-width,

$$ A_{FF} = (Z[:,F])^H H\,Z[:,F] = V\,\mathrm{diag}(a)\,V^T, \qquad
   x_j = (a_j - c)/h, $$
$$ P_k \;=\; Z[\mathrm{core},F]\;V\,\mathrm{diag}\big(p_k(x_j)\big)\,V^T
       \;=\; Z[\mathrm{core},F]\;p_k(A_{FF}) , $$

and $P_k$ **is** the core rows of $T_0$ for filter $k$. The two indexed
objects, spelled out because both are easy to misread: a *table* $k$ is one
Chebyshev coefficient vector, i.e. one filter function — one per LDOS energy,
one for the density, one per window when `energy_windows` > 1; and
$p_k(x_j)$ is that filter evaluated by Clenshaw (`ClenshawEval`) at the
fragment eigenvalue $a_j$, mapped into the $[-1,1]$ the series is fitted on.
$V$ and $a$ come from $(Z,H)$ alone, so ONE dense eigensolve serves every
table: a 20-energy LDOS pays one extra $\mathcal O(n_c n_f^2)$ reweighting per
fragment per energy, never another eigensolve. In the limit of one
fragment holding everything, $Z[:,F]\,p(A_{FF}) = Z\,p(Z^HHZ) = T$
identically — the 2.3e-14 measurement below.

**Cost, and the one knob that moves it.** Per fragment: $2n_r^2n_f + 2n_rn_f^2$
to form $A_{FF}$, $\approx 9n_f^3$ for `dsyevd`, then $2n_cn_f^2$ per table.
With the row trim keeping $n_r = \mathcal O(n_f)$ this is $\mathcal O(n_f^3)$,
and $n_f$ is set by the domain volume,

$$ n_f \approx \rho_\mathrm{atom}\,\big(c^3 + 6c^2r + 3\pi c r^2 +
   \tfrac43\pi r^3\big)\, n_\mathrm{orb}, \qquad
   c = \texttt{fragment\_core},\; r = \texttt{fragment\_buffer}. $$

Diamond at $c = 4.5$ Å, $r = 4$ Å: domain $\approx 1520$ Å$^3 \approx 270$
atoms against a core of $\approx 16$. So the buffer is a CUBIC cost knob
against an exponential accuracy gain ($n_f \propto (c+2r)^3$, cost
$\propto n_f^3$), and at fixed buffer a smaller core is cheaper in total
($N/c^3$ fragments of $(c+2r)^3$ size) at the price of more redundant
dressing — 4.5 Å is a compromise, not an optimum. Two limits bind before
flops do: $H_d$ is a dense $n_r \times n_r$ per THREAD (fragments are the
parallel grain), and $P$ storage is $n_c \times n_f$ per fragment PER TABLE,
so a many-energy LDOS is memory-limited where the density is not.

Note what the shape of $P_k$ implies: only $n_c$ of $n_f$ rows are kept —
6% for the diamond numbers above — while `dsyevd` computes all of $V$. This
is verbatim the argument `isqrt_action.hpp` makes for the isqrtm seed ("a
full eigendecomposition costs a large multiple of $n^3$ to produce $n^2$
numbers of which only $n\cdot n_\mathrm{rhs}$ are used"), and
$P_k^T = p_k(A_{FF})\,Z[\mathrm{core},F]^T$ is exactly that kernel's thin-action
signature with $n_\mathrm{rhs} = n_c$. A block-Lanczos action started from
$Z[\mathrm{core},F]^T$ would serve every table from one Krylov space,
preserving the one-solve-serves-all-energies property; break-even against the
eigensolve is $m \approx 9n_f/n_c \approx 150$ steps at these sizes, so it
wins for the density filter and coarse $\eta$ and loses for sharp-$\eta$ LDOS,
where $m$ scales as $\mathrm{spread}/\eta$ for the same reason the main
recursion's degree does. Derived, not measured; and it removes only the
eigensolve share of the build unless $A_{FF}$ also stops being formed
densely. Recorded as an extension point, not a plan.

**$n_0$ is analytic. Only the difference is sampled.** This is the part the
identity at the top of §10 states but does not show. $T_0$ produces a map by
the same contraction as the true operator,

$$ n_0(r) = \sum_s \big|\Phi(r)^T T_0 e_s\big|^2
        = \sum_{ij}\varphi_i(r)\varphi_j(r)\,(T_0T_0^H)_{ij}, $$

with $D_0 = T_0T_0^H$ never formed — that would be the sparse product the
design forbids. Instead the COLUMNS of $T_0$ are streamed through the grid
over all $N$ basis vectors: no RNG, no reduction over samples, an exact column
sum. It is literally the `complete_basis` route applied to $T_0$ instead of
$T$, and it is affordable for the same reason that route is not affordable for
$T$: a column of $T$ costs a Chebyshev sweep, a column of $T_0$ is a stored
number (`ApplyIdentity`).

| term | how it is computed | error it carries |
|---|---|---|
| $n_0$ | deterministic sum over all $N$ columns of $T_0$ | none from sampling; exact for $T_0$, up to the collocation quadrature every term shares |
| $\bar X$ | $n_s$ Chebyshev sweeps of the true $T$ | full $\chi^2$ noise, ~17% at 64 samples |
| $\bar Y$ | $n_s$ fragment GEMMs against the SAME seeds | full $\chi^2$ noise, ~17%, correlated with $\bar X$ |
| $\bar X - \bar Y$ | their difference | variance $\propto$ the residual $T - T_0$ alone |

Read the assembly $\hat n = \bar X + \beta(n_0 - \bar Y)$ at $\beta = 1$ as
$n_0 + (\bar X - \bar Y)$: an exact deterministic map of the surrogate plus a
stochastic correction for what the surrogate got wrong. The third row is the
one that matters — $\bar Y$ is *just as noisy* as $\bar X$ on its own; what
makes the scheme work is that both are driven by the same $\xi$, so the
$\chi^2$ fluctuations cancel in the difference. Hence the buffer controls
VARIANCE and not bias: $n_0$ is exact for whatever $T_0$ was built, and the
samples are only ever asked to supply $T - T_0$. An insufficient buffer costs
sampling error, never a wrong density.

**Relation to the isqrtm graph seed.** The construction is the same family as
`graph_inverse_sqrt` — dense matrix function on a local neighbourhood, keep
only the centre's rows, assemble globally from disjoint row sets — with three
differences worth stating because the analogy otherwise over-promises:

| | isqrtm seed | fragment $T_0$ |
|---|---|---|
| subproblem | one per ATOM (block row) | one per CORE BIN (~16 atoms in diamond at 4.5 Å) |
| neighbourhood | $S$'s own row pattern — one hop, fixed by $r_c$ | geometric ball of radius `fragment_buffer` — multi-hop, tunable |
| local operator | $S[\mathrm{nbr},\mathrm{nbr}]$, ONE sparse matrix | $A_{FF}$, a compression of TWO ($Z$ and $H$) on one index set |
| function | $x^{-1/2}$ | the sqrt-filter $p_k$, all tables from one solve |
| kept | that atom's block row | the core's rows |
| the error | REMOVED later by Newton–Schulz; the seed only sets the step count | never removed; absorbed statistically by the sampled correction |

The last row is the substantive one. The isqrtm seed is an initial guess that
a global iteration cleans up, so its locality error disappears from the
answer. $T_0$ is never refined — its error is permanent and is paid for in
variance. Same locality argument, different currency, which is the same
distinction §10's opening draws between probing and the control variate.

### 10.2 Gauge, measured behaviour, and the safeguards

**The gauge lesson (measured, not hypothetical).** The surrogate must
compress the SAME $(Z, H)$ pair the recursion iterates,

$$ T_0 = Z[:,F]\; p\!\big(\, (Z[:,F])^H H\, Z[:,F] \,\big), $$

never an idealized $S_f^{-1/2}$-based fragment pencil: the isqrtm $Z$
satisfies $Z^H S Z = I$ to tolerance while differing from the symmetric
$S^{-1/2}$ by a near-orthogonal gauge factor $Q$. The map is $Q$-invariant,
but the per-sample states are not — an $(H,S)$-built surrogate leaves a
$|Q - I|$-sized residual the buffer cannot remove (measured floor ~1e-3 on
the chain fixture). The $(Z,H)$ compression is exact in the full-fragment
limit by construction (measured 2.3e-14).

**Measured, diamond sc444 density, 64 samples, ~50 s either way:**

| buffer | pointwise rel std | rms of peak |
|---|---|---|
| off | 16.2% | 6.93% |
| 3 Å | 2.5% | 1.05% |
| 4 Å | 0.7% | 0.27% |
| 5 Å | exact (cell-sized fragment) | 0.00% |

`correction_rms` in the task output is the residual's own magnitude — the
free a-posteriori indicator of what the samples still carry.

Scope and honesty: the build is defined over the global atom list and runs at
any rank count (it replicates O(nnz) block rows for the duration of the build;
the O(boundary) halo gather that would replace that at extreme scale is the
recorded extension point in `BuildFragmentCv`); sandwich form only, enforced by
a guard in `FilteredSquareMap` — see "the operator form the surrogate assumes"
below for what the solve form would take; the gain shrinks as the filter
delocalizes (fine-η LDOS pays ξ(η) = ħv/η exactly as §3.3b prices it — no linear-scaling
method escapes that regime).

**Regression coefficient (the do-no-harm guarantee).** The raw control
variate (β = 1) assumes the surrogate tracks the estimator sample for
sample. For sharp spectral filters it does not: a fragment's spectrum near a
probed energy is boundary-modified, and at the diamond sc888 band edge the
β = 1 estimator was 4× *worse* than plain sampling. The assembly therefore
uses $\hat n = \bar X - \beta(\bar Y - n_0)$ with
$\beta = \mathrm{Cov}(X,Y)/\mathrm{Var}(Y)$ estimated from the same samples —
the textbook regression control variate — with the covariance accumulated
**per grid point** before the spatial sum (the covariance of the spatially
integrated series reads ≈1 for any surrogate and is exactly the wrong
statistic; this was measured, not hypothesized). β is reported per channel
(`control_beta`): 1.0 for the density, 0.03–0.05 for the sc888 band-edge
LDOS, matching the post-hoc optimum against the dense reference, and the CV
is never worse than the plain estimator on any system. Measured sc888 (64
fragments, buffer 4 Å, 64 samples, 39 min vs 28 min plain): density 17.0% →
2.4% pointwise; LDOS 17.7/19.6/15.8% → 17.7/18.6/15.8%. Sharp-η full-cell
LDOS remains priced by §3.3b — the fragments simply decline to pretend
otherwise.

**Rank independence (2026-08 addendum, continued).** Three separate
partition dependencies were found and addressed. (1) The per-rank
`mt19937` seed streams made the SAMPLED estimate redraw under a different
rank count — measured 193% of peak between np=1 and np=2 on the sc444
density at 16 samples. Replaced by counter-based Rademacher draws keyed on
(seed, GLOBAL row, state index) — the construction class of Salmon et al.
(SC'11) — so every rank evaluates the identical sign without communication.
(2) The fragment build is now defined over the GLOBAL atom list from
replicated block rows: the surrogate operator is the same at any rank
count, and a core spanning ranks is built by each owner for its own rows.
(3) What remains is the isqrtm gauge of Z itself, which is
partition-sensitive upstream of any estimator; it moves finite-sample
values within their own statistical error (measured 5.8% of peak at 16
samples WITH the control variate — 33× less than the plain estimator's
sensitivity, because the deterministic baseline is partition-free). Exact
bitwise invariance across rank counts therefore requires a
partition-deterministic Z. The solve form is the closer configuration but not
a cure, and the distinction is worth stating precisely: its OPERATOR, S^-1 H,
carries no gauge at all, and its seed covariance is exactly S^-1 for any
invertible Zt (metric_sampler.hpp, Prop 4.7), so its EXPECTATION is gauge-free
where the sandwich map inherits ||Z^H S Z - I|| as bias. Its SEEDS, however,
do pass through Zt: c0 = Zt q(W) xi, and under Zt -> Zt Q the seed is simply
rotated, c0 = Zt q(W) (Q xi). Finite-sample values therefore move under a
gauge change exactly as the sandwich form's do. Derived, not measured.

**The operator form the surrogate assumes (and what S^-1 H would take).** The
control-variate identity itself needs only two things from the operator form:
a seed with identity covariance that both branches can see, and some operator
T_eff with X = |Phi T_eff xi|^2. BOTH hold in the solve form, where
T_eff = F(S^-1 H) Zt q(W) and the raw Rademacher xi is drawn before the
sampler (the fragment side rebuilds it from the same counter keys, never from
c0). So Y would stay unbiased for n_0 there and the assembly would stay
unbiased for n. The guard is not protecting the identity; it is protecting the
surrogate's RELEVANCE, which is a variance question.

How relevant would the existing T_0 be? Write Zt = S^{-1/2} Q with
Q^H Q = W = Zt^H S Zt. When W = I the sandwich chain is S^{-1/2} p(A) Q and
the solve chain is S^{-1/2} F(A) Q with A = S^{-1/2} H S^{-1/2}: the SAME state,
seed for seed, not merely the same map. The two therefore differ by
O(||W - I||), which is `MetricSampler::residual_norm()` and is the same isqrtm
level that floors an (H,S)-built surrogate at ~1e-3. Enabling the present
surrogate for the solve form would thus reduce variance down to that floor,
while losing the exact full-fragment limit (it would bottom out at ||W - I||
plus the CG tolerance of each matvec) — so the 2.3e-14 regression above needs a
different threshold there, and that threshold is the acceptance criterion.
Derived, not measured; not implemented.

A gauge-EXACT solve-form surrogate has to compress all three factors, not one:
T_0 = [F(S^-1 H)]_frag * Zt[:,F] * q(W_FF), with W_FF = (Zt[:,F])^H S Zt[:,F]
carrying the sampler's own coefficient table (a fixed global polynomial, so
evaluating it on the fragment compression is well defined), and
[F(S^-1 H)]_frag taken from the generalized fragment problem H_FF v = S_FF v a
(`dsygvd`) — never from a nonsymmetric (S^-1 H)_FF. That costs one more dense
solve per fragment and one more `replicate` call for S, which the build's
lambda already supports generically. The interval-containment argument
survives the change: a generalized Rayleigh quotient of (H_FF, S_FF) is the
Rayleigh quotient of a zero-padded GLOBAL vector, so the fragment eigenvalues
still sit inside spec(S^-1 H) and the [-1, 1] clamp stays valid. Everything
downstream is form-agnostic — ApplyIdentity's stored-column trick is a property
of the storage, and the per-point beta, the per-box shrinkage and
`correction_rms` never look at the operator.

Position: not planned. The sandwich form is strictly cheaper for maps (no
sampler construction, no CG inside the recursion), and the solve form's real
advantage is bias, not variance — and if ||W - I|| is large enough for that
bias to matter, it is large enough to floor the control variate too. The fix
in both cases is converging Zt harder, not switching form.

**Energy windows (ew-sDFT / ew-efsDFT).** `energy_windows` on the density
task splits the occupation into Fermi-bounded windows AT THE SAME kT,
filtered and squared separately from the same sweep and seeds, maps summed.
Cross-window sampling variance is removed exactly; by Cauchy–Schwarz the
windowed variance can never exceed the plain one, and each window channel
carries its own fragment surrogate and regression coefficient — the hybrid
of Chen et al. (2021), with the regression safeguard on top. The telescoping
identity holds to the depth of the lowest window edge below the spectrum
(30 kT → e⁻³⁰, measured 4.7e-13 of peak; at 10 kT the deficit is exactly
f(e_min − 10kT) = e⁻¹⁰ ≈ 4e-5, which is how it was caught). This is the one
reduction that keeps working where fragments cannot — the published metals
benchmark (arXiv:2403.04203) measures fragments useless and windows worth
~2× at 300 K — so the composed estimator degrades gracefully across the
localization spectrum instead of failing with the material.

**Theoretical headroom, and the two levers that remain (2026-08).** The map
is a deterministic quantity, so there is no statistical floor in principle;
the constraint is computational. With S filtered applications plus O(1)-local
deterministic work, the achievable pointwise error interpolates between
Monte-Carlo sqrt(2/n) (S = n) and probing exactness (S = χ(η) ∝ η⁻³ seeds,
§5). Every published family sits on this curve: dilution / hierarchical
probing moves along it linearly (the pair-suppression scale for the LDOS is
r_c + ξ(η), re-derived — no shortcut); deflation is blocked by the extensive
window rank; sub-η energy windows cost degree ∝ N_w. MLMC degenerates to our
two-level scheme when level zero is computed exactly, which it is (the
baseline). Within the class, the regression coefficient already realizes the
variance-optimal mixing of the implemented controls (fitted β matches the
post-hoc optimum). The remaining recoverable gap is SURROGATE QUALITY below
the locality ceiling β_max(η): (i) spatially resolved β per grid box with
shrinkage to the global fit — implemented; strict generalization, its gain
appears for heterogeneous cells (defects, interfaces) where one global
coefficient under-serves every region; (ii) boundary-state screening — damp
the compression eigenvectors by their boundary weight (keeping the
gauge-exact Z[:,F] factors), removing the anti-correlated fragment boundary
spectrum that the measured LDOS failure consists of; bounded gain, roughly a
doubling of β at η ≳ 0.5 eV, nothing at η = 0.25 where ξ(η) exceeds any O(1)
fragment. Beyond both, better sharp-η accuracy at fixed cost means leaving
the class: probing (§5), region mode (§3), or the k-route for crystals.

**Cost of the surrogate itself (2026-08, efficiency pass).** The first
implementation paid three structural overheads that the accounting above
never required, measured 632 s of the 39-minute sc888 run (fragment_build
484 s, fragment_baseline 148 s — the entire 1.38× premium over the plain
estimator). All three are removed exactly — same arithmetic, same maps to
floating-point reordering (measured 2.5e-14 relative on the recorded np=2
fragment run):

1. *Fragments are the parallel grain.* The dense per-fragment solves ran in
   a serial loop, leaving the machine to threaded BLAS whose small GEMMs and
   memory-bound tridiagonalization cannot use it. The build now threads over
   fragments (one dense solve per thread, per-thread scratch marker arrays;
   BLAS runs sequential inside the active parallel region — MKL detects the
   nesting), and the domain search dropped its all-atoms scan — O(N²) over
   fragments — first for a hand-rolled cell list, then (rightly) for the
   library's own vbcsr::atomic::NeighborList on the replicated global
   positions: one build shared by every fragment, image-correct for sheared
   cells where a bare minimum-image round is not, and with the cutoff
   inflated by the same absolute 1e-9 Å tolerance AtomicData's own graph
   construction applies, so the buffer stays INCLUSIVE — a round
   buffer on a lattice with round spacings sits atoms exactly on the
   boundary, and the strict comparison silently dropped that shell (the
   chain fixture's correction tripled; with the inclusive convention it is
   marginally better than the original, whose fractional round-trip lost
   exact-boundary atoms to 1e-16 rounding). The cell inverse likewise comes
   from vbcsr::atomic::InvertCell3x3 instead of a private adjugate. Output
   slots are preallocated in geometric bin order, so the result is
   schedule-independent.

2. *T₀'s identity columns are stored numbers, not products.* The exact
   baseline streamed identity blocks through the same dense Apply the
   samples use — a GEMM against seeds that are all zeros but for one entry
   per column, whose product is literally a stored column of P. ApplyIdentity
   scatters those columns directly (binary search of the sorted domain rows
   against the block range); the baseline's flops fall to the write itself.

3. *The baseline contraction must not pay for zeros.* Each identity-block
   pass contracted every owned box against all columns, though a column of
   T₀ reaches only the cores of fragments holding it — the stream cost grew
   as (N/blk blocks) × O(N) boxes = O(N²), the one superlinear term in the
   method. SquareVectorsOntoBoxes now detects each box's active columns
   (one pass over rows already gathered; dense sample blocks settle the scan
   after a row or two and take the unchanged path), compacts the GEMM to
   them, and skips untouched boxes before the orbital evaluation. Active
   (block, box) pairs are set by the overlap geometry, not the basis size —
   the stream is O(N) again, and the wider block (512) amortizes the per-box
   orbital evaluation across passes.

Measured after the pass, same physics bit-for-bit at the tolerance above:
sc444 density (buffer 4 Å) fragment_build 1.64 s, baseline 0.29 s against a
41 s sweep — the control variate now costs ~2 s on a 49 s run, cheaper than
the plain estimator's own noise.

**The build is the submatrix method, and now runs on its machinery
(2026-08, second pass).** Because the fragment's variational space is
spanned by full-height columns of Z, the Rayleigh compression is exactly
the principal submatrix of the sandwich operator, A_FF = M[F, F] with
M = Z^T H Z — each fragment is one submatrix-method subproblem [Lass et
al. 2018; CP2K 2020; NOLSM, Schade et al. 2022] with a multi-atom anchor,
an arbitrary filter table instead of sign/isqrt, and (unlike NOLSM, which
recomputes a local S^{-1/2} per submatrix and would hit the measured 1e-3
gauge floor) prolongation through the true global Z columns. The build
therefore now shares the graph seed's infrastructure: only Z's PATTERN with
per-block Frobenius weights is replicated globally (~20 bytes/block — the
partition-free fragment definition needs exactly that and no more), and the
block payloads are fetched per batched fragment window through
fetch_batched_block_payloads, one dense solve per thread. The dense
nrr × nrr intermediates of the first pass are gone too, and how they went
is a lesson in kernel shape — four variants of the SAME arithmetic,
sc888 build (2 calls), all producing identical maps:

| window-product kernel | build | peak |
|---|---|---|
| dense nrr² windows (replicated payloads) | 133 s | 123 GB |
| per-block 13×13 GEMMs | 910 s | 52 GB |
| single-block-row panels (m = 13) | 3710 s | 46 GB |
| chunked panels (m ≈ 512) — SHIPPED | 246 s | 54 GB |

The 1e-3-trimmed Z window at sc888 spans ~850 of 1024 atoms — nearly
dense — so tiny GEMMs with strided output run at a tenth of BLAS3 speed;
single-row panels are worse still because every m = 13 GEMM re-streams
the full nrr × nf right-hand side (1.6 flops/byte across 64 threads is a
bandwidth wall, not a compute kernel); and the full dense window sits one
buffer increment from the 125 GB machine limit. The chunked panel is the
dense GEMM evaluated in ~512-row slabs: dense-BLAS3 arithmetic intensity,
O(chunk × nrr) scratch, never nrr². Numbers are unchanged throughout: the
same fragments, the same trim, the same correction_rms to the printed
digit.

One benchmarking trap, chased to ground: across this work the untouched
Chebyshev sweep read ~1815 s (five consecutive runs) against the recorded
1467 s, with bit-identical outputs — and the cause was neither code nor
machine state but THREADS. `rsatb run` with OMP_NUM_THREADS unset uses
nproc = 96 on this 48-core SMT box; 2× oversubscription costs a
bandwidth-bound spmm ~25% while leaving cache-resident sc444 unharmed
(which is what made it look environmental). With OMP_NUM_THREADS=48,
OMP_PLACES=cores, OMP_PROC_BIND=spread the sweep measured 1450 s. Final
clean sc888 whole-run at proper threading: 33m46s against the recorded
39m14s of the pre-optimization control variate — sweep unchanged (1450 vs
1467 s), fragment_build 484 → 261 s, fragment_baseline 148 → 73 s, build
peak 45 GB where the dense-window route needed 123 GB.

**Smeared spectral weights: tried, measured, removed (2026-08).** The
submatrix view suggests a refinement: each Ritz state's energy variance
under M is exactly computable in the window (sigma_j² = ||M[:,F] v_j||² −
a_j²), and evaluating the filter Gauss-Hermite-smeared over N(a_j, sigma_j)
is the L2-optimal diagonal weight IF the state's spectral measure were
Gaussian. Measured: density correction 20× WORSE (0.0053 → 0.106 at sc444
b4.0), and the LDOS gains erased at every eta (eta=1.0: 14.5/15.1/16.0%
back to 16.3/17.1/17.7% against the 17.7% plain floor; eta=0.5: back to
plain). The mechanism, understood after the fact: the spectral measure of a
spatially truncated state is a dominant peak plus low-weight far tails,
sigma² is tail-dominated (Heisenberg broadening of the truncation, eV
scale), and a Gaussian moves the PEAK's weight by a width the peak does not
have. The regression coefficient kept every smeared run no worse than plain
— the designed safety worked — but point evaluation p(a_j) won everywhere
measured and is again the only weight. Recorded next attempt if sharp-eta
surrogate quality is ever worth another round: a peak-plus-tail model with
second-order embedding weights w_g = |(M_GF v_j)_g|² / (M_gg − a_j)² from
the window diagonal, which respects the measure's actual shape where the
Gaussian cannot.

## 11. Subspace deflation for sharp-filter maps (2026-08-25)

The sharp-eta LDOS lever: the filtered sample states double as a
spectrum-slicing sketch of the energy window, the SQUARED filter rides the
same recursion so the sketch's true filtered image is exact, and the
estimator splits into a deterministic captured map plus a cross-fitted
deflated remainder -- unbiased for any capture, never worse than plain
pointwise, no fitted coefficient, no extra matvecs. Full self-contained
definitions, derivations, algorithm, cost accounting, and the
system-size scaling law live in the dedicated document:

    doc/design_subspace_deflation.md

Headline measurements (64 samples, eta = 0.25 eV, vs eta-matched
references): sc444 maps exact to filter leakage (1e-9..5e-7 relative;
plain 7-15%); sc888 gap 19.6% -> 0.02%, conduction edge 15.8% -> 1.5%,
deep valence 17.7% -> 7.5% (rank-saturated); cost +9% wall at sc444,
0.3% of the sc888 LDOS task; np1-vs-np2 at the isqrtm-gauge level.

The one remaining structural extension point: sharing the fragment
eigensolves across tasks in one run (each task still builds its own tables
from the same decomposition).
