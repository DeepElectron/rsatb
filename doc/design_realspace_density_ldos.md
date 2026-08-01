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
`release_overlap_after_build` discarded the overlap.

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
count. Verified: `BindsPointsToTheAtomsTheyLieOn` at 2, 3, 4, 8 and 16 ranks,
plus determinism, balance, and the degenerate case of more ranks than atoms.

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
