# Position Commutators and Derivative Lifts

How RSATB evaluates quantities that need the position operator $\hat x$ under
periodic boundary conditions, where $\hat x$ has no matrix representation at
all. The mean-square displacement is the case that forces the issue, but the
same construction is what the velocity operator, the AC and DC conductivities
and the quantum metric are all built from, so this note states it once in
general form and then points at each consumer.

Companion to [formula.md](formula.md) (non-orthogonal conventions, KPM, Kubo)
and [velocity_operator.md](velocity_operator.md) (the two-centre integrals).

---

## 1. The obstruction

The Einstein form of the Kubo conductivity needs the spread of a wavepacket,

$$
\Delta X^2(E,t)=\frac{\langle\varphi(t)|\,\delta(E-\hat H)\,|\varphi(t)\rangle}
{\langle\psi|\,\delta(E-\hat H)\,|\psi\rangle},
\qquad
|\varphi(t)\rangle=\big[\hat x,\,U(t)\big]|\psi\rangle,
$$

with $U(t)=e^{-i\hat Ht/\hbar}$. Written this way it asks for $\hat x$
explicitly, and under periodic boundary conditions $\hat x$ is not a periodic
operator: it grows without bound across cells and has no representation in a
lattice basis. Truncating it to a "sawtooth" introduces an artificial
discontinuity; going to open boundaries throws away the periodicity that the
supercell exists to provide.

The resolution is that **the commutator is bounded even though the operator is
not**. Everything below is a way of never forming $\hat x$ while still getting
$[\hat x, f(\hat H)]$ exactly.

## 2. The principle

$\hat x\mapsto[\hat x,\cdot]$ is a *derivation*: it is linear and obeys the
Leibniz rule $[\hat x,AB]=[\hat x,A]B+A[\hat x,B]$. A derivation applied to a
function of an operator gives the Fréchet derivative of that function,

$$
\big[\hat x,\,f(\hat H)\big]=L_f\!\big(\hat H,\ [\hat x,\hat H]\big),
$$

where $L_f(H,E)=\left.\frac{d}{ds}f(H+sE)\right|_{s=0}$. The only input on the
right is $[\hat x,\hat H]$, which **is** a bounded lattice operator: the
position difference across a hopping is the bond vector, and the bond vector is
bounded by the interaction range. The unboundedness cancels in the commutator
before anything is ever assembled.

Two equivalent machines evaluate $L_f$. They are the same recursion written in
two notations, and RSATB currently uses the second.

### 2a. Commutator recursion

Expand $f$ in Chebyshev polynomials and apply the derivation to the recurrence
$T_{m+1}=2\tilde HT_m-T_{m-1}$:

$$
\big[\hat x,T_{m+1}(\tilde H)\big]
=2\Big(\big[\hat x,\tilde H\big]T_m(\tilde H)+\tilde H\big[\hat x,T_m(\tilde H)\big]\Big)
-\big[\hat x,T_{m-1}(\tilde H)\big].
$$

Carrying $\psi_m=T_m(\tilde H)\psi$ and $\varphi_m=[\hat x,T_m(\tilde H)]\psi$
together:

$$
\boxed{
\begin{aligned}
\psi_{m+1}&=2\tilde H\psi_m-\psi_{m-1}, &\psi_0&=\psi, &\psi_1&=\tilde H\psi,\\
\varphi_{m+1}&=2\big(\tilde H\varphi_m+\tilde V\psi_m\big)-\varphi_{m-1},
&\varphi_0&=0, &\varphi_1&=\tilde V\psi,
\end{aligned}}
$$

with $\tilde H=(H-c)/w$, $\tilde V=[\hat x,H]/w$, $c$ the spectral centre and
$w$ the half-width.

### 2b. Nilpotent lift

Equivalently, form the block operator on pairs

$$
\mathsf M=\begin{pmatrix}\hat H & \hat V\\ 0 & \hat H\end{pmatrix},
\qquad \hat V=[\hat x,\hat H],
$$

for which any matrix function is block upper-triangular with the Fréchet
derivative in the corner:

$$
f(\mathsf M)=\begin{pmatrix}f(\hat H) & L_f(\hat H,\hat V)\\ 0 & f(\hat H)\end{pmatrix}.
$$

This is the classical 2×2 trick for derivatives of matrix functions (Higham,
*Functions of Matrices*). Because $\mathsf M$ is $\hat H\otimes\mathbb 1+\hat
V\otimes\varepsilon$ with $\varepsilon^2=0$, it is arithmetic in the dual
numbers $\mathbb R[\varepsilon]/(\varepsilon^2)$ — forward-mode automatic
differentiation, with $\varepsilon$ tracking one derivative.

**RSATB's affine representation is this lift.** `GammaAffineState` stores the
pair $(a,b)$ of a translation-covariant state $u_R=a+Rb$, and
`propagate_affine` iterates

$$
\mathsf H_{\rm aff}=\begin{pmatrix}H_0 & H_1\\ 0 & H_0\end{pmatrix},
$$

with $H_1$ the first lattice moment. Applying $\hat x$ to a cell-periodic
$\psi$ gives the affine pair $(\hat d\psi,\ \psi)$, propagating gives
$U(t)\hat x\psi$, and the $R$-linear parts cancel in the commutator, leaving a
cell-periodic result. That cancellation is the affine-picture statement of the
same fact as $\varphi_0=0$ above.

## 3. The driver operator, orthogonal basis

One symbol carries the whole construction, so fix it once, in the convention
the code stores. The **bond moment** of a lattice operator $O$ with images
$O(R)$, along Cartesian direction $\alpha$, sampled at $k$ in the atomic gauge
($s_{ji}=f_j-f_i$ the fractional intra-cell displacement, the same phases
`sample_k` gives $H$):

$$
\mathcal M_\alpha[O](k)_{ij}
\equiv\sum_R\big(R_\alpha+r_{j,\alpha}-r_{i,\alpha}\big)\,O_{ij}(R)\,
e^{-2\pi ik\cdot(R+s_{ji})}
\;=\;i\,\partial_{k_\alpha}O(k).
$$

The second equality is the reason this object matters: differentiating the
phase pulls down $-i(R+r_j-r_i)_\alpha$, so **bond-vector weighting is
$k$-differentiation** — no finite differences, no second sampling.

In an orthogonal basis the position is diagonal with entries $r_i$
(diagonal_position.hpp), and the commutator closes on each image block: the
bra orbital of block $(i,j,R)$ sits at $r_i$, its ket at $r_j+R$, so

$$
[\hat H,\hat x_\alpha]_{ij}(R)
=\big(r_j+R\big)_\alpha H_{ij}(R)-r_{i,\alpha}H_{ij}(R)
=\big(R+r_j-r_i\big)_\alpha H_{ij}(R),
$$

i.e. $[\hat H,\hat x_\alpha]=\mathcal M_\alpha[H]$ and the stored velocity is

$$
V_\alpha(k)=i\,\mathcal M_\alpha[H](k)=-\,\partial_{k_\alpha}H(k)
\qquad(\hbar=1;\ \text{eigenbasis diagonal }=-\,\partial E_n/\partial k_\alpha).
$$

Every hopping contributes its own displacement; the weight is bounded by the
hopping range even though $\hat x$ is unbounded — the §1 obstruction resolved.
The on-site block has zero weight and drops out, which is why a dispersionless
model has no velocity. This is exactly what `operator/commutator_velocity.hpp`
assembles (its header carries the same derivation), and its **bare matrix** is
valid only for an orthogonal basis — the overlap-dressed case is §4.

(Older revisions of this note used $\mathcal D=-\mathcal M$, weighting by
$(r_i-r_j-R)$; every formula below is now written in $\mathcal M$.)

## 4. Non-orthogonal basis

With overlap $S$, operator products compose as $\hat A\hat B\to AS^{-1}B$
(formula.md, *Composition of Operators*) and the effective generator on
coefficient vectors is $A_0=S_0^{-1}H_0$. Two things change relative to §3,
and each contributes one term.

**Ingredient 1 — the pencil derivative.** The velocity is still a
$k$-derivative, but now of $S^{-1}H$, and the metric varies with $k$ too.
The product rule, with $\mathcal M=i\partial_k$ from §3:

$$
i\,\partial_k\big(S^{-1}H\big)
=S_0^{-1}\big(i\partial_kH\big)-S_0^{-1}\big(i\partial_kS\big)S_0^{-1}H
=S_0^{-1}\big(\mathcal M[H]-\mathcal M[S]\,A_0\big).
$$

Dropping $\mathcal M[S]$ is the same error as differentiating $S^{-1}H$ while
treating $S$ as constant.

**Ingredient 2 — the intra-atomic dipole.** In §3 the position was exhausted
by the atom coordinates. With extended orbitals it is not: split $\hat x$
about each **ket** orbital's own centre $r_j+R$,

$$
X_{ij}(R)=\langle\phi_{i0}|\hat x|\phi_{jR}\rangle
=\underbrace{(R+r_j)\,S_{ij}(R)}_{\text{which atom, and where}}
+\underbrace{\Delta_{ij}(R)}_{\text{ket-centred dipole}},
$$

so the Bloch-summed dipole is $\Delta(k)=X_0(k)-\mathcal K[S](k)$, with
$\mathcal K[S]_{ij}=\sum_R(R+r_j)S_{ij}(R)e^{-2\pi ik\cdot(R+s_{ji})}$ the
ket-position moment. $\Delta$ is bounded and short-ranged — the unbounded part
of $\hat x$ went entirely into the monopole piece, which ingredient 1 already
carries through the bond weights. $\Delta$ is *new information*: the charge
displacement inside an orbital, invisible to any point-charge bookkeeping.

**The operator.** The dipole enters as a commutator — moving in $k$ rotates
the basis representation as well as the Hamiltonian, and a change of
representation acts by similarity, whose generator is always
$[\,\cdot\,,A_0]$ (the same structure as a Berry connection). In the stored
matrix convention (§3: eigenbasis diagonal $=-\,\partial E_n/\partial k$):

$$
\boxed{\;
V_\alpha(k)=i\Big(\mathcal M_\alpha[H]-\mathcal M_\alpha[S]\,A_0
\;-\;S_0\big[\,S_0^{-1}\Delta_\alpha,\ A_0\,\big]\Big).}
$$

**Checks.**
* *Orthogonal limit*: $\mathcal M[S]=0$, and $X_0=\mathrm{diag}(r)$ with
  $\mathcal K[S]=\mathrm{diag}(r)$ gives $\Delta=0$, so $V=i\,\mathcal M[H]$ —
  §3 recovered.
* *Diagonal*: with $A_0c_n=E_nc_n$ and $c_n^\dagger H_0=E_nc_n^\dagger S_0$,
  the commutator's diagonal cancels identically,
  $c^\dagger S_0[S_0^{-1}\Delta,A_0]c=E\,c^\dagger\Delta c-E\,c^\dagger\Delta c=0$,
  leaving $c^\dagger Vc=i\,c^\dagger(\mathcal M[H]-E\,\mathcal M[S])c
  =-\,\partial E_n/\partial k$ — Hellmann–Feynman for the pencil, exactly.
  Band velocities never see the dipole; its entire effect is interband.
* *Expanded form* (what the code applies; types.hpp, §8): with
  $\bar X\equiv X_0-\mathrm{diag}(r)S_0=\mathcal M[S]+\Delta$,
  $$Vx=i\big(\mathcal M[H]\,x-\bar X\,S_0^{-1}H_0\,x+H_0\,S_0^{-1}\Delta\,x\big),$$
  the same operator with the commutator multiplied out — three sparse pieces
  and two metric solves, never a materialized $S^{-1}$.

**The Γ-point lattice-moment form.** At $\Gamma$ the same operator can be
regrouped around lattice moments $O_1\equiv\sum_RR\,O(R)$, using
$\mathcal M[O]|_\Gamma=O_1-[\mathrm{diag}(r),O_0]$:

$$
V\big|_\Gamma=-\,i\,S_0\,C,\qquad
C=-S_0^{-1}\big(H_1-S_1A_0\big)
+\Big[\,S_0^{-1}\big(X_0-S_1\big),\ A_0\,\Big],
$$

which is how the affine machinery of §2b states its driver
($C=[\mathcal X,A_0]$ on coefficients, $\mathcal X=S_0^{-1}X_0$-dressed
position). The two groupings differ by $[\mathrm{diag}(r),\cdot]$ terms that
cancel between the cell and dipole pieces — verified symbolically and, at
general $k$ against SK models where both routes are the same operator, to
$10^{-16}$ (§7). Getting this regrouping wrong (bond moments paired with the
full $X_0$) double-counts the displacement; §7's forensics caught exactly
that.

Two consequences worth stating plainly:

* The new content relative to the LSQT literature is the **overlap moment**
  $\mathcal M[S]$ (equivalently $S_1$) and the **dipole commutator**: the
  metric varies from cell to cell, and the orbitals carry internal dipoles.
  §7 measures that neither is a small correction — the dipole term alone
  carries half the operator, and dropping it inflates the f-sum by three
  orders of magnitude.
* $V$ is assembled with **no gradient integrals and no β projectors**.
  $\mathcal M[H]$, $\mathcal M[S]$, $\mathcal K[S]$ need only $H(R)$, $S(R)$
  and geometry; $X_0$ is the one ingredient requiring real orbital integrals,
  and they are plain two-centre dipoles (`overlap_position`), not the
  pseudopotential machinery of the explicit-integral route. This is why LCAO
  now has a velocity **unconditionally** — `.upf` files are needed only for
  the op2c cross-check operator — while a non-orthogonal **SKTB/DFTB** model
  still has none: no orbital shapes, no $X_0$, and the dipole cannot be
  dropped.

## 5. Spectral weight

Whether $\delta(E-A_0)$ needs a metric depends on whether the observable is a
**trace** or a **matrix element on a specific state** — and the two behave
oppositely.

**Traces need no metric.** $A_0=S_0^{-1}H_0$ has spectral projectors
$c_nc_n^\dagger S_0$ (the eigenvectors are $S$-orthonormal), so inside a full
trace the weights close on themselves,
$\mathrm{tr}[c_nc_n^\dagger S_0]=c_n^\dagger S_0c_n=1$, and

$$\mathrm{Tr}\,\delta(E-A_0)=\sum_n\delta(E-E_n)=\rho(E)$$

exactly — plain $\delta(E-S_0^{-1}H_0)$ or $\delta(E-Z^HH Z)$, plain
Euclidean stochastic estimation. This is why the DOS never sees $S_0$.

**State-resolved quantities do.** The MSD (and the LDOS) evaluates
$\langle\varphi|\delta(E-\hat H)|\varphi\rangle$ on one state, whose
physical weight is $|\langle n|\varphi\rangle|^2=|c_n^\dagger S_0c|^2$ —
both $S$'s. Sandwiching the projectors shows the correct coefficient-space
expression carries one explicit metric,

$$
\mu_n=\langle\varphi|\,S_0\,T_n(A_0)\,|\varphi\rangle,
\qquad
\Delta X^2(E,t)=\frac{\langle\varphi(t)|S_0\,\delta(E-A_0)|\varphi(t)\rangle}
{\langle\psi|S_0\,\delta(E-A_0)|\psi\rangle},
$$

while the plain Euclidean $c^\dagger\delta(E-A_0)c
=\sum_n\delta(E-E_n)(c^\dagger c_n)(c_n^\dagger S_0c)$ has the $S$ on one
side only — not real in general, let alone a weight. (Equivalently:
$A_0$ is self-adjoint in the $S$ inner product, not the Euclidean one.)

**The sandwich form makes the metric disappear — by construction, not by
luck.** With $c=Z\eta$ and $Z^HS_0Z=\mathbb 1$,

$$\langle\varphi|\delta(E-\hat H)|\varphi\rangle
=\eta^\dagger\,\delta(E-Z^HH Z)\,\eta:$$

plain η-space dots *are* the physical inner product, which is exactly the
intuition that $\delta(E-Z^HHZ)$ "should just work" — it does, provided the
state is carried in η-space consistently. An earlier version of this section
claimed the sandwich was unavailable for the periodic route because the lift
would need $Z$'s lattice moment $Z_1$, which Newton–Schulz never produces.
That applied to the OLD affine coupling, which needed the coupling as a
standalone MATRIX in the η representation — and a matrix moment of $Z$ does
not exist. The commutator route (§8) needs the coupling only as an
APPLICATION: `apply_dual` computes $Z^HV(Z\eta)$ at runtime, using $Z$ purely
as a linear map, and lattice moments are taken only of $H$, $S$, $X_0$, which
have image decompositions. Blockwise conjugation is exact
($Z^{-1}S^{-1}HZ=Z^HHZ$, $Z^{-1}S^{-1}VZ=Z^HVZ$ via $ZZ^H=S^{-1}$), so the
η-space pair recursion is the similarity transform of the coefficient one and
the sandwich is fully available. Note the production default is the
**Solve** form (`preconditioner.operator_form = "solve"`); the sandwich is the
opt-in that trades the per-matvec CG for two SpMVs, releases $S$'s memory, and
enables the Hermitian two-moments-per-matvec KPM shortcut — at the price of
carrying the factor's residual $\|Z^HSZ-\mathbb 1\|$ as a spectral error.

Implementation (`metric_density_moments`, `calc/msd.hpp`): plain dots for the
Orthogonal and Sandwich forms, one $S_0$-multiply per spectral readout for the
Solve form. For stochastic traces over the metric ($\mathrm{Tr}[S_0^{-1}\cdot]$-type
estimators, as the real-space LDOS needs), draw samples with covariance
$S_0^{-1}$ via `calc/metric_sampler.hpp`.

## 6. Time evolution and the MSD

With $\tau=wt/\hbar$ and Jacobi–Anger,

$$
U(t)=e^{-ict/\hbar}\sum_{m\ge0}(2-\delta_{m0})(-i)^mJ_m(\tau)\,T_m(\tilde A),
$$

which is `timeprop_coeff`. The scalar phase commutes with $\hat x$, so

$$
[\hat x,U(t)]|\psi\rangle=e^{-ict/\hbar}\sum_m c_m(\tau)\,\varphi_m,
$$

with $\varphi_m$ from §2a driven by $\tilde V=[\mathcal X_\alpha,A_0]/w
=\tfrac{i}{w}S_0^{-1}V_\alpha$ — which is exactly one form-dressed
`apply_dual` of the §4 velocity per Chebyshev step, times $i/w$
(`corr_msd(cheby, velocity, ...)`, msd.hpp). The diffusivity and conductivity
follow as usual,

$$
D(E,t)=\tfrac12\,\partial_t\Delta X^2(E,t),\qquad
\sigma(E)=e^2\rho(E)\,D(E,t\to\infty).
$$

### Why $\varphi_0=0$ is worth having

The orthogonal Γ-affine route (§2b, kept for its cheaper per-step cost) forms
the commutator as $c(t)=D_0\,b(t)-a(t)$, the difference of two propagated
states. In a Γ-only supercell the intra-cell coordinate spans the *whole*
supercell, so both terms grow like the supercell dimension $L$ while their
difference is the physical wavepacket spread. The subtraction loses roughly
$\log_{10}(L/\text{spread})$ digits, worst at short times — exactly where the
ballistic slope is read off. The commutator route of §2a starts at
$\varphi_0=0$ and never forms either term.

Cost is unchanged: both routes apply the generator twice and the driver once
per Chebyshev step.

## 7. The same construction elsewhere in RSATB

| quantity | order in $\hat x$ | what it needs | status |
|---|---|---|---|
| velocity operator | 1 | $V$ (§4) | one production source, the §4 commutator (`kGeneric`): the bare bond moment when orthogonal, the composite when not; op2c's explicit integrals stay on the model (`BuildVelocity`) as a cross-check only |
| AC conductivity (`ac.hpp`) | 1 | $V$ dressed by the operator form | all bases, all forms; `apply_dual` supplies the dressing |
| DC transport (`fftcheb_transport.hpp`) | 1 | same | all bases, all forms |
| MSD / diffusion (`msd.hpp`) | 1, but needs $[\hat x,U(t)]$ | $V$ + the §2a recursion | all bases, all forms; open boundaries via the position matrix, periodic via the commutator route (§8) |
| quantum metric (`observables.hpp`) | 2 | two derivative directions | computed through the AC correlator, so it inherits AC's coverage |
| higher moments $\langle x^2\rangle$ | 2 | a 3×3 lift (below) | not implemented |

Only the MSD needs the *recursion*, because only it needs $[\hat x,f(\hat H)]$
as a state rather than an operator applied once. The first-order quantities
need $V$ itself, which is why they are unaffected by the lift machinery but
**are** affected by §4: any of them evaluated in a non-orthogonal basis needs
the $\mathcal M[S]$ and dipole terms.

### Which velocity operator? Commutator versus explicit integrals

Because the §4 velocity is itself the commutator $i[\hat H,\hat x]$, every
consumer above can take it from images plus the dipole integrals, with no
gradient integrals and no $\beta$ projectors. Whether it *should* — rather
than op2c's explicit momentum integrals — was a real question when this
section was first written, and the honest answer is that both routes carry
error, of different kinds. This subsection is the record of the arguments,
the measurements that settled them, and two real defects the measurements
flushed out along the way.

**The nonlocal term is not lost.** `build_overlap_velocity` needs $\beta$
because it assembles $[\hat H,\hat x]$ *piecewise*: a 2-body gradient part plus
a 3-body `v_comm` term (`orb_r_beta` / `ncomm_IKJ`) for
$[\hat V_{\rm NL},\hat x]$, the nonlocal pseudopotential commutator. It refuses
without projectors rather than silently dropping that term. The
commutator-from-images route never splits $\hat H$, so the nonlocal
contribution rides along inside $H_{ij}(R)$ automatically. This is the standard
argument for preferring a commutator formulation when nonlocal potentials are
present, and it is why route (b) below needs no projectors.

**What it trades into is basis incompleteness.** The composition rule
$\langle\phi_i|\hat A\hat B|\phi_j\rangle=(AS^{-1}B)_{ij}$ inserts a resolution
of identity that is a projector $P$ onto the basis span, not $\mathbb 1$. So
the commutator route evaluates $XS^{-1}H-HS^{-1}X$ where the true matrix
element is $\langle\phi_i|(\hat x\hat H-\hat H\hat x)|\phi_j\rangle$, and the
difference is

$$
\langle\phi_i|\,\hat x\,(1-P)\,\hat H\,|\phi_j\rangle
-\langle\phi_i|\,\hat H\,(1-P)\,\hat x\,|\phi_j\rangle,
$$

the part of the operator product that leaves the basis. It vanishes for a
complete basis and is not small for a minimal one. The explicit-integral route
does not have this error, because it evaluates each product as an integral.

**What the commutator route does give, exactly.** For any two eigenpairs of the
finite-basis problem $HC_n=E_nSC_n$,

$$
C_n^\dagger\big(XS^{-1}H-HS^{-1}X\big)C_m=(E_m-E_n)\,C_n^\dagger XC_m
$$

holds exactly — the length-gauge/velocity-gauge relation, within the basis. The
commutator velocity is therefore precisely the one consistent with the model's
own spectrum and its own dipole matrix.

That reframes the question from "which is more accurate" to "accurate with
respect to what":

* **AC, DC and MSD** are spectral functions of the *same* finite-basis $H$ that
  the Chebyshev recursion propagates. Consistency with that $H$ is what makes
  the discrete sum rules hold — the f-sum rule, and the Kubo–Greenwood relation
  between $\sigma(E)$ and the DOS. The explicit-integral velocity is closer to
  the continuum operator but is *not* the exact commutator of the propagated
  $H$, so it can violate them. For these observables the commutator route is
  arguably the more appropriate one, not merely the cheaper one.
* **Absolute oscillator strengths**, compared against experiment or an
  all-electron reference, are the case where the explicit integrals are the
  better estimate and basis incompleteness is the error that matters.

**Three tiers, of which two are defensible:**

| | route | needs | error |
|---|---|---|---|
| (a) | explicit $\nabla$ + $\beta$ 3-body (op2c `BuildVelocity`, cross-check only) | pseudopotentials | none from incompleteness; not the exact commutator of the finite-basis $H$; and, **as first built**, not Bloch-periodic away from $\Gamma$ — since repaired, see the forensics below |
| (b) | the §4 commutator with exact $\mathcal M[H],\mathcal M[S],X_0$ — **the production operator** (`kGeneric`) | images **+ orbital dipole integrals $X_0$**; no pseudopotentials | basis incompleteness |
| (c) | pencil derivative only, dipole dropped ($\Delta=0$) — what a model without orbital shapes could build | images only | (b) **plus** the dropped intra-atomic dipole — measured fatal in a non-orthogonal basis, see below |

Route (c) discards $\Delta$, which §4's checks show is invisible to band
velocities but carries the entire interband weight; the measurement below
puts numbers on "fatal". It is why an overlap-bearing model without dipole
integrals is given **no** velocity rather than a wrong one
(`model/operator_source.hpp` refuses the combination by name), and why route
(b) is the one that removes the projector requirement honestly.

**A resolved inconsistency.** When this section was first written, AC and DC
on an LCAO model took route (a) while every $[\hat x,f(\hat H)]$ quantity was
forced onto the commutator — two velocity operators on one model. The
migration recorded in §8 removed it: route (b) is now the single production
velocity for every consumer, and route (a) survives only as the model-level
cross-check.

**How it was settled.** Not by comparing the two routes against each other —
that can only show *that* they differ — and not by diagonalising a reference
Hamiltonian, which is no referee at all, since
$\langle n|[\hat H,\hat x]|m\rangle=(E_n-E_m)\langle n|X|m\rangle$ holds
exactly in any finite basis. The neutral standard is the Hellmann–Feynman
theorem for the pencil actually being propagated. For
$(H(k)-E S(k))c=0$ with $c^\dagger Sc=1$,

$$
\frac{\partial E_n}{\partial k_\alpha}
= c_n^\dagger\Big(\frac{\partial H}{\partial k_\alpha}-E_n\frac{\partial S}{\partial k_\alpha}\Big)c_n ,
$$

which is variational and assumes nothing about basis completeness. In the
atomic gauge, $\partial_k$ of a Bloch-sampled operator is $-i\mathcal M[\cdot]$
(§3), so the right-hand side is
$-i\,c^\dagger\big(\mathcal M[H]-E\,\mathcal M[S]\big)c$ — precisely the
**diagonal of the §4 velocity** (up to the shared $-1$), because the dipole
commutator's diagonal cancels identically (§4, *Checks*). So the band velocity
probes the pencil-derivative term alone: no position operator, no gauge
choice, no pseudopotential. And $\partial E_n/\partial k$ is obtainable
independently, by central differences of the generalized eigenvalues of
$H(k),S(k)$.

`calc/test/probe_velocity_gauge.cpp` runs exactly this, along a Cartesian axis
at a low-symmetry $k$ (all band velocities vanish at $\Gamma$ by time reversal,
so the test is empty there). Relative error against $\partial E_n/\partial k$,
one complex scale factor fitted over all bands — **as first measured**, i.e.
before the `invert_cell` repair described below (the route (a) LCAO entry is
the number that repair later reduced to $0.75$; route (b) is unaffected by
it):

| model | route (b), §4 | route (a), explicit integrals |
|---|---|---|
| orthogonal SK, periodic | $1.1\times10^{-5}$ | $1.1\times10^{-5}$ |
| non-orthogonal SK, $S$ from the same table | $7.4\times10^{-6}$ | *does not exist* |
| diamond LCAO, 2s2p1d | $8.7\times10^{-6}$ | $0.996$ *(pre-repair; $0.75$ after)* |

$10^{-5}$–$10^{-6}$ is the $O(h^2)$ truncation of the difference quotient, i.e.
route (b) is exact to the precision the reference is known. The fitted
constants recovered the physical constants rather than assuming them: route
(b) returned $\kappa=i$ to six digits, and route (a) on the orthogonal model
returned $\kappa=-0.658211=-\hbar$ in eV·fs — the value it had in the
**pre-unification** Å/fs storage; in today's canonical eV·Å convention the
same fit returns $\kappa=-1$ exactly, which `test_velocity_units` asserts.

Two things follow, and they are worth keeping apart.

**The spectral measurement (f-sum rule).** The band velocity arbitrates only
the diagonal; spectral functions live on $|v_{nm}|^2$ across pairs. To settle
the off-diagonals, `probe_velocity_gauge --spectral` first validates the §4
boxed operator at general $k$ on SK models, where the two routes are the same
operator and must agree as full matrices — they do, to $1.3\times10^{-16}$
(cubic) and $4.8\times10^{-17}$ (triclinic), and this is also the check that
caught the displacement double-counting §4's regrouping note warns about
(bond moments paired with the full $X_0$ first came out at $1.12$, not
$10^{-16}$; the ket-centred split fixed it).

**A unit hazard this measurement exposed, worth recording first.** The two
velocity builders **used to** store their operators in conventions differing
by $\hbar$: the generic orthogonal builder divided by $\hbar$, storing plain
velocity in Å/fs, while op2c (`build_overlap_velocity`, factor `erfactor`
$=14.399647=(\hbar^2/m_e)\times(\mathrm{Bohr}^{-1}/\text{Å}^{-1})$ to seven
digits) stored $\hbar\times$velocity in eV·Å. A first pass of this analysis
calibrated on SK and applied that constant to op2c, suppressing every LCAO
route-(a) weight by $\hbar^2=0.433$ and producing a spurious "60% sum-rule
deficit". The same hazard was live in production — `ComputeAc` and the
quantum-metric path consume `velocity_state` with no per-model conversion —
**and has since been fixed by unifying the convention**: the canonical stored
velocity is $\hbar\times v$ in eV·Å (the $\hbar=1$ system the whole pipeline
lives in — the correlators step time in $1/\mathrm{eV}$, and `sigma_au_to_si`
is exactly $e^2/(\hbar\cdot\text{Å})=2.4341\times10^{6}$ S/m for these
units). op2c already stored it; the generic builder's $/\hbar$ was removed
(`commutator_velocity.hpp` UNITS), which rescales **SK-model** $\sigma$
outputs by $\hbar^{-2}=2.31$ relative to anything recorded before the
unification. `test_velocity_units` locks the convention across families: the
eigenbasis diagonal equals $-\,dE_n/dk$ at the FD floor on **both** families
($1.3\times10^{-5}$ SK, $3.2\times10^{-6}$ LCAO), and the LCAO f-sum lands in
its calibrated window (the deliberately coarse $2^3$ lock grid gives
$\sim7.0$ against a converged $7.9$–$8.1$), where an $\hbar^2$ unit error
would land at $\sim3$ or $\sim16$.

With units corrected, the probe dumps every $\tilde v_{nm}=\hbar v_{nm}$ over
shifted k-grids on diamond LCAO (2s2p1d) and scores both conventions against
the Thomas–Reiche–Kuhn f-sum rule,
$n_{\rm eff}=\tfrac{2}{N_k}\sum_{k}\sum_{n\,{\rm occ}}\sum_{m\,{\rm unocc}}
|\tilde v_{nm}|^2/(3.80998\,\omega_{nm})$, exact value 8 valence electrons per
cell:

| convention | $n_{\rm eff}$, $4^3$ grid | $n_{\rm eff}$, $6^3$ grid | exact |
|---|---|---|---|
| route (b), commutator | 7.894 | 8.133 | 8 |
| route (a), explicit integrals | 7.423 | 7.678 | 8 |

Route (b) brackets the exact value within k-grid noise ($\pm1.7\%$); route (a)
sits $4$–$8\%$ below it — the classic finite-basis underestimate of momentum
matrix elements, at the magnitude the LCAO optical literature reports. The
energy-resolved structure is textbook: the median per-pair amplitude ratio
$|v^{(a)}/v^{(b)}|$ is $0.98$–$1.02$ for $\omega_{nm}<18$ eV — near parity in
the physically relevant window — degrading to $0.84$–$0.89$ above $25$ eV,
where the basis cannot follow $\hat p$. The f-sum-weighted overall deviation
is $15\%$. Diagonals scatter around parity ($1.16$ valence, $0.85$ low
conduction in $|v_{nn}|^2$) but with large per-band spread: the scale-free
band-velocity fit for (a) has residual $0.75$ where (b) sits at $10^{-6}$.

**The dipole term is not a correction — it is half the operator.** Since
$\Delta=X_0-\mathcal K[S]$ (§4) is the only ingredient that needs real
orbital integrals, it is worth knowing what a model lacking them loses.
Setting $\Delta=0$ and keeping the pencil-derivative term alone (the same
run, `include_dipole=false`) gives:

| | $n_{\rm eff}$ (exact 8) |
|---|---|
| full §4 commutator | 7.894 |
| **pencil derivative only, $\Delta=0$** | **29545** |
| explicit integrals | 7.423 |

Three orders of magnitude, not a few percent. The diagonal is untouched
($1.3\times10^{-13}$ — the commutator vanishes there, as §7's band-velocity
argument requires), so the entire damage is interband, growing with transition
energy: the median $|v_{\rm no\,dipole}/v_{\rm full}|$ runs $1.3$ at
$\omega<10$ eV, $1.8$ at $14$–$18$ eV, and $40$ above $40$ eV.

The reason is structural, not numerical. The pencil-derivative term alone is
**not gauge covariant**: shifting the intracell atomic positions changes it by
a commutator. Only the sum with the dipole commutator is a physical operator.
The two pieces are individually large and largely cancel; dropping either
leaves the residue, not the answer. So $\Delta\approx0$ is not an available
approximation in a non-orthogonal basis — which is exactly why the bare bond
moment (`commutator_velocity.hpp`) serves only orthogonal models, where
$\Delta$ vanishes identically and the pencil derivative *is* the whole
velocity, and why an overlap-bearing model without $X_0$ is refused a
velocity outright rather than handed route (c).

**What route (b) therefore does and does not buy.** It removes the
*pseudopotential* requirement, not the *integral* requirement: $X_0$ comes
from plain two-centre orbital dipole integrals (`overlap_position`), with no
$\beta$ projectors and no 3-body term. For LCAO that made the velocity
unconditional — position was always available while the op2c velocity was
gated on `.upf` files, and the production `kGeneric` route now runs on the
dipoles alone. For a non-orthogonal SKTB/DFTB model it buys nothing yet:
those declare position `kUnavailable` because there are no orbital shapes to
integrate, and the measurement above forbids approximating $\Delta$ away.
Giving DFTB a velocity needs dipole integrals over its Slater orbitals (or a
Löwdin orthogonalisation), and that is a separate piece of work.

**Verdict: the commutator convention remains the correct default, on
consistency rather than magnitude grounds.** The honest summary is that the
two conventions agree closely where a 2s2p1d basis is trustworthy, and route
(a) is *not* badly wrong — but route (b) is *exactly* right for the model
being propagated: band velocities at the FD floor, f-sum at the electron
count within grid noise, the unique generator consistent with the Chebyshev
dynamics (DC, MSD), the only definition of $x_{nm}$ under PBC (quantum
metric), and the only velocity that exists at all for SK/DFTB models — with
no pseudopotential files needed. Route (a)'s remaining role is what it is
good at: an independent cross-check, and the better estimate only when the
target is the continuum matrix element itself rather than the model's
spectrum. Where the two disagree above $\sim25$ eV, neither should be
trusted: that is the basis's failure domain, not a convention question.

**Route (b) is the band velocity of the propagated pencil, exactly, orthogonal
or not.** The non-orthogonal SK row is the load-bearing one: $S_1\neq0$ there,
$H$ and $S$ come from one table, and no explicit velocity exists for such a
model at all (`sktb_model.hpp` declares `OperatorSource::kUnavailable`). That
row is the §4 extension validated on its own terms.

**The LCAO failure of route (a) is not yet an incompleteness measurement.**
Three checks were run to attribute it, and the first two came back clean:

* *Provenance.* op2c computes $S$ from its own orbital integrals while $H$ is
  read from file, so they could have been different bases. They are not:
  op2c's overlap matches the `overlaps.h5` shipped with the Hamiltonian to
  $1.7\times10^{-7}$.
* *Symmetry at $\Gamma$.* Route (a) is Hermitian to $6\times10^{-16}$ and its
  subspace-averaged diagonal vanishes to $7\times10^{-14}$ — time reversal is
  respected, so the operator is not grossly malformed.
* *Bloch periodicity.* Any legitimate $k$-space operator satisfies
  $A(k+G)=U_G^\dagger A(k)U_G$ with $U_G=\mathrm{diag}(e^{-2\pi iG\cdot f_i})$
  — mechanically, for **any** block values, whenever every deposit is phased
  as $e^{-2\pi ik\cdot(\Delta R+f_j-f_i)}$. First measurement: $H$
  $4.8\times10^{-16}$, $S$ $9.9\times10^{-16}$, op2c velocity $0.77$, op2c
  position $1.85$.

**Root cause — not op2c.** Tracing that violation to its source (per-element
forensics with pbc off, then per-stored-block forensics with pbc on) showed
op2c's own phasing to be correct in every term; the defect was one level down.
`vbcsr`'s `AtomicData::invert_cell` applied the plain inverse $C^{-1}$ of the
cell matrix where the library's own row-vector convention
($r=\sum_iR_ia_i$, i.e. `get_edge_vec`, `CellShiftComponent`,
`collocation.hpp`'s transposed indexing) requires $(C^{-1})^{\mathsf T}$. For a
symmetric cell the two coincide — every cubic test was blind to it — but for
diamond's triclinic primitive cell it maps the lattice vectors themselves to
$(1,-0.577,-0.408)$, $(0.5,0.711,-0.558)$, $(0.5,0.045,0.678)$ instead of the
identity. Any Bloch phase built from a **full Cartesian bond vector** (op2c's
velocity and position) then acquires a non-integer lattice part and stops being
a Bloch operator at all. Phases built from **integer image keys plus intra-cell
deltas** (`sample_k`, the commutator velocity) used the same wrong fractionals
*consistently*, which is a legitimate — if unintended — diagonal gauge: spectra
and every trace-type observable (DOS, AC, DC, MSD) computed within that family
were unaffected. Only mixed-provenance combinations at $k\neq\Gamma$ in
non-symmetric cells were wrong, and everything at $\Gamma$ was exact.

With the transpose fixed in `invert_cell`, all consistency checks pass at
machine precision (covariance of op2c $V$: $5.4\times10^{-16}$; of op2c $X$:
$5.6\times10^{-16}$), every previously passing number is unchanged, and route
(a) is a genuine Bloch operator.

**Which makes the remaining gap a measurement.** Against the fixed operator,
route (a) still misses $\partial E_n/\partial k$: relative error $0.75$ at the
test $k$, $0.51$–$0.57$ closer to $\Gamma$, with a fitted scale drifting far
from $\hbar$ — while route (b) sits at the finite-difference floor throughout.
With provenance excluded ($10^{-7}$), symmetry and Bloch consistency verified,
and the instrument validated to $10^{-15}$ on SK, what remains **is** the
finite-basis error of the explicit momentum route: the well-documented fact
that $\langle n|\hat p/m + \tfrac{i}{\hbar}[\hat V_{\rm NL},\hat x]|n\rangle$
equals $\partial E_n/\partial k$ only in a complete basis, while the
Hellmann–Feynman/commutator route contains the basis correction exactly. For
this 2s2p1d carbon basis the discrepancy is large — order one on the fitted
band-velocity scale — not a rounding-level concern.

### Higher moments

For the $n$-th moment, lift to the truncated polynomial algebra
$\mathbb R[\varepsilon]/(\varepsilon^{n+1})$ — an $(n{+}1)$-jet. An operator
becomes upper-triangular Toeplitz in its moments,

$$
\mathsf O^{(2)}=\begin{pmatrix}O_0&O_1&O_2\\0&O_0&O_1\\0&0&O_0\end{pmatrix},
$$

matrix functions act blockwise, and the $k$-th superdiagonal of $f(\mathsf
O^{(n)})$ is the $k$-th derivative. The cost is $n{+}1$ generator applies plus
$n$ driver applies per Chebyshev step. This would give $\langle x^2\rangle$
directly rather than through $|[\hat x,U]|^2$, and is the natural route to
quantities needing second derivatives at a single $k$-point.

## 8. Implementation status

**The commutator is now the production velocity, for every family.** The
operator layer carries it as `VelocityAxis`: a bare sparse matrix when the
basis is orthogonal (`BuildCommutatorVelocity`, the bond moment of $H$), and
the three sparse pieces $\{\mathcal M[H],\ \bar X,\ \Delta\}$ of

$$Vx = i\big(\mathcal M[H]x-\bar XS^{-1}Hx+HS^{-1}\Delta x\big),\qquad
\bar X=X_0-\mathrm{diag}(r)S_0,\quad \Delta=X_0-\textstyle\sum_R(R{+}r_j)S(R)e^{i\phi}$$

when it is not (`BuildNonorthogonalCommutatorVelocity`). The exact operator
contains $S^{-1}$ and is applied, never materialized:
`Chebyshev::apply_dual` owns the metric solves (CG in the Solve form,
$ZZ^{\mathsf H}$ in the Sandwich form) and dresses the composite exactly as it
dressed a bare matrix — one dispatch point, so AC, DC (both kernels) and the
quantum metric all inherited the commutator without touching their algorithms.
MSD needed nothing: all its routes were already commutator-built.

Consequences: LCAO velocity no longer needs pseudopotential files (the dipole
integrals suffice); op2c's explicit-integral velocity remains on the model as
a cross-check (`BuildVelocity`) but is not what production propagates; every
piece lives on the atomic-graph sparsity, cheaper per application than op2c's
two-hop projector pattern, with the two extra metric solves paid once per
correlator sample, not per Chebyshev moment. `test_velocity_units` locks the
chain end to end: parts vs dense reference at $10^{-16}$, eigenbasis diagonal
$=-\,\partial E_n/\partial k$ at the FD floor on **both** families
($1.3\times10^{-5}$ SK, $3.2\times10^{-6}$ LCAO — a lock the explicit route
could never pass), the production `apply_dual` path vs dense $S^{-1}Vx$ at the
CG floor ($2.5\times10^{-7}$), and the f-sum in its grid window.

**The non-orthogonal periodic MSD is done.** `corr_msd(cheby, velocity, ...)`
is the §2a derivation recursion driven by the commutator: the pair
$(\varphi,\psi)$ with $\varphi_m=[\mathcal X,T_m(\tilde A)]\psi$ propagates
under $[[\tilde A,\tilde C],[0,\tilde A]]$, where the coupling
$\tilde C=\tfrac{i}{w}\,\mathrm{apply\_dual}(V,\cdot)$ is the same
form-dressed composite every other consumer applies; $\varphi(0)=0$ and
$\varphi(t)$ **is** $[\mathcal X,U(t)]|\psi\rangle$ — no position operator,
no diagonal split, no gauge choice. Spectral readouts are metric-weighted per
§5 (`metric_density_moments`: $\langle\varphi|S\,T_n|\varphi\rangle$ in the
Solve form; η-space dots already are the $S$-metric in the Sandwich form —
the OBC routes now use the same weighting). Validated against an exact dense
$2\times2$ non-orthogonal reference — the block generator exponentiated, the
driver assembled independently, S-weighted moments — to $10^{-6}$ per energy
point per time step, and end-to-end on LCAO diamond through
`compute_msd_spectrum`, where the $t{=}0$ row vanishes identically as
$\varphi(0)=0$ demands. The orthogonal Γ-affine route is kept: the diagonal
split is cheaper per step where it is valid.

Still staged: a DFTB velocity, which needs dipole integrals over Slater
orbitals before any of this applies (the gate now refuses by name:
`compute_msd_spectrum` on a velocity-less non-orthogonal model).

### Original staging notes

Kept as the record of how the work was scoped; every "missing" item below is
now **done**, by the code named in brackets.

Present from the start:

* `BuildLatticeMoment(atomic_data, hamiltonian_images_)` → $H_1$.
* Γ-point position matrix $X_0$ (`position_axis.matrix`, from op2c's
  `overlap_position`) and the orthogonal diagonal position.
* `propagate_affine` / `GammaAffineState` — the lift of §2b.
* `MetricSampler` — the $S$-weighted estimator of §5.

Was missing for the non-orthogonal periodic route:

1. **The overlap moment.** `BuildLatticeMoment` weights each image by
   $R_\alpha$ alone with no Bloch phase — the lattice-gauge moment, correct
   only at $\Gamma$ — and was only ever called on $H$. The atomic-gauge
   general-$k$ object is the bond moment $\mathcal M[\cdot]$ of §3, and no
   $\mathcal M[S]$ builder existed. **Done**: `AccumulateBondMoment`
   (commutator_velocity.hpp) builds $\mathcal M$ and $\mathcal K$ of any image
   container, at any $k$.
2. **The §4 operator assembly**, replacing the bare `H1.mult` coupling —
   undressed at the time, one of the two reasons the periodic route was gated
   to orthogonal bases (the other being the elementwise
   `pointwise_mult(diagonal_position)`, removed by §2a). **Done**:
   `BuildNonorthogonalCommutatorVelocity` (nonorthogonal_velocity.hpp) +
   `apply_dual`'s composite branch (cheby.hpp) + `corr_msd(cheby, velocity,
   ...)` (msd.hpp).
3. **Metric-weighted moments** per §5. **Done**: `metric_density_moments`
   (msd.hpp), used by every MSD route.

Settled since, and what it leaves open (§7):

4. **Which velocity operator.** Route (b) is verified to reproduce
   $\partial E_n/\partial k$ exactly, orthogonal and non-orthogonal alike, so
   it is the velocity consistent with the propagated pencil. The intended
   yardstick — route (a) — initially failed the Bloch-covariance law; the
   forensics traced that to the `invert_cell` transpose defect in vbcsr (§7),
   which is now fixed, with op2c's own assembly exonerated. Against the fixed
   route (a), the residual $\sim0.5$–$0.75$ **is** the finite-basis error of
   the explicit momentum route for this 2s2p1d basis. The follow-up —
   $\sigma(\omega)$ both ways plus the f-sum rule — is done (§7): route (b)
   saturates the f-sum at $8.0\pm0.13$ of 8 valence electrons, route (a)
   reaches $7.4$–$7.7$, and the two agree to $\sim2\%$ below 18 eV. The
   commutator stays the default on exactness-for-the-model grounds. The
   $\hbar$ unit mismatch between the two velocity builders is fixed — the
   canonical convention is $\hbar v$ in eV·Å, locked by
   `test_velocity_units` across both families (§7).

   Independently, route (b) gives a velocity to **LCAO** models carrying no
   pseudopotentials, which have none today, since the dipole integrals it
   needs are unconditionally available there. It does *not* yet reach
   non-orthogonal SKTB/DFTB: those have no position operator at all, and §7
   measures that dropping the dipole term inflates the f-sum by three orders
   of magnitude, so it cannot simply be omitted.

## 9. Prior art

The commutator recursion of §2a is standard in linear-scaling quantum
transport. Fan et al., *Efficient linear-scaling quantum transport calculations
on graphics processing units* (Comput. Phys. Commun. 185, 28 (2014);
[arXiv:1307.0288](https://arxiv.org/abs/1307.0288)) give it explicitly as

$$
[X,T_m(H)]=2[X,H]T_{m-1}(H)+2H[X,T_{m-1}(H)]-[X,T_{m-2}(H)],
\quad [X,T_0]=0,\ [X,T_1]=[X,H],
$$

driven by $V=(i/\hbar)[H,X]=(i/\hbar)\sum_{\langle mn\rangle}(X_n-X_m)H_{mn}|m\rangle\langle n|$
— the same bond-vector weighting as §3. It is implemented in GPUQT, GPUTB and
LSQT, and reviewed in *Linear scaling quantum transport methodologies*
(Phys. Rep. 903, 1 (2021)).

The block-triangular identity of §2b predates the physics application and
belongs to matrix-function numerics (Higham, *Functions of Matrices*, ch. 3;
Al-Mohy & Higham on Fréchet derivatives).

**RSATB's affine representation is a re-derivation of this method, not a new
one, and should not be presented as novel.**

**Nor is the non-orthogonal velocity formula of §4 new — it is published, term
for term.** Lee, Ozaki *et al.*, *Tight-binding calculations of optical matrix
elements for conductivity using nonorthogonal atomic orbitals: Anomalous Hall
conductivity in bcc Fe* (Phys. Rev. B 98, 115115 (2018);
[arXiv:1805.08918](https://arxiv.org/abs/1805.08918)) derive
$\langle\vec k m|\hat{\vec p}|\vec k n\rangle$ starting from
$\hat{\vec p}=(im_e/\hbar)[\hat H,\hat{\vec r}]$ and inserting the
non-orthogonal resolution of identity
$\hat I=\sum|\vec R M\rangle S^{-1}_{\vec RM,\vec R'N}\langle \vec R'N|$
(their Eq. 16). Their Eq. 7 contains all three of our pieces:

| §4 term | their Eq. 7 |
|---|---|
| $\mathcal M[H]$ / $H_1$ | $\partial H_{MN}(\vec k)/\partial\vec k$ |
| $-\mathcal M[S]A_0$ / $-S_1A_0$ | $-\epsilon_{\vec km}\,\partial S_{MN}(\vec k)/\partial\vec k$ |
| the dipole commutator $[S_0^{-1}\Delta,A_0]$ | $i(\epsilon_{\vec km}-\epsilon_{\vec kn})\sum_{\vec R}\langle 0M|\hat{\vec r}|\vec RN\rangle e^{i\vec k\cdot\vec R}$ |

The eigenvalue prefactors are what our commutator form $[\,\cdot\,,A_0]$
becomes once $A_0$ acts on an eigenvector, so the two statements carry the same
content. Same physics is in *Optical matrix elements in tight-binding models
with overlap* (Phys. Rev. B 72, 125105 (2005)) and, most thoroughly, in
Esteve-Paredes & Palacios, *A comprehensive study of the velocity, momentum and
position matrix elements for Bloch states: application to a local orbital
basis* (SciPost Phys. Core 6, 002 (2023);
[arXiv:2201.12290](https://arxiv.org/abs/2201.12290)), which surveys both the
commutator and Berry-connection routes and makes exactly our point that in a
local-orbital basis the velocity is *not* the $k$-derivative of the Bloch
Hamiltonian. Anyone reviewing a novelty claim here will know these papers.

The non-orthogonal KPM machinery is old too: Kernel polynomial method for a
nonorthogonal electronic-structure calculation of amorphous diamond
(Phys. Rev. B 55, 15382 (1997)) already applies $S^{-1}H$ to a vector by
preconditioned conjugate gradient at $O(N)$ — the "solve" form of
`cheby.hpp`. But it computes electronic structure, not transport: no velocity
operator appears.

What survives as a contribution is narrower, and worth stating precisely:

1. **The three literatures have not been joined.** The non-orthogonal velocity
   work is $k$-space and diagonalization-based (Ozaki 2018 uses a
   $150^3$ $k$-mesh on a 2-atom bcc Fe cell, with a generalized eigensolve at
   every point); the non-orthogonal KPM work is diagonalization-free but
   computes no transport; and the linear-scaling transport literature — LSQT,
   GPUQT, GPUTB, the Fourier–Chebyshev Kubo method (PRL 132, 076302 (2024)),
   and the Phys. Rep. 903 review — is orthogonal throughout. §4 in *operator*
   form is what lets the $S$-dressed commutator ride inside a Chebyshev
   recursion applied to a vector, with no $k$-points and no eigenvectors: the
   published formulas presuppose the eigenpairs their prefactors are written
   in, so they cannot be used this way as they stand. That is the only route
   to the system sizes this code targets.
2. **The same object serves the periodic MSD**, where the position operator has
   no matrix at all and there is therefore no alternative construction. That
   application does not appear in the non-orthogonal optical literature, which
   only ever needs $\langle n|v|m\rangle$ between eigenstates.
3. **The consistency argument of §7**, made quantitative: that for spectral
   functions of a propagated $H$ the commutator velocity is the correct choice
   on sum-rule grounds rather than the cheap one, verified against
   $\partial E_n/\partial k$ to $10^{-6}$ in a non-orthogonal basis.

Claim (1) and (2); do not claim the non-orthogonal velocity operator itself.
