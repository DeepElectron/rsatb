# Square-Root Filtered Stochastic Chebyshev LDOS from LCAO H/S

**Design of record for the real-space LDOS path.** Supersedes the
region-seeded deterministic construction in
`design_realspace_density_ldos.md` sections 1-3, which is retained only for
its density-path material (probing on the S-pattern, k-sampling equivalence)
and for the collocation/grid/partition layers, which this design reuses
unchanged.

Selected inversion of `(E + i.eta) S - H`, discussed at length in the design
review, moves to the **density** path and is not used here: it prices per
energy and per factorization, whereas the estimator below serves every energy
from one recursion.

---

*Author: project owner; revised during implementation review (sections 2, 3.5
and 6.3-6.4 were rewritten - see section 12 for what changed and why).*

A complete formalism for linear-scaling real-space LDOS extraction on
nonorthogonal numerical atomic orbitals. Generalization of the time-dependent
random state method (Zhou & Yuan, arXiv:2203.03465) to (i) nonorthogonal LCAO
bases, (ii) energy-resolved rather than occupation-resolved quantities, and
(iii) direct Chebyshev filtering in place of time propagation.

## Notation

| Symbol | Meaning |
|---|---|
| $\phi_\mu(\mathbf r)$ | NAO basis function, $\mu=1\dots N$, strictly zero for $\Vert \mathbf r-\boldsymbol\tau_\mu\Vert >r_\mu^{\rm cut}$ |
| $H_{\mu\nu},\,S_{\mu\nu}$ | Hamiltonian / overlap in NAO basis, both sparse on the pattern $\mathcal P$ |
| $\mathcal P$ | $\{(\mu,\nu): \Vert \boldsymbol\tau_\mu-\boldsymbol\tau_\nu\Vert  < r_\mu^{\rm cut}+r_\nu^{\rm cut}\}$ |
| $c_n,\varepsilon_n$ | Generalized eigenpairs: $Hc_n=\varepsilon_nSc_n$, $c_n^\dagger Sc_m=\delta_{nm}$ |
| $Z$ | Orthogonalizer, $Z^\dagger SZ=I_{N'}$, $ZZ^\dagger=S^{-1}$ (or its pseudo-inverse) |
| $\tilde H = Z^\dagger H Z$ | Orthogonalized Hamiltonian, spectrum $=\{\varepsilon_n\}$ |
| $\bar H=(\tilde H-b)/a$ | Rescaled, spectrum $\subset[-1,1]$ |
| $\eta$ | Energy broadening (Gaussian standard deviation) |
| $M$ | Chebyshev expansion order |
| $S_{\rm smp}$ | Number of random samples |
| $N_E,\ N_{\rm grid}$ | Number of energy points, number of real-space grid points ($dr$) |
| $\boldsymbol\xi$ | Random vector, $\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I$ |

---

## 1. Exact reduction of the LDOS

### 1.1 Definition

$$
\rho(\mathbf r,E)\;=\;\sum_n \big|\psi_n(\mathbf r)\big|^2\,\delta(E-\varepsilon_n),
\qquad \psi_n(\mathbf r)=\sum_\mu c_{n\mu}\,\phi_\mu(\mathbf r).
$$

Introduce the **spectral density matrix**

$$
A(E)\;\equiv\;\sum_n c_n c_n^\dagger\,\delta(E-\varepsilon_n)\;\in\;\mathbb C^{N\times N},
$$

so that, with $\boldsymbol\phi(\mathbf r)=(\phi_1(\mathbf r),\dots,\phi_N(\mathbf r))^T$,

$$
\rho(\mathbf r,E)=\boldsymbol\phi(\mathbf r)^T A(E)\,\boldsymbol\phi(\mathbf r). \tag{1.1}
$$

### 1.2 The sparsity theorem

Because the NAOs have **strict** cutoffs, $\phi_\mu(\mathbf r)\phi_\nu(\mathbf r)\equiv 0$
for all $\mathbf r$ whenever $(\mu,\nu)\notin\mathcal P$. Hence in (1.1) only
$A|_{\mathcal P}$ contributes:

$$
\rho(\mathbf r,E)=\sum_{(\mu\nu)\in\mathcal P}\phi_\mu(\mathbf r)A_{\mu\nu}(E)\phi_\nu(\mathbf r).\tag{1.2}
$$

**This restriction is exact, not an approximation.** It is the single fact that
makes the whole method linear-scaling: we never need, store, or compute the
$N^2$ elements of $A$ - only the $O(N)$ elements on the overlap pattern.
$\mathcal P$ is available for free; it is the pattern already stored for $H$
and $S$.

### 1.3 Removal of the metric

Let $Z$ satisfy $Z^\dagger SZ=I_{N'}$ (construction in section 2). Then
$c_n=Zu_n$ with $\tilde Hu_n=\varepsilon_nu_n$, $u_n^\dagger u_m=\delta_{nm}$, and

$$
A(E)=Z\Big(\sum_n u_nu_n^\dagger\delta(E-\varepsilon_n)\Big)Z^\dagger=Z\,\delta(E-\tilde H)\,Z^\dagger.\tag{1.3}
$$

Substituting into (1.1) and defining the **point vector**
$\mathbf w(\mathbf r)\equiv Z^\dagger\boldsymbol\phi(\mathbf r)\in\mathbb C^{N'}$,

$$
\rho(\mathbf r,E)=\mathbf w(\mathbf r)^\dagger\,\delta(E-\tilde H)\,\mathbf w(\mathbf r).\tag{1.4}
$$

Equation (1.4) is the master identity: **every real-space point is a quadratic
form of the same orthogonal-basis spectral projector.** Any algorithm written
for orthogonal tight-binding site-LDOS, $\langle i|\delta(E-H)|i\rangle$,
generalizes by the substitution $|i\rangle\to\mathbf w(\mathbf r)$.

- **Gauge invariance.** $\rho$ depends on $Z$ only through $ZZ^\dagger=S^{-1}$,
  hence is invariant under $Z\to ZQ$ for unitary $Q$. Loewdin ($S^{-1/2}$),
  Cholesky ($L^{-T}$), and AINV factors give *identical* results. Choose on
  numerical grounds alone. Since only the *product* $ZZ^\dagger$ survives, a
  formula free of $Z$ altogether must exist; section 2.3 exhibits it, and
  section 4.3 shows where a factor is nevertheless irreducibly required.
- **Exact sum rule (I).** $\int A(E)\,dE=\sum_nc_nc_n^\dagger=S^{-1}$, therefore

  $$
  \int \rho(\mathbf r,E)\,dE=\boldsymbol\phi(\mathbf r)^TS^{-1}\boldsymbol\phi(\mathbf r)=\Vert \mathbf w(\mathbf r)\Vert ^2 .\tag{1.5}
  $$

---

## 2. Overlap handling: the two modes, $\tilde Z$, and spectral bounds

### 2.1 The two overlap modes of this package

There is exactly one $Z$ object, `calc/isqrtm.hpp`: a sparse
$\tilde Z\approx S^{-1/2}$ built by Newton-Schulz (order-3 iteration
$\tfrac{15}{8}-\tfrac{10}{8}X+\tfrac38X^2$) with block filtering at `sptol`,
seeded from a subgraph dense diagonalization. It converges on, and reports,
precisely the quantity this design needs to track:

$$
E\;\equiv\;\tilde Z^\dagger S\tilde Z-I,\qquad \Vert E\Vert _{\max}\le\mathrm{tol}.\tag{2.1}
$$

That one object is consumed in **two different ways**, and `calc/cheby.hpp`
already provides both as distinct constructors. They are not "Cholesky versus
iterative" - they are *the same $\tilde Z$, loose or purified*:

| | **Mode B - orthogonalized** | **Mode A - adjoint** |
|---|---|---|
| Constructor | `Chebyshev(H, prec, Symmetric, ...)` | `Chebyshev(H, S, prec, Symmetric, ...)` |
| $\tilde Z$ is used as | **congruence / orthogonalizer** | **preconditioner** |
| Operator | $\tilde Z^\dagger H\tilde Z$ | $S^{-1}H$ (`NORMAL`), $HS^{-1}$ (`ADJOINT`) |
| Per step | 1 SpMV($H$) + 2 SpMV($\tilde Z$) | 1 SpMV($H$) + $\hat B_k$ apply, section 2.5 |
| Needs $\tilde Z$ | **purified** | **loose is fine** |
| Bias | $O(\mathrm{tol})$ in the spectrum | **none** |
| Accuracy bought by | $\tilde Z$'s density | CG iterations |

Mode B is the Loewdin route: if $E\to0$ then $\tilde Z^\dagger H\tilde Z\to\tilde H$
exactly and no solver is needed anywhere. Mode A never forms $\tilde H$; it uses
$\tilde Z$ only to make the inner system well-conditioned, so $\tilde Z$'s error
moves out of the answer and into the iteration count. Section 2.3 gives the
identity that makes Mode A legitimate, and section 2.4 decides between them.

**The truncated-canonical option** ($Z=U_{\rm keep}s_{\rm keep}^{-1/2}$, dropping
the null space) remains the fallback for $\lambda_{\min}(S)\lesssim10^{-3}$; note
`isqrtm_init` already floors the initial guess at $x>10^{-12}$.

### 2.2 Spectral radius - the LCAO cost trap

The Chebyshev order is set by the half-bandwidth of the **pencil**, not of $H$:

$$
a=\tfrac12(\varepsilon_{\max}-\varepsilon_{\min})(1+\epsilon_s),\qquad
b=\tfrac12(\varepsilon_{\max}+\varepsilon_{\min}),\qquad \epsilon_s\simeq0.01,
$$

with $\varepsilon_{\rm extremal}$ the extremal eigenvalues of $(H,S)$. Since
$\varepsilon_{\max}\lesssim\Vert H\Vert /\lambda_{\min}(S)$, a poorly conditioned
overlap **directly multiplies the cost**. For DZP NAOs $\lambda_{\min}(S)$ is
typically $10^{-2}$-$10^{-1}$, inflating $a$ by $3$-$5\times$ relative to
$\Vert H\Vert $.

1. Estimate $\lambda_{\min}(S)$ by a few LOBPCG / inverse-Lanczos iterations.
2. If $\lambda_{\min}(S)>10^{-3}$: proceed with $\tilde Z$ from `isqrtm`.
3. If $\lambda_{\min}(S)\le10^{-3}$: near-null directions carry no physics but
   enormous fictitious eigenvalues. Project them out (truncated canonical
   orthogonalization on the low-$S$ subspace only, by sparse eigensolve - a
   handful of vectors, not a full diagonalization), or prune the basis.
4. Obtain $\varepsilon_{\min},\varepsilon_{\max}$ by ~30 Lanczos steps on
   $\tilde H$; pad by $\epsilon_s$. **Under-estimating $a$ is fatal** (the
   recursion diverges); over-estimating merely costs proportional extra
   moments.

### 2.3 The adjoint (metric-free) representation

Mode B is exact only in the limit $E\to0$, and driving $E$ down costs $\tilde Z$'s
sparsity: every Newton-Schulz step is an SpGEMM whose result is refiltered at
`sptol`, so accuracy is bought with fill, and the iteration stalls once `sptol`
truncation dominates (`isqrtm.hpp:89` exits on
$|{\rm norm}_{\rm pre}-{\rm norm}|<0.005\,{\rm norm}$ for exactly this reason).
Mode A escapes that trade entirely, and the following identity is why it is
legitimate rather than an approximation.

**Theorem (intertwining).** Let $Z^\dagger SZ=I$, $ZZ^\dagger=S^{-1}$. Then for
every $m\ge0$

$$
Z\,(Z^\dagger HZ)^m\,Z^\dagger=(ZZ^\dagger H)^m\,ZZ^\dagger=(S^{-1}H)^m\,S^{-1},
$$

hence for any $f$ with a convergent polynomial expansion

$$
\boxed{\;Z\,f(\tilde H)\,Z^\dagger\;=\;f(S^{-1}H)\,S^{-1}\;=\;S^{-1}f(HS^{-1})\;}\tag{2.2}
$$

*Proof.* Induction: $m=0$ is $ZZ^\dagger=S^{-1}$; and
$Z(Z^\dagger HZ)^mZ^\dagger=(ZZ^\dagger H)\,Z(Z^\dagger HZ)^{m-1}Z^\dagger$.
Extend by linearity. The second equality follows because the left side is
Hermitian and $(S^{-1}H)^\dagger=HS^{-1}$. $\square$

Applied to (1.3) and (1.1):

$$
A(E)=\delta_\eta(E-S^{-1}H)\,S^{-1}=S^{-1}\delta_\eta(E-HS^{-1}),\qquad
\rho(\mathbf r,E)=\boldsymbol\phi(\mathbf r)^T\delta_\eta(E-S^{-1}H)\,S^{-1}\boldsymbol\phi(\mathbf r).\tag{2.3}
$$

**No orthogonalizer appears.** Define the dual pair

$$
G\equiv S^{-1}H\quad(\text{contravariant}),\qquad
G^\dagger= HS^{-1}\quad(\text{covariant}),
$$

$G$ self-adjoint in $\langle x,y\rangle_S=x^\dagger Sy$ with eigenvectors $c_n$;
$G^\dagger$ self-adjoint in $\langle\cdot,\cdot\rangle_{S^{-1}}$ with eigenvectors
$Sc_n$. **Both have spectrum $\{\varepsilon_n\}$**, so section 2.2's $(a,b)$ and
section 3.2's $c_m(\bar E)$ are shared verbatim; only the recursion changes:

$$
X_{m+1}=2\bar GX_m-X_{m-1},\qquad \bar Gx=\big(S^{-1}(Hx)-bx\big)/a\tag{2.4}
$$

- one SpMV with $H$ and one **solve** with $S$ per step, in place of Mode B's two
  applies of $\tilde Z$. `calc/cheby.hpp` already implements exactly this as
  `ChebyMode::NORMAL`, and $G^\dagger$ as `ChebyMode::ADJOINT`.

**Duality is free.** If $Y_m\equiv SX_m$ then (2.4) implies

$$
Y_{m+1}=2\big(t_m-bY_m\big)/a-Y_{m-1},\qquad t_m\equiv HX_m,\tag{2.5}
$$

and $t_m$ is already computed by (2.4). So the covariant iterates
$Y_m=T_m(\bar G^\dagger)Sx_0$ cost **two AXPYs, no extra SpMV and no extra
solve**. Two consequences used below: the two-sided estimator of section 4.4
needs only one recursion pass per column pair, and

$$
\Vert X_m\Vert _S^2=X_m^\dagger Y_m
$$

is a free stability monitor. Since $\Vert T_m(\bar G)\Vert _S=1$, it must not exceed
$\Vert X_0\Vert _S$; in the Euclidean norm the recursion can grow only by the bounded
factor $\sqrt{\kappa(S)}$, not exponentially - which is what makes
$M\sim10^4$ steps of a non-symmetric recursion safe. (One honesty note: with
an inexact inner solve $Y_m$ tracks $SX_m$ only up to the accumulated solve
error - $O(M\,\mathrm{tol}^{k+1})\approx10^{-11}$ for the fixed polynomial of
section 2.5 - which is far below anything a divergence monitor needs to see.)

**The inner $S^{-1}$ is nearly free, and $E$ is why.** With $\tilde Z$ as a
`Symmetric` preconditioner the solve runs on $\tilde Z^\dagger S\tilde Z=I+E$,
whose condition number is $1+2\,\mathrm{tol}$ - each CG iteration gains a full
factor $\mathrm{tol}$:

$$
n_{\rm it}\;\simeq\;\left\lceil\frac{\ln(2/\tau_{\rm CG})}{\ln(2/\mathrm{tol})}\right\rceil
\;=\;3\ \text{ at }\ \mathrm{tol}=10^{-5},\ \tau_{\rm CG}=10^{-10}.\tag{2.6}
$$

Section 2.5 replaces even these few CG iterations with a fixed polynomial; (2.6)
remains the price of the PCG fallback (section 6.7).

### 2.4 Which mode - a measured decision, not a doctrine

Mode B's bias is not a vague $O(\mathrm{tol})$: it is an error in the *spectrum*
of the operator being filtered. The pencil $(\tilde Z^\dagger H\tilde Z,\,I+E)$
has the exact eigenvalues $\varepsilon_n$; Mode B discards the metric $I+E$, so
its eigenvalues are displaced by $\delta\varepsilon\sim\Vert E\Vert \,a$. A displacement
on the energy axis is seen by a filter of width $\eta$ as

$$
\frac{\delta\rho}{\rho}\;\sim\;\frac{|\partial_E\rho|}{\rho}\,\delta\varepsilon
\;\sim\;\mathrm{tol}\cdot\frac{a}{\eta}\;\simeq\;\frac{\mathrm{tol}\cdot M}{4}.\tag{2.7}
$$

(2.7) is a *budget*, not a disqualification. At the `ISQRTMPreconditioner`
default $\mathrm{tol}=10^{-5}$ with $a=100$ eV and $\eta=25$ meV it is
$4\times10^{-2}$ - a 4% error, so the default is not usable for LDOS in Mode B.
But Newton-Schulz is **cubically convergent**: one further sweep maps
$E\to O(E^3)$, so $10^{-5}\to\sim10^{-14}$ in a *single* extra iteration, two
from $10^{-2}$. Iteration count is never the obstacle to purification. What
purification actually costs is **density**: the achievable floor is set by the
drop tolerance ($\mathrm{tol}\gtrsim c\cdot\mathrm{sptol}$), so the last sweep
must run at $\mathrm{sptol}\sim10^{-11}$, and $d_Z(\mathrm{sptol})$ - the
density `isqrtm` prints - is governed by the decay length of $S^{-1/2}$,
$\propto\sqrt{\kappa(S)}$ lattice constants. Well-conditioned overlaps decay in
a few neighbour shells and the purified $\tilde Z$ stays a small multiple of
$S$'s density; ill-conditioned ones fill in, and *that* is when Mode B loses.

| | **Mode B** (purified $\tilde Z$) | **Mode A** (loose $\tilde Z$) |
|---|---|---|
| $\tilde Z$ setup | + one cubic sweep at $\mathrm{sptol}\sim\mathrm{tol}_B/10$ | first sweep only, $\mathrm{sptol}=10^{-5}$ |
| Bias budget | $\mathrm{tol}_B\le4\epsilon/M$ (2.7), one sweep away | $\mathrm{tol}^{k+1}M/4$, $k=2$ (2.8) |
| Per step | 1 SpMV($H$) + 2 applies at density $d_Z$ | 1 SpMV($H$) + 8 applies at density $d_S,d_{\tilde Z}$ |
| Price of accuracy | density $d_Z(\mathrm{sptol})$ | series length $k$ - logarithmic |
| Sampler (4.3) | trivial: $\mathbf c_0=\tilde Z\boldsymbol\xi$ | two-term series (4.8) |
| Loses when | $d_Z\gtrsim4\,d_S$, or $\lambda_{\min}(S)$ small | never much - the conservative choice |

**The decision is a measurement, not a doctrine.** Flop parity sits near
$d_Z\approx4\,d_S$ (Mode B's two dense-ish applies against Mode A's eight
sparse ones). So: purify once, read the density `isqrtm` reports, and

- $d_Z\lesssim4\,d_S$: **Mode B** - fewer applies per step, no inner
  polynomial, the sampler collapses to $\tilde Z\boldsymbol\xi$, and the bias
  is verified (not assumed) by the reported $\Vert E\Vert \le4\epsilon/M$;
- fill-in beyond that, memory-tight, or $\lambda_{\min}(S)\lesssim10^{-3}$:
  **Mode A** - the loose first sweep is all the setup there is, and accuracy is
  priced logarithmically.

Both configurations are correct; the two *mismatches* are the failure modes:
a loose $\tilde Z$ inside Mode B ((2.7) at $\mathrm{tol}=10^{-5}$: the 4%
trap), and pointless purification inside Mode A (density paid in all $M$ steps
for accuracy the identity already guarantees). The trace observables (DOS, AC,
MSD) sit in Mode B at the loose default legitimately - their $M$ is small and
(2.7) is then harmless. In short: Mode A never *trusts* $\tilde Z$ (its error
sets the series length and cancels in the sampler); Mode B *verifies* it - the
reported $\Vert E\Vert $ against the $4\epsilon/M$ budget, one number at setup.

### 2.5 The inner solve should be a fixed polynomial, not CG

CG has a hidden cost at scale that flops do not show: **two global allreduces
per iteration** (the $\alpha,\beta$ dot products), i.e.
$2\,n_{\rm it}M\sim10^5$ latency-bound synchronizations per sample block. At
$10^7$ atoms over thousands of ranks the hot loop becomes latency-dominated.
And an inexact CG solve applies a slightly *different* operator at every step -
the recursion is then not exactly a polynomial of any fixed matrix, which is
what forces the coherent $\tau_{\rm CG}M$ book-keeping of section 6.7.

Both problems disappear at once, because the solve is not a generic
$S^{-1}$ solve: $S^{-1}=\tilde Z(I+E)^{-1}\tilde Z^\dagger$ **exactly**, and
$\Vert E\Vert \le\mathrm{tol}\ll1$. So truncate the Neumann series:

$$
\hat B_k\;\equiv\;\tilde Z\Big(\sum_{j=0}^{k}(-E)^j\Big)\tilde Z^\dagger
=\tilde Z\big(I-E+E^2-\dots\big)\tilde Z^\dagger,\qquad
\Vert \hat B_kS-I\Vert \;\lesssim\;\mathrm{tol}^{\,k+1},\tag{2.8}
$$

and iterate $\hat G=(\hat B_kH-b)/a$ instead of $\bar G$. Consequences:

- **No reductions.** $\hat B_k$ is SpMVs only: $1+3k+1$ applies per step
  ($Ev=\tilde Z^\dagger(S(\tilde Zv))-v$ costs three). At $k=2$ that is 8
  SpMVs - the same flops as 2-3 PCG iterations, minus every dot product.
- **A fixed operator.** The recursion now computes $T_m(\hat G)$ *exactly* for
  one fixed matrix. $\hat B_k$ is SPD (a positive polynomial of $E$ congruent
  by $\tilde Z$), so $\hat B_kH$ is similar to the Hermitian
  $\hat B_k^{1/2}H\hat B_k^{1/2}$: real spectrum, and the $\epsilon_s$ pad
  absorbs its $O(\mathrm{tol}^{k+1})$ displacement. No divergence mode exists.
  Run the Lanczos bounds of section 2.2 on $\hat G$ itself - the operator
  actually iterated.
- **Controlled bias, same currency as (2.7).** The error is a pure operator
  perturbation $\Vert \hat G-\bar G\Vert \lesssim\mathrm{tol}^{k+1}$, an
  energy-axis displacement amplified by $a/\eta$ like everything else:
  relative LDOS error $\sim\mathrm{tol}^{k+1}M/4$. At
  $\mathrm{tol}=10^{-5}$: $k=1$ gives $4\times10^{-7}$ (marginal), **$k=2$
  gives $4\times10^{-12}$** (negligible). Default $k=2$.
- **One primitive for everything.** The sampler (4.7) is
  $\tilde Z\,p(E)\,\boldsymbol\xi$ with $p\approx(1+x)^{-1/2}$; the inner
  solve is $\tilde Z\,p(E)\,\tilde Z^\dagger$ with $p\approx(1+x)^{-1}$.
  Implement one routine "apply $\tilde Z\cdot p(E)\cdot$" and both fall out.

Keep PCG (section 6.7) as the fallback for a $\tilde Z$ too loose for (2.8) -
$\mathrm{tol}\gtrsim10^{-2}$, where the series needs many terms and CG's
adaptivity wins.

---

## 3. The filter and its square root

### 3.1 Why the square root

Write $\delta_\eta$ for a normalized Gaussian of width $\eta$. The key
structural choice is to factor

$$
\delta_\eta(E-\tilde H)=F_E F_E^\dagger,\qquad F_E\equiv\delta_\eta^{1/2}(E-\tilde H),\tag{3.1}
$$

and to estimate $\rho$ as a **norm** rather than a bilinear form. Because the
square root of a Gaussian is again a Gaussian,

$$
\delta_\eta^{1/2}(x)=(2\pi\eta^2)^{-1/4}\exp\!\Big(-\frac{x^2}{4\eta^2}\Big)
\quad\text{- a Gaussian of width }\sigma_F=\sqrt2\,\eta,\tag{3.2}
$$

$F_E$ is *cheaper* to expand than $\delta_\eta$ itself (wider => fewer
moments), and it is real symmetric whenever $\tilde H$ is. This is the
LCAO/energy-resolved analogue of the $\sqrt{f(H)}$ vs $f(H)$ distinction in the
source paper's SM section 2. Section 3.5 quantifies what it buys - and the
moment count is the *smallest* part of it.

### 3.2 Chebyshev expansion

With $\bar H=(\tilde H-b)/a$, $\bar E=(E-b)/a$, $\bar\eta=\eta/a$:

$$
F_E \;=\; \mathcal N_\eta\sum_{m=0}^{M}c_m(\bar E)\,T_m(\bar H),
\qquad \mathcal N_\eta=(2\pi\eta^2)^{-1/4},\tag{3.3}
$$

$$
c_m(\bar E)=\frac{2-\delta_{m0}}{\pi}\int_{-1}^{1}\frac{dx}{\sqrt{1-x^2}}\,
\exp\!\Big(-\frac{(\bar E-x)^2}{4\bar\eta^2}\Big)T_m(x).\tag{3.4}
$$

**Evaluate (3.4) by DCT**, not closed form: $x=\cos\theta$ turns it into
$c_m=\frac{2-\delta_{m0}}{\pi}\int_0^\pi g(\cos\theta)\cos m\theta\,d\theta$, a
cosine transform computed to machine precision by an $N_{\rm FFT}\ge 4M$ point
DCT in negligible time. (Closed forms in modified Bessel functions exist but
overflow badly for small $\bar\eta$.)

### 3.3 Truncation order

Near $\theta_E=\arccos\bar E$ the integrand is Gaussian in $\theta$ with width
$\Delta\theta=2\bar\eta/\sin\theta_E$ (Jacobian $dx=-\sin\theta\,d\theta$). Its
cosine transform decays as

$$
|c_m|\;\sim\;\exp\!\Big(-\frac{m^2\eta^2}{a^2\,(1-\bar E^2)}\Big)
\quad\Longrightarrow\quad M \;\ge\; \frac{a}{\eta}\sqrt{\ln(1/\epsilon_M)}\tag{3.5}
$$

at band centre, and fewer moments near band edges - the $x=\cos\theta$ map
stretches edge features, so they resolve more cheaply. Note the **Gaussian
(super-exponential)** decay: this is why the Gaussian filter is preferred over
the Lorentzian $-\mathrm{Im}\,G(E+i\eta)/\pi$, whose Chebyshev coefficients
decay only as $e^{-m\eta/a}$.

Practical: $\epsilon_M=10^{-8}\Rightarrow M\approx4.3\,a/\eta$;
$\epsilon_M=10^{-12}\Rightarrow M\approx5.3\,a/\eta$. $\epsilon_M$ sets the
**dynamic range floor**: an absolute error $\sim\epsilon_M\Vert \mathbf w\Vert ^2$
contaminates gap regions, so resolving LDOS at $10^{-p}$ of the peak requires
$\epsilon_M\ll10^{-p}$.

### 3.4 Damping (optional)

No kernel damping is needed for positivity (section 4.2 gives it
structurally). If ringing from hard truncation is visible in a very-low-DOS
region, apply Lorentz damping $g_m=\sinh[\lambda(1-m/M)]/\sinh\lambda$ with
$\lambda\simeq3$. Jackson damping is *not* recommended here: it enforces a
positivity the estimator already has, at the cost of $\sqrt2$ resolution.


### 3.5 What the square root buys over plain KPM - a theorem, not an estimate

Fair question, since KPM with a damping kernel is the standard tool. The
comparison below is exact and step by step; the statistical machinery
(moments of $b_n$, the variance lemma) is the same as section 6.3 and is
derived there in full.

**Step 0 - moments alone would not justify it.** Expanding $\delta_\eta$
directly (width $\eta$) at $\epsilon_M=10^{-8}$ needs $M=4.3\,a/\eta$;
expanding $\delta_\eta^{1/2}$ (width $\sqrt2\,\eta$) needs $M=3.0\,a/\eta$ -
exactly $\sqrt2$ fewer. Against Jackson-damped KPM even that vanishes:
Jackson resolves $\sigma_J\simeq\pi a/M=3.14\,a/M$ per moment, ours
$\eta=3.03\,a/M$ - equal to 4%.

**Step 1 - the two estimators, built from identical data.** Both start from
the same recursion vectors $V_m=T_m(\bar H)\boldsymbol\xi$ and the same DCT
coefficient machinery; they differ only in *where* the stochastic identity
$\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I$ is inserted into
$\delta_\eta=F_EF_E^\dagger$:

$$
\hat\rho_{\rm sqrt}=\big|\mathbf w^\dagger F_E\boldsymbol\xi\big|^2
\quad\text{(split: }\delta_\eta=F_E\,\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]\,F_E^\dagger\text{)},
$$

$$
\hat\rho_{\rm KPM}=\mathrm{Re}\big[(\mathbf w^\dagger\delta_\eta(E-\tilde H)\boldsymbol\xi)(\boldsymbol\xi^\dagger\mathbf w)\big]
\quad\text{(one-sided: }A\approx\tfrac12(\delta_\eta\boldsymbol\xi\boldsymbol\xi^\dagger+\boldsymbol\xi\boldsymbol\xi^\dagger\delta_\eta)\text{)}.
$$

The second is what "stochastic KPM for $A(E)\vert_{\mathcal P}$" means in
practice, whether assembled per energy or from moment matrices.

**Step 2 - both are the same quadratic form with different spectral weights.**
Insert the eigenbasis exactly as in section 6.3 Step 0
($b_n=u_n^\dagger\boldsymbol\xi$, $\psi_n=\mathbf w^\dagger u_n$,
$\delta_n=\delta_\eta(E-\varepsilon_n)=f_n^2$), and for the KPM form
symmetrize the real part by relabeling $n\leftrightarrow m$. Both estimators
become

$$
\hat F=\sum_{nm}g_{nm}\,H_{nm}\,b_n\bar b_m,
\qquad
H_{nm}=\psi_n\overline{\psi_m}\ \text{(pointwise)}
\ \text{or}\ \textstyle\int h\,\psi_n\overline{\psi_m}\ \text{(aggregated)},
\tag{3.6}
$$

with the **only** difference in the weight $g_{nm}$:

$$
g^{\rm sqrt}_{nm}=f_nf_m=\sqrt{\delta_n\delta_m},
\qquad
g^{\rm KPM}_{nm}=\tfrac12(\delta_n+\delta_m).
$$

Their diagonals coincide, $g_{nn}=\delta_n$, so both are unbiased with the
same mean $\sum_n\delta_nH_{nn}$.

**Step 3 - variance of a quadratic form.** For complex random phases the
pairing identity of section 6.3 Step 1 gives, for any Hermitian kernel
$A_{nm}=g_{nm}H_{nm}$,

$$
\mathrm{Var}\Big[\sum_{nm}A_{nm}b_n\bar b_m\Big]
=\sum_{n\ne m}|A_{nm}|^2+v_b\sum_n A_{nn}^2 ,
\tag{3.7}
$$

(the two pairings $n{=}m,k{=}l$ and $n{=}l,k{=}m$, minus the IPR remainder).

*Why no $v_b$ on the off-diagonal term:* every slot contributes (variance of
its random coefficient) x (weight squared), and the off-diagonal coefficient
$b_n\bar b_m$ ($n\ne m$) has mean zero and variance

$$
\mathrm{Var}(b_n\bar b_m)
=\mathbb E[|b_n|^2|b_m|^2]-|\mathbb E[b_n\bar b_m]|^2
=\mathbb E|b_n|^2\,\mathbb E|b_m|^2=1 ,
$$

a product of *second* moments only - and $\mathbb E|b_n|^2=1$ is pinned by
unbiasedness ($\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I$) itself. Only
the diagonal slot $|b_n|^2$ involves a fourth moment of a single component,
$\mathrm{Var}(|b_n|^2)=v_b$, so only there can the sampler distribution enter.
(Real $\boldsymbol\xi$ with real eigenvectors: $\mathbb E[(b_n\bar b_m)^2]=1$
instead of $0$, doubling the off-diagonal coefficient - the $v_z=2$ of section
6.3.) Consequence: **sampler engineering can only tune the diagonal term**; the
coherent cross-term noise of section 6.3 is frozen at coefficient 1 by
normalization and is removable only by dephasing, aggregation, or k-averaging.

The diagonal term is **identical** for both estimators - so on global traces
($h=1$, $H_{nm}=\delta_{nm}$) they are literally the same random variable.
Everything is decided off-diagonal.

**Step 4 - the theorem.** By the identity
$\tfrac14(x+y)^2-xy=\tfrac14(x-y)^2$ applied to $x=\delta_n$, $y=\delta_m$:

$$
\mathrm{Var}[\hat F_{\rm KPM}]-\mathrm{Var}[\hat F_{\rm sqrt}]
=\frac14\sum_{n\ne m}\big(\delta_n-\delta_m\big)^2\,|H_{nm}|^2\;\ge\;0 ,
\tag{3.8}
$$

**for every point, every energy, every readout $h$, every spectrum.** KPM is
never better; equality holds only when the filter is flat across all pairs the
readout couples ($h=1$, or $\eta\to\infty$). The excess is concentrated
exactly where $\delta_n\gg\delta_m$: one state inside the window paired with
one far outside - the white, unfiltered side of the bilinear form. (Verified
numerically during review: 60-state toy, complex phases, $4\times10^5$
samples - lemma (3.7) and identity (3.8) reproduce the empirical variances of
both estimators; the KPM excess was 27x even for a window holding a quarter of
the spectrum.)

**Step 5 - evaluating the theorem: how large is the excess at one point?**
Call the left side of (3.8) the **excess**,
$\Delta\equiv\mathrm{Var}_{\rm KPM}-\mathrm{Var}_{\rm sqrt}$; the goal of this
step is to evaluate $\Delta$ pointwise and divide by
$\mathrm{Var}_{\rm sqrt}=\rho_\eta^2$ (Eq. 6.1) to get the variance *ratio* -
i.e. how many times more samples plain KPM needs at this point.

Pointwise, $|H_{nm}|^2=|\psi_n|^2|\psi_m|^2$, so

$$
\Delta=\frac14\sum_{n\ne m}(\delta_n-\delta_m)^2\,|\psi_n|^2|\psi_m|^2 .
$$

Split the pairs by position relative to the window. Pairs with both states
inside have $\delta_n\approx\delta_m$ - their $(\delta_n-\delta_m)^2$ is small.
The dominant pairs have $n$ inside ($\delta_n$ large) and $m$ far outside
($\delta_m\approx0$), so $(\delta_n-\delta_m)^2\approx\delta_n^2$; each
unordered pair occurs twice in the ordered sum:

$$
\Delta\;\approx\;2\cdot\frac14\Big(\sum_n\delta_n^2|\psi_n|^2\Big)\Big(\sum_m|\psi_m|^2\Big).
$$

Both factors are known quantities:

$$
\sum_m|\psi_m(\mathbf r)|^2=\int\rho(\mathbf r,E')\,dE'
\quad\text{(the sum rule 1.5 - the point's total spectral weight)},
$$

$$
\sum_n\delta_n^2|\psi_n|^2
=\frac{1}{2\sqrt\pi\,\eta}\,\rho_{\eta/\sqrt2}(\mathbf r,E)
\approx\frac{\rho(\mathbf r,E)}{2\sqrt\pi\,\eta},
$$

the second using the Gaussian-square identity
$\delta_\eta(x)^2=\delta_{\eta/\sqrt2}(x)/(2\sqrt\pi\eta)$ (same algebra as
$\int\delta_\eta^2=(2\sqrt\pi\eta)^{-1}$ in section 6.3) and smoothness of
$\rho$ on the scale $\eta$. Therefore

$$
\frac{\mathrm{Var}_{\rm KPM}}{\mathrm{Var}_{\rm sqrt}}
=1+\frac{\Delta}{\rho_\eta^2}
\;\approx\;1+\frac{W_{\rm loc}}{4\sqrt\pi\,\eta},
\qquad
W_{\rm loc}(\mathbf r,E)\equiv\frac{\int\rho(\mathbf r,E')\,dE'}{\rho(\mathbf r,E)} ,
$$

where $W_{\rm loc}$ is the energy width the point's weight would span if
spread at its current height. **$W_{\rm loc}$ is not an input parameter - it is
computable from the run itself**: its numerator is the sum-rule value
$\boldsymbol\phi^TS^{-1}\boldsymbol\phi$ that ladder test 1 already produces for
free, and its denominator is the map being computed. Its meaning and range:

- For a flat LDOS, $W_{\rm loc}$ = the spectral support width. Anchor from
  this repo: the graphene $p_z$ spectrum spans $17.2$ eV
  (`energy_bounds=(-8.6, 8.6)` in `test_graphene_pz_dos.py`), so a mid-band
  point has $W_{\rm loc}\sim15$-$20$ eV. The pencil half-width
  $a\approx100$ eV is **not** the relevant scale - the high-lying states that
  inflate $a$ carry little weight at physical points, and $W_{\rm loc}$ is
  weight-weighted by construction.
- Limits: window swallowing the whole spectrum $\Rightarrow$ ratio $\to1$
  (consistent with (3.8) for a flat filter); gaps and band tails $\Rightarrow$
  $W_{\rm loc}$ diverges - the KPM penalty is worst exactly where LDOS maps
  are read.

Representative number: $W_{\rm loc}=17$ eV (the graphene $p_z$ span),
$\eta=25$ meV: $W_{\rm loc}/(4\sqrt\pi\eta)\approx96$ - **plain KPM needs about
a hundred times more samples per mid-band point**. Recommended
practice: report the measured $1+W_{\rm loc}/(4\sqrt\pi\eta)$ alongside the map -
it is the per-point "KPM-equivalent sample multiplier", free to evaluate.
Aggregated readouts interpolate per (6.2b): both estimators improve, the
ordering (3.8) is preserved, and they merge only at the full trace, where
$H_{nm}=\delta_{nm}$ makes $\Delta=0$ exactly.

**Positivity is the same choice seen structurally.** $g^{\rm sqrt}$ is the
unique weight in (3.6) that factorizes, $g_{nm}=f_nf_m$, making $\hat F$ a
norm: $\hat\rho\ge0$ per sample, Gamma-concentrated (section 6.4), no
negative LDOS in gaps - without a damping kernel, which is what frees the
truncation of section 3.3 from Jackson's $\pi a/M$. The bilinear form goes
negative sample by sample.

**Memory for the full map.** Assembling from KPM moment matrices stores
$M\cdot\mathrm{nnz}$ against our $N_E\cdot\mathrm{nnz}$ - $\sim85\times$ at the
worked example; per-energy KPM assembly matches our memory but keeps (3.8)'s
variance excess.

**Where plain KPM *does* win: few points.** Deterministic single-vector KPM -
recursion started from $\mathbf w(\mathbf r)$ itself, moments
$\mu_m=\mathbf w^\dagger T_m\mathbf w$ - has *no* stochastic error at all and
costs one recursion per point. For $N_{\rm pt}$ requested points versus
$S_{\rm smp}$ samples, it wins whenever $N_{\rm pt}<S_{\rm smp}$: an STS
curve at a handful of tip positions should use it. The stochastic square-root
estimator is for maps, $N_{\rm grid}\sim10^8\gg S_{\rm smp}$; there its cost
advantage is $N_{\rm grid}/S_{\rm smp}\sim10^{5}$.

**And versus stochastic KPM it costs nothing.** Same $V_m$, same DCT; only the
coefficient set changes. Strictly better at identical flops, by (3.8).

**Relation to prior art.** The split-insertion principle is established: for
occupations, stochastic DFT computes densities from square-root-filtered random
orbitals $|(\sqrt{f}\,\xi)(\mathbf r)|^2$ (Baer, Neuhauser & Rabani, PRL 111,
106402 (2013) and lineage), and the source paper (Zhou & Yuan,
arXiv:2203.03465, SM sec. 2) analyzes $\sqrt f$ vs $f$ directly. For energies,
the quasi-eigenstate method (Yuan, De Raedt & Katsnelson, PRB 82, 115448
(2010); the engine of TBPM/TBPLaS) is a filtered-norm LDOS estimator in
disguise - our sections 3.1-3.2 differ in engineering (choosing
$F=\delta_\eta^{1/2}$ deliberately, DCT coefficients, band-limit priced at
$\sqrt2\eta$) rather than principle. Deterministic local-vector KPM for
few-site LDOS is textbook (Weisse et al., RMP 78, 275 (2006); the Kwant KPM
tutorial states the few-sites rule verbatim). Nonorthogonal stochastic
electronic structure exists (e.g. Gaussian-basis sDFT, arXiv:2504.17115), so
section 4.1's metric-correct state has precedent, though the trace-vs-resolved
distinction is usually implicit. What we have *not* found stated in the
literature: the exact readout-resolved identity (3.8) and its corollaries
(trace coincidence, the computable per-point penalty $W_{\rm loc}/(4\sqrt\pi\eta)$),
and the two-regime pointwise/aggregated variance analysis of section 6.3 for
the single-shot (non-dephased) Chebyshev estimator. Those should be treated as
results of this design until a prior statement surfaces.

---

## 4. The stochastic estimator

### 4.1 Metric-correct random states

The resolution of identity in a nonorthogonal basis is
$\hat P=\sum_{\mu\nu}|\phi_\mu\rangle(S^{-1})_{\mu\nu}\langle\phi_\nu|$
(check: $\hat P|\phi_\lambda\rangle=\sum_{\mu\nu}|\phi_\mu\rangle(S^{-1})_{\mu\nu}S_{\nu\lambda}=|\phi_\lambda\rangle$).
Consequently the random **coefficient** vector must satisfy

$$
\mathbb E\big[\mathbf c_0\mathbf c_0^\dagger\big]=S^{-1},\quad\text{realized by}\quad
\mathbf c_0=Z\boldsymbol\xi,\;\;\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I_{N'}.\tag{4.1}
$$

> **This is the principal trap in porting the method to LCAO.** Drawing random
> phases directly on the NAO coefficients
> ($\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=I$) biases $\rho$ by
> $O(\Vert S-I\Vert )$, which for DZP NAOs is a first-order error, not a small
> correction.

**Where (4.1) comes from - it is not an algebraic trick.** Let
$|g\rangle=\sqrt{\Delta V}\sum_pg_p|\mathbf r_p\rangle$ be real-space white noise
on a quadrature mesh, $\mathbb E[g_pg_q]=\delta_{pq}$, so
$\mathbb E|g\rangle\langle g|=\Delta V\sum_p|\mathbf r_p\rangle\langle\mathbf r_p|\to\hat 1$.
Project it onto the basis span: $\hat P|g\rangle=\sum_\mu|\phi_\mu\rangle c_{0\mu}$
with

$$
\mathbf c_0=S^{-1}\boldsymbol\xi,\qquad
\xi_\mu=\langle\phi_\mu|g\rangle=\sqrt{\Delta V}\sum_p\phi_\mu(\mathbf r_p)g_p,\tag{4.1a}
$$

$$
\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=\Delta V\,\Phi^T\Phi\equiv S_g,
\qquad \mathbb E[\mathbf c_0\mathbf c_0^\dagger]=S^{-1}S_gS^{-1}\;\longrightarrow\;S^{-1}.
$$

So the metric-correct state **is** the LCAO representation of the same
real-space random state the source method uses on its uniform grid, and the
naive $\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=I$ is the error of treating NAO
coefficients as if they were amplitudes on an orthonormal mesh. Section 4.3
grounds sampler (S3) below.

**Which estimators need this, and which do not.** A *trace* is invariant under
similarity, and $G=S^{-1}H$ is similar to $\tilde H$, so

$$
\mathbb E\big[\boldsymbol\xi^\dagger f(G)\boldsymbol\xi\big]=\mathrm{Tr}\,f(G)=\sum_nf(\varepsilon_n)
\qquad\text{already holds for }\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I.
$$

Total DOS, Kubo conductivity and MSD are traces, so the plain unit-norm random
vector of `SampleVectors` is **correct** for them and must not be "fixed".
$\rho(\mathbf r,E)$ is not a trace - it is a *resolved matrix element* of
$A(E)$, and $\mathbb E[\hat D]=P(G)\,\mathbb E[\mathbf c_0\mathbf c_0^\dagger]\,P(G)^\dagger$
collapses to $A_\eta$ only when the covariance is $S^{-1}$. The same applies to
the density matrix on the density path. Metric correction is the price of
resolution, not of nonorthogonality per se.

Choice of $\boldsymbol\xi$:

- **Complex random phases** $\xi_i=e^{i\theta_i}$: $\mathbb E|b_n|^4-1=1$.
- **Real Rademacher** $\xi_i=\pm1$: allowed when $H,S$ real (Gamma-only);
  halves flops and memory but doubles $\mathrm{Var}(|b_n|^2)$ to 2. Net wash -
  choose for memory.
- Gaussian vectors are strictly worse than both (no IPR gain).

Write $v_b\equiv\mathrm{Var}(|b_n|^2)$: $v_b=1$ (complex phase), $2$ (real).

### 4.2 The estimator

For sample $s$ and energy $E$, define the **quasi-eigenstate coefficient
vector**

$$
\boldsymbol\chi_s(E)\;=\;Z\,F_E\,\boldsymbol\xi_s\;\in\;\mathbb C^{N},\tag{4.2}
$$

and the estimators

$$
\hat D(E)=\frac{1}{S_{\rm smp}}\sum_{s=1}^{S_{\rm smp}}
\boldsymbol\chi_s(E)\boldsymbol\chi_s(E)^\dagger\Big|_{\mathcal P},
\qquad
\hat\rho(\mathbf r,E)=\boldsymbol\phi(\mathbf r)^T\hat D(E)\,\boldsymbol\phi(\mathbf r).\tag{4.3}
$$

Equivalently, $\hat\rho=\frac{1}{S_{\rm smp}}\sum_s|\psi_{s,E}(\mathbf r)|^2$
with $\psi_{s,E}(\mathbf r)=\boldsymbol\phi(\mathbf r)^T\boldsymbol\chi_s(E)$ -
a genuine real-space wavefunction, a random superposition of eigenstates within
$\eta$ of $E$.

**Unbiasedness.** Using $\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I$ and
$F_E=F_E^\dagger$:

$$
\mathbb E[\hat D]=Z\,F_E\,I\,F_E^\dagger Z^\dagger
=Z\,\delta_\eta(E-\tilde H)\,Z^\dagger=A_\eta(E),\tag{4.4}
$$

exactly, for every $S_{\rm smp}$ and every $M$ (up to $\epsilon_M$).
**There is no statistical bias, only variance.**

**Positivity.** $\hat D$ is a sum of outer products, hence PSD, hence
$\hat\rho\ge0$ at every point for every sample count - regardless of damping.
This is what makes *relative*-error concentration bounds available.

**Restriction to $\mathcal P$ loses nothing** - by section 1.2 the discarded
entries never enter (4.3).

In adjoint form, (4.2) never mentions $Z$: by (2.2), $\boldsymbol\chi_s=P(\bar G)\mathbf c_0$
with $P\equiv\delta_\eta^{1/2}$ reproduces (4.4) exactly,

$$
\mathbb E[\hat D]=P(G)\,S^{-1}P(G)^\dagger=P(G)S^{-1}P(HS^{-1})
=S^{-1}P^2(HS^{-1})=A_\eta(E),\tag{4.6}
$$

using the commutation rule $P(G)S^{-1}=S^{-1}P(G^\dagger)$ implied by (2.2).
$\mathbf w(\mathbf r)=Z^\dagger\boldsymbol\phi$ of section 1.3 is a proof device
only - the algorithm assembles $\hat D$ on $\mathcal P$ and contracts with
$\boldsymbol\phi$ directly, and even sum rule (1.5) is
$\boldsymbol\phi^TS^{-1}\boldsymbol\phi$, one solve.

### 4.3 Samplers for $\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=S^{-1}$

> **Lemma (positivity requires a factor).** Suppose $\hat D=\boldsymbol\chi\boldsymbol\chi^\dagger$
> with $\boldsymbol\chi=B\boldsymbol\xi$, $\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I$,
> and $\mathbb E[\hat D]=A(E)$. Then $BB^\dagger=A=P(G)S^{-1}P(G)^\dagger$, and since
> $P(G)=\delta_\eta^{1/2}(E-G)$ is positive definite (a Gaussian is nowhere zero)
> it is invertible, so $B=P(G)Y$ with $YY^\dagger=S^{-1}$ up to a unitary. **A
> square root of $S^{-1}$ is unavoidable for any rank-1 PSD estimator.**

This is the exact sense in which (2.2) does *not* make the metric free. But the
factor is needed **once per sample vector, not once per Chebyshev step**, and -
crucially - *it does not have to be accurate*.

**Proposition (any $\tilde Z$ suffices, exactly).** Let $\tilde Z$ be any
invertible matrix and $E=\tilde Z^\dagger S\tilde Z-I$ as in (2.1). Then

$$
\mathbf c_0=\tilde Z\,(I+E)^{-1/2}\,\boldsymbol\xi,\qquad
\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I
\quad\Longrightarrow\quad
\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=S^{-1}\quad\text{exactly}.\tag{4.7}
$$

*Proof.* $E$ is Hermitian and $I+E=\tilde Z^\dagger S\tilde Z$ is SPD, so
$(I+E)^{-1/2}$ is well defined and Hermitian - and $\Vert E\Vert <1$ itself
certifies that $\tilde Z$ is invertible, since $\tilde Z^\dagger S\tilde Z\succ0$
forces $\tilde Z$ full rank. Then

$$
\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=\tilde Z(I+E)^{-1}\tilde Z^\dagger
=\tilde Z(\tilde Z^\dagger S\tilde Z)^{-1}\tilde Z^\dagger
=\tilde Z\tilde Z^{-1}S^{-1}\tilde Z^{-\dagger}\tilde Z^\dagger=S^{-1}.\qquad\square
$$

**No property of $\tilde Z$ was used.** The symmetrization $(I+E)^{-1/2}$
absorbs whatever error the construction left behind. What *does* depend on
$\Vert E\Vert $ is only how $(I+E)^{-1/2}$ is applied. Write $W\equiv I+E=\tilde Z^\dagger S\tilde Z$,
applied matrix-free as $Wv=\tilde Z^\dagger(S(\tilde Zv))$ - three SpMVs - and
distinguish the two $\tilde Z$'s this package actually produces:

- **Newton-Schulz $\tilde Z$** (loose or purified): $\Vert E\Vert \le\mathrm{tol}\ll1$,
  and the binomial series

  $$
  (I+E)^{-1/2}=I-\tfrac12E+\tfrac38E^2-\tfrac{5}{16}E^3+\dots\tag{4.8}
  $$

  converges at rate $\mathrm{tol}$: $k$ terms leave $O(\mathrm{tol}^{k+1})$.
- **Graph-seed $\tilde Z$** (`isqrtm_init` subgraph diagonalization alone, no
  purification - used in practice as a preconditioner for near-singular $S$):
  here $E$ is $O(1)$ and (4.8) is useless or divergent. Empirically PCG on $W$
  converges in $\sim50$ rounds, which pins $\kappa(W)\approx20$-$30$ via
  $n_{\rm it}\simeq\tfrac12\sqrt\kappa\ln(2/\tau)$. Then apply
  $(I+E)^{-1/2}=q(W)$ with $q$ the **Chebyshev approximant of $x^{-1/2}$ on the
  measured interval** $[\lambda_{\min}(W),\lambda_{\max}(W)]$ (~20 Lanczos
  steps on $W$, which simultaneously certify $\lambda_{\min}(W)>0$, i.e.
  invertibility of $\tilde Z$ - never assume it). Degree
  $k_q\approx\tfrac12\sqrt{\kappa(W)}\ln(2/\epsilon_q)\approx50$ at
  $\epsilon_q=10^{-8}$ - the same $\sim50$ as the CG rounds, because it is the
  same conditioning. Bias $2\epsilon_q$, a priori; set $\epsilon_q\le\epsilon_M$.

(4.8) is the small-$\Vert E\Vert $ special case of $q(W)$; one routine
"apply $\tilde Z\,q(W)$" covers both.

| $\tilde Z$ | applicator | applies of $W$ | cost per sample vector |
|---|---|---|---|
| purified, $\Vert E\Vert =10^{-10}$ | $\mathbf c_0=\tilde Z\boldsymbol\xi$ | 0 | 1 SpMV |
| loose NS, $\Vert E\Vert =10^{-5}$ | (4.8), 2 terms | 2 | ~7 SpMV |
| graph seed, $\kappa(W)\sim25$ | Chebyshev $x^{-1/2}$, degree ~50 | ~50 | ~150 SpMV |

Even the last row is $\sim0.1\%$ of the $\sim9M\approx1.5\times10^5$ SpMVs per
sample in the recursion: **the sampler is effectively free in every regime,
because the factor is paid once per sample, not once per step.** The place a
crude $\tilde Z$ *does* hurt is Mode A's inner solve, which pays its polynomial
degree per Chebyshev step: with the graph seed, $\hat B_k$ of (2.8) generalizes
the same way (fixed Chebyshev approximant of $x^{-1}$ on the measured interval,
still reduction-free and a fixed operator) but at degree $\sim50$ - roughly
$20\times$ the purified-regime step cost. There, one Newton-Schulz sweep pays
for itself after $\sim M$ steps if the density budget of section 2.4 permits;
if it does not (the near-singular-$S$ case), accept the degree or the 50-round
PCG - and note that near-singular $S$ also triggers section 2.2 step 3, where
the *target* $S^{-1}$ itself needs the null-space projection.

**One fallback, for near-singular $S$** - **(S3) grid Gram.** Eq. (4.1a),
$\mathbf c_0=S^{-1}\sqrt{\Delta V}\,\Phi^Tg$, reusing `calc/collocation.hpp`
transposed: the same $\Phi$ that Eq. (5.4) contracts *with* generates the state
when applied *adjointly*. Exact iff $S$ is the grid Gram; ours is from op2c
two-center integrals, so it carries $\Delta\equiv S_g-S$, sparse on
$\mathcal P$ and computable, removable to second order by
$\mathbf c_0=(I-\tfrac12S^{-1}\Delta)S^{-1}\boldsymbol\xi$. Its advantage is
structural: $\boldsymbol\xi\in\mathrm{range}(\Phi^T)$ **never excites the
near-null space of $S$**, where (4.7) amplifies by $\lambda^{-1/2}$. This is
the sampler for when section 2.2 step 3 triggers.

Use Rademacher $\boldsymbol\xi$ or $g_p=\pm1$ throughout - the covariance is
exact either way and the projections $b_n$ are asymptotically Gaussian
regardless.

### 4.4 Estimator with no factor at all (two-sided form)

If no sampler is admissible, unbiasedness survives without one; positivity does
not. Draw a plain $\boldsymbol\xi$, $\mathbb E[\boldsymbol\xi\boldsymbol\xi^\dagger]=I$, and set

$$
\boldsymbol\chi=P(\bar G)\,S^{-1}\boldsymbol\xi,\qquad
\boldsymbol\zeta=P(\bar G)\,\boldsymbol\xi,\qquad
\hat D=\tfrac12\big(\boldsymbol\chi\boldsymbol\zeta^\dagger+\boldsymbol\zeta\boldsymbol\chi^\dagger\big)\Big|_{\mathcal P}.\tag{4.10}
$$

$$
\mathbb E[\boldsymbol\chi\boldsymbol\zeta^\dagger]
=P(G)S^{-1}\,I\,P(G)^\dagger=S^{-1}P^2(G^\dagger)=A_\eta(E),\tag{4.11}
$$

so the symmetrized form is unbiased and Hermitian. One recursion on the 2-wide
block $[S^{-1}\boldsymbol\xi\,|\,\boldsymbol\xi]$ produces both.

**What it costs.** $\hat D$ is no longer PSD: $\hat\rho=\mathrm{Re}\,[u\bar v]$
with $u=\boldsymbol\phi^T\boldsymbol\chi$, $v=\boldsymbol\phi^T\boldsymbol\zeta$
can be negative on a single sample, so section 6.4's *relative* concentration
bound degrades to an additive Bernstein bound and the "physical-looking map"
property of section 4.2 is lost. The variance is inflated by the
Cauchy-Schwarz factor $\mathbb E|u|^2\mathbb E|v|^2/\rho^2\ge1$, which is $1$
only when $u\parallel v$, i.e. only when $S=I$; expect a factor of a few for
DZP. Note (4.10) reduces to (4.2) identically in the orthogonal limit.

**Recommendation: (4.2) with sampler (4.7).** Keep (4.10) as an
independent cross-check - it shares no sampler code with (4.2), so agreement
between them validates the metric correction itself (ladder test 8).

---

## 5. Algorithm

### 5.1 Chebyshev recursion (multi-vector)

Let $\Xi=[\boldsymbol\xi_1\cdots\boldsymbol\xi_{S_{\rm blk}}]$.

$$
V_0=\Xi,\qquad V_1=\bar H\Xi,\qquad V_{m+1}=2\bar HV_m-V_{m-1},\tag{5.1}
$$

with $\bar H$ the Mode B congruence $(\tilde Z^\dagger H\tilde Zv-bv)/a$ or the
Mode A operator $\hat G$ of (2.8). Accumulate, for all energies simultaneously,

$$
\mathbf X(E_k)\mathrel{+}=\mathcal N_\eta\,c_m(\bar E_k)\,V_m,\qquad k=1\dots N_E,\tag{5.2}
$$

then $\boldsymbol\chi(E_k)=Z\,\mathbf X(E_k)$.

> **One recursion serves all $N_E$ energies.** The marginal cost of an extra
> energy point is one AXPY per Chebyshev step - negligible against the SpMM.
> This is the single largest efficiency advantage over per-energy methods
> (selected inversion, shift-invert) when a dense energy grid is wanted.

Use **SpMM with $S_{\rm blk}=8$-$64$ columns**, never $S_{\rm blk}=1$ SpMV:
arithmetic intensity rises ~10x and the kernel maps onto GPU tensor pipelines.

### 5.2 Assembly on $\mathcal P$

$$
\hat D_{\mu\nu}(E_k)\mathrel{+}=\tfrac{1}{S_{\rm smp}}\chi_\mu(E_k)\overline{\chi_\nu(E_k)},
\qquad (\mu\nu)\in\mathcal P.\tag{5.3}
$$

Cost $S_{\rm smp}N_E\,\mathrm{nnz}(S)$ - subdominant.

### 5.3 Grid contraction

Partition space into boxes of side $\sim2$-$4$ A. For box $\mathcal B$ with
$n_{\rm pts}$ grid points, let $\mathcal L$ be the orbitals (including periodic
images) whose cutoff spheres intersect $\mathcal B$, $|\mathcal L|=n_{\rm loc}$.
Build once $\Phi\in\mathbb R^{n_{\rm pts}\times n_{\rm loc}}$,
$\Phi_{p\mu}=\phi_\mu(\mathbf r_p-\boldsymbol\tau_\mu-\mathbf R_\mu)$, by cubic-spline
radial tables times recursively generated real spherical harmonics. Then per
energy

$$
\boldsymbol\rho_{\mathcal B}(E)=\mathrm{rowsum}\Big[\big(\Phi\,\hat D_{\mathcal L\mathcal L}(E)\big)\circ\Phi\Big]\tag{5.4}
$$

- one GEMM plus one Hadamard-and-reduce. $\Phi$ is amortized over all $N_E$.

**Two algebraically identical routes.** By the sparsity theorem 1.2,

$$
\boldsymbol\phi^T\big(\boldsymbol\chi\boldsymbol\chi^\dagger\vert_{\mathcal P}\big)\boldsymbol\phi
=\big|\boldsymbol\phi^T\boldsymbol\chi\big|^2
$$

exactly, so the sample index can be collapsed on either side of the grid:

- **(5.3a) assemble-then-contract** (above): accumulate $\hat D$ on
  $\mathcal P$ over all samples, then one $\Phi$ GEMM per box per energy.
  Grid flops $2N_EN_{\rm grid}n_{\rm loc}^2$; memory $N_E\,\mathrm{nnz}$ for
  $\hat D$ (the dominant allocation of section 7).
- **(5.3b) square-on-grid**: per sample block, evaluate the real-space random
  wavefunction $\psi_s(\mathbf r_p)=[\Phi\,\boldsymbol\chi_{\mathcal L}]_p$
  (one thin GEMM) and accumulate $|\psi_s|^2$ into $\rho$ directly. Grid flops
  $2S_{\rm smp}N_EN_{\rm grid}n_{\rm loc}$; **no $\hat D$ is ever formed** -
  the $N_E\,\mathrm{nnz}$ memory item vanishes, and positivity is manifest
  per sample.

**Crossover: $S_{\rm smp}$ vs $n_{\rm loc}$.** (5.3b) wins on flops whenever
$S_{\rm smp}\lesssim n_{\rm loc}\ (\sim10^2$-$10^3)$ - the common LDOS regime
$S_{\rm smp}\sim10^2$ - and wins on memory always. (5.3a) wins for
$S_{\rm smp}\gg n_{\rm loc}$, and is in any case the structure the density
path needs ($\hat D$ on $\mathcal P$ is its output). (5.3b) re-enters the grid
stage once per sample block; $\Phi$ per box amortizes over the $N_E$ energies
within each pass, so its build cost stays subdominant.

**k-space convention for (5.3b).** No obstacle, because the LDOS has no
cross-k interference: different $\mathbf k$ are orthogonal eigenstates
(and independent random draws), so the k-sum sits *outside* the square,

$$
\hat\rho(\mathbf r,E)=\sum_{\mathbf k}w_{\mathbf k}\,
\frac{1}{S_{\rm smp}}\sum_s\big|\boldsymbol\phi^{\mathbf k}(\mathbf r)^T\boldsymbol\chi^{\mathbf k}_s(E)\big|^2,
\qquad
\phi^{\mathbf k}_\mu(\mathbf r)=e^{i\mathbf k\cdot\mathbf R^*_\mu}\,\phi_\mu(\mathbf r-\boldsymbol\tau_\mu-\mathbf R^*_\mu),
$$

with $\mathbf R^*_\mu$ the minimum image - the collocation's unique-image guard
(cell width $>2r^{\rm cut}$) makes the Bloch sum a single phased term, so
$\Phi$ merely becomes complex per $\mathbf k$. This squares per
$(\mathbf k,s)$ and $w_{\mathbf k}$-sums, and is identical to (5.3a)'s
$\hat D(\mathbf R,E)$ Fourier assembly followed by real contraction.

### 5.4 Pseudocode - Mode B (orthogonalized)

Correct only when `tol` satisfies (2.7); for LDOS at fine $\eta$ use 5.5.

```
# ---- setup ----
P            = sparsity_pattern(S)
Zt           = isqrtm(S, tol, sptol)                     # PURIFIED: tol <= 4*eps/M
E            = Zt^H @ S @ Zt - I ;  assert |E| <= tol    # isqrtm reports this
eps_min, eps_max = lanczos_bounds(Zt^H @ H @ Zt, iters=30)
a, b         = 0.5*(eps_max-eps_min)*(1+0.01), 0.5*(eps_max+eps_min)
M            = ceil(4.3 * a / eta)                       # eps_M = 1e-8
c[m][k]      = dct_chebyshev_coeffs(E_k, eta, a, b, M)   # (M+1) x N_E

# ---- spectral stage ----
Dhat = zeros_sparse(P, N_E)
for sblk in range(0, S_smp, S_blk):
    Xi   = random_phases(N, S_blk)                       # xi, not c_0
    Vm1, Vm = Xi, Hbar @ Xi                              # Hbar v = (Zt^H H Zt v - b v)/a
    X[k] = N_eta * (c[0][k]*Vm1 + c[1][k]*Vm)  for all k
    for m in 2..M:
        Vp  = 2*(Hbar @ Vm) - Vm1                        # SpMM  <-- hot loop
        X[k] += N_eta * c[m][k] * Vp   for all k         # AXPY
        Vm1, Vm = Vm, Vp
    for k in 1..N_E:
        chi = Zt @ X[k]                                  # apply Z, one SpMM
        Dhat[k] += chi @ chi^H  restricted to P
Dhat /= S_smp

# ---- grid stage, route (5.3a); route (5.3b) instead squares Phi @ X[k]
# per sample block inside the spectral loop and never forms Dhat ----
for box in spatial_boxes:
    Phi  = eval_orbitals(box, dr)                        # n_pts x n_loc
    for k in 1..N_E:
        rho[box,k] = rowsum((Phi @ Dhat[k][L,L]) * Phi)  # GEMM
    stream_write(rho[box,:])
```

### 5.5 Pseudocode - Mode A (adjoint)

Same $\tilde Z$ as 5.4 at its loose first sweep - the configuration chosen
when the 2.4 density measurement rules Mode B out; replaces the setup and
the operator apply. Assembly and grid stages are untouched.

```
# ---- setup: no purification ----
P            = sparsity_pattern(S)
Zt           = isqrtm(S, tol=1e-5, sptol=1e-5)      # loose first sweep; Mode B rejected by 2.4
E_apply      = lambda v: Zt^H @ (S @ (Zt @ v)) - v  # matrix-free, ||E|| <= tol
eps_min, eps_max = lanczos_bounds_pencil(H, S, Zt, iters=30)     # spectrum of G
a, b         = 0.5*(eps_max-eps_min)*(1+0.01), 0.5*(eps_max+eps_min)
M            = ceil(4.3 * a / eta)
tau_cg       = min(1e-10, 4*eps_target/M)           # only if PCG fallback; (6.4)
k_inner      = 2                                    # (2.8): bias tol^3 * M/4 ~ 4e-12

# ---- spectral stage ----
for sblk in range(0, S_smp, S_blk):
    Xi   = random_phases(N, S_blk)
    # metric-correct state, Prop. (4.7): c0 = Zt (I+E)^-1/2 Xi, series (4.8)
    W    = Xi - 0.5*E_apply(Xi) + 0.375*E_apply(E_apply(Xi))     # k=2 -> O(tol^3)
    C0   = Zt @ W
    Xm1  = C0
    Xm   = Gbar(C0)                       # t = H@C0 ; solve S z = t ; (z-b*C0)/a
    Ym1, Ym = S@C0, S@Xm                  # tracked by (2.5), 2 AXPYs/step
    X[k] = N_eta*(c[0][k]*Xm1 + c[1][k]*Xm)  for all k
    for m in 2..M:
        t   = H @ Xm                                     # SpMV
        z   = Sinv_poly(Zt, t, k_inner)   # Zt(I-E+E^2)Zt^H t: 8 SpMV, 0 reductions (2.5)
        Xp  = 2*(z - b*Xm)/a - Xm1
        Yp  = 2*(t - b*Ym)/a - Ym1                       # free covariant iterate
        X[k] += N_eta * c[m][k] * Xp   for all k
        assert dot(Xp, Yp) <= dot(C0, S@C0) * (1+tol)    # ||X_m||_S monitor (2.3);
                                                         # infrequent: it is the ONLY reduction
        Xm1, Xm, Ym1, Ym = Xm, Xp, Ym, Yp
    for k in 1..N_E:
        Dhat[k] += X[k] @ X[k]^H  restricted to P        # chi == X[k]; no Z
Dhat /= S_smp
```

Differences from 5.4: `Zt` is never purified; and it disappears entirely from
the energy loop, because $\boldsymbol\chi=\mathbf X(E_k)$ already - by (2.2) the
$Z$ that Mode B applies to every $\mathbf X(E_k)$ has been absorbed into the
operator. $\tilde Z$ survives in only two places, the PCG preconditioner and the
one-off sampler, and in neither does its accuracy reach the answer.

---

## 6. Error analysis

$$
\hat\rho-\rho=\underbrace{(\rho_\eta-\rho)}_{\rm (a)\ broadening}
+\underbrace{(\rho_{\eta,M}-\rho_\eta)}_{\rm (b)\ truncation}
+\underbrace{(\hat\rho-\rho_{\eta,M})}_{\rm (c)\ sampling}
+\underbrace{O(\mathrm{tol}\cdot M)\ \text{(B)}\ \text{or}\ O(\mathrm{tol}^{k+1})\ \text{(A)}}_{\rm (d)\ metric}
+\underbrace{O(\tau_{\rm CG}M)}_{\rm (e)\ inner\ solve}.
$$

### 6.1 (a) Broadening - deterministic

$\rho_\eta=\rho*\delta_\eta$, hence
$|\rho_\eta-\rho|\le\frac{\eta^2}{2}\sup_E|\partial_E^2\rho|$. Independent of
$N$ and $S_{\rm smp}$. Not an error to eliminate - it *is* the energy
resolution, and should be reported as such.

> **Resolution floor.** $\eta$ must exceed the mean level spacing $\Delta$ near
> $E$, else "LDOS" is a comb of individual states. Enforce $\eta\ge3\Delta(E)$
> and warn otherwise. Hard constraint for small cells at Gamma, and also the
> condition for any self-averaging at all.

### 6.2 (b) Truncation - deterministic

$|\rho_{\eta,M}-\rho_\eta|\lesssim\epsilon_M\,\Vert \mathbf w(\mathbf r)\Vert ^2$, with
$\epsilon_M=e^{-M^2\eta^2/a^2}$. Independent of $N$ and $S_{\rm smp}$.
Absolute, not relative - it sets dynamic range, not peak accuracy.

### 6.3 (c) Sampling - the full derivation

*(Re-derived during implementation review; the first draft's sampling-error
claims held only in the dephased/aggregated regime. Every step is explicit.)*

**Step 0 - the estimator at one point is a coherent sum.** Fix one energy $E$
and one grid point $\mathbf r$, one sample $s$. The algorithm forms
$\boldsymbol\chi=ZF_E\boldsymbol\xi$ (Eq. 4.2), assembles
$\hat D=\boldsymbol\chi\boldsymbol\chi^\dagger\vert_{\mathcal P}$, and contracts
$\hat\rho_1=\boldsymbol\phi^T\hat D\boldsymbol\phi$ (Eq. 5.4). By the sparsity
theorem 1.2 the restriction to $\mathcal P$ drops nothing, so

$$
\hat\rho_1=\big|\boldsymbol\phi(\mathbf r)^T\boldsymbol\chi\big|^2=|z|^2,
\qquad
z\equiv\mathbf w^\dagger F_E\,\boldsymbol\xi,
\qquad
\mathbf w=Z^\dagger\boldsymbol\phi(\mathbf r),
$$

using $\boldsymbol\phi^TZ=\mathbf w^\dagger$ (real $\boldsymbol\phi$). Insert the
eigenbasis of $\tilde H$: $\tilde Hu_n=\varepsilon_nu_n$, $u_n^\dagger u_m=\delta_{nm}$.
Since $c_n=Zu_n$, the amplitude of eigenstate $n$ at the point is
$\mathbf w^\dagger u_n=\boldsymbol\phi^TZu_n=\psi_n(\mathbf r)$. With
$f_n\equiv\delta_\eta^{1/2}(E-\varepsilon_n)$ (the filter eigenvalue) and
$b_n\equiv u_n^\dagger\boldsymbol\xi$ (the random amplitude),

$$
z=\sum_n c_n\,b_n,\qquad c_n\equiv f_n\,\overline{\psi_n(\mathbf r)},
$$

$$
a_n\equiv|c_n|^2=\delta_\eta(E-\varepsilon_n)\,|\psi_n(\mathbf r)|^2\ \ge0,
\qquad
\rho_\eta(\mathbf r,E)=\sum_na_n .
$$

$a_n$ is eigenstate $n$'s contribution to the broadened LDOS at this point;
everything below is statistics of $z=\sum_nc_nb_n$.

**Step 1 - moments of the random amplitudes $b_n$.** Take
$\xi_i=e^{i\theta_i}$, i.i.d. uniform phases. Then $\mathbb E[\xi_i]=0$,
$\mathbb E[\xi_i\xi_j]=0$, $\mathbb E[\xi_i\bar\xi_j]=\delta_{ij}$, and the only
nonzero fourth moments are pairings:

$$
\mathbb E[\xi_i\bar\xi_j\xi_k\bar\xi_l]
=\delta_{ij}\delta_{kl}+\delta_{il}\delta_{kj}-\delta_{ijkl},
$$

the last term because $|\xi_i|^4=1$ (both pairings coincide when all four
indices are equal, and are then counted twice). Transforming by the unitary
$U=[u_1\dots u_{N'}]$, $b_n=\sum_i\bar U_{in}\xi_i$:

$$
\mathbb E[b_n\bar b_m]=\delta_{nm},
\qquad
\mathbb E[b_n\bar b_mb_k\bar b_l]
=\delta_{nm}\delta_{kl}+\delta_{nl}\delta_{km}-R_{nmkl},
$$

$$
R_{nmkl}=\sum_i\bar U_{in}U_{im}\bar U_{ik}U_{il},\qquad
|R|\le R_{nnnn}=\sum_i|U_{in}|^4\equiv\mathrm{IPR}_n\sim1/N'.
$$

$R$ is an inverse-participation-ratio remainder, negligible for delocalized
eigenvectors; it is what makes $v_b\equiv\mathrm{Var}(|b_n|^2)=1-\mathrm{IPR}_n\approx1$
for complex phases (section 4.1's $v_b=1$).

**Step 2 - mean.** Unbiased, as (4.4) said:

$$
\mathbb E[\hat\rho_1]=\sum_{nm}c_n\bar c_m\,\mathbb E[b_n\bar b_m]=\sum_n|c_n|^2=\rho_\eta .
$$

**Step 3 - variance.** Expand the fourth power and apply Step 1 term by term:

$$
\mathbb E[\hat\rho_1^2]
=\sum_{nmkl}c_n\bar c_mc_k\bar c_l\,\mathbb E[b_n\bar b_mb_k\bar b_l]
=\underbrace{\Big(\sum_n|c_n|^2\Big)^2}_{n=m,\,k=l}
+\underbrace{\Big(\sum_n|c_n|^2\Big)^2}_{n=l,\,k=m}
-\sum_{nmkl}c_n\bar c_mc_k\bar c_lR_{nmkl}.
$$

Subtracting $\rho_\eta^2$ and dropping the $O(\mathrm{IPR})$ remainder:

$$
\mathrm{Var}[\hat\rho_1]
=\rho_\eta^2
=\underbrace{\sum_{n\ne m}a_na_m}_{\text{cross terms}}
+\underbrace{\sum_na_n^2}_{\text{diagonal}} .
\tag{6.1}
$$

**The single-sample relative variance is exactly 1** (complex phases; for real
Rademacher $\boldsymbol\xi$ with real eigenvectors the same computation has a
third pairing $\delta_{nk}\delta_{ml}$ and gives relative variance 2 - call
this factor $v_z$, $=1$ complex, $=2$ real). Note what did *not* happen: the
diagonal term alone is small,
$\sum a_n^2/(\sum a_n)^2\equiv1/\tilde N$ with $\tilde N$ the **participation
number** of the window at $\mathbf r$ - but the cross terms restore the total
to $\rho_\eta^2$ regardless of $\tilde N$. Many contributing states do not
average; they interfere.

**Step 4 - why, structurally (the Gaussian limit).** $z$ is a sum of $\tilde N\gg1$
independent zero-mean complex terms, so by the CLT it is complex Gaussian with
variance $\rho_\eta$. For a complex Gaussian, $|z|^2$ is **exponentially
distributed**: $\mathbb E|z|^2=\rho_\eta$, $\mathrm{Var}|z|^2=\rho_\eta^2$,
relative variance 1 *independent of how many states built it* - consistent
with (6.1). Averaging $S_{\rm smp}$ independent samples,
$\hat\rho/\rho_\eta\sim\mathrm{Gamma}(S_{\rm smp})/S_{\rm smp}$:

$$
\frac{\sigma[\hat\rho]}{\rho_\eta}
=\sqrt{\frac{v_z}{S_{\rm smp}}}
\qquad\text{pointwise - no $N_{\rm at}$ anywhere.}
\tag{6.2a}
$$

(A factor-2 pointwise preference for complex phases even at $\Gamma$ - the
opposite of 4.1's memory-driven advice; choose per run.) Spatially, one sample
$|z(\mathbf r)|^2$ is a **speckle pattern**: envelope $\rho_\eta(\mathbf r,E)$,
multiplicative $O(1)$ grain whose correlation length is that of the filtered
states, $\ell_c(E)\approx\pi/k(E)$ with $k(E)$ the local de Broglie
wavevector - a few angstroms at eV energies.

**Step 5 - aggregation, and the true origin of $1/N_{\rm at}$.** Now contract
the same sample against a weight $h(\mathbf r)\ge0$ (a voxel average, an STM
pixel, an energy window, or $h=1$):

$$
\hat F_h\equiv\int h\,\hat\rho_1\,d^3r
=\sum_{nm}f_nf_m\,H_{nm}\,\bar b_nb_m,
\qquad
H_{nm}\equiv\int h\,\overline{\psi_n}\psi_m\,d^3r,
$$

by the same expansion as Step 0. Mean: $\sum_nf_n^2H_{nn}=\langle\rho_\eta,h\rangle$.
Variance, by the identical Step-3 pairing argument:

$$
\mathrm{Var}[\hat F_h]
=\sum_{n\ne m}f_n^2f_m^2\,|H_{nm}|^2
+v_b\sum_nf_n^4\,H_{nn}^2 .
$$

The pointwise case is $h\to\delta^3(\mathbf r-\mathbf r_0)$, where
$H_{nm}=\overline{\psi_n}\psi_m(\mathbf r_0)$ and $|H_{nm}|^2=H_{nn}H_{mm}$ -
the cross terms are as large as they can be, giving (6.1). Aggregation
shrinks precisely the off-diagonal $H_{nm}$:

- $h=1$: $H_{nm}=\delta_{nm}$ **exactly** (eigenstate orthogonality) - the
  cross terms vanish identically.
- $h$ smooth over a region containing
  $K\equiv(L_h/\ell_c)^d$ independent speckle cells: for extended states the
  standard random-wave estimate gives
  $|H_{nm}|^2\approx H_{nn}H_{mm}/K$ for $n\ne m$ ($H_{nm}$ is a sum of $K$
  cell contributions of random phase; $h=1$ is the $K\to\infty$ limit).

Insert both pieces and normalize:

$$
\frac{\mathrm{Var}[\hat F_h]}{\langle\rho_\eta,h\rangle^2}
\;\simeq\;\frac1K
+\frac{v_b\,\kappa\,g}{N_{\rm lev}},
\qquad\text{per sample; divide by }S_{\rm smp}.
\tag{6.2b}
$$

Here the diagonal term was evaluated exactly as in the original text: its
spectral factor is

$$
\frac{\sum_nf_n^4}{(\sum_nf_n^2)^2}=\frac{\sum_n\delta_\eta^2}{(\sum_n\delta_\eta)^2}
=\frac{1}{2\sqrt\pi\,\eta\,D(E)}\equiv\frac{1}{N_{\rm lev}},
$$

using $\int\delta_\eta^2\,dE=(2\sqrt\pi\eta)^{-1}$; $N_{\rm lev}=2\sqrt\pi\eta D(E)$
is the number of levels in the broadening window ($D=d\cdot N_{\rm at}$ the
total DOS, $d$ the DOS per atom). Amplitude fluctuations multiply it by
$\kappa=\langle H_{nn}^2\rangle/\langle H_{nn}\rangle^2$ ($\kappa=2$
complex-ergodic, 3 real-ergodic, $\gg3$ localized), and an exact degeneracy $g$
(spin, valley) divides the independent-level count.

**Reading (6.2b).** The first term is residual speckle coherence; the second is
the orthogonality-limited floor. The floor is reached when $K\gtrsim N_{\rm lev}$,
and only then does the error scale as

$$
\frac{\sigma}{\langle\rho\rangle}=\frac{1}{\sqrt{S_{\rm smp}N_{\rm eff}}},
\qquad
N_{\rm eff}\equiv\frac{2\sqrt\pi\,\eta\,d(E)}{v_b\,\kappa\,g}\,N_{\rm at}
\;\propto\;\eta\,N_{\rm at}.
\tag{6.1a}
$$

**This is the only place $1/N_{\rm at}$ enters.** Three operations reach the
floor: spatial or energy coarse-graining ($K>1$; the sum rules of section 9
are $K\to\infty$), k-point averaging (independent draws, section 8.1), and
time/dephasing averaging (section 6.6 - the source paper's route to pointwise
$N_e$ scaling; its $1/N_w$ is exactly the $1/K$ of (6.2b) with the time
average playing the role of aggregation).

The error is **multiplicative** in both regimes ($\hat\rho$ is
Gamma-distributed pointwise): gaps quiet in absolute terms, peaks noisy in
proportion - the maps look physical, not salt-and-peppered.

### 6.4 Uniform bound over the whole map

Pointwise, $\hat\rho/\rho_\eta\sim\mathrm{Gamma}(S_{\rm smp})/S_{\rm smp}$,
so $\Pr(|\hat\rho/\rho-1|>\varepsilon)\le2e^{-S_{\rm smp}\varepsilon^2/4}$, and a
union bound over the $P=N_{\rm grid}N_E$ outputs gives

$$
S_{\rm smp}\;\ge\;\frac{4}{\varepsilon^{2}}\,\ln\frac{2N_{\rm grid}N_E}{\delta}.
\tag{6.3}
$$

- **No $N_{\rm at}$ in it** - this replaces the original claim that samples
  decrease with system size, which holds only for aggregated functionals
  (there, replace $4/\varepsilon^2$ by $4/(\varepsilon^2N_{\rm eff})$ per
  (6.2b)).
- **Map size costs a logarithm**: uniform 10% over the full 4D TBG map
  ($P\sim10^{10}$, $\delta=10^{-2}$) needs $S_{\rm smp}\approx1.2\times10^4$;
  *typical* (non-uniform) 10% needs $S_{\rm smp}\approx10^2$. Report which one
  is being quoted.
- Because $\hat\rho\ge0$ pointwise, an assumption-free Markov backstop is
  always available.

Linear scaling is unaffected - $S_{\rm smp}$ in (6.3) is $N$-independent, so
every cost term in section 7 stays $O(N)$; only the original claim of a
*sublinear* spectral stage is lost.

### 6.5 (d) Metric

In **Mode B** this is the dominant systematic term and it is *not* $O(\mathrm{tol})$
but $O(\mathrm{tol}\cdot M/4)$, by (2.7): the metric error is an energy-axis
displacement, and a filter of width $\eta$ amplifies it by $a/\eta$. This is the
term that disqualifies Mode B for LDOS at the `ISQRTMPreconditioner` default.

In **Mode A** the operator carries no metric error - (2.2) is an identity -
so only the *sampler* contributes: the applicator truncation of (4.7) -
$O(\mathrm{tol}^{k+1})$ for series (4.8), $2\epsilon_q$ for the measured-interval
$q(W)$ - or
$O(\Vert S_g-S\Vert ^2)$ for (S3). This is a *bias*: it does not average down
with $S_{\rm smp}$ and must be made small, not sampled away.

### 6.6 Why not time propagation

Time-averaging suppresses off-diagonal coherences for
$|\varepsilon_n-\varepsilon_m|\gtrsim1/T_{\rm tot}$, adding a second,
size-independent floor $1/N_w$ with $N_w=\sqrt2\,\eta T_{\rm tot}$:
$\mathrm{Var}/\rho^2=S_{\rm smp}^{-1}(N_{\rm eff}^{-1}+N_w^{-1})$ - the
$1/N_w$ term is exactly the residual coherence $1/K$ of (6.2b), with time
average playing the role of aggregation. Since
$M\simeq aT_{\rm tot}$, cost at fixed accuracy is
$\propto x(N_{\rm eff}^{-1}+x^{-1})$ with $x=\eta T_{\rm tot}$, minimized at
$x^*\simeq\sqrt{N_{\rm eff}}$ where the bracket is $\simeq1$; at $x=1$ it is
$\simeq2$.

> **$S_{\rm smp}$ and $T_{\rm tot}$ are interchangeable to within a factor of
> two.** Dephasing buys exactly the propagation time it saves in samples.
> Meanwhile the source paper's Table S2 shows time-stepping wastes up to $6\times$
> the operations of a single-shot expansion at small $\tau$, because each
> restart discards low-order Bessel terms; direct Chebyshev *is* the
> $\tau\to\infty$ limit. Hence: **use direct Chebyshev, spend the budget on
> $S_{\rm smp}$.**
>
> Exception: if conductivities, diffusion, or wave-packet dynamics are wanted
> alongside, the intermediate $\mathbf c(t)$ are physically meaningful and
> $T_m(\bar H)\boldsymbol\xi$ are not. Then the time route earns its keep.

### 6.7 (e) Inner-solve error - the one new knob

**With the fixed polynomial (2.5), this term is closed in advance:** relative
error $\mathrm{tol}^{k+1}M/4$, deterministic, no tolerance to tune, and
tests 9/12 verify it. The rest of this section prices the **PCG fallback**.

A PCG solve to relative residual $\tau$ returns $z=S^{-1}t+S^{-1}r$ with
$\Vert r\Vert \le\tau\Vert t\Vert $, so the applied operator is $G+\mathcal E$ with

$$
\Vert \mathcal E\Vert \;\le\;\tau\,\Vert S^{-1}\Vert \,\Vert H\Vert \;\simeq\;\tau\,a
$$

- the same $\Vert H\Vert /\lambda_{\min}(S)$ that set $a$ in section 2.2. A perturbation
of this size shifts the argument of the filter, giving relative error
$\sim|\partial_E\rho|\,\Vert \mathcal E\Vert /\rho\sim\tau a/\eta$. Using
$M\simeq4.3\,a/\eta$:

$$
\boxed{\;\tau_{\rm CG}\;\lesssim\;\epsilon_{\rm target}\,\frac{\eta}{a}\;\simeq\;\frac{4\,\epsilon_{\rm target}}{M}\;}\tag{6.4}
$$

**The inner tolerance must beat the target accuracy by the Chebyshev order.**
At $a=100$ eV, $\eta=25$ meV, $\epsilon_{\rm target}=10^{-6}$: $\tau_{\rm CG}\le2.5\times10^{-10}$.
The `LinearSolver` default of $10^{-8}$ in `calc/cheby.hpp:61` is **two orders
too loose for this regime** and must be lowered for the LDOS path. The bound is
the coherent worst case; independent residuals accumulate as $\sqrt M$ instead,
so measured behaviour is usually better - confirm by re-running one energy at
$\tau/10$ (ladder test 9). The insurance is cheap: with the $\tilde Z$
preconditioner, (2.6) prices $10^{-8}\to10^{-10}$ at one extra CG iteration.

---

## 7. Cost and memory

$$
C_{\rm tot}=\underbrace{2S_{\rm smp}M\,\mathrm{nnz}}_{\text{spectral}}
+\underbrace{2S_{\rm smp}N_E\,\mathrm{nnz}(S)}_{\text{assembly (5.3a)}}
+\underbrace{2N_EN_{\rm grid}\min(n_{\rm loc},S_{\rm smp})\,n_{\rm loc}}_{\text{grid, best route}}
$$

Substituting $M=\Theta(a/\eta)$ and (6.3): $S_{\rm smp}$ is set by the target
pointwise accuracy alone, **independent of $N$**, so every term is strictly
$O(N)$ with $N$-independent prefactors. (For aggregated observables (6.2b)
gives the stronger $S_{\rm smp}\propto1/N_{\rm at}$; do not book that bonus for
the pointwise map.) Overall: **linear scaling for the complete 4D map**, with
the grid contraction - not the eigenproblem - as the asymptotic bottleneck.

In the adjoint route the spectral term becomes

$$
2S_{\rm smp}M\big(\mathrm{nnz}(H)+(2+3k)\,\mathrm{nnz}(S)\big)
\;+\;\underbrace{6kS_{\rm smp}\,\mathrm{nnz}(S)}_{\text{sampler (4.7)},\ k\simeq2},
$$

With the fixed $k=2$ polynomial of (2.5) - $1+3k+1=8$ overlap-pattern SpMVs
per step against Mode B's 3 - the first term is $\sim3\times$ Mode B's
$2S_{\rm smp}M\,\mathrm{nnz}$; the sampler
term is $\sim k/M\sim10^{-4}$ of it and **never appears in the cost
model**. That $3\times$ is the entire price of removing (2.7)'s bias, against
Mode B's alternative of driving `tol` down five orders through SpGEMM fill -
which raises $\mathrm{nnz}(\tilde Z)$ in *every one* of the $M$ steps and may
simply stall. Every term above is $O(N)$ with an $N$-independent prefactor.

### Memory

| Item | Size | Note |
|---|---|---|
| $H,S,\tilde Z$ | $O(\mathrm{nnz})$ | $\tilde Z$ density controlled by `sptol`; loose is cheap |
| Recursion vectors | $3N'S_{\rm blk}$ | 3 multivectors only |
| Energy accumulators $\mathbf X(E_k)$ | $N_EN'S_{\rm blk}$ | **dominant working set** |
| $\hat D$ (per rank) | $N_E\,\mathrm{nnz}_{\rm local}$ | route (5.3a) only; (5.3b) never forms it |
| Output | $N_{\rm grid}N_E$ | irreducible; stream |

Concretely for the worked TBG example: $N_E\,\mathrm{nnz}=6\times10^{10}$
entries - 960 GB aggregate in complex128, i.e. ~10 GB/rank on
96 ranks, halved at $\Gamma$ (real) and halved again in float32 accumulation
(the estimator averages $10^2$-$10^4$ samples; 7 digits per entry is ample).
If still tight the honest options are: more ranks ($\mathrm{nnz}_{\rm local}$
shrinks linearly); fewer energies per pass (multiplies the *spectral* cost by
the number of passes - the SpMMs rerun); or per-sample-block grid contraction
(multiplies the *grid* cost by $S_{\rm smp}/S_{\rm blk}$) - the last being
exactly route (5.3b), which at $S_{\rm smp}\lesssim n_{\rm loc}$ is the
preferred answer anyway: it removes this row entirely.

**Block over samples, keep all energies in flight.** With $S_{\rm blk}=4$-$8$
the accumulator stays under 1 GB per rank while still wide enough for efficient
SpMM.

Full output at $dr=0.1$ A for a magic-angle slab is $\sim3.6\times10^8$ points
$\times N_E$ - ~300 GB at $N_E=200$ in float32. Stream chunked, compressed
HDF5; expose a z-slab / energy-subset API.

### Parallelization

- **Spatial domain decomposition.** Each rank owns atoms/orbitals, their
  $\mathcal P$-pairs, and the grid boxes they cover. The recursion is an SpMM
  with one halo exchange per step; halo width fixed by $r^{\rm cut}$, so
  communication is $O(N^{2/3}/P^{2/3})$ surface against $O(N/P)$ volume.
- **Zero global reductions in the hot loop** with the fixed-polynomial inner
  solve (2.5): halo exchanges are neighbor-only. The stability monitor's dot
  product is the sole collective; run it every ~100 steps. (The PCG fallback
  forfeits this: $2n_{\rm it}M\sim10^5$ allreduces per block.)
- $\hat D$ is accumulated **locally** - no cross-rank sample reduction.
- The grid stage is embarrassingly parallel, no communication.
- Samples and energies give two further trivially-parallel dimensions.

---

## 8. Extensions

### 8.1 k-points

Run the recursion independently per $\mathbf k$ on $H(\mathbf k),S(\mathbf k)$,
obtain $\hat D^{\mathbf k}(E)$, then
$\hat D_{\mu\nu}(\mathbf R,E)=\sum_{\mathbf k}w_{\mathbf k}e^{-i\mathbf k\cdot\mathbf R}\hat D^{\mathbf k}_{\mu\nu}(E)$,
and contract in the standard Gint structure. **Each k-point is an independent
random draw**, so $N_{\mathbf k}$ multiplies $S_{\rm smp}$ at no extra cost -
k-sampling is free variance reduction, and also raises $D(E)$ and hence
$N_{\rm eff}$.

### 8.2 Deterministic limit (probing)

$\mathbf w(\mathbf r)$ is supported within $r^{\rm cut}$ of $\mathbf r$; $F_E$
has range $\ell_\eta\sim\hbar v/\eta$. Two points separated by more than
$2(r^{\rm cut}+\ell_\eta)$ have disjoint supports, so replacing random vectors
by a distance-$d$ graph coloring makes the estimator **exact** with
$N_{\rm col}=O((\ell_\eta/dr)^{d})$ vectors, independent of $N$.

**Use probing when $N_{\rm col}<S_{\rm smp}$, stochastic otherwise.** Large
$\eta$ favours probing; small $\eta$ favours sampling. Probing vectors also make
excellent control variates.

### 8.3 Spectral window mode

For STS restricted to $|E-E_F|<0.5$ eV, Chebyshev-filter a subspace onto the
window, Rayleigh-Ritz, and obtain *actual eigenpairs*: exact LDOS with arbitrary
broadening at cost $O(N N_{\rm win})$. For flat-band physics often both cheapest
and most accurate.

### 8.4 Vacuum extrapolation for STM

Beyond $r^{\rm cut}$ above the surface the NAO basis is identically zero, so raw
LDOS in the tunneling gap is a basis artifact. Match at a plane $z_0$ where the
basis is still complete, FFT in-plane, and propagate:

$$
\rho(\mathbf k_\parallel+\mathbf G,z,E)=\rho(\mathbf k_\parallel+\mathbf G,z_0,E)\,
e^{-2\kappa(z-z_0)},\qquad
\kappa=\sqrt{\tfrac{2m}{\hbar^2}(\Phi_w-E)+|\mathbf k_\parallel+\mathbf G|^2}.
$$

---

## 9. Validation ladder

| # | Test | Reference | Cost |
|---|---|---|---|
| 1 | $\int\hat\rho(\mathbf r,E)\,dE$ | $\boldsymbol\phi^TS^{-1}\boldsymbol\phi$, Eq. (1.5) | free, exact |
| 2 | $\int\hat\rho\,d^3r$ | total DOS $D(E)$ | free |
| 3 | $\int f(E)\hat\rho\,dE$ | parent code's SCF density $n(\mathbf r)$ | cheap, brutal end-to-end |
| 4 | $\hat\rho$ at $M$ vs $M/2$ | internal truncation estimator | free (same moments) |
| 5 | Split-half variance over $S_{\rm smp}$ | (6.2a) pointwise / (6.2b) aggregated | free - **tells you which regime you are in** |
| 6 | Small cell (~200 atoms) | ELPA dense diagonalization + explicit $\delta_\eta$ | one-off |
| 7 | $\Vert E\Vert _{\max}=\Vert \tilde Z^\dagger S\tilde Z-I\Vert _{\max}$ | $\le\mathrm{tol}$; sets $k$, **not** accuracy | free, `isqrtm` reports it |
| 8 | (4.2)+(4.7) vs two-sided (4.10) | each other | 2x one run - **isolates the metric** |
| 9 | one energy re-run at $\tau_{\rm CG}/10$ | itself | cheap; validates (6.4) |
| 10 | $\Vert S_g-S\Vert _{\max}$ on $\mathcal P$ | grid quadrature error | free, if (S3) used |
| 11 | Mode A vs Mode B at purified `tol` | each other | one-off; validates (2.2) |
| 12 | Mode A at $\mathrm{tol}=10^{-3}$ vs $10^{-7}$ | **must agree to $\epsilon_M$** | cheap; proves $\tilde Z$ is not in the answer |

Sum rules 1-3 are exact identities; assert them in unit tests. Test 5 says
whether the next core-hour should go to $S_{\rm smp}$ (sampling-limited) or to
$M$/$\eta$ (resolution-limited). Tests 8-12 are specific to the adjoint route:
test 8 isolates the metric correction ((4.2) and (4.10) share the recursion but
no sampler code - run it on the row-6 small cell for a three-way check), and
test 12 is the cheapest check of sections 2.4/4.3 - if degrading $\tilde Z$ by
four orders moves $\hat\rho$ by more than $\epsilon_M$, $\tilde Z$ has leaked
into the answer (congruence where a preconditioner was meant, or a sampler
missing its $(I+E)^{-1/2}$).

---

## 10. Parameter recipe

| Parameter | Choice | Rationale |
|---|---|---|
| $\eta$ | $\max(3\Delta(E),\ \text{target resolution})$ | section 6.1 |
| $a,b$ | Lanczos on pencil, $\times1.01$ pad | section 2.2; under-estimation diverges |
| $M$ | $\lceil 4.3\,a/\eta\rceil$ ($\epsilon_M=10^{-8}$) | Eq. (3.5) |
| $c_m$ | DCT, $N_{\rm FFT}\ge4M$ | section 3.2 |
| $\boldsymbol\xi$ | real Rademacher (Gamma), complex phase (k) | section 4.1 |
| $S_{\rm smp}$ | Eq. (6.3); start from split-half estimate | typically $10^2$-$10^4$ |
| $S_{\rm blk}$ | 4-8 (memory-bound), up to 64 (GPU) | section 7 |
| $\tilde Z$ | Mode B: + one cubic sweep to $\mathrm{tol}\le4\epsilon/M$; Mode A: loose first sweep | section 2.4 |
| mode | B if purified $d_Z\lesssim4d_S$ (measure); A otherwise | section 2.4 |
| sampler | **(4.7)** $\tilde Z\,q(W)\boldsymbol\xi$ on the measured spectrum of $W$; (S3) if $\lambda_{\min}(S)$ small | section 4.3 |
| $\tau_{\rm CG}$ | $\min(10^{-10},\,4\epsilon_{\rm target}/M)$ | Eq. (6.4) - **not** the $10^{-8}$ default |
| inner solve | fixed $\hat B_2$ polynomial (2.8); PCG only as fallback | section 2.5; zero reductions |
| contraction | (5.3b) if $S_{\rm smp}\lesssim n_{\rm loc}$ (usual); (5.3a) otherwise / density path | section 5.3 |
| box size | 2-4 A | balances $n_{\rm loc}^2$ against $\Phi$ reuse |

### Worked estimate - magic-angle TBG

$N_{\rm at}=1.1\times10^4$, DZP => $N=1.45\times10^5$,
$\mathrm{nnz}\approx3\times10^8$, $a\approx100$ eV, $\eta=25$ meV =>
$M\approx1.7\times10^4$, $N_E=200$, $dr=0.1$ A.

- Spectral, per sample: $2M\,\mathrm{nnz}\approx1.0\times10^{13}$ flops - seconds on a GPU.
- $S_{\rm smp}=100$: $\approx10^{15}$ flops - ~300x below the deterministic all-columns route.
- Grid: $2N_EN_{\rm grid}n_{\rm loc}^2\approx6\times10^{15}$ flops of GEMM - minutes on GPU, the true bottleneck.

**Caveat, and it matters here.** Pointwise, $S_{\rm smp}$ carries the whole
burden regardless of size (6.2a). For the *aggregated* diagnostics and any
coarse-grained readout, the floor (6.2b) needs $N_{\rm lev}\gg1$: a single
moire cell at Gamma has four flat bands in ~10 meV, $D\approx0.4$ meV$^{-1}$,
$N_{\rm eff}\sim3$ - so even aggregation does not average there. Genuine
supercells ($10\times10$ moire: $N_{\rm eff}\sim300$) or k-point sampling
restore it. Two further warnings:

- **Degeneracy $g$.** Spin and valley degeneracies are exact and cannot be
  dephased or sampled away; they divide $N_{\rm eff}$ directly. Break them
  explicitly or accept the factor.
- **$\kappa$ in reconstructed structures.** Flat-band states in AA regions are
  strongly non-ergodic and $\kappa$ may substantially exceed 3. Measure it on a
  small cell before trusting (6.3).

---

## 11. Summary of the generalization

| Ingredient | Zhou-Yuan (rsDFT) | This work (LCAO LDOS) |
|---|---|---|
| Basis | uniform real-space grid, orthogonal | NAO, nonorthogonal |
| Random state | $\mathbb E[\mathbf c\mathbf c^\dagger]=I/N$ | $\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=S^{-1}$, via $\mathbf c_0=Z\boldsymbol\xi$ |
| Filter | $\sqrt{f(H)}$ (occupation) | $\delta_\eta^{1/2}(E-\tilde H)$ (energy) |
| Propagator | $e^{-iHt}$, Chebyshev-Bessel, $N_t$ steps | direct Chebyshev, single expansion |
| Dephasing | time average over $t$ | absorbed into sample average |
| Self-averaging scale | $N_e$, via time-dephasing | $N_{\rm eff}$ (6.1a), aggregated readouts only |
| Output | $\rho(\mathbf r)$ | $\rho(\mathbf r,E)$ on full grid x energy |
| Error | $1/\sqrt{SN_e}$ | $1/\sqrt{S_{\rm smp}}$ pointwise (6.2a); $1/\sqrt{S_{\rm smp}N_{\rm eff}}$ aggregated (6.2b) |

The three things that must not be dropped in implementation: **the
metric-corrected random vector** (4.1), **the square-root factorization** (3.1)
- worth $W/(4\sqrt\pi\eta)\sim10^2$ in variance against a bilinear KPM
estimator pointwise, and never worse, by theorem (3.8) - and **assembly on $\mathcal P$ before grid contraction** (5.3).

---

## 12. Implementation status and reuse (added by implementer)

**Reused unchanged from the removed deterministic path:**

| Component | File | Role in this design |
|---|---|---|
| Uniform grid, box tiling, bounding spheres | `calc/real_space_grid.hpp` | section 5.3 |
| Inertial partition, `box_owner == atom_owner` | vbcsr `core/atomic/inertial_partition.hpp` | section 7 parallelization |
| Orbital evaluation (op2c splines x $Y_{lm}$), box GEMM | `calc/collocation.hpp` | Eq. (5.4) |
| Minimum-image geometry | `calc/pbc_geometry.hpp` | images in $\mathcal L$ |
| Chebyshev recursion, spectral bounds | `calc/cheby.hpp`, `setup/spectral_bounds.hpp` | Eq. (5.1), section 2.2 |
| DCT Chebyshev coefficients | `calc/kpm.hpp` | Eq. (3.4) |
| $\bar G=S^{-1}H$ and $\bar G^\dagger=HS^{-1}$ applies | `calc/cheby.hpp` `ChebyMode::NORMAL`/`ADJOINT` | Eq. (2.4) - **already exact** |
| Batched PCG over `DistMultiVector` | `calc/solver.hpp` | inner solve in (2.4) |
| Block-Jacobi / ISQRTM preconditioners | `calc/preconditioner.hpp` | PCG fallback path only |
| Newton-Schulz sparse $S^{-1/2}$ ($\tilde Z$) | `calc/isqrtm.hpp` | the single metric object: sections 2.1, 2.5, 4.7 |
| $\Phi$ applied adjointly | `calc/collocation.hpp` | sampler (S3), Eq. (4.1a) |

**Removed as superseded:** region seeding (`LdosRegion`, `SelectLdosRegion`,
`LdosSeedPlan`, `LdosPattern`, `ComputeRegionSpectralBlocks`), the d-separated
probe classes (`calc/probe_classes.hpp`), and the dense $|R|^2$ region block.
The deterministic direct-seeding path cost $O(N^2)$ and the probing variant was
measured at 5-15% contamination at the cutoff separation; section 8.2 here
re-derives probing from a different premise (disjoint supports of
$\mathbf w(\mathbf r)$ and $F_E$) and would be reinstated on that basis, not the
old one.

**Built and benchmarked:** the sampler (4.7) with both applicators, as
`calc/metric_sampler.hpp` (`MetricSampler`: Lanczos measurement of
$W=\tilde Z^\dagger S\tilde Z$, automatic binomial/Chebyshev selection, the
deterministic certificate `Residual()`), validated in
`calc/test/test_metric_sampler.cpp`. Measured on tridiagonal chains: loose NS
$\tilde Z$ bias $2\times10^{-11}\approx\mathrm{tol}^2$ at 4 SpMV/sample;
graph-seed $\tilde Z$ on $\kappa_S\sim200$ gives $\kappa(W)=40$
($\Vert E\Vert =0.96$), Chebyshev degree 66, bias $4\times10^{-10}$ at 199
SpMV/sample; all regimes agree with the CG reference for
$u^\dagger S^{-1}u$ to statistical precision while the naive $c_0=\xi$ is
71-128% wrong; a deliberately wrong applicator is exposed by the certificate
at $9\times10^{-2}$.

On the real diamond sc444 overlap ($N=1664$), two builds were benchmarked.
The DeePTB-stored $S(\mathbf R)$ summed to $\Gamma$ is **indefinite**
($\lambda_{\min}=-1.7\times10^{-2}$, dense-verified; R-truncation error), which
the sampler's $w_{\min}>0$ guard rejects as required - the definiteness audit
of the stored overlaps is tracked separately, and the guard is unit-tested on a
synthetic indefinite chain. The **op2c two-center integrals** build (the
production path) gives the true Gram matrix: $\mathrm{diag}(S)=1$ exactly,
$\lambda\in[3.8\times10^{-3},16.1]$, $\kappa(S)\approx4.2\times10^3$ - genuine
DZP-class conditioning, $\lambda_{\min}$ at the section 2.2 pruning threshold.
There: the no-$\tilde Z$ identity baseline is **unusable** (degree 690,
certificate stalls at $3\times10^{-5}$, 2071 SpMV/sample - section 2's claim as
a measurement); every $\tilde Z$ regime reaches $\sim10^{-14}$ certificates at
4 SpMV/sample, a factor $\sim500$ in cost and the difference between certified
and unconverged on the same matrix. At 128 atoms every pattern-subgraph
spans the whole cell, so the graph seed is near-exact there; the genuine
regime was measured on **sc888** ($N=13312$, subgraph ~300 of 1024 atoms):
$W\in[0.66,1.56]$, i.e. $\Vert E\Vert =0.56$, $\kappa(W)=2.35$ - **the
unpurified graph seed turns $\kappa(S)=4200$ into $\kappa(W)=2.35$**. The
binomial series is correctly rejected at that $\Vert E\Vert $; the
measured-interval Chebyshev applicator needs only degree 19 (58 SpMV/sample,
certificate $5.5\times10^{-13}$, covariance agreeing with the CG reference
while the naive sampler errs by 97%). Runtime is dominated by the seed itself
- 1024 dense ~3900-dim subgraph diagonalizations, ~20 min at 64 threads under
the memory-capped batch; the sampling tail is minutes (vbcsr SpMV is
OpenMP-threaded within a rank).

**Not yet built:** the fixed inner solve $\hat B_k$ (2.8) sharing the
`MetricSampler` machinery; the $\delta_\eta^{1/2}$ DCT coefficients (3.4); the covariant
co-iterate (2.5) and its $\Vert X_m\Vert _S$ monitor; multi-energy accumulation
(5.2); assembly on $\mathcal P$ (5.3a) and the square-on-grid route (5.3b)
with its complex per-k $\Phi$; the two-sided estimator (4.10); the
grid-Gram sampler (S3); and the validation ladder (section 9).

**Two findings that change existing code rather than adding to it:**

1. `SampleVectors::initialize` (`calc/observables.hpp:70`) draws
   `set_random_normal` on NAO coefficients, i.e.
   $\mathbb E[\mathbf c_0\mathbf c_0^\dagger]=I$. This is **correct as it stands**
   for its current consumers - `compute_dos` and `compute_ac` are Hutchinson
   trace estimators (`calc/dos.hpp:29`, "$\langle\psi|P|\psi\rangle\sim\mathrm{Tr}(P)/n$")
   and traces are similarity-invariant, per the box in section 4.1. The LDOS
   sampler is therefore an **addition**, not a repair: add a metric-correct
   factory alongside `initialize` and leave the existing path alone. Do not
   apply (4.1) to the trace observables - it would bias them.
2. `calc/cheby.hpp:61` fixes the inner `LinearSolver` tolerance at $10^{-8}$ in
   all four constructors. Equation (6.4) requires $\sim10^{-10}$ at $M\sim10^4$.
   Make it a constructor parameter derived from $M$ and $\epsilon_{\rm target}$;
   by (2.6) it costs one extra CG iteration, not a factor.
3. **Resolved.** `isqrtm`'s seed now runs through `vbcsr::graph_inverse_sqrt`
   (the thin inverse-square-root action, see the vbcsr tree's
   `doc/design_isqrt_action.md`): the per-problem working set dropped from
   ~4 dim^2 scalars (dense diagonalization, ~0.78 GB/thread at dim ~3800) to
   ~2 dim^2 (~0.45 GB), and the explicit memory budget that capped the batch
   was removed - one problem runs per OpenMP thread, so `OMP_NUM_THREADS` is
   both the thread and the memory lever (measured 8 GB peak at 12 concurrent
   sc888 problems).
4. `ISQRTMPreconditioner`'s defaults (`tol=1e-5, sptol=1e-5, max_iter=10`) are
   right for Mode A and **wrong for Mode B at LDOS resolution** by four orders,
   per (2.7). Since Mode B is selected by a *different constructor* and not by a
   flag, nothing warns. Either gate Mode B on $\mathrm{tol}\le4\epsilon/M$ at
   construction, or make `isqrtm`'s achieved $\Vert E\Vert $ a required argument to the
   LDOS entry point so the check cannot be skipped.
