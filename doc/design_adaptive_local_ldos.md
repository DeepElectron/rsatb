# Adaptive, distance-certified local capture for real-space LDOS

Status: proposed enhancement of `rs_ldos.hpp`.  This document defines the
algorithm and validation needed before the packed local-capture path is called
state of the art.  It does not change the paper's claims until the measurements
in section 13 have passed.

## 1. Objective

For each requested energy, compute the real-space diagonal of the positive
spectral operator

$$
K_k=g_{\eta_k}(E_k-H)=F_kF_k^\dagger
$$

with an error requested by the user rather than parameters chosen by hand.
The implementation must retain all current distribution constraints:

* atom ownership is fixed by the existing orbital-weighted RIB/ParMETIS
  partition before the LDOS calculation starts;
* capture regions subdivide only a rank's owned atoms and never move atoms;
* real-space grid ownership remains independent;
* no object proportional to the global atom or region count is replicated;
* at fixed nonzero broadening and fixed error target, memory per rank and
  communication per rank remain bounded under weak scaling;
* the unpacked limit reproduces the exact projector identity, while every
  approximation in the scalable packed path has a measured error.

The core operation remains unchanged: filtered random vectors identify a
local projector, its contribution is evaluated directly, and only the
remainder is sampled.  The enhancement concerns when that operation is used,
how large each local projector is, how its image columns are packed, and how
the resulting error is certified.

## 2. Why the present packing is insufficient

The current implementation assigns each RIB-region center to a virtual
fractional cell and colors the cell periodically.  It has four limitations.

1. The number of colors is capped by the maximum number of regions on one
   rank.  One region on each of many ranks therefore collapses to one global
   color.
2. A periodic color is not a distance guarantee for irregular regions.  RIB
   regions on opposite sides of an MPI boundary may be close and share a
   column.
3. One sign realization gives no internal estimate of packing noise.
4. Fixed `local_capture`, `local_pack`, and the global-capture threshold do
   not solve a stated error-versus-cost problem.

The unpacked estimator is algebraically exact but has
$r_{\rm loc}N_B$ global image columns.  The packed estimator has a fixed
number of columns but replaces this exactness with random cross-region terms.
A scalable claim is therefore credible only if those cross terms are both
controlled geometrically and measured statistically.

## 3. Choose the side of the square-root filter before optimizing packing

Write the positive LDOS operator as $K=FF^\dagger$ and condition on a frozen
block projector $P=\sum_B P_B$.  There are two exact projector identities.
The current implementation projects the *filtered output*:

$$
K=\underbrace{PK+KP-PKP}_{D_{\rm out}}
 +\underbrace{(I-P)K(I-P)}_{R_{\rm out}}.
$$

An equally exact alternative projects the random vector *before* the
square-root filter:

$$
\boxed{
K=\underbrace{FPF^\dagger}_{D_{\rm in}}
 +\underbrace{F(I-P)F^\dagger}_{R_{\rm in}}.}
$$

The second identity should be the reference design for the next
implementation.  It has four concrete advantages.

1. Both terms are positive and $0\le R_{\rm in}\le K$.  Hence its pointwise
   Gaussian residual variance can never exceed the plain estimator's.
   $D_{\rm out}$ need not be positive and no analogous pointwise ordering is
   available for the present split.
2. Direct local images require $Fq_j$, not $Kq_j$.  For the Gaussian LDOS
   filter the $K$ polynomial is about $\sqrt 2$ more expensive than the $F$
   polynomial.
3. $P=I$ continuously becomes complete-identity probing through $F$; no
   separate polarization formula is needed at the full-rank endpoint.
4. The two splits discard exactly the same *integrated* spectral weight:

   $$
   \operatorname{Tr}R_{\rm out}
   =\operatorname{Tr}[(I-P)K]
   =\operatorname{Tr}R_{\rm in}.
   $$

The price is scheduling, not additional scalar filter work.  Training
vectors must first be filtered and used to freeze $P$; only then can
independent production vectors be filtered as $F(I-P)\xi$.  The current
single pass over all training and production columns must therefore become
two sequential block sweeps.  Keep the output-side split as an A/B baseline:
a strongly nonuniform map can in principle prefer its pointwise residual,
but it must overcome the higher-degree images and loss of positivity.

### 3.1 Error decomposition of the preferred split

Let $U_B=(u_{B1},\ldots,u_{Br_B})$ and
$P=\sum_BU_BU_B^\dagger$.  Pack equally indexed directions of same-color
regions into

$$
x_{gj}^{(\ell)}=
\sum_{B:c(B)=g}\sigma_{Bj}^{(\ell)}u_{Bj},
\qquad X_\ell=(x_{gj}^{(\ell)})_{g,j}.
$$

Then $\mathbb E_\sigma[X_\ell X_\ell^\dagger]=P$.  The captured map from
one replica is the positive square-root-filter map

$$
\widehat D_\ell(\mathbf r)
=\langle\mathbf r|F X_\ell X_\ell^\dagger F^\dagger|\mathbf r\rangle
=\sum_{g,j}|\langle\mathbf r|F x_{gj}^{(\ell)}\rangle|^2.
$$

Its packing error is exactly

$$
\epsilon_{P,\ell}(\mathbf r)
=\langle\mathbf r|F(X_\ell X_\ell^\dagger-P)F^\dagger
|\mathbf r\rangle.
$$

The full learned estimator is

$$
\widehat\rho_P(\mathbf r)=
\frac1L\sum_{\ell=1}^L\widehat D_\ell(\mathbf r)
+\frac1{n_R}\sum_{i=1}^{n_R}
|\langle\mathbf r|F(I-P)\xi_i\rangle|^2.
$$

For real Gaussian production vectors, define

$$
\rho_R(\mathbf r)=
\langle\mathbf r|F(I-P)F^\dagger|\mathbf r\rangle.
$$

Independent residual vectors and packing phases then give the exact
conditional variance model

$$
\operatorname{Var}\widehat\rho_P(\mathbf r)
=\frac{2\rho_R(\mathbf r)^2}{n_R}
+\frac{v_P(\mathbf r)}{L}.
$$

Real sign residuals have the usual nonpositive fourth-cumulant correction.
The learning, tuning, residual, packing-pilot, and packing-production domains
must use disjoint counter keys.  `correction_rms` is not a substitute for
either production variance and must not be presented as an error bar.

### 3.2 Why learning a local projector can beat identity probing

For one training family $Y_B=\Pi_BF\Omega_L$,

$$
\mathbb E[Y_BY_B^\dagger]
=n_L\Pi_BK\Pi_B.
$$

Let $\lambda_{B1}\ge\cdots\ge\lambda_{Bd_B}$ be the eigenvalues of the
local covariance $\Pi_BK\Pi_B$.  By the Ky Fan maximum principle, the top
$r_B$ eigenvectors minimize the integrated residual among all rank-$r_B$
projectors supported in $B$:

$$
\min_{\operatorname{rank}P_B=r_B}
\operatorname{Tr}[F(I-P)F^\dagger]
=\operatorname{Tr}K-
\sum_B\sum_{j=1}^{r_B}\lambda_{Bj}.
$$

The local eigenvalue tail $\sum_{j>r_B}\lambda_{Bj}$ is therefore the
quantity exchanged for fewer image columns.  The additional Chebyshev
modulations should generate a larger candidate range; an independent
target-filter covariance should select and rank the final directions inside
that range so that enrichment does not silently optimize a different
spectral weighting.

There are two independent reductions in column count:

| construction | global filter columns |
|---|---:|
| complete identity, no packing | $N_Bd_B$ |
| learned local projector, no packing | $N_Br_B+n_{L,T,R}$ |
| packed identity | $L_I G_I d_B$ |
| packed learned projector | $L_PG_Pr_B+n_{L,T,R}$ |

The first reduction, $d_B\to r_B$, uses local spectral compressibility and
leaves a sampled residual.  The second, $N_B\to G$, uses spatial separation
and leaves packing interference.  Identity probing applies only the second;
the learned estimator applies both.

With the preferred input-side split, all entries in the table use $F$, so a
first equal-work comparison is

$$
C_{\rm identity}\simeq M_F L_I G_I d_B,
\qquad
C_{\rm learned}\simeq M_F
(n_{L,T,R}+L_PG_Pr_B).
$$

At equal $G$ and $L$, learning has a cost advantage precisely when

$$
d_B-r_B>\frac{n_{L,T,R}}{LG},
$$

provided its residual plus packing error meets the same target.  This is not
a universal dominance theorem.  If the local spectrum has exact rank
$r_*\ll d_B$, choosing $r_B\ge r_*$ removes the residual and learned capture
is cheaper whenever the inequality holds.  If the local spectrum is flat,
small residual error forces $r_B\simeq d_B$ and identity probing wins because
learning adds overhead without compressing anything.

For a fixed rank and coloring distance, define the masked map-averaged
residual and packing variances by $a(r_B)$ and $b(r_B,D)$.  The actual route
selection problem is

$$
\min_{r_B,D,n_R,L}
M_F[n_{L,T}+n_R+LG(D)r_B]
\quad\hbox{subject to}\quad
\frac{a(r_B)}{n_R}+\frac{b(r_B,D)}{L}\le\epsilon^2.
$$

Identity probing is the boundary point $r_B=d_B$, $n_R=0$ of this same
optimization.  It is not a competing physical idea.  The controller should
select the minimum measured cost, including the identity boundary, rather
than decide in advance that stochastic or deterministic computation is
preferable.

The current dense-window audit illustrates the prefactor.  The code targets
$d_B\simeq6r_B$: for $r_B=96$, $d_B\simeq576$, $G=8$, $n=32$, and $L=2$,
the preferred learned route has about
$32+2\times8\times96=1568$ square-root-filter columns.  One deterministic
packed-identity pattern has $8\times576=4608$ columns; two identity replicas
have 9216.  Thus identity is approximately 2.9 or 5.9 times wider in this
specific block construction.  An optimized coordinate coloring may change
$G_Id_B$ substantially, so this count is a benchmark prediction, not a
verdict.

### 3.3 Randomness roles and selection bias

Adaptive choices cannot use the same random values later reported as the
estimate.  Otherwise choosing a route, rank, distance, or sample count because
it looked favorable can bias the selected result.  Partition the stochastic
columns into three logical domains, accumulated during the same Chebyshev
recursion:

* **learning** $\Omega_L$: constructs nested global and local basis families;
* **tuning** $\Omega_T$: selects route and rank and estimates residual cost;
* **production** $\Omega_R$: estimates the final residual after every choice
  is frozen.

Packing likewise has disjoint pilot and production sign domains.  Pilot signs
select $D$, $G$, and a fixed production replica count $L$.  Only fresh
production signs enter the reported map and its post-hoc error bar.  Do not
stop production sequentially based on its ordinary sample variance.  If a
production run misses the requested error, treat it as a new pilot and launch
a fresh, pre-sized production batch; alternatively implement an explicitly
proved anytime-valid confidence sequence.

The global path can retain exchangeable leave-out reuse after the route has
been frozen by independent tuning data.  The local reference path uses one
learned projector and independent production residuals.  A symmetric two-fold
local cross-fit is optional: it recovers both sample halves but requires a
second local image sweep.

## 4. Route before paying for local images

The current code constructs and filters the local image columns before the
global projector has decided whether local capture is needed.  Reverse this
order.

1. Run a training/tuning filtered sweep.  Do not yet filter the local
   production vectors.
2. Construct the global candidate from learning vectors and evaluate its
   predicted production error with tuning vectors.
3. Freeze the route for every energy.  Accept the global path where the
   tuning error meets the requested target.
4. Only the remaining energies enter local basis construction.  Freeze their
   ranks and packing design.
5. Apply $F$ to the packed local directions and, in a separate production
   sweep, to $(I-P)\xi_R$.  These two stages supply the positive captured and
   residual maps in Sec. 3.1.

The route must be based on an estimated error, not a fixed capture value such
as 0.9.  Capture is still a useful diagnostic, but a window with small total
LDOS or strongly nonuniform spatial capture can violate a capture-only rule.

For a multi-energy scan, build a joint local basis only from locally routed
energies.  If their required ranks or kernel ranges differ by more than a
factor of two, cluster the energies and process the clusters separately.  A
single difficult energy must not set the image width of an otherwise cheap
scan.

## 5. Adaptive local rank

For routed energy $k$, let

$$
C_{Bk}=\Pi_BY_kY_k^\dagger\Pi_B
$$

be the first-half local covariance estimate after the existing per-energy
trace normalization.  For a joint energy group use its balanced sum.  If
$\lambda_{B1}\ge\lambda_{B2}\ge\cdots$ are its eigenvalues, choose the
smallest rank satisfying

$$
\frac{\sum_{j>r_B}\lambda_{Bj}}
     {\sum_j\lambda_{Bj}}
\le \tau_{\rm basis}^2,
\qquad r_B\le r_{\max}.
$$

This converts `local_capture` from the selected rank into a safety cap.
Numerically empty windows retain rank zero through the existing absolute
filter-scale threshold.

The tuning vectors provide an independent local check of every nested basis
constructed from learning vectors:

$$
\widehat e_B=
\frac{\sum_i\|(I-P_B)\Pi_By_i\|^2}
     {\sum_i\|\Pi_By_i\|^2}.
$$

If the tuning error exceeds the eigenvalue-tail prediction by more than its
sampling uncertainty, raise $r_B$ within the already learned nested family or
declare the rank cap active.  The final residual vectors never participate in
this decision.  Report the fraction of regions at the cap.  Regions may have
different ranks; packed columns are padded only within small rank buckets so
low-rank regions do not pay the global maximum unnecessarily.

### 5.1 Identity-probing endpoint

The admissible local ranks include the full block dimension.  Setting

$$
P_B=I_B\quad\hbox{for every region},\qquad
P_{\rm loc}=\sum_B I_B=I,
$$

eliminates the sampled remainder exactly:

$$
F(I-P_{\rm loc})F^\dagger=0,
\qquad F P_{\rm loc}F^\dagger=K.
$$

With separate image columns this is the complete deterministic-basis
calculation.  With packing, use the region coordinate directions
$e_{Bj}$:

$$
x_{gj}^{(\ell)}=
\sum_{B:c(B)=g}\sigma_{Bj}^{(\ell)}e_{Bj},
\qquad
Y_{gj}^{(\ell)}=Fx_{gj}^{(\ell)}.
$$

The positive packed estimate is

$$
\widehat\rho_I^{(\ell)}(\mathbf r)
=\sum_{g,j}|\langle\mathbf r|Y_{gj}^{(\ell)}\rangle|^2
=\langle\mathbf r|F X_\ell X_\ell^\dagger F^\dagger|\mathbf r\rangle.
$$

Since $\mathbb E_\sigma[XX^\dagger]=I$, this is an unbiased estimator of the
full LDOS and its only statistical error is packing error.  The two-sided
square-root form is preferred to one-sided $K$ polarization: it uses the
lower-degree filter, is positive for every replica, and inherits the
square-root variance advantage.  This is distance-colored stochastic
probing, not a deterministic method unless either no regions share columns
or the coloring eliminates every nonzero cross term.  A fixed phase pattern
is a deterministic approximation with unknown packing bias, not an exact
calculation merely because no random-state residual remains.

Identity probing needs no learned basis, square-root-filter training vectors,
or residual samples.  Its leading image width is

$$
G(D)d_B L,
$$

whereas learned local capture uses approximately
$G(D)r_BL+n_RF$-filter columns.  A first cost comparison is therefore

$$
C_{\rm identity}\simeq L G d_B M_F,
\qquad
C_{\rm learned}^{\rm preferred}
\simeq (n_{L,T,R}+L G r_B)M_F,
$$

The current output-side implementation instead pays
$n_{L,T,R}M_F+LGr_BM_K$.  Identity probing is favored when $r_B$ is close to
$d_B$, basis learning is expensive, or the distance coloring makes packing
exceptionally quiet.  Learned capture is favored when $r_B\ll d_B$ and the
sampled eigenvalue tail is cheap enough to meet the same accuracy target.

The controller must therefore compare three complete estimators with tuning
data and then freeze the route:

1. global learned projector;
2. local learned projector plus sampled residual;
3. local identity probing with packing error only.

For identity probing, region size is an independent geometric parameter; it
must not be defined as $6r_B$.  Varying region size trades coordinate width
against color count, and the relevant cost is the measured product
$G(D)d_B$, which is expected to approach the number of orbital directions in
one exclusion volume.  The identity endpoint is both a production candidate
and a mandatory baseline for deciding whether learned local bases provide
real value.

## 6. Distributed region-neighbor graph

Coloring is defined by physical separation, not MPI rank, row number, or grid
ownership.  Each region publishes a compact record

```
(stable_region_id, fractional_center, bounding_radius, rank_bucket)
```

where `stable_region_id` is based on its minimum original global atom id and
its local RIB leaf.  A global scalar `Allreduce` gives the total region count;
it never limits the number of colors by a per-rank count.

Construct the neighbor graph without gathering all records:

1. Choose global fractional directory cells whose physical edge is at most
   the trial exclusion distance $D$.
2. Send each region record to the owners of the directory cells intersecting
   its bounding sphere.  Directory ownership is a hash of the cell id and is
   unrelated to atom or grid ownership.
3. Within a cell and its periodic neighbors, form an edge $(B,C)$ when the
   minimum-image distance between region bounding spheres is below $D$.
4. Return only adjacency records to the two region owners.

This is a one-time metadata exchange of $O(N_B/P)$ records plus a bounded
geometric halo.  It neither moves atoms nor adds communication to applying
$P_B$.

## 7. Distance-certified coloring

Color the neighbor graph so that

$$
c(B)=c(C)\quad\Longrightarrow\quad d(B,C)\ge D.
$$

Use deterministic distributed Jones--Plassmann coloring: a counter hash of
the stable region id gives each node a priority; locally maximal uncolored
nodes choose the smallest color unused by colored neighbors; boundary colors
are exchanged until complete.  A final distributed edge audit is mandatory.

For regular lattices, a multiplier-based periodic coloring may be supplied as
a fast path, but it must pass the same distance audit.  The irregular graph
coloring is the reference path.

For a fixed physical $D$ and bounded region density, graph degree and color
count $G(D)$ are independent of total system size.  Image width is therefore
$G(D)r_B$, not $N_Br_B$.  This is the actual linear-scaling argument.

The coloring diagnostics are

* global region count and color count;
* occupancy range per color;
* minimum and percentile same-color distances;
* conflict-edge count after coloring (must be zero);
* image width per rank bucket.

## 8. Kernel-aware choice of exclusion distance

Distance alone does not set an error tolerance; the finite-$\eta$ spectral
kernel does.  Estimate its relevant cross-region envelope with a fixed-size
pilot:

1. Select canary regions deterministically across region sizes and atom
   types.
2. Apply each routed $F_k$ to a few isolated canary basis directions (and
   retain the present $K_k$ measurement only for the output-side A/B path).
3. Bin the captured-map cross products by source--target distance to obtain
   an upper quantile $\widehat\kappa_k(d)$.
4. Predict the far-color variance

$$
\widehat v_{P,k}(D)
\propto\sum_{d(B,C)\ge D}\widehat\kappa_k(d(B,C))^2.
$$

Begin with the cheapest trial distance.  Using tuning vectors and packing
pilot signs only, raise $D$ while the predicted packing error is above its
budget and the color-width cap is not reached.  Freeze $D$ before generating
the production signs.  Canary images are independent validation data and are
reusable across color trials.  The shifted-color and isolated-canary checks
already developed in `design_probing_acefd.md` should be reused rather than
inventing a symmetry indicator; that note shows why an unshifted symmetry
check is blind in a periodic system.

## 9. Packing replicas and an error certificate

For each rank bucket and color, form $L$ packed copies with independent,
pairwise-independent counter signs.  The key is

```
(packing_seed_domain, replica, stable_region_id, local_direction)
```

and must not contain the MPI rank.  Process replicas in batches if
$LGr_B$ exceeds the preferred SpMM width.

Choose $L$ from pilot signs and freeze it.  From the $L$ fresh production
captured maps, compute their mean and Welford variance at every owned grid
point.  For $L\ge2$ this gives

$$
\widehat{\operatorname{se}}_P(\mathbf r)
=\sqrt{s_P^2(\mathbf r)/L}.
$$

The independent residual blocks provide the analogous
$\widehat{\operatorname{se}}_R$.  Report their quadratic sum as pointwise,
masked-RMS, 95th-percentile, and maximum error indicators.  The maximum is a
diagnostic unless a simultaneous-confidence correction is applied.

Pilot adaptation is monotone:

* increase $r_B$ when the holdout residual dominates;
* increase $D$ when a small set of nearby cross terms dominates packing;
* increase $L$ when the distance-certified far field remains noisy;
* increase $n_R$ when residual sampling dominates after the rank cap.

If $a/n_R+b/L$ is the measured variance and one residual sample and one
packing replica cost $c_R$ and $c_P$, respectively, the cost-optimal local
allocation obeys

$$
\frac{L}{n_R}=\sqrt{\frac{b\,c_R}{a\,c_P}}.
$$

Use this ratio only after tuning vectors and pilot signs have measured $a$
and $b$; it is not an input heuristic.  Freeze the resulting $n_R$ and $L$
before the production columns are accumulated.

## 10. Public control and diagnostics

The preferred interface exposes one accuracy request and hard resource caps:

```
local_target_rms       # 0 disables adaptive local capture
local_rank_max
local_image_width_max
local_pack_replicas_max
local_setup_memory_max
```

Expert overrides may expose a fixed exclusion distance, rank, color count,
or replica count, but the default path is adaptive.  Keep `local_capture` and
`local_pack` as a legacy fixed-parameter mode for reproducibility.

Per energy, return

```
route
global_capture, global_error_estimate
local_rank_min/mean/max, local_rank_cap_fraction
basis_holdout_residual
color_count, minimum_same_color_distance
packing_replicas, packing_error_estimate
residual_error_estimate, total_error_estimate
image_columns, image_matvecs
```

`route` distinguishes `global_learned`, `local_learned`, and
`local_identity_probing`.

An estimator that cannot meet the target under the caps must say so.  It must
not silently return the best attempted map as a certified result.

## 11. Communication and scaling

Under weak scaling with fixed atoms per rank and fixed $\eta$:

| stage | per-rank scaling |
|---|---:|
| owned-atom RIB regions | $O(1)$ |
| stored filtered families | $O(JnN/P)$ |
| local Grams and bases | $O(1)$ regions of fixed dimension |
| region-directory metadata | bounded records plus scalar reductions |
| distance coloring | bounded-degree neighbor exchanges |
| packed image sweeps | fixed width $LGr_B$ through the existing operator halo |
| grid contraction | $O(N_{\rm grid}/P)$ at fixed width |

Total work is $O(N)$, which is order-optimal in system size for producing an
$O(N)$ real-space map.  No claim is made that the dependence on the requested
error is optimal until a matching lower bound is proved for the finite-range
spectral-kernel model.

## 12. Implementation sequence

### Phase A: correctness and observability

1. Fix the color count using the global region count, never the maximum local
   count.
2. Replace rank-based packing keys with stable region ids.
3. Add $L=2$ independent packing replicas and separate packing/residual error
   diagnostics.
4. Move the global route before the local image sweep.

These changes retain the current coloring and should land first because they
expose its real error.

### Phase B: settle the estimator before optimizing its coloring

5. Add positive square-root packed-identity probing as the full-rank
   reference path.
6. Add the two-stage input-side learned split: training/tuning first, then
   packed $F P F^\dagger$ images and $F(I-P)\xi_R$ production vectors.
7. A/B it against the current output-side $K$-image split at equal
   filter-vector work and equal error.  Do not replace the current route
   until the exact small-system identities and multi-seed comparison pass.

### Phase C: distance guarantee

8. Add the distributed region directory and conflict graph.
9. Add deterministic distributed graph coloring and the zero-conflict audit.
10. Retain the current periodic coloring only as a test baseline.

### Phase D: adaptation

11. Add eigenvalue-tail and holdout selection of $r_B$.
12. Add canary filter-envelope measurements and adaptive $D$.
13. Add cost-optimal adaptation of $n_R$ and $L$ under width/memory caps.
14. Cluster heterogeneous energy windows before local processing.
15. Route continuously over $0\le r_B\le d_B$; the endpoint $r_B=d_B$ is
    packed identity and has no residual samples.

### Phase E: optional advanced paths

16. Add nested/hierarchical color refinement so earlier image work contributes
    to later accuracy levels.
17. Add the regular-lattice multiplier coloring fast path.
18. Evaluate mixed precision for image sweeps only after the error certificate
    includes arithmetic error.

## 13. Required validation before a SOTA claim

### Correctness tests

* exact row cover and immutable atom ownership at multiple MPI counts;
* deterministic region ids and colors for a fixed atom partition;
* zero conflict edges for periodic, nonperiodic, anisotropic, and disordered
  cells;
* unpacked result agrees with the exact projector identity;
* for every test projector, the input-side captured and residual operators
  are positive, sum to $K$, and have the same residual trace as the present
  output-side split;
* unpacked square-root $P_B=I_B$ agrees with the complete-basis LDOS and has
  zero residual;
* packing mean converges to unpacked as $L$ grows;
* reported packing and residual standard errors have calibrated coverage over
  at least 30 independent trials;
* empty-window, rank-zero, unequal-regions-per-rank, and rank-cap cases.

### Scaling tests

* weak scaling with at least one region per rank, specifically catching the
  old $G=1$ collapse;
* fixed-$\eta$, fixed-target error through at least an order of magnitude in
  $N$;
* setup metadata, image halo bytes, grid halo bytes, memory, matvecs, and wall
  time reported separately;
* demonstrate bounded $G$, $r_B$, $L$, and $n_R$ at fixed target.

### Equal-cost baselines

On identical operators, grids, seeds, and matvec or wall-time budgets compare

* positive square-root Hutchinson;
* XDiag/Diag++-style global low-rank plus residual estimation;
* stochastic distance probing and hierarchical probing;
* deterministic fixed-phase and randomized square-root identity probing;
* the current periodic packed local capture;
* the preferred input-side packed local capture;
* unpacked local capture;
* real/energy-space embedded-fragment control variates where applicable.

Use periodic metal, disordered metal, semiconductor edge/defect, insulator,
and an inhomogeneous interface.  Report multi-seed confidence intervals for
pointwise masked RMS, 95th percentile, maximum error, integrated error,
positivity violations, wall time, and communication.

## 14. Publication claim enabled by this design

If the validation succeeds, the defensible claim is:

> An adaptive local-projection estimator for positive real-space spectral
> maps whose exact local split is made linear-scaling by distance-certified
> random-sign packing.  Independent estimates of residual and packing error
> select the local rank, exclusion distance, repetitions, and sample count;
> at fixed nonzero broadening and requested accuracy, all parameters and the
> per-rank work remain bounded under weak scaling.

This is stronger and more precise than calling the algorithm optimal.  A true
optimality claim additionally needs a lower bound in the same Hamiltonian
access and pointwise-error model.

## 15. Relation to the advanced literature

The design deliberately combines rather than renames established strengths:

* XDiag/XTrace exchangeable low-rank-plus-residual estimation:
  <https://doi.org/10.1137/23M1548323>;
* Diag++ and probabilistic diagonal bounds:
  <https://arxiv.org/abs/2201.10684>;
* hierarchical graph-distance probing:
  <https://doi.org/10.1137/18M1176427>;
* multiplier-based intermediate lattice coloring:
  <https://arxiv.org/abs/2606.00394>;
* domain-decomposed local-coherence deflation:
  <https://arxiv.org/abs/2401.14724>;
* real- and energy-space fragmented stochastic DFT:
  <https://doi.org/10.1063/5.0044163>;
* positive estimators for full local spectral maps:
  <https://doi.org/10.1103/qb1w-44r1>.

The proposed novelty remains the application-specific composition: filtered
random vectors learn trace-optimal block-local projectors; the positive
square-root split continuously connects low-rank residual sampling to
full-rank identity probing; and a distance-certified packing layer supplies a
separate statistical error certificate.

## 16. Disposition (2026-08-28)

This review was triaged against the implementation in `rs_ldos.hpp` and the
following was adopted, deferred, or adapted.  Section 13's full validation
battery remains the bar for any "SOTA" wording in the paper; nothing below
claims it.

**Adopted and landed (the original observability work):**

1. *Global color bound* (12.1).  `G` is now bounded by the global region
   count (`MPI_Allreduce SUM`), not the per-rank maximum.  The G = 1
   collapse -- one region on each of many ranks sharing one column -- is
   fixed and regression-tested at two ranks
   (`DistributedLocalCaptureKeepsTheExistingAtomPartition`).
2. *Stable packing keys* (12.2).  Sign keys are the region's minimum
   original global atom id (unique because atoms are uniquely owned), never
   the MPI rank: the packed estimate is invariant under repartitioning, the
   same property the counter-based sample seeds already had.
3. *Sign replicas with an error certificate* (12.3, section 9 reduced to
   its measurable core).  `local_pack_replicas` (default 2) independent
   realizations; the map averages them (crosstalk falls as 1/L) and their
   between-replica spread is reported per energy as `local_packing_se`,
   kept separate from `correction_rms` exactly as section 3 demands.
4. *Route before paying for local images* (12.4, section 4).  The global
   pass now runs first for every energy; regions, bases, and image sweeps
   are built only if some energy's measured capture routes local, and the
   shared basis and squared-table sweep are built from the routed energies
   alone (section 4's joint-basis rule).  A scan the global path fully
   serves pays nothing for having `local_capture` enabled.
5. *Distance audit* (the measurement half of sections 6-7).  Region records
   (center, bounding radius, color; 5 doubles) are gathered once and the
   minimum-image gap between same-color bounding spheres is reported as
   `min_same_color_distance` -- negative means overlapping spheres shared a
   column.  Skipped above 16384 regions (reported as unaudited) so setup
   metadata stays bounded.

**Deferred, with reasons:**

* *Positive square-root identity probing and the input-side local split*
  (section 3 and Phase B).  These are now the highest-priority estimator A/B
  tests.  They are algebraically stronger than the present output-side split,
  but no production code or equal-work measurement exists yet; the paper must
  not claim their cost or accuracy until those tests pass.
* *Distributed directory + Jones-Plassmann coloring + zero-conflict audit*
  (sections 6-7, Phase C).  Correct at scale, but heavy distributed
  infrastructure; at current rank counts the audit above measures the same
  defect the certified coloring would fix, and the replica certificate
  bounds its damage.  Trigger: an audited negative gap that the replica
  average cannot absorb on a production system.
* *Error-targeted routing, adaptive rank, filter-envelope distance, and
  cost-optimal (n_R, L)* (sections 4-5, 8-9, Phase D).  The 0.9 capture
  route stays: it is outcome-based (measured capture, not a Gram heuristic)
  and the rank-heuristic misroute it replaced is documented in the code.
  Per-fragment rank already adapts below the `local_capture` cap through
  the relative + absolute eigenvalue filters.
* *Public error-target interface* (section 10).  Pre-release the fixed
  knobs stay the interface; the new diagnostics (route, packing SE, region
  and color counts, audited distance) are exactly the observability that
  interface would need, so it can be added without further estimator work.
* *Strict three-domain randomness separation* (section 3.3).  The
  first/second-half split and cross-fitted LOO already separate basis
  construction from contraction; routing still reads the production
  capture.  Accepted as a known, documented second-order bias until the
  interface work above gives adaptivity a place to live.
