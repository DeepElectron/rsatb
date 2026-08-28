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

## 3. Error decomposition

Condition on the vectors used to construct and tune the block projector
$P=\sum_B P_B$.  The exact local split is

$$
K=(I-P)K(I-P)+PK+KP-PKP.
$$

The first term is sampled with independent vectors.  Let $D(\mathbf r)$ be
the diagonal of the last three terms.  With one separate image column for
every local basis direction, $D$ is evaluated exactly.

Packing introduces independent signs $\sigma^{(\ell)}_{Bj}$, where $\ell$
labels a packing replica.  The captured map from replica $\ell$ can be
written

$$
\widehat D_\ell(\mathbf r)
=D(\mathbf r)+
\sum_{B\ne C\atop c(B)=c(C)}
\sigma^{(\ell)}_{Bj}\sigma^{(\ell)}_{Cj}
d_{BCj}(\mathbf r).
$$

Pairwise-independent counter signs give
$\mathbb E_\sigma\widehat D_\ell=D$.  Averaging $L$ replicas gives the final
estimator

$$
\widehat\rho(\mathbf r)=
\frac1L\sum_{\ell=1}^L\widehat D_\ell(\mathbf r)
+\frac1{n_R}\sum_{i=1}^{n_R}
|G(I-P)F\xi_i|^2(\mathbf r).
$$

The learning, tuning, residual, packing-pilot, and packing-production domains
must use disjoint counter keys.  Conditional on the frozen estimator design,
the two production errors are then independent and

$$
\operatorname{Var}\widehat\rho(\mathbf r)
=\frac{v_R(\mathbf r)}{n_R}
+\frac{v_P(\mathbf r)}{L}.
$$

This equation is the controller's error model.  `correction_rms` is not a
substitute for either variance and must not be presented as an error bar.

### 3.1 Randomness roles and selection bias

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

1. Run the ordinary filtered sample sweep and store the learning, tuning, and
   production families needed by the global and local projectors.
2. Construct the global candidate from learning vectors and evaluate its
   predicted production error with tuning vectors.
3. Freeze the route for every energy.  Accept the global path where the
   tuning error meets the requested target.
4. Only the remaining energies enter local basis construction and image
   sweeps; their final residual uses production vectors only.

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
2. Apply each routed $K_k$ to a few isolated canary basis directions.
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

### Phase B: distance guarantee

5. Add the distributed region directory and conflict graph.
6. Add deterministic distributed graph coloring and the zero-conflict audit.
7. Retain the current periodic coloring only as a test baseline.

### Phase C: adaptation

8. Add eigenvalue-tail and holdout selection of $r_B$.
9. Add canary kernel-envelope measurements and adaptive $D$.
10. Add cost-optimal adaptation of $n_R$ and $L$ under width/memory caps.
11. Cluster heterogeneous energy windows before local processing.

### Phase D: optional advanced paths

12. Add nested/hierarchical color refinement so earlier image work contributes
    to later accuracy levels.
13. Add the regular-lattice multiplier coloring fast path.
14. Evaluate mixed precision for image sweeps only after the error certificate
    includes arithmetic error.

## 13. Required validation before a SOTA claim

### Correctness tests

* exact row cover and immutable atom ownership at multiple MPI counts;
* deterministic region ids and colors for a fixed atom partition;
* zero conflict edges for periodic, nonperiodic, anisotropic, and disordered
  cells;
* unpacked result agrees with the exact projector identity;
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
* the current periodic packed local capture;
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

The proposed novelty remains the application-specific composition: learned
energy-filtered local projectors, the exact LDOS restoration identity, and an
adaptive distance-certified packing layer with a separate statistical error
certificate.

## 16. Disposition (2026-08-28)

This review was triaged against the implementation in `rs_ldos.hpp` and the
following was adopted, deferred, or adapted.  Section 13's full validation
battery remains the bar for any "SOTA" wording in the paper; nothing below
claims it.

**Adopted and landed (Phase A, plus two pieces of B/C):**

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

* *Distributed directory + Jones-Plassmann coloring + zero-conflict audit*
  (sections 6-7, Phase B).  Correct at scale, but heavy distributed
  infrastructure; at current rank counts the audit above measures the same
  defect the certified coloring would fix, and the replica certificate
  bounds its damage.  Trigger: an audited negative gap that the replica
  average cannot absorb on a production system.
* *Error-targeted routing, adaptive rank, kernel-envelope distance, and
  cost-optimal (n_R, L)* (sections 4-5, 8-9, Phase C).  The 0.9 capture
  route stays: it is outcome-based (measured capture, not a Gram heuristic)
  and the rank-heuristic misroute it replaced is documented in the code.
  Per-fragment rank already adapts below the `local_capture` cap through
  the relative + absolute eigenvalue filters.
* *Public error-target interface* (section 10).  Pre-release the fixed
  knobs stay the interface; the new diagnostics (route, packing SE, region
  and color counts, audited distance) are exactly the observability that
  interface would need, so it can be added without further estimator work.
* *Strict three-domain randomness separation* (section 3.1).  The
  first/second-half split and cross-fitted LOO already separate basis
  construction from contraction; routing still reads the production
  capture.  Accepted as a known, documented second-order bias until the
  interface work above gives adaptivity a place to live.
