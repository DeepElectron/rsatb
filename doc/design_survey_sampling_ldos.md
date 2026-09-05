# A survey-sampling estimator for local density-of-states maps: fixed accuracy at linear cost

*Written 2026-09-03; since the cleanup of 2026-09-05 this is the LDOS map's only estimator (`rsatb/calc/rs_ldos.hpp`, `ComputeRealSpaceLdos`; options `n_samples`, `block`, `sketch_families`; outputs `ldos`, `capture`, `sketch_rank`, `predicted_rel_error`). Validation in `pub_exp/spectrum_realspace/results/val8/tpl_*` (Section 9). Literature context in `pub_exp/variance_reduction/SYNTHESIS.md`.*

---

## 1. The quantity we want

Let $H$ be the tight-binding Hamiltonian of a system with $N$ basis functions, with eigenpairs $H\psi_i = \varepsilon_i \psi_i$. For a grid point $\mathbf r$ let $\phi_{\mathbf r}$ be the vector whose entries are the values of the basis functions at $\mathbf r$; it has nonzero entries only for the basis functions whose atoms lie within the basis range of $\mathbf r$. The local density of states at energy $E$ and resolution $\eta$ is

$$
n(\mathbf r) \;=\; \sum_i \, w_i \, \phi_i^2 , \qquad w_i = \exp\!\left[-\frac{(\varepsilon_i - E)^2}{2\eta^2}\right], \qquad \phi_i = \langle \psi_i , \phi_{\mathbf r}\rangle .
\tag{1}
$$

The weights $w_i$ form a Gaussian window around $E$. Writing $p(\varepsilon) = \exp[-(\varepsilon-E)^2/4\eta^2]$ for the square root of the window, equation (1) is the squared norm of a filtered vector,

$$
n(\mathbf r) \;=\; \big\| \, p(H)\,\phi_{\mathbf r} \, \big\|^2 .
\tag{2}
$$

Two facts about the window matter for everything below.

* **Few states carry weight.** Only states with $|\varepsilon_i - E| \lesssim 4\eta$ have an appreciable $w_i$. Call the span of those states the *window space* $S$ and its dimension $m_S$. Because the number of states per unit energy is proportional to the size of the system, $m_S = \rho_S N$ with $\rho_S$ a fixed small fraction (about 2 % in silicon at $\eta = 0.25$ eV).
* **Every state in the window is spatially extended.** A state of energy width $\eta$ spreads over a distance $\hbar v/\eta \approx 50$ Å. No small neighbourhood of $\mathbf r$ contains the information needed for $n(\mathbf r)$; this is why the local surrogates that solve the density problem do not apply here.

## 2. The plain estimator and its noise

Draw a random vector $\xi$ with independent entries of mean zero and unit variance, so that $\mathbb E[\xi\xi^{\mathsf T}] = I$. One Chebyshev sweep gives $y = p(H)\,\xi$, and

$$
\mathbb E\big[\,|\phi_{\mathbf r}^{\mathsf T} y|^2\,\big] \;=\; \phi_{\mathbf r}^{\mathsf T}\, p(H)\,\mathbb E[\xi\xi^{\mathsf T}]\, p(H)\, \phi_{\mathbf r} \;=\; n(\mathbf r).
\tag{3}
$$

The scalar $\phi_{\mathbf r}^{\mathsf T} y = \sum_i p(\varepsilon_i)\,\phi_i\,\xi_i$ is a sum of many independent terms, hence (very nearly) Gaussian, with variance $n(\mathbf r)$. Its square is therefore $n(\mathbf r)$ times a $\chi^2$ variable with one degree of freedom, whose variance is twice its mean squared:

$$
\operatorname{Var}\big[\,|\phi_{\mathbf r}^{\mathsf T} y|^2\,\big] \;=\; 2\,n(\mathbf r)^2 .
\tag{4}
$$

Averaging over $n$ independent probes gives a relative error of $\sqrt{2/n}$ at every point, independent of $N$, of $\eta$, and of the point itself: 17.7 % for 64 probes, 5 % for 800. This is what we measure (0.97–0.98 of the bound), and it is a floor: no estimator that squares a single random probe can do better, because each probe delivers one Gaussian amplitude per point and the square of one Gaussian amplitude has this variance. We call it the *phase noise*: it comes from the random signs $\xi_i$ with which the probe excites each window state.

## 3. A second kind of randomness: which states a sketch captures

The same $n$ sweeps also produce a *sketch* of the window space. Alongside $y = p(H)\xi$, the recursion delivers, for free, the vectors

$$
Y \;=\; \big[\, T_{js}(H)\, q(H)\, \xi_l \,\big]_{\,j=0,\dots,J-1;\; l=1,\dots,n}
\tag{5}
$$

where $q$ is a *sketch filter* concentrated on the window (Section 5 explains the choice), and $T_{js}$ are Chebyshev polynomials of orders $0, s, 2s, \dots$ with the stride $s$ chosen so that successive families are decorrelated over the window. $Y$ has $k = Jn$ columns, all lying (essentially) in $S$. Let $\Pi$ be the orthogonal projector onto the span of $Y$; it has rank $k$ (or $m_S$, if $k \ge m_S$).

The key property, established in the subspace-deflation note, is that the *images* $p(H)\,Y$ are also free: they are further tables on the same recursion. Consequently the exact quantity

$$
n_{\rm cap}(\mathbf r) \;=\; \big\| \, \Pi\, p(H)\, \phi_{\mathbf r} \, \big\|^2
\tag{6}
$$

can be assembled for every grid point without any additional sweep: it is a quadratic form of stored vectors. It is the part of $n(\mathbf r)$ that lives in the captured subspace. Our existing "deflated" estimator adds to it a stochastic estimate of the remainder,

$$
n(\mathbf r) \;=\; n_{\rm cap}(\mathbf r) \;+\; \mathbb E\big[\,|\phi_{\mathbf r}^{\mathsf T} p(H)(I-\Pi)\xi|^2\,\big],
\tag{7}
$$

and the remainder carries phase noise scaled by the uncaptured fraction: relative error $(1-c)\sqrt{2/n}$, where $c$ is the captured fraction of the map. Making the remainder vanish means $c \to 1$, i.e. $k \approx m_S \propto N$: the cost of an exact map is quadratic in $N$.

The estimator proposed here **drops the remainder** and uses the captured part alone:

$$
\boxed{\; n_T(\mathbf r) \;=\; \frac{n_{\rm cap}(\mathbf r)}{c} \;=\; \frac{\|\Pi\, p(H)\,\phi_{\mathbf r}\|^2}{c} \;}
\tag{8}
$$

with $c$ one global number, the fraction of the whole map that the sketch captured (Section 6). The idea is that a random sketch is a random *sample of the window's states*; the captured part is the exact contribution of that sample; and dividing by the sampling fraction extrapolates it to the whole population, exactly as a survey extrapolates from the people it interviewed. The only randomness left is *which* states were sampled.

## 4. Why the extrapolation is unbiased, and what its noise is

The precise hypothesis is that the captured subspace is *isotropic*: $\Pi$ is a uniformly random $k$-dimensional subspace of $S$, with no preferred directions. (Section 5 says how to arrange this.)

**Proposition 1 (no bias).** If $\Pi$ is a uniformly random rank-$k$ projector on $S$, then for every point $\mathbf r$ and every window $p$ supported in $S$,

$$
\mathbb E\big[\, \|\Pi\, p(H)\,\phi_{\mathbf r}\|^2 \,\big] \;=\; \frac{k}{m_S}\; \|p(H)\,\phi_{\mathbf r}\|^2 \;=\; c\, n(\mathbf r), \qquad c = \frac{k}{m_S}.
\tag{9}
$$

*Proof.* The average projector $\mathbb E[\Pi]$ is unchanged by any rotation of $S$, because the distribution of $\Pi$ is. The only linear maps on $S$ with that property are multiples of the identity, and the trace of $\mathbb E[\Pi]$ equals the trace of $\Pi$, which is $k$. Hence $\mathbb E[\Pi] = (k/m_S)\, I_S$. For any fixed vector $v \in S$, $\mathbb E\|\Pi v\|^2 = v^{\mathsf T}\mathbb E[\Pi] v = (k/m_S)\|v\|^2$. Apply this to $v = p(H)\phi_{\mathbf r}$, which lies in $S$ because $p$ vanishes outside the window. $\square$

Nothing about locality, smoothness, or the spatial shape of the window states was used; the statement holds point by point and for any kernel.

**Proposition 2 (noise).** Under the same hypothesis, the captured fraction of a fixed vector,

$$
\frac{\|\Pi v\|^2}{\|v\|^2}, 
$$

follows a Beta distribution with parameters $\tfrac{k}{2}$ and $\tfrac{m_S-k}{2}$, whose mean is $c$ and whose variance is $\dfrac{2\,c\,(1-c)}{m_S+2}$. Therefore the relative error of $n_T(\mathbf r)$ is

$$
\boxed{\;\frac{\sqrt{\operatorname{Var}[n_T(\mathbf r)]}}{n(\mathbf r)} \;=\; \sqrt{\frac{2\,(1-c)}{c\,m_S + 2}} \;\approx\; \sqrt{\frac{2\,(1-c)}{k}}\;}
\tag{10}
$$

*Proof.* Rotate coordinates in $S$ so that $v$ points along the first axis. A uniformly random $k$-dimensional subspace captures of the first axis the same fraction that a uniformly random unit vector of $\mathbb R^{m_S}$ has in its first $k$ coordinates. The squared coordinates of a uniformly random unit vector are Dirichlet-distributed, and the sum of $k$ of them out of $m_S$ is Beta$(\tfrac k2, \tfrac{m_S-k}{2})$. Its variance follows from the Beta formula $\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$ with $\alpha = k/2$, $\beta = (m_S-k)/2$. Dividing by $c^2$ gives (10). $\square$

Equation (10) is the classical formula for the sampling error of a survey: $k$ is the sample size, $m_S$ the population size, and $(1-c)$ the finite-population correction, the fraction of the population *not* interviewed. It replaces the phase-noise formula $\sqrt{2/n}$ of (4). The two are of the same form, but the denominator is now $k = Jn$ rather than $n$, and the numerator carries the factor $(1-c)$.

## 5. Making the sketch isotropic: the flat-top filter

Proposition 1 needs $\Pi$ to be uniformly random within $S$. Whether it is depends on the sketch filter $q$.

A sketch column is $\sum_i q(\varepsilon_i)\,\xi_i\,\psi_i$: a random vector in $S$ whose component along state $i$ is weighted by $q(\varepsilon_i)$. If $q$ were the estimator's own filter $p$, states near the centre of the window (large $p$) would be captured preferentially and states near its edge rarely; $\mathbb E[\Pi]$ would then be diagonal in the eigenbasis with entries increasing with $p(\varepsilon_i)$ rather than a multiple of the identity. The estimate would still be unbiased *in space*, but it would correspond to a narrower, distorted energy kernel. The remedy is to make the sketch weights equal across the window: choose a *flat-top* sketch filter

$$
q(\varepsilon) \;=\; \tfrac12\Big[\operatorname{erf}\!\Big(\frac{\varepsilon - E + 4\eta}{\eta}\Big) - \operatorname{erf}\!\Big(\frac{\varepsilon - E - 4\eta}{\eta}\Big)\Big],
\tag{11}
$$

which equals one on $|\varepsilon - E| < 4\eta$ (where the window $w$ exceeds $e^{-8}$), zero outside, and turns over in a width $\eta$ so that its Chebyshev degree is the same as the Gaussian's. With equal weights the sketch columns are isotropic Gaussian vectors in $S$, their span is a uniformly random subspace, and Proposition 1 applies with the estimator's Gaussian kernel unchanged. The images needed for (6) are then the tables of the product $p\,q$ instead of $p^2$; the split (7) remains valid for any projector, so the deflated map is still available beside $n_T$.

A flat sketch has a second benefit. A Gaussian sketch has weight tails, and a random sketch needs about twice as many columns as there are states carrying 99.9 % of the weight before it spans them all (measured on silicon 12³: capture 0.76 at $k = 1728$, 1.0000 at $k = 3696$). A flat spectrum has no tails: $k = m_S$ isotropic vectors span $S$ with probability one. Exactness therefore arrives at $k \approx m_S$, with no oversampling.

The $J$ families of one probe are not $J$ independent random vectors; they are one random vector modulated by $J$ different Chebyshev polynomials. The stride $s$ is chosen so that the modulation pattern across the window states is effectively random, and the observation that the captured fraction grows in proportion to $J$ (the "capacity multiplies by $J$" result of the deflation note) is evidence that the span behaves as if the columns were independent. Equation (10) tests the same assumption at the level of the variance; where it fails, the effective $k$ is smaller than $Jn$ and the measured error exceeds the formula.

## 6. The normalisation

The fraction $c$ is a single number for the whole map. It is measured on the probes themselves, in state space, and *cross-fitted*: for each probe block $b$, the sketch $\Pi_{-b}$ built from the other blocks' columns is applied to the block's own filtered probes,

$$
c_{-b} \;=\; \frac{\sum_{l\in b} \|\Pi_{-b}\, p(H)\,\xi_l\|^2}{\sum_{l\in b} \|p(H)\,\xi_l\|^2}, \qquad
c \;=\; \min\!\Big(1,\ \frac{\sum_b c_{-b}\,\sum_{l\in b}\|p(H)\xi_l\|^2 \cdot \operatorname{rank}\Pi/\operatorname{rank}\Pi_{-b}}{\sum_l \|p(H)\xi_l\|^2}\Big).
\tag{14}
$$

Three properties make this the right choice. (i) It is *out of sample*: a probe is never projected onto columns it helped build (the in-sample projection over-captures a probe's own state, by a large factor when $k \ll m_S$). (ii) Under the isotropy of Proposition 1, $c_{-b} = \operatorname{rank}\Pi_{-b}/m_S$, so the rescaling by the rank ratio gives the full sketch's $c = \operatorname{rank}\Pi/m_S$ with no further assumption. (iii) It is *exactly* one when the sketch spans the window: every $p(H)\xi_l$ then lies in the span and the ratios are 1, so the exact regime carries no normalisation noise at all (measured: capture 1.000000 and a map error of $8\times10^{-8}$ on the chain fixture). A grid-integral normalisation, by contrast, would inherit the phase noise of the plain estimate's integral, $\sim 1/\sqrt{n\,m_S}$, which is negligible on production windows but not zero. The leave-one-block-out Grams are principal submatrices of the one Gram already formed; their eigensolves cost $O(n_{\rm blocks}\,k^3)$, minutes at $k = 4000$.

The noise of (14) is that of a survey over the probes' states, $\sim\sqrt{2(1-c)/k}\,/\sqrt{n_{\rm blocks}}$; on the chain fixture (a few hundred window states, four blocks) it is ~8 %, on the silicon runs below 1 %.

## 7. Why this is linear scaling at a fixed accuracy

Fix the run: $n$ probes, $J$ families, hence $k = Jn$; fix a target relative error $\epsilon$.

**The cost is proportional to $N$.** The sweeps are $n$ Chebyshev recursions of degree $d(1+J/10)$ on a sparse matrix, i.e. $O(nJdN)$ operations. The captured part needs one Gram matrix of size $k$, one eigenvalue problem of size $k$, and one contraction of $k$ vectors onto the grid: $O(k^2 N + k^3)$. At fixed $(n, J)$ all of this is linear in $N$.

**The error is bounded independently of $N$.** Insert $c = k/(\rho_S N)$ into (10):

$$
\epsilon(N) \;=\; \sqrt{\frac{2}{k}\Big(1 - \frac{k}{\rho_S N}\Big)} \;\le\; \sqrt{\frac{2}{k}} \qquad\text{for every } N .
\tag{12}
$$

The error grows with the system, because a fixed sample is a smaller fraction of a larger population, but it never exceeds $\sqrt{2/k}$; at finite $N$ the factor $(1-c)$ is a bonus, and once $\rho_S N \le k$ the sketch spans the whole window and the estimate is exact.

Hence a fixed setting delivers $\epsilon \le \sqrt{2/k}$ at every size, at a cost proportional to $N$. To guarantee a target $\epsilon$ at *all* sizes one chooses

$$
k = \frac{2}{\epsilon^2}, \qquad\text{i.e.}\qquad n = \frac{2}{J\,\epsilon^2}\ \text{probes},
\tag{13}
$$

a number that does not depend on $N$. That is the statement "linear scaling at a fixed accuracy". For the plain estimator the same target needs $n = 2/\epsilon^2$ probes: $J$ times more sweeps. Since the $J$ families lengthen each sweep by the factor $1 + J/10$, the gain at equal cost is $J/(1+J/10)$ in variance: 5.5 at $J = 12$, approaching 10 for large $J$.

The three estimators side by side:

| estimator | relative error per point | probes for a target $\epsilon$ at all $N$ |
|---|---|---|
| plain, eq. (3) | $\sqrt{2/n}$ | $2/\epsilon^2$ |
| deflated, eq. (7) | $(1-c)\sqrt{2/n}$, $c = k/m_S$ | $2/\epsilon^2$ as $N\to\infty$; exact only for $k \approx \rho_S N$ (quadratic) |
| survey, eq. (8) | $\sqrt{2(1-c)/k}$, $k = Jn$ | $2/(J\epsilon^2)$ |

The survey estimator is better than the deflated one whenever $(1-c)\,J > 1$, i.e. whenever the sketch captures less than $1 - 1/J$ of the map; the deflated estimator wins only close to exactness, where its factor $(1-c)^2$ vanishes faster. In the regime the question is about, large $N$ at a fixed budget, $c$ is small and the survey estimator wins by the factor $J$ in variance.

Numbers for silicon at $\eta = 0.25$ eV with $n = 64$, $J = 12$ ($k = 768$):

| cell | $N$ | $m_S$ | $c$ | survey error, eq. (10) | plain error |
|---|---|---|---|---|---|
| $8^3$ | 13,312 | ≈ 700 | 1 | exact | 17.7 % |
| $12^3$ | 44,928 | ≈ 2,200 | 0.35 | 4.1 % | 17.7 % |
| $20^3$ | 208,000 | ≈ 10,000 | 0.08 | 4.9 % | 17.7 % |
| $N \to \infty$ | — | — | 0 | 5.1 % ($J = 6$: the survey law holds exactly; $J = 12$ carries a design effect of 2.5, Section 9) | 17.7 % |

## 8. What was exploited, and what was not

The density problem is solved by *spatial* locality: the density matrix decays within a buffer, a local surrogate captures the dominant part exactly, and only a small residual fraction $f$ is left to the probes, giving an error $f\sqrt{2/n}$. For the LDOS the window states are extended, so no local surrogate captures a dominant part, and the residual of the deflated estimator, $(1-c)$, grows to one with $N$.

The survey estimator uses *spectral* locality instead. The population it samples is the window space, a small fraction $\rho_S$ of all states; the sample is drawn inside that population by the flat-top filter; and the sample size $k$ is fixed by the run, not by the system. The residual fraction $(1-c)$ appears in the noise (10) linearly, as the finite-population correction, divided by $k$; the phase noise of the plain estimator does not appear at all, because no random probe is ever squared. The estimate is a deterministic quadratic form of the captured vectors, and its only randomness is the choice of sample.

Two things the estimator does not do. It does not make an *exact* map cheaper: exactness still requires $k \approx m_S$, which is proportional to $N$. And it does not push the error below $\sqrt{2/k}$ at large $N$: the target error sets $k$, and $k$ sets the cost per atom. What it changes is the constant that multiplies $N$ for a given accuracy, by a factor $J$ in variance.

## 9. Validation on silicon $8^3$

Three runs on the 1,024-atom cell (13,312 basis functions), compared with the exact $k$-point fold `kconv/si_rs_k8` at the three energies 8.8, 16.8 and 18.3 eV. Errors are the rms deviation over the map relative to the map's rms; "deflated" is the cross-fitted estimator (7) from the same run; $c$ and $k$ are the realised capture and sketch rank.

| run | estimator | E = 8.8 | 16.8 | 18.3 |
|---|---|---|---|---|
| plain, $n=64$ | eq. (3) | 17.4 % | 18.0 % | 18.3 % |
| $n=64$, $J=4$ | deflated | 12.3 % | 14.1 % | 8.4 % |
| flat sketch, $n=64$, $J=12$ ($k=768$) | deflated | 6.4 % | 7.0 % | 2.9 % |
| | **survey**, $c$ = 0.90 / 0.93 / 0.99 | **2.5 %** | **1.9 %** | **0.6 %** |
| | eq. (10) | 1.6 % | 1.5 % | 0.5 % |
| flat sketch, $n=32$, $J=12$ ($k=384$) | deflated | 19.8 % | 20.4 % | 16.1 % |
| | **survey**, $c$ = 0.46 / 0.42 / 0.66 | **7.8 %** | **8.3 %** | **9.6 %** |
| | eq. (10) | 5.3 % | 5.7 % | 4.2 % |
| Gaussian sketch, $n=64$, $J=12$ | survey, $c$ = 0.99 / 0.96 / 1.00 | 0.7 % | 1.4 % | 0.3 % |

Three conclusions.

1. **Unbiased, as Proposition 1 says.** The mean error of the survey map equals that of the deflated map at every energy (−0.85, +0.26, −0.45 % at $n=64$; it is inherited from the normalisation, Section 6, and is at the level of the $k$-point reference's own broadening match).
2. **The noise is survey noise, not phase noise.** At the same sweeps the survey map is 3–7 times more accurate than the plain estimator and 2.5–3 times more accurate than the deflated one; halving the probes ($k$ from 768 to 384, $c$ from 0.9 to 0.45) raised the error by the factor $\sqrt{(1-c)/k}$ predicts, from ≈2 % to ≈8 %.
3. **A design effect that comes from the families, not the probes.** At $n = 64$, $J = 12$ the realised variance exceeds equation (10) by a factor 2.5 (1.6 in the error). In survey language the sample is *clustered*: the $J$ families derived from one probe are correlated draws. The diagnostic runs isolate this:

| run | $k$ | $c$ (E = 8.8 eV) | survey error | eq. (10) | design effect |
|---|---|---|---|---|---|
| $n = 64$, $J = 12$ | 768 | 0.904 | 2.51 % | 1.58 % | 2.5 |
| $n = 128$, $J = 6$ | 768 | 0.963 | 1.01 % | 0.99 % | 1.0 |
| $n = 64$, $J = 24$ | 976–1079 | 1.000 | 0.00 % | exact | — |

   With six families per probe the survey law holds exactly; with twelve, the effective sample size is about $k/2.5$. Families are cheap (they lengthen a sweep by $1 + J/10$) but not free in this sense, so the efficient setting for the survey estimator is a moderate $J \approx 6$ and the probes supplying the rest of $k$: variance per unit cost $\propto J/[d_{\rm eff}(J)(1+J/10)]$ is 3.8 at $J = 6$ against 2.2 at $J = 12$. The $J = 24$ run shows that clustering does not limit the *rank*: the sketch reached $k \ge m_S$ and the map became exact on this cell, which also measures $m_S \approx 760$–1080 for the three windows, consistent with the estimate of Section 1.

The Gaussian-sketch run reaches $c \approx 0.99$ on this small cell, so the kernel distortion of Section 5 is invisible here (it is a $(1-c)$ effect); it needs a cell with $c \ll 1$ to be seen.

## 10. Choosing families or probes

Both knobs buy sketch columns, $k = Jn$, and the map cost splits into a part that depends on how $k$ is split and a part that does not. The sweeps cost $n\,(1 + J/10)$ recursions: a probe is a full recursion, a family one more accumulation per step, about a tenth of one. The Gram matrix, its eigenproblem and the contraction onto the grid depend on $k$ alone. Per sketch column the sweep price is therefore

| way to add a column | sweep units per column |
|---|---|
| one more probe at $J = 6$ | $1.6/6 = 0.27$ |
| one more probe at $J = 12$ | $2.2/12 = 0.18$ |
| one more family, at any $n$ | $0.10$ |

so on price alone families always win. What limits them is their worth. The $J$ columns of one probe are the same random vector seen through $J$ energy combs; the diagnostic of Section 9 shows that up to about six combs they act as six independent draws ($d_{\rm eff} = 1.0$), while at twelve the effective sample is $k/2.5$. Beyond $J \approx 6$ extra families are redundant *as random draws* of the window, though they remain linearly independent and keep raising the rank.

This gives one rule with two regimes.

**Survey regime, $k < m_S$.** Take $J = 6$ and buy accuracy with probes: for a target $\epsilon$, $k \approx 2(1-c)/\epsilon^2$ and $n = k/6$ rounded up to whole blocks. Raising $J$ past six here spends sweep time on columns that add rank but not information.

**Exact regime, $k \ge m_S$.** Raise $J$ instead. Exactness needs the columns to span the window, not to be independent draws, and the $n = 64$, $J = 24$ run of Section 9 spanned the whole $8^3$ window with the same probes that at $J = 12$ captured 0.9. Here the cheapest setting is the largest $J$ memory allows, with $n = m_S/J$, and the error drops to the numerical floor.

**Which regime to aim for.** The survey regime at $J = 6$ costs about $1.6\,n = 0.53\,(1-c)/\epsilon^2$ sweep units; the exact regime at $J = 24$ costs $3.4\,n = 0.14\,m_S$. Exactness is cheaper whenever

$$
\epsilon \;<\; \sqrt{\frac{3.8\,(1-c)}{m_S}} .
\tag{15}
$$

For silicon $12^3$ ($m_S \approx 4{,}000$ for the flat-top window, $c \approx 0.2$) the crossover is about 3 %: a target above it is met most cheaply by the survey estimator at $J = 6$, a target below it by the exact route at large $J$. The crossover falls as $1/\sqrt{m_S}$, so at larger cells the survey regime covers an ever wider range of targets, which is the regime Section 7 is about.

## 11. Scope

The construction applies to any quantity of the form $\|f(H)\,\phi\|^2$ whose filter $f$ is supported on a small fraction of the states: LDOS maps at any resolution and with any kernel, spectral functions, energy-resolved densities. It also applies to the density itself ($f$ the square root of the Fermi function, support the occupied band, $\rho_S \approx 1/2$), where the survey noise $\sqrt{2/k}$ still improves on $\sqrt{2/n}$ by $\sqrt J$ and could serve as the residual estimator behind the fragment surrogate. It does not apply to quantities that are not squared norms of a single filtered vector, and it loses efficiency when the window is so narrow that the $J$ modulated families cease to be independent samples (Section 5).

## 12. Code map (after the cleanup of 2026-09-05)

`rsatb/calc/rs_ldos.hpp` holds one engine, `detail::FilteredSquareMap`, driven by
`MapEngineOptions`, and two thin entry points:

* `ComputeRealSpaceLdos` (survey = true): tables for the estimator filter $p$
  (sqrt-Gaussian, `SqrtGaussianTables`), the flat-top sketch $q$
  (`FlatTopTables`, eq. 11) with its $J$ comb modulations, and the images $pq$
  (`ProductTable`); one Chebyshev recursion per probe block accumulates all of
  them (`rs_ldos/sweep`); the survey stage (`rs_ldos/survey`) forms the sketch
  Gram, whitens it, contracts the images of the orthonormal basis onto the grid
  (the captured map, eq. 6), measures the capture fraction on the probes in
  state space from the leave-one-block-out principal submatrices of that Gram
  (eq. 14), and reports `ldos`, `capture`, `sketch_rank`,
  `predicted_rel_error` (eq. 10). Every table is truncated where its
  coefficients fall below $10^{-4}\,\epsilon$ of its largest. The options
  are `n_samples`, `block` and `sketch_families` (Section 10 for the split).
* `ComputeRealSpaceDensity` (`rs_density.hpp`, survey = false): the sqrt-Fermi
  table, plain squares, the fragment surrogate control variate of
  `rs_fragment.hpp` (`fragment_core`, `fragment_buffer`, `fragment_workspace_mb`,
  `fragment_coefficient`) for gapped systems, and `energy_windows` for metals.

The two never combine; the engine refuses a fragment surrogate under the survey
estimator. `complete_basis` sweeps the identity instead of probes for both and is
the validation mode the tests use.
