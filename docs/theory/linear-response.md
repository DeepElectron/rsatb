---
description: Kubo–Greenwood DC conductivity, optical conductivity and dielectric function, mean-square displacement and diffusion, and the quantum metric.
---

<p class="rs-kicker">Theory</p>

# Linear response

<div class="rs-lead" markdown>
Transport observables are correlation functions of the velocity operator.
RSATB evaluates them with the same random-state and Chebyshev machinery as the
DOS, so they scale the same way.
</div>

## DC conductivity

The Kubo–Greenwood conductivity at zero temperature is

$$
\sigma_{\alpha\beta}(E) = \frac{\pi\hbar e^2 g_s}{\Omega}\,
\operatorname{Tr}\big[\hat v_\alpha\,\delta(E-\hat H)\,\hat v_\beta\,\delta(E-\hat H)\big],
$$

with $\Omega$ the normalization volume and $g_s$ the spin degeneracy. Both
delta functions are expanded in Chebyshev polynomials, giving a matrix of
moments $\mu_{mn} = \operatorname{Tr}[\hat v_\alpha T_m(\tilde H)\hat v_\beta
T_n(\tilde H)]$ that RSATB evaluates with row FFTs over the Chebyshev nodes
(`method = "fftcheb"`). The `dc_spectrum` task returns $\sigma(E)$ on those
nodes; `dc` folds it with $-\partial f_F/\partial E$ at one $(\mu, T)$.
`formulation = "bastin"` adds the Fermi-sea term to the Fermi-surface
(Greenwood) term.

**Normalization.** For a 2D sheet padded with vacuum, a conductivity per
volume is meaningless; `axis` chooses what $\Omega$ is — the cell volume
(`"abc"`), the area of a lattice plane (`"ab"`, …) for a sheet, or a lattice
vector's length for a wire.

## Optical conductivity

The frequency-dependent conductivity follows from the current–current
correlation in time,

$$
\sigma_{\alpha\beta}(\omega) = \frac{e^2 g_s\,(1-e^{-\beta\hbar\omega})}{\omega\,\Omega}
\int_0^\infty dt\; e^{i(\omega+i\eta)t}\,
\operatorname{Tr}\big[F\,\hat v_\alpha(t)\,(1-F)\,\hat v_\beta\big],
\qquad F = f_F(\hat H),
$$

with $\hat v_\alpha(t) = e^{i\hat Ht}\hat v_\alpha e^{-i\hat Ht}$; the factor
$1-e^{-\beta\hbar\omega}$ is the KMS condition. In a non-orthogonal basis
every operator is taken in the dual basis, so $F = f_F(S^{-1}H)$. The `ac`
task propagates $f_F$-weighted random states with Chebyshev time evolution —
`nt` steps sampled at the Nyquist step of the spectral window, alias-free —
and Fourier transforms the correlator. The imaginary part follows by a
Kramers–Kronig (Hilbert) transform, and from the complex $\sigma(\omega)$:

$$
\varepsilon(\omega) = 1 + \frac{i\sigma(\omega)}{\varepsilon_0\omega},
\qquad n + i\kappa = \sqrt{\varepsilon},
\qquad \alpha(\omega) = \frac{2\omega\kappa}{c}.
$$

## Diffusion

The mean-square displacement of a wave packet at energy $E$,

$$
\Delta X^2(E,t) = \frac{\operatorname{Tr}\big[\delta(E-\hat H)\,
(\hat X(t)-\hat X(0))^2\big]}{\operatorname{Tr}\,\delta(E-\hat H)},
\qquad
D(E,t) = \frac12\,\frac{\partial\,\Delta X^2}{\partial t},
$$

distinguishes transport regimes at a glance: ballistic motion gives
$\Delta X^2 \propto t^2$ and $D \propto t$, diffusive motion a plateau in $D$
that defines the semiclassical conductivity $\sigma = e^2\rho(E)D$, and
localization a $D$ that falls back to zero. RSATB evaluates it through the
commutator velocity correlator, which is valid in any basis and under periodic
boundaries (`msd`, Gamma point only). Two kernels compute the same trace:
`stepwise` re-expands the energy resolution at each output time, `spectral`
runs one segmented Chebyshev chain at a cost independent of the output grid.

## Quantum metric

The longitudinal quantum metric is a frequency integral of the optical
absorption,

$$
G_{xx} \propto \int_{\omega_{\min}}^{\infty} \frac{\operatorname{Re}\sigma_{xx}(\omega)}{\omega}\,d\omega ,
$$

so the `quantum_metric` task runs the AC correlator and integrates it.
`omega_min_ev` excludes the intraband (Drude) region at small frequency.
