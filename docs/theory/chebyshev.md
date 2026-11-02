---
description: Kernel polynomial method — spectral scaling, Chebyshev moments, the Jackson kernel, and how dE and n_moments set resolution and cost.
---

<p class="rs-kicker">Theory</p>

# Chebyshev expansions

<div class="rs-lead" markdown>
A smooth function of a matrix can be approximated by a polynomial of that
matrix. RSATB uses Chebyshev polynomials, which are numerically stable, for
all its observables.
</div>

## Scaling the spectrum

Chebyshev polynomials $T_n(x)=\cos(n\arccos x)$ are defined on $[-1,1]$, so
the operator is first mapped onto that interval:

$$
\tilde H = \frac{H - b}{a}, \qquad
a = \frac{E_{\max} - E_{\min}}{2}, \quad b = \frac{E_{\max} + E_{\min}}{2}.
$$

$[E_{\min}, E_{\max}]$ is the **spectral window** reported at setup. It is
estimated with a few eigenvalue iterations (`run.bounds.mode = "auto"`) or
given in the input (`"fixed"`). Results at or beyond the ends of the window
are zero, so the window must contain the whole spectrum.

## Moments and the recurrence

The density of states is expanded as

$$
\rho(\tilde E) = \frac{1}{\pi\sqrt{1-\tilde E^2}}
\Big[g_0\mu_0 + 2\sum_{n\ge1} g_n\,\mu_n\,T_n(\tilde E)\Big],
\qquad
\mu_n = \operatorname{Tr}\,T_n(\tilde H) \approx \langle\xi|T_n(\tilde H)|\xi\rangle ,
$$

and the vectors $|\xi_n\rangle = T_n(\tilde H)|\xi\rangle$ follow from the
three-term recurrence

$$
|\xi_{n+1}\rangle = 2\tilde H|\xi_n\rangle - |\xi_{n-1}\rangle ,
$$

with one sparse matrix–vector product per step. For a Hermitian operator the
identity $2T_mT_n = T_{m+n}+T_{|m-n|}$ gives two moments per product, which
halves the cost.

## Kernels and resolution

Truncating the series at $N$ moments causes Gibbs oscillations. Damping
factors $g_n$, the **kernel**, suppress them. With the **Jackson** kernel each
eigenvalue becomes a nearly Gaussian peak of width

$$
\sigma \approx \frac{\pi a}{N},
$$

and the result is positive everywhere. A finer resolution therefore needs
more moments, and the number of moments for a given resolution is
proportional to the width of the window. The `dos` task takes the resolution
directly: `dE` is the full width at half maximum, and
$N = \sqrt{8\ln 2}\,\pi a/\mathrm{dE}$ moments are computed from the window
at run time.

The `costaper` kernel, available for conductivity spectra, is not positive
everywhere, but its bias decreases faster with $N$. It is the better choice
when the spectrum is only used inside an energy integral, as in `dc`.

!!! warning "A wide window is expensive"
    The window is set by the extreme eigenvalues, not by the energies of
    interest. LCAO bases with polarization orbitals have states hundreds of
    eV above the Fermi level; the diamond test basis spans about 270 eV. A
    0.1 eV resolution then needs about twenty times the moments it would
    need in a 13 eV window. Use the coarsest resolution that the analysis
    allows.

## Time evolution

The same method propagates states in time. The Jacobi–Anger expansion

$$
e^{-i\tilde H\tau} = J_0(\tau) + 2\sum_{n\ge1}(-i)^n J_n(\tau)\,T_n(\tilde H)
$$

converges once $n$ exceeds $\tau$, so a time step of length $\tau$ costs
about $\tau$ products. Correlation functions, such as the AC conductivity and
the mean-square displacement, are built from these propagations. RSATB
reuses Chebyshev states across time steps where possible
(`ac.method = "concurrent"`, `msd.method = "spectral"`).

## Fermi functions

Occupations use the Fermi–Dirac function $f_F(E)$. The sharper the function,
the higher the degree: an expansion that is accurate at every energy needs
$N \sim a/k_BT$. The `occupation` and `fermi_level` tasks instead expand the
step function with a smooth high-order taper and choose the degree from the
sampled spectrum after the moments are computed. For extended systems the
degree is then set by the smoothness of the DOS near $\mu$, not by the
temperature. The `density` task uses $\sqrt{f_F}$, and its degree grows as
$1/T$, so its temperature affects the cost as well as the result.
