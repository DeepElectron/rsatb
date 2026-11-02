---
description: Kernel polynomial method — spectral scaling, Chebyshev moments, the Jackson kernel, and how dE and n_moments set resolution and cost.
---

<p class="rs-kicker">Theory</p>

# Chebyshev expansions

<div class="rs-lead" markdown>
A smooth function of a matrix is a polynomial of that matrix, and Chebyshev
polynomials are the numerically stable choice. Every RSATB observable is
assembled from them.
</div>

## Scaling the spectrum

Chebyshev polynomials $T_n(x)=\cos(n\arccos x)$ live on $[-1,1]$, so the
operator is first mapped onto that interval:

$$
\tilde H = \frac{H - b}{a}, \qquad
a = \frac{E_{\max} - E_{\min}}{2}, \quad b = \frac{E_{\max} + E_{\min}}{2}.
$$

$[E_{\min}, E_{\max}]$ is the **spectral window** reported at setup. It is
estimated with a few PRIMME eigenvalue iterations (`run.bounds.mode = "auto"`)
or given directly (`"fixed"`). Every observable is defined on the open
interval: energies at or beyond its ends reconstruct to zero, so the window
must contain the whole spectrum.

## Moments and the recurrence

The density of states expands as

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

one sparse matrix–vector product per step. For a Hermitian operator the
product identity $2T_mT_n = T_{m+n}+T_{|m-n|}$ yields two moments per product,
halving the cost.

## Kernels and resolution

Truncating at $N$ moments causes Gibbs oscillations; the damping factors
$g_n$ — the **kernel** — suppress them. The **Jackson** kernel turns each
eigenvalue into a near-Gaussian of width

$$
\sigma \approx \frac{\pi a}{N},
$$

strictly positive and pointwise clean. That fixes the trade-off every
spectral task makes: **resolution costs moments, and moments cost
matrix–vector products in proportion to the window width.** The `dos` task
asks for the resolution directly — `dE` is the full width at half maximum, and
$N = \sqrt{8\ln 2}\,\pi a/\mathrm{dE}$ moments are derived from the window at
run time.

`costaper`, offered for conductivity spectra, trades pointwise positivity for a
faster-decaying bias, which wins when the spectrum is only consumed inside an
energy integral (as in `dc`).

!!! warning "A wide window is expensive"
    The window is set by the extreme eigenvalues, not by the energies you care
    about. LCAO bases with polarization orbitals place states hundreds of eV
    above the Fermi level — the diamond test basis spans about 270 eV — so a
    0.1 eV resolution there needs roughly twenty times the moments it would
    in a 13 eV window. Choose the coarsest resolution the physics allows.

## Time evolution

The same machinery propagates states in time. The Jacobi–Anger expansion

$$
e^{-i\tilde H\tau} = J_0(\tau) + 2\sum_{n\ge1}(-i)^n J_n(\tau)\,T_n(\tilde H)
$$

converges once $n$ exceeds $\tau$, so a step of length $\tau$ costs about
$\tau$ products. Correlation functions — the AC conductivity, the
mean-square displacement — are built from such propagations, and RSATB shares
the Chebyshev states across time steps where it can (`ac.method =
"concurrent"`, `msd.method = "spectral"`).

## Fermi functions

Occupations use the Fermi–Dirac function $f_F(E)$. Its sharpness sets the
degree: a pointwise-accurate expansion needs $N \sim a/k_BT$. The
`occupation` and `fermi_level` tasks instead expand the exact step with a
smooth high-order taper and choose the degree *a posteriori* from the sampled
spectral measure, which on extended systems is set by the smoothness of the
DOS near $\mu$ rather than by the temperature. The `density` task uses
$\sqrt{f_F}$ and its degree does grow as $1/T$: there the smearing is a
numerical knob as much as a physical one.
