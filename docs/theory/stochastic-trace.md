---
description: Hutchinson–Girard stochastic trace estimation, why its relative error shrinks with system size, and how nsamples controls it.
---

<p class="rs-kicker">Theory</p>

# Random-state traces

<div class="rs-lead" markdown>
The trace of an $N\times N$ matrix function costs $N$ matrix–vector products
if computed column by column. A random vector probes all columns at once.
</div>

## The estimator

Let $|\xi\rangle$ have independent random entries with zero mean and unit
variance — random phases $e^{i\phi_j}$, say. Then
$\mathbb E\big[\,|\xi\rangle\langle\xi|\,\big] = I$, and for any matrix $A$

$$
\operatorname{Tr} A = \mathbb E\,\langle\xi|A|\xi\rangle
\approx \frac{1}{R}\sum_{r=1}^{R}\langle\xi_r|A|\xi_r\rangle .
$$

Each sample needs only the action $A|\xi\rangle$, which a
[Chebyshev recurrence](chebyshev.md) delivers with sparse matrix–vector
products. `run.nsamples` is $R$.

## Dependence on system size

For a matrix whose spectral weight is spread over many states — a density of
states, an occupation — the variance of one sample grows like $N$ while the
trace grows like $N$ as well, so the *relative* error is

$$
\frac{\delta\,\operatorname{Tr}A}{\operatorname{Tr}A} \sim \frac{1}{\sqrt{R\,N}} .
$$

Doubling the system doubles the cost of one sample but halves the number
of samples needed for the same relative accuracy, so the total cost at fixed
accuracy grows linearly with size. In practice a cell with millions of
orbitals needs a few random vectors for a smooth DOS, while a two-atom cell
needs hundreds.

!!! tip "In practice"
    Fix the physics first — supercell, k-points, resolution — then raise
    `nsamples` until the observable stops changing beyond the noise you can
    accept. Run two seeds and compare: the difference *is* the error bar.
    See [Converge a calculation](../guides/convergence.md).

## Local quantities

A diagonal element or a real-space map — the LDOS at a point, the electron
density — is a local quantity and does not self-average: its relative error
does not shrink with $N$. RSATB therefore uses variance-reduced estimators for
the real-space tasks:

- **`ldos`** uses a *survey-sampling* estimator. Each random probe also yields
  a flat-top Chebyshev sketch of the energy window; the map is the captured
  part of the window divided by its measured capture fraction. The relative
  error is $\sqrt{2(1-c)/r}$, with capture $c$ and sketch rank
  $r = J\times n_{\text{samples}}$ — bounded at every system size, and exact once
  the rank reaches the number of states in the window.
- **`density`** offers a *fragment control variate*: a deterministic fragment
  calculation carries most of the map and the random samples correct only the
  residual, so in gapped systems the noise falls exponentially with
  `fragment_buffer`. For metals, `energy_windows` splits the occupation into
  separately sampled windows instead.

Both report their own error diagnostics with the result — `capture`,
`predicted_rel_error`, `correction_rms` — listed under
[Output files](../reference/outputs.md).

## Reproducibility

A run repeats exactly at the same seed **and** the same MPI rank count. Each
rank draws its own random stream, so a different rank count draws different
states and gives a statistically equivalent result that differs by the
sampling error.
