---
description: Hutchinson–Girard stochastic trace estimation, why its relative error shrinks with system size, and how nsamples controls it.
---

<p class="rs-kicker">Theory</p>

# Random-state traces

<div class="rs-lead" markdown>
Computed column by column, the trace of an $N\times N$ matrix function costs
$N$ matrix–vector products. A random vector probes all columns at once.
</div>

## The estimator

Let $|\xi\rangle$ have independent random entries with zero mean and unit
variance, for example random phases $e^{i\phi_j}$. Then
$\mathbb E\big[\,|\xi\rangle\langle\xi|\,\big] = I$, and for any matrix $A$

$$
\operatorname{Tr} A = \mathbb E\,\langle\xi|A|\xi\rangle
\approx \frac{1}{R}\sum_{r=1}^{R}\langle\xi_r|A|\xi_r\rangle .
$$

Each sample needs only the product $A|\xi\rangle$, which a
[Chebyshev recurrence](chebyshev.md) computes with sparse matrix–vector
products. `run.nsamples` is $R$.

## Dependence on system size

For a quantity that is spread over many states, such as a density of states
or an occupation, the variance of one sample grows like $N$, and so does the
trace. The *relative* error is therefore

$$
\frac{\delta\,\operatorname{Tr}A}{\operatorname{Tr}A} \sim \frac{1}{\sqrt{R\,N}} .
$$

Doubling the system doubles the cost of one sample but halves the number of
samples needed for the same relative accuracy, so at fixed accuracy the
total cost grows linearly with system size. A cell with millions of orbitals
needs a few random vectors for a smooth density of states; a two-atom cell
needs hundreds.

!!! tip "In practice"
    First converge the supercell, the k-points and the resolution. Then
    increase `nsamples` until the result changes by less than the accepted
    noise. Two runs with different seeds give an estimate of the error. See
    [Converge a calculation](../guides/convergence.md).

## Local quantities

A diagonal element or a real-space map, such as the LDOS at a point or the
electron density, is a local quantity. Its relative error does not decrease
with $N$. RSATB therefore uses estimators with reduced variance for the
real-space tasks:

- **`ldos`** uses a survey-sampling estimator. Each random vector also gives
  a Chebyshev sketch of the energy window, and the map is the part of the
  window that the sketch captures, divided by the measured capture fraction.
  The relative error is $\sqrt{2(1-c)/r}$, with capture fraction $c$ and
  sketch rank $r = J\times n_{\text{samples}}$. It does not grow with system
  size, and it is zero once the rank reaches the number of states in the
  window.
- **`density`** can use a fragment correction. A deterministic calculation on
  local fragments gives most of the map, and the random vectors correct the
  remainder. In gapped systems the noise decreases exponentially with
  `fragment_buffer`. For metals, `energy_windows` divides the occupation
  into windows that are sampled separately.

Both tasks report error estimates with the result (`capture`,
`predicted_rel_error`, `correction_rms`); they are listed in
[Output files](../reference/outputs.md).

## Reproducibility

The random vectors are generated from a fixed seed and tied to the atoms,
not to the MPI processes. A run with the same input therefore gives the same
result with any number of processes, apart from rounding differences in the
sums.
