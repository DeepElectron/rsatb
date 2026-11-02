---
description: The four settings that determine whether an RSATB result is converged — system size, resolution, number of random vectors, and the spectral window.
---

<p class="rs-kicker">Guide</p>

# Converge a calculation

<div class="rs-lead" markdown>
A result has four independent sources of error. Converge them in this
order, changing one setting at a time.
</div>

| Error | Controlled by | Symptom when too small |
|---|---|---|
| **Finite size** | `structure.supercell`, `run.kpoints` | a comb of peaks instead of bands; quantities change with tiling |
| **Resolution** | `dos.dE`, `n_moments`, `n_kpm`, `nt`, `eta` | features too smooth; band edges and gaps blurred |
| **Sampling noise** | `run.nsamples`, `ldos.n_samples` | wiggles that change with the seed; negative absorption |
| **Spectral window** | `run.bounds` | spectrum cut off at the edges; invalid values |

## 1 · Size

A finite cell at one k-point has a discrete spectrum. Enlarge the cell
until its level spacing — roughly the bandwidth divided by the number of
orbitals — is below the resolution. An $8\times8\times8$ supercell at
$\Gamma$ samples the same states as an $8\times8\times8$ k-mesh of the
primitive cell. The $\Gamma$-only tasks (`msd`, `ldos`, `density`) require
the supercell form; other tasks can use either.

## 2 · Resolution

The `dos` task takes the resolution directly as `dE`; the others take a
number of moments $N$, with $\delta E \approx \pi(E_{\max}-E_{\min})/(2N)$.
Refine until the reported quantity no longer changes. The cost is
proportional to the number of moments — see
[Chebyshev expansions](../theory/chebyshev.md#kernels-and-resolution).

## 3 · Samples

Increase `nsamples` until two runs with different seeds agree within the
required tolerance. The relative error of a trace falls as
$1/\sqrt{n_{\text{samples}}\,N_{\text{orbital}}}$, so larger systems need
fewer random vectors: [Tutorial 1](../tutorials/graphene-dos.md) uses 8 for
32,768 orbitals, while a two-atom cell needs hundreds.

For the real-space maps (`ldos`, `density`) the error does not decrease with
system size; increase their own `n_samples` and check the reported error
estimates (`predicted_rel_error`, `correction_rms`).

## 4 · Window

`run.bounds.mode = "auto"` estimates the extreme eigenvalues during setup
and adds a margin. With `"fixed"` and `"energy": [emin, emax]` the estimate
is skipped; the window **must** then contain all eigenvalues, otherwise the
results are wrong.

## Checklist for a reported number

- [x] Result unchanged (within its error bar) when the supercell grows by one step
- [x] Result unchanged when the resolution doubles
- [x] Two seeds agree within the stated noise
- [x] `summary.json` archived with the result — it holds the resolved input,
      the window and the version
- [x] For a non-orthogonal model, `preconditioner.tol` at its default or
      tighter
