---
description: The four knobs that decide whether an RSATB result is converged — system size, resolution, samples, and the spectral window — and a checklist for publishable numbers.
---

<p class="rs-kicker">How-to guide</p>

# Converge a calculation

<div class="rs-lead" markdown>
A stochastic Chebyshev result has four independent sources of error. Converge
them in this order, changing one at a time.
</div>

| Error | Controlled by | Symptom when too small |
|---|---|---|
| **Finite size** | `structure.supercell`, `run.kpoints` | a comb of peaks instead of bands; quantities change with tiling |
| **Resolution** | `dos.dE`, `n_moments`, `n_kpm`, `nt`, `eta` | features too smooth; band edges and gaps blurred |
| **Sampling noise** | `run.nsamples`, `ldos.n_samples` | wiggles that change with the seed; negative absorption |
| **Spectral window** | `run.bounds` | spectrum clipped at the edges; NaNs or a blow-up |

## 1 · Size

A single k-point of a finite cell has a discrete spectrum. Tile the cell until
its levels are closer than your resolution: the level spacing is roughly the
bandwidth divided by the number of orbitals. Gamma-only tasks (`msd`, `ldos`,
`density`) need this most; k-resolved tasks can trade tiling for a k-mesh.

!!! example "Rule of thumb"
    An $8\times8\times8$ supercell at Gamma samples the same states as an
    $8\times8\times8$ k-mesh of the primitive cell. Large systems are cheap
    in RSATB — prefer tiling.

## 2 · Resolution

Each spectral task exposes its resolution: `dos.dE` directly, the others as a
moment count ($\delta E \approx \pi(E_{\max}-E_{\min})/(2N)$). Halve the width
until the feature you report stops changing. Cost is linear in the number of
moments and in the width of the window — see
[Chebyshev expansions](../theory/chebyshev.md#kernels-and-resolution).

## 3 · Samples

Raise `nsamples` until two runs with different seeds agree within the
tolerance you need. The noise of a trace falls as
$1/\sqrt{n_{\text{samples}}\,N_{\text{orbital}}}$, so large systems need few
samples:

| System | Typical `nsamples` for a smooth DOS |
|---|---|
| tens of atoms | 100–500 |
| thousands of atoms | 8–32 |
| 10⁵ atoms and up | 1–4 |

Real-space maps (`ldos`, `density`) do not self-average; use their own
`n_samples` and read the reported error diagnostics (`predicted_rel_error`,
`correction_rms`).

## 4 · Window

`run.bounds.mode = "auto"` estimates the extreme eigenvalues with PRIMME and
pads them. If you know the spectrum, `"fixed"` with
`"energy": [emin, emax]` skips the estimate — faster and more robust on very
large cells — but the window **must** contain every eigenvalue; anything
outside it is aliased into nonsense.

## Checklist for a reported number

- [x] Result unchanged (within its error bar) when the supercell grows by one step
- [x] Result unchanged when the resolution doubles
- [x] Two seeds agree within the stated noise
- [x] `summary.json` archived with the result — it holds the resolved input,
      the window and the version
- [x] For a non-orthogonal model, `preconditioner.tol` left at its default
      or tightened, never loosened, for production numbers
