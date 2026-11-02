---
description: Tutorial — diamond from a precomputed DFT Hamiltonian and numerical atomic orbitals; the overlap, the S^-1/2 preconditioner, and why LCAO spectral windows are wide.
---

<p class="rs-kicker">Tutorial 2</p>

# A non-orthogonal LCAO model

<div class="rs-lead" markdown>
Production electronic structure comes from DFT codes in atomic-orbital bases,
and those bases overlap. This tutorial runs diamond from a precomputed
Hamiltonian and the numerical orbitals it was written in, and shows what
non-orthogonality changes.
</div>

**You will learn:** the `lcao` model · how the basis is derived from `.orb`
files · the overlap and the preconditioner · k-point paths · the profile
table.

## The input

<!-- rsatb-input: examples/02_lcao_diamond.json -->

- **`structure`** is a POSCAR of the two-atom primitive cell.
- **`model.hamiltonian`** is $H(\mathbf R)$ in DeePTB's HDF5 layout. ABACUS
  and NextHam files work the same way — see
  [Import DFT Hamiltonians](../guides/hamiltonian-files.md).
- **`model.orbitals`** lists one `.orb` file per species. They are required:
  the overlap $S(\mathbf R)$ is computed from them, and so are the orbital
  count and the graph cutoff. Nothing about the basis is declared.
- **`run.kpoints`** is a line from L $(\tfrac12,\tfrac12,\tfrac12)$ to
  $\Gamma$ with 11 points; the DOS comes back with one row per k-point.
- **`run.preconditioner.operator_form`** is `"sandwich"`: expansions apply
  $S^{-1/2}HS^{-1/2}$ and $S$ is released after setup.

## Run it

```bash
rsatb run examples/02_lcao_diamond.json
```

```text
  model         lcao
  capabilities  non-orthogonal, velocity, position
  k-points      11 (line through 2 vertices)
  ...
  basis         C · 13 orbitals · cutoff 3.704 Å
  bounds        [-8.621902, 263.989131] eV
```

Three things to notice.

**The basis was read, not declared.** 13 orbitals per carbon — the 2s2p1d
basis is $2\times1 + 2\times3 + 1\times5$ — and a 3.70 Å cutoff, both from
`C_gga_7au_100Ry_2s2p1d.orb`.

**Velocity is available.** An LCAO model computes the dipole integrals from
its orbitals, so the commutator velocity $i[H, r]$ exists, and with it every
transport task (Tutorial 3).

**The window is 270 eV wide.** The valence and conduction bands of diamond
span about 30 eV, but the small eigenvalues of the overlap push the highest
eigenvalues of $S^{-1}H$ to 264 eV. Every Chebyshev expansion is scaled to
this window, so resolution is expensive: this input uses `dE = 2.8` eV to
stay cheap. See [Non-orthogonal bases](../theory/overlap.md#conditioning).

## Where the time went

The run ends with a profile table:

```text
  phase                                 wall(s)  peak(GB)       %    calls
  setup/spectral_bounds                    0.74      0.14   45.8%        1
  setup/isqrt_ns_ZH_S_Z                    0.24      0.14   14.9%       11
  setup/overlap_sample_k                   0.16      0.14    9.9%       11
  setup/hamiltonian_sample_k               0.13      0.14    8.0%       11
  setup/hamiltonian_read                   0.11      0.14    6.7%        1
  setup/overlap_build                      0.09      0.14    5.4%        1
  setup/isqrt_newton_schulz                0.06      0.14    3.5%       11
  setup/atomic_graph                       0.04      0.14    2.6%        1
  setup/isqrt_seed                         0.04      0.14    2.3%       11
```

The `isqrt_*` rows are the $S^{-1/2}$ construction — a graph-local seed
refined by Newton–Schulz, once per k-point (11 calls). On large cells this is
the setup's memory peak; the table is where to look first when a run is
slow or runs out of memory.

## Choosing the operator form

| | `"sandwich"` | `"solve"` (default) |
|---|---|---|
| expansion operator | $ZHZ$, $Z=S^{-1/2}$ | $S^{-1}H$ via CG |
| keeps $S$ | no | yes |
| best when | $Z$ fits in memory | $Z$ is too dense to store |

For cells up to a few thousand atoms in a compact basis, `"sandwich"` is
usually fastest. The full trade-off is in
[Non-orthogonal bases](../theory/overlap.md#two-equivalent-operators).

## Try next

- Replace the line with `{"mesh": [4, 4, 4]}` for a Brillouin-zone-averaged
  DOS.
- Set `"operator_form": "solve"` and compare the `task/dos` time.
- Continue to [Tutorial 3](optics.md), which tiles this cell to 128 atoms and
  computes its optical response.
