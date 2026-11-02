---
description: Tutorial 2 — diamond from a precomputed DFT Hamiltonian and its numerical atomic orbitals; how RSATB reads the basis, and why such models cost more.
---

<p class="rs-kicker">Tutorial 2 · The basics</p>

# A DFT Hamiltonian

<div class="rs-lead" markdown>
This tutorial computes the density of states of diamond from a DFT
Hamiltonian in a numerical atomic-orbital basis, at k-points along a path
through the Brillouin zone.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--beginner">Beginner</span>
  <span class="rs-chip">Model: lcao</span>
</div>

<div class="rs-learn" markdown>
**You will learn** how to point RSATB at a Hamiltonian file and its orbital
files, how to sample several k-points, and what is different about a DFT
basis.
</div>

## The input

<!-- rsatb-input: examples/02_lcao_diamond.json -->

Compared with Tutorial 1, two things are new:

- **`model` points at files** instead of listing parameters: the Hamiltonian
  `hamiltonians.h5` and one orbital file, `C_gga_7au_100Ry_2s2p1d.orb`.
  The number of orbitals per atom and their cutoff radius are read from the
  orbital file.
- **`kpoints` is a path** from the L point to $\Gamma$ with 11 points. The
  result has one density-of-states row per k-point.

## Run it

```bash
rsatb run examples/02_lcao_diamond.json
```

```text
  model         lcao
  capabilities  non-orthogonal, velocity, position
  k-points      11 (line through 2 vertices)
  basis         C · 13 orbitals · cutoff 3.704 Å
  bounds        [-8.621902, 263.989131] eV
```

The report shows:

- **13 orbitals per carbon atom, with a cutoff radius of 3.7 Å,** read from
  the orbital file.
- **`velocity` is available**, so this model can also compute conductivity
  and optical spectra ([Tutorial 4](transport.md) and
  [Tutorial 5](optics.md)).
- **The energy range is about 270 eV wide**, although the valence and
  low-lying conduction bands span about 30 eV. The upper part comes from the
  small eigenvalues of the overlap matrix. Because the cost of a given
  resolution is proportional to this range, the input uses `dE = 2.8` eV.

!!! tip "Other DFT codes"
    The same model reads ABACUS and NextHam output; set
    `hamiltonian.format` accordingly. [Tutorial 7](abacus-nextham.md) does this for a
    metal and a spin-orbit semiconductor.

## Summary

- A DFT model consists of **a Hamiltonian file and the orbital files** it
  was computed in.
- The basis is read from those files.
- A DFT basis has more orbitals per atom and a wider energy range than a
  $p_z$ model, so each step costs more; use the coarsest resolution the
  analysis allows.

??? deep "Overlapping orbitals"
    Atomic orbitals on neighbouring atoms overlap, so the basis is
    *non-orthogonal*: the overlap matrix $S$ is not the identity. RSATB
    computes $S$ from the orbital files and applies either $S^{-1} H$ or
    $S^{-1/2} H S^{-1/2}$. `run.preconditioner.operator_form` chooses which:

    | | `"sandwich"` (this input) | `"solve"` (default) |
    |---|---|---|
    | operator | $Z H Z$ with $Z = S^{-1/2}$ | $S^{-1} H$, by an iterative solve |
    | keeps $S$ in memory | no | yes |
    | use when | $Z$ fits in memory | $Z$ is too large to store |
    | `maxiter` | 20, so that $Z$ is refined | 0 (default): an approximate $Z$ preconditions the solve |

    The small eigenvalues of $S$ widen the energy range to 270 eV. The
    method is described in [Non-orthogonal bases](../theory/overlap.md).

## Next steps

- Replace the path with `{"mesh": [4, 4, 4]}` for a zone-averaged DOS.
- [Tutorial 3](other-models.md) covers the two remaining model types.
