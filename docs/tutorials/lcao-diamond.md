---
description: Tutorial 2 — diamond from a precomputed DFT Hamiltonian and its numerical atomic orbitals; how RSATB reads the basis, and why such models cost more.
---

<p class="rs-kicker">Tutorial 2 · The basics</p>

# A DFT Hamiltonian

<div class="rs-lead" markdown>
Most production Hamiltonians come from DFT codes, written in a basis of atomic
orbitals. This tutorial loads diamond from such a calculation and computes its
density of states along a path through the Brillouin zone.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--beginner">Beginner</span>
  <span class="rs-chip">&lt; 1 min</span>
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
  Everything about the basis — how many orbitals per atom, how far they
  reach — is read from the orbital file. You don't declare it.
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

Read the report line by line:

- **13 orbitals per carbon, reaching 3.7 Å** — read from the orbital file.
- **`velocity` is available**, so this model can also compute conductivity
  and optical spectra ([Tutorial 4](transport.md) and
  [Tutorial 5](optics.md)).
- **The energy range is about 270 eV wide**, although diamond's bands span
  only about 30 eV. This is typical of DFT bases, and it is why this input
  uses a coarse resolution (`dE = 2.8` eV) to stay fast.

!!! tip "Other DFT codes"
    The same input works for ABACUS and NextHam output — change
    `hamiltonian.format`. See [Import DFT Hamiltonians](../guides/hamiltonian-files.md).

## What to take away

- A DFT model is **a Hamiltonian file plus the orbital files** it was written
  in.
- RSATB **derives the basis** from those files; there is nothing to declare.
- DFT bases make **every step more expensive** than a hand-written model, so
  choose the coarsest resolution that answers your question.

??? deep "Under the hood: overlapping orbitals"
    Atomic orbitals on neighbouring atoms overlap, so the basis is
    *non-orthogonal*: the overlap matrix $S$ is not the identity. RSATB
    computes $S$ from the orbital files and works with the operator
    $S^{-1/2} H S^{-1/2}$, built once per k-point during setup.

    `run.preconditioner.operator_form` chooses how:

    | | `"sandwich"` (this input) | `"solve"` (default) |
    |---|---|---|
    | operator | $Z H Z$ with $Z = S^{-1/2}$ | $S^{-1} H$, by an iterative solve |
    | keeps $S$ in memory | no | yes |
    | best when | $Z$ fits in memory | $Z$ is too large to store |

    The small eigenvalues of $S$ are also what stretch the energy range to
    270 eV. Background: [Non-orthogonal bases](../theory/overlap.md).

??? deep "Under the hood: where the time goes"
    Every run ends with a profile table. For this input:

    ```text
      phase                                 wall(s)  peak(GB)       %    calls
      setup/spectral_bounds                    0.74      0.14   45.8%        1
      setup/isqrt_ns_ZH_S_Z                    0.24      0.14   14.9%       11
      setup/overlap_sample_k                   0.16      0.14    9.9%       11
      setup/hamiltonian_sample_k               0.13      0.14    8.0%       11
      setup/hamiltonian_read                   0.11      0.14    6.7%        1
    ```

    The `isqrt_*` rows build $S^{-1/2}$, once per k-point. On large cells this
    is where memory peaks — the first place to look if a run is slow.

## Try next

- Replace the path with `{"mesh": [4, 4, 4]}` for a zone-averaged DOS.
- Continue to [Tutorial 3](other-models.md) for the two remaining model types.
