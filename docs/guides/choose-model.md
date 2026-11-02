---
description: Which model.type to use for Slater-Koster parameters, DFTB .skf sets, DFT Hamiltonians with numerical orbitals, or orthogonal H(R) files — and what each can compute.
---

<p class="rs-kicker">How-to guide</p>

# Choose a model

<div class="rs-lead" markdown>
The model family follows from what you have. It then decides which
observables are available, because some need operators — velocity, position,
real-space orbitals — that only some families can build.
</div>

## Start from your data

```mermaid
flowchart TD
  Q{What do you have?}
  Q -->|Slater-Koster parameters you can write down| SK[sktb]
  Q -->|a directory of DFTB .skf files| DF[dftb]
  Q -->|H&#40;R&#41; from a DFT or ML code + .orb files| LC[lcao]
  Q -->|an H&#40;R&#41; file already in an orthogonal basis| TB[tb_file]
```

| You have | `model.type` | Overlap | Basis comes from |
|---|---|---|---|
| Onsite energies and two-centre hoppings you write | [`sktb`](../reference/input/model/sktb.md) | only if you give `overlap` | `basis` you declare |
| DFTB parameters (mio, 3ob, …) | [`dftb`](../reference/input/model/dftb.md) | always | `basis` you declare |
| A DFT/ML Hamiltonian (DeePTB, ABACUS, NextHam) plus its numerical orbitals | [`lcao`](../reference/input/model/lcao.md) | always, computed from the `.orb` files | the `.orb` files |
| An orthogonal H(R) file, e.g. a DeePTB Slater-Koster export | [`tb_file`](../reference/input/model/tb_file.md) | never | `type_norb`, `cutoff_radius` you declare |

## What each can compute

| | DOS, PDOS, occupation | DC / AC / quantum metric | MSD | LDOS, density |
|---|:---:|:---:|:---:|:---:|
| `sktb`, orthogonal | ✓ | ✓ | ✓ | — |
| `sktb` with `overlap` | ✓ | — | — | — |
| `dftb` | ✓ | — | — | — |
| `lcao` | ✓ | ✓ | ✓ | ✓ |
| `tb_file` | ✓ | ✓ | ✓ | — |

Why the gaps:

- **Velocity** is the commutator $i[H, r]$. Orthogonal models get the position
  operator from the atomic coordinates; `lcao` computes the dipole integrals
  from its orbitals. A non-orthogonal `sktb` or `dftb` model has neither, so
  no transport.
- **Real-space maps** evaluate orbitals on a grid, so they need radial
  functions — only `.orb` files provide them.
- **MSD and the real-space maps** are Gamma-point observables: put the
  periodicity into a supercell rather than a k-mesh.

`rsatb check` reports every one of these before a run starts:

```text
error: tasks[0]: 'ac' needs the velocity operator, which the 'sktb' model with these inputs does not supply
error: tasks[1]: 'ldos' needs the radial orbital files, which only an LCAO model carries
error: tasks[1]: 'ldos' currently requires a single Gamma point, but run.kpoints has 4
```

## Orthogonal or not?

An orthogonal model is the fast path: each Chebyshev step is one sparse
product with $H$. A non-orthogonal model also needs $S^{-1/2}$ (or a solve
with $S$) at each step and a wider spectral window, typically costing several
times more per step — the price of a basis that came straight from DFT. See
[Non-orthogonal bases](../theory/overlap.md).

## Spin

`spin` on any model is one of `none`, `collinear`, `noncollinear` or
`spinorbit`. `noncollinear` and `spinorbit` double the basis. For an `lcao`
model read from an nspin = 4 ABACUS or NextHam file, use `spinorbit`: see
[Import DFT Hamiltonians](hamiltonian-files.md#spin-orbit).
