---
description: Tutorial 3 — the dftb model from a directory of .skf files, and the tb_file model for an orthogonal tight-binding Hamiltonian.
---

<p class="rs-kicker">Tutorial 3 · The basics</p>

# Other model types

<div class="rs-lead" markdown>
This tutorial runs one example of each of the two remaining model types:
DFTB parameter files and tight-binding Hamiltonian files.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--beginner">Beginner</span>
  <span class="rs-chip">Models: dftb, tb_file</span>
</div>

## DFTB parameter files

A DFTB parameter set is a directory of `.skf` files, one per pair of
elements. `skf_dir` gives the directory, and `basis` the orbitals of each
element:

<!-- rsatb-input: examples/03_dftb_skf.json -->

```bash
rsatb run examples/03_dftb_skf.json
```

```text
  model         dftb  C
  capabilities  non-orthogonal
  k-points      8 (2x2x2 mesh)
  basis         C · 4 orbitals · cutoff 1.500 Å
```

`kpoints` here is a $2 \times 2 \times 2$ mesh, so the result has eight rows.

!!! note "The example parameter file is synthetic"
    The example includes a small synthetic `C-C.skf` file, so that it runs
    without downloads. For research, set `skf_dir` to a published parameter
    set such as mio or 3ob.

## A tight-binding Hamiltonian file

An orthogonal Hamiltonian $H(\mathbf R)$ produced by another program, for
example a DeePTB Slater-Koster export, is read with `tb_file`:

<!-- rsatb-input: examples/04_tb_file_chain.json -->

```bash
rsatb run examples/04_tb_file_chain.json
```

```text
  model         tb_file
  capabilities  orthogonal, velocity, position
  bounds        [-2.000000, 3.000000] eV
```

The file describes a 1D chain whose band runs from −1.5 to 2.5 eV.

For this model type the basis is declared in the input (`type_norb`,
`cutoff_radius`), because the files do not record it.

## Choosing a model type

| You have | Use |
|---|---|
| hopping parameters you can write down | `sktb` ([Tutorial 1](graphene-dos.md)) |
| output of a DFT code, plus orbital files | `lcao` ([Tutorial 2](lcao-diamond.md)) |
| a DFTB parameter set | `dftb` |
| an orthogonal $H(\mathbf R)$ file | `tb_file` |

??? deep "What each model can compute"
    DFTB parameters include an overlap but no position information, so a
    `dftb` model computes densities of states and populations, but not
    conductivities or optical spectra. A `tb_file` model is orthogonal, so
    the velocity operator is available and all transport tasks can be run.
    The full table is in [Choose a model](../guides/choose-model.md).

## Next steps

[Tutorial 4](transport.md) computes the conductivity and diffusion of
graphene.
