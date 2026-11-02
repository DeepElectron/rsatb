---
description: Write, validate and run a first RSATB calculation — graphene from a template — and load the result in Python.
---

<p class="rs-kicker">Getting started</p>

# Quick start

<div class="rs-lead" markdown>
A calculation is one JSON file and one command. This page writes a graphene
input from a template, validates it, runs it, and loads the density of states
it produced. It needs no data files and finishes in seconds.
</div>

## 1 · Start from a template

```bash
rsatb template sktb -o graphene.json
```

`rsatb template --list` shows the other starting points (`lcao`, `dftb`,
`tb_file`, `sktb-transport`, `minimal`). The `sktb` template is graphene
written inline: two carbon atoms, one $p_z$ orbital each, and a single
nearest-neighbour hopping $t = -2.7$ eV:

<!-- rsatb-template: sktb as graphene.json -->

## 2 · Check it

```console
$ rsatb check graphene.json
structure     2 atoms, inline
model         sktb  C
capabilities  orthogonal, velocity, position
k-points      1 (written out)
bounds        estimated at setup
parallel      npar=1, nsamples=64
tasks         energy_bounds, dos

graphene.json: valid
```

`check` needs neither MPI nor the compiled solver. It catches malformed
fields, missing files, and sections that disagree — a conductivity task on a
model with no velocity operator, say — before anything is queued. Try
misspelling a key to see what an error looks like:

```text
error: run.kpoint: unknown key 'kpoint'
    hint: did you mean 'kpoints'?
```

## 3 · Run it

```console
$ rsatb run graphene.json
rsatb 0.1.0

  structure     2 atoms, inline
  model         sktb  C
  capabilities  orthogonal, velocity, position
  ...
── setup ─────────────────────────────────────────────────────

  basis         C · 1 orbitals · cutoff 0.800 Å
  bounds        [-8.600000, 8.600000] eV

── tasks ─────────────────────────────────────────────────────

  [1/2] energy_bounds
        ✓ 0.0s
  [2/2] dos
        ✓ 0.0s
  wrote results/dos.npz

── results ───────────────────────────────────────────────────

  wrote results/summary.json
  energy_bounds: emin=-8.6, emax=8.6

done in 5.5s
```

The window is an analytic check. Each carbon has three neighbours, so the
Gamma-point eigenvalues are $\pm 3|t| = \pm 8.1$ eV, and RSATB pads the
estimated spectrum by 0.5 eV on each side.

The same input runs in parallel unchanged:

```bash
mpirun -n 4 rsatb run graphene.json
```

## 4 · Read the result

```python
import json
import numpy as np

dos = np.load("results/dos.npz")
energy, rho = dos["energy"], dos["dos"][0]       # [0]: the first (only) k-point

summary = json.load(open("results/summary.json"))
print(summary["setup"]["energy_bounds"])          # [-8.6, 8.6]
```

Every k-resolved array has a leading k-point axis. What each task writes is
listed in [Output files](../reference/outputs.md).

## Where next

A two-atom cell at one k-point has two eigenvalues, so its "DOS" is two
peaks. The [first tutorial](../tutorials/graphene-dos.md) tiles the same model
to 32,768 atoms and recovers the full graphene density of states, Dirac point
and van Hove singularities included.

!!! tip "Discover the format from the command line"
    ```bash
    rsatb doc                 # overview and topic index
    rsatb doc tasks.dos       # every key of one task, with defaults
    rsatb show graphene.json  # the input with every default filled in
    ```
    The [Input reference](../reference/input/index.md) on this site is
    generated from the same schema.
