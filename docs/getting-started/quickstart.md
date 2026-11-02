---
description: Write, check and run a first RSATB calculation — graphene from a template — and load the result in Python.
---

<p class="rs-kicker">Get started</p>

# Quick start

<div class="rs-lead" markdown>
A graphene calculation in four steps: create the input from a template,
check it, run it, and load the result. No data files are needed.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--beginner">Beginner</span>
  <span class="rs-chip">≈ 5 min</span>
</div>

## 1 · Create an input

```bash
rsatb template sktb -o graphene.json
```

This writes a complete, working input for graphene:

??? example "What's inside `graphene.json`"
    <!-- rsatb-template: sktb as graphene.json -->

    A structure (two carbon atoms), a model (one orbital per atom, one
    hopping value), and two tasks: report the energy range, then compute the
    density of states. [How an input works](concepts.md) explains each part.

## 2 · Check it

```console
$ rsatb check graphene.json
  ...
graphene.json: valid
```

`check` reports unknown keys, missing files, and tasks the model cannot
provide. A misspelled key, for example, gives:

```text
error: run.kpoint: unknown key 'kpoint'
    hint: did you mean 'kpoints'?
```

## 3 · Run it

```bash
rsatb run graphene.json
```

```text
  [1/2] energy_bounds   ✓
  [2/2] dos             ✓
  wrote results/dos.npz
done in 5.5s
```

??? example "The full output"
    ```text
    rsatb 0.1.0

      structure     2 atoms, inline
      model         sktb  C
      capabilities  orthogonal, velocity, position
      k-points      1 (written out)

    ── setup ──────────────────────────────────────────

      basis         C · 1 orbitals · cutoff 0.800 Å
      bounds        [-8.600000, 8.600000] eV

    ── tasks ──────────────────────────────────────────

      [1/2] energy_bounds
            ✓ 0.0s
      [2/2] dos
            ✓ 0.0s
      wrote results/dos.npz

    ── results ────────────────────────────────────────

      wrote results/summary.json
      energy_bounds: emin=-8.6, emax=8.6

    done in 5.5s
    ```

    The run also prints a table of where the time went; it's useful once
    calculations get large.

For a parallel run, start it with MPI:

```bash
mpirun -n 4 rsatb run graphene.json
```

## 4 · Look at the result

```python
import numpy as np
import matplotlib.pyplot as plt

d = np.load("results/dos.npz")
plt.plot(d["energy"], d["dos"][0])     # [0]: the first (and only) k-point
plt.xlabel("Energy (eV)")
plt.show()
```

Every task writes one `.npz` file named after it, plus `summary.json`, a
record of the whole run.

## Next steps

A two-atom cell at one k-point has two energy levels, so this density of
states consists of two peaks. [Tutorial 1](../tutorials/graphene-dos.md)
uses a 128 × 128 supercell (32,768 atoms) and compares the result with the
exact density of states.

!!! tip "Help on the command line"
    `rsatb doc` lists the input options, `rsatb doc tasks.dos` describes one
    task, and `rsatb show graphene.json` prints the input with all defaults
    filled in.
