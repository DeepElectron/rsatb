---
description: Write, check and run a first RSATB calculation — graphene from a template — and load the result in Python.
---

<p class="rs-kicker">Get started</p>

# Quick start

<div class="rs-lead" markdown>
Your first calculation in four steps: create an input, check it, run it, and
plot the result. It needs no data files and finishes in seconds.
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

`check` catches typos, missing files and impossible requests before
anything runs — try misspelling a key to see the message it gives:

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

To use more processors, launch the same file with MPI — nothing in the input
changes:

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

This tiny cell has only two energy levels, so its "density of states" is two
peaks. [Tutorial 1](../tutorials/graphene-dos.md) makes it 32,768 atoms with
one line and recovers the real graphene spectrum.

!!! tip "Get help from the command line"
    `rsatb doc` lists every input option, `rsatb doc tasks.dos` explains one
    task, and `rsatb show graphene.json` prints your input with all defaults
    filled in.
