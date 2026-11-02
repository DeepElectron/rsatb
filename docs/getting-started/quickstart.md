---
description: A first RSATB calculation, the density of states of graphene from a template, in four steps.
---

<p class="rs-kicker">Get started</p>

# Quick start

<div class="rs-lead" markdown>
This page runs a first calculation, the density of states of graphene,
from a template. No data files are needed.
</div>

## 1 · Create an input

```bash
rsatb template sktb -o graphene.json
```

??? example "What's inside `graphene.json`"
    <!-- rsatb-template: sktb as graphene.json -->

    A structure (two carbon atoms), a model (one orbital per atom, one
    hopping value), and two tasks: the energy range and the density of
    states. [How an input works](concepts.md) explains each part.

## 2 · Check it

```console
$ rsatb check graphene.json
graphene.json: valid
```

Errors, such as a misspelled key or a missing file, are reported here,
before the run.

## 3 · Run it

```bash
rsatb run graphene.json
```

In parallel, with MPI processes and OpenMP threads:

```bash
OMP_NUM_THREADS=4 mpirun -n 2 rsatb run graphene.json
```

How to choose the numbers is described in [Run on a cluster](../guides/parallel.md).

## 4 · The result

Each task writes a file to `results/`; the density of states is in
`results/dos.npz`:

```python
import numpy as np
import matplotlib.pyplot as plt

d = np.load("results/dos.npz")
plt.plot(d["energy"], d["dos"][0])
plt.show()
```

The two-atom cell has only two energy levels, so this shows two peaks.
[Tutorial 1](../tutorials/graphene-dos.md) uses a 32,768-atom supercell and
obtains the full graphene density of states.
