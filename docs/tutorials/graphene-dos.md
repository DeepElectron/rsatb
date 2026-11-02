---
description: Tutorial 1 — the density of states of graphene from a hand-written Slater-Koster model, scaled to 32,768 atoms and checked against the exact result.
---

<p class="rs-kicker">Tutorial 1 · The basics</p>

# Graphene density of states

<div class="rs-lead" markdown>
Your first real calculation: write a small graphene model by hand, run it,
then make it 16,000 times bigger with a one-line change.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--beginner">Beginner</span>
  <span class="rs-chip">≈ 2 min</span>
  <span class="rs-chip">Model: sktb</span>
</div>

<div class="rs-learn" markdown>
**You will learn** how a model is written, how to make a large system from a
small cell, and how to set the energy resolution of a density of states.
</div>

## Step 1 — a two-atom cell

Graphene has two carbon atoms per cell, each with one $p_z$ orbital, joined to
three neighbours by a hopping of $-2.7$ eV. That is the whole model, and it
fits in one file:

<!-- rsatb-input: examples/01_sktb_graphene.json -->

Four sections, read top to bottom:

| Section | Here it says |
|---|---|
| `structure` | two carbon atoms in a hexagonal cell, periodic in-plane |
| `model` | one $p_z$ orbital per carbon, hopping $-2.7$ eV to neighbours within 1.6 Å |
| `run` | use 64 random vectors at the $\Gamma$ point |
| `tasks` | report the energy range, then compute the density of states |

Run it:

```bash
rsatb run examples/01_sktb_graphene.json
```

```text
  bounds        [-8.600000, 8.600000] eV
```

A quick sanity check: a two-atom cell at $\Gamma$ has exactly two energies,
$\pm 3 \times 2.7 = \pm 8.1$ eV, and RSATB pads the range by 0.5 eV — the check passes.

## Step 2 — a real system

Two energy levels are not yet a density of states. To get the real thing,
repeat the cell 128 × 128 times — 32,768 atoms — by adding one key to
`structure`:

```json
"supercell": [128, 128, 1]
```

We also ask for the density of states twice, at a coarse and a fine
resolution, from the same run:

```json
"tasks": [
  {"type": "dos", "label": "coarse", "dE": 0.1},
  {"type": "dos", "label": "fine",   "dE": 0.02}
]
```

??? example "The complete input — `examples/tutorials/graphene_dos.json`"
    <!-- rsatb-input: examples/tutorials/graphene_dos.json -->

```bash
rsatb run examples/tutorials/graphene_dos.json
```

<figure markdown>
![Graphene DOS from RSATB against the exact result at two resolutions](../assets/figures/tut_graphene_dos.png)
<figcaption><strong>RSATB (teal) on top of the exact answer (amber).</strong>
Left, at 0.1 eV: the familiar graphene density of states, with a V-shaped
Dirac point at zero and sharp van Hove peaks at ±2.7 eV. Right, at 0.02 eV:
the resolution is now finer than the gaps between the energy levels of this
finite cell, so individual levels show through.</figcaption>
</figure>

## What to take away

- **A bigger cell is one line.** `supercell` repeats the structure; the model
  needs no change.
- **`dE` sets the resolution.** Finer is more expensive, and only meaningful
  if the system is large enough to have levels that close together.
- **Large systems need few random vectors.** Eight were enough here — see
  below for why.

??? deep "Under the hood: why so few random vectors?"
    RSATB never diagonalizes the Hamiltonian. It estimates the density of
    states from random vectors, and the statistical error falls as
    $1/\sqrt{n_{\text{samples}} \times N_{\text{orbitals}}}$. With 32,768 orbitals,
    eight vectors already put the noise below the line width. Small cells need
    many more. The details are in
    [Random-state traces](../theory/stochastic-trace.md).

??? deep "Under the hood: normalizing the result"
    The `dos` array is scaled so that it integrates to the width of the
    energy range, $E_{\max} - E_{\min}$. Divide by that width to get states per eV
    per orbital:

    ```python
    import json
    import numpy as np

    run = "examples/tutorials/results/graphene_dos"
    summary = json.load(open(f"{run}/summary.json"))
    emin, emax = summary["setup"]["energy_bounds"]

    d = np.load(f"{run}/coarse.npz")
    energy, rho = d["energy"], d["dos"][0] / (emax - emin)
    ```

## Try next

- Set `nsamples` to 1 and watch noise appear on the fine curve.
- Continue to [Tutorial 2](lcao-diamond.md): the same workflow with a
  Hamiltonian from a DFT code.
