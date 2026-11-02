---
description: Tutorial — the density of states of graphene from a Slater-Koster model, tiled to 32,768 atoms and validated against the exact result.
---

<p class="rs-kicker">Tutorial 1</p>

# Graphene density of states

<div class="rs-lead" markdown>
The simplest model with real physics in it: one $p_z$ orbital per carbon and
a nearest-neighbour hopping. You will write it inline, check it against an
analytic answer, scale it to 32,768 atoms, and see what the resolution and the
number of random vectors control.
</div>

**You will learn:** the `sktb` model · supercells · `energy_bounds` · the
`dos` task and its resolution · labelling two runs of one task.

## The model

Graphene's $\pi$ bands come from one $p_z$ orbital per atom coupled to its
three neighbours by $t = V_{pp\pi} = -2.7$ eV:

$$
E_\pm(\mathbf k) = \pm |t|\,\Big|\sum_{j=1}^{3} e^{i\mathbf k\cdot\boldsymbol\delta_j}\Big| .
$$

At the Gamma point the sum is 3, so a two-atom cell has exactly two levels,
$\pm 3|t| = \pm 8.1$ eV. That is our first check.

## Step 1 — two atoms

<!-- rsatb-input: examples/01_sktb_graphene.json -->

Reading it top to bottom:

- **`structure`** is written inline. The second carbon sits at fractional
  $(-\tfrac13, \tfrac23)$, outside the cell, which is fine: coordinates are
  used exactly as given and periodic images are built from them. `pbc` is
  false along $c$, the vacuum direction.
- **`model`** *is* the parameters: a basis (`"C": ["pz"]`), an onsite energy,
  and one radial function for the `pp_pi` channel of the `C-C` pair —
  constant out to 1.6 Å, so first neighbours only. There is no `overlap`, so
  the model is orthogonal.
- **`run`** asks for 64 random vectors at Gamma.
- **`tasks`**: `energy_bounds` reports the spectral window; `dos` computes the
  density of states at 0.1 eV resolution.

```bash
rsatb run examples/01_sktb_graphene.json
```

```text
  basis         C · 1 orbitals · cutoff 0.800 Å
  bounds        [-8.600000, 8.600000] eV
```

The derived basis reports one orbital per carbon and a cutoff of 0.8 Å per
atom — half the 1.6 Å pair cutoff, since an edge exists when
$d \le r_i + r_j$. The window is $\pm 8.1$ eV padded by 0.5 eV: the analytic
check passes.

## Step 2 — a real system

Two levels are not a density of states. Tiling the cell $128\times128$ at
Gamma samples exactly the same states as a $128\times128$ k-mesh of the
primitive cell — 32,768 atoms. One line changes, `supercell`, and we ask for
two resolutions from the same setup:

<!-- rsatb-input: examples/tutorials/graphene_dos.json -->

Two tasks of the same type need a `label` each; their results are written to
`coarse.npz` and `fine.npz`.

```bash
rsatb run examples/tutorials/graphene_dos.json
```

<figure markdown>
![Graphene DOS from RSATB against the exact result at two resolutions](../assets/figures/tut_graphene_dos.png)
<figcaption><strong>Left:</strong> at 0.1 eV the 32,768-atom cell gives the
familiar graphene DOS — linear around the Dirac point, logarithmic van Hove
peaks at $\pm|t|$. <strong>Right:</strong> at 0.02 eV the resolution is finer
than the level spacing of the finite cell, and its discrete levels appear. In
both panels RSATB (teal, 8 random vectors) lies on the exact DOS of the same
k-points broadened to the same resolution (amber).</figcaption>
</figure>

Two lessons are in this figure:

1. **The method is exact up to noise.** With eight random vectors on 32,768
   orbitals the stochastic error is already below the line width. Large
   systems need *fewer* samples, because the trace self-averages —
   see [Random-state traces](../theory/stochastic-trace.md).
2. **Resolution must match system size.** A finite cell has a discrete
   spectrum. Asking for a resolution finer than its level spacing shows the
   levels, not the bulk DOS. Converge the supercell *and* the resolution
   together — see [Converge a calculation](../guides/convergence.md).

## Reading the result

```python
import json
import numpy as np

run = "examples/tutorials/results/graphene_dos"
summary = json.load(open(f"{run}/summary.json"))
emin, emax = summary["setup"]["energy_bounds"]

d = np.load(f"{run}/coarse.npz")
energy = d["energy"]
rho = d["dos"][0] / (emax - emin)     # states / eV / orbital; integrates to 1
```

The raw `dos` array integrates to the width of the window; dividing by it
gives the DOS per orbital. See [Output files](../reference/outputs.md#dos).

## Try next

- Halve `nsamples` to 4 and to 1 and watch the noise appear on the fine run.
- Add a second-neighbour hopping: another channel entry needs a longer
  `r_cut`, and the derived graph cutoff grows with it.
- Give `"bounds": {"mode": "fixed", "energy": [-8.7, 8.7]}` under `run` to skip
  the eigenvalue estimate.
