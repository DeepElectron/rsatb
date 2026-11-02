---
description: Tutorial — electron density and local density of states of diamond on a real-space grid; Fermi level, Gamma-only tasks, cell-size requirement and distributed per-rank output.
---

<p class="rs-kicker">Tutorial 5</p>

# Real-space maps

<div class="rs-lead" markdown>
The density of states says *which* energies electrons occupy; real-space maps
say *where*. This tutorial computes the electron density and the local density
of states of a 128-atom diamond cell on a 0.25 Å grid, and reassembles the
output that every MPI rank writes separately.
</div>

**You will learn:** `fermi_level` · `density` · `ldos` · the Gamma-only and
cell-size requirements · distributed output and how to reassemble it.

## The input

<!-- rsatb-input: examples/tutorials/diamond_realspace.json -->

- **The cell is tiled $4\times4\times4$.** Real-space tasks are Gamma-point
  only, so the supercell is what samples the Brillouin zone. It also has to
  satisfy a geometric requirement — see the warning below.
- **`fermi_level`** finds the chemical potential at which the stochastic
  electron count is 256: two valence pairs per carbon, with no spin factor.
- **`density`** evaluates the electron density at a given `mu` and smearing
  temperature on a grid of spacing 0.25 Å.
- **`ldos`** evaluates the local density of states at seven energies from 8 to
  20 eV with 0.5 eV broadening.

!!! warning "Every cell width must exceed twice the orbital cutoff"
    The grid contraction needs each periodic width of the cell to be larger
    than $2r_c$ — here $2 \times 3.70 = 7.41$ Å. A $3\times3\times3$ diamond
    cell is only 6.19 Å across and fails at run time:

    ```text
    RuntimeError: Real-space contraction needs every periodic cell width to exceed
    twice the orbital cutoff, but width along a is 6.189493 A against a cutoff of
    3.704241 A. Use a supercell at least 7.408481 A across.
    ```

    `rsatb check` does not catch this yet, so size the supercell first.

## Run it

```bash
rsatb run examples/tutorials/diamond_realspace.json
```

```text
  fermi_level: n_electrons=256, band_energy=1090.96, mu=13.5143, ...
  density: n_point=68921, filter_degree=6325, n_states=64, electrons_on_grid=511.902
  ldos: n_point=68921, filter_degree=745, n_states=32, capture_min=0.78327,
        sketch_rank_min=18, predicted_rel_error_max=0.0661694
```

The run took 19 minutes on one loaded rank. Most of it went to the density's
Chebyshev filter, whose degree (6,325) scales as the spectral span over
$k_BT$.

**The density integrates to the electron count.** `electrons_on_grid` is
511.9 against the exact $2\times256 = 512$: the grid quadrature check passes
to 0.02%.

**The Fermi level is in the gap.** In an insulator, every $\mu$ between the
band edges gives the same electron count, so `fermi_level` returns one
point in the gap — here 13.51 eV, near the valence-band top. The density
itself was computed at mid-gap, $\mu = 16.23$ eV; for an insulator at 1000 K
the choice makes no difference.

**The LDOS reports its own error.** `capture` is the fraction of each energy
window the random sketch captured; it is 1.0 — exact — at six of the seven
energies, because this small cell has fewer states in each window than the
sketch's rank. The predicted relative error is at most 6.6%, at 20 eV.

<figure markdown>
![Electron density and LDOS of diamond on a lattice plane](../assets/figures/tut_diamond_realspace.png)
<figcaption>A lattice plane of the 128-atom cell, drawn in Cartesian
coordinates. <strong>Left:</strong> the valence electron density, peaked on
the carbon sites of the plane. <strong>Right:</strong> the LDOS deep in the
valence band, inside the gap (an order of magnitude weaker; the residual
weight is the 0.5 eV broadening of the band edge) and in the conduction band,
whose states have a distinctly different spatial character.</figcaption>
</figure>

## Reassembling the output

Each MPI rank owns part of the grid and writes only that part:

```text
examples/tutorials/results/diamond_realspace/
├── density.rank00000.npz
├── ldos.rank00000.npz
└── summary.json
```

With more ranks there are more files. Each holds the global grid `shape`,
the `point_index` of its own points, and their values. The pieces are a
partition, so reassemble them by scattering:

```python
import glob
import numpy as np

run = "examples/tutorials/results/diamond_realspace"

parts = [np.load(p) for p in sorted(glob.glob(f"{run}/density.rank*.npz"))]
shape = tuple(parts[0]["shape"])                 # (41, 41, 41)
rho = np.zeros(int(np.prod(shape)))
for part in parts:
    rho[part["point_index"]] = part["density"]
rho = rho.reshape(shape)                         # electrons / Å³

parts = [np.load(p) for p in sorted(glob.glob(f"{run}/ldos.rank*.npz"))]
energy = parts[0]["energy"]
ldos = np.zeros((len(energy), int(np.prod(shape))))
for part in parts:
    ldos[:, part["point_index"]] = part["ldos"]
ldos = ldos.reshape((-1,) + shape)               # (n_energy, n0, n1, n2)
```

Grid point $(i, j, k)$ sits at fractional coordinates
$(i/n_0, j/n_1, k/n_2)$ of the supercell, so `rho[:, :, 0]` is the lattice
plane spanned by $\mathbf a_1$ and $\mathbf a_2$ — the slice in the figure.

!!! tip "Why not gather to rank 0?"
    At production sizes no rank can hold the whole grid: a 0.25 Å grid of a
    million-atom cell has billions of points. Writing per rank, with no
    collective, is what keeps the output step scalable.

## Try next

- Add `"fragment_buffer": 3.0` to the density task. A deterministic fragment
  calculation then carries most of the map, and the pointwise noise drops at
  the same sweep cost; the result reports `correction_rms`.
- Raise `ldos.n_samples` and watch `predicted_rel_error` fall as
  $1/\sqrt{n}$.
