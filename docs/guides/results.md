---
description: Load RSATB output in Python — npz archives, summary.json, normalizing the DOS, conductivity components, and reassembling distributed real-space grids.
---

<p class="rs-kicker">Guide</p>

# Read and plot results

<div class="rs-lead" markdown>
Everything a run writes is a NumPy archive or JSON. This page collects the
snippets for loading each kind; the full list of arrays is in
[Output files](../reference/outputs.md).
</div>

## Archives and the summary

```python
import json
import numpy as np

run = "results/graphene"
summary = json.load(open(f"{run}/summary.json"))
emin, emax = summary["setup"]["energy_bounds"]

for result in summary["results"]:
    print(result["label"], result["type"], result["seconds"], "s", result["scalars"])

dos = np.load(f"{run}/dos.npz")
energy = dos["energy"]
rho = dos["dos"][0] / (emax - emin)     # per-orbital DOS, integrates to 1
```

Scalars such as `emin` or a Fermi level live only in `summary.json`; arrays
live in the archive named by each result's `file`.

## Density of states

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(energy, rho, lw=1.2)
ax.set(xlabel="E (eV)", ylabel="DOS (states/eV/orbital)")
fig.tight_layout()
fig.savefig("dos.png", dpi=200)
```

To reference energies to the Fermi level, run a
[`fermi_level`](../reference/input/tasks/fermi_level.md) task in the same input
and read its `mu` from the summary.

## Conductivity tensors

Tensor components are stored in the order **(xx, xy, xz, yy, yz, zz)**:

```python
XX, XY, XZ, YY, YZ, ZZ = range(6)
dc = np.load(f"{run}/dc_spectrum.npz")
sigma_xx = dc["sigma"][0, :, XX]          # S/m, k-point 0
```

For a 2D material set `axis: "ab"` on the task, so that $\sigma$ is normalized
by the sheet area rather than a volume padded with vacuum.

## Optical spectra

```python
ac = np.load(f"{run}/ac.npz")
omega_ev = ac["omega_cm1"] / 8065.56
eps = ac["epsilon"][0, :, XX]              # complex dielectric function
alpha = ac["absorption_coefficient"][0, :, XX]   # 1/cm
```

## Real-space grids { #real-space-grids }

`ldos` and `density` are written one file per MPI process, each holding
some of the grid points and their indices. Put each file's points in place:

```python
import glob
import numpy as np

parts = [np.load(p) for p in sorted(glob.glob("results/ldos.rank*.npz"))]
shape = tuple(parts[0]["shape"])
energy = parts[0]["energy"]

ldos = np.zeros((len(energy), int(np.prod(shape))))
for part in parts:
    ldos[:, part["point_index"]] = part["ldos"]
ldos = ldos.reshape((-1,) + shape)          # (n_energy, n0, n1, n2)
```

For `density`, the array is `density` with no energy axis:

```python
rho = np.zeros(int(np.prod(shape)))
for part in parts:                      # parts = density.rank*.npz
    rho[part["point_index"]] = part["density"]
rho = rho.reshape(shape)
```

Grid axis $i$ runs along lattice vector $\mathbf a_i$, so `rho[:, :, k]` is a
slice parallel to the $\mathbf a_1\mathbf a_2$ plane.

## Mean-square displacement

```python
msd = np.load(f"{run}/msd.npz")
t, E = msd["time_fs"], msd["energy"]
D = msd["diffusion"][0]                 # (n_t, n_E), Å²/fs
i = np.argmin(abs(E - 1.0))             # energy closest to 1 eV
plt.plot(t, D[:, i] * 0.1)              # cm²/s
```
