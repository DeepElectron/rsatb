---
description: What every task writes to disk — array names, shapes, units — and the summary.json run record.
---

<p class="rs-kicker">Reference</p>

# Output files

<div class="rs-lead" markdown>
A run writes into `output.dir`: one archive per task, named after the task's
`label` (default: the task type), plus `summary.json`, the record of the run.
`output.prefix` is prepended to every name. Rank 0 writes everything except
the real-space grids of [`ldos`](#ldos) and [`density`](#density), which every
rank writes its own piece of.
</div>

```text
results/
├── summary.json          # resolved input, setup, every scalar, every array shape
├── dos.npz               # one archive per task, named for its label
├── dc_spectrum.npz
└── ldos.rank00000.npz    # real-space tasks: one file per MPI rank
```

`output.format` chooses `npz` (default, compressed NumPy archives), `json`
(arrays as nested lists) or `none` (summary only).

## Conventions

| | |
|---|---|
| Leading axis | Every k-resolved array starts with a k-point axis, in the order `run.kpoints` generates them — even with a single k-point. |
| Energies | eV, absolute (not shifted to the Fermi level). |
| Tensors | Six components of the symmetric conductivity tensor on the last axis, in the order **(xx, xy, xz, yy, yz, zz)**. |
| Spin | Transport quantities include both spins: a spin-degenerate basis is counted twice, a spinor basis already holds both. |
| Conductivity | Normalized by the task's `axis`: S/m for the cell volume (`"abc"`), S — a sheet conductance — for a lattice plane (`"ab"`, `"bc"`, `"ca"`), S·m for a single lattice vector. |

## `summary.json` { #summary }

```json
{
  "input":   { "...": "the resolved input, every default filled in" },
  "setup":   {"model": "SKTB", "type_symbols": ["C"], "type_norb": [1],
              "cutoff_radius": [0.85],
              "capabilities": {"overlap": false, "velocity": true, "position": true},
              "n_kpoint": 1, "n_rank": 1, "energy_bounds": [-9.0, 9.0]},
  "results": [
    {"label": "msd", "type": "msd", "seconds": 2.736, "scalars": {},
     "distributed": false,
     "arrays": {"msd": {"shape": [1, 41, 256], "dtype": "float64"}, "...": {}},
     "file": "graphene_msd.npz"}
  ]
}
```

The summary is enough to reproduce a run: `input` is the fully resolved input,
`setup.energy_bounds` is the spectral window every expansion was scaled to,
and each result lists its scalars, the shape of every array and the file that
holds them. Arrays are written as soon as their task finishes, so a job that
runs out of wall time keeps the tasks it completed.

## Spectrum and populations

### `energy_bounds` { #energy_bounds }

Scalars only: `emin`, `emax` (eV), the window the Chebyshev expansions use.

### `dos` { #dos }

| Array | Shape | Units |
|---|---|---|
| `energy` | `(n_E,)` | eV |
| `dos` | `(n_k, n_E)` | see below |

!!! note "Normalization"
    `dos` follows the amplitude convention of the Chebyshev reconstruction:
    its integral over energy equals the width of the spectral window,
    $\int \mathrm{dos}\,dE = E_{\max} - E_{\min}$. Divide by that width for a
    density of states per orbital (integrating to one), and multiply by the
    orbital count of the cell for states/eV of the whole cell:

    ```python
    emin, emax = summary["setup"]["energy_bounds"]
    rho_per_orbital = dos / (emax - emin)        # ∫ rho dE = 1
    ```

### `pdos` { #pdos }

| Array | Shape | Units |
|---|---|---|
| `energy` | `(n_E,)` | eV |
| `pdos` | `(n_k, n_group, n_E)` | as `dos` |
| `total` | `(n_k, n_E)` | as `dos`; equals `pdos.sum(axis=1)` to rounding |
| `group_type`, `group_slot` | `(n_group,)` | `orbital_slot` projection only |

Scalars: `n_group`, and `population` — `"lowdin"` under the sandwich operator
form, `"mulliken"` under the solve form (Mulliken populations may legitimately
be negative).

### `occupation` { #occupation }

Scalars: `n_electrons` (no spin factor), `band_energy` (eV), `mu` (eV),
`n_moments`, `indicator`, `indicator_fired`. `indicator_fired = false` means
the degree cap `n_max` was reached before the error indicator fell below its
threshold.

### `fermi_level` { #fermi_level }

The same scalars as `occupation`, evaluated at the chemical potential `mu`
whose electron count matches the requested `n_electrons`.

## Real-space electronic structure

### `ldos` { #ldos }

Written **one file per rank**: `<prefix><label>.rank00000.npz`, …. Each file
is self-describing and holds only the grid points that rank owns.

| Array | Shape | Meaning |
|---|---|---|
| `shape` | `(3,)` | grid dimensions `(n0, n1, n2)` |
| `energy` | `(n_E,)` | eV |
| `point_index` | `(n_owned,)` | flat global index of each owned point |
| `ldos` | `(n_E, n_owned)` | LDOS at those points |
| `capture` | `(n_E,)` | captured fraction of the energy window |
| `sketch_rank` | `(n_E,)` | effective sketch rank |
| `predicted_rel_error` | `(n_E,)` | $\sqrt{2(1-\text{capture})/\text{rank}}$ |

Scalars: `n_point` (global point count), `filter_degree`, `n_states`,
`sampler_residual`, `capture_min`, `sketch_rank_min`,
`predicted_rel_error_max`. To reassemble the grid, see
[Read and plot results](../guides/results.md#real-space-grids).

### `density` { #density }

Per-rank files like `ldos`, with `density` of shape `(n_owned,)` in place of
`ldos` and no energy axis.

Scalars: `n_point`, `filter_degree`, `n_states`, `sampler_residual` and
`electrons_on_grid` — the grid integral of the map, which approaches the
occupation sum as the grid is refined and is a free quadrature check. With the
fragment control variate on (`fragment_buffer > 0`) also `n_fragments`,
`correction_rms`, `control_beta`; with `energy_windows > 1`, `n_windows`.

## Electrical and optical response

### `dc` { #dc }

| Array | Shape | Units |
|---|---|---|
| `sigma` | `(n_k, 6)` | S/m |

### `dc_spectrum` { #dc_spectrum }

| Array | Shape | Units |
|---|---|---|
| `energy` | `(n_moments,)` | eV, on the Chebyshev nodes |
| `sigma` | `(n_k, n_moments, 6)` | S/m, at zero temperature |

### `ac` { #ac }

| Array | Shape | Units |
|---|---|---|
| `omega_cm1` | `(nt+1,)` | frequency, cm⁻¹ (divide by 8065.56 for eV) |
| `sigma` | `(n_k, nt+1, 6)` | complex S/m; the imaginary part is the Kramers–Kronig transform of the real part |
| `epsilon` | `(n_k, nt+1, 6)` | complex dielectric function, $1 + i\sigma/(\varepsilon_0\omega)$ |
| `refractive_index` | `(n_k, nt+1, 6)` | $n = \operatorname{Re}\sqrt{\varepsilon}$ |
| `extinction_coefficient` | `(n_k, nt+1, 6)` | $\kappa = \operatorname{Im}\sqrt{\varepsilon}$ |
| `absorption_coefficient` | `(n_k, nt+1, 6)` | $\alpha = 2\omega\kappa/c$, cm⁻¹ |

## Dynamics and quantum geometry

### `msd` { #msd }

| Array | Shape | Units |
|---|---|---|
| `time_fs` | `(n_steps+1,)` | fs |
| `energy` | `(n_kpm,)` | eV |
| `msd` | `(1, n_steps+1, n_kpm)` | $\Delta X^2(E,t)$, Å² |
| `diffusion` | `(1, n_steps+1, n_kpm)` | $D(E,t)=\tfrac12\,\partial_t\Delta X^2$, Å²/fs |

The k axis is always a singleton: MSD is a Gamma-point observable.
1 Å²/fs = 0.1 cm²/s.

### `quantum_metric` { #quantum_metric }

| Array | Shape | Units |
|---|---|---|
| `gxx` | `(n_k,)` | longitudinal quantum metric |
| `sigma_xx` | `(n_k, nt+1)` | the AC conductivity it is integrated from |
| `omega_ev` | `(nt+1,)` | eV |
