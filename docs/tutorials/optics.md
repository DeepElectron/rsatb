---
description: Tutorial 5 — density of states, optical conductivity and dielectric function of a 128-atom diamond cell from one DFT-based input.
---

<p class="rs-kicker">Tutorial 5 · Observables</p>

# Optical spectra

<div class="rs-lead" markdown>
This tutorial computes the optical conductivity, the dielectric function
and the density of states of a 128-atom diamond supercell from one input.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--intermediate">Intermediate</span>
  <span class="rs-chip">Model: lcao</span>
</div>

<div class="rs-learn" markdown>
**You will learn** to compute an optical spectrum, to sample a crystal with a
supercell instead of many k-points, and to judge whether a result is
converged.
</div>

## The input

<!-- rsatb-input: examples/tutorials/diamond_optics.json -->

New in this input:

- **A supercell instead of k-points.** The two-atom cell is repeated
  4 × 4 × 4 (128 atoms) and computed at a single k-point; the supercell
  samples the Brillouin zone.
- **The `ac` task** computes the optical conductivity. `mu` is the chemical
  potential, here a value inside the band gap of diamond, and `temp` the
  temperature.

## Run it

```bash
rsatb run examples/tutorials/diamond_optics.json
```

<figure markdown>
![DOS, optical conductivity and dielectric function of diamond](../assets/figures/tut_diamond_optics.png)
<figcaption><strong>Left:</strong> the density of states around the chemical
potential (dashed); the band gap is visible.
<strong>Centre:</strong> the optical conductivity. <strong>Right:</strong>
the dielectric function derived from it.</figcaption>
</figure>

## Checks

- **The absorption starts near 5.4 eV and peaks at 11.5 eV.** The onset is
  at the direct gap of this Hamiltonian; the maximum is close to the main
  absorption peak of diamond, near 12 eV.
- **Cubic symmetry.** The $xx$, $yy$ and $zz$ components agree within 6% at
  the peak, and the $xy$ component is below 1% of them.

!!! warning "Tutorial-sized settings"
    The input uses 16 random vectors and a coarse frequency grid (0.68 eV
    steps), so the curves are noisy, and the absorption is slightly negative
    in places. For publishable spectra, use
    `examples/07_lcao_diamond_dos_ac.json` (128 vectors, twice the
    resolution), or `examples/08_lcao_diamond888_dos_ac.json` on 1,024 atoms.
    Both are sized for a compute node.

## Summary

- The DOS and the optical spectrum share one setup.
- A supercell at one k-point samples the Brillouin zone; larger cells give
  smoother spectra.
- Convergence is checked by comparing with a run with more samples, finer
  resolution or a larger cell.

??? deep "Discrete levels in the DOS"
    A 128-atom cell at one k-point has a discrete set of energy levels, so the
    raw density of states (grey) is a comb of sharp lines. The blue curve is
    the same data with 0.6 eV of Gaussian smoothing, for display only. On
    this cell the gap runs from about 13.3 to 17.8 eV.

??? deep "What sets the frequency resolution"
    The `ac` task samples the electrons' response at `nt` time steps spread
    over the whole energy range of the model, 0 to about 136 eV here. The
    range is wide because of the overlap of the DFT basis, so 200 steps give a
    spacing of 0.68 eV. Doubling `nt` halves the spacing and doubles the
    cost.

??? deep "Reading the arrays"
    The arrays are complex, with six tensor components on the last axis in the
    order (xx, xy, xz, yy, yz, zz):

    ```python
    import numpy as np

    ac = np.load("examples/tutorials/results/diamond_optics/ac.npz")
    XX = 0
    omega_ev = ac["omega_cm1"] / 8065.56          # frequency, eV
    sigma = ac["sigma"][0, :, XX]                  # complex, S/m
    eps = ac["epsilon"][0, :, XX]                  # dielectric function
    alpha = ac["absorption_coefficient"][0, :, XX] # 1/cm
    ```

    The theory is in [Linear response](../theory/linear-response.md).

## Next steps

[Tutorial 6](realspace.md) computes the electron density and the local
density of states on a grid.
