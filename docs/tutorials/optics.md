---
description: Tutorial 5 — density of states, optical conductivity and dielectric function of a 128-atom diamond cell from one DFT-based input.
---

<p class="rs-kicker">Tutorial 5 · Observables</p>

# Optical spectra

<div class="rs-lead" markdown>
The optical conductivity and dielectric function of a 128-atom diamond
supercell, together with its density of states, from one input.
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

What's new:

- **A supercell instead of k-points.** The two-atom cell is repeated
  4 × 4 × 4 (128 atoms) and computed at a single k-point; the supercell
  samples the Brillouin zone.
- **The `ac` task** computes the optical conductivity. `mu` is the chemical
  potential — here a value inside diamond's band gap — and `temp` the
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

- **Absorption onset near 5.4 eV, maximum at 11.5 eV.** The onset is at the
  direct gap of this Hamiltonian; the maximum is close to the main absorption
  peak of diamond, near 12 eV.
- **Cubic symmetry.** The $xx$, $yy$ and $zz$ components agree within 6% at
  the peak, and the $xy$ component is below 1% of them.

!!! warning "This is a quick, tutorial-sized run"
    It uses 16 random vectors and a coarse frequency grid (0.68 eV steps), so
    the curves are noisy — the absorption even dips slightly below zero in
    places. For publishable spectra, use
    `examples/07_lcao_diamond_dos_ac.json` (128 vectors, twice the
    resolution), or `examples/08_lcao_diamond888_dos_ac.json` on 1,024 atoms.
    Both are sized for a compute node.

## What to take away

- The DOS and the optical spectrum share one setup.
- A supercell at one k-point samples the Brillouin zone; larger cells give
  smoother spectra.
- Convergence is checked by comparing with a run with more samples, finer
  resolution or a larger cell.

??? deep "Under the hood: the band picture in the DOS"
    A 128-atom cell at one k-point has a discrete set of energy levels, so the
    raw density of states (grey) is a comb of sharp lines. The blue curve is
    the same data with 0.6 eV of Gaussian smoothing, for display only. On
    this cell the gap runs from about 13.3 to 17.8 eV.

??? deep "Under the hood: what sets the frequency resolution"
    The `ac` task samples the electrons' response at `nt` time steps spread
    over the whole energy range of the model, 0 to about 136 eV here. That
    range is wide because of the DFT basis, not the physics, so 200 steps give
    0.68 eV spacing. Doubling `nt` halves it — at double the cost.

??? deep "Under the hood: reading the arrays"
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

## Try next

Continue to [Tutorial 6](realspace.md): where the electrons are in space.
