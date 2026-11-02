---
description: Tutorial 4 — DC conductivity and wave-packet diffusion of graphene from one input, checked against the analytic ballistic limit.
---

<p class="rs-kicker">Tutorial 4 · Observables</p>

# Transport

<div class="rs-lead" markdown>
How easily do electrons move, and how fast do they spread out? This tutorial
computes the conductivity and the diffusion of electrons in a 32,768-atom
graphene sheet, all from one input.
</div>

<div class="rs-chips">
  <span class="rs-chip rs-chip--intermediate">Intermediate</span>
  <span class="rs-chip">≈ 2 min</span>
  <span class="rs-chip">Model: sktb</span>
</div>

<div class="rs-learn" markdown>
**You will learn** to run several observables in one go, to compute a
conductivity spectrum and a diffusion coefficient, and how to treat a 2D
material correctly.
</div>

## The input

<!-- rsatb-input: examples/tutorials/graphene_transport.json -->

What's new compared with Tutorial 1:

- **Three tasks in one input.** They share a single setup, so asking for all
  three costs barely more than asking for one.
- **`dc_spectrum`** gives the conductivity at every energy.
  `"axis": "ab"` tells RSATB that this is a sheet: it divides by the area of
  the sheet, not by a volume padded with vacuum.
- **`msd`** follows electron wave packets for 100 fs and measures how far they
  spread along $x$.
- **`bounds` is fixed** at ±9 eV. When you know the energy range, giving it
  skips a setup step.

## Run it

```bash
rsatb run examples/tutorials/graphene_transport.json
```

<figure markdown>
![DOS, DC conductivity and diffusion of graphene](../assets/figures/tut_graphene_transport.png)
<figcaption><strong>Left:</strong> the density of states.
<strong>Centre:</strong> the conductivity of the sheet at each energy — zero
at the Dirac point and at the band edges. <strong>Right:</strong> the
diffusion coefficient grows steadily with time at every energy: electrons fly
freely through a perfect crystal.</figcaption>
</figure>

## Is it right? A check against theory

In a perfect crystal nothing scatters the electrons, so their spread grows
like $D(t) = \tfrac12 v_F^2\,t$, where $v_F$ is graphene's Fermi velocity:

$$
v_F = \frac{3 \times 2.7\ \text{eV} \times 1.42\ \text{Å}}{2\hbar} = 8.74\ \text{Å/fs}
\quad\Rightarrow\quad D(100\ \text{fs}) = 382\ \text{cm}^2/\text{s}.
$$

The curves for 0.5 and 1 eV reach exactly that.

## What to take away

- **List several tasks in one input** — they share the expensive setup.
- **Use `"axis": "ab"`** for 2D materials, so results are per sheet.
- **The diffusion curve tells you the transport regime** — straight line:
  ballistic; flat: diffusive; falling: localized.

??? deep "Under the hood: why the conductivity is so large"
    A perfect crystal has no resistance, so its DC conductivity is formally
    infinite. What the plot shows is limited only by the energy resolution
    (256 moments here). Physically meaningful DC values need scattering:
    disorder in the Hamiltonian, a cell large enough to hold it, and a
    resolution finer than the scattering rate. The diffusion curve reveals
    which regime you are in.

??? deep "Under the hood: reading the arrays"
    ```python
    import numpy as np

    run = "examples/tutorials/results/graphene_transport"
    XX = 0                                   # components: xx, xy, xz, yy, yz, zz

    dc = np.load(f"{run}/dc_spectrum.npz")
    energy, sigma_xx = dc["energy"], dc["sigma"][0, :, XX]   # siemens (sheet)

    msd = np.load(f"{run}/msd.npz")
    t, E = msd["time_fs"], msd["energy"]
    D = msd["diffusion"][0] * 0.1            # Å²/fs → cm²/s, shape (n_t, n_E)
    ```

    The theory behind both tasks is in
    [Linear response](../theory/linear-response.md).

## Try next

- Measure along $y$ with `"axis": "y"` in the `msd` task — graphene is
  isotropic, so the curves should match.
- Continue to [Tutorial 5](optics.md): optical spectra.
