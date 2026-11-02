---
description: Tutorial — DC conductivity spectrum and mean-square displacement of graphene from one setup, with 2D normalization and an analytic ballistic check.
---

<p class="rs-kicker">Tutorial 4</p>

# Transport and diffusion

<div class="rs-lead" markdown>
Conductivity and diffusion are correlation functions of the velocity
operator. This tutorial computes the zero-temperature DC conductivity
spectrum and the energy-resolved mean-square displacement of a 32,768-atom
graphene sheet, and checks the diffusion against the analytic ballistic limit.
</div>

**You will learn:** `dc_spectrum` · `msd` · the `axis` normalization for 2D
materials · fixed spectral bounds · several tasks against one setup.

## The input

<!-- rsatb-input: examples/tutorials/graphene_transport.json -->

- **The hopping tapers smoothly** between `r_on` = 1.5 Å and `r_cut` = 1.7 Å
  (`exp_poly` with a single coefficient and no decay). A smooth cutoff matters
  more for transport than for a DOS, because the velocity operator involves
  the bond vectors.
- **`bounds` is fixed** at $\pm 9$ eV, which contains the $\pm 8.1$ eV
  spectrum. That skips the eigenvalue estimate — on large cells the faster
  and more robust choice.
- **`dc_spectrum`** with `"axis": "ab"` normalizes by the area of the sheet,
  not by a volume padded with 20 Å of vacuum. The result is a sheet
  conductance, in siemens.
- **`msd`** propagates wave packets for 50 steps of 2 fs and resolves them in
  energy with 256 Chebyshev moments, measuring displacement along $x$.

An orthogonal Slater-Koster model supplies the velocity operator for free, so
`rsatb check` accepts both transport tasks.

## Run it

```bash
rsatb run examples/tutorials/graphene_transport.json
```

All three tasks reuse one setup: one graph, one Hamiltonian, one window.

<figure markdown>
![DOS, DC conductivity and diffusion of graphene](../assets/figures/tut_graphene_transport.png)
<figcaption><strong>Left:</strong> the DOS of the same setup.
<strong>Centre:</strong> the zero-temperature Kubo–Greenwood sheet
conductance $\sigma_{xx}(E)$, vanishing at the Dirac point and the band
edges. <strong>Right:</strong> the diffusion coefficient $D(E,t)$ grows
linearly in time at every energy — ballistic transport, as expected for a
perfect crystal.</figcaption>
</figure>

## Checking the ballistic limit

In a clean crystal nothing scatters, so wave packets move ballistically and
$\Delta X^2 \propto t^2$, i.e. $D(E,t) = \langle v_x^2\rangle_E\,t$. Near the
Dirac point graphene's velocity has a fixed magnitude

$$
v_F = \frac{3|t|a_{CC}}{2\hbar} = \frac{3 \times 2.7\ \text{eV} \times 1.42\ \text{Å}}{2 \times 0.658\ \text{eV fs}} = 8.74\ \text{Å/fs},
$$

and averaging over directions gives $\langle v_x^2\rangle = v_F^2/2$. At
$t = 100$ fs:

$$
D = \tfrac12 v_F^2\, t = 3.8\times10^{3}\ \text{Å}^2/\text{fs} = 382\ \text{cm}^2/\text{s},
$$

which is what the curves for 0.5 and 1 eV reach. At higher energies the band
bends — trigonal warping, then the van Hove point at $|t| = 2.7$ eV where the
group velocity collapses — and $D$ grows more slowly.

!!! note "Why the DC conductivity is so large"
    With no disorder, the Kubo–Greenwood conductivity of a perfect crystal is
    infinite; what you see is limited only by the energy resolution of 256
    moments. Physical DC values need scattering: disorder in the Hamiltonian,
    a supercell large enough to contain it, and enough moments that the
    resolution is finer than the scattering rate. The MSD tells you which
    regime you are in — ballistic ($D \propto t$), diffusive ($D$ plateaus),
    or localized ($D$ falls back).

## Reading the result

```python
import numpy as np

run = "examples/tutorials/results/graphene_transport"
XX = 0                                     # components: xx, xy, xz, yy, yz, zz

dc = np.load(f"{run}/dc_spectrum.npz")
energy, sigma_xx = dc["energy"], dc["sigma"][0, :, XX]   # S (sheet), k-point 0

msd = np.load(f"{run}/msd.npz")
t, E = msd["time_fs"], msd["energy"]
D = msd["diffusion"][0] * 0.1              # Å²/fs → cm²/s, shape (n_t, n_E)
```

## Try next

- Switch `dc_spectrum` to `"kernel": "costaper"` — better when the spectrum
  feeds an energy integral, as in the `dc` task at finite temperature.
- Add `{"type": "dc", "mu": 0.5, "temp": 300, "n_moments": 256, "axis": "ab"}`
  for the conductance at one chemical potential and temperature.
- Measure along $y$ with `"axis": "y"` on the `msd` task: graphene is
  isotropic, so the curves should agree.
