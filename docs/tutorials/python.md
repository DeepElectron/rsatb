---
description: Tutorial — a biaxial strain sweep of graphene through the RSATB Python API, with a distance-dependent Slater-Koster hopping built from sk_exp_poly.
---

<p class="rs-kicker">Tutorial 7</p>

# Scripting with Python

<div class="rs-lead" markdown>
An input file describes one calculation. A sweep — a structure generated on
the fly, a parameter varied in a loop, post-processing between runs — is a
program. This tutorial stretches graphene by up to 10% and watches its band
narrow, using the objects the JSON input is built from.
</div>

**You will learn:** `Structure.from_arrays` · `SktbModel` and `SkParameters` ·
`sk_exp_poly` · `SetupOptions` · `RSATB.compute_dos` · running a script under
MPI.

## The physics

Hopping integrals decay with bond length. A standard form for graphene is

$$
t(r) = t_0\,e^{-\beta\,(r/a_0 - 1)}, \qquad t_0 = -2.7\ \text{eV},\ a_0 = 1.42\ \text{Å},\ \beta = 3.37 .
$$

That is exactly RSATB's `exp_poly` radial form, $P(r)\,e^{-\lambda r}$, with a
one-term polynomial $P = t_0 e^{\beta}$ and $\lambda = \beta/a_0$. Stretching
the lattice lengthens the bonds, shrinks $|t|$, and narrows the band from
$\pm 3|t|$.

## The script

```python title="examples/tutorials/strain_sweep.py"
--8<-- "examples/tutorials/strain_sweep.py:21:84"
```

Line by line:

- **`from mpi4py import MPI` comes first.** MPI must be initialized before
  any native object is built.
- **`Structure.from_arrays`** takes flat Cartesian positions (three values per
  atom), atomic numbers, and a row-major $3\times3$ cell — and the same
  `supercell` tiling the JSON input has.
- **`sk_exp_poly`** builds the radial function; `r_on` and `r_cut` add a cosine
  taper so the hopping reaches zero smoothly before second neighbours.
- **`SkParameters`** keys pairs by a tuple, `("C", "C")`, where the JSON input
  writes `"C-C"`. Omitting `overlap` makes the model orthogonal.
- **`compute_dos`** returns the k-points concatenated into one flat array;
  `reshape(-1, len(energy))` splits them. Dividing by the window width gives
  the DOS per orbital.
- **Only rank 0 prints and writes.** Every rank runs the same loop; results
  are already reduced across ranks.

## Run it

```bash
python examples/tutorials/strain_sweep.py           # one rank
mpirun -n 4 python examples/tutorials/strain_sweep.py
```

```text
strain  0.0%   |t| = 2.700 eV   window [-8.60, 8.60] eV
strain  2.5%   |t| = 2.482 eV   window [-7.94, 7.94] eV
strain  5.0%   |t| = 2.281 eV   window [-7.34, 7.34] eV
strain  7.5%   |t| = 2.097 eV   window [-6.79, 6.79] eV
strain 10.0%   |t| = 1.928 eV   window [-6.28, 6.28] eV
wrote examples/tutorials/strain_sweep.npz
```

Each window is $3|t| + 0.5$ eV: the band edges follow the hopping exactly.

<figure markdown>
![Graphene DOS at five biaxial strains](../assets/figures/tut_strain_sweep.png)
<figcaption>Biaxial strain narrows the graphene band. The van Hove peaks
track $\pm|t(r)|$ inwards; the Dirac point stays pinned at zero because the
lattice keeps its symmetry. Each curve is a 32,768-atom calculation at 0.1 eV
resolution.</figcaption>
</figure>

## From an input file instead

A script can also start from a JSON input and go beyond it, because
`build_calculation` returns exactly these objects:

```python
from mpi4py import MPI
from rsatb.inputs import load_and_resolve, build_calculation

resolved, issues = load_and_resolve("examples/01_sktb_graphene.json")
calc, model, options = build_calculation(resolved)

print(model.capabilities)            # <Capabilities orthogonal, velocity, position>
dos, energy = calc.compute_dos(dE=0.05)
```

The full API is in the [Python reference](../reference/python.md).
