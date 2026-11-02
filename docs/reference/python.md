---
description: The RSATB Python API — Structure, the four model classes, SetupOptions, PreconditionerOptions and the RSATB calculation object with its compute methods.
---

<p class="rs-kicker">Reference</p>

# Python API

<div class="rs-lead" markdown>
The JSON input is the primary interface; this is the layer underneath it, for
calculations an input file cannot express — a structure generated on the fly,
a parameter sweep, post-processing between observables. Every input key maps
onto an argument documented here.
</div>

```python
from mpi4py import MPI      # initialise MPI before any native object is built
from rsatb import (RSATB, SetupOptions, SkParameters, SktbModel,
                   Structure, sk_exp_poly)

structure = Structure.from_file("graphene.vasp", supercell=[64, 64, 1])

hopping = sk_exp_poly(coeffs=[-2.7], r_cut=1.6)
model = SktbModel(
    basis={"C": ["pz"]},
    hamiltonian=SkParameters(onsite={"C": {"pz": 0.0}},
                             pairs={("C", "C"): {"pp_pi": hopping}}),
)
calc = RSATB(structure, model, SetupOptions(nsamples=8), npar=1)

dos, energy = calc.compute_dos(dE=0.05)
```

Three objects and one constructor: the structure says where the atoms are,
the model what the Hamiltonian is, the options how to run. A worked example is
[Tutorial 8](../tutorials/python.md).

## From an input file

`rsatb.inputs.build_calculation` turns a validated input into exactly these
objects, so an input can be the starting point a script goes beyond:

```python
from rsatb.inputs import load_and_resolve, build_calculation, run_tasks

resolved, issues = load_and_resolve("input.json")
if any(issue.is_error for issue in issues):
    raise SystemExit("\n".join(str(issue) for issue in issues))

calculation, model, options = build_calculation(resolved)
results = run_tasks(calculation, resolved["tasks"], options)
```

## Structure

```python
Structure.from_file(filename, format="", supercell=[1, 1, 1])
Structure.from_arrays(pos, z, cell, pbc=[True, True, True], supercell=[1, 1, 1])
```

| Argument | |
|---|---|
| `filename`, `format` | POSCAR/`.vasp`, `.xyz`, `.cif`; `format` overrides the extension |
| `pos` | Cartesian positions in Å, **flat**: three values per atom |
| `z` | atomic numbers, one per atom |
| `cell` | lattice vectors in Å, **row-major, nine values** |
| `pbc` | periodicity per lattice vector |
| `supercell` | integer tiling applied during setup; file Hamiltonians are mapped, not copied |

Attributes `pos`, `z`, `cell`, `pbc`, `n_atom`, `supercell`;
`atomic_number("C")` converts a symbol.

## Models

All models expose `.basis` (a `BasisSpec`), `.spin` and `.capabilities`, and
describe themselves:

```python
>>> model
<LCAO C(13) non-orthogonal>
>>> model.capabilities
<Capabilities non-orthogonal, velocity, position>
```

### `SktbModel`

```python
SktbModel(basis, hamiltonian, overlap=None, spin="none")
SkParameters(onsite={species: {orbital: value}},
             pairs={(species_a, species_b): {channel: SkRadialFunction}})
sk_spline(r, values, r_cut=None)
sk_exp_poly(coeffs, r_cut, lambda_value=0.0, r_on=0.0)
```

`hamiltonian` and `overlap` are separate `SkParameters`; omitting `overlap`
makes the model orthogonal. Channel names are shell pair plus bond type:
`"ss_sigma"`, `"sp_sigma"`, `"pp_pi"`, `"dd_delta"`, …. The radial forms match
the [input's](input/model/radial.md) `spline` and `exp_poly`.

### `DftbModel`

```python
DftbModel.from_skf_directory(skf_dir, basis, heteronuclear="reject_ambiguous", spin="none")
```

### `LcaoModel`

```python
LcaoModel(hamiltonian, orbitals, pseudopotentials=PseudopotentialFiles(),
          spin="none", log_file="op2c.log")
HamiltonianFile(path, format="deeptb")          # "deeptb" | "abacus" | "nextham"
OrbitalFiles(names, dir="")                     # one .orb per species, ascending Z
PseudopotentialFiles(names=[], dir="")          # optional cross-check velocity only
```

### `TightBindingFileModel`

```python
TightBindingFileModel(hamiltonian, type_symbols, type_norb, cutoff_radius, spin="none")
```

The one model that must be told its basis.

## Options

```python
SetupOptions(kpoints=[[0, 0, 0]], nsamples=1, sample_seed=20240917,
             compute_bounds=True, energy_bounds=None,
             bounds_zero_tol=1e-12, bounds_max_matvecs=10000,
             preconditioner=PreconditionerOptions())

PreconditionerOptions(operator_form="solve", tol=1e-7, sptol=1e-5, maxiter=0)
```

`energy_bounds=(emin, emax)` fixes the spectral window, like
`run.bounds.mode = "fixed"` in an input. Unknown keyword arguments raise a
`TypeError`.

Verbosity is process-wide, not an argument: call
`rsatb.set_log_level("silent" | "summary" | "progress" | "debug")` before
building a calculation.

## The calculation object

Every `compute_*` method returns NumPy arrays already reduced across MPI
ranks. Shapes are given in each method's docstring below; note that
`compute_dos` returns the k-points concatenated into one flat array, which
`dos.reshape(-1, len(energy))` splits. The JSON tasks apply these reshapes and
write the shapes listed under [Output files](outputs.md). A method the model
cannot support fails with a message naming the missing capability rather than
returning zeros:

```text
RuntimeError: compute_ac requires a velocity-capable model/operator setup.
```

::: rsatb.driver.RSATB
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - __init__
        - from_graph
        - energy_bounds
        - compute_dos
        - compute_pdos
        - compute_occupation
        - find_fermi_level
        - compute_rs_ldos
        - compute_rs_density
        - compute_dc
        - compute_dc_spectrum
        - compute_ac
        - compute_msd_spectrum
        - compute_diffusion_spectrum
        - fit_ballistic_velocity
        - extract_mean_free_path
        - compute_quantum_metric
