# RSATB Python API

The primary interface is a JSON input file and the `rsatb` command —
see [json_input.md](json_input.md). This document describes the Python objects
that layer is built on, which is what to reach for when a calculation needs
something an input file does not express: a structure generated on the fly, a
parameter sweep, or post-processing between observables.

The two meet at `rsatb.inputs.build_calculation`, which turns a validated input
into exactly the objects below:

```python
from rsatb.inputs import load_and_resolve, build_calculation

resolved, issues = load_and_resolve("input.json")
calculation, model, options = build_calculation(resolved)   # an RSATB, as constructed here
```

## The shape of every calculation

Three objects, one constructor, then observables:

```python
from rsatb import RSATB, Structure, SktbModel, SkParameters, SetupOptions, sk_spline

structure = Structure.from_file("graphene.vasp")          # where the atoms are
model     = SktbModel(...)                                 # what H and S are
calc      = RSATB(structure, model, SetupOptions(...), npar=4)

dos, energy = calc.compute_dos(dE=0.1)
```

`Structure` says where the atoms are. The model says what the Hamiltonian is.
`SetupOptions` says what the calculation should do. Nothing else is required,
and nothing is passed twice.

## The five design rules

**1. Nothing is declared that can be derived.**

The model knows its own basis, so orbital counts and graph cutoffs are never
arguments:

```python
model = LcaoModel(
    hamiltonian=HamiltonianFile("hamiltonians.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/"),
)
model.basis          # <BasisSpec C: 13 orb, r<=3.7042 A>   -- read from the .orb file
```

Those two numbers used to be constructor arguments (`type_norb=[13]`,
`r_max_per_type=[3.7]`), hand-transcribed from the basis-set file with nothing
checking them.

**2. A wrong input is an error, not a different calculation.**

```python
SetupOptions(kpoint=[[0, 0, 0]])       # TypeError: no such argument
PreconditionerOptions(target="H")      # explains what replaced the old "H" mode
LcaoModel(hamiltonian=..., orbitals=OrbitalFiles([]))
                                       # refuses: LCAO is non-orthogonal, orbitals are required
```

Config types are defined once in C++ and bound directly, so an unrecognised
keyword cannot be silently dropped.

**3. One way to express each thing.**

Hamiltonian and overlap are separate `SkParameters` objects, so there is no
shape to guess at:

```python
hopping = SkParameters(
    onsite={"C": {"pz": 0.0}},
    pairs={("C", "C"): {"pp_pi": sk_spline(r, v_pi, r_cut)}},
)
SktbModel(basis={"C": ["pz"]}, hamiltonian=hopping)                  # orthogonal
SktbModel(basis={"C": ["pz"]}, hamiltonian=hopping, overlap=overlap) # not
```

**4. Options say what they do.**

`PreconditionerOptions(mode="H")` used to free the overlap matrix as a side
effect of a mode name. It is now `operator_form="sandwich"` (previously `release_overlap_after_build=True`), naming the operator it selects -- Z H Z with S released -- rather than the memory side effect.

**5. Objects describe themselves.**

```python
>>> model
<LCAO C(13) non-orthogonal>
>>> model.capabilities
<Capabilities non-orthogonal, velocity, position>
>>> options
<SetupOptions nk=21 nsamples=500 bounds=auto>
```

## Choosing a model

| your input | class |
|---|---|
| Hamiltonian file + `.orb` (+ optional `.upf`) | `LcaoModel` — always non-orthogonal |
| Slater-Koster parameters you write | `SktbModel` — orthogonal unless `overlap=` |
| DFTB `.skf` parameter set | `DftbModel.from_skf_directory(...)` |
| Hamiltonian file, already orthogonal | `TightBindingFileModel` |

`TightBindingFileModel` is the only one that must be told its basis, because the
supported file formats carry no basis header. Every other model reads it.

## Building the structure

```python
Structure.from_file("POSCAR", supercell=[4, 4, 1]) # expansion happens during setup
Structure.from_arrays(pos=[...], z=[6, 6], cell=[...], pbc=[True, True, False])
```

`supercell=[nx, ny, nz]` belongs to the structure. `RSATB` expands the
coordinates immediately before it builds the atomic graph and operators. A
Hamiltonian file's primitive `(i,j,R)` blocks are mapped into the expanded atom
and image indexing during the read, whatever its format; no expanded file is
written.

`HamiltonianFile(path, format=...)` takes `"deeptb"` (DeePTB HDF5), `"abacus"`
(ABACUS's `out_mat_hs2` sparse H(R), `hrs1_nao.csr` or the older
`data-HR-sparse_SPIN0.csr`, text or binary) or `"nextham"` (NextHam's
`HR_atom_pairs.h5`). The last two are in ABACUS's orbital order and are
converted using the `.orb` files' shells, so they need `LcaoModel`, a structure
listing the atoms in ABACUS's order, and a spin-degenerate (nspin=1) run. The
overlap is computed from the orbital files as always; ABACUS's `srs1_nao.csr`
and NextHam's `SR_atom_pairs.h5` are not read.

Every format is read distributed: each MPI rank asks for exactly the blocks
of its own atoms and holds nothing else. NextHam HDF5 and ABACUS's binary CSR
are read in parallel, each rank reading only its rows. ABACUS's text CSR cannot
be read out of order, so rank 0 parses it in one streaming pass and sends each
rank its rows; its time does not drop with more ranks. Measured on a
1728-atom, 4.3e7-nonzero Al file: NextHam 2.5 s on 1 rank and 0.23 s on 16,
binary 3.0 s and 0.20 s, text about 20 s at any rank count
(`calc/test/benchmark_hamiltonian_read.cpp`).

For a neighbour list you already have, `RSATB.from_graph(...)` takes it
directly. That is a **distributed** description — each rank passes its own
atoms and edges — and the local counts are checked against the declared totals,
so handing every rank the full graph raises instead of silently duplicating the
system.

## Observables

`compute_dos`, `compute_ac`, `compute_dc`, `compute_dc_spectrum`,
`compute_msd_spectrum`, `compute_diffusion_spectrum`, `compute_quantum_metric`.

Each returns numpy arrays with a leading k-point axis, already reduced across
ranks. An observable a model cannot supply fails with a message naming the
missing capability rather than returning zeros:

```python
RuntimeError: compute_ac requires a velocity-capable model/operator setup.
```

Check ahead of time with `model.capabilities`.

## Migration

| before | now |
|---|---|
| `RSATB.from_file(npar, f, r_max_per_type, type_norb)` + `setup_lcao(...)` | `RSATB(Structure.from_file(f), model, options, npar)` |
| `RSATB.from_raw_data(...)` + `setup_sktb(...)` | `RSATB.from_graph(..., model=..., options=...)` |
| `model.prepare_atomic_spec_from_file(f)` | deleted — the basis is derived |
| `SKTBModel.native(basis, onsite, pairs)` | `SktbModel(basis, hamiltonian=SkParameters(onsite, pairs))` |
| `LCAOModel.hdf5(f, integrals=LCAOIntegralInput(...))` | `LcaoModel(hamiltonian=HamiltonianFile(f), orbitals=..., pseudopotentials=...)` |
| `Hdf5Input(f, nspin=..., lspinorb=...)` | `HamiltonianFile(f, format=...)` + `spin=` on the model |
| `PreconditionerOptions(mode="H")` | `PreconditionerOptions(operator_form="sandwich")` |
| `PreconditionerOptions(mode="None")` | `PreconditionerOptions(target="none")` |
| `setup_model` / `setup_lcao` / `setup_sktb` / `setup_dftb` / `setup_config` | the constructor |

Observable methods keep their signatures and return shapes.
