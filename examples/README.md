# Examples

A calculation is one JSON file and one command. These are that file, one per
model family, and they are the place to start reading.

```bash
rsatb check examples/01_sktb_graphene.json    # validate, without running
rsatb run   examples/01_sktb_graphene.json    # compute
mpirun -n 4 rsatb run examples/02_lcao_diamond.json
```

Paths inside each file are relative to the file itself, so they run from
anywhere. `rsatb doc` explains every key; the format reference is
[`doc/json_input.md`](../doc/json_input.md).

| file | model | what it shows |
|---|---|---|
| `01_sktb_graphene.json` | `sktb` | the model *is* the parameters — no data files at all |
| `02_lcao_diamond.json` | `lcao` | a precomputed H plus the `.orb` files the overlap comes from |
| `03_dftb_skf.json` | `dftb` | Slater-Koster tables from a directory of `.skf` |
| `04_tb_file_chain.json` | `tb_file` | an orthogonal H(R) straight from a file |
| `05_graphene_transport.json` | `sktb` | several observables against one setup |
| `06_matblg_transport.json` | `sktb` | a full research run — PRL 134, 126301 |

01 through 05 run as written: everything they read is in `data/`. 06 is the one
that needs something you supply — a relaxed moiré unit cell — and says so.

`publication/` holds the calculations of the paper, as run: the inputs, the
run scripts, the diagonalization references and the figure scripts for the
spectrum / real-space gallery (`publication/spectrum_realspace/`) and the
transport figure (`publication/transport/`). They read the primitive cells in
`test/data/` and are sized for a 48-core node with 256 GB, hours to days per
run; each folder's README says what it reproduces and in which order.

## What each one is for

**01 — SKTB.** Graphene, one pz orbital per carbon, a single −2.7 eV pp_pi
hopping out to 1.6 Å. Each carbon has three neighbours at 1.42 Å, so the
Gamma-point eigenvalues are ±3|t| = ±8.1 eV and the reported window is that
padded by 0.5 eV — the analytic check. Omitting `overlap` leaves the basis
orthogonal, which is why velocity and position come for free.

**02 — LCAO.** Orbital files are required, not optional: an LCAO basis is
non-orthogonal and they are what the overlap is computed from. The velocity is
the commutator i[H, r] from the model's dipole integrals, so no pseudopotential
files are needed. Nothing about the
basis is declared — the orbital count (13) and the graph cutoff (3.7042 Å) are
read from `C_gga_7au_100Ry_2s2p1d.orb`; those two numbers used to be
constructor arguments transcribed by hand. Uses the diamond test data.

**03 — DFTB.** The basis is declared, not inferred: a `.skf` tabulates all ten
channels up to *d* whether the parameterization uses them or not. `data/skf`
holds a small synthetic C–C set so this runs with no downloads; point `skf_dir`
at a released set (mio-1-1, 3ob, …) for real work.

**04 — TightBindingFileModel.** What a DeePTB Slater-Koster export produces:
H(R) already orthogonal, with no overlap and nothing to evaluate. The only
model that must be told its basis, because the supported formats carry no basis
header. `data/chain.h5` is a 1D chain whose Gamma-point level e₀ + 2t = −1.5 eV
is known analytically.

**05 — several observables, one setup.** DOS, the DC conductivity spectrum, and
the mean-square displacement against a 4×4 graphene supercell. Tasks run in
order against the same setup, so three observables cost one setup, not three.
Sized to finish on a laptop; scale the supercell and the moment counts up for
production.

**06 — MATBLG transport.** The clean pz-only calculation from
P. Alcazar Guerrero *et al.*, Phys. Rev. Lett. **134**, 126301 (2025): the
paper's pp_pi / pp_sigma parameters tabulated in `data/matblg/`, a 16×16×1
tiling of the relaxed moiré cell, DOS, MSD and the quantum metric at Gamma.
Replace `structure.file` with your relaxed cell. Structural relaxation and
Anderson-disorder injection are not part of it — for the disordered workflow
you still need a wrapper around `rsatb::calc::apply_anderson_disorder`, or an
externally generated disordered Hamiltonian read through `lcao` / `tb_file`.

## Estimating the cost first

`estimate_matvecs.py` is the one piece that does not belong in an input file: a
cost model. Matrix-vector products are what a run spends its time on, and their
number follows from the Chebyshev expansions the requested tasks use, so it can
be worked out from the input alone before anything is queued.

```bash
python examples/estimate_matvecs.py examples/06_matblg_transport.json
```

```
  dos                      19,456 matvecs   1024 time steps x 19 Chebyshev terms
  msd                     839,398 matvecs   KPM density 706798 + affine time 132600
  quantum_metric           94,886 matvecs   H0 91808 + velocity 3078
  total                   953,740
```

It needs the spectral window, because two of the expansions are truncated
against it: an input with `run.bounds.mode = "fixed"` states it, and otherwise
pass `--bounds EMIN EMAX` or `--summary` from an earlier run. Like `rsatb
check`, it imports nothing compiled.

## Choosing the parallel layout

Three levels of parallelism, filled outside-in, because communication cost
increases inward:

1. **k-points** (`run.npar`): sub-communicator groups working independent
   k-points. Zero communication between groups, so this axis is linear —
   take all of it first. Set `npar` to the number of k-points or a divisor.
2. **MPI ranks within a group**: split the atoms; every rank must own some,
   and setup refuses layouts where one would own none.
3. **OpenMP threads within a rank** (`OMP_NUM_THREADS`): split the block
   rows. Effective while rows per thread stays large; physical cores only —
   the KPM loop is bandwidth-bound and SMT siblings just contend.

Two traps worth naming. Open MPI 5's default binds each rank to a **single
core**, so `mpirun -n 4` without a mapping gives four serial ranks whatever
`OMP_NUM_THREADS` says — always state the layout:

```bash
OMP_NUM_THREADS=12 mpirun -n 4 --map-by numa:PE=12 --bind-to core \
    rsatb run examples/02_lcao_diamond.json
```

And threads without work spin rather than sleep: a cell with 2 atoms has 2
block rows, so 96 threads on it burn ~100% CPU per thread at the barriers
while 2 do the work. If the profile table shows a KPM task consuming far more
CPU than wall time, the thread count is too high for the system, not the
other way around.

## Coordinates are used exactly as written

Positions may sit inside the cell or outside it; POSCAR, XYZ and CIF all allow
either. Nothing is wrapped, sorted or renumbered on the way in — atom order and
coordinates reach the graph as supplied, which is what a Hamiltonian file's own
atom indexing and R-vector convention refer to. Periodic images are expressed
against those coordinates, so `01_sktb_graphene.json` can write the second
carbon at Cartesian (0, 1.42), which is fractional (−1/3, 2/3), and still give
the analytic 3|t| = 8.1 eV.

## Going beyond an input file

An input file covers a structure, a model, run options and a list of
observables. When a calculation needs more than that — a structure generated on
the fly, a parameter sweep, post-processing between observables — the Python
API underneath is described in [`doc/python_api.md`](../doc/python_api.md), and
`rsatb.inputs.build_calculation` turns a validated input into exactly the
objects it documents:

```python
from rsatb.inputs import load_and_resolve, build_calculation

resolved, issues = load_and_resolve("examples/01_sktb_graphene.json")
calculation, model, options = build_calculation(resolved)
```
