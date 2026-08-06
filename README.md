# RSATB: Random State Ab-initio Tight-Binding

## Running a calculation

A calculation is one JSON file and one command:

```bash
rsatb template sktb -o graphene.json   # a working input to start from
rsatb check graphene.json              # validate it before queueing it
rsatb run graphene.json                # compute
mpirun -n 64 rsatb run graphene.json   # the same input, in parallel
```

```json
{
  "structure": {"file": "graphene.vasp"},
  "model": {
    "type": "sktb",
    "basis": {"C": ["pz"]},
    "hamiltonian": {
      "onsite": {"C": {"pz": 0.0}},
      "pairs": {"C-C": {"pp_pi": {"kind": "constant", "value": -2.7, "r_cut": 1.6}}}
    }
  },
  "tasks": [{"type": "dos", "nt": 200}]
}
```

Four sections, and they answer four separate questions: where the atoms are,
what the Hamiltonian is, how the calculation is carried out, and which
observables to compute. Nothing is written twice — the model reports the basis
it works in, so orbital counts and graph cutoffs are derived, not declared —
and an unknown key is an error rather than a silent default.

`rsatb doc` explains the whole format, one topic at a time, and is generated
from the same declaration the validator uses. `rsatb check` needs neither MPI
nor the compiled solver, so an input can be written and validated anywhere and
only then queued; it reports malformed fields, missing files, and sections that
disagree — a task asking for the velocity operator from a model that cannot
build one, say.

- Reference: [`doc/json_input.md`](doc/json_input.md)
- Example inputs: [`examples/`](examples/)
- The Python API underneath: [`doc/python_api.md`](doc/python_api.md)

## Installation

### Prerequisites
- Python >= 3.9
- MPI (OpenMPI or MPICH)
- CMake >= 3.15
- C++ Compiler (GCC/Clang/Intel)

Native libraries (conda-forge names):

```bash
conda install -c conda-forge \
    compilers cmake ninja openmpi mpi4py \
    blas lapack fftw hdf5 libmatio metis parmetis primme
```

Python build/runtime packages:

```bash
python -m pip install scikit-build-core pybind11 numpy scipy ase
```

`libmatio` is required by the bundled `op2c` because its rescumat `.mat`
reader is compiled into `nao_core`. `hdf5` must include the C++ library
(`libhdf5_cpp`). `primme`, `metis`, `parmetis`, `fftw`, MPI, and BLAS/LAPACK
are native build/link dependencies, not optional Python packages.

### Compute Canada Instructions
1. Load required modules, create or activate a virtual environment, then run:
   ```bash
   bash scripts/install_alliance.sh
   ```

The script loads the Alliance module stack by default, verifies the native
headers and shared libraries with a compile/run smoke test, installs Python
dependencies into the active virtual environment, builds `rsatb`, then checks
that the installed extension links to HDF5, HDF5 C++, MATIO, FFTW, MPI, METIS,
ParMETIS, PRIMME, and BLAS/LAPACK. With the default MKL backend, FFTW API calls
are provided consistently through MKL's FFTW interface rather than a separate
`libfftw3`. Override module names with `RSATB_MODULES` or add site-specific
modules with `RSATB_EXTRA_MODULES`.

### General Installation
1. Install the native prerequisites listed above.
2. Build and install:
   ```bash
   pip install .
   ```

CMake accepts `-DBLAS_BACKEND=Auto|MKL|OpenBLAS|Generic` and dependency roots
such as `-DHDF5_ROOT=...`, `-DFFTW_ROOT=...`, `-DMETIS_ROOT=...`,
`-DPARMETIS_ROOT=...`, `-DPRIMME_ROOT=...`, and `-DMATIO_ROOT=...` when the
libraries are outside the compiler or module search path.
