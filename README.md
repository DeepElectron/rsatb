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
    parmetis metis fftw hdf5 libmatio pybind11 mpi4py
```

`libmatio` is required by the bundled `op2c` (it links its rescumat `.mat`
reader into `nao_core` unconditionally, even though rsatb does not use that
format). Configuration fails with an explicit message if it is missing.

### Compute Canada Instructions
1. Load required modules:
   ```bash
   module load python/3.11
   module load scipy-stack
   module load petsc
   module load slepc
   module load mpi4py
   ```
   *Note: `dbcsr` is not available as a module and will be built locally.*

2. Create a virtual environment:
   ```bash
   virtualenv --no-download .venv
   source .venv/bin/activate
   ```

3. Install dependencies and build:
   ```bash
   # Install build dependencies
   pip install scikit-build-core pybind11 cmake ninja fypp

   # Build external dependencies (DBCSR)
   # This script detects if PETSc/SLEPc are loaded and skips them if so.
   source scripts/install_deps.sh

   # Install rsatb
   pip install .
   ```

### General Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Build and install:
   ```bash
   source scripts/install_deps.sh
   pip install .
   ```
