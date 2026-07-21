# RSATB: Random State Ab-initio Tight-Binding

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
