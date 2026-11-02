---
description: Install RSATB from source in a conda environment, on a Linux system with its own libraries, or on Digital Research Alliance of Canada clusters.
---

<p class="rs-kicker">Get started</p>

# Installation

<div class="rs-lead" markdown>
RSATB is a Python package with a compiled C++ solver. It is installed from
source; the build needs a C++17 compiler, MPI and several numerical libraries,
all available from conda-forge.
</div>

## Conda environment (recommended)

```bash
git clone --recurse-submodules https://github.com/DeepElectron/rsatb.git
cd rsatb

conda create -n rsatb -c conda-forge \
  python=3.11 compilers cmake ninja openmpi mpi4py \
  blas lapack fftw hdf5 libmatio metis parmetis primme
conda activate rsatb

python -m pip install .
```

Check the installation:

```console
$ rsatb info
rsatb 0.1.0
...
solver        available (...)
MPI           1 rank(s) in this launch
```

`solver available` means the compiled part was built and loads. If the
repository was cloned without `--recurse-submodules`, run
`git submodule update --init --recursive` before installing.

## Other systems

=== "Linux, system libraries"

    Requirements: Python ≥ 3.9, CMake ≥ 3.15, a C++17 compiler, MPI,
    BLAS/LAPACK, FFTW, HDF5 (with its C++ library), MATIO, METIS, ParMETIS
    and PRIMME. Then:

    ```bash
    python -m pip install .
    ```

    Libraries in non-standard locations are passed to CMake:

    ```bash
    python -m pip install . \
      -Ccmake.define.HDF5_ROOT=/opt/hdf5 \
      -Ccmake.define.PRIMME_ROOT=/opt/primme
    ```

    `FFTW_ROOT`, `METIS_ROOT`, `PARMETIS_ROOT` and `MATIO_ROOT` work the same
    way. A specific BLAS can be selected with
    `-Ccmake.define.BLAS_BACKEND=MKL` (or `OpenBLAS`, `Generic`).

=== "Alliance clusters"

    On Digital Research Alliance of Canada systems, activate the Python
    virtual environment to install into, then run:

    ```bash
    bash scripts/install_alliance.sh
    ```

    The script loads the required modules, builds RSATB and checks the
    result. `RSATB_EXTRA_MODULES` adds site-specific modules.

To plot results as in these pages, also install matplotlib
(`python -m pip install ".[plot]"`).

## Troubleshooting

??? question "`rsatb info` says the solver is not available"
    The compiled part did not load. `python -c "import rsatb.driver"` shows
    the error. The cause is usually a shared library (HDF5, MPI or PRIMME)
    that is not on `LD_LIBRARY_PATH` at run time.

??? question "OpenMP threads do not speed up a run started with `mpirun`"
    Open MPI binds each process to one core by default, so its threads share
    that core. Give the process placement explicitly, as described in
    [Run on a cluster](../guides/parallel.md).
