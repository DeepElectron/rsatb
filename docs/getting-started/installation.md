---
description: Install RSATB from source in a conda environment, on a generic Linux system, or on Digital Research Alliance of Canada clusters.
---

<p class="rs-kicker">Getting started</p>

# Installation

<div class="rs-lead" markdown>
RSATB is a Python package around a C++17 solver. `pip install .` builds both;
what it needs from your system is a compiler, MPI and a handful of numerical
libraries. A fresh conda environment provides all of them.
</div>

## Recommended: a conda environment

```bash
git clone --recurse-submodules https://github.com/DeepElectron/rsatb.git
cd rsatb

conda create -n rsatb -c conda-forge \
  python=3.11 compilers cmake ninja openmpi mpi4py \
  blas lapack fftw hdf5 libmatio metis parmetis primme
conda activate rsatb

python -m pip install .
```

Then confirm the build:

```console
$ rsatb info
rsatb 0.1.0
input format version 1

models        dftb, lcao, sktb, tb_file
tasks         ac, dc, dc_spectrum, density, dos, energy_bounds, fermi_level, ldos, msd, occupation, pdos, quantum_metric
radial forms  constant, exp_poly, spline
H(R) formats  deeptb, abacus, nextham

solver        available (.../site-packages/rsatb/_rsatb.cpython-311-x86_64-linux-gnu.so)
MPI           1 rank(s) in this launch
```

`solver available` is the line that matters: it means the compiled extension
loaded. If you cloned without `--recurse-submodules`, run
`git submodule update --init --recursive`; CMake can also fetch the same
pinned sources itself when the submodule directories are absent.

## Other systems

=== "Linux, system libraries"

    You need Python ≥ 3.9, CMake ≥ 3.15, a C++17 compiler, MPI,
    BLAS/LAPACK, FFTW, HDF5 with its C++ library, MATIO, METIS, ParMETIS and
    PRIMME. Once CMake can find them:

    ```bash
    python -m pip install .
    ```

    Point CMake at non-standard locations with the usual roots, passed through
    scikit-build-core:

    ```bash
    python -m pip install . \
      -Ccmake.define.HDF5_ROOT=/opt/hdf5 \
      -Ccmake.define.PRIMME_ROOT=/opt/primme
    ```

    `FFTW_ROOT`, `METIS_ROOT`, `PARMETIS_ROOT` and `MATIO_ROOT` work the same
    way.

=== "Alliance clusters"

    On Digital Research Alliance of Canada systems, activate the virtual
    environment RSATB should live in, then:

    ```bash
    bash scripts/install_alliance.sh
    ```

    The script loads the module stack, checks native headers and shared
    libraries, installs the Python dependencies, builds RSATB and checks the
    extension. `RSATB_MODULES` replaces the default module list;
    `RSATB_EXTRA_MODULES` appends site-specific modules.

### Choosing a BLAS

CMake picks a BLAS automatically. To force one:

```bash
python -m pip install . -Ccmake.define.BLAS_BACKEND=MKL   # Auto | MKL | OpenBLAS | Generic
```

Threading is controlled by OpenMP alone — `OMP_NUM_THREADS` — whichever BLAS
is linked. RSATB sets `MKL_THREADING_LAYER=GNU` on import so MKL and the GNU
OpenMP runtime agree; set it yourself beforehand to override.

## Optional extras

```bash
python -m pip install ".[plot]"   # matplotlib, for the plotting snippets in these docs
python -m pip install ".[dev]"    # pytest, to run the test suite
```

## Troubleshooting

??? question "`rsatb info` says the solver is not available"
    The package imported but the extension did not load. Run
    `python -c "import rsatb.driver"` to see the underlying error; it is
    almost always a shared library (HDF5, MPI, PRIMME) that is visible at build
    time but not on `LD_LIBRARY_PATH` at run time.

??? question "`mpirun -n 4` is no faster than one rank"
    Open MPI 5 binds each rank to a single core by default, so threads have
    nowhere to run. State the mapping explicitly — see
    [Run on a cluster](../guides/parallel.md).

??? question "`python` runs Python 2"
    On some clusters the bare `python` is the system Python 2. Use the
    environment's `python3`, or `python -m pip` from inside the activated
    environment.
