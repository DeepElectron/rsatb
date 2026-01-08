#!/bin/bash
set -e

echo "Configuring environment..."
# Add system mpi4py to PYTHONPATH so pip sees it as installed
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/gcc12/openmpi4/mpi4py/4.0.3/lib/python3.11/site-packages:$PYTHONPATH

echo "Installing build dependencies..."
pip install scikit-build-core pybind11 cmake ninja fypp

echo "Installing runtime dependencies..."
# Install runtime dependencies explicitly using available wheels
pip install numpy scipy matplotlib ase --no-index

echo "Installing rsatb with --no-build-isolation..."
# We use --no-build-isolation because mpi4py is a system module on Compute Canada
# and cannot be installed via pip in an isolated environment.
module load hdf5 fftw metis parmetis primme flexiblas imkl

# Explicitly pass BLAS/LAPACK libraries to CMake using the environment variables set by the modules
# We find the absolute path to libflexiblas.so to ensure CMake finds it
FLEXIBLAS_LIB=$(find $EBROOTFLEXIBLAS -name "libflexiblas.so" | head -n 1)
echo "Found FlexiBLAS lib: $FLEXIBLAS_LIB"

# Find ScaLAPACK library
SCALAPACK_LIB=$(find $EBROOTSCALAPACK -name "libscalapack.so" | head -n 1)
echo "Found ScaLAPACK lib: $SCALAPACK_LIB"

# Find FFTW pkgconfig path
FFTW_PKGCONFIG=$(dirname $(find $EBROOTFFTW -name "fftw3.pc" | head -n 1))
echo "Found FFTW pkgconfig: $FFTW_PKGCONFIG"
export PKG_CONFIG_PATH=$FFTW_PKGCONFIG:$PKG_CONFIG_PATH

export pybind11_DIR=/home/zhanghao/workplace/rsatb/rsenv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11
pip install . --no-build-isolation -v \
    -C cmake.define.BLAS_LIBRARIES="$FLEXIBLAS_LIB" \
    -C cmake.define.LAPACK_LIBRARIES="$FLEXIBLAS_LIB" \
    -C cmake.define.SCALAPACK_LIBRARIES="$SCALAPACK_LIB"
