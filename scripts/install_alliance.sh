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
module load hdf5 fftw metis parmetis primme imkl

# Explicitly pass BLAS/LAPACK libraries to CMake using the environment variables set by the modules
# We find the absolute path to libmkl_intel_lp64.so and its dependencies to ensure CMake finds them
# We grep for "intel64" to avoid picking up 32-bit (ia32) libraries
MKL_INTERFACE=$(find $EBROOTIMKL -name "libmkl_intel_lp64.so" | grep "intel64" | head -n 1)
MKL_CORE=$(find $EBROOTIMKL -name "libmkl_core.so" | grep "intel64" | head -n 1)
MKL_THREAD=$(find $EBROOTIMKL -name "libmkl_sequential.so" | grep "intel64" | head -n 1)
BLAS_LIB="${MKL_INTERFACE};${MKL_THREAD};${MKL_CORE}"
echo "Found BLAS libs: $BLAS_LIB"

# Find ScaLAPACK library
MKL_SCALAPACK=$(find $EBROOTIMKL -name "libmkl_scalapack_lp64.so" | grep "intel64" | head -n 1)
MKL_BLACS=$(find $EBROOTIMKL -name "libmkl_blacs_openmpi_lp64.so" | grep "intel64" | head -n 1)
SCALAPACK_LIB="${MKL_SCALAPACK};${MKL_BLACS}"
echo "Found ScaLAPACK libs: $SCALAPACK_LIB"

# Find FFTW pkgconfig path
FFTW_PKGCONFIG=$(dirname $(find $EBROOTFFTW -name "fftw3.pc" | head -n 1))
echo "Found FFTW pkgconfig: $FFTW_PKGCONFIG"
export PKG_CONFIG_PATH=$FFTW_PKGCONFIG:$PKG_CONFIG_PATH

export pybind11_DIR=/home/zhanghao/workplace/rsatb/rsenv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11
pip install . --no-build-isolation -v \
    -C cmake.define.BLAS_LIBRARIES="$BLAS_LIB" \
    -C cmake.define.LAPACK_LIBRARIES="$BLAS_LIB" \
    -C cmake.define.SCALAPACK_LIBRARIES="$SCALAPACK_LIB"
