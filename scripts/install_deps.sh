#!/bin/bash
set -e

# Directory to install dependencies
DEPS_DIR=$(pwd)/deps
mkdir -p $DEPS_DIR
INSTALL_DIR=$DEPS_DIR/install
mkdir -p $INSTALL_DIR

echo "Installing dependencies to $INSTALL_DIR"

# 1. PETSc
if [ -n "$PETSC_DIR" ]; then
    echo "Using system PETSc at $PETSC_DIR"
else
    echo "PETSc not found in environment."
    echo "On Compute Canada, please run: module load petsc"
    echo "Building PETSc from source..."
    if [ ! -d "$INSTALL_DIR/lib/pkgconfig/PETSc.pc" ]; then
        if [ ! -d "$DEPS_DIR/petsc" ]; then
            git clone -b release https://gitlab.com/petsc/petsc.git $DEPS_DIR/petsc
        fi
        pushd $DEPS_DIR/petsc
        ./configure --prefix=$INSTALL_DIR --with-debugging=0 --with-shared-libraries=1 --with-scalar-type=complex --download-fblaslapack --download-mpich --download-cmake
        make
        make install
        popd
    else
        echo "PETSc already installed locally."
    fi
    export PETSC_DIR=$INSTALL_DIR
    export PETSC_ARCH=""
fi

# 2. SLEPc
if [ -n "$SLEPC_DIR" ]; then
    echo "Using system SLEPc at $SLEPC_DIR"
else
    echo "SLEPc not found in environment."
    echo "On Compute Canada, please run: module load slepc"
    echo "Building SLEPc from source..."
    if [ ! -d "$INSTALL_DIR/lib/pkgconfig/SLEPc.pc" ]; then
        if [ ! -d "$DEPS_DIR/slepc" ]; then
            git clone -b release https://gitlab.com/slepc/slepc.git $DEPS_DIR/slepc
        fi
        pushd $DEPS_DIR/slepc
        ./configure --prefix=$INSTALL_DIR
        make
        make install
        popd
    else
        echo "SLEPc already installed locally."
    fi
    export SLEPC_DIR=$INSTALL_DIR
fi

# 3. DBCSR
if [ ! -d "$INSTALL_DIR/lib/cmake/dbcsr/DBCSRConfig.cmake" ]; then
    echo "Downloading and building DBCSR..."
    if [ ! -d "$DEPS_DIR/dbcsr" ]; then
        git clone -b v2.6.0 https://github.com/cp2k/dbcsr.git $DEPS_DIR/dbcsr
    fi
    pushd $DEPS_DIR/dbcsr
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DUSE_SMM=blas -DUSE_ACCEL=off -DBUILD_SHARED_LIBS=ON -DBLA_VENDOR=OpenBLAS
    make -j4
    make install
    popd
else
    echo "DBCSR already installed."
fi

echo "Dependencies installed successfully."
export CMAKE_PREFIX_PATH=$INSTALL_DIR:$CMAKE_PREFIX_PATH
echo "CMAKE_PREFIX_PATH set to $CMAKE_PREFIX_PATH"
