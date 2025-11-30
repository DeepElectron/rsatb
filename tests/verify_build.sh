#!/bin/bash
# Verify build on Compute Canada

# Attempt to initialize modules if not present
if ! command -v module &> /dev/null; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /etc/profile.d/lmod.sh ]; then
        source /etc/profile.d/lmod.sh
    fi
fi

# Load modules
echo "Loading modules..."
module purge
module load python/3.11 scipy-stack petsc-complex slepc-complex mpi4py cmake/3.27.7 openblas
export BLA_VENDOR=OpenBLAS

# Create venv
echo "Creating virtual environment..."
if [ -d ".venv_test" ]; then
    rm -rf .venv_test
fi
python3 -m venv --system-site-packages .venv_test
source .venv_test/bin/activate

# Install build dependencies
echo "Installing build dependencies..."
pip install scikit-build-core pybind11 ninja fypp

# Run install_deps.sh (should skip if modules are loaded)
echo "Running install_deps.sh..."
rm -rf deps/dbcsr/build # Ensure clean build for DBCSR
source scripts/install_deps.sh

# Build rsatb
echo "Building rsatb..."
pip install . -v --no-build-isolation > build.log 2>&1

echo "Build verification complete."
