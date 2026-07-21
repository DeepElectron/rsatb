#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${RSATB_VENV:-${VIRTUAL_ENV:-/home/zhanghao/envs/rsatb}}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Could not find Python in ${VENV_DIR}." >&2
    echo "Activate the rsatb environment first, or set RSATB_VENV=/path/to/env." >&2
    exit 1
fi

if ! command -v module >/dev/null 2>&1; then
    # Alliance/Compute Canada exposes the module command through this init script
    # in non-interactive shells.
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi

if ! command -v module >/dev/null 2>&1; then
    echo "Could not initialize the Alliance module command." >&2
    exit 1
fi

RSATB_MODULES="${RSATB_MODULES:-StdEnv/2023 gcc/12.3 openmpi/4.1.5 cmake/3.31.0 imkl/2025.2.0 hdf5-mpi/1.14.6 fftw-mpi/3.3.10 metis/5.1.0 parmetis/4.0.3 primme/3.2 matio/1.5.23}"
read -r -a MODULES <<< "${RSATB_MODULES}"

echo "Loading Alliance modules..."
module load "${MODULES[@]}"

# The Alliance mpi4py module is tied to the module Python and can inject an
# incompatible site-packages directory into this Python 3.13 venv. Build/install
# mpi4py into the venv instead.
module unload mpi4py >/dev/null 2>&1 || true

# Loading modules can prepend their own Python. Activate after module changes so
# every Python build step uses the requested rsatb environment.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
hash -r

PYTHON="${VENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip)

if [[ "$("${PYTHON}" -c 'import sys; print(sys.executable)')" != "${PYTHON}" ]]; then
    echo "The active Python is not ${PYTHON}; module loading likely changed PATH." >&2
    exit 1
fi

find_one() {
    local root="$1"
    local name="$2"
    local path
    path="$(find "${root}" -name "${name}" | sort | head -n 1 || true)"
    if [[ -z "${path}" ]]; then
        echo "Could not find ${name} under ${root}" >&2
        exit 1
    fi
    printf '%s\n' "${path}"
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "${name} is not set. Did the required module fail to load?" >&2
        exit 1
    fi
}

prepend_path_var() {
    local name="$1"
    shift
    local value
    for value in "$@"; do
        if [[ -n "${value}" && -d "${value}" ]]; then
            export "${name}=${value}:${!name:-}"
        fi
    done
}

require_env MKLROOT
require_env EBROOTHDF5
require_env EBROOTFFTW
require_env EBROOTMETIS
require_env EBROOTPARMETIS
require_env EBROOTPRIMME
# libmatio is a hard build dependency of the bundled op2c (rescumat .mat reader).
require_env EBROOTMATIO

MKL_RT="$(find_one "${MKLROOT}" "libmkl_rt.so")"
HDF5_LIB="$(find_one "${EBROOTHDF5}" "libhdf5.so")"
FFTW_LIB="$(find_one "${EBROOTFFTW}" "libfftw3.so")"
METIS_LIB="$(find_one "${EBROOTMETIS}" "libmetis.so")"
PARMETIS_LIB="$(find_one "${EBROOTPARMETIS}" "libparmetis.so")"
PRIMME_LIB="$(find_one "${EBROOTPRIMME}" "libprimme.so")"
MATIO_LIB="$(find_one "${EBROOTMATIO}" "libmatio.so")"

for header in \
    "${EBROOTHDF5}/include/H5Cpp.h" \
    "${EBROOTFFTW}/include/fftw3.h" \
    "${EBROOTMETIS}/include/metis.h" \
    "${EBROOTPARMETIS}/include/parmetis.h" \
    "${EBROOTPRIMME}/include/primme.h" \
    "${EBROOTMATIO}/include/matio.h"; do
    if [[ ! -f "${header}" ]]; then
        echo "Could not find required header: ${header}" >&2
        exit 1
    fi
done

export CC="${CC:-$(command -v mpicc)}"
export CXX="${CXX:-$(command -v mpicxx)}"
export FC="${FC:-$(command -v mpifort || command -v mpif90 || true)}"
export MPICC="${MPICC:-${CC}}"
export MPICXX="${MPICXX:-${CXX}}"

if [[ -z "${CC}" || -z "${CXX}" ]]; then
    echo "Could not find mpicc/mpicxx after loading modules." >&2
    exit 1
fi

if [[ -z "${FC}" ]]; then
    echo "Could not find mpifort/mpif90 after loading modules." >&2
    exit 1
fi

export HDF5_ROOT="${EBROOTHDF5}"
export FFTW_ROOT="${EBROOTFFTW}"
export METIS_ROOT="${EBROOTMETIS}"
export PARMETIS_ROOT="${EBROOTPARMETIS}"
export PRIMME_ROOT="${EBROOTPRIMME}"

prepend_path_var CMAKE_PREFIX_PATH \
    "${MKLROOT}" \
    "${EBROOTHDF5}" \
    "${EBROOTFFTW}" \
    "${EBROOTFFTWMPI:-}" \
    "${EBROOTMETIS}" \
    "${EBROOTPARMETIS}" \
    "${EBROOTPRIMME}" \
    "${PROJECT_DIR}/deps/install"

prepend_path_var LD_LIBRARY_PATH \
    "$(dirname "${MKL_RT}")" \
    "$(dirname "${HDF5_LIB}")" \
    "$(dirname "${FFTW_LIB}")" \
    "$(dirname "${METIS_LIB}")" \
    "$(dirname "${PARMETIS_LIB}")" \
    "$(dirname "${PRIMME_LIB}")"

prepend_path_var CPATH \
    "${EBROOTHDF5}/include" \
    "${EBROOTFFTW}/include" \
    "${EBROOTMETIS}/include" \
    "${EBROOTPARMETIS}/include" \
    "${EBROOTPRIMME}/include"

echo "Using Python: $(${PYTHON} -c 'import sys; print(sys.executable)')"
echo "Python version: $(${PYTHON} -c 'import sys; print(sys.version.split()[0])')"
echo "Using cmake: $(command -v cmake) ($(cmake --version | head -n 1))"
echo "Using mpicc: ${CC}"
echo "Using mpicxx: ${CXX}"
echo "Using mpifort: ${FC}"
echo "Using MKL: ${MKL_RT}"
echo "Using HDF5: ${HDF5_LIB}"
echo "Using FFTW: ${FFTW_LIB}"
echo "Using METIS: ${METIS_LIB}"
echo "Using ParMETIS: ${PARMETIS_LIB}"
echo "Using PRIMME: ${PRIMME_LIB}"

PIP_FLAGS=()
if [[ "${RSATB_NO_INDEX:-0}" == "1" ]]; then
    PIP_FLAGS+=(--no-index)
fi

if [[ "${RSATB_SKIP_PIP_DEPS:-0}" != "1" ]]; then
    echo "Installing Python build/runtime dependencies..."
    "${PIP[@]}" install "${PIP_FLAGS[@]}" \
        scikit-build-core pybind11 cmake ninja fypp \
        numpy scipy matplotlib ase
fi

if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import sys

import mpi4py

prefix = os.path.realpath(sys.prefix)
location = os.path.realpath(mpi4py.__file__)
if not location.startswith(prefix + os.sep):
    raise SystemExit(f"mpi4py is outside the active venv: {location}")
PY
then
    echo "Installing mpi4py into the active venv for the loaded OpenMPI..."
    PIP_ONLY_BINARY=:none: PIP_NO_BINARY=mpi4py "${PIP[@]}" install \
        --no-cache-dir \
        --no-binary=mpi4py \
        mpi4py
fi

echo "Installing rsatb..."
cd "${PROJECT_DIR}"
"${PIP[@]}" install . --no-build-isolation -v \
    -C cmake.define.CMAKE_C_COMPILER="${CC}" \
    -C cmake.define.CMAKE_CXX_COMPILER="${CXX}" \
    -C cmake.define.CMAKE_Fortran_COMPILER="${FC}" \
    -C cmake.define.BLAS_BACKEND=MKL \
    -C cmake.define.MKL_RT_LIB="${MKL_RT}" \
    -C cmake.define.PARMETIS_LIB="${PARMETIS_LIB}" \
    -C cmake.define.METIS_LIB="${METIS_LIB}" \
    -C cmake.define.PRIMME_LIB="${PRIMME_LIB}" \
    -C cmake.define.USE_ILP64=OFF \
    -C cmake.define.VBCSR_BLAS_ILP64=OFF \
    -C cmake.define.RSATB_BUILD_TESTS=OFF

echo "Checking installed import..."
CHECK_DIR="$(mktemp -d)"
trap 'rm -rf "${CHECK_DIR}"' EXIT
cd "${CHECK_DIR}"
"${PYTHON}" - <<'PY'
import sys

import mpi4py
from mpi4py import MPI
import rsatb
import _rsatb

print("Python:", sys.executable)
print("mpi4py:", mpi4py.__version__, mpi4py.__file__)
print("MPI:", MPI.Get_library_version().splitlines()[0])
print("rsatb:", rsatb.__file__)
print("_rsatb:", _rsatb.__file__)
PY

echo "rsatb installation completed."
