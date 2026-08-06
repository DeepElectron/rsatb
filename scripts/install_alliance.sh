#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

path_prepend() {
    local name="$1"
    shift
    local value current
    current="${!name:-}"
    for value in "$@"; do
        if [[ -n "${value}" && -d "${value}" ]]; then
            case ":${current}:" in
                *":${value}:"*) ;;
                *)
                    if [[ -n "${current}" ]]; then
                        current="${value}:${current}"
                    else
                        current="${value}"
                    fi
                    ;;
            esac
        fi
    done
    export "${name}=${current}"
}

split_path_var() {
    local name="$1"
    local item
    local _parts=()
    if [[ -n "${!name:-}" ]]; then
        IFS=':' read -r -a _parts <<< "${!name}"
        for item in "${_parts[@]}"; do
            [[ -n "${item}" ]] && printf '%s\n' "${item}"
        done
    fi
}

init_module_command() {
    if type module >/dev/null 2>&1; then
        return 0
    fi

    local init
    for init in \
        /cvmfs/soft.computecanada.ca/config/profile/bash.sh \
        /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/share/Modules/init/bash; do
        if [[ -r "${init}" ]]; then
            # shellcheck disable=SC1090
            source "${init}"
            if type module >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    return 1
}

load_one_module() {
    local mod="$1"
    local fallback

    [[ -z "${mod}" ]] && return 0
    if module load "${mod}" >/dev/null 2>&1; then
        return 0
    fi

    fallback="${mod%%/*}"
    if [[ "${fallback}" != "${mod}" ]] && module load "${fallback}" >/dev/null 2>&1; then
        log "Loaded ${fallback} because ${mod} was not available."
        return 0
    fi

    die "could not load module '${mod}'. Set RSATB_MODULES to modules available on this cluster, or RSATB_MODULES=none if dependencies are already in your environment."
}

load_alliance_modules() {
    local modules_value modules mod

    if [[ "${RSATB_MODULES+x}" == "x" ]]; then
        modules_value="${RSATB_MODULES}"
    else
        # Versionless module names make the script portable across Alliance
        # clusters while still respecting each cluster's default toolchain.
        modules_value="StdEnv/2023 gcc openmpi cmake imkl hdf5-mpi fftw-mpi metis parmetis primme matio"
    fi

    case "${modules_value}" in
        ""|none|NONE|skip|SKIP)
            log "Skipping module loads because RSATB_MODULES=${modules_value:-<empty>}."
            return 0
            ;;
    esac

    if [[ -n "${RSATB_EXTRA_MODULES:-}" ]]; then
        modules_value="${modules_value} ${RSATB_EXTRA_MODULES}"
    fi

    if ! init_module_command; then
        die "could not initialize the module command. On Alliance, check that /cvmfs/soft.computecanada.ca is mounted, or set RSATB_MODULES=none and provide dependencies yourself."
    fi

    read -r -a modules <<< "${modules_value}"
    log "Loading modules: ${modules[*]}"
    for mod in "${modules[@]}"; do
        load_one_module "${mod}"
    done

    # Avoid mixing an Alliance mpi4py module built for the module Python with
    # this venv. The script installs/verifies mpi4py inside the active venv.
    module unload mpi4py >/dev/null 2>&1 || true
}

resolve_venv_dir() {
    local candidate
    if [[ -n "${RSATB_VENV:-}" ]]; then
        candidate="${RSATB_VENV}"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        candidate="${VIRTUAL_ENV}"
    elif [[ -x "$(dirname "${PROJECT_DIR}")/bin/python" ]]; then
        candidate="$(dirname "${PROJECT_DIR}")"
    elif [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
        candidate="${PROJECT_DIR}/.venv"
    elif [[ -x "${PWD}/.venv/bin/python" ]]; then
        candidate="${PWD}/.venv"
    else
        die "could not find the rsatb Python environment. Activate it first, or set RSATB_VENV=/path/to/env."
    fi

    candidate="$(cd "${candidate}" && pwd)"
    [[ -x "${candidate}/bin/python" ]] || die "could not find Python at ${candidate}/bin/python"
    printf '%s\n' "${candidate}"
}

find_file_in_roots() {
    local label="$1"
    shift
    local patterns=()
    while [[ "$#" -gt 0 && "$1" != "--" ]]; do
        patterns+=("$1")
        shift
    done
    [[ "$#" -gt 0 ]] || die "internal error while searching for ${label}: missing --"
    shift

    local roots=("$@")
    local root pattern expr=() path
    for root in "${roots[@]}"; do
        [[ -n "${root}" && -d "${root}" ]] || continue
        expr=()
        for pattern in "${patterns[@]}"; do
            if [[ "${#expr[@]}" -gt 0 ]]; then
                expr+=(-o)
            fi
            expr+=(-name "${pattern}")
        done
        path="$(find "${root}" \( -type f -o -type l \) \( "${expr[@]}" \) 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -n "${path}" ]]; then
            printf '%s\n' "${path}"
            return 0
        fi
    done

    die "could not find ${label}. Load the matching module or add its prefix to CMAKE_PREFIX_PATH."
}

find_file_in_roots_optional() {
    local label="$1"
    shift
    local patterns=()
    while [[ "$#" -gt 0 && "$1" != "--" ]]; do
        patterns+=("$1")
        shift
    done
    [[ "$#" -gt 0 ]] || die "internal error while searching for ${label}: missing --"
    shift

    local roots=("$@")
    local root pattern expr=() path
    for root in "${roots[@]}"; do
        [[ -n "${root}" && -d "${root}" ]] || continue
        expr=()
        for pattern in "${patterns[@]}"; do
            if [[ "${#expr[@]}" -gt 0 ]]; then
                expr+=(-o)
            fi
            expr+=(-name "${pattern}")
        done
        path="$(find "${root}" \( -type f -o -type l \) \( "${expr[@]}" \) 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -n "${path}" ]]; then
            printf '%s\n' "${path}"
            return 0
        fi
    done

    return 0
}

prefix_from_file() {
    local path="$1"
    case "${path}" in
        */include/*) printf '%s\n' "${path%%/include/*}" ;;
        */lib64/*) printf '%s\n' "${path%%/lib64/*}" ;;
        */lib/*) printf '%s\n' "${path%%/lib/*}" ;;
        *) dirname "${path}" ;;
    esac
}

collect_search_roots() {
    local root
    for root in \
        "${VENV_DIR}" \
        "${CONDA_PREFIX:-}" \
        "${MKLROOT:-}" \
        "${EBROOTIMKL:-}" \
        "${EBROOTHDF5:-}" \
        "${HDF5_ROOT:-}" \
        "${EBROOTFFTW:-}" \
        "${EBROOTFFTWMPI:-}" \
        "${FFTW_ROOT:-}" \
        "${EBROOTMETIS:-}" \
        "${METIS_ROOT:-}" \
        "${EBROOTPARMETIS:-}" \
        "${PARMETIS_ROOT:-}" \
        "${EBROOTPRIMME:-}" \
        "${PRIMME_ROOT:-}" \
        "${EBROOTMATIO:-}" \
        "${MATIO_ROOT:-}" \
        "${PROJECT_DIR}/deps/install"; do
        [[ -n "${root}" ]] && printf '%s\n' "${root}"
    done
    split_path_var RSATB_EXTRA_PREFIXES
    split_path_var CMAKE_PREFIX_PATH
    split_path_var CPATH
    split_path_var LIBRARY_PATH
    split_path_var LD_LIBRARY_PATH
}

unique_lines() {
    awk '!seen[$0]++'
}

require_command() {
    local name="$1"
    command -v "${name}" >/dev/null 2>&1 || die "could not find ${name} on PATH"
}

verify_native_dependencies() {
    local tmp_dir test_src test_bin status
    local blas_link_args=()

    tmp_dir="$(mktemp -d)"
    test_src="${tmp_dir}/rsatb_native_deps.cpp"
    test_bin="${tmp_dir}/rsatb_native_deps"

    cat > "${test_src}" <<'CPP'
#include <H5Cpp.h>
#include <fftw3.h>
#include <matio.h>
#include <metis.h>
#include <parmetis.h>
#include <primme.h>

int main() {
    H5::H5File h5("rsatb_native_deps.h5", H5F_ACC_TRUNC);

    fftw_complex in[1] = {{0.0, 0.0}};
    fftw_complex out[1] = {{0.0, 0.0}};
    fftw_plan plan = fftw_plan_dft_1d(1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (plan == nullptr) return 2;
    fftw_destroy_plan(plan);

    idx_t metis_options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(metis_options);

    primme_params primme;
    primme_initialize(&primme);
    primme_free(&primme);

    mat_t* mat = Mat_CreateVer("rsatb_native_deps.mat", nullptr, MAT_FT_DEFAULT);
    if (mat != nullptr) Mat_Close(mat);

    return 0;
}
CPP

    log "Checking native dependency headers, link flags, and runtime loader..."
    if [[ -n "${MKL_RT:-}" ]]; then
        blas_link_args+=("${MKL_RT}")
    else
        blas_link_args+=(-llapack -lblas)
    fi

    status=0
    "${CXX}" -std=c++17 \
        -I"$(dirname "${HDF5_HEADER}")" \
        -I"$(dirname "${FFTW_HEADER}")" \
        -I"$(dirname "${METIS_HEADER}")" \
        -I"$(dirname "${PARMETIS_HEADER}")" \
        -I"$(dirname "${PRIMME_HEADER}")" \
        -I"$(dirname "${MATIO_HEADER}")" \
        "${test_src}" -o "${test_bin}" \
        "${HDF5_CPP_LIB}" "${HDF5_LIB}" \
        "${FFTW_LIB}" \
        "${PARMETIS_LIB}" "${METIS_LIB}" \
        "${PRIMME_LIB}" \
        "${MATIO_LIB}" \
        "${blas_link_args[@]}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
        rm -rf "${tmp_dir}"
        die "native dependency link test failed. Check the module set, CMAKE_PREFIX_PATH, and LD_LIBRARY_PATH above."
    fi

    status=0
    (cd "${tmp_dir}" && "${test_bin}") || status=$?
    rm -rf "${tmp_dir}"
    [[ "${status}" -eq 0 ]] || die "native dependency runtime test failed with exit code ${status}"
}

check_installed_shared_objects() {
    local so output all_output label pattern i
    local import_check_dir
    local installed_extensions=()

    if ! command -v ldd >/dev/null 2>&1; then
        log "Skipping installed shared-object dependency check because ldd is not available."
        return 0
    fi

    import_check_dir="$(mktemp -d)"
    mapfile -t installed_extensions < <(cd "${import_check_dir}" && "${PYTHON}" - <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec("rsatb")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("could not locate installed rsatb package")

pkg_dir = pathlib.Path(next(iter(spec.submodule_search_locations)))
for path in sorted(pkg_dir.glob("*.so")):
    print(path)
PY
)
    rm -rf "${import_check_dir}"

    [[ "${#installed_extensions[@]}" -gt 0 ]] || die "could not find installed rsatb extension modules"

    all_output=""
    for so in "${installed_extensions[@]}"; do
        log "Checking linked libraries for ${so}"
        output="$(ldd "${so}" 2>&1)" || die "ldd failed for ${so}"
        if grep -q "not found" <<< "${output}"; then
            printf '%s\n' "${output}" >&2
            die "unresolved shared library dependency in ${so}"
        fi
        all_output+="${output}"$'\n'
    done

    local required_links=(
        "HDF5" "libhdf5"
        "HDF5 C++" "libhdf5_cpp"
        "MATIO" "libmatio"
        "MPI" "libmpi"
        "ParMETIS" "libparmetis"
        "PRIMME" "libprimme"
    )
    if [[ "${RSATB_BLAS_BACKEND}" == "MKL" ]]; then
        required_links+=("MKL/FFTW interface" "libmkl_rt")
    else
        required_links+=("FFTW" "libfftw3")
    fi
    for ((i = 0; i < ${#required_links[@]}; i += 2)); do
        label="${required_links[i]}"
        pattern="${required_links[i + 1]}"
        if ! grep -Eq "${pattern}" <<< "${all_output}"; then
            die "installed extensions are not linked against ${label}; this usually means CMake found the wrong dependency set"
        fi
    done

    if [[ "${RSATB_BLAS_BACKEND}" == "MKL" ]] && grep -Eq "libfftw3" <<< "${all_output}"; then
        die "MKL build is also linked against external FFTW; this mixes FFTW providers and can crash at runtime"
    fi

    # METIS is allowed to be linked statically by ParMETIS on Alliance, so it
    # may not appear in ldd output. verify_native_dependencies checked it
    # directly before the package build.
    if ! grep -Eq "libmetis" <<< "${all_output}"; then
        log "METIS does not appear as a shared dependency; assuming it was linked statically after the native smoke test."
    fi
}

ensure_git_external() {
    local name="$1"
    local dir="$2"
    local url="$3"
    local ref="$4"
    local required_path="$5"

    if [[ "${RSATB_UPDATE_EXTERNALS:-1}" != "1" ]]; then
        [[ -e "${dir}/${required_path}" ]] || die "${name} is missing ${required_path}. Set RSATB_UPDATE_EXTERNALS=1 or update ${dir} manually."
        log "Using existing ${name} checkout because RSATB_UPDATE_EXTERNALS=0."
        return 0
    fi

    require_command git
    log "Synchronizing ${name} external dependency to ${ref}..."

    if [[ -d "${dir}/.git" ]]; then
        git -C "${dir}" fetch --depth 1 origin "${ref}" || git -C "${dir}" fetch origin "${ref}"
        git -C "${dir}" switch --detach FETCH_HEAD || git -C "${dir}" checkout --detach FETCH_HEAD
    else
        if [[ -d "${dir}" ]] && find "${dir}" -mindepth 1 -maxdepth 1 | read -r; then
            die "${dir} exists but is not a usable git checkout and is missing ${required_path}"
        fi
        git clone --depth 1 "${url}" "${dir}"
        git -C "${dir}" fetch --depth 1 origin "${ref}" || git -C "${dir}" fetch origin "${ref}"
        git -C "${dir}" switch --detach FETCH_HEAD || git -C "${dir}" checkout --detach FETCH_HEAD
    fi

    [[ -e "${dir}/${required_path}" ]] || die "${name} still does not contain ${required_path} after updating"
}

load_alliance_modules

VENV_DIR="$(resolve_venv_dir)"

# Loading modules can prepend their own Python. Activate after module changes so
# every Python build step uses the requested rsatb environment.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
hash -r

PYTHON="${VENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip)

if [[ "$("${PYTHON}" -c 'import sys; print(sys.executable)')" != "${PYTHON}" ]]; then
    die "the active Python is not ${PYTHON}; module loading likely changed PATH"
fi

export CC="${CC:-$(command -v mpicc || true)}"
export CXX="${CXX:-$(command -v mpicxx || true)}"
export FC="${FC:-$(command -v mpifort || command -v mpif90 || true)}"
export MPICC="${MPICC:-${CC}}"
export MPICXX="${MPICXX:-${CXX}}"

[[ -n "${CC}" ]] || die "could not find mpicc after loading modules"
[[ -n "${CXX}" ]] || die "could not find mpicxx after loading modules"
[[ -n "${FC}" ]] || die "could not find mpifort/mpif90 after loading modules"

PIP_FLAGS=()
if [[ "${RSATB_NO_INDEX:-0}" == "1" ]]; then
    PIP_FLAGS+=(--no-index)
fi

if [[ "${RSATB_SKIP_PIP_DEPS:-0}" != "1" ]]; then
    log "Installing Python build/runtime dependencies..."
    "${PIP[@]}" install "${PIP_FLAGS[@]}" \
        scikit-build-core pybind11 cmake ninja \
        numpy scipy ase
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
    log "Installing mpi4py into the active venv for the loaded MPI..."
    "${PIP[@]}" install "${PIP_FLAGS[@]}" \
        --no-cache-dir \
        --no-binary=mpi4py \
        mpi4py
fi

require_command cmake

mapfile -t SEARCH_ROOTS < <(collect_search_roots | unique_lines)

RSATB_BLAS_BACKEND="${RSATB_BLAS_BACKEND:-MKL}"
MKL_RT=""
if [[ "${RSATB_BLAS_BACKEND}" == "MKL" ]]; then
    MKL_RT="$(find_file_in_roots "libmkl_rt" "libmkl_rt.so" "libmkl_rt.so.*" -- "${SEARCH_ROOTS[@]}")"
    MKL_PREFIX="$(prefix_from_file "${MKL_RT}")"
    export MKLROOT="${MKLROOT:-${MKL_PREFIX}}"
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
else
    MKL_PREFIX=""
fi

HDF5_HEADER="$(find_file_in_roots "H5Cpp.h" "H5Cpp.h" -- "${SEARCH_ROOTS[@]}")"
FFTW_HEADER="$(find_file_in_roots "fftw3.h" "fftw3.h" -- "${SEARCH_ROOTS[@]}")"
METIS_HEADER="$(find_file_in_roots "metis.h" "metis.h" -- "${SEARCH_ROOTS[@]}")"
PARMETIS_HEADER="$(find_file_in_roots "parmetis.h" "parmetis.h" -- "${SEARCH_ROOTS[@]}")"
PRIMME_HEADER="$(find_file_in_roots "primme.h" "primme.h" -- "${SEARCH_ROOTS[@]}")"
MATIO_HEADER="$(find_file_in_roots "matio.h" "matio.h" -- "${SEARCH_ROOTS[@]}")"

HDF5_LIB="$(find_file_in_roots "libhdf5" "libhdf5.so" "libhdf5.so.*" -- "${SEARCH_ROOTS[@]}")"
HDF5_CPP_LIB="$(find_file_in_roots "libhdf5_cpp" "libhdf5_cpp.so" "libhdf5_cpp.so.*" -- "${SEARCH_ROOTS[@]}")"
FFTW_LIB="$(find_file_in_roots "libfftw3" "libfftw3.so" "libfftw3.so.*" -- "${SEARCH_ROOTS[@]}")"
FFTW_MPI_LIB="$(find_file_in_roots_optional "libfftw3_mpi" "libfftw3_mpi.so" "libfftw3_mpi.so.*" -- "${SEARCH_ROOTS[@]}")"
METIS_LIB="$(find_file_in_roots "libmetis" "libmetis.so" "libmetis.so.*" -- "${SEARCH_ROOTS[@]}")"
PARMETIS_LIB="$(find_file_in_roots "libparmetis" "libparmetis.so" "libparmetis.so.*" -- "${SEARCH_ROOTS[@]}")"
PRIMME_LIB="$(find_file_in_roots "libprimme" "libprimme.so" "libprimme.so.*" -- "${SEARCH_ROOTS[@]}")"
MATIO_LIB="$(find_file_in_roots "libmatio" "libmatio.so" "libmatio.so.*" -- "${SEARCH_ROOTS[@]}")"

if [[ "${RSATB_BLAS_BACKEND}" == "MKL" ]]; then
    FFTW_HEADER="$(find_file_in_roots "MKL fftw3.h" "fftw3.h" -- "${MKLROOT}" "${MKL_PREFIX}")"
    FFTW_LIB="${MKL_RT}"
fi

HDF5_PREFIX="$(prefix_from_file "${HDF5_LIB}")"
HDF5_CPP_PREFIX="$(prefix_from_file "${HDF5_CPP_LIB}")"
FFTW_PREFIX="$(prefix_from_file "${FFTW_LIB}")"
if [[ -n "${FFTW_MPI_LIB}" ]]; then
    FFTW_MPI_PREFIX="$(prefix_from_file "${FFTW_MPI_LIB}")"
else
    FFTW_MPI_PREFIX=""
fi
METIS_PREFIX="$(prefix_from_file "${METIS_LIB}")"
PARMETIS_PREFIX="$(prefix_from_file "${PARMETIS_LIB}")"
PRIMME_PREFIX="$(prefix_from_file "${PRIMME_LIB}")"
MATIO_PREFIX="$(prefix_from_file "${MATIO_LIB}")"

export HDF5_ROOT="${HDF5_ROOT:-${HDF5_PREFIX}}"
export FFTW_ROOT="${FFTW_ROOT:-${FFTW_PREFIX}}"
export METIS_ROOT="${METIS_ROOT:-${METIS_PREFIX}}"
export PARMETIS_ROOT="${PARMETIS_ROOT:-${PARMETIS_PREFIX}}"
export PRIMME_ROOT="${PRIMME_ROOT:-${PRIMME_PREFIX}}"
export MATIO_ROOT="${MATIO_ROOT:-${MATIO_PREFIX}}"

prefixes=(
    "${MKL_PREFIX:-}"
    "${HDF5_PREFIX}"
    "${HDF5_CPP_PREFIX}"
    "${FFTW_PREFIX}"
    "${FFTW_MPI_PREFIX}"
    "$(prefix_from_file "${HDF5_HEADER}")"
    "$(prefix_from_file "${FFTW_HEADER}")"
    "${METIS_PREFIX}"
    "${PARMETIS_PREFIX}"
    "${PRIMME_PREFIX}"
    "${MATIO_PREFIX}"
    "${PROJECT_DIR}/deps/install"
)

lib_dirs=()
include_dirs=()
for prefix in "${prefixes[@]}"; do
    [[ -n "${prefix}" ]] || continue
    lib_dirs+=("${prefix}/lib" "${prefix}/lib64")
    include_dirs+=("${prefix}/include")
done

path_prepend CMAKE_PREFIX_PATH "${prefixes[@]}"
path_prepend LD_LIBRARY_PATH "${lib_dirs[@]}"
path_prepend LIBRARY_PATH "${lib_dirs[@]}"
path_prepend CPATH "${include_dirs[@]}"

log "Using Python: $(${PYTHON} -c 'import sys; print(sys.executable)')"
log "Python version: $(${PYTHON} -c 'import sys; print(sys.version.split()[0])')"
log "Using cmake: $(command -v cmake) ($(cmake --version | head -n 1))"
log "Using mpicc: ${CC}"
log "Using mpicxx: ${CXX}"
log "Using mpifort: ${FC}"
log "Using BLAS backend: ${RSATB_BLAS_BACKEND}"
[[ -n "${MKL_RT}" ]] && log "Using MKL: ${MKL_RT}"
[[ -n "${MKL_RT}" ]] && log "Using MKL threading layer: ${MKL_THREADING_LAYER}"
log "Using HDF5: ${HDF5_LIB}"
log "Using HDF5 C++: ${HDF5_CPP_LIB}"
if [[ "${RSATB_BLAS_BACKEND}" == "MKL" ]]; then
    log "Using FFTW interface: ${FFTW_HEADER} via ${FFTW_LIB}"
else
    log "Using FFTW: ${FFTW_LIB}"
fi
[[ -n "${FFTW_MPI_LIB}" ]] && log "Using FFTW MPI: ${FFTW_MPI_LIB}"
log "Using METIS: ${METIS_LIB}"
log "Using ParMETIS: ${PARMETIS_LIB}"
log "Using PRIMME: ${PRIMME_LIB}"
log "Using MATIO: ${MATIO_LIB}"

verify_native_dependencies

CMAKE_CONFIG=(
    -C cmake.define.CMAKE_C_COMPILER="${CC}"
    -C cmake.define.CMAKE_CXX_COMPILER="${CXX}"
    -C cmake.define.CMAKE_Fortran_COMPILER="${FC}"
    -C cmake.define.BLAS_BACKEND="${RSATB_BLAS_BACKEND}"
    -C cmake.define.HDF5_ROOT="${HDF5_PREFIX}"
    -C cmake.define.PARMETIS_ROOT="${PARMETIS_PREFIX}"
    -C cmake.define.METIS_ROOT="${METIS_PREFIX}"
    -C cmake.define.PRIMME_ROOT="${PRIMME_PREFIX}"
    -C cmake.define.MATIO_ROOT="${MATIO_PREFIX}"
    -C cmake.define.PARMETIS_LIB="${PARMETIS_LIB}"
    -C cmake.define.METIS_LIB="${METIS_LIB}"
    -C cmake.define.PARMETIS_INCLUDE_DIR="$(dirname "${PARMETIS_HEADER}")"
    -C cmake.define.METIS_INCLUDE_DIR="$(dirname "${METIS_HEADER}")"
    -C cmake.define.PRIMME_LIB="${PRIMME_LIB}"
    -C cmake.define.MATIO_INCLUDE_DIR="$(dirname "${MATIO_HEADER}")"
    -C cmake.define.MATIO_LIBRARY="${MATIO_LIB}"
    -C cmake.define.RSATB_MATIO_INCLUDE_DIR="$(dirname "${MATIO_HEADER}")"
    -C cmake.define.RSATB_MATIO_LIBRARY="${MATIO_LIB}"
    -C cmake.define.USE_ILP64=OFF
    -C cmake.define.VBCSR_BLAS_ILP64=OFF
    -C cmake.define.RSATB_BUILD_TESTS=OFF
)

if [[ "${RSATB_BLAS_BACKEND}" == "MKL" ]]; then
    CMAKE_CONFIG+=(-C cmake.define.MKL_RT_LIB="${MKL_RT}")
else
    CMAKE_CONFIG+=(-C cmake.define.FFTW_ROOT="${FFTW_PREFIX}")
fi

ensure_git_external \
    "VBCSR" \
    "${PROJECT_DIR}/rsatb/extern/vbcsr" \
    "https://github.com/floatingCatty/vbcsr.git" \
    "${RSATB_VBCSR_REF:-main}" \
    "vbcsr/core/atomic/atomic_data.hpp"

log "Installing rsatb..."
cd "${PROJECT_DIR}"
"${PIP[@]}" install . --no-build-isolation -v "${CMAKE_CONFIG[@]}"

check_installed_shared_objects

log "Checking installed import..."
CHECK_DIR="$(mktemp -d)"
trap 'rm -rf "${CHECK_DIR}"' EXIT
cd "${CHECK_DIR}"
"${PYTHON}" - <<'PY'
import sys

import mpi4py
from mpi4py import MPI
import rsatb
import _rsatb
from rsatb import RSATB, SetupOptions, Structure

print("Python:", sys.executable)
print("mpi4py:", mpi4py.__version__, mpi4py.__file__)
print("MPI:", MPI.Get_library_version().splitlines()[0])
print("rsatb:", rsatb.__file__)
print("_rsatb:", _rsatb.__file__)
print("native API:", RSATB.__name__, SetupOptions.__name__, Structure.__name__)
PY

log "rsatb installation completed."
