cmake_minimum_required(VERSION 3.15...4.0)
project(rsatb LANGUAGES CXX Fortran C)

# --- Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set CONDA_PREFIX if available
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options ---
option(USE_OPENMP "Use OpenMP" ON)
set(BLAS_BACKEND "Generic" CACHE STRING "BLAS Backend (MKL, OpenBLAS, Generic)")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS "MKL" "OpenBLAS" "Generic")
option(USE_ILP64 "Use 64-bit integers for BLAS/LAPACK" OFF)
option(RSATB_BUILD_TESTS "Build rsatb C++ test executables" ON)

# Location of the test data tree, baked into the test binaries (see
# test_data_path.hpp). Overridable at runtime with the same-named env var.
set(RSATB_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test/data"
    CACHE PATH "Directory containing the rsatb test data tree")
add_compile_definitions(RSATB_TEST_DATA_DIR="${RSATB_TEST_DATA_DIR}")

# --- Dependencies ---
include(FetchContent)

# Set MPI_HOME to help FindMPI with Conda
if(DEFINED CONDA_PREFIX)
    set(MPI_HOME ${CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH ${CONDA_PREFIX})
endif()

# MPI
find_package(MPI REQUIRED)

# Threads
find_package(Threads REQUIRED)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# BLAS/LAPACK
if(BLAS_BACKEND STREQUAL "MKL")
    # Use mkl_rt (Single Dynamic Library) to avoid threading layer conflicts.
    # Individual MKL libs (mkl_gf_lp64 + mkl_gnu_thread) lock in GNU OpenMP,
    # which conflicts if any consumer uses Intel OpenMP (mkl_intel_thread).
    # mkl_rt provides a single entry point with runtime-configurable threading.
    find_library(MKL_RT_LIB NAMES mkl_rt mkl_rt.2 mkl_rt.1
                 HINTS $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64 $ENV{CONDA_PREFIX}/lib)
    if(NOT MKL_RT_LIB)
        message(FATAL_ERROR "BLAS_BACKEND=MKL but libmkl_rt not found. "
                "Set BLAS_BACKEND=Generic, or install MKL.")
    endif()
    message(STATUS "Using MKL SDL: ${MKL_RT_LIB}")

    # Create BLAS/LAPACK targets pointing to mkl_rt
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES "${MKL_RT_LIB}")
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    set_target_properties(LAPACK::LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${MKL_RT_LIB}")
    set(BLAS_LIBRARIES ${MKL_RT_LIB})
    set(BLAS_FOUND TRUE)

    if(USE_ILP64)
        add_compile_definitions(MKL_ILP64)
        set(RSATB_USE_ILP64 ON)
    else()
        set(RSATB_USE_ILP64 OFF)
    endif()
    add_compile_definitions(__MKL RSATB_USE_MKL VBCSR_USE_MKL)
    if(DEFINED ENV{MKLROOT})
        include_directories($ENV{MKLROOT}/include)
    endif()
elseif(BLAS_BACKEND STREQUAL "OpenBLAS")
    set(BLA_VENDOR "OpenBLAS")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_compile_definitions(__OPENBLAS RSATB_USE_OPENBLAS VBCSR_USE_OPENBLAS)
else()
    # Generic
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# Generate config.hpp
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/calc/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/calc/config.hpp"
)
# Add binary dir to include path so config.hpp is found
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# FFTW
find_package(FFTW REQUIRED)
add_compile_definitions(__FFTW)
if(NOT TARGET FFTW::FFTW)
    add_library(FFTW::FFTW INTERFACE IMPORTED)
    set_target_properties(FFTW::FFTW PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" 
        INTERFACE_LINK_LIBRARIES     "${FFTW_LIBRARIES}")
endif()
include_directories(${FFTW_INCLUDE_DIRS})

# Python & Pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sys, sysconfig; print(f'{sys.prefix}/include/python{sysconfig.get_python_version()}')"
  OUTPUT_VARIABLE PYTHON_ENV_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import mpi4py, sys; sys.stdout.write(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "mpi4py include: ${MPI4PY_INCLUDE_DIR}")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# --- Local / Fetched Dependencies ---

# GTest (FetchContent)
if(RSATB_BUILD_TESTS)
  find_package(GTest)
  if(NOT GTest_FOUND)
    message(STATUS "GTest not found, fetching from source...")
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
  endif()
endif()

# 1. ParMETIS & METIS (Required System/Conda)
find_library(PARMETIS_LIB parmetis HINTS ${CONDA_PREFIX}/lib)
find_library(METIS_LIB metis HINTS ${CONDA_PREFIX}/lib)

if(NOT PARMETIS_LIB OR NOT METIS_LIB)
    message(FATAL_ERROR "ParMETIS or METIS not found! Please install them (e.g., 'conda install -c conda-forge parmetis metis') or set CMAKE_PREFIX_PATH.")
endif()

# 2. DBCSR (Use existing sources in extern/, or fetch if missing)
set(RSATB_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/install")
list(APPEND CMAKE_PREFIX_PATH "${RSATB_DEPS_DIR}")

# Disable git updates for FetchContent (faster rebuilds)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# find_package(DBCSR 2.0.0 CONFIG QUIET HINTS "${RSATB_DEPS_DIR}/lib/cmake/dbcsr")
# if(NOT DBCSR_FOUND)
#     set(DBCSR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/dbcsr")
#     
#     if(EXISTS "${DBCSR_SOURCE_DIR}/CMakeLists.txt")
#         message(STATUS "DBCSR sources found in extern/dbcsr, using existing sources")
#         # Add subdirectory (works whether freshly fetched or already exists)
#         set(WITH_C_API ON CACHE BOOL "" FORCE)
#         add_subdirectory(${DBCSR_SOURCE_DIR} ${CMAKE_BINARY_DIR}/extern/dbcsr)
#     else()
#         message(STATUS "DBCSR not found, fetching from GitHub...")
#         FetchContent_Declare(
#             dbcsr
#             GIT_REPOSITORY https://github.com/cp2k/dbcsr.git
#             GIT_TAG        v2.6.0
#             SOURCE_DIR     "${DBCSR_SOURCE_DIR}"
#             BINARY_DIR     "${CMAKE_BINARY_DIR}/extern/dbcsr"
#             PATCH_COMMAND  sed -i "s|\${CMAKE_SOURCE_DIR}/DBCSR.md|\${CMAKE_CURRENT_SOURCE_DIR}/../DBCSR.md|g" docs/CMakeLists.txt
#         )
#         FetchContent_MakeAvailable(dbcsr)
#     endif()
#     
#     # Create alias
#     if(TARGET dbcsr_c AND NOT TARGET DBCSR::dbcsr_c)
#          add_library(DBCSR::dbcsr_c ALIAS dbcsr_c) 
#     elseif(TARGET dbcsr AND NOT TARGET DBCSR::dbcsr_c)
#          add_library(DBCSR::dbcsr_c ALIAS dbcsr)
#     endif()
# endif()

# 3. PRIMME (Use existing sources in extern/, or fetch if missing)
find_library(PRIMME_LIB primme HINTS ${CONDA_PREFIX}/lib "${RSATB_DEPS_DIR}/lib")
if(NOT PRIMME_LIB)
    set(PRIMME_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/primme")
    
    if(EXISTS "${PRIMME_SOURCE_DIR}/include/primme.h")
        message(STATUS "PRIMME sources found in extern/primme, using existing sources")
    else()
        message(STATUS "PRIMME not found, fetching from GitHub...")
        FetchContent_Declare(
            primme
            GIT_REPOSITORY https://github.com/primme/primme.git
            GIT_TAG        v3.2
            SOURCE_DIR     "${PRIMME_SOURCE_DIR}"
            BINARY_DIR     "${CMAKE_BINARY_DIR}/extern/primme"
        )
        FetchContent_MakeAvailable(primme)
    endif()
    
    # PRIMME has no CMakeLists.txt, so we define the target manually
    if(NOT TARGET primme)
        file(GLOB_RECURSE PRIMME_SOURCES "${PRIMME_SOURCE_DIR}/src/*.c")
        add_library(primme STATIC ${PRIMME_SOURCES})
        target_include_directories(primme PUBLIC 
            "$<BUILD_INTERFACE:${PRIMME_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${PRIMME_SOURCE_DIR}/src/include>"
        )
        target_link_libraries(primme PRIVATE BLAS::BLAS LAPACK::LAPACK)
        target_compile_definitions(primme PRIVATE F77UNDERSCORE)
    endif()
    set(PRIMME_LIB primme)
else()
    message(STATUS "PRIMME library found: ${PRIMME_LIB}")
endif()

# 4. VBCSR (Use existing sources in extern/, or fetch if missing)
if(USE_ILP64)
    set(VBCSR_BLAS_ILP64 ON CACHE BOOL "Use 64-bit integers for BLAS/LAPACK interfaces" FORCE)
else()
    set(VBCSR_BLAS_ILP64 OFF CACHE BOOL "Use 64-bit integers for BLAS/LAPACK interfaces" FORCE)
endif()

set(VBCSR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/vbcsr")

if(EXISTS "${VBCSR_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "VBCSR sources found in extern/vbcsr, using existing sources")
    # Add subdirectory (works whether freshly fetched or already exists)
    add_subdirectory(${VBCSR_SOURCE_DIR} ${CMAKE_BINARY_DIR}/extern/vbcsr)

    # Fix include path for namespaced includes (vbcsr/core/...)
    if(TARGET vbcsr)
        target_include_directories(vbcsr INTERFACE "$<BUILD_INTERFACE:${VBCSR_SOURCE_DIR}>")
    endif()
else()
    message(STATUS "VBCSR not found, fetching from GitHub...")
    FetchContent_Declare(
        vbcsr
        GIT_REPOSITORY https://github.com/floatingCatty/vbcsr.git
        GIT_TAG        main
        SOURCE_DIR     "${VBCSR_SOURCE_DIR}"
        BINARY_DIR     "${CMAKE_BINARY_DIR}/extern/vbcsr"
    )
    FetchContent_MakeAvailable(vbcsr)
endif()

# 5. Op2c (Use existing sources in extern/, or fetch if missing)
#
# Op2c hard-requires libmatio for its rescumat *_AtomicData.mat reader. rsatb
# never reads that format, but op2c compiles io/rescumat_mat.cpp into nao_core
# unconditionally and links MATIO::MATIO PUBLIC, so it is a build dependency of
# rsatb too. Check it here: op2c's own FATAL_ERROR points users at the rescupp
# environment, which is misleading in an rsatb build.
find_path(RSATB_MATIO_INCLUDE_DIR NAMES matio.h HINTS ${CONDA_PREFIX}/include)
find_library(RSATB_MATIO_LIBRARY NAMES matio HINTS ${CONDA_PREFIX}/lib)
if(NOT RSATB_MATIO_INCLUDE_DIR OR NOT RSATB_MATIO_LIBRARY)
    message(FATAL_ERROR
        "libmatio not found, but it is required to build the bundled op2c.\n"
        "  conda:           conda install -c conda-forge libmatio\n"
        "  Compute Canada:  module load matio   (or build it into your venv)\n"
        "Searched CONDA_PREFIX=${CONDA_PREFIX}.")
endif()
message(STATUS "Found libmatio (required by op2c): ${RSATB_MATIO_LIBRARY}")

set(OP2C_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/op2c")

if(EXISTS "${OP2C_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "Op2c sources found in extern/op2c, using existing sources")
    add_subdirectory(${OP2C_SOURCE_DIR} ${CMAKE_BINARY_DIR}/extern/op2c)
else()
    message(STATUS "Op2c not found, fetching from GitHub...")
    FetchContent_Declare(
        op2c
        GIT_REPOSITORY https://github.com/floatingCatty/op2c.git
        GIT_TAG        main
        SOURCE_DIR     "${OP2C_SOURCE_DIR}"
        BINARY_DIR     "${CMAKE_BINARY_DIR}/extern/op2c"
    )
    FetchContent_MakeAvailable(op2c)
endif()

list(APPEND math_libs FFTW::FFTW LAPACK::LAPACK BLAS::BLAS)

# --- Subdirectories ---
add_subdirectory(model)
add_subdirectory(io)

# --- Targets ---

# Common Link Libraries
set(RSATB_LINK_LIBS
    vbcsr
    io
    nao_int2c
    nao_pseudo
    nao_core
    nao_math
    nao_utils
    MPI::MPI_CXX
    # DBCSR::dbcsr_c
    HDF5::HDF5
    FFTW::FFTW
    LAPACK::LAPACK
    ${PRIMME_LIB}
    ${PARMETIS_LIB}
    ${METIS_LIB}
)

if(USE_OPENMP)
    list(APPEND RSATB_LINK_LIBS OpenMP::OpenMP_CXX)
endif()

# Common Include Directories
set(RSATB_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    model
    model/sktb
    extern
    extern/op2c/src
    extern/op2c/src/utils/container
    extern/vbcsr
    extern/vbcsr/vbcsr
)

if(DEFINED RSATB_DEPS_DIR)
    list(APPEND RSATB_INCLUDE_DIRS "${RSATB_DEPS_DIR}/include")
endif()
if(DEFINED CONDA_PREFIX)
    list(APPEND RSATB_INCLUDE_DIRS "${CONDA_PREFIX}/include")
endif()


if(RSATB_BUILD_TESTS)
    # Executable: test_rsatb
    add_executable(test_rsatb test_rsatb.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_rsatb PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_rsatb PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_fftcheb_transport
    add_executable(test_fftcheb_transport calc/test/test_fftcheb_transport.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_fftcheb_transport PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_fftcheb_transport PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_kpm_extensions
    add_executable(test_kpm_extensions calc/test/test_kpm_extensions.cpp)
    target_include_directories(test_kpm_extensions PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_kpm_extensions PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_msd
    add_executable(test_msd calc/test/test_msd.cpp)
    target_include_directories(test_msd PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_msd PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_operator_api
    add_executable(test_operator_api test_operator_api.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_operator_api PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_operator_api PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_hdf5_io
    add_executable(test_hdf5_io io/test.cpp)
    target_include_directories(test_hdf5_io PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_hdf5_io PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_preconditioner
    add_executable(test_preconditioner calc/test/test_preconditioner.cpp)
    target_include_directories(test_preconditioner PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_preconditioner PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Standalone MPI test drivers (own main(), not GTest). Registered so they
    # are compiled by the normal build and cannot silently bit-rot again.
    foreach(_driver test_graphmf test_solver test_dos)
        add_executable(${_driver} calc/test/${_driver}.cpp)
        target_include_directories(${_driver} PRIVATE ${RSATB_INCLUDE_DIRS})
        target_link_libraries(${_driver} PRIVATE ${RSATB_LINK_LIBS})
    endforeach()
endif()

# Python Module: _rsatb
pybind11_add_module(_rsatb pybind_rsatb.cpp)
# Prevent MKL threading symbols from leaking into the shared object.
# This avoids conflicts when numpy or other packages bring their own MKL.
if(UNIX AND NOT APPLE)
    target_link_options(_rsatb PRIVATE
        "-Wl,--exclude-libs,libmkl_intel_thread.so"
        "-Wl,--exclude-libs,libiomp5.so")
endif()
target_include_directories(_rsatb BEFORE PRIVATE ${PYTHON_ENV_INCLUDE_DIR})
target_include_directories(_rsatb PRIVATE ${RSATB_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${MPI4PY_INCLUDE_DIR})
target_link_libraries(_rsatb PRIVATE ${RSATB_LINK_LIBS})

# Install
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS _rsatb DESTINATION .)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/calc/config.hpp" DESTINATION include/rsatb/calc)

# Test: test_isqrtm_vbcsr
# add_executable(test_isqrtm_vbcsr calc/test/test_isqrtm_vbcsr.cpp)
# target_include_directories(test_isqrtm_vbcsr PRIVATE ${RSATB_INCLUDE_DIRS})
# target_link_libraries(test_isqrtm_vbcsr PRIVATE ${RSATB_LINK_LIBS})

# Benchmark: benchmark_isqrtm_heavy
# add_executable(benchmark_isqrtm_heavy calc/test/benchmark_isqrtm_heavy.cpp)
# target_include_directories(benchmark_isqrtm_heavy PRIVATE ${RSATB_INCLUDE_DIRS})
# target_link_libraries(benchmark_isqrtm_heavy PRIVATE ${RSATB_LINK_LIBS})

# add_executable(test_lapack_query calc/test/test_lapack_query.cpp)
# target_include_directories(test_lapack_query PRIVATE ${RSATB_INCLUDE_DIRS})
# target_link_libraries(test_lapack_query PRIVATE ${RSATB_LINK_LIBS})
