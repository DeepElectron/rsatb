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
set(BLAS_BACKEND "Auto" CACHE STRING "BLAS Backend (Auto, MKL, OpenBLAS, Generic)")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS "Auto" "MKL" "OpenBLAS" "Generic")
option(USE_ILP64 "Use 64-bit integers for BLAS/LAPACK" OFF)
option(RSATB_BUILD_TESTS "Build rsatb C++ test executables" ON)
option(RSATB_ENABLE_X86_SIMD "Compile rsatb backend with explicit x86 AVX2/FMA flags when supported" OFF)

# Location of the test data tree, baked into the test binaries (see
# test_data_path.hpp). Overridable at runtime with the same-named env var.
set(RSATB_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test/data"
    CACHE PATH "Directory containing the rsatb test data tree")
add_compile_definitions(RSATB_TEST_DATA_DIR="${RSATB_TEST_DATA_DIR}")

# --- Dependencies ---
include(FetchContent)
include(CMakeParseArguments)

set(RSATB_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/install"
    CACHE PATH "Prefix containing rsatb-built native dependencies")
list(APPEND CMAKE_PREFIX_PATH "${RSATB_DEPS_DIR}")

function(rsatb_import_header_library target cache_prefix header)
    set(options REQUIRED)
    set(one_value_args)
    set(multi_value_args LIB_NAMES ROOT_HINTS LINK_LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(TARGET ${target})
        return()
    endif()

    set(_include_hints)
    set(_library_hints)
    foreach(_root IN LISTS ARG_ROOT_HINTS)
        if(_root)
            list(APPEND _include_hints "${_root}/include")
            list(APPEND _library_hints "${_root}/lib" "${_root}/lib64")
        endif()
    endforeach()

    find_path(${cache_prefix}_INCLUDE_DIR NAMES "${header}" HINTS ${_include_hints})
    find_library(${cache_prefix}_LIBRARY NAMES ${ARG_LIB_NAMES} HINTS ${_library_hints})

    if(NOT ${cache_prefix}_INCLUDE_DIR OR NOT ${cache_prefix}_LIBRARY)
        if(ARG_REQUIRED)
            message(FATAL_ERROR
                "${cache_prefix} not found. Expected header ${header} and libraries ${ARG_LIB_NAMES}. "
                "Set CMAKE_PREFIX_PATH or ${cache_prefix}_ROOT to a prefix containing include/ and lib/.")
        endif()
        return()
    endif()

    add_library(${target} UNKNOWN IMPORTED)
    set_target_properties(${target} PROPERTIES
        IMPORTED_LOCATION "${${cache_prefix}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${${cache_prefix}_INCLUDE_DIR}")
    if(ARG_LINK_LIBRARIES)
        set_target_properties(${target} PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ARG_LINK_LIBRARIES}")
    endif()
endfunction()

function(rsatb_define_hdf5_target)
    if(TARGET RSATB::HDF5)
        return()
    endif()

    add_library(rsatb_hdf5 INTERFACE)
    if(TARGET hdf5::hdf5 AND TARGET hdf5::hdf5_cpp)
        target_link_libraries(rsatb_hdf5 INTERFACE hdf5::hdf5 hdf5::hdf5_cpp)
    elseif(TARGET HDF5::HDF5)
        target_link_libraries(rsatb_hdf5 INTERFACE HDF5::HDF5)
    else()
        target_include_directories(rsatb_hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(rsatb_hdf5 INTERFACE
            ${HDF5_CXX_LIBRARIES}
            ${HDF5_C_LIBRARIES}
            ${HDF5_LIBRARIES})
    endif()
    add_library(RSATB::HDF5 ALIAS rsatb_hdf5)
endfunction()

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
rsatb_define_hdf5_target()

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# BLAS/LAPACK
if(BLAS_BACKEND STREQUAL "Auto")
    find_library(MKL_RT_LIB NAMES mkl_rt mkl_rt.2 mkl_rt.1
                 HINTS $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64 $ENV{CONDA_PREFIX}/lib)
    if(MKL_RT_LIB)
        message(STATUS "Auto-detected MKL SDL: ${MKL_RT_LIB}")
        set(BLAS_BACKEND "MKL" CACHE STRING "BLAS Backend (Auto, MKL, OpenBLAS, Generic)" FORCE)
    else()
        message(STATUS "MKL SDL not found; using Generic BLAS/LAPACK")
        set(BLAS_BACKEND "Generic" CACHE STRING "BLAS Backend (Auto, MKL, OpenBLAS, Generic)" FORCE)
    endif()
endif()

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
    # MKL always ships the grouped batch-GEMM API, which the SpGEMM numeric
    # phase uses (vbcsr gates it on this define).
    add_compile_definitions(__MKL RSATB_USE_MKL VBCSR_USE_MKL VBCSR_BLAS_HAS_BATCH_GEMM)
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
add_compile_definitions(__FFTW)
if(BLAS_BACKEND STREQUAL "MKL")
    find_path(RSATB_MKL_FFTW_INCLUDE_DIR
        NAMES fftw3.h
        HINTS $ENV{MKLROOT}/include/fftw $ENV{MKLROOT}/include
        PATH_SUFFIXES fftw
    )
    if(NOT RSATB_MKL_FFTW_INCLUDE_DIR)
        message(FATAL_ERROR "BLAS_BACKEND=MKL but MKL's FFTW interface header fftw3.h was not found. "
                "Set MKLROOT correctly, or use BLAS_BACKEND=Generic with an external FFTW module.")
    endif()
    if(NOT TARGET FFTW::FFTW)
        add_library(FFTW::FFTW INTERFACE IMPORTED)
        set_target_properties(FFTW::FFTW PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${RSATB_MKL_FFTW_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES     "${MKL_RT_LIB}")
    endif()
    set(FFTW_INCLUDE_DIRS "${RSATB_MKL_FFTW_INCLUDE_DIR}")
    set(FFTW_LIBRARIES "${MKL_RT_LIB}")
    message(STATUS "Using MKL FFTW interface: ${RSATB_MKL_FFTW_INCLUDE_DIR}; ${MKL_RT_LIB}")
else()
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    if(NOT FFTW_DOUBLE_LIB)
        message(FATAL_ERROR "FFTW double-precision library libfftw3 was not found.")
    endif()
    if(NOT TARGET FFTW::FFTW)
        add_library(FFTW::FFTW INTERFACE IMPORTED)
        set_target_properties(FFTW::FFTW PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES     "${FFTW_DOUBLE_LIB}")
    endif()
    set(FFTW_LIBRARIES "${FFTW_DOUBLE_LIB}")
    message(STATUS "Using external FFTW double library: ${FFTW_DOUBLE_LIB}")
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
  RESULT_VARIABLE MPI4PY_INCLUDE_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT MPI4PY_INCLUDE_RESULT EQUAL 0 OR NOT MPI4PY_INCLUDE_DIR)
    message(FATAL_ERROR
        "mpi4py is required at build time to provide mpi4py/mpi4py.h. "
        "Install mpi4py into ${Python3_EXECUTABLE}'s environment.")
endif()
message(STATUS "mpi4py include: ${MPI4PY_INCLUDE_DIR}")

set(PYBIND11_FINDPYTHON ON)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKE_DIR_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
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
rsatb_import_header_library(METIS::METIS METIS metis.h
    REQUIRED
    LIB_NAMES metis
    ROOT_HINTS "${METIS_ROOT}" "${CONDA_PREFIX}" "$ENV{METIS_ROOT}" "$ENV{EBROOTMETIS}" "${RSATB_DEPS_DIR}")
rsatb_import_header_library(ParMETIS::ParMETIS PARMETIS parmetis.h
    REQUIRED
    LIB_NAMES parmetis
    ROOT_HINTS "${PARMETIS_ROOT}" "${CONDA_PREFIX}" "$ENV{PARMETIS_ROOT}" "$ENV{EBROOTPARMETIS}" "${RSATB_DEPS_DIR}"
    LINK_LIBRARIES METIS::METIS MPI::MPI_C)
get_target_property(METIS_LIB METIS::METIS IMPORTED_LOCATION)
get_target_property(PARMETIS_LIB ParMETIS::ParMETIS IMPORTED_LOCATION)

# 2. DBCSR (Use existing sources in extern/, or fetch if missing)
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
rsatb_import_header_library(PRIMME::PRIMME PRIMME primme.h
    LIB_NAMES primme
    ROOT_HINTS "${PRIMME_ROOT}" "${CONDA_PREFIX}" "$ENV{PRIMME_ROOT}" "$ENV{EBROOTPRIMME}" "${RSATB_DEPS_DIR}"
    LINK_LIBRARIES "LAPACK::LAPACK;BLAS::BLAS")
if(NOT TARGET PRIMME::PRIMME)
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
    if(NOT TARGET PRIMME::PRIMME)
        add_library(PRIMME::PRIMME ALIAS primme)
    endif()
    set(PRIMME_LIB PRIMME::PRIMME)
else()
    get_target_property(PRIMME_LIBRARY PRIMME::PRIMME IMPORTED_LOCATION)
    set(PRIMME_LIB PRIMME::PRIMME)
    message(STATUS "PRIMME library found: ${PRIMME_LIBRARY}")
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
        GIT_REPOSITORY https://github.com/DeepElectron/vbcsr.git
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
#
# Do not create MATIO::MATIO here: op2c owns that imported target and creates it
# in its subdirectory. Defining it twice makes CMake configuration fail. We only
# locate/cache the paths early so the error message points at rsatb and op2c can
# reuse the same MATIO_INCLUDE_DIR / MATIO_LIBRARY values.
set(_RSATB_MATIO_INCLUDE_HINTS)
set(_RSATB_MATIO_LIBRARY_HINTS)
foreach(_root IN ITEMS "${MATIO_ROOT}" "${CONDA_PREFIX}" "$ENV{MATIO_ROOT}" "$ENV{EBROOTMATIO}" "${RSATB_DEPS_DIR}")
    if(_root)
        list(APPEND _RSATB_MATIO_INCLUDE_HINTS "${_root}/include")
        list(APPEND _RSATB_MATIO_LIBRARY_HINTS "${_root}/lib" "${_root}/lib64")
    endif()
endforeach()
find_path(MATIO_INCLUDE_DIR NAMES matio.h HINTS ${_RSATB_MATIO_INCLUDE_HINTS})
find_library(MATIO_LIBRARY NAMES matio HINTS ${_RSATB_MATIO_LIBRARY_HINTS})
if(NOT MATIO_INCLUDE_DIR OR NOT MATIO_LIBRARY)
    message(FATAL_ERROR
        "libmatio not found, but it is required to build the bundled op2c.\n"
        "  conda:           conda install -c conda-forge libmatio\n"
        "  Compute Canada:  module load matio   (or build it into your venv)\n"
        "Set CMAKE_PREFIX_PATH or MATIO_ROOT to a prefix containing include/ and lib/.")
endif()
set(MATIO_INCLUDE_DIR "${MATIO_INCLUDE_DIR}" CACHE PATH "libmatio include directory" FORCE)
set(MATIO_LIBRARY "${MATIO_LIBRARY}" CACHE FILEPATH "libmatio library" FORCE)
set(RSATB_MATIO_INCLUDE_DIR "${MATIO_INCLUDE_DIR}" CACHE PATH "libmatio include directory" FORCE)
set(RSATB_MATIO_LIBRARY "${MATIO_LIBRARY}" CACHE FILEPATH "libmatio library" FORCE)
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
    RSATB::HDF5
    FFTW::FFTW
    LAPACK::LAPACK
    BLAS::BLAS
    PRIMME::PRIMME
    ParMETIS::ParMETIS
    METIS::METIS
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

    # Executable: test_cheby - the Chebyshev operator shapes and application
    # modes. Was present in the tree but registered nowhere, so its adjoint
    # check had never been compiled, let alone run.
    add_executable(test_cheby calc/test/test_cheby.cpp)
    target_include_directories(test_cheby PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_cheby PRIVATE ${RSATB_LINK_LIBS})

    # Executable: test_preconditioner
    add_executable(test_preconditioner calc/test/test_preconditioner.cpp)
    target_include_directories(test_preconditioner PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_preconditioner PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_metric_sampler
    add_executable(test_metric_sampler calc/test/test_metric_sampler.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_metric_sampler PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_metric_sampler PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: test_rs_ldos - the full-cell stochastic LDOS map against
    # dense diagonalization
    add_executable(test_rs_ldos calc/test/test_rs_ldos.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_rs_ldos PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_rs_ldos PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

    # Executable: benchmark_spgemm - the SpGEMM numeric phase cost breakdown
    # on synthetic block matrices: full vs Hermitian-halved, batched vs
    # per-pair, at sparse and dense block densities.
    add_executable(benchmark_spgemm calc/test/benchmark_spgemm.cpp)
    target_include_directories(benchmark_spgemm PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(benchmark_spgemm PRIVATE ${RSATB_LINK_LIBS})

    # Executable: test_spmm_hermitian - the half-computed Hermitian product
    # against the full one, on synthetic serial matrices. No integral engine.
    add_executable(test_spmm_hermitian calc/test/test_spmm_hermitian.cpp)
    target_include_directories(test_spmm_hermitian PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_spmm_hermitian PRIVATE ${RSATB_LINK_LIBS})

    # Executable: test_square_polynomial - the fused c2 A^2 + c1 A + c0 I
    # against both a dense reference and the three-step sequence it replaces.
    add_executable(test_square_polynomial calc/test/test_square_polynomial.cpp)
    target_include_directories(test_square_polynomial PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_square_polynomial PRIVATE ${RSATB_LINK_LIBS})

    # Executable: test_spmm_inplace - A <- A B against the ordinary product,
    # which is the only way a premature storage release shows up (it corrupts
    # the answer, it does not crash). Drive it at 1, 2 and 4 ranks.
    add_executable(test_spmm_inplace calc/test/test_spmm_inplace.cpp)
    target_include_directories(test_spmm_inplace PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_spmm_inplace PRIVATE ${RSATB_LINK_LIBS})

    # Executable: test_rarh - the fused triple product C = A B A^H against a
    # dense reference, on synthetic matrices. Runs at any comm size; drive it at
    # 1, 2, 3 and 4 ranks, because the two-level ghost exchange is only
    # exercised once a row's neighbours live on another rank.
    add_executable(test_rarh calc/test/test_rarh.cpp)
    target_include_directories(test_rarh PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_rarh PRIVATE ${RSATB_LINK_LIBS})

    # Executable: experiment_ns_variants - which congruence (Z^H S Z vs Z S Z)
    # the Newton-Schulz refinement should iterate on, measured on the real
    # diamond overlap. An experiment, not a test: it reports, it does not assert.
    add_executable(experiment_ns_variants calc/test/experiment_ns_variants.cpp
                   $<TARGET_OBJECTS:backend>)
    target_include_directories(experiment_ns_variants PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(experiment_ns_variants PRIVATE ${RSATB_LINK_LIBS})

    # Executable: benchmark_isqrt_diamond - the thin inverse-square-root action
    # timed on the real diamond sc888 subgraph problems (needs the integral
    # engine, hence the backend objects). Single rank, OMP_NUM_THREADS=1.
    add_executable(benchmark_isqrt_diamond calc/test/benchmark_isqrt_diamond.cpp
                   $<TARGET_OBJECTS:backend>)
    target_include_directories(benchmark_isqrt_diamond PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(benchmark_isqrt_diamond PRIVATE ${RSATB_LINK_LIBS})

    # Executable: benchmark_fermi_diamond - ACE-FD on the real non-orthogonal
    # problem (file H, integral S, Op = S^-1 H) against dense diagonalization.
    add_executable(benchmark_fermi_diamond calc/test/benchmark_fermi_diamond.cpp
                   $<TARGET_OBJECTS:backend>)
    target_include_directories(benchmark_fermi_diamond PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(benchmark_fermi_diamond PRIVATE ${RSATB_LINK_LIBS})

    # Inverse square root (calc/isqrt_action.hpp, calc/isqrtm.hpp): the graph
    # driver against the legacy per-subgraph diagonalization, and the synthetic
    # sweep of the dense thin-action kernel (doc/design_isqrt_action.md).
    # Standalone MPI drivers, no backend objects needed.
    add_executable(test_isqrt_graph calc/test/test_isqrt_graph.cpp)
    target_include_directories(test_isqrt_graph PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(test_isqrt_graph PRIVATE ${RSATB_LINK_LIBS})
    add_executable(benchmark_isqrt_action calc/test/benchmark_isqrt_action.cpp)
    target_include_directories(benchmark_isqrt_action PRIVATE ${RSATB_INCLUDE_DIRS})
    target_link_libraries(benchmark_isqrt_action PRIVATE ${RSATB_LINK_LIBS})

    # Standalone MPI test drivers (own main(), not GTest). Registered so they
    # are compiled by the normal build and cannot silently bit-rot again.
    foreach(_driver test_isqrtm test_graphmf test_solver test_dos test_ac test_fermi_ace)
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
