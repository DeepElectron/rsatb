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
    set(BLA_VENDOR "Intel10_64lp")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
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
  COMMAND ${Python3_EXECUTABLE} -c "import mpi4py, sys; sys.stdout.write(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "mpi4py include: ${MPI4PY_INCLUDE_DIR}")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# --- Local / Fetched Dependencies ---

# GTest (FetchContent)
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

list(APPEND math_libs FFTW::FFTW LAPACK::LAPACK BLAS::BLAS)

# --- Subdirectories ---
add_subdirectory(backend)
add_subdirectory(io)
add_subdirectory(extern/nao-abacus)

# --- Targets ---

# Common Link Libraries
set(RSATB_LINK_LIBS
    vbcsr
    io
    base
    numerical_atomic_orbitals
    estate
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
    backend
    extern
    extern/nao-abacus
    extern/nao-abacus/source_base/module_container
    extern/vbcsr
)

if(DEFINED RSATB_DEPS_DIR)
    list(APPEND RSATB_INCLUDE_DIRS "${RSATB_DEPS_DIR}/include")
endif()
if(DEFINED CONDA_PREFIX)
    list(APPEND RSATB_INCLUDE_DIRS "${CONDA_PREFIX}/include")
endif()


# Executable: test_rsatb
add_executable(test_rsatb test_rsatb.cpp $<TARGET_OBJECTS:backend>)
target_include_directories(test_rsatb PRIVATE ${RSATB_INCLUDE_DIRS})
target_link_libraries(test_rsatb PRIVATE ${RSATB_LINK_LIBS} GTest::gtest)

# Python Module: _rsatb
pybind11_add_module(_rsatb pybind_rsatb.cpp)
target_include_directories(_rsatb PRIVATE ${RSATB_INCLUDE_DIRS})
target_link_libraries(_rsatb PRIVATE ${RSATB_LINK_LIBS})

# Install
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS _rsatb DESTINATION .)

# Test: test_isqrtm_vbcsr
# add_executable(test_isqrtm_vbcsr calc/test/test_isqrtm_vbcsr.cpp)
# target_include_directories(test_isqrtm_vbcsr PRIVATE ${RSATB_INCLUDE_DIRS})
# target_link_libraries(test_isqrtm_vbcsr PRIVATE ${RSATB_LINK_LIBS})

# Benchmark: benchmark_isqrtm_heavy
# add_executable(benchmark_isqrtm_heavy calc/test/benchmark_isqrtm_heavy.cpp)
# target_include_directories(benchmark_isqrtm_heavy PRIVATE ${RSATB_INCLUDE_DIRS})
# target_link_libraries(benchmark_isqrtm_heavy PRIVATE ${RSATB_LINK_LIBS})
