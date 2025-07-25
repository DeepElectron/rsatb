cmake_minimum_required(VERSION 3.15...4.0)
project(Calculator LANGUAGES CXX)

# Get important environment variables
set(CONDA_PREFIX $ENV{CONDA_PREFIX})
set(PETSC_DIR $ENV{PETSC_DIR})
set(SLEPC_DIR $ENV{SLEPC_DIR})

# Add subdirectories (assumes each has its own CMakeLists.txt or source files)
add_subdirectory(backend)
add_subdirectory(io)
add_subdirectory(calc)

# Find SLEPc and PETSc libraries manually
find_library(PETSC_LIB
    NAMES petsc
    HINTS ${PETSC_DIR}/lib
)
find_library(SLEPC_LIB
    NAMES slepc slepc4c
    HINTS ${SLEPC_DIR}/lib
)

# Find HDF5 if you use it
find_library(HDF5_LIB NAMES hdf5 HINTS ${CONDA_PREFIX}/lib)
find_library(HDF5_C_LIB NAMES hdf5_cpp HINTS ${CONDA_PREFIX}/lib)

# Find GTest if you use it
find_library(GTEST_LIB NAMES gtest HINTS ${CONDA_PREFIX}/lib)

# Print status messages
message(STATUS "PETSc library: ${PETSC_LIB}")
message(STATUS "SLEPc library: ${SLEPC_LIB}")
message(STATUS "HDF5 library: ${HDF5_LIB}")
message(STATUS "HDF5 C++ library: ${HDF5_C_LIB}")
message(STATUS "GTest library: ${GTEST_LIB}")

# Core library: hr2hklib
add_library(hr2hklib STATIC hr2hk.cpp hr2hk.hpp)
target_compile_features(hr2hklib PUBLIC cxx_std_17)
target_include_directories(hr2hklib PUBLIC
    ./
    ${PETSC_DIR}/include
    ${SLEPC_DIR}/include
)

target_link_libraries(hr2hklib PUBLIC ${PETSC_LIB} ${HDF5_LIB})

# Solver library
add_library(solverlib STATIC solver.cpp solver.hpp)
target_compile_features(solverlib PUBLIC cxx_std_17)
target_include_directories(solverlib PUBLIC
    ./
    ${PETSC_DIR}/include
    ${SLEPC_DIR}/include
)
target_link_libraries(solverlib PUBLIC hr2hklib ${SLEPC_LIB})

# Test executable
add_executable(test test.cpp)
target_link_libraries(test PRIVATE solverlib hr2hklib petsclib ${GTEST_LIB} ${HDF5_LIB} ${HDF5_C_LIB} ${PETSC_LIB} ${MPI_LIB})