cmake_minimum_required(VERSION 3.15...4.0)
project(calc LANGUAGES CXX)

set(PETSC_DIR $ENV{PETSC_DIR})
set(SLEPC_DIR $ENV{SLEPC_DIR})
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

# Interface library for calc
add_library(calclib INTERFACE)
target_compile_features(calclib INTERFACE cxx_std_17)
target_include_directories(calclib INTERFACE
    ${PETSC_DIR}/include
    ${SLEPC_DIR}/include
    ${CONDA_PREFIX}/include
    ../extern/
)

find_library(PETSC_LIB NAMES petsc HINTS ${PETSC_DIR}/lib)
find_library(SLEPC_LIB NAMES slepc HINTS ${SLEPC_DIR}/lib)
find_library(MPI_LIB NAMES mpi HINTS ${CONDA_PREFIX}/lib)
find_library(GTEST_LIB NAMES gtest HINTS ${CONDA_PREFIX}/lib)
find_library(GTEST_MAIN_LIB NAMES gtest_main HINTS ${CONDA_PREFIX}/lib)

message(STATUS "PETSc library: ${PETSC_LIB}")
message(STATUS "SLEPc library: ${SLEPC_LIB}")
message(STATUS "MPI library: ${MPI_LIB}")
message(STATUS "GTest library: ${GTEST_LIB}")
message(STATUS "GTest Main library: ${GTEST_MAIN_LIB}")

target_link_libraries(calclib INTERFACE ${PETSC_LIB} ${SLEPC_LIB} ${MPI_LIB})

add_executable(test_calc test.cpp)
target_link_libraries(test_calc PRIVATE calclib ${GTEST_LIB} ${GTEST_MAIN_LIB})