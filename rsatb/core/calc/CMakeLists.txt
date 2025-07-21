cmake_minimum_required(VERSION 3.15...4.0)
project(Calculator LANGUAGES CXX)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

add_library(calclib INTERFACE)
target_compile_features(calclib INTERFACE cxx_std_11)
target_include_directories(calclib INTERFACE ${CONDA_PREFIX}/petsc/build/include ${CONDA_PREFIX}/petsc/arch-linux-c-opt/include ${CONDA_PREFIX}/include)

find_library(PETSC_LIB
    NAMES petsc 
    HINTS ${CONDA_PREFIX}/petsc/arch-linux-c-opt/lib
)

message(STATUS "PETSc library: ${PETSC_LIB}")

find_library(GTEST_LIB
    NAMES gtest
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "GTEST library: ${GTEST_LIB}")

find_library(MPI_LIB
    NAMES mpi
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "MPI library: ${MPI_LIB}")

target_link_libraries(calclib INTERFACE ${PETSC_LIB} ${MPI_LIB})

add_executable(test_calc ./test.cpp)
target_link_libraries(test_calc PRIVATE calclib ${GTEST_LIB})