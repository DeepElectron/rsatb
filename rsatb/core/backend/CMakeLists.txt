cmake_minimum_required(VERSION 3.15...4.0)
project(Calculator LANGUAGES CXX)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

add_library(petsclib STATIC petsc.cpp petsc.hpp)
target_compile_features(petsclib PUBLIC cxx_std_11)
target_include_directories(petsclib PUBLIC ${CONDA_PREFIX}/include ${CONDA_PREFIX}/petsc/build/include ${CONDA_PREFIX}/petsc/arch-linux-c-opt/include)

find_library(PETSC_LIB
    NAMES petsc
    HINTS ${CONDA_PREFIX}/petsc/arch-linux-c-opt/lib
)

message(STATUS "PETSc library: ${PETSC_LIB}")

find_library(MPI_LIB
    NAMES mpi
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "MPI library: ${MPI_LIB}")


find_library(GTEST_LIB
    NAMES gtest
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "GTEST library: ${GTEST_LIB}")

find_library(PTHREAD_LIB NAMES pthread PATHS ${CONDA_PREFIX}/lib)

message(STATUS "PTHREAD library: ${PTHREAD_LIB}")

target_link_libraries(petsclib PUBLIC ${PETSC_LIB} ${PTHREAD_LIB} ${MPI_LIB})


add_executable(test_backend ./test.cpp)
target_link_libraries(test_backend PUBLIC petsclib ${GTEST_LIB})