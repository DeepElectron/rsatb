cmake_minimum_required(VERSION 3.15...4.0)
project(backend LANGUAGES CXX)

set(PETSC_DIR $ENV{PETSC_DIR})
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

add_library(petsclib STATIC petsc.cpp petsc.hpp)
target_compile_features(petsclib PUBLIC cxx_std_17)
target_include_directories(petsclib PUBLIC
    ${PETSC_DIR}/include
    ${CONDA_PREFIX}/include
)

find_library(PETSC_LIB NAMES petsc HINTS ${PETSC_DIR}/lib)
find_library(MPI_LIB NAMES mpi HINTS ${CONDA_PREFIX}/lib)
find_library(PTHREAD_LIB NAMES pthread PATHS ${CONDA_PREFIX}/lib)

message(STATUS "PETSc library: ${PETSC_LIB}")
message(STATUS "MPI library: ${MPI_LIB}")
message(STATUS "PTHREAD library: ${PTHREAD_LIB}")

target_link_libraries(petsclib PUBLIC ${PETSC_LIB} ${MPI_LIB} ${PTHREAD_LIB})

find_library(GTEST_LIB
    NAMES gtest
    HINTS ${CONDA_PREFIX}/lib
)
find_library(GTEST_MAIN_LIB
    NAMES gtest_main
    HINTS ${CONDA_PREFIX}/lib
)
message(STATUS "GTEST library: ${GTEST_LIB}")
message(STATUS "GTEST_MAIN library: ${GTEST_MAIN_LIB}")

add_executable(test_backend test.cpp)
target_link_libraries(test_backend PRIVATE petsclib ${GTEST_LIB} ${GTEST_MAIN_LIB})