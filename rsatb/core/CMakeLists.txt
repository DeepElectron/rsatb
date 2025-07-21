cmake_minimum_required(VERSION 3.15...4.0)
project(Calculator LANGUAGES CXX)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

add_subdirectory(backend)
add_subdirectory(io)

find_library(SLEPC_LIB
    NAMES slepc
    HINTS ${CONDA_PREFIX}/petsc/build/lib
)

message(STATUS "SLEPC library: ${SLEPC_LIB}")

# add hr2hk lib
add_library(hr2hklib STATIC hr2hk.cpp hr2hk.hpp)
target_compile_features(hr2hklib PUBLIC cxx_std_11)
target_include_directories(
    hr2hklib PUBLIC
    ./ 
    ${CONDA_PREFIX}/include 
    ${CONDA_PREFIX}/petsc/build/include 
    ${CONDA_PREFIX}/petsc/arch-linux-c-opt/include
    )
target_link_libraries(hr2hklib PUBLIC petsclib hdf5lib)

add_library(solverlib STATIC solver.cpp solver.hpp)
target_compile_features(solverlib PUBLIC cxx_std_11)
target_include_directories(
    solverlib PUBLIC 
    ./
    ${CONDA_PREFIX}/petsc/build/include/
)
target_link_libraries(solverlib PUBLIC hr2hklib ${SLEPC_LIB})

add_executable(test ./test.cpp)
target_link_libraries(test PRIVATE hr2hklib solverlib ${GTEST_LIB} ${HDF5_LIB} ${HDF5_C_LIB})
