cmake_minimum_required(VERSION 3.15...4.0)
project(Calculator LANGUAGES CXX)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

add_library(hdf5lib INTERFACE)
target_compile_features(hdf5lib INTERFACE cxx_std_11)
target_include_directories(hdf5lib INTERFACE ${CONDA_PREFIX}/include)

find_library(HDF5_LIB
    NAMES hdf5
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "HDF5 library: ${HDF5_LIB}")

find_library(HDF5_C_LIB
    NAMES hdf5_cpp
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "HDF5 library: ${HDF5_C_LIB}")

find_library(GTEST_LIB
    NAMES gtest
    HINTS ${CONDA_PREFIX}/lib
)

message(STATUS "GTEST library: ${GTEST_LIB}")

target_link_libraries(hdf5lib INTERFACE ${HDF5_LIB} ${HDF5_C_LIB})

add_executable(test_io ./test.cpp)
target_link_libraries(test_io PRIVATE hdf5lib ${GTEST_LIB})