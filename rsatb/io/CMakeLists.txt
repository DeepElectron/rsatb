cmake_minimum_required(VERSION 3.15...4.0)
project(io LANGUAGES CXX)

if (PROJECT_IS_TOP_LEVEL)
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
    find_package(MPI REQUIRED)
    if(RSATB_BUILD_TESTS)
        find_package(GTest REQUIRED)
    endif()
endif()

add_library(io INTERFACE)
target_compile_features(io INTERFACE cxx_std_11)

target_link_libraries(io INTERFACE hdf5::hdf5 hdf5::hdf5_cpp)

if(RSATB_BUILD_TESTS)
    add_executable(test_io ./test.cpp)
    target_link_libraries(test_io PRIVATE io GTest::gtest MPI::MPI_CXX)
endif()
