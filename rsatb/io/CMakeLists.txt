cmake_minimum_required(VERSION 3.15...4.0)
project(io LANGUAGES CXX)

if(NOT TARGET RSATB::HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C CXX)

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
endif()

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    if(RSATB_BUILD_TESTS)
        find_package(GTest REQUIRED)
    endif()
endif()

add_library(io INTERFACE)
target_compile_features(io INTERFACE cxx_std_11)

target_link_libraries(io INTERFACE RSATB::HDF5)

if(RSATB_BUILD_TESTS)
    add_executable(test_io ./test.cpp)
    target_link_libraries(test_io PRIVATE io GTest::gtest MPI::MPI_CXX)
endif()
