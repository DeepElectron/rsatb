cmake_minimum_required(VERSION 3.15...4.0)
project(source_base LANGUAGES CXX)

add_library(
    base
    STATIC
    math_integral.cpp
    math_sphbes.cpp
    memory.cpp
    ylm.cpp
    parallel_common.cpp
    spherical_bessel_transformer.cpp
    cubic_spline.cpp
    projgen.cpp
    matrix.cpp
    realarray.cpp
    complexarray.cpp
    complexmatrix.cpp
)

if (PROJECT_IS_TOP_LEVEL)
    # include_directories(./module_device)
    # include_directories(./module_container)
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG REQUIRED)
    list(APPEND math_libs MKL::MKL)
endif()


target_link_libraries(base PUBLIC container MPI::MPI_CXX ${math_libs} LAPACK::LAPACK BLAS::BLAS)

target_include_directories(base PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/int2c>
)
add_subdirectory(module_device)
add_subdirectory(module_container)

if(ENABLE_COVERAGE)
  add_coverage(base)
endif()

