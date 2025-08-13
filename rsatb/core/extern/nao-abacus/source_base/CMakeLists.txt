cmake_minimum_required(VERSION 3.15...4.0)
project(source_base LANGUAGES CXX)

add_library(
    base
    OBJECT
    math_integral.cpp
    math_sphbes.cpp
    memory.cpp
    ylm.cpp
    parallel_common.cpp
    spherical_bessel_transformer.cpp
    cubic_spline.cpp
    projgen.cpp
)

if (PROJECT_IS_TOP_LEVEL)
    include_directories(./module_device)
    include_directories(./module_container)
endif()

find_package(MKL CONFIG REQUIRED)
target_link_libraries(base PUBLIC container MKL::MKL)
add_subdirectory(module_device)
add_subdirectory(module_container)

if(ENABLE_COVERAGE)
  add_coverage(base)
endif()

