add_library(
numerical_atomic_orbitals
STATIC
  numerical_radial.cpp
  radial_set.cpp
  atomic_radials.cpp
  radial_set.cpp
  radial_collection.cpp
  real_gaunt_table.cpp
  two_center_table.cpp
  two_center_integrator.cpp
  two_center_bundle.cpp
  beta_radials.cpp
)

target_link_libraries(numerical_atomic_orbitals
    PUBLIC
    base
    MPI::MPI_CXX
)

target_include_directories(numerical_atomic_orbitals
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/int2c>
)


if(ENABLE_COVERAGE)
  add_coverage(numerical_atomic_orbitals)
endif()
