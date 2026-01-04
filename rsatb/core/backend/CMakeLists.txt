cmake_minimum_required(VERSION 3.15...4.0)
project(backend LANGUAGES CXX Fortran C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    find_package(GTest REQUIRED)
    find_package(Threads REQUIRED)
    find_package(PETSc REQUIRED)
    find_package(DBCSR 2.0.0 CONFIG REQUIRED)
    find_package(BLAS REQUIRED)
endif()
# Wrap into a modern imported target so the rest of your code can stay target-based
if(NOT TARGET PETSc::PETSc)
add_library(PETSc::PETSc INTERFACE IMPORTED)
set_target_properties(PETSc::PETSc PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSCCONF_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES     "${PETSC_LIBRARIES}")
endif()

add_library(backend OBJECT petsc.cpp petsc.hpp dbcsr.hpp petsc_hack.h operator_construction.hpp)
target_compile_features(backend PUBLIC cxx_std_11)

target_link_libraries(backend PUBLIC MPI::MPI_CXX Threads::Threads PETSc::PETSc DBCSR::dbcsr_c BLAS::BLAS)
target_include_directories(backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus/source_base/module_container
)

# add_executable(test_backend ./test.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_backend PUBLIC backend GTest::gtest)

add_executable(test_dm_atomic ./test_dmplex.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_dm_atomic PUBLIC backend GTest::gtest)

add_executable(test_complex_dist_vector ./test_complex_dist_vector.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_complex_dist_vector PUBLIC backend MPI::MPI_CXX)

add_executable(test_block_csr ./test_block_csr.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_block_csr PUBLIC backend MPI::MPI_CXX)

add_executable(test_dbcsr_conversions ./test_dbcsr_conversions.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_dbcsr_conversions PUBLIC backend MPI::MPI_CXX)

add_executable(test_isqrtm ../calc_/test_isqrtm.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_isqrtm PUBLIC backend MPI::MPI_CXX)
target_include_directories(test_isqrtm PUBLIC ../calc_)