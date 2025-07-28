cmake_minimum_required(VERSION 3.15...4.0)
project(backend LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    find_package(GTest REQUIRED)
    find_package(Threads REQUIRED)
    find_package(PETSc REQUIRED)
endif()
# Wrap into a modern imported target so the rest of your code can stay target-based
if(NOT TARGET PETSc::PETSc)
add_library(PETSc::PETSc INTERFACE IMPORTED)
set_target_properties(PETSc::PETSc PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSCCONF_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES     "${PETSC_LIBRARIES}")
endif()

add_library(backend OBJECT petsc.cpp petsc.hpp)
target_compile_features(backend PUBLIC cxx_std_11)

target_link_libraries(backend PUBLIC MPI::MPI_CXX Threads::Threads PETSc::PETSc)

add_executable(test_backend ./test.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_backend PUBLIC backend GTest::gtest)