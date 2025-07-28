cmake_minimum_required(VERSION 3.15...4.0)
project(calc LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    find_package(GTest REQUIRED)
    find_package(PETSc REQUIRED)
    find_package(FFTW REQUIRED)
endif()
# Wrap into a modern imported target so the rest of your code can stay target-based
if(NOT TARGET PETSc::PETSc)
    add_library(PETSc::PETSc INTERFACE IMPORTED)
    set_target_properties(PETSc::PETSc PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSCCONF_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES     "${PETSC_LIBRARIES}")
endif()

if(NOT TARGET FFTW::FFTW)
    add_library(FFTW::FFTW INTERFACE IMPORTED)
    set_target_properties(FFTW::FFTW PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" 
      INTERFACE_LINK_LIBRARIES     "${FFTW_LIBRARIES}")
endif()

add_library(calc INTERFACE)
target_compile_features(calc INTERFACE cxx_std_11)
target_include_directories(calc INTERFACE ../extern/)

target_link_libraries(calc INTERFACE MPI::MPI_CXX PETSc::PETSc FFTW::FFTW)

add_executable(test_calc ./test.cpp)
target_link_libraries(test_calc PRIVATE calc GTest::gtest)