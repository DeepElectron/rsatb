cmake_minimum_required(VERSION 3.15...4.0)
project(calc LANGUAGES CXX C Fortran)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    find_package(GTest REQUIRED)
    find_package(PETSc REQUIRED)
    find_package(FFTW REQUIRED)
    find_package(DBCSR 2.0.0 CONFIG REQUIRED)
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

if(NOT TARGET SLEPC_LIB)
  find_library(SLEPC_LIB
      NAMES slepc
      HINTS ${PETSC_INCLUDE_DIR}/../lib
  )

  message(STATUS "SLEPC library: ${SLEPC_LIB}")
endif()

add_library(calc INTERFACE)
target_compile_features(calc INTERFACE cxx_std_11)
target_include_directories(calc INTERFACE ../extern/)

if (PROJECT_IS_TOP_LEVEL)
  target_link_libraries(calc INTERFACE MPI::MPI_CXX PETSc::PETSc FFTW::FFTW ${SLEPC_LIB} DBCSR::dbcsr_c)
else()
  target_link_libraries(calc INTERFACE MPI::MPI_CXX PETSc::PETSc ${math_libs} ${SLEPC_LIB} DBCSR::dbcsr_c)
endif()
add_executable(test_calc ./test.cpp)
target_link_libraries(test_calc PRIVATE calc GTest::gtest)

add_executable(test_cheby_conjugate ./test_cheby_conjugate.cpp)
target_link_libraries(test_cheby_conjugate PRIVATE calc GTest::gtest)