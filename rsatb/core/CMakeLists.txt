cmake_minimum_required(VERSION 3.15...4.0)
project(rsatb LANGUAGES CXX)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(MPI REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(PETSc REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(FFTW REQUIRED)
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

add_subdirectory(backend)
add_subdirectory(io)
add_subdirectory(calc)

find_library(SLEPC_LIB
    NAMES slepc
    HINTS ${PETSC_INCLUDE_DIR}/../lib
)

message(STATUS "SLEPC library: ${SLEPC_LIB}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import mpi4py, sys; sys.stdout.write(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "mpi4py include: ${MPI4PY_INCLUDE_DIR}")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


# add hr2hk lib
add_library(hr2hk OBJECT hr2hk.cpp hr2hk.hpp $<TARGET_OBJECTS:backend>)
target_compile_features(hr2hk PUBLIC cxx_std_11)

# target_include_directories(
#     hr2hklib PUBLIC
#     ./ 
#     ${CONDA_PREFIX}/include 
#     ${CONDA_PREFIX}/petsc/build/include 
#     ${CONDA_PREFIX}/petsc/arch-linux-c-opt/include
#     )

target_link_libraries(hr2hk PUBLIC backend io)

# add solver lib
add_library(solver SHARED solver.cpp solver.hpp)
target_compile_features(solver PUBLIC cxx_std_11)
target_include_directories(
    solver PUBLIC 
    ./
)
target_link_libraries(solver PUBLIC hr2hk calc backend io ${SLEPC_LIB})

# add the binding lib

pybind11_add_module(binding SHARED binding.cpp)
target_compile_features(binding PUBLIC cxx_std_11)
target_include_directories(
    binding PUBLIC 
    ./ 
    ${MPI4PY_INCLUDE_DIR}
)

target_link_libraries(binding PUBLIC solver)

install(TARGETS binding DESTINATION .)

add_executable(test ./test.cpp)
target_link_libraries(test PRIVATE solver GTest::gtest)
