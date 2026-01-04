cmake_minimum_required(VERSION 3.15...4.0)
project(rsatb LANGUAGES CXX Fortran C)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MPI REQUIRED)
find_package(GTest)
if(NOT GTest_FOUND)
  message(STATUS "GTest not found, fetching from source...")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

find_package(Threads REQUIRED)
find_package(PETSc REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(DBCSR 2.0.0 CONFIG REQUIRED)
# find_package(FFTW REQUIRED)

# add math_libs
if(DEFINED ENV{MKLROOT} AND NOT DEFINED MKLROOT)
    set(MKLROOT "$ENV{MKLROOT}")
endif()

if(MKLROOT)
    set(MKL_INTERFACE lp64)
    set(USE_OPENMP ON)
    if(USE_OPENMP)
        find_package(OpenMP REQUIRED)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_link_options(${OpenMP_CXX_LIBRARIES})
    endif()
    find_package(MKL REQUIRED)
    add_definitions(-D__MKL)
    include_directories(${MKL_INCLUDE} ${MKL_INCLUDE}/fftw)

    list(APPEND math_libs IntelMKL::MKL)

    if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    list(APPEND math_libs -lifcore)
    endif()
else()
    find_package(LAPACK REQUIRED)
    find_package(FFTW REQUIRED)
    add_compile_definitions(__FFTW)
    if(NOT TARGET FFTW::FFTW)
        add_library(FFTW::FFTW INTERFACE IMPORTED)
        set_target_properties(FFTW::FFTW PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" 
            INTERFACE_LINK_LIBRARIES     "${FFTW_LIBRARIES}")
    endif()
    include_directories(${FFTW_INCLUDE_DIRS})
    message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}")
    list(APPEND math_libs FFTW::FFTW LAPACK::LAPACK)
endif()


# Wrap into a modern imported target so the rest of your code can stay target-based
if(NOT TARGET PETSc::PETSc)
    add_library(PETSc::PETSc INTERFACE IMPORTED)
    set_target_properties(PETSc::PETSc PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSCCONF_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES     "${PETSC_LIBRARIES}")
endif()

# if(NOT TARGET FFTW::FFTW)
#     add_library(FFTW::FFTW INTERFACE IMPORTED)
#     set_target_properties(FFTW::FFTW PROPERTIES
#       INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" 
#       INTERFACE_LINK_LIBRARIES     "${FFTW_LIBRARIES}")
# endif()

add_subdirectory(backend)
add_subdirectory(io)
add_subdirectory(extern/nao-abacus)
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
add_library(hr2hk SHARED hr2hk.cpp hr2hk.hpp $<TARGET_OBJECTS:backend>)
target_compile_features(hr2hk PUBLIC cxx_std_11)

target_include_directories(
    hr2hk PUBLIC
    ./ 
    ./extern/nao-abacus/
    ./extern/nao-abacus/source_base/module_container 
    ./extern/nao-abacus/source_base/module_device 
    )

target_link_libraries(hr2hk PUBLIC backend io base numerical_atomic_orbitals estate)

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

install(TARGETS binding solver hr2hk DESTINATION rsatb/core)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Manually add DBCSR path since it's in a non-standard location
set(DBCSR_LIB_PATH "${CMAKE_SOURCE_DIR}/../../deps/install/lib64")

# Extract PETSc lib dir manually to ensure it's in RPATH (fixes local deployment issues)
list(GET PETSC_LIBRARIES 0 PETSC_MAIN_LIB)
get_filename_component(PETSC_LIB_DIR "${PETSC_MAIN_LIB}" DIRECTORY)

# Extract SLEPc lib dir manually
get_filename_component(SLEPC_LIB_DIR "${SLEPC_LIB}" DIRECTORY)

set_target_properties(binding PROPERTIES INSTALL_RPATH "$ORIGIN:${DBCSR_LIB_PATH}:${PETSC_LIB_DIR}:${SLEPC_LIB_DIR}")
set_target_properties(solver PROPERTIES INSTALL_RPATH "$ORIGIN:${DBCSR_LIB_PATH}:${PETSC_LIB_DIR}:${SLEPC_LIB_DIR}")
set_target_properties(hr2hk PROPERTIES INSTALL_RPATH "$ORIGIN:${DBCSR_LIB_PATH}:${PETSC_LIB_DIR}:${SLEPC_LIB_DIR}")

add_executable(test ./test.cpp)
target_link_libraries(test PRIVATE solver GTest::gtest)

add_executable(test_hr2hk ./test_hr2hk.cpp)
target_link_libraries(test_hr2hk PRIVATE solver GTest::gtest)

add_executable(test_operator_construction backend/test_operator_construction.cpp $<TARGET_OBJECTS:backend>)
target_include_directories(test_operator_construction PRIVATE backend extern/nao-abacus extern/nao-abacus/source_base/module_container)
target_link_libraries(test_operator_construction PRIVATE io base numerical_atomic_orbitals estate GTest::gtest)
