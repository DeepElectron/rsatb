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
# find_package(PETSc REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(DBCSR 2.0.0 CONFIG REQUIRED)
# find_package(FFTW REQUIRED)
# ParMETIS
find_library(PARMETIS_LIB parmetis HINTS $ENV{CONDA_PREFIX}/lib)
find_library(METIS_LIB metis HINTS $ENV{CONDA_PREFIX}/lib)
include_directories($ENV{CONDA_PREFIX}/include)

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
# if(NOT TARGET PETSc::PETSc)
#     add_library(PETSc::PETSc INTERFACE IMPORTED)
#     set_target_properties(PETSc::PETSc PROPERTIES
#       INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSCCONF_INCLUDE_DIR}"
#       INTERFACE_LINK_LIBRARIES     "${PETSC_LIBRARIES}")
# endif()

# if(NOT TARGET FFTW::FFTW)
#     add_library(FFTW::FFTW INTERFACE IMPORTED)
#     set_target_properties(FFTW::FFTW PROPERTIES
#       INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}" 
#       INTERFACE_LINK_LIBRARIES     "${FFTW_LIBRARIES}")
# endif()

add_subdirectory(backend)
add_subdirectory(io)
add_subdirectory(extern/nao-abacus)
# add_subdirectory(calc)

# find_library(SLEPC_LIB
#     NAMES slepc
#     HINTS ${PETSC_INCLUDE_DIR}/../lib
# )

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

find_library(PRIMME_LIB primme HINTS $ENV{CONDA_PREFIX}/lib ${CMAKE_SOURCE_DIR}/../../deps/install/lib)
message(STATUS "PRIMME library: ${PRIMME_LIB}")

# add the binding lib

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Manually add DBCSR path since it's in a non-standard location
set(DBCSR_LIB_PATH "${CMAKE_SOURCE_DIR}/../../deps/install/lib64")

add_executable(test_rsatb test_rsatb.cpp $<TARGET_OBJECTS:backend>)
target_include_directories(test_rsatb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../.. backend extern extern/nao-abacus extern/nao-abacus/source_base/module_container)
target_link_libraries(test_rsatb PRIVATE io base numerical_atomic_orbitals estate GTest::gtest MPI::MPI_CXX DBCSR::dbcsr_c HDF5::HDF5 FFTW::FFTW LAPACK::LAPACK ${PRIMME_LIB} ${PARMETIS_LIB} ${METIS_LIB})

