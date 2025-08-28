cmake_minimum_required(VERSION 3.15...4.0)

# project settings
project(
  int2c 
  LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

if (PROJECT_IS_TOP_LEVEL)
  find_package(GTest REQUIRED)
  find_package(MPI REQUIRED)
endif()
include_directories(${MPI_CXX_INCLUDE_PATH})
add_compile_definitions(__MPI)

# set source path
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/")
message(STATUS "set source_dir as ${SOURCE_DIR}")
set(BASE_PATH "${SOURCE_DIR}/source_base")
message(STATUS "set base_path as ${BASE_PATH}")
set(NAO_PATH "${SOURCE_DIR}/source_basis/module_nao")
message(STATUS "set nao_path as ${NAO_PATH}")

# set(MKL_INTERFACE lp64)
# find_package(MKL CONFIG REQUIRED)

# add math_libs

if (PROJECT_IS_TOP_LEVEL)
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
endif()

# add include directories

include_directories(
    ${BASE_PATH} 
    ${SOURCE_DIR} 
    ${SOURCE_DIR}/source_base/module_container 
    ${SOURCE_DIR}/source_base/module_device 
    )

# add basic libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add base
set(BASE_BINARY_DIR "${PROJECT_SOURCE_DIR}/build/base")
add_subdirectory(${SOURCE_DIR}/source_base ${BASE_BINARY_DIR})
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(ORB_BINARY_DIR "${PROJECT_SOURCE_DIR}/build/orb")
add_subdirectory(${SOURCE_DIR}/source_basis/module_nao ${ORB_BINARY_DIR})

add_executable(test_nao ./test.cpp)
target_link_libraries(test_nao PRIVATE base numerical_atomic_orbitals MPI::MPI_CXX GTest::gtest)
