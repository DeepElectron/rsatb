cmake_minimum_required(VERSION 3.15...4.0)
project(backend LANGUAGES CXX Fortran C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    find_package(GTest REQUIRED)
    find_package(Threads REQUIRED)
    
    # ParMETIS
    find_library(PARMETIS_LIB parmetis HINTS $ENV{CONDA_PREFIX}/lib)
    find_library(METIS_LIB metis HINTS $ENV{CONDA_PREFIX}/lib)
    include_directories($ENV{CONDA_PREFIX}/include)

    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../../../deps/install")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(FFTW REQUIRED)
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
    
    # Add dependencies
    add_subdirectory(../io io_build)
    add_subdirectory(../extern/op2c op2c_build)
endif()

add_library(backend OBJECT operator_construction.hpp operator_set.hpp)
set_target_properties(backend PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(backend PUBLIC cxx_std_17)
target_compile_options(backend PUBLIC -mavx2 -mfma)

target_link_libraries(backend PUBLIC MPI::MPI_CXX Threads::Threads BLAS::BLAS LAPACK::LAPACK vbcsr)
if(TARGET primme)
    target_link_libraries(backend PUBLIC primme)
endif()
if(USE_OPENMP)
    target_link_libraries(backend PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/primme/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/vbcsr/
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/vbcsr/vbcsr/
)
if(DEFINED CONDA_PREFIX)
    target_include_directories(backend PUBLIC ${CONDA_PREFIX}/include)
endif()


add_executable(test_operator_construction test/test_operator_construction.cpp $<TARGET_OBJECTS:backend>)
target_include_directories(test_operator_construction PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
)
target_link_libraries(test_operator_construction PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)

add_executable(test_operator_set test/test_operator_set.cpp $<TARGET_OBJECTS:backend>)
target_include_directories(test_operator_set PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
)
target_link_libraries(test_operator_set PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)
