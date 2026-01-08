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
    find_package(DBCSR 2.0.0 CONFIG REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(FFTW REQUIRED)
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
    
    # Add dependencies
    add_subdirectory(../io io_build)
    add_subdirectory(../extern/nao-abacus nao_abacus_build)
endif()

add_library(backend OBJECT operator_construction.hpp)
set_target_properties(backend PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(backend PUBLIC cxx_std_11)
target_compile_options(backend PUBLIC -mavx2 -mfma)

target_link_libraries(backend PUBLIC MPI::MPI_CXX Threads::Threads DBCSR::dbcsr_c BLAS::BLAS LAPACK::LAPACK vbcsr)
if(USE_OPENMP)
    target_link_libraries(backend PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus/source_base/module_container
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/vbcsr/
)

add_executable(test_atomic_data test/test_atomic_data.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_atomic_data PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB})

add_executable(test_image_container test/test_image_container.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_image_container PUBLIC backend MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB})

add_executable(test_operator_construction test/test_operator_construction.cpp $<TARGET_OBJECTS:backend>)
target_include_directories(test_operator_construction PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus/source_base/module_container
)
target_link_libraries(test_operator_construction PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io base numerical_atomic_orbitals estate LAPACK::LAPACK BLAS::BLAS FFTW::Double)