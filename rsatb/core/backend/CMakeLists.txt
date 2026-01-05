cmake_minimum_required(VERSION 3.15...4.0)
project(backend LANGUAGES CXX Fortran C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    find_package(MPI REQUIRED)
    find_package(GTest REQUIRED)
    find_package(Threads REQUIRED)
    # find_package(PETSc REQUIRED)
    
    # ParMETIS
    find_library(PARMETIS_LIB parmetis HINTS $ENV{CONDA_PREFIX}/lib)
    find_library(METIS_LIB metis HINTS $ENV{CONDA_PREFIX}/lib)
    include_directories($ENV{CONDA_PREFIX}/include)

    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../../../deps/install")
    find_package(DBCSR 2.0.0 CONFIG REQUIRED)
    find_package(BLAS REQUIRED)
endif()
# Wrap into a modern imported target so the rest of your code can stay target-based
# if(NOT TARGET PETSc::PETSc)
# add_library(PETSc::PETSc INTERFACE IMPORTED)
# set_target_properties(PETSc::PETSc PROPERTIES
#   INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSCCONF_INCLUDE_DIR}"
#   INTERFACE_LINK_LIBRARIES     "${PETSC_LIBRARIES}")
# endif()

add_library(backend OBJECT operator_construction.hpp)
target_compile_features(backend PUBLIC cxx_std_11)

target_link_libraries(backend PUBLIC MPI::MPI_CXX Threads::Threads DBCSR::dbcsr_c BLAS::BLAS)
target_include_directories(backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus 
    ${CMAKE_CURRENT_SOURCE_DIR}/../extern/nao-abacus/source_base/module_container
)

# add_executable(test_backend ./test.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_backend PUBLIC backend GTest::gtest)

# add_executable(test_dm_atomic ./test_dmplex.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_dm_atomic PUBLIC backend GTest::gtest)

# add_executable(test_complex_dist_vector ./test_complex_dist_vector.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_complex_dist_vector PUBLIC backend MPI::MPI_CXX)

# add_executable(test_block_csr ./test_block_csr.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_block_csr PUBLIC backend MPI::MPI_CXX)

# add_executable(test_dbcsr_conversions ./test_dbcsr_conversions.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_dbcsr_conversions PUBLIC backend MPI::MPI_CXX)

# add_executable(test_isqrtm ../calc_/test_isqrtm.cpp $<TARGET_OBJECTS:backend>)
# target_link_libraries(test_isqrtm PUBLIC backend MPI::MPI_CXX)
# target_include_directories(test_isqrtm PUBLIC ../calc_)

add_executable(test_atomic_data test/test_atomic_data.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_atomic_data PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB})

add_executable(test_image_container test/test_image_container.cpp $<TARGET_OBJECTS:backend>)
target_link_libraries(test_image_container PUBLIC backend MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB})