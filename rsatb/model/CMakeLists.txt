cmake_minimum_required(VERSION 3.15...4.0)
project(model LANGUAGES CXX Fortran C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

if (PROJECT_IS_TOP_LEVEL)
    # Standalone build: the parent rsatb/CMakeLists.txt normally supplies this.
    set(RSATB_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../test/data"
        CACHE PATH "Directory containing the rsatb test data tree")
    add_compile_definitions(RSATB_TEST_DATA_DIR="${RSATB_TEST_DATA_DIR}")

    find_package(MPI REQUIRED)
    find_package(Threads REQUIRED)
    if(RSATB_BUILD_TESTS)
        find_package(GTest REQUIRED)
    endif()
    
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

# One object target over the header tree. The layering it compiles is
# operator/ -> model/ -> setup/, one way; see doc/operator_layer_design.md.
add_library(
    backend OBJECT
    # operator/: what an operator is, and the constructions that need no model
    ../operator/types.hpp
    ../operator/geometry.hpp
    ../operator/block_assembly.hpp
    ../operator/commutator_velocity.hpp
    ../operator/diagonal_position.hpp
    ../operator/lattice_moment.hpp
    ../operator/builders.hpp
    # model/: the interface, what every family shares, and one dir per family
    setup.hpp
    basis_spec.hpp
    model.hpp
    operator_source.hpp
    hamiltonian_images.hpp
    lcao/lcao_model.hpp
    lcao/op2c_engine.hpp
    tbfile/tb_file_model.hpp
    sktb/orbital_labels.hpp
    sktb/radial.hpp
    sktb/sk_table.hpp
    sktb/skf_table.hpp
    sktb/sktb_model.hpp
    sktb/sktb_construction.hpp
    # setup/: assembling and caching them
    ../setup/operator_set.hpp
    ../setup/operator_assembly.hpp
    ../setup/kpoint_partition.hpp
    ../setup/spectral_bounds.hpp
    ../calc/observables.hpp
)
set_target_properties(backend PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(backend PUBLIC cxx_std_17)
target_compile_options(backend PUBLIC -mavx2 -mfma)

target_link_libraries(backend PUBLIC MPI::MPI_CXX Threads::Threads BLAS::BLAS LAPACK::LAPACK vbcsr)
if(TARGET primme)
    target_link_libraries(backend PUBLIC primme)
elseif(PRIMME_LIB)
    target_link_libraries(backend PUBLIC ${PRIMME_LIB})
endif()
if(USE_OPENMP)
    target_link_libraries(backend PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sktb
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


if(RSATB_BUILD_TESTS)
    add_executable(test_op2c_engine lcao/test/test_op2c_engine.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_op2c_engine PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src 
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    )
    target_link_libraries(test_op2c_engine PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)

    add_executable(test_operator_set test/test_operator_set.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_operator_set PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    )
    target_link_libraries(test_operator_set PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)

    add_executable(test_sktb sktb/test/test_sktb.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_sktb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    )
    target_link_libraries(test_sktb PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)

    add_executable(test_setup_stages ../setup/test/test_setup_stages.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_setup_stages PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    )
    target_link_libraries(test_setup_stages PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)

    add_executable(test_collocation ../calc/test/test_collocation.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_collocation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    )
    target_link_libraries(test_collocation PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)

    add_executable(test_skf sktb/test/test_skf.cpp $<TARGET_OBJECTS:backend>)
    target_include_directories(test_skf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/op2c/src/utils/container
    )
    target_link_libraries(test_skf PUBLIC backend GTest::gtest MPI::MPI_CXX ${PARMETIS_LIB} ${METIS_LIB} io nao_int2c nao_pseudo nao_core nao_math nao_utils LAPACK::LAPACK BLAS::BLAS FFTW::Double)
endif()
