cmake_minimum_required(VERSION 3.15...4.0)

# project settings
project(
  int2c 
  LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(GNUInstallDirs)

if (PROJECT_IS_TOP_LEVEL)
  find_package(GTest REQUIRED)
  find_package(MPI REQUIRED)
endif()
# # include_directories(${MPI_CXX_INCLUDE_PATH})
# add_compile_definitions(__MPI)
# add_compile_definitions(__LCAO)

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
        set(INT2C_USE_MKL ON)
        # add_definitions(-D__MKL)
        # include_directories(${MKL_INCLUDE} ${MKL_INCLUDE}/fftw)

        list(APPEND math_libs IntelMKL::MKL)

        if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
          list(APPEND math_libs -lifcore)
        endif()
    else()
        find_package(LAPACK REQUIRED)
        
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(FFTW3 REQUIRED fftw3)
        set(INT2C_USE_FFTW ON)
        
        # add_compile_definitions(__FFTW)
        if(NOT TARGET FFTW::FFTW)
          add_library(FFTW::FFTW INTERFACE IMPORTED)
          set_target_properties(FFTW::FFTW PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}" 
            INTERFACE_LINK_LIBRARIES     "${FFTW3_LIBRARIES}")
        endif()
        # include_directories(${FFTW3_INCLUDE_DIRS})
        message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW3_INCLUDE_DIRS}")
        list(APPEND math_libs FFTW::FFTW LAPACK::LAPACK)
    endif()
endif()

# add include directories

# include_directories(
#     ${BASE_PATH} 
#     ${SOURCE_DIR} 
#     ${SOURCE_DIR}/source_base/module_container 
#     ${SOURCE_DIR}/source_base/module_device 
#     )

# add basic libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add base
set(BASE_BINARY_DIR "${PROJECT_SOURCE_DIR}/build/base")
add_subdirectory(${SOURCE_DIR}/source_base ${BASE_BINARY_DIR})
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(ORB_BINARY_DIR "${PROJECT_SOURCE_DIR}/build/orb")
add_subdirectory(${SOURCE_DIR}/source_basis/module_nao ${ORB_BINARY_DIR})

set(CELL_BINARY_DIR "${PROJECT_SOURCE_DIR}/build/cell")
add_subdirectory(${SOURCE_DIR}/source_cell ${CELL_BINARY_DIR})

set(ESTATE_BINARY_DIR "${PROJECT_SOURCE_DIR}/build/estate")
add_subdirectory(${SOURCE_DIR}/source_estate ${ESTATE_BINARY_DIR})

add_executable(test_nao test/test.cpp)
target_link_libraries(test_nao PRIVATE base numerical_atomic_orbitals MPI::MPI_CXX GTest::gtest)

add_executable(test_pos test/test_pos.cpp)
target_link_libraries(test_pos PRIVATE base numerical_atomic_orbitals MPI::MPI_CXX GTest::gtest)

add_executable(test_pseudo test/test_pseudo.cpp)
target_link_libraries(test_pseudo PRIVATE base numerical_atomic_orbitals estate cell MPI::MPI_CXX GTest::gtest)
target_compile_definitions(test_pseudo PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/pporb/")

add_executable(test_two_center_bundle test/test_two_center_bundle.cpp)
target_link_libraries(test_two_center_bundle PRIVATE base numerical_atomic_orbitals estate cell MPI::MPI_CXX GTest::gtest)
target_compile_definitions(test_two_center_bundle PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/pporb/")

add_executable(test_api_usage test/test_api_usage.cpp)
target_link_libraries(test_api_usage PRIVATE base numerical_atomic_orbitals estate cell MPI::MPI_CXX GTest::gtest)
target_compile_definitions(test_api_usage PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/pporb/")

add_executable(test_op2c test/test_op2c.cpp)
target_link_libraries(test_op2c PRIVATE base numerical_atomic_orbitals estate cell MPI::MPI_CXX GTest::gtest)
target_compile_definitions(test_op2c PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/pporb/")

# --- Target-based definitions ---
target_compile_definitions(base PUBLIC __MPI __LCAO)
if(INT2C_USE_MKL)
    target_compile_definitions(base PUBLIC __MKL)
endif()
if(INT2C_USE_FFTW)
    target_compile_definitions(base PUBLIC __FFTW)
endif()


# --- Installation ---

install(TARGETS container base numerical_atomic_orbitals cell estate
    EXPORT int2cTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT int2cTargets
    FILE int2cTargets.cmake
    NAMESPACE int2c::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/int2c
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/int2cConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/int2cConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/int2cConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/int2c
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/int2cConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/int2cConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/int2c
)

# Install headers
install(DIRECTORY source_base/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/int2c/source_base FILES_MATCHING PATTERN "*.h")
install(DIRECTORY source_basis/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/int2c/source_basis FILES_MATCHING PATTERN "*.h")
install(DIRECTORY source_cell/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/int2c/source_cell FILES_MATCHING PATTERN "*.h")
install(DIRECTORY source_estate/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/int2c/source_estate FILES_MATCHING PATTERN "*.h")
install(FILES source_cell/read_pp.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/int2c/source_cell)

