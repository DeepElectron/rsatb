add_library(estate
    read_pseudo.cpp
    cal_wfc.cpp
    op2c.cpp
)

target_link_libraries(estate
    PUBLIC
    base
    numerical_atomic_orbitals
    cell
    MPI::MPI_CXX
)

target_include_directories(estate
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/int2c>
)
