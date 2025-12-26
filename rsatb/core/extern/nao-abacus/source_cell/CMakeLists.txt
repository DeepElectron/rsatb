add_library(cell
    atom_pseudo.cpp
    pseudo.cpp
    read_pp.cpp
    read_pp_upf100.cpp
    read_pp_upf201.cpp
    read_pp_complete.cpp
    soc.cpp
)

target_link_libraries(cell
    PUBLIC
    base
    numerical_atomic_orbitals
    MPI::MPI_CXX
)

target_include_directories(cell
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/int2c>
)
