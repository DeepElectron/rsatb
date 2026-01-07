cmake_minimum_required(VERSION 3.15)
project(test_install LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(int2c REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build/install)

add_executable(test_install main.cpp)
target_link_libraries(test_install PRIVATE int2c::estate)
