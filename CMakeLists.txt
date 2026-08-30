cmake_minimum_required(VERSION 3.20)
project(AxiomQuant VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings and optimization flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# OpenMP for parallel simulations
find_package(OpenMP REQUIRED)

# Eigen3 linear algebra library
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3")

# Core engine modules
add_subdirectory(third_party)
add_subdirectory(data)
add_subdirectory(indicators)
add_subdirectory(backtest)
add_subdirectory(risk)
add_subdirectory(simulation)
add_subdirectory(optimization)
add_subdirectory(cli)
add_subdirectory(tests)
