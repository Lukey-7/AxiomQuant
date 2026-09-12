cmake_minimum_required(VERSION 3.20)
project(AxiomQuant
    VERSION 1.1.0
    DESCRIPTION "High-performance quantitative research engine: backtesting, risk, Monte Carlo and portfolio optimization"
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(AXIOM_BUILD_TESTS   "Build the unit test suite"               ON)
option(AXIOM_BUILD_CLI     "Build the axiomquant command-line tool"  ON)
option(AXIOM_ENABLE_OPENMP "Parallelize simulations with OpenMP"     ON)
option(AXIOM_NATIVE        "Tune code generation for the host CPU"   OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Eigen 3.4 (header-only, vendored). Included as SYSTEM so its internals don't trip our warnings.
add_library(axiom_eigen INTERFACE)
target_include_directories(axiom_eigen SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3")

# OpenMP is optional: without it every parallel region runs serially with identical results.
add_library(axiom_openmp INTERFACE)
if(AXIOM_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(axiom_openmp INTERFACE OpenMP::OpenMP_CXX)
        message(STATUS "AxiomQuant: OpenMP ${OpenMP_CXX_VERSION} enabled")
    else()
        message(STATUS "AxiomQuant: OpenMP not found - building serial simulation kernels")
    endif()
endif()

# Vendored SQLite is compiled before our warning flags are applied.
add_subdirectory(third_party)

# ---------------------------------------------------------------------------
# Project-wide compile flags (apply to targets defined below this point)
# ---------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /Zc:__cplusplus)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(AXIOM_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# ---------------------------------------------------------------------------
# Engine modules
# ---------------------------------------------------------------------------
add_subdirectory(data)
add_subdirectory(indicators)
add_subdirectory(backtest)
add_subdirectory(risk)
add_subdirectory(simulation)
add_subdirectory(optimization)
add_subdirectory(analysis)

if(AXIOM_BUILD_CLI)
    add_subdirectory(cli)
endif()

if(AXIOM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
