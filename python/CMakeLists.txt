# Python bindings (pybind11). Only configured with -DAXIOM_BUILD_PYTHON=ON, so the core build keeps
# no dependency beyond a C++20 compiler.
#
# pybind11 is located through its CMake package. A pip install provides one; when pybind11_DIR is
# not set, ask the selected interpreter where it lives (`python -m pybind11 --cmakedir`).

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _axiom_pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _axiom_pybind11_status
        ERROR_QUIET
    )
    if(_axiom_pybind11_status EQUAL 0)
        set(pybind11_DIR "${_axiom_pybind11_dir}" CACHE PATH "pybind11 CMake package directory")
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "AxiomQuant: Python module for ${Python_EXECUTABLE} (pybind11 ${pybind11_VERSION})")

pybind11_add_module(axiomquant_python src/module.cpp)
set_target_properties(axiomquant_python PROPERTIES
    OUTPUT_NAME axiomquant
    # $<1:...> stops multi-config generators (MSVC) from appending a Release/ or Debug/ folder,
    # so the module is always at <build>/python and PYTHONPATH can point there.
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/python>"
)
target_link_libraries(axiomquant_python PRIVATE
    quant_data
    quant_backtest
    quant_risk
    quant_analysis
)

# MinGW: CPython does not search PATH for an extension's dependencies. Link the C++ runtime
# statically, and copy the OpenMP runtime (which FindOpenMP links as a DLL) next to the module,
# where Windows does look.
if(MINGW)
    target_link_options(axiomquant_python PRIVATE -static)
    get_filename_component(_axiom_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(_dll libgomp-1.dll libwinpthread-1.dll)
        if(TARGET OpenMP::OpenMP_CXX AND EXISTS "${_axiom_mingw_bin}/${_dll}")
            add_custom_command(TARGET axiomquant_python POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_axiom_mingw_bin}/${_dll}"
                        "${CMAKE_BINARY_DIR}/python/${_dll}"
            )
        endif()
    endforeach()
endif()

if(AXIOM_BUILD_TESTS)
    set(_axiom_python_env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        "AXIOMQUANT_DATA=${PROJECT_SOURCE_DIR}/sample_data"
    )
    if(AXIOM_BUILD_CLI)
        list(APPEND _axiom_python_env "AXIOMQUANT_CLI=$<TARGET_FILE:quant_engine_cli>")
    endif()
    add_test(NAME python_bindings
        COMMAND "${Python_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -v
    )
    set_tests_properties(python_bindings PROPERTIES ENVIRONMENT "${_axiom_python_env}")
endif()
