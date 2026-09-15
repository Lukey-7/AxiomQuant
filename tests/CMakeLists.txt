add_executable(quant_tests
    test_main.cpp
    test_data.cpp
    test_indicators.cpp
    test_backtest.cpp
    test_risk.cpp
    test_monte_carlo.cpp
    test_optimization.cpp
    test_analysis.cpp
    test_significance.cpp
)

target_include_directories(quant_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(quant_tests PRIVATE
    quant_data
    quant_indicators
    quant_backtest
    quant_risk
    quant_simulation
    quant_optimization
    quant_analysis
    axiom_eigen
    axiom_openmp
)

add_test(NAME unit_tests COMMAND quant_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# End-to-end smoke test of the full CLI pipeline on the bundled sample universe.
if(TARGET quant_engine_cli)
    add_test(NAME cli_end_to_end
        COMMAND quant_engine_cli --data ${PROJECT_SOURCE_DIR}/sample_data --paths 2000 --no-db
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()
