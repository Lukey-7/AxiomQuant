add_executable(quant_tests
    test_main.cpp
    test_data.cpp
    test_indicators.cpp
    test_backtest.cpp
    test_risk.cpp
    test_monte_carlo.cpp
    test_optimization.cpp
)

target_include_directories(quant_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(quant_tests PRIVATE
    quant_data
    quant_indicators
    quant_backtest
    quant_risk
    quant_simulation
    quant_optimization
    OpenMP::OpenMP_CXX
)
