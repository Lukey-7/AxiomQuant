add_executable(quant_engine_cli
    src/main.cpp
)

target_include_directories(quant_engine_cli PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(quant_engine_cli PRIVATE
    quant_data
    quant_indicators
    quant_backtest
    quant_risk
    quant_simulation
    quant_optimization
    OpenMP::OpenMP_CXX
)
