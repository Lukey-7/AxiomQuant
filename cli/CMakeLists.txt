add_executable(quant_engine_cli
    src/main.cpp
)

set_target_properties(quant_engine_cli PROPERTIES OUTPUT_NAME axiomquant)

target_link_libraries(quant_engine_cli PRIVATE
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
