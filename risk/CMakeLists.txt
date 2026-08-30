add_library(quant_risk STATIC
    src/metrics.cpp
    src/var_cvar.cpp
    src/report.cpp
)

target_include_directories(quant_risk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_risk PUBLIC
    quant_backtest
    quant_data
)
