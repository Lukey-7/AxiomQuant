add_library(quant_backtest STATIC
    src/portfolio.cpp
    src/engine.cpp
)

target_include_directories(quant_backtest PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_backtest PUBLIC
    quant_data
    quant_indicators
)
