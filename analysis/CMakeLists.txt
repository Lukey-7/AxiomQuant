add_library(quant_analysis STATIC
    src/walk_forward.cpp
)

target_include_directories(quant_analysis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_analysis PUBLIC
    quant_backtest
    quant_risk
    axiom_openmp
)
