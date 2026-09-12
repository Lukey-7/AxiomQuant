add_library(quant_optimization STATIC
    src/portfolio_stats.cpp
    src/unconstrained.cpp
    src/constrained_qp.cpp
    src/efficient_frontier.cpp
)

target_include_directories(quant_optimization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_optimization PUBLIC
    quant_data
    quant_risk
    axiom_eigen
)
