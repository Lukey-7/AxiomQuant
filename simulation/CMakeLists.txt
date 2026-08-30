add_library(quant_simulation STATIC
    src/monte_carlo.cpp
)

target_include_directories(quant_simulation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(quant_simulation PUBLIC
    quant_risk
    quant_data
    OpenMP::OpenMP_CXX
)
