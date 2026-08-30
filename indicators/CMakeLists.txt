add_library(quant_indicators STATIC
    src/indicators.cpp
)

target_include_directories(quant_indicators PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_indicators PUBLIC
    quant_data
)
