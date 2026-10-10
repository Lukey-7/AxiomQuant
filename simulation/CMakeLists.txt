add_library(quant_simulation STATIC
    src/monte_carlo.cpp
)

target_include_directories(quant_simulation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_simulation PUBLIC
    quant_risk
    quant_data
    axiom_eigen
    axiom_openmp
)

# Keep the simulation's arithmetic identical on every platform. Apple Clang on arm64 (and GCC/Clang
# with -march=native) would otherwise fuse multiply-adds into FMAs, which changes the last bit of
# each path and breaks the exact match against the reference algorithm in test_monte_carlo.cpp.
if(NOT MSVC)
    set_source_files_properties(src/monte_carlo.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()
