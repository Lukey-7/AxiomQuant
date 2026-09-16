add_library(quant_data STATIC
    src/csv_loader.cpp
    src/membership.cpp
    src/universe.cpp
    src/sqlite_storage.cpp
)

target_include_directories(quant_data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quant_data PUBLIC
    sqlite3_lib
)
