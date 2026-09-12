add_library(sqlite3_lib STATIC sqlite3/sqlite3.c)
target_include_directories(sqlite3_lib SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/sqlite3")
target_compile_definitions(sqlite3_lib PRIVATE SQLITE_DQS=0 SQLITE_THREADSAFE=1)

# Vendored amalgamation: silence its warnings, they are not ours to fix.
if(MSVC)
    target_compile_options(sqlite3_lib PRIVATE /w)
else()
    target_compile_options(sqlite3_lib PRIVATE -w)
endif()

find_package(Threads REQUIRED)
target_link_libraries(sqlite3_lib PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
