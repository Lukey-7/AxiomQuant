cmake_minimum_required(VERSION 3.20)
project(quant_third_party C CXX)

add_library(sqlite3_lib STATIC sqlite3/sqlite3.c)
target_include_directories(sqlite3_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/sqlite3")

find_package(Threads REQUIRED)
target_link_libraries(sqlite3_lib PUBLIC Threads::Threads)
