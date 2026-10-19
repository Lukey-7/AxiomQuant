# CSV loader fuzzing.
#
# fuzz_csv_loader_replay is built with every test build: it replays the seed corpus and a fixed number
# of seeded random mutations, so the target runs under every compiler (and under ASan/UBSan in the
# sanitizer job) without needing libFuzzer.
#
# With -DAXIOM_BUILD_FUZZERS=ON and Clang, fuzz_csv_loader is the real coverage-guided libFuzzer binary:
#   ./build/bin/fuzz_csv_loader -max_total_time=120 tests/fuzz/corpus sample_data

add_executable(fuzz_csv_loader_replay fuzz_csv_loader.cpp standalone_driver.cpp)
target_link_libraries(fuzz_csv_loader_replay PRIVATE quant_data)

add_test(NAME fuzz_csv_loader_replay
    COMMAND fuzz_csv_loader_replay --mutations 3000 --seed 1
            ${CMAKE_CURRENT_SOURCE_DIR}/corpus
)

if(AXIOM_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "AXIOM_BUILD_FUZZERS needs Clang (libFuzzer); got ${CMAKE_CXX_COMPILER_ID}")
    endif()
    add_executable(fuzz_csv_loader fuzz_csv_loader.cpp)
    target_link_libraries(fuzz_csv_loader PRIVATE quant_data)
    target_link_options(fuzz_csv_loader PRIVATE -fsanitize=fuzzer)
endif()
