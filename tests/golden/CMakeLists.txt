# Golden-output regression test: the full CLI output on sample_data, pinned in expected/.
# Any numeric change outside one unit in the last printed digit fails it. After an intended change:
#   cmake --build build --target update_golden     (then review and commit the diff)

add_executable(golden_diff golden_diff.cpp)

set(GOLDEN_SCRIPT_ARGS
    -DCLI=$<TARGET_FILE:quant_engine_cli>
    -DDIFF=$<TARGET_FILE:golden_diff>
    -DDATA=${PROJECT_SOURCE_DIR}/sample_data
    -DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/expected
    -DWORK=${CMAKE_CURRENT_BINARY_DIR}/golden_run
)

add_test(NAME golden_output
    COMMAND ${CMAKE_COMMAND} ${GOLDEN_SCRIPT_ARGS} -P ${CMAKE_CURRENT_SOURCE_DIR}/run_golden.cmake
)

add_custom_target(update_golden
    COMMAND ${CMAKE_COMMAND} ${GOLDEN_SCRIPT_ARGS} -DUPDATE=ON -P ${CMAKE_CURRENT_SOURCE_DIR}/run_golden.cmake
    DEPENDS quant_engine_cli golden_diff
    COMMENT "Regenerating the golden CLI output in tests/golden/expected"
    VERBATIM
)
