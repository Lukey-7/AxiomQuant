# Runs the CLI on the sample data with fixed settings and compares its output with tests/golden/expected.
#
#   cmake -DCLI=<axiomquant> -DDIFF=<golden_diff> -DDATA=<sample_data> -DGOLDEN=<tests/golden/expected>
#         -DWORK=<scratch dir> [-DUPDATE=ON] -P run_golden.cmake
#
# With UPDATE=ON the fresh output replaces the golden copies instead of being compared with them.

foreach(var CLI DIFF DATA GOLDEN WORK)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "run_golden.cmake: -D${var}=... is required")
    endif()
endforeach()

# Every pinned figure depends on these; keep them in sync with the golden copies.
set(CLI_ARGS --data ${DATA} --no-db --paths 2000 --bootstrap 1000 --seed 42)
set(PINNED_EXPORTS
    strategy_summary.csv
    capacity.csv
    parameter_sweep.csv
    walk_forward.csv
    regime_breakdown.csv
    portfolio_weights.csv
    efficient_frontier.csv
)
# Lines that legitimately differ between runs and machines.
set(IGNORE_ARGS
    --ignore "Parallelism:"
    --ignore "Ingesting market data from"
    --ignore " ms"
    --ignore "Computation Time"
    --ignore "Exported CSV results to"
)

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK}/exports)
execute_process(
    COMMAND ${CLI} ${CLI_ARGS} --export-dir ${WORK}/exports
    OUTPUT_FILE ${WORK}/stdout.txt
    ERROR_VARIABLE cli_stderr
    RESULT_VARIABLE cli_status
)
if(NOT cli_status EQUAL 0)
    message(FATAL_ERROR "axiomquant exited with ${cli_status}:\n${cli_stderr}")
endif()

set(files stdout.txt)
foreach(name ${PINNED_EXPORTS})
    list(APPEND files exports/${name})
endforeach()

if(UPDATE)
    foreach(name ${files})
        get_filename_component(leaf ${name} NAME)
        execute_process(COMMAND ${DIFF} ${IGNORE_ARGS} --write ${WORK}/${name} ${GOLDEN}/${leaf}
                        RESULT_VARIABLE write_status)
        if(NOT write_status EQUAL 0)
            message(FATAL_ERROR "could not write ${GOLDEN}/${leaf}")
        endif()
    endforeach()
    message(STATUS "Golden copies in ${GOLDEN} regenerated; review the diff before committing it.")
    return()
endif()

set(failed "")
foreach(name ${files})
    get_filename_component(leaf ${name} NAME)
    execute_process(
        COMMAND ${DIFF} ${IGNORE_ARGS} ${GOLDEN}/${leaf} ${WORK}/${name}
        RESULT_VARIABLE diff_status
    )
    if(NOT diff_status EQUAL 0)
        list(APPEND failed ${leaf})
    endif()
endforeach()

if(failed)
    message(FATAL_ERROR "Output differs from the golden copy in: ${failed}")
endif()
