// Fuzz target for CsvLoader: arbitrary bytes in, and either a clean exception or a series in which
// every bar is usable by the rest of the engine. libFuzzer drives it in CI; standalone_driver.cpp
// replays a corpus (plus random mutations) everywhere else.

#include "quant/data/csv_loader.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>

namespace {

[[noreturn]] void fail(const char* what, size_t index) {
    std::fprintf(stderr, "csv loader invariant violated at bar %zu: %s\n", index, what);
    std::abort();
}

bool positive_finite(double value) {
    return std::isfinite(value) && value > 0.0;
}

void check_series(const quant::data::TimeSeries& series) {
    const size_t n = series.size();
    if (series.dates.size() != n || series.timestamps.size() != n || series.open.size() != n ||
        series.high.size() != n || series.low.size() != n || series.adj_close.size() != n ||
        series.volume.size() != n) {
        fail("column lengths differ", 0);
    }
    for (size_t i = 0; i < n; ++i) {
        if (!positive_finite(series.open[i])) fail("open is not a positive finite price", i);
        if (!positive_finite(series.high[i])) fail("high is not a positive finite price", i);
        if (!positive_finite(series.low[i])) fail("low is not a positive finite price", i);
        if (!positive_finite(series.close[i])) fail("close is not a positive finite price", i);
        if (!positive_finite(series.adj_close[i])) fail("adjusted close is not a positive finite price", i);
        if (!(series.high[i] >= series.low[i])) fail("high below low", i);
        if (!std::isfinite(series.volume[i]) || series.volume[i] < 0.0)
            fail("volume is negative or not finite", i);
        if (i > 0 && series.timestamps[i] <= series.timestamps[i - 1])
            fail("timestamps not strictly increasing", i);
    }
}

}   // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::istringstream input(std::string(reinterpret_cast<const char*>(data), size));
    try {
        const auto series = quant::data::CsvLoader::load_stream(input, "FUZZ", "<fuzz input>");
        check_series(series);
    } catch (const std::exception&) {
        // Rejecting malformed input with an exception is the documented behaviour.
    }
    return 0;
}
