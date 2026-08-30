#pragma once

#include "quant/data/types.hpp"
#include <string>
#include <filesystem>
#include <unordered_map>

namespace quant::data {

class CsvLoader {
public:
    /**
     * @brief Load a single ticker's historical OHLCV data from a CSV file.
     * @param filepath Path to the CSV file.
     * @param ticker Symbol/ticker for the series.
     * @return Populated TimeSeries object.
     */
    static TimeSeries load_file(const std::filesystem::path& filepath, const std::string& ticker);

    /**
     * @brief Load multiple CSV files from a directory. Assumes filename without extension is ticker symbol.
     * @param directory_path Path to directory containing CSV files.
     * @return Map of ticker symbol to TimeSeries.
     */
    static std::unordered_map<std::string, TimeSeries> load_directory(const std::filesystem::path& directory_path);

    /**
     * @brief Parses an ISO-8601 date string (YYYY-MM-DD) to Unix epoch seconds.
     */
    static int64_t parse_date_to_timestamp(const std::string& date_str);
};

} // namespace quant::data
