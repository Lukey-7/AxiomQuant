#include "quant/data/csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <ctime>

namespace quant::data {

namespace {

std::string trim(const std::string& str) {
    const auto start = str.find_first_not_of(" \t\r\n\"");
    if (start == std::string::npos) return "";
    const auto end = str.find_last_not_of(" \t\r\n\"");
    return str.substr(start, end - start + 1);
}

std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return str;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            tokens.push_back(trim(token));
            token.clear();
        } else {
            token += c;
        }
    }
    tokens.push_back(trim(token));
    return tokens;
}

} // namespace

int64_t CsvLoader::parse_date_to_timestamp(const std::string& date_str) {
    std::tm tm{};
    std::string s = trim(date_str);
    
    // Support YYYY-MM-DD or YYYY/MM/DD
    if (s.size() >= 10) {
        try {
            tm.tm_year = std::stoi(s.substr(0, 4)) - 1900;
            tm.tm_mon = std::stoi(s.substr(5, 2)) - 1;
            tm.tm_mday = std::stoi(s.substr(8, 2));
            tm.tm_hour = 0;
            tm.tm_min = 0;
            tm.tm_sec = 0;
            tm.tm_isdst = -1;
#if defined(_WIN32)
            return static_cast<int64_t>(_mkgmtime(&tm));
#else
            return static_cast<int64_t>(timegm(&tm));
#endif
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

TimeSeries CsvLoader::load_file(const std::filesystem::path& filepath, const std::string& ticker) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filepath.string());
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("CSV file is empty: " + filepath.string());
    }

    auto headers = split_csv_line(header_line);
    int date_idx = -1, open_idx = -1, high_idx = -1, low_idx = -1, close_idx = -1, adj_close_idx = -1, vol_idx = -1;

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string col = to_lower(headers[i]);
        if (col == "date" || col == "timestamp" || col == "datetime") date_idx = static_cast<int>(i);
        else if (col == "open") open_idx = static_cast<int>(i);
        else if (col == "high") high_idx = static_cast<int>(i);
        else if (col == "low") low_idx = static_cast<int>(i);
        else if (col == "close") close_idx = static_cast<int>(i);
        else if (col == "adj close" || col == "adjclose" || col == "adjusted_close") adj_close_idx = static_cast<int>(i);
        else if (col == "volume" || col == "vol") vol_idx = static_cast<int>(i);
    }

    if (date_idx == -1 || close_idx == -1) {
        throw std::runtime_error("CSV must contain at least 'Date' and 'Close' columns: " + filepath.string());
    }

    std::vector<Bar> raw_bars;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tokens = split_csv_line(line);
        if (tokens.size() <= static_cast<size_t>(std::max({date_idx, open_idx, high_idx, low_idx, close_idx}))) {
            continue;
        }

        try {
            Bar bar;
            bar.date = tokens[date_idx];
            bar.timestamp = parse_date_to_timestamp(bar.date);
            bar.close = std::stod(tokens[close_idx]);
            bar.open = (open_idx != -1 && open_idx < static_cast<int>(tokens.size())) ? std::stod(tokens[open_idx]) : bar.close;
            bar.high = (high_idx != -1 && high_idx < static_cast<int>(tokens.size())) ? std::stod(tokens[high_idx]) : std::max(bar.open, bar.close);
            bar.low = (low_idx != -1 && low_idx < static_cast<int>(tokens.size())) ? std::stod(tokens[low_idx]) : std::min(bar.open, bar.close);
            bar.adj_close = (adj_close_idx != -1 && adj_close_idx < static_cast<int>(tokens.size())) ? std::stod(tokens[adj_close_idx]) : bar.close;
            bar.volume = (vol_idx != -1 && vol_idx < static_cast<int>(tokens.size())) ? std::stod(tokens[vol_idx]) : 0.0;

            if (bar.is_valid()) {
                raw_bars.push_back(bar);
            }
        } catch (...) {
            // Skip bad rows
            continue;
        }
    }

    // Ensure chronological order
    std::sort(raw_bars.begin(), raw_bars.end(), [](const Bar& a, const Bar& b) {
        return a.timestamp < b.timestamp;
    });

    TimeSeries ts(ticker);
    ts.reserve(raw_bars.size());
    for (const auto& bar : raw_bars) {
        ts.push_back(bar);
    }

    return ts;
}

std::unordered_map<std::string, TimeSeries> CsvLoader::load_directory(const std::filesystem::path& directory_path) {
    if (!std::filesystem::exists(directory_path) || !std::filesystem::is_directory(directory_path)) {
        throw std::runtime_error("Directory does not exist: " + directory_path.string());
    }

    std::unordered_map<std::string, TimeSeries> dataset;
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string ticker = entry.path().stem().string();
            dataset[ticker] = load_file(entry.path(), ticker);
        }
    }

    return dataset;
}

} // namespace quant::data
