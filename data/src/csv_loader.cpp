#include "quant/data/csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <optional>

namespace quant::data {

namespace {

std::string trim(const std::string& str) {
    const auto start = str.find_first_not_of(" \t\r\n\"");
    if (start == std::string::npos) return "";
    const auto end = str.find_last_not_of(" \t\r\n\"");
    return str.substr(start, end - start + 1);
}

std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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

/// Parses a whole field as a number. std::stod alone accepts a numeric prefix, which would read
/// "1,000.5" (a thousands separator inside quotes) as 1.0; any trailing text is an error instead.
double parse_number(const std::string& field) {
    size_t consumed = 0;
    const double value = std::stod(field, &consumed);
    if (consumed != field.size()) throw std::invalid_argument("trailing characters in number: " + field);
    return value;
}

/// Parses the leading YYYY-MM-DD (or YYYY/MM/DD) of `text` to Unix seconds at midnight UTC.
/// Anything after the date (e.g. a time of day) is ignored. Returns nullopt for anything that is not
/// a real calendar date, so 2021-02-30 is rejected rather than silently rolled into March.
std::optional<int64_t> parse_iso_date(const std::string& text) {
    const std::string s = trim(text);
    if (s.size() < 10) return std::nullopt;
    const auto digit = [&s](size_t i) { return std::isdigit(static_cast<unsigned char>(s[i])) != 0; };
    static constexpr size_t kDigitPositions[] = {0, 1, 2, 3, 5, 6, 8, 9};
    for (const size_t i : kDigitPositions) {
        if (!digit(i)) return std::nullopt;
    }
    if ((s[4] != '-' && s[4] != '/') || s[7] != s[4]) return std::nullopt;

    const auto number = [&s](size_t pos, size_t len) {
        int value = 0;
        for (size_t i = pos; i < pos + len; ++i)
            value = value * 10 + (s[i] - '0');
        return value;
    };
    const int year = number(0, 4);
    const int month = number(5, 2);
    const int day = number(8, 2);
    if (month < 1 || month > 12 || day < 1) return std::nullopt;
    static constexpr int kDaysInMonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    if (day > kDaysInMonth[month - 1] + ((month == 2 && leap) ? 1 : 0)) return std::nullopt;

    // Days since 1970-01-01 in the proleptic Gregorian calendar (H. Hinnant's days_from_civil), so
    // the result does not depend on the platform's timegm/_mkgmtime range limits.
    const int y = year - (month <= 2 ? 1 : 0);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const int yoe = y - era * 400;
    const int doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const int64_t days = static_cast<int64_t>(era) * 146097 + doe - 719468;
    return days * 86400;
}

}   // namespace

int64_t CsvLoader::parse_date_to_timestamp(const std::string& date_str) {
    return parse_iso_date(date_str).value_or(0);
}

TimeSeries CsvLoader::load_file(const std::filesystem::path& filepath, const std::string& ticker) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filepath.string());
    }
    return load_stream(file, ticker, filepath.string());
}

TimeSeries CsvLoader::load_stream(std::istream& file, const std::string& ticker, const std::string& source) {
    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("CSV file is empty: " + source);
    }

    auto headers = split_csv_line(header_line);
    int date_idx = -1, open_idx = -1, high_idx = -1, low_idx = -1, close_idx = -1, adj_close_idx = -1,
        vol_idx = -1;

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string col = to_lower(headers[i]);
        if (col == "date" || col == "timestamp" || col == "datetime") date_idx = static_cast<int>(i);
        else if (col == "open") open_idx = static_cast<int>(i);
        else if (col == "high") high_idx = static_cast<int>(i);
        else if (col == "low") low_idx = static_cast<int>(i);
        else if (col == "close") close_idx = static_cast<int>(i);
        else if (col == "adj close" || col == "adjclose" || col == "adjusted_close")
            adj_close_idx = static_cast<int>(i);
        else if (col == "volume" || col == "vol") vol_idx = static_cast<int>(i);
    }

    if (date_idx == -1 || close_idx == -1) {
        throw std::runtime_error("CSV must contain at least 'Date' and 'Close' columns: " + source);
    }

    std::vector<Bar> raw_bars;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tokens = split_csv_line(line);
        if (tokens.size() <=
            static_cast<size_t>(std::max({date_idx, open_idx, high_idx, low_idx, close_idx}))) {
            continue;
        }

        try {
            Bar bar;
            bar.date = tokens[date_idx];
            const auto timestamp = parse_iso_date(bar.date);
            if (!timestamp) continue;
            bar.timestamp = *timestamp;
            bar.close = parse_number(tokens[close_idx]);
            bar.open = (open_idx != -1 && open_idx < static_cast<int>(tokens.size()))
                           ? parse_number(tokens[open_idx])
                           : bar.close;
            bar.high = (high_idx != -1 && high_idx < static_cast<int>(tokens.size()))
                           ? parse_number(tokens[high_idx])
                           : std::max(bar.open, bar.close);
            bar.low = (low_idx != -1 && low_idx < static_cast<int>(tokens.size()))
                          ? parse_number(tokens[low_idx])
                          : std::min(bar.open, bar.close);
            bar.adj_close = (adj_close_idx != -1 && adj_close_idx < static_cast<int>(tokens.size()))
                                ? parse_number(tokens[adj_close_idx])
                                : bar.close;
            bar.volume = (vol_idx != -1 && vol_idx < static_cast<int>(tokens.size()))
                             ? parse_number(tokens[vol_idx])
                             : 0.0;

            if (bar.is_valid()) {
                raw_bars.push_back(bar);
            }
        } catch (...) {
            // Skip bad rows
            continue;
        }
    }

    // Chronological order; a date that appears twice keeps its first row in file order.
    std::stable_sort(raw_bars.begin(), raw_bars.end(),
                     [](const Bar& a, const Bar& b) { return a.timestamp < b.timestamp; });
    raw_bars.erase(std::unique(raw_bars.begin(), raw_bars.end(),
                               [](const Bar& a, const Bar& b) { return a.timestamp == b.timestamp; }),
                   raw_bars.end());

    TimeSeries ts(ticker);
    ts.reserve(raw_bars.size());
    for (const auto& bar : raw_bars) {
        ts.push_back(bar);
    }

    return ts;
}

std::unordered_map<std::string, TimeSeries> CsvLoader::load_directory(
    const std::filesystem::path& directory_path) {
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

}   // namespace quant::data
