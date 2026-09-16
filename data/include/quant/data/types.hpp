#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <iomanip>
#include <sstream>

namespace quant::data {

/**
 * @brief Represents a single price candle / OHLCV bar.
 */
struct Bar {
    std::string date;       // ISO-8601 string, e.g. "2023-01-03"
    int64_t timestamp{0};   // Unix timestamp in seconds
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double adj_close{0.0};
    double volume{0.0};

    [[nodiscard]] bool is_valid() const noexcept {
        return open > 0.0 && high >= low && high >= open && high >= close && low <= open && low <= close &&
               close > 0.0 && !std::isnan(close) && !std::isinf(close);
    }
};

/**
 * @brief Columnar time series data structure for high-performance vectorized operations.
 */
class TimeSeries {
public:
    std::string ticker;
    std::vector<std::string> dates;
    std::vector<int64_t> timestamps;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> adj_close;
    std::vector<double> volume;

    TimeSeries() = default;
    explicit TimeSeries(std::string symbol) : ticker(std::move(symbol)) {}

    void push_back(const Bar& bar) {
        dates.push_back(bar.date);
        timestamps.push_back(bar.timestamp);
        open.push_back(bar.open);
        high.push_back(bar.high);
        low.push_back(bar.low);
        close.push_back(bar.close);
        adj_close.push_back(bar.adj_close);
        volume.push_back(bar.volume);
    }

    void reserve(size_t n) {
        dates.reserve(n);
        timestamps.reserve(n);
        open.reserve(n);
        high.reserve(n);
        low.reserve(n);
        close.reserve(n);
        adj_close.reserve(n);
        volume.reserve(n);
    }

    void clear() noexcept {
        dates.clear();
        timestamps.clear();
        open.clear();
        high.clear();
        low.clear();
        close.clear();
        adj_close.clear();
        volume.clear();
    }

    [[nodiscard]] size_t size() const noexcept { return close.size(); }
    [[nodiscard]] bool empty() const noexcept { return close.empty(); }

    [[nodiscard]] Bar get_bar(size_t index) const {
        if (index >= size()) {
            throw std::out_of_range("TimeSeries index out of range");
        }
        return Bar{dates[index], timestamps[index], open[index],      high[index],
                   low[index],   close[index],      adj_close[index], volume[index]};
    }

    /**
     * @brief Computes simple percentage returns: R_t = (P_t - P_{t-1}) / P_{t-1}
     */
    [[nodiscard]] std::vector<double> simple_returns() const {
        if (close.size() < 2) return {};
        std::vector<double> ret(close.size() - 1);
        for (size_t i = 1; i < close.size(); ++i) {
            ret[i - 1] = (close[i] - close[i - 1]) / close[i - 1];
        }
        return ret;
    }

    /**
     * @brief Computes log returns: r_t = ln(P_t / P_{t-1})
     */
    [[nodiscard]] std::vector<double> log_returns() const {
        if (close.size() < 2) return {};
        std::vector<double> ret(close.size() - 1);
        for (size_t i = 1; i < close.size(); ++i) {
            ret[i - 1] = std::log(close[i] / close[i - 1]);
        }
        return ret;
    }
};

/**
 * @brief Represents a multi-asset cross-sectional market snapshot at a specific point in time.
 */
struct MarketSnapshot {
    std::string date;
    int64_t timestamp{0};
    std::unordered_map<std::string, Bar> bars;

    [[nodiscard]] bool has_ticker(const std::string& ticker) const { return bars.find(ticker) != bars.end(); }

    [[nodiscard]] const Bar& get_bar(const std::string& ticker) const {
        auto it = bars.find(ticker);
        if (it == bars.end()) {
            throw std::invalid_argument("Ticker not found in MarketSnapshot: " + ticker);
        }
        return it->second;
    }
};

}   // namespace quant::data
