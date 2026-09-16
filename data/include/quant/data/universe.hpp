#pragma once

#include "quant/data/types.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>

namespace quant::data {

class MarketDataUniverse {
public:
    MarketDataUniverse() = default;

    /**
     * @brief Adds a TimeSeries for a ticker to the universe.
     */
    void add_asset(const std::string& ticker, TimeSeries series);

    /**
     * @brief Aligns all added assets along a common timeline.
     * @param require_all_assets If true, only keeps dates where ALL assets have data (intersection).
     *                           If false, uses union and forward-fills missing prices.
     */
    void synchronize_timeline(bool require_all_assets = true);

    [[nodiscard]] const std::vector<std::string>& get_tickers() const noexcept { return tickers_; }
    [[nodiscard]] const std::vector<std::string>& get_timeline() const noexcept { return timeline_; }
    [[nodiscard]] size_t size() const noexcept { return timeline_.size(); }
    [[nodiscard]] bool empty() const noexcept { return timeline_.empty(); }

    [[nodiscard]] const std::vector<int64_t>& get_timestamps() const noexcept { return timestamps_; }
    [[nodiscard]] const TimeSeries& get_series(const std::string& ticker) const;
    [[nodiscard]] const MarketSnapshot& get_snapshot(size_t timeline_index) const;

    /**
     * @brief Close prices of one ticker aligned to the synchronized timeline (element t == timeline index t).
     *
     * Strategies that index indicators by timeline position must use this instead of get_series():
     * the raw series may contain dates that synchronization dropped, which would silently shift
     * every indicator value relative to the bar being traded. Missing bars are NaN.
     */
    [[nodiscard]] std::vector<double> get_aligned_closes(const std::string& ticker) const;

    /**
     * @brief Returns a synchronized sub-universe covering timeline positions [begin, end).
     *        Used for walk-forward windows; get_series() on the slice still returns the full raw series.
     */
    [[nodiscard]] MarketDataUniverse slice(size_t begin, size_t end) const;

    /**
     * @brief Returns a matrix of close prices (rows = timeline points, cols = tickers in get_tickers()
     * order).
     */
    [[nodiscard]] std::vector<std::vector<double>> get_aligned_close_matrix() const;

    /**
     * @brief Returns a matrix of simple percentage returns (rows = timeline points - 1, cols = tickers).
     */
    [[nodiscard]] std::vector<std::vector<double>> get_aligned_returns_matrix() const;

    /**
     * @brief Returns a matrix of log returns (rows = timeline points - 1, cols = tickers).
     */
    [[nodiscard]] std::vector<std::vector<double>> get_aligned_log_returns_matrix() const;

private:
    std::unordered_map<std::string, TimeSeries> assets_;
    std::vector<std::string> tickers_;
    std::vector<std::string> timeline_;
    std::vector<int64_t> timestamps_;
    std::vector<MarketSnapshot> snapshots_;
};

}   // namespace quant::data
