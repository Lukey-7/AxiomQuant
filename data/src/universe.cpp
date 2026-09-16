#include "quant/data/universe.hpp"
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace quant::data {

void MarketDataUniverse::add_asset(const std::string& ticker, TimeSeries series) {
    if (series.empty()) return;
    if (std::find(tickers_.begin(), tickers_.end(), ticker) == tickers_.end()) {
        tickers_.push_back(ticker);
    }
    assets_[ticker] = std::move(series);
}

void MarketDataUniverse::synchronize_timeline(bool require_all_assets) {
    if (assets_.empty()) {
        timeline_.clear();
        timestamps_.clear();
        snapshots_.clear();
        return;
    }

    // Sort tickers for deterministic ordering
    std::sort(tickers_.begin(), tickers_.end());

    if (require_all_assets) {
        // Find intersection of dates across all assets
        std::vector<std::string> common_dates = assets_[tickers_.front()].dates;
        for (size_t i = 1; i < tickers_.size(); ++i) {
            const auto& current_dates = assets_[tickers_[i]].dates;
            std::set<std::string> current_set(current_dates.begin(), current_dates.end());
            std::vector<std::string> filtered;
            for (const auto& d : common_dates) {
                if (current_set.find(d) != current_set.end()) {
                    filtered.push_back(d);
                }
            }
            common_dates = std::move(filtered);
        }

        timeline_ = std::move(common_dates);
    } else {
        // Union of all dates
        std::set<std::string> all_dates;
        for (const auto& [ticker, ts] : assets_) {
            for (const auto& d : ts.dates) {
                all_dates.insert(d);
            }
        }
        timeline_.assign(all_dates.begin(), all_dates.end());
    }

    // Build fast lookup index for each asset: date -> Bar
    std::unordered_map<std::string, std::unordered_map<std::string, Bar>> asset_date_map;
    for (const auto& ticker : tickers_) {
        const auto& ts = assets_[ticker];
        for (size_t i = 0; i < ts.size(); ++i) {
            asset_date_map[ticker][ts.dates[i]] = ts.get_bar(i);
        }
    }

    // Build synchronized snapshots and timestamps
    snapshots_.clear();
    timestamps_.clear();
    snapshots_.reserve(timeline_.size());
    timestamps_.reserve(timeline_.size());

    std::unordered_map<std::string, Bar> last_known_bar;

    for (const auto& date : timeline_) {
        MarketSnapshot snap;
        snap.date = date;
        int64_t ts_val = 0;

        for (const auto& ticker : tickers_) {
            auto& date_map = asset_date_map[ticker];
            auto it = date_map.find(date);
            if (it != date_map.end()) {
                snap.bars[ticker] = it->second;
                last_known_bar[ticker] = it->second;
                if (ts_val == 0) ts_val = it->second.timestamp;
            } else if (last_known_bar.find(ticker) != last_known_bar.end()) {
                // Forward fill if union was used
                Bar filled = last_known_bar[ticker];
                filled.date = date;
                filled.volume = 0.0;
                snap.bars[ticker] = filled;
            }
        }

        snap.timestamp = ts_val;
        timestamps_.push_back(ts_val);
        snapshots_.push_back(std::move(snap));
    }
}

const TimeSeries& MarketDataUniverse::get_series(const std::string& ticker) const {
    auto it = assets_.find(ticker);
    if (it == assets_.end()) {
        throw std::invalid_argument("Ticker not found in universe: " + ticker);
    }
    return it->second;
}

const MarketSnapshot& MarketDataUniverse::get_snapshot(size_t timeline_index) const {
    if (timeline_index >= snapshots_.size()) {
        throw std::out_of_range("Timeline index out of range in MarketDataUniverse");
    }
    return snapshots_[timeline_index];
}

MarketDataUniverse MarketDataUniverse::slice(size_t begin, size_t end) const {
    end = std::min(end, snapshots_.size());
    if (begin > end) {
        throw std::out_of_range("MarketDataUniverse::slice: begin > end");
    }
    MarketDataUniverse out;
    out.assets_ = assets_;
    out.tickers_ = tickers_;
    out.timeline_.assign(timeline_.begin() + static_cast<std::ptrdiff_t>(begin),
                         timeline_.begin() + static_cast<std::ptrdiff_t>(end));
    out.timestamps_.assign(timestamps_.begin() + static_cast<std::ptrdiff_t>(begin),
                           timestamps_.begin() + static_cast<std::ptrdiff_t>(end));
    out.snapshots_.assign(snapshots_.begin() + static_cast<std::ptrdiff_t>(begin),
                          snapshots_.begin() + static_cast<std::ptrdiff_t>(end));
    return out;
}

std::vector<double> MarketDataUniverse::get_aligned_closes(const std::string& ticker) const {
    if (assets_.find(ticker) == assets_.end()) {
        throw std::invalid_argument("Ticker not found in universe: " + ticker);
    }
    std::vector<double> closes(snapshots_.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t t = 0; t < snapshots_.size(); ++t) {
        auto it = snapshots_[t].bars.find(ticker);
        if (it != snapshots_[t].bars.end()) {
            closes[t] = it->second.close;
        }
    }
    return closes;
}

std::vector<std::vector<double>> MarketDataUniverse::get_aligned_close_matrix() const {
    if (snapshots_.empty() || tickers_.empty()) return {};

    std::vector<std::vector<double>> matrix(snapshots_.size(), std::vector<double>(tickers_.size(), 0.0));
    for (size_t t = 0; t < snapshots_.size(); ++t) {
        const auto& snap = snapshots_[t];
        for (size_t i = 0; i < tickers_.size(); ++i) {
            auto it = snap.bars.find(tickers_[i]);
            if (it != snap.bars.end()) {
                matrix[t][i] = it->second.close;
            }
        }
    }
    return matrix;
}

std::vector<std::vector<double>> MarketDataUniverse::get_aligned_returns_matrix() const {
    auto close_mat = get_aligned_close_matrix();
    if (close_mat.size() < 2 || tickers_.empty()) return {};

    std::vector<std::vector<double>> ret_mat(close_mat.size() - 1, std::vector<double>(tickers_.size(), 0.0));
    for (size_t t = 1; t < close_mat.size(); ++t) {
        for (size_t i = 0; i < tickers_.size(); ++i) {
            double prev = close_mat[t - 1][i];
            double curr = close_mat[t][i];
            ret_mat[t - 1][i] = (prev > 0.0) ? ((curr - prev) / prev) : 0.0;
        }
    }
    return ret_mat;
}

std::vector<std::vector<double>> MarketDataUniverse::get_aligned_log_returns_matrix() const {
    auto close_mat = get_aligned_close_matrix();
    if (close_mat.size() < 2 || tickers_.empty()) return {};

    std::vector<std::vector<double>> log_ret_mat(close_mat.size() - 1,
                                                 std::vector<double>(tickers_.size(), 0.0));
    for (size_t t = 1; t < close_mat.size(); ++t) {
        for (size_t i = 0; i < tickers_.size(); ++i) {
            double prev = close_mat[t - 1][i];
            double curr = close_mat[t][i];
            log_ret_mat[t - 1][i] = (prev > 0.0 && curr > 0.0) ? std::log(curr / prev) : 0.0;
        }
    }
    return log_ret_mat;
}

}   // namespace quant::data
