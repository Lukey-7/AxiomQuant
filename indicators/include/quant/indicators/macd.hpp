#pragma once

#include "quant/indicators/indicator.hpp"
#include "quant/indicators/ema.hpp"
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace quant::indicators {

struct MacdResult {
    std::vector<double> macd_line;
    std::vector<double> signal_line;
    std::vector<double> histogram;
};

class MACD {
public:
    /**
     * @brief Computes MACD line, Signal line, and MACD histogram.
     * @param prices Input price series.
     * @param fast_period Fast EMA period (default 12).
     * @param slow_period Slow EMA period (default 26).
     * @param signal_period Signal line EMA period (default 9).
     * @return MacdResult containing macd_line, signal_line, and histogram vectors.
     */
    [[nodiscard]] static MacdResult calculate(
        const std::vector<double>& prices,
        size_t fast_period = 12,
        size_t slow_period = 26,
        size_t signal_period = 9
    ) {
        if (fast_period >= slow_period) {
            throw std::invalid_argument("MACD fast_period must be strictly less than slow_period");
        }
        if (fast_period == 0 || slow_period == 0 || signal_period == 0) {
            throw std::invalid_argument("MACD periods must be >= 1");
        }

        const size_t n = prices.size();
        MacdResult result{
            std::vector<double>(n, NaN),
            std::vector<double>(n, NaN),
            std::vector<double>(n, NaN)
        };

        if (n < slow_period) {
            return result;
        }

        auto fast_ema = EMA::calculate(prices, fast_period);
        auto slow_ema = EMA::calculate(prices, slow_period);

        // Compute MACD Line = Fast EMA - Slow EMA
        std::vector<double> valid_macd_vals;
        std::vector<size_t> valid_indices;

        for (size_t i = 0; i < n; ++i) {
            if (is_valid(fast_ema[i]) && is_valid(slow_ema[i])) {
                result.macd_line[i] = fast_ema[i] - slow_ema[i];
                valid_macd_vals.push_back(result.macd_line[i]);
                valid_indices.push_back(i);
            }
        }

        // Compute Signal Line = EMA of MACD Line
        if (valid_macd_vals.size() >= signal_period) {
            auto signal_sub = EMA::calculate(valid_macd_vals, signal_period);
            for (size_t j = 0; j < signal_sub.size(); ++j) {
                if (is_valid(signal_sub[j])) {
                    size_t orig_idx = valid_indices[j];
                    result.signal_line[orig_idx] = signal_sub[j];
                    result.histogram[orig_idx] = result.macd_line[orig_idx] - result.signal_line[orig_idx];
                }
            }
        }

        return result;
    }
};

} // namespace quant::indicators
