#pragma once

#include "quant/indicators/indicator.hpp"
#include "quant/indicators/sma.hpp"
#include <vector>
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace quant::indicators {

struct BollingerBandsResult {
    std::vector<double> middle;
    std::vector<double> upper;
    std::vector<double> lower;
    std::vector<double> percent_b;
    std::vector<double> bandwidth;
};

class BollingerBands {
public:
    /**
     * @brief Computes Bollinger Bands (Middle, Upper, Lower, %B, Bandwidth).
     * @param prices Input price series.
     * @param period Rolling lookback window (default 20).
     * @param num_std Multiplier for sample standard deviation (default 2.0).
     * @return BollingerBandsResult struct containing the aligned vectors.
     */
    [[nodiscard]] static BollingerBandsResult calculate(const std::vector<double>& prices,
                                                        size_t period = 20,
                                                        double num_std = 2.0) {
        if (period == 0) {
            throw std::invalid_argument("Bollinger Bands period must be >= 1");
        }
        if (num_std < 0.0) {
            throw std::invalid_argument("num_std must be non-negative");
        }

        const size_t n = prices.size();
        BollingerBandsResult result{std::vector<double>(n, NaN), std::vector<double>(n, NaN),
                                    std::vector<double>(n, NaN), std::vector<double>(n, NaN),
                                    std::vector<double>(n, NaN)};

        if (n < period) {
            return result;
        }

        result.middle = SMA::calculate(prices, period);

        // Rolling standard deviation with Welford / two-pass rolling window
        for (size_t i = period - 1; i < n; ++i) {
            double mean = result.middle[i];
            double sum_sq_diff = 0.0;
            for (size_t j = i + 1 - period; j <= i; ++j) {
                double diff = prices[j] - mean;
                sum_sq_diff += diff * diff;
            }
            // Sample standard deviation (divided by N - 1, or population N if N==1)
            double variance = (period > 1) ? (sum_sq_diff / static_cast<double>(period - 1)) : 0.0;
            double std_dev = std::sqrt(variance);

            result.upper[i] = mean + num_std * std_dev;
            result.lower[i] = mean - num_std * std_dev;

            double width = result.upper[i] - result.lower[i];
            if (width > 0.0) {
                result.percent_b[i] = (prices[i] - result.lower[i]) / width;
            } else {
                result.percent_b[i] = 0.5;
            }

            if (mean != 0.0) {
                result.bandwidth[i] = width / mean;
            }
        }

        return result;
    }
};

}   // namespace quant::indicators
