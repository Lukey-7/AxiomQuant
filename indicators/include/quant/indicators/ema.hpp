#pragma once

#include "quant/indicators/indicator.hpp"
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace quant::indicators {

class EMA {
public:
    /**
     * @brief Computes Exponential Moving Average: EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
     * @param data Input price series.
     * @param period Smoothing period (N).
     * @param smoothing Multiplier factor (default = 2.0, giving alpha = 2 / (N + 1)).
     * @return Vector of same length, with first (period - 1) elements as NaN.
     */
    [[nodiscard]] static std::vector<double> calculate(const std::vector<double>& data, size_t period, double smoothing = 2.0) {
        if (period == 0) {
            throw std::invalid_argument("EMA period must be >= 1");
        }

        const size_t n = data.size();
        std::vector<double> result(n, NaN);
        if (n < period) {
            return result;
        }

        const double alpha = smoothing / (static_cast<double>(period) + 1.0);

        // Seed with the Simple Moving Average of the first `period` elements
        double seed_sum = 0.0;
        for (size_t i = 0; i < period; ++i) {
            seed_sum += data[i];
        }
        double prev_ema = seed_sum / static_cast<double>(period);
        result[period - 1] = prev_ema;

        for (size_t i = period; i < n; ++i) {
            prev_ema = alpha * data[i] + (1.0 - alpha) * prev_ema;
            result[i] = prev_ema;
        }

        return result;
    }
};

} // namespace quant::indicators
