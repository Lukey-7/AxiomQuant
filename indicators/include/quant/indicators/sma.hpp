#pragma once

#include "quant/indicators/indicator.hpp"
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace quant::indicators {

class SMA {
public:
    /**
     * @brief Computes Simple Moving Average using an O(N) rolling window accumulator.
     * @param data Input price series.
     * @param period Lookback window period (must be >= 1).
     * @return Vector of same length as data, with first (period - 1) elements filled with NaN.
     */
    [[nodiscard]] static std::vector<double> calculate(const std::vector<double>& data, size_t period) {
        if (period == 0) {
            throw std::invalid_argument("SMA period must be >= 1");
        }
        
        const size_t n = data.size();
        std::vector<double> result(n, NaN);
        if (n < period) {
            return result;
        }

        double window_sum = 0.0;
        for (size_t i = 0; i < period; ++i) {
            window_sum += data[i];
        }
        result[period - 1] = window_sum / static_cast<double>(period);

        for (size_t i = period; i < n; ++i) {
            window_sum += data[i] - data[i - period];
            result[i] = window_sum / static_cast<double>(period);
        }

        return result;
    }
};

} // namespace quant::indicators
