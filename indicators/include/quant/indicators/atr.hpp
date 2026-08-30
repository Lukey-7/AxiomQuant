#pragma once

#include "quant/indicators/indicator.hpp"
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace quant::indicators {

class ATR {
public:
    /**
     * @brief Computes Average True Range using Wilder's smoothing method.
     * @param high High prices.
     * @param low Low prices.
     * @param close Close prices.
     * @param period Smoothing lookback period (default 14).
     * @return Vector of ATR values with initial period - 1 as NaN.
     */
    [[nodiscard]] static std::vector<double> calculate(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        size_t period = 14
    ) {
        if (period == 0) {
            throw std::invalid_argument("ATR period must be >= 1");
        }
        const size_t n = close.size();
        if (high.size() != n || low.size() != n) {
            throw std::invalid_argument("High, Low, and Close vectors must have the same length");
        }

        std::vector<double> atr(n, NaN);
        if (n <= period) {
            return atr;
        }

        // True Range vector
        std::vector<double> tr(n, 0.0);
        tr[0] = high[0] - low[0];
        for (size_t i = 1; i < n; ++i) {
            double hl = high[i] - low[i];
            double hc = std::abs(high[i] - close[i - 1]);
            double lc = std::abs(low[i] - close[i - 1]);
            tr[i] = std::max({hl, hc, lc});
        }

        // Initial ATR = simple average of first `period` TRs
        double sum_tr = 0.0;
        for (size_t i = 0; i < period; ++i) {
            sum_tr += tr[i];
        }
        double current_atr = sum_tr / static_cast<double>(period);
        atr[period - 1] = current_atr;

        const double d_period = static_cast<double>(period);
        for (size_t i = period; i < n; ++i) {
            current_atr = (current_atr * (d_period - 1.0) + tr[i]) / d_period;
            atr[i] = current_atr;
        }

        return atr;
    }
};

} // namespace quant::indicators
