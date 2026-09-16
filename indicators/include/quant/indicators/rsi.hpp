#pragma once

#include "quant/indicators/indicator.hpp"
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace quant::indicators {

class RSI {
public:
    /**
     * @brief Computes Relative Strength Index using Wilder's smoothed moving averages.
     * @param prices Input price series (e.g. close prices).
     * @param period RSI lookback period (standard is 14).
     * @return Vector of RSI values [0.0, 100.0] with warmup period filled with NaN.
     */
    [[nodiscard]] static std::vector<double> calculate(const std::vector<double>& prices,
                                                       size_t period = 14) {
        if (period == 0) {
            throw std::invalid_argument("RSI period must be >= 1");
        }

        const size_t n = prices.size();
        std::vector<double> rsi(n, NaN);
        if (n <= period) {
            return rsi;
        }

        // Calculate price changes and initial average gain/loss
        double sum_gain = 0.0;
        double sum_loss = 0.0;

        for (size_t i = 1; i <= period; ++i) {
            double change = prices[i] - prices[i - 1];
            if (change > 0.0) {
                sum_gain += change;
            } else {
                sum_loss += -change;
            }
        }

        double avg_gain = sum_gain / static_cast<double>(period);
        double avg_loss = sum_loss / static_cast<double>(period);

        if (avg_loss == 0.0) {
            rsi[period] = (avg_gain == 0.0) ? 50.0 : 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            rsi[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Wilder's smoothing for subsequent bars
        const double d_period = static_cast<double>(period);
        for (size_t i = period + 1; i < n; ++i) {
            double change = prices[i] - prices[i - 1];
            double gain = (change > 0.0) ? change : 0.0;
            double loss = (change < 0.0) ? -change : 0.0;

            avg_gain = (avg_gain * (d_period - 1.0) + gain) / d_period;
            avg_loss = (avg_loss * (d_period - 1.0) + loss) / d_period;

            if (avg_loss == 0.0) {
                rsi[i] = (avg_gain == 0.0) ? 50.0 : 100.0;
            } else {
                double rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        return rsi;
    }
};

}   // namespace quant::indicators
