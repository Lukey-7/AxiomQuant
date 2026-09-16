#pragma once

#include "quant/indicators/indicator.hpp"
#include <vector>
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace quant::indicators {

class RollingVolatility {
public:
    /**
     * @brief Computes annualized rolling volatility of log returns.
     * @param prices Input price series.
     * @param window Rolling window length (number of return observations, default 20).
     * @param annualization_factor Factor to annualize daily volatility (e.g. 252.0 for trading days).
     * @return Vector of annualized volatilities aligned with prices.
     */
    [[nodiscard]] static std::vector<double> calculate(const std::vector<double>& prices,
                                                       size_t window = 20,
                                                       double annualization_factor = 252.0) {
        if (window < 2) {
            throw std::invalid_argument("Rolling volatility window must be >= 2");
        }

        const size_t n = prices.size();
        std::vector<double> result(n, NaN);
        if (n <= window) {
            return result;
        }

        // Compute log returns
        std::vector<double> log_returns(n, 0.0);
        for (size_t i = 1; i < n; ++i) {
            if (prices[i] > 0.0 && prices[i - 1] > 0.0) {
                log_returns[i] = std::log(prices[i] / prices[i - 1]);
            }
        }

        const double sqrt_ann = std::sqrt(annualization_factor);
        const double d_win = static_cast<double>(window);

        for (size_t i = window; i < n; ++i) {
            // Mean of log returns in window [i - window + 1, i]
            double sum_ret = 0.0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                sum_ret += log_returns[j];
            }
            double mean_ret = sum_ret / d_win;

            // Sample variance
            double sum_sq = 0.0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                double diff = log_returns[j] - mean_ret;
                sum_sq += diff * diff;
            }
            double sample_var = sum_sq / (d_win - 1.0);
            double daily_vol = std::sqrt(sample_var);

            result[i] = daily_vol * sqrt_ann;
        }

        return result;
    }
};

}   // namespace quant::indicators
