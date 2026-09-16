#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>

namespace quant::simulation {

class BootstrapSimulator {
public:
    /**
     * @brief Simulates a portfolio path by bootstrapping (resampling with replacement) historical portfolio
     * returns.
     * @param historical_returns Vector of historical daily percentage returns.
     * @param initial_equity Starting equity S_0.
     * @param days Number of days to simulate.
     * @param rng Random number generator.
     * @return Vector of simulated portfolio equity values of length (days + 1).
     */
    template <typename Rng>
    [[nodiscard]] static std::vector<double> simulate_portfolio_path(
        const std::vector<double>& historical_returns, double initial_equity, size_t days, Rng& rng) {
        if (historical_returns.empty()) {
            throw std::invalid_argument("Historical returns cannot be empty for bootstrapping");
        }

        std::uniform_int_distribution<size_t> dist(0, historical_returns.size() - 1);
        std::vector<double> path(days + 1, initial_equity);

        for (size_t t = 1; t <= days; ++t) {
            size_t idx = dist(rng);
            double ret = historical_returns[idx];
            path[t] = path[t - 1] * (1.0 + ret);
            if (path[t] < 0.0) path[t] = 0.0;   // Prevent negative equity
        }

        return path;
    }

    /**
     * @brief Multi-asset synchronized bootstrap: resamples joint cross-sectional return vectors.
     * @param historical_return_matrix Matrix where rows = days, cols = assets.
     * @param initial_prices Vector of starting asset prices.
     * @param days Number of days to simulate.
     * @param rng Random number generator.
     * @return Simulated price matrix: (days + 1) x num_assets.
     */
    template <typename Rng>
    [[nodiscard]] static Eigen::MatrixXd simulate_multi_asset_path(
        const Eigen::MatrixXd& historical_return_matrix,
        const Eigen::VectorXd& initial_prices,
        size_t days,
        Rng& rng) {
        const size_t n_history = historical_return_matrix.rows();
        const size_t n_assets = historical_return_matrix.cols();
        if (n_history == 0 || n_assets == 0) {
            throw std::invalid_argument("Return matrix cannot be empty");
        }

        std::uniform_int_distribution<size_t> dist(0, n_history - 1);
        Eigen::MatrixXd prices(days + 1, n_assets);
        prices.row(0) = initial_prices.transpose();

        for (size_t t = 1; t <= days; ++t) {
            size_t sampled_row = dist(rng);
            for (size_t i = 0; i < n_assets; ++i) {
                double ret = historical_return_matrix(sampled_row, i);
                double prev_p = prices(t - 1, i);
                prices(t, i) = prev_p * (1.0 + ret);
                if (prices(t, i) < 0.0) prices(t, i) = 0.0;
            }
        }

        return prices;
    }
};

}   // namespace quant::simulation
