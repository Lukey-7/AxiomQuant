#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>

namespace quant::simulation {

class GbmSimulator {
public:
    /**
     * @brief Simulates a single-asset Geometric Brownian Motion path.
     * @param s0 Initial asset price.
     * @param mu Annualized drift (expected return).
     * @param sigma Annualized volatility.
     * @param days Number of simulation steps (trading days).
     * @param dt Time step size (default 1.0 / 252.0).
     * @param rng Random number generator.
     * @return Vector of prices of length (days + 1).
     */
    template <typename Rng>
    [[nodiscard]] static std::vector<double> simulate_single_path(
        double s0, double mu, double sigma, size_t days, double dt, Rng& rng) {
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> path(days + 1, s0);

        const double drift = (mu - 0.5 * sigma * sigma) * dt;
        const double vol = sigma * std::sqrt(dt);

        for (size_t t = 1; t <= days; ++t) {
            double z = dist(rng);
            path[t] = path[t - 1] * std::exp(drift + vol * z);
        }

        return path;
    }

    /**
     * @brief Simulates correlated multi-asset return paths using Cholesky decomposition of covariance matrix.
     * @param initial_prices Vector of S_0 for each asset.
     * @param mu Expected returns vector.
     * @param cholesky_L Lower triangular matrix L where L * L^T = Sigma * dt.
     * @param days Number of steps.
     * @param dt Time step size (1.0 / 252.0).
     * @param rng Random number generator.
     * @return Matrix of prices: dimensions (days + 1) x num_assets.
     */
    template <typename Rng>
    [[nodiscard]] static Eigen::MatrixXd simulate_correlated_paths(const Eigen::VectorXd& initial_prices,
                                                                   const Eigen::VectorXd& mu,
                                                                   const Eigen::MatrixXd& cholesky_L,
                                                                   size_t days,
                                                                   double dt,
                                                                   Rng& rng) {
        const size_t n_assets = initial_prices.size();
        Eigen::MatrixXd prices(days + 1, n_assets);
        prices.row(0) = initial_prices.transpose();

        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd z_uncorr(n_assets);

        // Precompute drift vector: (mu_i - 0.5 * var_i) * dt
        // Notice variance_i = Sigma(i, i), which is row_norm^2 of L / dt
        Eigen::VectorXd drift(n_assets);
        for (size_t i = 0; i < n_assets; ++i) {
            double var_i = cholesky_L.row(i).squaredNorm() / dt;
            drift(i) = (mu(i) - 0.5 * var_i) * dt;
        }

        for (size_t t = 1; t <= days; ++t) {
            for (size_t i = 0; i < n_assets; ++i) {
                z_uncorr(i) = dist(rng);
            }
            // Correlated random shock: L * z_uncorr
            Eigen::VectorXd correlated_shock = cholesky_L * z_uncorr;

            for (size_t i = 0; i < n_assets; ++i) {
                double prev_p = prices(t - 1, i);
                prices(t, i) = prev_p * std::exp(drift(i) + correlated_shock(i));
            }
        }

        return prices;
    }
};

}   // namespace quant::simulation
