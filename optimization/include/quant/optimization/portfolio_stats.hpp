#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace quant::optimization {

class PortfolioStats {
public:
    /**
     * @brief Computes sample expected return vector (annualized).
     * @param return_matrix Matrix of daily returns (rows = days, cols = assets).
     * @param ann_factor Annualization factor (default 252.0).
     */
    [[nodiscard]] static Eigen::VectorXd compute_expected_returns(const Eigen::MatrixXd& return_matrix,
                                                                  double ann_factor = 252.0);

    /**
     * @brief Computes sample covariance matrix (annualized).
     * @param return_matrix Matrix of daily returns (rows = days, cols = assets).
     * @param ann_factor Annualization factor (default 252.0).
     */
    [[nodiscard]] static Eigen::MatrixXd compute_sample_covariance(const Eigen::MatrixXd& return_matrix,
                                                                   double ann_factor = 252.0);

    /**
     * @brief Ledoit-Wolf (2004) shrinkage towards the constant-correlation target.
     *
     *   Sigma_LW = delta * F + (1 - delta) * S,   F_ii = s_ii,  F_ij = r_bar * sqrt(s_ii * s_jj)
     *
     * with the asymptotically optimal intensity delta = clamp((pi_hat - rho_hat) / (T * gamma_hat), 0, 1)
     * from "Honey, I Shrunk the Sample Covariance Matrix" (J. Portfolio Management, 2004).
     *
     * @param return_matrix Matrix of daily returns (rows = days, cols = assets).
     * @param ann_factor Annualization factor (default 252.0).
     * @param shrinkage_intensity Optional output: the estimated delta in [0, 1].
     */
    [[nodiscard]] static Eigen::MatrixXd compute_ledoit_wolf_covariance(
        const Eigen::MatrixXd& return_matrix,
        double ann_factor = 252.0,
        double* shrinkage_intensity = nullptr);

    /**
     * @brief Computes portfolio annualized return: w^T * mu.
     */
    [[nodiscard]] static double portfolio_return(const Eigen::VectorXd& weights,
                                                 const Eigen::VectorXd& expected_returns);

    /**
     * @brief Computes portfolio annualized volatility: sqrt(w^T * Sigma * w).
     */
    [[nodiscard]] static double portfolio_volatility(const Eigen::VectorXd& weights,
                                                     const Eigen::MatrixXd& cov_matrix);

    /**
     * @brief Computes portfolio Sharpe ratio: (return - Rf) / vol.
     */
    [[nodiscard]] static double portfolio_sharpe(const Eigen::VectorXd& weights,
                                                 const Eigen::VectorXd& expected_returns,
                                                 const Eigen::MatrixXd& cov_matrix,
                                                 double risk_free_rate = 0.02);
};

}   // namespace quant::optimization
