#pragma once

#include <Eigen/Dense>

namespace quant::optimization {

struct OptimizationResult {
    Eigen::VectorXd weights;
    double expected_return{0.0};
    double volatility{0.0};
    double sharpe_ratio{0.0};
    bool converged{true};
    std::string method;
};

class UnconstrainedMarkowitz {
public:
    /**
     * @brief Computes the Global Minimum Variance (GMV) portfolio analytically.
     * @param cov_matrix Annualized covariance matrix (N x N).
     */
    [[nodiscard]] static OptimizationResult global_minimum_variance(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        double risk_free_rate = 0.02
    );

    /**
     * @brief Computes Minimum Variance portfolio for a specific target return analytically.
     */
    [[nodiscard]] static OptimizationResult target_return_portfolio(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        double target_return,
        double risk_free_rate = 0.02
    );

    /**
     * @brief Computes the Tangency (Maximum Sharpe Ratio) portfolio analytically.
     */
    [[nodiscard]] static OptimizationResult maximum_sharpe_portfolio(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        double risk_free_rate = 0.02
    );

    /**
     * @brief Computes Equal Risk Contribution (Risk Parity) portfolio iteratively.
     */
    [[nodiscard]] static OptimizationResult risk_parity_portfolio(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        double risk_free_rate = 0.02,
        size_t max_iter = 1000,
        double tol = 1e-7
    );
};

} // namespace quant::optimization
