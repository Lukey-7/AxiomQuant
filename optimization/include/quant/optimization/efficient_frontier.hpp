#pragma once

#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/constrained_qp.hpp"
#include <vector>
#include <string>

namespace quant::optimization {

struct FrontierPoint {
    double expected_return{0.0};
    double volatility{0.0};
    double sharpe_ratio{0.0};
    Eigen::VectorXd weights;
};

class EfficientFrontier {
public:
    /**
     * @brief Computes unconstrained efficient frontier curve points.
     */
    [[nodiscard]] static std::vector<FrontierPoint> compute_unconstrained_frontier(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        size_t num_points = 50,
        double risk_free_rate = 0.02
    );

    /**
     * @brief Computes constrained (long-only / box constrained) efficient frontier curve points.
     */
    [[nodiscard]] static std::vector<FrontierPoint> compute_constrained_frontier(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        const ConstrainedQpConfig& config = ConstrainedQpConfig{},
        size_t num_points = 50
    );

    /**
     * @brief Renders an ASCII scatter/curve plot of the Efficient Frontier.
     */
    [[nodiscard]] static std::string render_ascii_frontier(
        const std::vector<FrontierPoint>& frontier,
        const FrontierPoint& gmv_point,
        const FrontierPoint& tangency_point,
        size_t width = 60,
        size_t height = 15
    );

    /**
     * @brief Generates comprehensive text report comparing portfolios and weights.
     */
    [[nodiscard]] static std::string generate_portfolio_report(
        const std::vector<std::string>& tickers,
        const OptimizationResult& gmv_unconstrained,
        const OptimizationResult& tangency_unconstrained,
        const OptimizationResult& gmv_constrained,
        const OptimizationResult& tangency_constrained,
        const OptimizationResult& risk_parity
    );
};

} // namespace quant::optimization
