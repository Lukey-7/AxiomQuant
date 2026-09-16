#pragma once

#include "quant/optimization/unconstrained.hpp"
#include <Eigen/Dense>

namespace quant::optimization {

struct ConstrainedQpConfig {
    double min_weight{0.0};   // 0.0 for long-only (no short selling)
    double max_weight{1.0};   // 1.0 for standard, e.g. 0.30 for 30% cap per asset
    double risk_free_rate{0.02};
    size_t max_iterations{20000};
    double tolerance{1e-12};   // max-norm change of the iterate between steps
};

class ConstrainedQpOptimizer {
public:
    explicit ConstrainedQpOptimizer(ConstrainedQpConfig config = ConstrainedQpConfig{}) : config_(config) {}

    /**
     * @brief Solves the constrained Global Minimum Variance (GMV) portfolio:
     *        min 0.5 * w^T * Sigma * w s.t. 1^T * w = 1, min_w <= w_i <= max_w.
     */
    [[nodiscard]] OptimizationResult global_minimum_variance(const Eigen::VectorXd& expected_returns,
                                                             const Eigen::MatrixXd& cov_matrix) const;

    /**
     * @brief Solves constrained mean-variance trade-off for a given risk aversion gamma:
     *        min 0.5 * w^T * Sigma * w - gamma * mu^T * w s.t. 1^T * w = 1, min_w <= w_i <= max_w.
     *
     * Accelerated projected gradient (FISTA) with step 1/lambda_max and gradient-based adaptive
     * restart (O'Donoghue & Candes, 2015), which restores linear convergence on this strongly convex
     * problem.
     */
    [[nodiscard]] OptimizationResult optimize_risk_aversion(const Eigen::VectorXd& expected_returns,
                                                            const Eigen::MatrixXd& cov_matrix,
                                                            double gamma) const;

    /**
     * @brief Solves the constrained Maximum Sharpe Ratio portfolio.
     *
     * Along the constrained frontier traced by the risk aversion gamma, the tangency portfolio is the
     * point where  gamma * (mu^T w - rf) = w^T Sigma w  (the frontier slope d(sigma^2)/d(return) equals
     * 2 gamma whatever constraints are active). That condition is bracketed on a log-spaced gamma grid
     * and solved by bisection, which is accurate to the precision of the inner QP rather than to the
     * flatness of the Sharpe ratio near its maximum. The best grid point by Sharpe is kept as a
     * fallback when no bracket exists (e.g. no portfolio earns more than the risk-free rate).
     */
    [[nodiscard]] OptimizationResult maximum_sharpe_portfolio(const Eigen::VectorXd& expected_returns,
                                                              const Eigen::MatrixXd& cov_matrix) const;

    /**
     * @brief Projects vector v onto the bounded simplex:
     *        { w in R^N : sum(w_i) = 1, min_w <= w_i <= max_w }.
     */
    [[nodiscard]] static Eigen::VectorXd project_onto_bounded_simplex(const Eigen::VectorXd& v,
                                                                      double min_w = 0.0,
                                                                      double max_w = 1.0);

    [[nodiscard]] const ConstrainedQpConfig& get_config() const noexcept { return config_; }
    void set_config(const ConstrainedQpConfig& config) noexcept { config_ = config; }

private:
    ConstrainedQpConfig config_;
};

}   // namespace quant::optimization
