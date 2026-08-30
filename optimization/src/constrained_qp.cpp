#include "quant/optimization/constrained_qp.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace quant::optimization {

Eigen::VectorXd ConstrainedQpOptimizer::project_onto_bounded_simplex(
    const Eigen::VectorXd& v,
    double min_w,
    double max_w
) {
    const size_t n = v.size();
    if (n == 0) return v;

    double min_sum = static_cast<double>(n) * min_w;
    double max_sum = static_cast<double>(n) * max_w;
    if (min_sum > 1.0 + 1e-9 || max_sum < 1.0 - 1e-9) {
        throw std::invalid_argument("Infeasible constraints: N * min_weight > 1 or N * max_weight < 1");
    }

    // Root finding for theta: g(theta) = sum(clamp(v_i - theta, min_w, max_w)) - 1 = 0
    double low = v.minCoeff() - max_w - 1.0;
    double high = v.maxCoeff() - min_w + 1.0;

    auto eval_sum = [&](double theta) -> double {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double val = v(i) - theta;
            sum += std::clamp(val, min_w, max_w);
        }
        return sum;
    };

    // 40 iterations of bisection achieves machine precision (< 1e-12)
    for (int iter = 0; iter < 45; ++iter) {
        double mid = 0.5 * (low + high);
        double s = eval_sum(mid);
        if (s > 1.0) {
            low = mid;
        } else {
            high = mid;
        }
    }

    double optimal_theta = 0.5 * (low + high);
    Eigen::VectorXd w(n);
    for (size_t i = 0; i < n; ++i) {
        w(i) = std::clamp(v(i) - optimal_theta, min_w, max_w);
    }

    // Minor normalization to guarantee exact sum = 1.0
    double s = w.sum();
    if (std::abs(s - 1.0) > 1e-12 && s > 0.0) {
        w /= s;
    }

    return w;
}

OptimizationResult ConstrainedQpOptimizer::optimize_risk_aversion(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    double gamma
) const {
    const size_t n = expected_returns.size();
    if (cov_matrix.rows() != static_cast<int>(n) || cov_matrix.cols() != static_cast<int>(n)) {
        throw std::invalid_argument("Dimension mismatch between expected_returns and cov_matrix");
    }

    // Compute spectral radius / max eigenvalue for step size L
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov_matrix);
    double L = (eigensolver.info() == Eigen::Success)
        ? eigensolver.eigenvalues().maxCoeff()
        : cov_matrix.norm(); // Frobenius norm as upper bound

    if (L <= 1e-12) L = 1.0;
    double step_size = 1.0 / L;

    // Gradient of objective: grad = Sigma * w - gamma * mu
    Eigen::VectorXd linear_term = gamma * expected_returns;

    // Initial feasible point: 1/N
    Eigen::VectorXd w = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
    w = project_onto_bounded_simplex(w, config_.min_weight, config_.max_weight);

    Eigen::VectorXd y = w;
    Eigen::VectorXd w_prev = w;
    double t = 1.0;
    bool converged = false;

    // Accelerated Projected Gradient Descent (FISTA / Nesterov)
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        Eigen::VectorXd grad = cov_matrix * y - linear_term;
        Eigen::VectorXd step_point = y - step_size * grad;

        w = project_onto_bounded_simplex(step_point, config_.min_weight, config_.max_weight);

        double diff = (w - w_prev).cwiseAbs().maxCoeff();
        if (diff < config_.tolerance) {
            converged = true;
            break;
        }

        double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
        double beta = (t - 1.0) / t_next;
        y = w + beta * (w - w_prev);

        w_prev = w;
        t = t_next;
    }

    OptimizationResult res;
    res.weights = w;
    res.expected_return = PortfolioStats::portfolio_return(w, expected_returns);
    res.volatility = PortfolioStats::portfolio_volatility(w, cov_matrix);
    res.sharpe_ratio = PortfolioStats::portfolio_sharpe(w, expected_returns, cov_matrix, config_.risk_free_rate);
    res.converged = converged;
    res.method = "Constrained QP (gamma=" + std::to_string(gamma) + ")";

    return res;
}

OptimizationResult ConstrainedQpOptimizer::global_minimum_variance(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix
) const {
    auto res = optimize_risk_aversion(expected_returns, cov_matrix, 0.0);
    res.method = "Constrained Long-Only GMV";
    return res;
}

OptimizationResult ConstrainedQpOptimizer::maximum_sharpe_portfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix
) const {
    // Find risk aversion gamma > 0 that maximizes Sharpe ratio along the constrained efficient frontier
    double best_sharpe = -1e9;
    OptimizationResult best_result;

    // Log-spaced search grid for gamma parameter
    const size_t grid_steps = 150;
    double log_min = -3.0; // 10^-3
    double log_max = 2.5;  // 10^2.5 ~ 316.0

    for (size_t i = 0; i <= grid_steps; ++i) {
        double log_gamma = log_min + (static_cast<double>(i) / grid_steps) * (log_max - log_min);
        double gamma = std::pow(10.0, log_gamma);

        auto candidate = optimize_risk_aversion(expected_returns, cov_matrix, gamma);
        if (candidate.sharpe_ratio > best_sharpe) {
            best_sharpe = candidate.sharpe_ratio;
            best_result = candidate;
        }
    }

    best_result.method = "Constrained Max Sharpe (Tangency)";
    return best_result;
}

} // namespace quant::optimization
