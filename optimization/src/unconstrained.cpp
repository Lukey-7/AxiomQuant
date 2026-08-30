#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace quant::optimization {

OptimizationResult UnconstrainedMarkowitz::global_minimum_variance(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    double risk_free_rate
) {
    const size_t n = expected_returns.size();
    if (cov_matrix.rows() != static_cast<int>(n) || cov_matrix.cols() != static_cast<int>(n)) {
        throw std::invalid_argument("Dimension mismatch between expected_returns and cov_matrix");
    }

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);
    // Solve Sigma * x = 1 via LLT (Cholesky) or ColPivHouseholderQR
    Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix);
    Eigen::VectorXd inv_cov_ones;

    if (llt.info() == Eigen::Success) {
        inv_cov_ones = llt.solve(ones);
    } else {
        inv_cov_ones = cov_matrix.colPivHouseholderQr().solve(ones);
    }

    double denom = ones.dot(inv_cov_ones);
    if (std::abs(denom) < 1e-12) {
        throw std::runtime_error("Singular covariance matrix in GMV computation");
    }

    Eigen::VectorXd w = inv_cov_ones / denom;

    OptimizationResult res;
    res.weights = w;
    res.expected_return = PortfolioStats::portfolio_return(w, expected_returns);
    res.volatility = PortfolioStats::portfolio_volatility(w, cov_matrix);
    res.sharpe_ratio = PortfolioStats::portfolio_sharpe(w, expected_returns, cov_matrix, risk_free_rate);
    res.converged = true;
    res.method = "Analytical GMV";

    return res;
}

OptimizationResult UnconstrainedMarkowitz::target_return_portfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    double target_return,
    double risk_free_rate
) {
    const size_t n = expected_returns.size();
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);

    Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix);
    Eigen::VectorXd inv_cov_ones, inv_cov_mu;

    if (llt.info() == Eigen::Success) {
        inv_cov_ones = llt.solve(ones);
        inv_cov_mu = llt.solve(expected_returns);
    } else {
        inv_cov_ones = cov_matrix.colPivHouseholderQr().solve(ones);
        inv_cov_mu = cov_matrix.colPivHouseholderQr().solve(expected_returns);
    }

    double A = ones.dot(inv_cov_ones);
    double B = ones.dot(inv_cov_mu);
    double C = expected_returns.dot(inv_cov_mu);
    double delta = A * C - B * B;

    if (std::abs(delta) < 1e-12) {
        // Fallback to GMV if target cannot be formed
        return global_minimum_variance(expected_returns, cov_matrix, risk_free_rate);
    }

    double lambda1 = (C - target_return * B) / delta;
    double lambda2 = (target_return * A - B) / delta;

    Eigen::VectorXd w = lambda1 * inv_cov_ones + lambda2 * inv_cov_mu;

    OptimizationResult res;
    res.weights = w;
    res.expected_return = PortfolioStats::portfolio_return(w, expected_returns);
    res.volatility = PortfolioStats::portfolio_volatility(w, cov_matrix);
    res.sharpe_ratio = PortfolioStats::portfolio_sharpe(w, expected_returns, cov_matrix, risk_free_rate);
    res.converged = true;
    res.method = "Analytical Target Return";

    return res;
}

OptimizationResult UnconstrainedMarkowitz::maximum_sharpe_portfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    double risk_free_rate
) {
    const size_t n = expected_returns.size();
    Eigen::VectorXd excess_returns = expected_returns - (Eigen::VectorXd::Ones(n) * risk_free_rate);

    Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix);
    Eigen::VectorXd inv_cov_excess;

    if (llt.info() == Eigen::Success) {
        inv_cov_excess = llt.solve(excess_returns);
    } else {
        inv_cov_excess = cov_matrix.colPivHouseholderQr().solve(excess_returns);
    }

    double sum_weights = inv_cov_excess.sum();
    if (std::abs(sum_weights) < 1e-12) {
        return global_minimum_variance(expected_returns, cov_matrix, risk_free_rate);
    }

    Eigen::VectorXd w = inv_cov_excess / sum_weights;

    OptimizationResult res;
    res.weights = w;
    res.expected_return = PortfolioStats::portfolio_return(w, expected_returns);
    res.volatility = PortfolioStats::portfolio_volatility(w, cov_matrix);
    res.sharpe_ratio = PortfolioStats::portfolio_sharpe(w, expected_returns, cov_matrix, risk_free_rate);
    res.converged = true;
    res.method = "Analytical Tangency (Max Sharpe)";

    return res;
}

OptimizationResult UnconstrainedMarkowitz::risk_parity_portfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    double risk_free_rate,
    size_t max_iter,
    double tol
) {
    const size_t n = expected_returns.size();
    // Cyclical coordinate descent for Equal Risk Contribution (Spinu, 2013)
    // Objective: min sum_i (w_i * (Sigma * w)_i - sigma_p^2 / N)^2
    // w_i = (sqrt((Sigma * w)_(-i)^2 + 4 * Sigma_{ii} * b_i) - (Sigma * w)_(-i)) / (2 * Sigma_{ii})

    Eigen::VectorXd w = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
    double target_b = 1.0 / static_cast<double>(n); // Target risk budget per asset

    bool converged = false;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd w_prev = w;

        for (size_t i = 0; i < n; ++i) {
            double sigma_ii = cov_matrix(i, i);
            double sigma_w_minus_i = (cov_matrix.row(i).dot(w)) - sigma_ii * w(i);

            // Quadratic formula for positive root of: sigma_ii * w_i^2 + sigma_w_minus_i * w_i - target_b = 0
            double discriminant = sigma_w_minus_i * sigma_w_minus_i + 4.0 * sigma_ii * target_b;
            if (discriminant >= 0.0 && sigma_ii > 0.0) {
                w(i) = (-sigma_w_minus_i + std::sqrt(discriminant)) / (2.0 * sigma_ii);
            }
        }

        // Normalize weights: sum w_i = 1
        double sum_w = w.sum();
        if (sum_w > 1e-12) {
            w /= sum_w;
        }

        double diff = (w - w_prev).norm();
        if (diff < tol) {
            converged = true;
            break;
        }
    }

    OptimizationResult res;
    res.weights = w;
    res.expected_return = PortfolioStats::portfolio_return(w, expected_returns);
    res.volatility = PortfolioStats::portfolio_volatility(w, cov_matrix);
    res.sharpe_ratio = PortfolioStats::portfolio_sharpe(w, expected_returns, cov_matrix, risk_free_rate);
    res.converged = converged;
    res.method = "Equal Risk Contribution (Risk Parity)";

    return res;
}

} // namespace quant::optimization
