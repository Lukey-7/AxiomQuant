#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace quant::optimization {

OptimizationResult UnconstrainedMarkowitz::global_minimum_variance(const Eigen::VectorXd& expected_returns,
                                                                   const Eigen::MatrixXd& cov_matrix,
                                                                   double risk_free_rate) {
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

OptimizationResult UnconstrainedMarkowitz::target_return_portfolio(const Eigen::VectorXd& expected_returns,
                                                                   const Eigen::MatrixXd& cov_matrix,
                                                                   double target_return,
                                                                   double risk_free_rate) {
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

OptimizationResult UnconstrainedMarkowitz::maximum_sharpe_portfolio(const Eigen::VectorXd& expected_returns,
                                                                    const Eigen::MatrixXd& cov_matrix,
                                                                    double risk_free_rate) {
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

OptimizationResult UnconstrainedMarkowitz::risk_parity_portfolio(const Eigen::VectorXd& expected_returns,
                                                                 const Eigen::MatrixXd& cov_matrix,
                                                                 double risk_free_rate,
                                                                 size_t max_iter,
                                                                 double tol) {
    const size_t n = expected_returns.size();
    // Cyclical coordinate descent for Equal Risk Contribution (Spinu, 2013) on the strictly convex
    //   min_y  0.5 * y^T Sigma y - b * sum_i log(y_i),   b = 1/N,
    // whose optimality conditions y_i (Sigma y)_i = b are exactly equal risk contributions. Each
    // coordinate is minimised in closed form (positive root of Sigma_ii y_i^2 + c_i y_i - b = 0, with
    // c_i = (Sigma y)_i - Sigma_ii y_i). The weights are y / sum(y). Normalising inside the loop would
    // change the scale the other coordinates were solved at and converge to the wrong point.
    Eigen::VectorXd y = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
    const double budget = 1.0 / static_cast<double>(n);

    bool converged = false;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double sigma_ii = cov_matrix(i, i);
            if (!(sigma_ii > 0.0)) continue;
            const double c = cov_matrix.row(i).dot(y) - sigma_ii * y(i);
            const double updated = (-c + std::sqrt(c * c + 4.0 * sigma_ii * budget)) / (2.0 * sigma_ii);
            max_change = std::max(max_change, std::abs(updated - y(i)));
            y(i) = updated;
        }
        if (max_change <= tol * y.cwiseAbs().maxCoeff()) {
            converged = true;
            break;
        }
    }
    const Eigen::VectorXd w = y / y.sum();

    OptimizationResult res;
    res.weights = w;
    res.expected_return = PortfolioStats::portfolio_return(w, expected_returns);
    res.volatility = PortfolioStats::portfolio_volatility(w, cov_matrix);
    res.sharpe_ratio = PortfolioStats::portfolio_sharpe(w, expected_returns, cov_matrix, risk_free_rate);
    res.converged = converged;
    res.method = "Equal Risk Contribution (Risk Parity)";

    return res;
}

}   // namespace quant::optimization
