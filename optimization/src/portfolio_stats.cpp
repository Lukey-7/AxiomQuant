#include "quant/optimization/portfolio_stats.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace quant::optimization {

Eigen::VectorXd PortfolioStats::compute_expected_returns(
    const Eigen::MatrixXd& return_matrix,
    double ann_factor
) {
    if (return_matrix.rows() == 0 || return_matrix.cols() == 0) {
        throw std::invalid_argument("Empty return matrix");
    }
    // Column-wise mean
    Eigen::VectorXd mean_daily = return_matrix.colwise().mean();
    return mean_daily * ann_factor;
}

Eigen::MatrixXd PortfolioStats::compute_sample_covariance(
    const Eigen::MatrixXd& return_matrix,
    double ann_factor
) {
    const size_t T = return_matrix.rows();
    const size_t N = return_matrix.cols();
    if (T < 2 || N == 0) {
        throw std::invalid_argument("Insufficient observations for covariance estimation");
    }

    Eigen::VectorXd mean_daily = return_matrix.colwise().mean();
    Eigen::MatrixXd centered = return_matrix.rowwise() - mean_daily.transpose();

    // S = (X_c^T * X_c) / (T - 1) * ann_factor
    Eigen::MatrixXd cov_daily = (centered.transpose() * centered) / static_cast<double>(T - 1);
    return cov_daily * ann_factor;
}

Eigen::MatrixXd PortfolioStats::compute_ledoit_wolf_covariance(
    const Eigen::MatrixXd& return_matrix,
    double ann_factor
) {
    const size_t T = return_matrix.rows();
    const size_t N = return_matrix.cols();
    if (T < 2 || N == 0) {
        throw std::invalid_argument("Insufficient observations for Ledoit-Wolf estimation");
    }

    Eigen::MatrixXd S = compute_sample_covariance(return_matrix, ann_factor);

    // Target F: Diagonal matrix with sample variances on diagonal
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(N, N);
    for (size_t i = 0; i < N; ++i) {
        F(i, i) = S(i, i);
    }

    // Centered observations
    Eigen::VectorXd mean_daily = return_matrix.colwise().mean();
    Eigen::MatrixXd centered = return_matrix.rowwise() - mean_daily.transpose();

    // Estimate variance of elements: sum Var(s_ij)
    double pi_hat = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum_dev = 0.0;
            double s_ij_daily = S(i, j) / ann_factor;
            for (size_t t = 0; t < T; ++t) {
                double term = (centered(t, i) * centered(t, j)) - s_ij_daily;
                sum_dev += term * term;
            }
            pi_hat += (sum_dev / static_cast<double>(T)) * (ann_factor * ann_factor);
        }
    }

    // Distance gamma_hat = ||S - F||_F^2
    double gamma_hat = (S - F).squaredNorm();

    // Optimal shrinkage intensity delta = pi_hat / (T * gamma_hat)
    double delta = 0.0;
    if (gamma_hat > 1e-12) {
        delta = (pi_hat / static_cast<double>(T)) / gamma_hat;
    }
    delta = std::clamp(delta, 0.0, 1.0);

    return (1.0 - delta) * S + delta * F;
}

double PortfolioStats::portfolio_return(
    const Eigen::VectorXd& weights,
    const Eigen::VectorXd& expected_returns
) {
    return weights.dot(expected_returns);
}

double PortfolioStats::portfolio_volatility(
    const Eigen::VectorXd& weights,
    const Eigen::MatrixXd& cov_matrix
) {
    double var = weights.dot(cov_matrix * weights);
    return std::sqrt(std::max(0.0, var));
}

double PortfolioStats::portfolio_sharpe(
    const Eigen::VectorXd& weights,
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    double risk_free_rate
) {
    double ret = portfolio_return(weights, expected_returns);
    double vol = portfolio_volatility(weights, cov_matrix);
    if (vol <= 1e-9) return 0.0;
    return (ret - risk_free_rate) / vol;
}

} // namespace quant::optimization
