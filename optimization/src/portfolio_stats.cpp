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
    double ann_factor,
    double* shrinkage_intensity
) {
    const Eigen::Index T = return_matrix.rows();
    const Eigen::Index N = return_matrix.cols();
    if (T < 2 || N == 0) {
        throw std::invalid_argument("Insufficient observations for Ledoit-Wolf estimation");
    }
    const double dT = static_cast<double>(T);

    // Centered returns and the (1/T-normalized) sample covariance used by the LW estimator.
    const Eigen::RowVectorXd mean_daily = return_matrix.colwise().mean();
    const Eigen::MatrixXd X = return_matrix.rowwise() - mean_daily;
    const Eigen::MatrixXd S = (X.transpose() * X) / dT;

    if (N == 1) {
        if (shrinkage_intensity) *shrinkage_intensity = 0.0;
        return S * ann_factor;
    }

    const Eigen::VectorXd var = S.diagonal();
    const Eigen::VectorXd sd = var.array().sqrt();

    // Average pairwise correlation r_bar.
    double corr_sum = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index j = 0; j < N; ++j) {
            if (i != j && sd(i) > 0.0 && sd(j) > 0.0) corr_sum += S(i, j) / (sd(i) * sd(j));
        }
    }
    const double r_bar = corr_sum / static_cast<double>(N * (N - 1));

    // Constant-correlation target F.
    Eigen::MatrixXd F = r_bar * (sd * sd.transpose());
    F.diagonal() = var;

    // pi_hat = sum_ij AsyVar[sqrt(T) s_ij]:  pi_ij = (1/T) sum_t (x_ti x_tj - s_ij)^2
    const Eigen::MatrixXd X2 = X.array().square().matrix();
    const Eigen::MatrixXd pi_mat = (X2.transpose() * X2) / dT - S.cwiseProduct(S);
    const double pi_hat = pi_mat.sum();

    // rho_hat = sum_i pi_ii + r_bar * sum_{i != j} (sd_j / sd_i) * theta_ii,ij
    //   theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij) = (1/T) sum_t x_ti^3 x_tj - s_ii s_ij
    const Eigen::MatrixXd X3 = X.array().cube().matrix();
    const Eigen::MatrixXd term1 = (X3.transpose() * X) / dT;
    double rho_off = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index j = 0; j < N; ++j) {
            if (i == j || !(sd(i) > 0.0)) continue;
            const double theta = term1(i, j) - var(i) * S(i, j);
            rho_off += (sd(j) / sd(i)) * theta;
        }
    }
    const double rho_hat = pi_mat.diagonal().sum() + r_bar * rho_off;

    // gamma_hat = ||F - S||_F^2  (misspecification of the target)
    const double gamma_hat = (F - S).squaredNorm();

    double delta = 0.0;
    if (gamma_hat > 1e-300) {
        const double kappa = (pi_hat - rho_hat) / gamma_hat;
        delta = std::clamp(kappa / dT, 0.0, 1.0);
    }
    if (shrinkage_intensity) *shrinkage_intensity = delta;

    return (delta * F + (1.0 - delta) * S) * ann_factor;
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
