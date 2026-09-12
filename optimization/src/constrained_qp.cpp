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
    const Eigen::Index n = v.size();
    if (n == 0) return v;
    if (min_w > max_w) {
        throw std::invalid_argument("Infeasible constraints: min_weight > max_weight");
    }

    const double dn = static_cast<double>(n);
    if (dn * min_w > 1.0 + 1e-9 || dn * max_w < 1.0 - 1e-9) {
        throw std::invalid_argument("Infeasible constraints: N * min_weight > 1 or N * max_weight < 1");
    }

    // The KKT conditions give w_i = clamp(v_i - theta, lo, hi) for the unique theta solving
    //   g(theta) = sum_i clamp(v_i - theta, lo, hi) = 1.
    // g is continuous, non-increasing and piecewise linear with 2N breakpoints:
    //   theta = v_i - hi  (coordinate i leaves its upper bound and becomes free)
    //   theta = v_i - lo  (coordinate i reaches its lower bound)
    // Sweeping the sorted breakpoints while maintaining the active set finds the exact root in O(N log N).
    struct Breakpoint {
        double theta;
        Eigen::Index idx;
        bool becomes_free;
    };
    std::vector<Breakpoint> events;
    events.reserve(static_cast<size_t>(2 * n));
    for (Eigen::Index i = 0; i < n; ++i) {
        events.push_back({v(i) - max_w, i, true});
        events.push_back({v(i) - min_w, i, false});
    }
    std::sort(events.begin(), events.end(), [](const Breakpoint& a, const Breakpoint& b) {
        if (a.theta != b.theta) return a.theta < b.theta;
        return a.becomes_free && !b.becomes_free; // a coordinate must become free before it can hit the floor
    });

    // Left of every breakpoint all coordinates sit at the upper bound.
    double count_hi = dn, count_lo = 0.0, count_free = 0.0, free_sum = 0.0;
    auto g = [&](double theta) {
        return count_hi * max_w + count_lo * min_w + free_sum - count_free * theta;
    };

    double theta = events.back().theta;
    for (const auto& e : events) {
        if (g(e.theta) <= 1.0) {
            // Root lies on the current linear segment (ending at e.theta).
            theta = (count_free > 0.0)
                ? (count_hi * max_w + count_lo * min_w + free_sum - 1.0) / count_free
                : e.theta;
            break;
        }
        if (e.becomes_free) {
            count_hi -= 1.0;
            count_free += 1.0;
            free_sum += v(e.idx);
        } else {
            count_free -= 1.0;
            free_sum -= v(e.idx);
            count_lo += 1.0;
        }
    }

    Eigen::VectorXd w(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        w(i) = std::clamp(v(i) - theta, min_w, max_w);
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
    // The constrained tangency portfolio lies on the constrained frontier traced by the risk-aversion
    // parameter gamma. Stage 1: coarse log-spaced scan (plus gamma = 0, the GMV end of the frontier).
    // Stage 2: golden-section refinement of log10(gamma) around the best grid point.
    const size_t grid_steps = 120;
    const double log_min = -3.0; // 10^-3
    const double log_max = 2.5;  // 10^2.5 ~ 316
    const double grid_h = (log_max - log_min) / static_cast<double>(grid_steps);

    auto solve_at = [&](double log_gamma) {
        return optimize_risk_aversion(expected_returns, cov_matrix, std::pow(10.0, log_gamma));
    };

    OptimizationResult best_result = optimize_risk_aversion(expected_returns, cov_matrix, 0.0);
    double best_log_gamma = log_min;
    for (size_t i = 0; i <= grid_steps; ++i) {
        const double log_gamma = log_min + static_cast<double>(i) * grid_h;
        auto candidate = solve_at(log_gamma);
        if (candidate.sharpe_ratio > best_result.sharpe_ratio) {
            best_result = std::move(candidate);
            best_log_gamma = log_gamma;
        }
    }

    constexpr double inv_phi = 0.6180339887498949; // 1 / golden ratio
    double a = std::max(log_min, best_log_gamma - grid_h);
    double b = std::min(log_max, best_log_gamma + grid_h);
    double c = b - inv_phi * (b - a);
    double d = a + inv_phi * (b - a);
    auto fc = solve_at(c);
    auto fd = solve_at(d);
    for (int iter = 0; iter < 40 && (b - a) > 1e-9; ++iter) {
        if (fc.sharpe_ratio > fd.sharpe_ratio) {
            b = d; d = c; fd = std::move(fc);
            c = b - inv_phi * (b - a);
            fc = solve_at(c);
        } else {
            a = c; c = d; fc = std::move(fd);
            d = a + inv_phi * (b - a);
            fd = solve_at(d);
        }
    }
    for (auto* cand : {&fc, &fd}) {
        if (cand->sharpe_ratio > best_result.sharpe_ratio) best_result = *cand;
    }

    best_result.method = "Constrained Max Sharpe (Tangency)";
    return best_result;
}

} // namespace quant::optimization
