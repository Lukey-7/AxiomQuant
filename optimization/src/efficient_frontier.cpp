#include "quant/optimization/efficient_frontier.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace quant::optimization {

std::vector<FrontierPoint> EfficientFrontier::compute_unconstrained_frontier(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    size_t num_points,
    double risk_free_rate) {
    std::vector<FrontierPoint> points;
    if (num_points == 0) return points;

    auto gmv = UnconstrainedMarkowitz::global_minimum_variance(expected_returns, cov_matrix, risk_free_rate);
    double min_ret = gmv.expected_return;
    double max_ret = expected_returns.maxCoeff() * 1.5;

    for (size_t i = 0; i < num_points; ++i) {
        double target_r = min_ret + (static_cast<double>(i) / (num_points - 1)) * (max_ret - min_ret);
        auto opt = UnconstrainedMarkowitz::target_return_portfolio(expected_returns, cov_matrix, target_r,
                                                                   risk_free_rate);
        points.push_back({opt.expected_return, opt.volatility, opt.sharpe_ratio, opt.weights});
    }

    return points;
}

std::vector<FrontierPoint> EfficientFrontier::compute_constrained_frontier(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    const ConstrainedQpConfig& config,
    size_t num_points) {
    std::vector<FrontierPoint> points;
    if (num_points == 0) return points;

    ConstrainedQpOptimizer optimizer(config);

    // Sweep log-spaced risk aversion gamma from 0.001 to 100
    for (size_t i = 0; i < num_points; ++i) {
        double gamma =
            (i == 0) ? 0.0 : std::pow(10.0, -3.0 + (static_cast<double>(i) / (num_points - 1)) * 5.0);
        auto opt = optimizer.optimize_risk_aversion(expected_returns, cov_matrix, gamma);
        points.push_back({opt.expected_return, opt.volatility, opt.sharpe_ratio, opt.weights});
    }

    // Sort by volatility
    std::sort(points.begin(), points.end(),
              [](const FrontierPoint& a, const FrontierPoint& b) { return a.volatility < b.volatility; });

    return points;
}

std::string EfficientFrontier::render_ascii_frontier(const std::vector<FrontierPoint>& frontier,
                                                     const FrontierPoint& gmv_point,
                                                     const FrontierPoint& tangency_point,
                                                     size_t width,
                                                     size_t height) {
    if (frontier.empty() || width == 0 || height == 0) return "";

    double min_vol = 1e9, max_vol = -1e9;
    double min_ret = 1e9, max_ret = -1e9;

    auto update_bounds = [&](double v, double r) {
        if (v < min_vol) min_vol = v;
        if (v > max_vol) max_vol = v;
        if (r < min_ret) min_ret = r;
        if (r > max_ret) max_ret = r;
    };

    for (const auto& pt : frontier) {
        update_bounds(pt.volatility, pt.expected_return);
    }
    update_bounds(gmv_point.volatility, gmv_point.expected_return);
    update_bounds(tangency_point.volatility, tangency_point.expected_return);

    double vol_margin = (max_vol - min_vol) * 0.05;
    double ret_margin = (max_ret - min_ret) * 0.05;
    if (vol_margin <= 1e-6) vol_margin = 0.01;
    if (ret_margin <= 1e-6) ret_margin = 0.01;

    min_vol -= vol_margin;
    max_vol += vol_margin;
    min_ret -= ret_margin;
    max_ret += ret_margin;

    std::vector<std::string> grid(height, std::string(width, ' '));

    auto plot_char = [&](double v, double r, char ch) {
        double norm_x = (v - min_vol) / (max_vol - min_vol);
        double norm_y = (r - min_ret) / (max_ret - min_ret);

        int col = static_cast<int>(std::floor(norm_x * (width - 1)));
        int row = static_cast<int>(std::floor(norm_y * (height - 1)));

        col = std::clamp(col, 0, static_cast<int>(width - 1));
        row = std::clamp(row, 0, static_cast<int>(height - 1));

        int grid_row = (height - 1) - row;
        grid[grid_row][col] = ch;
    };

    for (const auto& pt : frontier) {
        plot_char(pt.volatility, pt.expected_return, '.');
    }

    plot_char(gmv_point.volatility, gmv_point.expected_return, 'G');
    plot_char(tangency_point.volatility, tangency_point.expected_return, 'T');

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);

    ss << "\n--- ASCII EFFICIENT FRONTIER ---\n";
    ss << "Return (%)\n";
    for (size_t r = 0; r < height; ++r) {
        double level = max_ret - (static_cast<double>(r) / (height - 1)) * (max_ret - min_ret);
        ss << std::setw(6) << (level * 100.0) << "% |";
        ss << grid[r] << "\n";
    }
    ss << "       +" << std::string(width, '-') << "\n";
    ss << "        " << std::setw(6) << (min_vol * 100.0) << "%"
       << std::string(width > 20 ? width - 20 : 1, ' ') << std::setw(6) << (max_vol * 100.0)
       << "% Volatility\n";
    ss << "Legend: [.] Frontier   [G] Global Min Variance   [T] Max Sharpe Tangency\n\n";

    return ss.str();
}

std::string EfficientFrontier::generate_portfolio_report(const std::vector<std::string>& tickers,
                                                         const OptimizationResult& gmv_uncon,
                                                         const OptimizationResult& tan_uncon,
                                                         const OptimizationResult& gmv_con,
                                                         const OptimizationResult& tan_con,
                                                         const OptimizationResult& risk_parity) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "=================================================================================================="
          "======\n";
    ss << "                               MARKOWITZ PORTFOLIO OPTIMIZATION REPORT                            "
          "      \n";
    ss << "=================================================================================================="
          "======\n";
    ss << std::left << std::setw(12) << "Asset" << std::right << std::setw(18) << "GMV (Uncon)"
       << std::setw(18) << "Tangency (Uncon)" << std::setw(18) << "GMV (Long-Only)" << std::setw(18)
       << "Max Sharpe (Long)" << std::setw(18) << "Risk Parity" << "\n";
    ss << "--------------------------------------------------------------------------------------------------"
          "------\n";

    for (size_t i = 0; i < tickers.size(); ++i) {
        ss << std::left << std::setw(12) << tickers[i] << std::right << std::setw(17)
           << (gmv_uncon.weights(i) * 100.0) << "%" << std::setw(17) << (tan_uncon.weights(i) * 100.0) << "%"
           << std::setw(17) << (gmv_con.weights(i) * 100.0) << "%" << std::setw(17)
           << (tan_con.weights(i) * 100.0) << "%" << std::setw(17) << (risk_parity.weights(i) * 100.0) << "%"
           << "\n";
    }

    ss << "--------------------------------------------------------------------------------------------------"
          "------\n";
    ss << std::left << std::setw(12) << "Exp Return" << std::right << std::setw(17)
       << (gmv_uncon.expected_return * 100.0) << "%" << std::setw(17) << (tan_uncon.expected_return * 100.0)
       << "%" << std::setw(17) << (gmv_con.expected_return * 100.0) << "%" << std::setw(17)
       << (tan_con.expected_return * 100.0) << "%" << std::setw(17) << (risk_parity.expected_return * 100.0)
       << "%" << "\n";

    ss << std::left << std::setw(12) << "Volatility" << std::right << std::setw(17)
       << (gmv_uncon.volatility * 100.0) << "%" << std::setw(17) << (tan_uncon.volatility * 100.0) << "%"
       << std::setw(17) << (gmv_con.volatility * 100.0) << "%" << std::setw(17)
       << (tan_con.volatility * 100.0) << "%" << std::setw(17) << (risk_parity.volatility * 100.0) << "%"
       << "\n";

    ss << std::left << std::setw(12) << "Sharpe (2%)" << std::right << std::setprecision(3) << std::setw(18)
       << gmv_uncon.sharpe_ratio << std::setw(18) << tan_uncon.sharpe_ratio << std::setw(18)
       << gmv_con.sharpe_ratio << std::setw(18) << tan_con.sharpe_ratio << std::setw(18)
       << risk_parity.sharpe_ratio << "\n";
    ss << "=================================================================================================="
          "======\n";

    return ss.str();
}

}   // namespace quant::optimization
