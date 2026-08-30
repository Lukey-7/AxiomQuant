#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <Eigen/Dense>

namespace quant::simulation {

struct MonteCarloConfig {
    size_t num_simulations{50000};
    size_t horizon_days{252};
    double initial_wealth{100000.0};
    uint64_t seed{42};
    bool use_bootstrap{false};
};

struct MonteCarloReport {
    size_t num_simulations{0};
    size_t horizon_days{0};
    double initial_wealth{0.0};

    double mean_terminal_wealth{0.0};
    double median_terminal_wealth{0.0};
    double std_terminal_wealth{0.0};

    // Terminal wealth percentiles
    double p01_wealth{0.0};
    double p05_wealth{0.0};
    double p25_wealth{0.0};
    double p50_wealth{0.0};
    double p75_wealth{0.0};
    double p95_wealth{0.0};
    double p99_wealth{0.0};

    // Drawdown distribution
    double mean_max_drawdown{0.0};
    double p50_max_drawdown{0.0};
    double p95_max_drawdown{0.0};
    double p99_max_drawdown{0.0};

    // Risk probabilities
    double prob_loss{0.0};              // Prob(Ending Wealth < Initial Wealth)
    double prob_loss_gt_10pct{0.0};     // Prob(Ending Wealth < 0.9 * Initial)
    double prob_loss_gt_20pct{0.0};     // Prob(Ending Wealth < 0.8 * Initial)
    double var_95_terminal{0.0};
    double cvar_95_terminal{0.0};

    double elapsed_ms{0.0};
};

class MonteCarloEngine {
public:
    explicit MonteCarloEngine(MonteCarloConfig config = MonteCarloConfig{}) : config_(config) {}

    /**
     * @brief Run Monte Carlo simulation for a single return stream (e.g. backtest returns).
     * @param historical_returns Daily historical returns.
     */
    [[nodiscard]] MonteCarloReport run_simulation(const std::vector<double>& historical_returns) const;

    /**
     * @brief Run Monte Carlo simulation for a multi-asset portfolio with given weights.
     * @param expected_returns Asset annualized expected returns vector.
     * @param cov_matrix Asset daily covariance matrix.
     * @param weights Portfolio allocation weights vector.
     */
    [[nodiscard]] MonteCarloReport run_gbm_portfolio(
        const Eigen::VectorXd& expected_returns,
        const Eigen::MatrixXd& cov_matrix,
        const Eigen::VectorXd& weights
    ) const;

    /**
     * @brief Formats report as rich text table.
     */
    [[nodiscard]] static std::string generate_text_report(const MonteCarloReport& rep);

private:
    MonteCarloConfig config_;
};

} // namespace quant::simulation
