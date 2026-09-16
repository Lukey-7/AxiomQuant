#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>

namespace quant::simulation {

struct MonteCarloConfig {
    size_t num_simulations{50000};
    size_t horizon_days{252};
    double initial_wealth{100000.0};
    uint64_t seed{42};
    bool use_bootstrap{false};     // run_simulation: resample history (true) or fit a GBM (false)
    double block_length{1.0};      // bootstrap mean block length in days; 1 = i.i.d. resampling
    bool rebalance_daily{false};   // run_gbm_portfolio: constant-mix (true) or buy-and-hold (false)
    size_t num_threads{0};         // 0 = OpenMP default
};

struct MonteCarloReport {
    std::string method;
    size_t num_simulations{0};
    size_t horizon_days{0};
    double initial_wealth{0.0};
    int threads_used{1};

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
    double prob_loss{0.0};            // Prob(Ending Wealth < Initial Wealth)
    double prob_loss_gt_10pct{0.0};   // Prob(Ending Wealth < 0.9 * Initial)
    double prob_loss_gt_20pct{0.0};   // Prob(Ending Wealth < 0.8 * Initial)
    double var_95_terminal{0.0};
    double cvar_95_terminal{0.0};

    double elapsed_ms{0.0};
    double paths_per_second{0.0};
};

/**
 * @brief Parallel Monte Carlo engine.
 *
 * Every path owns an independent xoshiro256** stream seeded from (seed, path index), and normals
 * are drawn with a portable Marsaglia polar sampler. Results are therefore bit-identical for a
 * given seed regardless of thread count, scheduling, or standard library implementation.
 * Paths are simulated in streaming fashion (no per-path allocation).
 */
class MonteCarloEngine {
public:
    explicit MonteCarloEngine(MonteCarloConfig config = MonteCarloConfig{}) : config_(config) {}

    /**
     * @brief Simulate a single return stream (e.g. backtest daily returns), either by bootstrap
     *        resampling or by a GBM fitted to the stream's mean and volatility.
     *
     * With `block_length` above 1 the bootstrap is the stationary block bootstrap of Politis & Romano
     * (1994): each day continues the previous day's block with probability 1 - 1/L and otherwise
     * jumps to a fresh uniform start, wrapping at the end of the history. Blocks keep volatility
     * clustering and short-range autocorrelation that i.i.d. resampling destroys, which matters for
     * drawdown statistics far more than for terminal wealth.
     */
    [[nodiscard]] MonteCarloReport run_simulation(const std::vector<double>& historical_returns) const;

    /**
     * @brief Correlated multi-asset GBM simulation of a weighted portfolio.
     *
     * Asset log-returns are drawn as  x_t = drift + L z_t  with  L L^T = Sigma_daily  (Cholesky),
     * so the full cross-asset correlation structure is preserved.
     *
     * @param expected_returns Annualized arithmetic expected returns (N).
     * @param cov_matrix       DAILY covariance matrix (N x N).
     * @param weights          Portfolio weights (N), typically summing to 1.
     */
    [[nodiscard]] MonteCarloReport run_gbm_portfolio(const Eigen::VectorXd& expected_returns,
                                                     const Eigen::MatrixXd& cov_matrix,
                                                     const Eigen::VectorXd& weights) const;

    /**
     * @brief Formats report as rich text table.
     */
    [[nodiscard]] static std::string generate_text_report(const MonteCarloReport& rep);

    [[nodiscard]] const MonteCarloConfig& get_config() const noexcept { return config_; }

private:
    MonteCarloConfig config_;
};

}   // namespace quant::simulation
