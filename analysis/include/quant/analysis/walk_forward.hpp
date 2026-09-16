#pragma once

#include "quant/backtest/engine.hpp"
#include "quant/data/universe.hpp"
#include <string>
#include <vector>

namespace quant::analysis {

/**
 * @brief Everything needed to spin up independent backtest engines (one per thread / window).
 */
struct BacktestSetup {
    double initial_cash{100000.0};
    backtest::ExecutionConfig execution{};
    backtest::EngineConfig engine{};
    double risk_free_rate{0.02};
};

struct WindowPerformance {
    double total_return{0.0};
    double sharpe_ratio{0.0};
    double max_drawdown{0.0};
    size_t trades{0};
    size_t days{0};
};

/**
 * @brief Performance statistics of a daily simple-return series.
 */
[[nodiscard]] WindowPerformance performance_from_returns(const std::vector<double>& returns,
                                                         double risk_free_rate);

/**
 * @brief Backtests `strategy` on timeline window [begin, end).
 *
 * Up to `warmup_bars` of preceding history are replayed first so indicators are initialised, but
 * trading is disabled until `begin`: every window starts flat, and only returns inside the window
 * are measured.
 *
 * @param window_returns Optional output: daily returns inside the window.
 */
[[nodiscard]] WindowPerformance evaluate_window(const data::MarketDataUniverse& universe,
                                                backtest::Strategy& strategy,
                                                const BacktestSetup& setup,
                                                size_t begin,
                                                size_t end,
                                                size_t warmup_bars,
                                                std::vector<double>* window_returns = nullptr);

struct SmaSweepPoint {
    size_t fast_period{0};
    size_t slow_period{0};
    WindowPerformance performance;
};

/**
 * @brief Evaluates an SMA crossover for every (fast < slow) pair on window [begin, end).
 *        Pairs are evaluated in parallel (OpenMP) and returned sorted by Sharpe ratio, best first.
 */
[[nodiscard]] std::vector<SmaSweepPoint> sweep_sma_parameters(const data::MarketDataUniverse& universe,
                                                              const std::string& ticker,
                                                              const std::vector<size_t>& fast_grid,
                                                              const std::vector<size_t>& slow_grid,
                                                              const BacktestSetup& setup,
                                                              size_t begin,
                                                              size_t end,
                                                              size_t warmup_bars);

struct WalkForwardConfig {
    size_t train_bars{504};   // ~2 years of daily bars
    size_t test_bars{126};    // ~6 months; windows roll forward by this amount
    std::vector<size_t> fast_grid{5, 10, 20, 30, 50};
    std::vector<size_t> slow_grid{50, 100, 150, 200};
};

struct WalkForwardFold {
    std::string train_start;
    std::string train_end;
    std::string test_start;
    std::string test_end;
    size_t fast_period{0};
    size_t slow_period{0};
    WindowPerformance in_sample;       // best parameters on the training window
    WindowPerformance out_of_sample;   // same parameters on the following unseen window
    WindowPerformance benchmark;       // buy-and-hold on the same unseen window
};

struct WalkForwardResult {
    size_t train_bars{0};
    size_t test_bars{0};
    std::vector<WalkForwardFold> folds;
    std::vector<double> oos_returns;         // stitched out-of-sample strategy returns
    std::vector<double> benchmark_returns;   // stitched buy-and-hold returns over the same days
    WindowPerformance stitched_strategy;
    WindowPerformance stitched_benchmark;
    double mean_in_sample_sharpe{0.0};
    double mean_out_of_sample_sharpe{0.0};
    size_t folds_beating_benchmark{0};
};

/**
 * @brief Rolling walk-forward optimisation of the SMA crossover.
 *
 * For each fold: grid-search (fast, slow) on the training window by in-sample Sharpe, then trade
 * the winning pair unchanged on the next `test_bars` bars, which the optimiser never saw. The
 * windows roll forward by `test_bars`, so the test windows tile the data without overlap.
 */
[[nodiscard]] WalkForwardResult run_sma_walk_forward(const data::MarketDataUniverse& universe,
                                                     const std::string& ticker,
                                                     const WalkForwardConfig& config,
                                                     const BacktestSetup& setup);

[[nodiscard]] std::string format_walk_forward_report(const WalkForwardResult& result,
                                                     const std::string& ticker);

}   // namespace quant::analysis
