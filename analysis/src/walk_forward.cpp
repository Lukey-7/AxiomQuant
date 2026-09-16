#include "quant/analysis/walk_forward.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/backtest/strategies/buy_and_hold.hpp"
#include "quant/risk/metrics.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace quant::analysis {

WindowPerformance performance_from_returns(const std::vector<double>& returns, double risk_free_rate) {
    WindowPerformance perf;
    perf.days = returns.size();

    double equity = 1.0, peak = 1.0;
    for (double r : returns) {
        equity *= 1.0 + r;
        peak = std::max(peak, equity);
        perf.max_drawdown = std::max(perf.max_drawdown, (peak - equity) / peak);
    }
    perf.total_return = equity - 1.0;
    perf.sharpe_ratio = risk::RiskMetrics::sharpe_ratio(returns, risk_free_rate);
    return perf;
}

WindowPerformance evaluate_window(const data::MarketDataUniverse& universe,
                                  backtest::Strategy& strategy,
                                  const BacktestSetup& setup,
                                  size_t begin,
                                  size_t end,
                                  size_t warmup_bars,
                                  std::vector<double>* window_returns) {
    end = std::min(end, universe.size());
    if (begin + 1 >= end) {
        throw std::invalid_argument("evaluate_window: window must contain at least two bars");
    }

    const size_t slice_begin = begin > warmup_bars ? begin - warmup_bars : 0;
    const size_t offset = begin - slice_begin;

    backtest::EngineConfig engine_cfg = setup.engine;
    engine_cfg.trading_start_index = offset;
    backtest::BacktestEngine engine(universe.slice(slice_begin, end), backtest::Portfolio(setup.initial_cash),
                                    backtest::ExecutionModel(setup.execution), engine_cfg);
    const auto result = engine.run(strategy);

    std::vector<double> returns;
    returns.reserve(result.equity_curve.size() - offset);
    for (size_t k = offset + 1; k < result.equity_curve.size(); ++k) {
        const double prev = result.equity_curve[k - 1].equity;
        const double curr = result.equity_curve[k].equity;
        returns.push_back(prev > 0.0 ? curr / prev - 1.0 : 0.0);
    }

    WindowPerformance perf = performance_from_returns(returns, setup.risk_free_rate);
    perf.trades = result.trades.size();
    if (window_returns) *window_returns = std::move(returns);
    return perf;
}

std::vector<SmaSweepPoint> sweep_sma_parameters(const data::MarketDataUniverse& universe,
                                                const std::string& ticker,
                                                const std::vector<size_t>& fast_grid,
                                                const std::vector<size_t>& slow_grid,
                                                const BacktestSetup& setup,
                                                size_t begin,
                                                size_t end,
                                                size_t warmup_bars) {
    if (begin + 1 >= std::min(end, universe.size())) {
        throw std::invalid_argument("sweep_sma_parameters: window must contain at least two bars");
    }

    std::vector<SmaSweepPoint> points;
    for (size_t fast : fast_grid) {
        for (size_t slow : slow_grid) {
            if (fast > 0 && fast < slow) points.push_back({fast, slow, {}});
        }
    }

    // Every pair builds its own engine and strategy, so iterations share no mutable state.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(points.size()); ++i) {
        auto& point = points[static_cast<size_t>(i)];
        backtest::strategies::SmaCrossoverStrategy strategy(ticker, point.fast_period, point.slow_period,
                                                            0.95);
        point.performance = evaluate_window(universe, strategy, setup, begin, end, warmup_bars);
    }

    std::stable_sort(points.begin(), points.end(), [](const SmaSweepPoint& a, const SmaSweepPoint& b) {
        return a.performance.sharpe_ratio > b.performance.sharpe_ratio;
    });
    return points;
}

WalkForwardResult run_sma_walk_forward(const data::MarketDataUniverse& universe,
                                       const std::string& ticker,
                                       const WalkForwardConfig& config,
                                       const BacktestSetup& setup) {
    if (config.train_bars < 2 || config.test_bars < 2) {
        throw std::invalid_argument("run_sma_walk_forward: train and test windows need at least two bars");
    }
    if (config.slow_grid.empty() || config.fast_grid.empty()) {
        throw std::invalid_argument("run_sma_walk_forward: empty parameter grid");
    }

    const size_t n = universe.size();
    const size_t warmup = *std::max_element(config.slow_grid.begin(), config.slow_grid.end());
    const auto& timeline = universe.get_timeline();

    WalkForwardResult out;
    out.train_bars = config.train_bars;
    out.test_bars = config.test_bars;

    for (size_t train_begin = 0; train_begin + config.train_bars + config.test_bars <= n;
         train_begin += config.test_bars) {
        const size_t train_end = train_begin + config.train_bars;
        const size_t test_end = train_end + config.test_bars;

        // 1. In-sample: pick the parameters with the best Sharpe on the training window only.
        const auto sweep = sweep_sma_parameters(universe, ticker, config.fast_grid, config.slow_grid, setup,
                                                train_begin, train_end, warmup);
        if (sweep.empty()) {
            throw std::invalid_argument("run_sma_walk_forward: grid has no pair with fast < slow");
        }
        const auto& best = sweep.front();

        WalkForwardFold fold;
        fold.train_start = timeline[train_begin];
        fold.train_end = timeline[train_end - 1];
        fold.test_start = timeline[train_end];
        fold.test_end = timeline[test_end - 1];
        fold.fast_period = best.fast_period;
        fold.slow_period = best.slow_period;
        fold.in_sample = best.performance;

        // 2. Out-of-sample: trade the frozen parameters on data the optimiser never saw.
        std::vector<double> oos, bench;
        backtest::strategies::SmaCrossoverStrategy strategy(ticker, best.fast_period, best.slow_period, 0.95);
        fold.out_of_sample = evaluate_window(universe, strategy, setup, train_end, test_end, warmup, &oos);

        backtest::strategies::BuyAndHoldStrategy benchmark(ticker, 0.99);
        fold.benchmark = evaluate_window(universe, benchmark, setup, train_end, test_end, 0, &bench);

        out.oos_returns.insert(out.oos_returns.end(), oos.begin(), oos.end());
        out.benchmark_returns.insert(out.benchmark_returns.end(), bench.begin(), bench.end());
        if (fold.out_of_sample.total_return > fold.benchmark.total_return) ++out.folds_beating_benchmark;
        out.folds.push_back(std::move(fold));
    }

    if (!out.folds.empty()) {
        double is_sum = 0.0, oos_sum = 0.0;
        for (const auto& f : out.folds) {
            is_sum += f.in_sample.sharpe_ratio;
            oos_sum += f.out_of_sample.sharpe_ratio;
        }
        out.mean_in_sample_sharpe = is_sum / static_cast<double>(out.folds.size());
        out.mean_out_of_sample_sharpe = oos_sum / static_cast<double>(out.folds.size());
        out.stitched_strategy = performance_from_returns(out.oos_returns, setup.risk_free_rate);
        out.stitched_benchmark = performance_from_returns(out.benchmark_returns, setup.risk_free_rate);
        for (const auto& f : out.folds) {
            out.stitched_strategy.trades += f.out_of_sample.trades;
            out.stitched_benchmark.trades += f.benchmark.trades;
        }
    }
    return out;
}

std::string format_walk_forward_report(const WalkForwardResult& r, const std::string& ticker) {
    std::ostringstream ss;
    ss << std::fixed;
    ss << "=========================================================================\n";
    ss << "        WALK-FORWARD OUT-OF-SAMPLE EVALUATION: SMA CROSSOVER (" << ticker << ")\n";
    ss << "=========================================================================\n";
    ss << "Train " << r.train_bars << " bars -> test " << r.test_bars
       << " bars, rolled forward by the test length.\n";
    ss << "Each fold picks (fast, slow) by in-sample Sharpe, then trades it unchanged on unseen data.\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "Fold  Test window               Params   IS Sharpe OOS Sharpe  OOS Ret  B&H Ret\n";
    for (size_t i = 0; i < r.folds.size(); ++i) {
        const auto& f = r.folds[i];
        const std::string params = std::to_string(f.fast_period) + "/" + std::to_string(f.slow_period);
        ss << std::setw(4) << (i + 1) << "  " << f.test_start << ".." << f.test_end << "  " << std::left
           << std::setw(7) << params << std::right << std::setprecision(3) << std::setw(10)
           << f.in_sample.sharpe_ratio << std::setw(11) << f.out_of_sample.sharpe_ratio
           << std::setprecision(2) << std::setw(8) << (f.out_of_sample.total_return * 100.0) << "%"
           << std::setw(8) << (f.benchmark.total_return * 100.0) << "%\n";
    }
    ss << "-------------------------------------------------------------------------\n";
    if (r.folds.empty()) {
        ss << "Not enough data for a single train/test fold.\n";
    } else {
        ss << "Stitched out-of-sample (" << r.oos_returns.size() << " trading days):\n";
        ss << std::setprecision(2);
        ss << "  SMA (walk-forward):  return " << std::setw(7) << (r.stitched_strategy.total_return * 100.0)
           << " %   Sharpe " << std::setprecision(3) << std::setw(6) << r.stitched_strategy.sharpe_ratio
           << "   MaxDD " << std::setprecision(2) << std::setw(6)
           << (r.stitched_strategy.max_drawdown * 100.0) << " %\n";
        ss << "  Buy & hold:          return " << std::setw(7) << (r.stitched_benchmark.total_return * 100.0)
           << " %   Sharpe " << std::setprecision(3) << std::setw(6) << r.stitched_benchmark.sharpe_ratio
           << "   MaxDD " << std::setprecision(2) << std::setw(6)
           << (r.stitched_benchmark.max_drawdown * 100.0) << " %\n";
        ss << std::setprecision(3);
        ss << "  Mean Sharpe: in-sample " << r.mean_in_sample_sharpe << "  ->  out-of-sample "
           << r.mean_out_of_sample_sharpe << "  (decay "
           << (r.mean_in_sample_sharpe - r.mean_out_of_sample_sharpe) << ")\n";
        ss << "  Folds where the strategy out-returned buy & hold: " << r.folds_beating_benchmark << " / "
           << r.folds.size() << "\n";
    }
    ss << "=========================================================================\n";
    return ss.str();
}

}   // namespace quant::analysis
