#include "test_harness.hpp"
#include "test_support.hpp"
#include "quant/analysis/cost_sensitivity.hpp"
#include "quant/analysis/walk_forward.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/backtest/strategies/vol_target.hpp"
#include "quant/backtest/strategies/buy_and_hold.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

quant::data::MarketDataUniverse trending_universe(size_t bars) {
    std::vector<double> close(bars);
    for (size_t t = 0; t < bars; ++t) {
        const double x = static_cast<double>(t);
        close[t] = 100.0 + 0.05 * x + 8.0 * std::sin(x / 15.0) + 2.0 * std::sin(x / 3.7);
    }
    return quant::tests::make_universe({{"AAA", close, {}}});
}

quant::analysis::BacktestSetup frictionless_setup() {
    quant::analysis::BacktestSetup setup;
    setup.execution = quant::tests::frictionless_execution();
    return setup;
}

}   // namespace

TEST_CASE(TestAnalysis_EvaluateWindow_MeasuresOnlyTheWindow) {
    auto universe = trending_universe(200);
    quant::backtest::strategies::BuyAndHoldStrategy strategy("AAA", 0.99);

    std::vector<double> returns;
    const auto perf =
        quant::analysis::evaluate_window(universe, strategy, frictionless_setup(), 100, 160, 20, &returns);

    EXPECT_EQ(returns.size(), 59u);   // 60 bars in the window -> 59 daily returns
    EXPECT_EQ(perf.days, 59u);
    EXPECT_TRUE(perf.trades >= 1u);   // warm-up trading is blocked, but the window itself trades
}

TEST_CASE(TestAnalysis_Sweep_CoversGridAndIsSortedBySharpe) {
    auto universe = trending_universe(300);
    const auto sweep = quant::analysis::sweep_sma_parameters(universe, "AAA", {5, 10}, {20, 40},
                                                             frictionless_setup(), 0, universe.size(), 0);

    EXPECT_EQ(sweep.size(), 4u);   // every (fast < slow) pair
    for (const auto& point : sweep) {
        EXPECT_TRUE(point.fast_period < point.slow_period);
    }
    for (size_t i = 1; i < sweep.size(); ++i) {
        EXPECT_TRUE(sweep[i - 1].performance.sharpe_ratio >= sweep[i].performance.sharpe_ratio);
    }
}

TEST_CASE(TestAnalysis_WalkForward_TestWindowsAreUnseenAndDisjoint) {
    auto universe = trending_universe(400);

    quant::analysis::WalkForwardConfig cfg;
    cfg.train_bars = 120;
    cfg.test_bars = 40;
    cfg.fast_grid = {3, 5};
    cfg.slow_grid = {10, 20};

    const auto result = quant::analysis::run_sma_walk_forward(universe, "AAA", cfg, frictionless_setup());

    // train_begin = 0, 40, ..., 240 while train_begin + 160 <= 400
    EXPECT_EQ(result.folds.size(), 7u);
    EXPECT_EQ(result.oos_returns.size(), result.folds.size() * (cfg.test_bars - 1));
    EXPECT_EQ(result.benchmark_returns.size(), result.oos_returns.size());

    for (size_t i = 0; i < result.folds.size(); ++i) {
        const auto& fold = result.folds[i];
        // Parameters must come from the grid...
        EXPECT_TRUE(fold.fast_period == 3 || fold.fast_period == 5);
        EXPECT_TRUE(fold.slow_period == 10 || fold.slow_period == 20);
        // ...and the tested window must start strictly after the window they were fitted on.
        EXPECT_TRUE(fold.test_start > fold.train_end);
        if (i > 0) {
            EXPECT_TRUE(fold.test_start > result.folds[i - 1].test_end);
        }
    }
}

TEST_CASE(TestAnalysis_CostSweep_ReturnsFallAsCostsRise) {
    auto universe = trending_universe(400);
    quant::analysis::BacktestSetup setup;   // default (non-zero) costs

    const auto rows = quant::analysis::sweep_costs(
        universe,
        []() {
            std::vector<std::unique_ptr<quant::backtest::Strategy>> built;
            built.push_back(
                std::make_unique<quant::backtest::strategies::SmaCrossoverStrategy>("AAA", 5, 20));
            return built;
        },
        setup, {2.0, 0.0, 1.0});   // deliberately unsorted

    EXPECT_EQ(rows.size(), 1u);
    const auto& points = rows.front().points;
    EXPECT_EQ(points.size(), 3u);
    EXPECT_NEAR(points[0].multiplier, 0.0, 1e-12);   // sorted ascending
    EXPECT_NEAR(points[1].multiplier, 1.0, 1e-12);
    EXPECT_NEAR(points[2].multiplier, 2.0, 1e-12);
    // More expensive trading can never improve a strategy that trades at all.
    EXPECT_TRUE(points[0].performance.total_return >= points[1].performance.total_return);
    EXPECT_TRUE(points[1].performance.total_return >= points[2].performance.total_return);
    EXPECT_TRUE(points[0].performance.trades > 0u);
}

TEST_CASE(TestAnalysis_ScaleCosts_ScalesEveryComponent) {
    quant::backtest::ExecutionConfig base;
    const auto doubled = quant::analysis::scale_costs(base, 2.0);
    EXPECT_NEAR(doubled.per_share_commission, base.per_share_commission * 2.0, 1e-15);
    EXPECT_NEAR(doubled.min_commission, base.min_commission * 2.0, 1e-15);
    EXPECT_NEAR(doubled.percentage_commission, base.percentage_commission * 2.0, 1e-15);
    EXPECT_NEAR(doubled.spread_bps, base.spread_bps * 2.0, 1e-15);
    EXPECT_NEAR(doubled.fixed_slippage_bps, base.fixed_slippage_bps * 2.0, 1e-15);
    EXPECT_NEAR(doubled.market_impact_factor, base.market_impact_factor * 2.0, 1e-15);

    const auto free_trading = quant::analysis::scale_costs(base, 0.0);
    EXPECT_NEAR(free_trading.min_commission, 0.0, 1e-15);
    EXPECT_NEAR(free_trading.spread_bps, 0.0, 1e-15);
}

TEST_CASE(TestAnalysis_VolTarget_SizesDownWhenVolatilityRises) {
    // Calm first half (volatility below the 10% target, so the position is capped at 100%), then the
    // same drift with a much larger wiggle (volatility well above target, so the position shrinks).
    const size_t bars = 400;
    std::vector<double> close(bars);
    for (size_t t = 0; t < bars; ++t) {
        const double x = static_cast<double>(t);
        const double amplitude = t < bars / 2 ? 0.4 : 6.0;
        close[t] = 100.0 + 0.02 * x + amplitude * std::sin(x / 2.3);
    }
    auto universe = quant::tests::make_universe({{"AAA", close, {}}});

    quant::backtest::strategies::VolatilityTargetStrategy strategy("AAA", 0.10, 20);
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(100000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()),
        quant::backtest::EngineConfig{});
    const auto result = engine.run(strategy);

    // Exposure is 1 - cash/equity. Compare two steady-state windows, skipping the indicator warm-up
    // at the start and the bars right after the regime change, where the rolling window is mixed.
    const auto average_exposure = [&result](size_t begin, size_t end) {
        double total = 0.0;
        size_t count = 0;
        for (size_t i = begin; i < end && i < result.equity_curve.size(); ++i) {
            const auto& point = result.equity_curve[i];
            if (point.equity <= 0.0) continue;
            total += 1.0 - point.cash / point.equity;
            ++count;
        }
        return count > 0 ? total / static_cast<double>(count) : 0.0;
    };

    const double calm = average_exposure(100, 190);
    const double rough = average_exposure(300, 390);
    EXPECT_TRUE(calm > 0.0);
    EXPECT_TRUE(rough > 0.0);
    EXPECT_TRUE(calm > rough);   // four times the volatility must mean a smaller position
}
