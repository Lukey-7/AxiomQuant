#include "test_harness.hpp"
#include "test_support.hpp"
#include "quant/analysis/walk_forward.hpp"
#include "quant/backtest/strategies/buy_and_hold.hpp"
#include <cmath>
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

} // namespace

TEST_CASE(TestAnalysis_EvaluateWindow_MeasuresOnlyTheWindow) {
    auto universe = trending_universe(200);
    quant::backtest::strategies::BuyAndHoldStrategy strategy("AAA", 0.99);

    std::vector<double> returns;
    const auto perf = quant::analysis::evaluate_window(universe, strategy, frictionless_setup(),
                                                       100, 160, 20, &returns);

    EXPECT_EQ(returns.size(), 59u);   // 60 bars in the window -> 59 daily returns
    EXPECT_EQ(perf.days, 59u);
    EXPECT_TRUE(perf.trades >= 1u);   // warm-up trading is blocked, but the window itself trades
}

TEST_CASE(TestAnalysis_Sweep_CoversGridAndIsSortedBySharpe) {
    auto universe = trending_universe(300);
    const auto sweep = quant::analysis::sweep_sma_parameters(
        universe, "AAA", {5, 10}, {20, 40}, frictionless_setup(), 0, universe.size(), 0);

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
