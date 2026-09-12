#include "test_harness.hpp"
#include "quant/risk/report.hpp"
#include "quant/backtest/engine.hpp"
#include "quant/risk/metrics.hpp"
#include "quant/risk/var_cvar.hpp"
#include <cmath>

TEST_CASE(TestRisk_CAGR) {
    // 100% total return over 2 years -> CAGR = sqrt(2.0) - 1 ~ 41.42%
    double cagr_2yr = quant::risk::RiskMetrics::cagr(1.0, 2.0);
    EXPECT_NEAR(cagr_2yr, std::sqrt(2.0) - 1.0, 1e-6);

    // 44% total return over 2 years -> sqrt(1.44) - 1 = 1.2 - 1 = 20%
    double cagr_exact = quant::risk::RiskMetrics::cagr(0.44, 2.0);
    EXPECT_NEAR(cagr_exact, 0.20, 1e-6);
}

TEST_CASE(TestRisk_Sharpe_Sortino) {
    // Returns: 1%, 2%, 3%, -1%, 2%
    std::vector<double> rets = {0.01, 0.02, 0.03, -0.01, 0.02};
    double mean_r = quant::risk::RiskMetrics::mean(rets);
    EXPECT_NEAR(mean_r, 0.014, 1e-6);

    double sharpe = quant::risk::RiskMetrics::sharpe_ratio(rets, 0.0, 252.0);
    EXPECT_TRUE(sharpe > 0.0);

    double sortino = quant::risk::RiskMetrics::sortino_ratio(rets, 0.0, 252.0);
    // Sortino should be strictly greater than Sharpe because upside volatility is not penalized
    EXPECT_TRUE(sortino > sharpe);
}

TEST_CASE(TestRisk_MaxDrawdown) {
    std::vector<double> equity = {100.0, 120.0, 150.0, 120.0, 105.0, 130.0, 160.0};
    auto dd_info = quant::risk::RiskMetrics::calculate_drawdown(equity);

    // Peak = 150 (index 2), Trough = 105 (index 4) -> MaxDD = (150 - 105) / 150 = 30%
    EXPECT_NEAR(dd_info.max_drawdown, 0.30, 1e-6);
    EXPECT_EQ(dd_info.peak_index, 2);
    EXPECT_EQ(dd_info.trough_index, 4);
    EXPECT_EQ(dd_info.recovery_index, 6); // recovers back to 160 >= 150 at index 6
}

TEST_CASE(TestRisk_VaR_CVaR) {
    std::vector<double> rets(100);
    for (size_t i = 0; i < 100; ++i) {
        rets[i] = -0.05 + static_cast<double>(i) * 0.001; // from -5% to +4.9%
    }

    auto hist = quant::risk::ValueAtRisk::historical(rets, 0.95);
    // 5% worst loss cutoff
    EXPECT_TRUE(hist.var > 0.0);
    // Expected shortfall must be strictly greater than or equal to VaR
    EXPECT_TRUE(hist.cvar >= hist.var);

    auto param = quant::risk::ValueAtRisk::parametric(rets, 0.95);
    EXPECT_TRUE(param.var > 0.0);
    EXPECT_TRUE(param.cvar >= param.var);
}

TEST_CASE(TestRisk_HistoricalVaR_ExactQuantile) {
    // Returns -5.0%, -4.9%, ... in 0.1% steps: the 5% worst observation is -4.5%.
    std::vector<double> rets(100);
    for (size_t i = 0; i < 100; ++i) rets[i] = -0.05 + static_cast<double>(i) * 0.001;

    const auto hist = quant::risk::ValueAtRisk::historical(rets, 0.95);
    EXPECT_NEAR(hist.var, 0.045, 1e-12);
    // CVaR is the mean of the six observations at or below that quantile.
    EXPECT_NEAR(hist.cvar, 0.0475, 1e-12);
}

// --- Regression: annualization must use the calendar span, not the bar count ---
TEST_CASE(TestRisk_CagrUsesCalendarDates) {
    quant::backtest::BacktestResult result;
    result.strategy_name = "Synthetic";
    result.initial_cash = 100000.0;
    result.final_equity = 144000.0;
    result.total_return = 0.44;
    result.total_bars = 400;              // deliberately inconsistent with the calendar span
    result.start_timestamp = 1000000000;
    result.end_timestamp = result.start_timestamp + static_cast<int64_t>(2.0 * 365.25 * 86400.0);

    const auto summary = quant::risk::RiskReport::evaluate(result, 0.02);
    EXPECT_NEAR(summary.years, 2.0, 1e-9);
    EXPECT_NEAR(summary.cagr, 0.20, 1e-9);   // sqrt(1.44) - 1
}
