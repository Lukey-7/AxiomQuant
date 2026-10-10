#include "test_harness.hpp"
#include "quant/analysis/regimes.hpp"
#include "quant/analysis/walk_forward.hpp"
#include "quant/risk/metrics.hpp"
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using quant::analysis::RegimeConfig;
using quant::analysis::TrendRegime;
using quant::analysis::VolatilityRegime;

/// Log-price path built from (bars, drift, swing) phases; daily log returns alternate drift +/- swing,
/// so every even-length window has exactly the same dispersion within a phase.
std::vector<double> scripted_closes(const std::vector<std::vector<double>>& phases) {
    std::vector<double> closes{100.0};
    size_t t = 0;
    for (const auto& phase : phases) {
        const auto bars = static_cast<size_t>(phase[0]);
        for (size_t k = 0; k < bars; ++k, ++t) {
            const double r = phase[1] + (t % 2 == 0 ? phase[2] : -phase[2]);
            closes.push_back(closes.back() * std::exp(r));
        }
    }
    return closes;
}

/// Calm rise, then a volatile fall, then a very calm recovery: 801 closes.
std::vector<double> three_phase_closes() {
    return scripted_closes({{300, 0.001, 0.01}, {300, -0.002, 0.03}, {200, 0.003, 0.003}});
}

RegimeConfig test_config() {
    RegimeConfig config;
    config.vol_window = 20;   // even, so the alternating swing contributes identically to every window
    config.min_vol_history = 100;
    return config;
}

std::vector<double> simple_returns(const std::vector<double>& closes) {
    std::vector<double> returns;
    for (size_t t = 1; t < closes.size(); ++t)
        returns.push_back(closes[t] / closes[t - 1] - 1.0);
    return returns;
}

bool same_value(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || a == b;
}

}   // namespace

TEST_CASE(TestRegimes_Classify_TrendFlipsWhereTheScriptSaysSo) {
    const auto closes = three_phase_closes();
    const auto labels = quant::analysis::classify_regimes(closes, test_config());

    EXPECT_EQ(labels.size(), closes.size());
    for (size_t t = 0; t < 200; ++t) {
        EXPECT_TRUE(labels.trend[t] == TrendRegime::Unknown);   // no 200-day history yet
        EXPECT_TRUE(std::isnan(labels.trailing_return[t]));
    }
    for (size_t t = 200; t <= 300; ++t)
        EXPECT_TRUE(labels.trend[t] == TrendRegime::Rising);
    for (size_t t = 460; t <= 600; ++t)
        EXPECT_TRUE(labels.trend[t] == TrendRegime::Falling);
    for (size_t t = 760; t < closes.size(); ++t)
        EXPECT_TRUE(labels.trend[t] == TrendRegime::Rising);
    EXPECT_NEAR(labels.trailing_return[250], closes[250] / closes[50] - 1.0, 1e-12);
}

TEST_CASE(TestRegimes_Classify_VolatilityTercilesFollowTheShifts) {
    const auto closes = three_phase_closes();
    const auto config = test_config();
    const auto labels = quant::analysis::classify_regimes(closes, config);

    // No volatility before vol_window returns, and no tercile before min_vol_history observations.
    for (size_t t = 0; t < config.vol_window; ++t)
        EXPECT_TRUE(std::isnan(labels.realized_vol[t]));
    const size_t first_label = config.vol_window + config.min_vol_history - 1;
    for (size_t t = 0; t < first_label; ++t)
        EXPECT_TRUE(labels.volatility[t] == VolatilityRegime::Unknown);
    EXPECT_TRUE(labels.volatility[first_label] != VolatilityRegime::Unknown);

    // Once a window lies wholly in the volatile phase it is above the calm history's upper tercile,
    // until the volatile observations themselves make up a third of the expanding history.
    for (size_t t = 321; t <= 460; ++t)
        EXPECT_TRUE(labels.volatility[t] == VolatilityRegime::High);
    // The very calm phase is below everything seen before, so it sits in the lowest tercile.
    for (size_t t = 621; t <= 700; ++t)
        EXPECT_TRUE(labels.volatility[t] == VolatilityRegime::Low);

    for (size_t t = 0; t < labels.size(); ++t) {
        if (!std::isnan(labels.low_cut[t])) EXPECT_TRUE(labels.low_cut[t] <= labels.high_cut[t]);
    }
    // Realized vol of the calm phase: ten +0.01 and ten -0.01 swings, sample stdev, annualized.
    EXPECT_NEAR(labels.realized_vol[200], 0.01 * std::sqrt(20.0 / 19.0) * std::sqrt(252.0), 1e-3);
}

TEST_CASE(TestRegimes_Classify_HasNoLookAhead) {
    const auto closes = three_phase_closes();
    const auto config = test_config();
    const auto base = quant::analysis::classify_regimes(closes, config);

    // Rewrite the future from bar 500: a crash, then wild swings. Nothing before bar 500 may move.
    auto altered = closes;
    for (size_t t = 500; t < altered.size(); ++t) {
        altered[t] = 20.0 + 15.0 * std::sin(static_cast<double>(t) * 0.7);
    }
    const auto changed = quant::analysis::classify_regimes(altered, config);

    for (size_t t = 0; t < 500; ++t) {
        EXPECT_TRUE(base.trend[t] == changed.trend[t]);
        EXPECT_TRUE(base.volatility[t] == changed.volatility[t]);
        EXPECT_TRUE(same_value(base.trailing_return[t], changed.trailing_return[t]));
        EXPECT_TRUE(same_value(base.realized_vol[t], changed.realized_vol[t]));
        EXPECT_TRUE(same_value(base.low_cut[t], changed.low_cut[t]));
        EXPECT_TRUE(same_value(base.high_cut[t], changed.high_cut[t]));
    }
    bool future_differs = false;
    for (size_t t = 500; t < closes.size(); ++t) {
        if (base.volatility[t] != changed.volatility[t] || base.trend[t] != changed.trend[t]) {
            future_differs = true;
        }
    }
    EXPECT_TRUE(future_differs);   // the alteration was large enough to matter
}

TEST_CASE(TestRegimes_Classify_MissingClosesAreUnknown) {
    auto closes = three_phase_closes();
    closes[400] = std::numeric_limits<double>::quiet_NaN();
    const auto config = test_config();
    const auto labels = quant::analysis::classify_regimes(closes, config);

    EXPECT_TRUE(labels.trend[400] == TrendRegime::Unknown);
    EXPECT_TRUE(labels.trend[600] == TrendRegime::Unknown);   // its 200-day anchor is the gap
    EXPECT_TRUE(labels.trend[601] == TrendRegime::Falling);
    for (size_t t = 400; t <= 400 + config.vol_window; ++t)
        EXPECT_TRUE(labels.volatility[t] == VolatilityRegime::Unknown);
    EXPECT_TRUE(labels.volatility[401 + config.vol_window] != VolatilityRegime::Unknown);

    EXPECT_THROW((void)quant::analysis::classify_regimes(closes, RegimeConfig{0, 21, 252, 252}),
                 std::invalid_argument);
}

TEST_CASE(TestRegimes_Breakdown_PartitionsReconcileWithTheWholeSample) {
    const auto closes = three_phase_closes();
    const auto labels = quant::analysis::classify_regimes(closes, test_config());
    const auto returns = simple_returns(closes);
    std::vector<bool> invested(returns.size());
    for (size_t i = 0; i < invested.size(); ++i)
        invested[i] = (i % 4 != 0);   // in the market three days out of four

    const double rf = 0.02;
    const auto rows = quant::analysis::regime_breakdown(returns, invested, labels, rf);
    EXPECT_EQ(rows.size(), 8u);
    EXPECT_TRUE(rows[0].regime == "All");

    const auto whole = quant::analysis::performance_from_returns(returns, rf);
    EXPECT_EQ(rows[0].days, returns.size());
    EXPECT_NEAR(rows[0].total_return, whole.total_return, 1e-12);
    EXPECT_NEAR(rows[0].sharpe_ratio, quant::risk::RiskMetrics::sharpe_ratio(returns, rf), 1e-12);
    EXPECT_NEAR(rows[0].max_drawdown, whole.max_drawdown, 1e-12);
    EXPECT_NEAR(rows[0].exposure, 600.0 / 800.0, 1e-12);

    // Trend rows (1..3) and volatility rows (4..7) each partition the days: day counts add up and
    // compounded returns multiply back to the whole-sample return.
    for (const auto& [first, last] : {std::pair<size_t, size_t>{1, 3}, std::pair<size_t, size_t>{4, 7}}) {
        size_t days = 0;
        double growth = 1.0, invested_days = 0.0;
        for (size_t r = first; r <= last; ++r) {
            days += rows[r].days;
            growth *= 1.0 + rows[r].total_return;
            invested_days += rows[r].exposure * static_cast<double>(rows[r].days);
        }
        EXPECT_EQ(days, rows[0].days);
        EXPECT_NEAR(growth - 1.0, rows[0].total_return, 1e-9);
        EXPECT_NEAR(invested_days, 600.0, 1e-9);
    }

    // Unknown trend days are exactly the first 200 returns (labels at bars 0..199).
    EXPECT_EQ(rows[3].days, 200u);
    // The scripted fall makes the falling-trend regime lose money and the rising one make money.
    EXPECT_TRUE(rows[1].total_return > 0.0);
    EXPECT_TRUE(rows[2].total_return < 0.0);
    // Annualized return is consistent with the regime's own compounded return.
    EXPECT_NEAR(rows[1].annualized_return,
                std::pow(1.0 + rows[1].total_return, 252.0 / static_cast<double>(rows[1].days)) - 1.0, 1e-12);

    // Returns are attributed to the label of the bar they start from.
    std::vector<double> rising_returns;
    for (size_t i = 0; i < returns.size(); ++i) {
        if (labels.trend[i] == TrendRegime::Rising) rising_returns.push_back(returns[i]);
    }
    EXPECT_NEAR(rows[1].sharpe_ratio, quant::risk::RiskMetrics::sharpe_ratio(rising_returns, rf), 1e-12);
}

TEST_CASE(TestRegimes_Breakdown_RejectsMismatchedInputs) {
    const auto labels = quant::analysis::classify_regimes(three_phase_closes(), test_config());
    const std::vector<double> too_many(labels.size() + 1, 0.0);
    EXPECT_THROW((void)quant::analysis::regime_breakdown(too_many, {}, labels, 0.0), std::invalid_argument);
    const std::vector<double> returns(10, 0.001);
    EXPECT_THROW((void)quant::analysis::regime_breakdown(returns, std::vector<bool>(9, true), labels, 0.0),
                 std::invalid_argument);
    const auto rows = quant::analysis::regime_breakdown(returns, {}, labels, 0.0);
    EXPECT_NEAR(rows[0].exposure, 0.0, 1e-15);   // no invested flags -> no exposure claimed
}

TEST_CASE(TestRegimes_RollingSharpe_ConstantAndPeriodicSeries) {
    // A constant return has zero volatility, so every window's Sharpe is 0 by convention.
    const std::vector<double> constant(300, 0.001);
    const auto flat = quant::analysis::rolling_sharpe(constant, 252, 0.0);
    EXPECT_EQ(flat.size(), 49u);
    for (double s : flat)
        EXPECT_NEAR(s, 0.0, 1e-12);

    // A period-2 series gives every even-length window the same sample, hence the same Sharpe.
    std::vector<double> periodic(400);
    for (size_t i = 0; i < periodic.size(); ++i)
        periodic[i] = (i % 2 == 0) ? 0.012 : -0.008;
    const auto rolling = quant::analysis::rolling_sharpe(periodic, 252, 0.02);
    EXPECT_EQ(rolling.size(), 149u);
    const std::vector<double> first(periodic.begin(), periodic.begin() + 252);
    const double expected = quant::risk::RiskMetrics::sharpe_ratio(first, 0.02);
    EXPECT_TRUE(expected > 1.0);
    for (double s : rolling)
        EXPECT_NEAR(s, expected, 1e-9);

    EXPECT_THROW((void)quant::analysis::rolling_sharpe(periodic, 1, 0.0), std::invalid_argument);
}

TEST_CASE(TestRegimes_EdgeCases_ShortHistories) {
    // 150 closes: shorter than the 200-day trend lookback and the 252-day rolling window.
    const auto closes = scripted_closes({{149, 0.001, 0.01}});
    const RegimeConfig config;
    const auto labels = quant::analysis::classify_regimes(closes, config);
    for (size_t t = 0; t < labels.size(); ++t) {
        EXPECT_TRUE(labels.trend[t] == TrendRegime::Unknown);
        EXPECT_TRUE(labels.volatility[t] == VolatilityRegime::Unknown);   // < 252 vol observations
    }

    const auto returns = simple_returns(closes);
    const auto report = quant::analysis::analyze_regimes("B&H", returns, {}, labels, config, 0.0);
    EXPECT_TRUE(report.rolling_sharpe.empty());
    EXPECT_EQ(report.regimes[0].days, returns.size());
    EXPECT_EQ(report.regimes[3].days, returns.size());   // all trend-unknown
    EXPECT_EQ(report.regimes[1].days, 0u);
    EXPECT_NEAR(report.regimes[1].total_return, 0.0, 1e-15);

    const auto text = quant::analysis::format_regime_report(labels, {report}, config, "AAA");
    EXPECT_TRUE(text.find("Skipped: fewer than 252") != std::string::npos);

    // Exactly 252 returns -> exactly one rolling window; empty input -> empty labels.
    EXPECT_EQ(quant::analysis::rolling_sharpe(std::vector<double>(252, 0.001), 252, 0.0).size(), 1u);
    EXPECT_EQ(quant::analysis::classify_regimes({}, config).size(), 0u);
}
