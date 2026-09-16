#include "test_harness.hpp"
#include "quant/analysis/significance.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

// Reference values below were computed independently in NumPy (statistics.NormalDist for Phi and
// Phi^-1) from the same deterministic series, using the same sample skewness / kurtosis estimators
// as RiskMetrics.

namespace {

std::vector<double> reference_series(size_t n) {
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        r[i] =
            0.0004 + 0.01 * std::sin(0.7 * t) + 0.006 * std::sin(0.013 * t * t) + 0.003 * std::cos(1.9 * t);
    }
    return r;
}

}   // namespace

TEST_CASE(TestSignificance_NormalCdf_KnownValues) {
    EXPECT_NEAR(quant::analysis::normal_cdf(0.0), 0.5, 1e-15);
    EXPECT_NEAR(quant::analysis::normal_cdf(1.959963984540054), 0.975, 1e-12);
    EXPECT_NEAR(quant::analysis::normal_cdf(-1.6448536269514722), 0.05, 1e-12);
}

TEST_CASE(TestSignificance_ProbabilisticSharpe_MatchesReference) {
    const auto r = reference_series(750);
    EXPECT_NEAR(quant::analysis::probabilistic_sharpe_ratio(r, 0.0, 0.02), 0.8386085863389722, 1e-9);
    EXPECT_NEAR(quant::analysis::probabilistic_sharpe_ratio(r, 0.5, 0.02), 0.5511162883541261, 1e-9);
    // A higher hurdle can only lower the probability of clearing it.
    EXPECT_TRUE(quant::analysis::probabilistic_sharpe_ratio(r, 1.0, 0.02) <
                quant::analysis::probabilistic_sharpe_ratio(r, 0.5, 0.02));
}

TEST_CASE(TestSignificance_ExpectedMaximumSharpe_MatchesReference) {
    EXPECT_NEAR(quant::analysis::expected_maximum_sharpe(1, 1.0), 0.0, 1e-15);
    EXPECT_NEAR(quant::analysis::expected_maximum_sharpe(41, 1.0), 2.1992205128938647, 1e-6);
    EXPECT_NEAR(quant::analysis::expected_maximum_sharpe(1000, 1.0), 3.255121513652723, 1e-6);
    EXPECT_NEAR(quant::analysis::expected_maximum_sharpe(41, 0.5), 0.5 * 2.1992205128938647, 1e-6);
}

TEST_CASE(TestSignificance_DeflatedSharpe_MatchesReference) {
    const auto r = reference_series(750);
    const std::vector<double> trials{0.9, 0.1, -0.3, 0.45, 0.2, -0.05, 0.6, 0.3, 0.0, -0.2};
    const auto dsr = quant::analysis::deflated_sharpe_ratio(r, trials, 0.02);

    EXPECT_EQ(dsr.trials, 10u);
    EXPECT_EQ(dsr.observations, 750u);
    EXPECT_NEAR(dsr.sharpe, 0.5746751972287912, 1e-12);
    EXPECT_NEAR(dsr.trial_sharpe_std, 0.3719318934070233, 1e-12);
    EXPECT_NEAR(dsr.expected_max_sharpe, 0.5856433275750075, 1e-6);
    EXPECT_NEAR(dsr.probabilistic_sharpe, 0.8386085863389722, 1e-9);
    EXPECT_NEAR(dsr.deflated_sharpe, 0.49247193587208515, 1e-6);
    EXPECT_TRUE(dsr.deflated_sharpe < dsr.probabilistic_sharpe);
}

TEST_CASE(TestSignificance_Bootstrap_IsDeterministicPerSeed) {
    const auto r = reference_series(400);
    quant::analysis::BootstrapConfig cfg;
    cfg.resamples = 500;
    cfg.seed = 7;
    const auto a = quant::analysis::bootstrap_sharpe(r, {}, cfg);
    const auto b = quant::analysis::bootstrap_sharpe(r, {}, cfg);
    cfg.seed = 8;
    const auto c = quant::analysis::bootstrap_sharpe(r, {}, cfg);

    EXPECT_EQ(a.strategy.lower, b.strategy.lower);
    EXPECT_EQ(a.strategy.upper, b.strategy.upper);
    EXPECT_EQ(a.strategy.p_value, b.strategy.p_value);
    EXPECT_TRUE(a.strategy.lower != c.strategy.lower || a.strategy.upper != c.strategy.upper);
    // The block length now comes from the data (Politis-White), capped at 3 sqrt(n).
    EXPECT_TRUE(a.mean_block_length >= 1.0 && a.mean_block_length <= 3.0 * std::sqrt(400.0));
    EXPECT_EQ(a.mean_block_length, b.mean_block_length);
    EXPECT_TRUE(a.strategy.lower <= a.strategy.estimate && a.strategy.estimate <= a.strategy.upper);
    EXPECT_FALSE(a.has_benchmark);
}

TEST_CASE(TestSignificance_Bootstrap_DetectsStrongEdgeAndNoEdge) {
    quant::analysis::BootstrapConfig cfg;
    cfg.resamples = 2000;
    cfg.risk_free_rate = 0.0;

    std::vector<double> strong(500), flat(500);
    for (size_t i = 0; i < 500; ++i) {
        const double t = static_cast<double>(i);
        strong[i] = 0.002 + 0.005 * std::sin(1.3 * t);
        flat[i] = (i % 2 == 0) ? 0.01 : -0.01;   // mean exactly zero
    }

    const auto edge = quant::analysis::bootstrap_sharpe(strong, {}, cfg);
    EXPECT_TRUE(edge.strategy.estimate > 1.0);
    EXPECT_TRUE(edge.strategy.lower > 0.0);
    EXPECT_TRUE(edge.strategy.p_value < 0.01);

    const auto none = quant::analysis::bootstrap_sharpe(flat, {}, cfg);
    EXPECT_NEAR(none.strategy.estimate, 0.0, 1e-9);
    EXPECT_TRUE(none.strategy.p_value > 0.2);
    EXPECT_TRUE(none.strategy.lower < 0.0 && none.strategy.upper > 0.0);
}

TEST_CASE(TestSignificance_Bootstrap_PairsStrategyAndBenchmarkIndices) {
    const auto r = reference_series(300);
    quant::analysis::BootstrapConfig cfg;
    cfg.resamples = 300;

    // Identical series resampled with shared indices differ by exactly zero in every resample.
    const auto same = quant::analysis::bootstrap_sharpe(r, r, cfg);
    EXPECT_TRUE(same.has_benchmark);
    EXPECT_EQ(same.difference.estimate, 0.0);
    EXPECT_EQ(same.difference.lower, 0.0);
    EXPECT_EQ(same.difference.upper, 0.0);
    EXPECT_NEAR(same.difference.p_value, 1.0, 1e-12);
    EXPECT_EQ(same.strategy.lower, same.benchmark.lower);
}

TEST_CASE(TestSignificance_Bootstrap_RejectsBadInput) {
    const auto r = reference_series(50);
    quant::analysis::BootstrapConfig cfg;
    EXPECT_THROW((void)quant::analysis::bootstrap_sharpe({0.01, 0.02}, {}, cfg), std::invalid_argument);
    EXPECT_THROW((void)quant::analysis::bootstrap_sharpe(r, reference_series(49), cfg),
                 std::invalid_argument);
    cfg.confidence = 1.0;
    EXPECT_THROW((void)quant::analysis::bootstrap_sharpe(r, {}, cfg), std::invalid_argument);
}

TEST_CASE(TestSignificance_BlockLength_ReflectsPersistence) {
    // White-ish noise: a deterministic but effectively uncorrelated series.
    std::vector<double> noise(600);
    unsigned long long state = 88172645463325252ULL;
    for (auto& x : noise) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        x = static_cast<double>(state % 20001) / 10000.0 - 1.0;
    }
    // AR(1) with phi = 0.9: the same shocks, but each day remembers the last.
    std::vector<double> persistent(noise.size());
    double level = 0.0;
    for (size_t i = 0; i < noise.size(); ++i) {
        level = 0.9 * level + noise[i];
        persistent[i] = level;
    }

    const double noise_block = quant::analysis::politis_white_block_length(noise);
    const double persistent_block = quant::analysis::politis_white_block_length(persistent);
    EXPECT_TRUE(noise_block >= 1.0);
    EXPECT_TRUE(noise_block < 5.0);                      // nothing to preserve: short blocks
    EXPECT_TRUE(persistent_block > 3.0 * noise_block);   // dependence demands longer ones
    EXPECT_TRUE(persistent_block <= 3.0 * std::sqrt(static_cast<double>(noise.size())));

    // Too short, or constant, gives no estimate rather than a wrong one.
    EXPECT_TRUE(std::isnan(quant::analysis::politis_white_block_length({0.1, 0.2, 0.3})));
    EXPECT_TRUE(std::isnan(quant::analysis::politis_white_block_length(std::vector<double>(200, 0.01))));

    // An explicit block length is still honoured.
    quant::analysis::BootstrapConfig cfg;
    cfg.resamples = 200;
    cfg.mean_block_length = 12.0;
    EXPECT_NEAR(quant::analysis::bootstrap_sharpe(persistent, {}, cfg).mean_block_length, 12.0, 1e-12);
}

TEST_CASE(TestSignificance_EffectiveTrials_DiscountsSimilarTrials) {
    const auto base = reference_series(300);
    std::vector<double> shifted(base.size()), opposite(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        shifted[i] = base[i] + 0.0001 * static_cast<double>(i % 7);
        opposite[i] = base[(i * 97 + 13) % base.size()];   // same values, scrambled order
    }

    EXPECT_NEAR(quant::analysis::effective_trials({base}), 1.0, 1e-12);
    EXPECT_NEAR(quant::analysis::effective_trials({base, base, base}), 1.0, 1e-6);   // identical trials
    const double mixed = quant::analysis::effective_trials({base, shifted, opposite});
    EXPECT_TRUE(mixed > 1.0 && mixed < 3.0);
    EXPECT_TRUE(quant::analysis::effective_trials({base, opposite}) > 1.5);

    // Fewer effective trials mean a lower luck hurdle, so the deflated Sharpe cannot fall.
    const std::vector<double> trial_sharpes{0.9, 0.1, -0.3, 0.45, 0.2, -0.05, 0.6, 0.3, 0.0, -0.2};
    const auto independent =
        quant::analysis::deflated_sharpe_ratio(reference_series(750), trial_sharpes, 0.02);
    const auto correlated =
        quant::analysis::deflated_sharpe_ratio(reference_series(750), trial_sharpes, 0.02, 252.0, 2.0);
    EXPECT_NEAR(independent.effective_trials, 10.0, 1e-12);
    EXPECT_NEAR(correlated.effective_trials, 2.0, 1e-12);
    EXPECT_TRUE(correlated.expected_max_sharpe < independent.expected_max_sharpe);
    EXPECT_TRUE(correlated.deflated_sharpe > independent.deflated_sharpe);
}
