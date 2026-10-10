#include "test_harness.hpp"
#include <Eigen/Dense>
#include <vector>
#include "quant/simulation/monte_carlo.hpp"
#include "quant/simulation/gbm.hpp"
#include "quant/simulation/bootstrap.hpp"
#include "quant/simulation/rng.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

TEST_CASE(TestSimulation_GBM_Theoretical_Convergence) {
    // 50,000 paths with S0 = 100, mu = 10%, sigma = 20%, 1 year (252 days)
    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = 40000;
    cfg.horizon_days = 252;
    cfg.initial_wealth = 100.0;
    cfg.seed = 12345;
    cfg.use_bootstrap = false;

    // Simulated historical returns with exact mu and sigma
    std::vector<double> mock_returns(252, 0.10 / 252.0);

    quant::simulation::MonteCarloEngine engine(cfg);
    auto rep = engine.run_simulation(mock_returns);

    // Theoretical expected value: E[S_T] = S_0 * exp(mu * T) = 100 * exp(0.10) ~ 110.517
    double theoretical_mean = 100.0 * std::exp(0.10);
    // Allow small Monte Carlo error margin (within 1.5%)
    EXPECT_NEAR(rep.mean_terminal_wealth, theoretical_mean, theoretical_mean * 0.015);
    EXPECT_TRUE(rep.elapsed_ms > 0.0);
    EXPECT_TRUE(rep.prob_loss >= 0.0 && rep.prob_loss <= 1.0);
}

TEST_CASE(TestSimulation_Bootstrap_Bounds) {
    std::vector<double> historical = {0.01, -0.005, 0.02, -0.015, 0.008};
    std::mt19937_64 rng(42);

    auto path = quant::simulation::BootstrapSimulator::simulate_portfolio_path(historical, 1000.0, 50, rng);
    EXPECT_EQ(path.size(), 51);
    EXPECT_NEAR(path[0], 1000.0, 1e-6);

    for (double val : path) {
        EXPECT_TRUE(val >= 0.0);
    }
}

// --- Regression: results must not depend on how paths are spread over threads ---
TEST_CASE(TestSimulation_ReproducibleAcrossThreadCounts) {
    const std::vector<double> history{0.012, -0.004, 0.007, -0.011, 0.003, 0.009, -0.006};

    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = 4000;
    cfg.horizon_days = 60;
    cfg.initial_wealth = 100000.0;
    cfg.seed = 2024;
    cfg.use_bootstrap = true;

    cfg.num_threads = 1;
    const auto serial = quant::simulation::MonteCarloEngine(cfg).run_simulation(history);
    cfg.num_threads = 4;
    const auto parallel = quant::simulation::MonteCarloEngine(cfg).run_simulation(history);

    EXPECT_NEAR(serial.mean_terminal_wealth, parallel.mean_terminal_wealth, 0.0);
    EXPECT_NEAR(serial.p05_wealth, parallel.p05_wealth, 0.0);
    EXPECT_NEAR(serial.p95_wealth, parallel.p95_wealth, 0.0);
    EXPECT_NEAR(serial.mean_max_drawdown, parallel.mean_max_drawdown, 0.0);
    EXPECT_NEAR(serial.prob_loss, parallel.prob_loss, 0.0);
}

TEST_CASE(TestSimulation_CorrelatedGbmPortfolio_BuyAndHoldMean) {
    // Two assets, 20% and 30% annual vol, correlation 0.5, held without rebalancing.
    // E[W_T] = W_0 * sum_i w_i * exp(mu_i * T)
    Eigen::VectorXd mu(2);
    mu << 0.08, 0.12;
    Eigen::MatrixXd cov_daily(2, 2);
    cov_daily << 0.04, 0.03, 0.03, 0.09;
    cov_daily /= 252.0;
    Eigen::VectorXd weights(2);
    weights << 0.6, 0.4;

    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = 40000;
    cfg.horizon_days = 252;
    cfg.initial_wealth = 100.0;
    cfg.seed = 7;
    const auto report = quant::simulation::MonteCarloEngine(cfg).run_gbm_portfolio(mu, cov_daily, weights);

    const double expected = 100.0 * (0.6 * std::exp(0.08) + 0.4 * std::exp(0.12));
    EXPECT_NEAR(report.mean_terminal_wealth, expected, expected * 0.015);
    EXPECT_TRUE(report.mean_max_drawdown > 0.0);
}

TEST_CASE(TestSimulation_CorrelatedGbmPortfolio_DailyRebalancedMean) {
    Eigen::VectorXd mu(2);
    mu << 0.08, 0.12;
    Eigen::MatrixXd cov_daily(2, 2);
    cov_daily << 0.04, 0.03, 0.03, 0.09;
    cov_daily /= 252.0;
    Eigen::VectorXd weights(2);
    weights << 0.6, 0.4;

    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = 40000;
    cfg.horizon_days = 252;
    cfg.initial_wealth = 100.0;
    cfg.seed = 11;
    cfg.rebalance_daily = true;
    const auto report = quant::simulation::MonteCarloEngine(cfg).run_gbm_portfolio(mu, cov_daily, weights);

    // Constant-mix: E[W_T] = W_0 * (sum_i w_i e^{mu_i dt})^H
    const double step = 0.6 * std::exp(0.08 / 252.0) + 0.4 * std::exp(0.12 / 252.0);
    const double expected = 100.0 * std::pow(step, 252.0);
    EXPECT_NEAR(report.mean_terminal_wealth, expected, expected * 0.015);
}

TEST_CASE(TestMonteCarlo_BlockBootstrap_PreservesContiguousBlocks) {
    // A cyclic rotation of the whole history has the same product of gross returns whatever the
    // starting point, so with a block far longer than the horizon every path ends at the same wealth.
    std::vector<double> history{0.02, -0.01, 0.03, -0.02, 0.015, -0.005, 0.01, 0.0, -0.03, 0.025};
    double product = 1.0;
    for (double r : history)
        product *= 1.0 + r;

    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = 200;
    cfg.horizon_days = history.size();
    cfg.use_bootstrap = true;
    cfg.block_length = 1e6;   // effectively never restarts inside the horizon
    cfg.seed = 5;
    const auto blocked = quant::simulation::MonteCarloEngine(cfg).run_simulation(history);

    EXPECT_NEAR(blocked.mean_terminal_wealth, cfg.initial_wealth * product, 1e-6);
    EXPECT_NEAR(blocked.std_terminal_wealth, 0.0, 1e-6);
    EXPECT_TRUE(blocked.method.find("block bootstrap") != std::string::npos);

    // i.i.d. resampling of the same history spreads terminal wealth out instead.
    cfg.block_length = 1.0;
    const auto iid = quant::simulation::MonteCarloEngine(cfg).run_simulation(history);
    EXPECT_TRUE(iid.std_terminal_wealth > 0.01 * cfg.initial_wealth);
    EXPECT_TRUE(iid.method.find("i.i.d.") != std::string::npos);
}

TEST_CASE(TestMonteCarlo_BlockBootstrap_IsDeterministicAndThreadIndependent) {
    std::vector<double> history(300);
    for (size_t i = 0; i < history.size(); ++i)
        history[i] = 0.0004 + 0.01 * std::sin(0.3 * static_cast<double>(i));

    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = 500;
    cfg.horizon_days = 60;
    cfg.use_bootstrap = true;
    cfg.block_length = 12.0;
    cfg.seed = 99;
    cfg.num_threads = 1;
    const auto one = quant::simulation::MonteCarloEngine(cfg).run_simulation(history);
    cfg.num_threads = 4;
    const auto four = quant::simulation::MonteCarloEngine(cfg).run_simulation(history);

    EXPECT_EQ(one.mean_terminal_wealth, four.mean_terminal_wealth);
    EXPECT_EQ(one.p05_wealth, four.p05_wealth);
    EXPECT_EQ(one.p50_max_drawdown, four.p50_max_drawdown);
}

// --- Correlated GBM: the blocked kernel must reproduce the original path-by-path algorithm ---
namespace {

struct ReferencePaths {
    std::vector<double> terminal;
    std::vector<double> drawdown;
};

// The original run_gbm_portfolio loop, one path at a time, kept as the bit-exact reference.
ReferencePaths reference_gbm_paths(const Eigen::VectorXd& expected_returns,
                                   const Eigen::MatrixXd& cov_matrix,
                                   const Eigen::VectorXd& weights,
                                   const quant::simulation::MonteCarloConfig& cfg) {
    const Eigen::MatrixXd L = Eigen::LLT<Eigen::MatrixXd>(cov_matrix).matrixL();
    const size_t un = static_cast<size_t>(weights.size());
    std::vector<double> L_flat(un * un, 0.0), drift(un), w(un);
    for (size_t i = 0; i < un; ++i) {
        for (size_t j = 0; j <= i; ++j)
            L_flat[i * un + j] = L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        const auto ii = static_cast<Eigen::Index>(i);
        drift[i] = expected_returns(ii) / 252.0 - 0.5 * cov_matrix(ii, ii);
        w[i] = weights(ii);
    }

    const double S0 = cfg.initial_wealth;
    ReferencePaths out{std::vector<double>(cfg.num_simulations), std::vector<double>(cfg.num_simulations)};
    std::vector<double> z(un), rel_price(un);
    for (size_t p = 0; p < cfg.num_simulations; ++p) {
        quant::simulation::PathRng rng(cfg.seed, p);
        std::fill(rel_price.begin(), rel_price.end(), 1.0);
        double wealth = S0, peak = S0, max_dd = 0.0;
        for (size_t t = 0; t < cfg.horizon_days; ++t) {
            for (size_t i = 0; i < un; ++i)
                z[i] = rng.normal();
            double port_growth = 0.0;
            double port_value = 0.0;
            for (size_t i = 0; i < un; ++i) {
                double shock = 0.0;
                const double* Li = &L_flat[i * un];
                for (size_t j = 0; j <= i; ++j)
                    shock += Li[j] * z[j];
                const double gross = std::exp(drift[i] + shock);
                if (cfg.rebalance_daily) {
                    port_growth += w[i] * gross;
                } else {
                    rel_price[i] *= gross;
                    port_value += w[i] * rel_price[i];
                }
            }
            wealth = cfg.rebalance_daily ? wealth * port_growth : S0 * port_value;
            if (wealth < 0.0) wealth = 0.0;
            if (wealth > peak) {
                peak = wealth;
            } else if (peak > 0.0) {
                max_dd = std::max(max_dd, (peak - wealth) / peak);
            }
        }
        out.terminal[p] = wealth;
        out.drawdown[p] = max_dd;
    }
    std::sort(out.terminal.begin(), out.terminal.end());
    std::sort(out.drawdown.begin(), out.drawdown.end());
    return out;
}

double reference_percentile(const std::vector<double>& sorted, double p) {
    const double idx = p * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(idx));
    const auto upper = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lower);
    return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}

// A dense, well-conditioned daily covariance with unequal volatilities and mixed-sign correlations.
Eigen::MatrixXd make_test_covariance(Eigen::Index n) {
    Eigen::MatrixXd a(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j)
            a(i, j) = std::sin(0.7 * static_cast<double>(i + 1) + 1.3 * static_cast<double>(j * j + 1));
    }
    Eigen::MatrixXd cov = a * a.transpose() / static_cast<double>(n);
    cov.diagonal().array() += 0.05;
    return cov * (0.04 / 252.0);
}

void expect_gbm_matches_reference(Eigen::Index n, size_t paths, bool rebalance, size_t threads) {
    Eigen::VectorXd mu(n), weights(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        mu(i) = 0.03 + 0.01 * static_cast<double>(i % 7);
        weights(i) = 1.0 + 0.25 * static_cast<double>(i % 3);
    }
    weights /= weights.sum();
    const Eigen::MatrixXd cov = make_test_covariance(n);

    quant::simulation::MonteCarloConfig cfg;
    cfg.num_simulations = paths;
    cfg.horizon_days = 40;
    cfg.initial_wealth = 1000.0;
    cfg.seed = 31337;
    cfg.rebalance_daily = rebalance;
    cfg.num_threads = threads;
    const auto rep = quant::simulation::MonteCarloEngine(cfg).run_gbm_portfolio(mu, cov, weights);
    const auto ref = reference_gbm_paths(mu, cov, weights, cfg);

    const auto count = static_cast<double>(paths);
    const double mean = std::accumulate(ref.terminal.begin(), ref.terminal.end(), 0.0) / count;
    double sum_sq = 0.0;
    for (double w : ref.terminal)
        sum_sq += (w - mean) * (w - mean);
    const double std_dev = paths > 1 ? std::sqrt(sum_sq / (count - 1.0)) : 0.0;
    const double mean_dd = std::accumulate(ref.drawdown.begin(), ref.drawdown.end(), 0.0) / count;

    // Exact equality: the kernel only regroups work, never the order of floating-point operations.
    EXPECT_EQ(rep.mean_terminal_wealth, mean);
    EXPECT_EQ(rep.std_terminal_wealth, std_dev);
    EXPECT_EQ(rep.mean_max_drawdown, mean_dd);
    EXPECT_EQ(rep.p01_wealth, reference_percentile(ref.terminal, 0.01));
    EXPECT_EQ(rep.p05_wealth, reference_percentile(ref.terminal, 0.05));
    EXPECT_EQ(rep.p25_wealth, reference_percentile(ref.terminal, 0.25));
    EXPECT_EQ(rep.p50_wealth, reference_percentile(ref.terminal, 0.50));
    EXPECT_EQ(rep.p75_wealth, reference_percentile(ref.terminal, 0.75));
    EXPECT_EQ(rep.p95_wealth, reference_percentile(ref.terminal, 0.95));
    EXPECT_EQ(rep.p99_wealth, reference_percentile(ref.terminal, 0.99));
    EXPECT_EQ(rep.p50_max_drawdown, reference_percentile(ref.drawdown, 0.50));
    EXPECT_EQ(rep.p95_max_drawdown, reference_percentile(ref.drawdown, 0.95));
    EXPECT_EQ(rep.p99_max_drawdown, reference_percentile(ref.drawdown, 0.99));
}

}   // namespace

TEST_CASE(TestSimulation_CorrelatedGbm_MatchesReferenceAlgorithm) {
    // Asset and path counts straddle the kernel's row-tile and path-block edges.
    for (const Eigen::Index n : {1, 2, 3, 5, 13, 37}) {
        for (const size_t paths : {1, 7, 37}) {
            expect_gbm_matches_reference(n, paths, false, 1);
            expect_gbm_matches_reference(n, paths, true, 1);
        }
    }
    expect_gbm_matches_reference(64, 50, false, 4);
    expect_gbm_matches_reference(64, 50, true, 3);
}

TEST_CASE(TestSimulation_CorrelatedGbm_IdenticalAcrossThreadCounts) {
    const Eigen::Index n = 29;
    const Eigen::VectorXd mu = Eigen::VectorXd::Constant(n, 0.07);
    const Eigen::VectorXd weights = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
    const Eigen::MatrixXd cov = make_test_covariance(n);

    for (const bool rebalance : {false, true}) {
        quant::simulation::MonteCarloConfig cfg;
        cfg.num_simulations = 203;
        cfg.horizon_days = 50;
        cfg.seed = 77;
        cfg.rebalance_daily = rebalance;
        cfg.num_threads = 1;
        const auto one = quant::simulation::MonteCarloEngine(cfg).run_gbm_portfolio(mu, cov, weights);
        for (const size_t threads : {2, 3, 8}) {
            cfg.num_threads = threads;
            const auto many = quant::simulation::MonteCarloEngine(cfg).run_gbm_portfolio(mu, cov, weights);
            EXPECT_EQ(one.mean_terminal_wealth, many.mean_terminal_wealth);
            EXPECT_EQ(one.std_terminal_wealth, many.std_terminal_wealth);
            EXPECT_EQ(one.p01_wealth, many.p01_wealth);
            EXPECT_EQ(one.p50_wealth, many.p50_wealth);
            EXPECT_EQ(one.p99_wealth, many.p99_wealth);
            EXPECT_EQ(one.mean_max_drawdown, many.mean_max_drawdown);
            EXPECT_EQ(one.p99_max_drawdown, many.p99_max_drawdown);
            EXPECT_EQ(one.cvar_95_terminal, many.cvar_95_terminal);
        }
    }
}
