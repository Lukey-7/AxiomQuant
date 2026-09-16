#include "test_harness.hpp"
#include <Eigen/Dense>
#include <vector>
#include "quant/simulation/monte_carlo.hpp"
#include "quant/simulation/gbm.hpp"
#include "quant/simulation/bootstrap.hpp"
#include <cmath>
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
