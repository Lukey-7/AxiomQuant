#include "test_harness.hpp"
#include "quant/simulation/monte_carlo.hpp"
#include "quant/simulation/gbm.hpp"
#include "quant/simulation/bootstrap.hpp"
#include <cmath>

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
