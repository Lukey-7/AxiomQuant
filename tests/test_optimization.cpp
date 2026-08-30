#include "test_harness.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/constrained_qp.hpp"
#include "quant/optimization/efficient_frontier.hpp"
#include <Eigen/Dense>

TEST_CASE(TestOptimization_TwoAsset_Analytical_GMV) {
    // 2-asset benchmark with known closed-form solution
    // sigma_1 = 0.20 (var1 = 0.04), sigma_2 = 0.30 (var2 = 0.09), rho = 0.30 -> cov12 = 0.3 * 0.2 * 0.3 = 0.018
    // w1_analytical = (var2 - cov12) / (var1 + var2 - 2*cov12)
    // w1 = (0.09 - 0.018) / (0.04 + 0.09 - 0.036) = 0.072 / 0.094 = 0.765957
    // w2 = 1 - w1 = 0.234043

    Eigen::VectorXd mu(2);
    mu << 0.10, 0.15;

    Eigen::MatrixXd cov(2, 2);
    cov << 0.04,  0.018,
           0.018, 0.09;

    auto gmv = quant::optimization::UnconstrainedMarkowitz::global_minimum_variance(mu, cov, 0.02);

    double expected_w1 = (0.09 - 0.018) / (0.04 + 0.09 - 2.0 * 0.018);
    double expected_w2 = 1.0 - expected_w1;

    EXPECT_NEAR(gmv.weights(0), expected_w1, 1e-5);
    EXPECT_NEAR(gmv.weights(1), expected_w2, 1e-5);
    EXPECT_NEAR(gmv.weights.sum(), 1.0, 1e-6);
}

TEST_CASE(TestOptimization_Simplex_Projection) {
    Eigen::VectorXd v(4);
    v << 0.8, -0.2, 1.5, 0.1;

    auto proj = quant::optimization::ConstrainedQpOptimizer::project_onto_bounded_simplex(v, 0.0, 1.0);
    EXPECT_NEAR(proj.sum(), 1.0, 1e-6);

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(proj(i) >= -1e-8);
        EXPECT_TRUE(proj(i) <= 1.0 + 1e-8);
    }
}

TEST_CASE(TestOptimization_Constrained_LongOnly_And_BoxCaps) {
    Eigen::VectorXd mu(3);
    mu << 0.25, 0.15, 0.08;

    Eigen::MatrixXd cov(3, 3);
    cov << 0.09, 0.02, 0.01,
           0.02, 0.05, 0.01,
           0.01, 0.01, 0.02;

    quant::optimization::ConstrainedQpConfig cfg;
    cfg.min_weight = 0.0;  // No short selling
    cfg.max_weight = 0.50; // Max 50% in any single asset

    quant::optimization::ConstrainedQpOptimizer optimizer(cfg);
    auto res = optimizer.maximum_sharpe_portfolio(mu, cov);

    EXPECT_NEAR(res.weights.sum(), 1.0, 1e-5);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(res.weights(i) >= -1e-6);
        EXPECT_TRUE(res.weights(i) <= 0.50 + 1e-5);
    }
}

TEST_CASE(TestOptimization_LedoitWolf_PositiveDefinite) {
    // 100 days x 3 assets random walk returns
    Eigen::MatrixXd rets(100, 3);
    for (int r = 0; r < 100; ++r) {
        for (int c = 0; c < 3; ++c) {
            rets(r, c) = (std::sin(r * 0.1 + c) * 0.01);
        }
    }

    auto cov_shrunk = quant::optimization::PortfolioStats::compute_ledoit_wolf_covariance(rets, 252.0);
    EXPECT_EQ(cov_shrunk.rows(), 3);
    EXPECT_EQ(cov_shrunk.cols(), 3);

    // Symmetric check
    EXPECT_NEAR(cov_shrunk(0, 1), cov_shrunk(1, 0), 1e-6);
    EXPECT_NEAR(cov_shrunk(0, 2), cov_shrunk(2, 0), 1e-6);

    // Positive definite check (all eigenvalues > 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov_shrunk);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(solver.eigenvalues()(i) > 0.0);
    }
}
