#include "test_harness.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>
#include "quant/optimization/portfolio_stats.hpp"
#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/constrained_qp.hpp"
#include "quant/optimization/efficient_frontier.hpp"
#include <Eigen/Dense>

TEST_CASE(TestOptimization_TwoAsset_Analytical_GMV) {
    // 2-asset benchmark with known closed-form solution
    // sigma_1 = 0.20 (var1 = 0.04), sigma_2 = 0.30 (var2 = 0.09), rho = 0.30 -> cov12 = 0.3 * 0.2 * 0.3 =
    // 0.018 w1_analytical = (var2 - cov12) / (var1 + var2 - 2*cov12) w1 = (0.09 - 0.018) / (0.04 + 0.09 -
    // 0.036) = 0.072 / 0.094 = 0.765957 w2 = 1 - w1 = 0.234043

    Eigen::VectorXd mu(2);
    mu << 0.10, 0.15;

    Eigen::MatrixXd cov(2, 2);
    cov << 0.04, 0.018, 0.018, 0.09;

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
    cov << 0.09, 0.02, 0.01, 0.02, 0.05, 0.01, 0.01, 0.01, 0.02;

    quant::optimization::ConstrainedQpConfig cfg;
    cfg.min_weight = 0.0;    // No short selling
    cfg.max_weight = 0.50;   // Max 50% in any single asset

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

TEST_CASE(TestOptimization_Simplex_Projection_ExactSolution) {
    // Projecting (0.8, -0.2, 1.5, 0.1) onto the unit simplex has the closed-form answer below.
    Eigen::VectorXd v(4);
    v << 0.8, -0.2, 1.5, 0.1;

    auto proj = quant::optimization::ConstrainedQpOptimizer::project_onto_bounded_simplex(v, 0.0, 1.0);
    EXPECT_NEAR(proj(0), 0.15, 1e-12);
    EXPECT_NEAR(proj(1), 0.00, 1e-12);
    EXPECT_NEAR(proj(2), 0.85, 1e-12);
    EXPECT_NEAR(proj(3), 0.00, 1e-12);
    EXPECT_NEAR(proj.sum(), 1.0, 1e-12);
}

TEST_CASE(TestOptimization_Simplex_Projection_MatchesBisection) {
    // The exact breakpoint sweep must agree with the bisection root-find it replaced.
    auto bisection_projection = [](const Eigen::VectorXd& v, double lo, double hi) {
        double low = v.minCoeff() - hi - 1.0;
        double high = v.maxCoeff() - lo + 1.0;
        for (int iter = 0; iter < 200; ++iter) {
            const double mid = 0.5 * (low + high);
            double sum = 0.0;
            for (Eigen::Index i = 0; i < v.size(); ++i)
                sum += std::clamp(v(i) - mid, lo, hi);
            if (sum > 1.0) low = mid;
            else high = mid;
        }
        Eigen::VectorXd w(v.size());
        const double theta = 0.5 * (low + high);
        for (Eigen::Index i = 0; i < v.size(); ++i)
            w(i) = std::clamp(v(i) - theta, lo, hi);
        return w;
    };

    std::mt19937_64 rng(20240517);
    std::uniform_real_distribution<double> dist(-1.5, 1.5);
    const std::vector<std::pair<double, double>> bounds{{0.0, 1.0}, {0.0, 0.40}, {-0.25, 0.60}, {0.05, 0.35}};

    for (const auto& [lo, hi] : bounds) {
        for (int trial = 0; trial < 50; ++trial) {
            Eigen::VectorXd v(8);
            for (Eigen::Index i = 0; i < v.size(); ++i)
                v(i) = dist(rng);

            auto exact = quant::optimization::ConstrainedQpOptimizer::project_onto_bounded_simplex(v, lo, hi);
            auto reference = bisection_projection(v, lo, hi);

            EXPECT_NEAR(exact.sum(), 1.0, 1e-9);
            for (Eigen::Index i = 0; i < v.size(); ++i) {
                EXPECT_NEAR(exact(i), reference(i), 1e-9);
                EXPECT_TRUE(exact(i) >= lo - 1e-12 && exact(i) <= hi + 1e-12);
            }
        }
    }
}

TEST_CASE(TestOptimization_ConstrainedMaxSharpe_IsOptimalOnTheFeasibleSet) {
    Eigen::VectorXd mu(3);
    mu << 0.25, 0.15, 0.08;
    Eigen::MatrixXd cov(3, 3);
    cov << 0.09, 0.02, 0.01, 0.02, 0.05, 0.01, 0.01, 0.01, 0.02;

    quant::optimization::ConstrainedQpConfig cfg;
    cfg.min_weight = 0.0;
    cfg.max_weight = 0.50;
    cfg.risk_free_rate = 0.02;
    quant::optimization::ConstrainedQpOptimizer optimizer(cfg);
    const auto best = optimizer.maximum_sharpe_portfolio(mu, cov);

    // No randomly sampled feasible portfolio may beat the optimizer's Sharpe ratio.
    std::mt19937_64 rng(987654321);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int trial = 0; trial < 3000; ++trial) {
        Eigen::VectorXd candidate(3);
        for (Eigen::Index i = 0; i < 3; ++i)
            candidate(i) = dist(rng);
        candidate = quant::optimization::ConstrainedQpOptimizer::project_onto_bounded_simplex(
            candidate, cfg.min_weight, cfg.max_weight);
        const double sharpe =
            quant::optimization::PortfolioStats::portfolio_sharpe(candidate, mu, cov, cfg.risk_free_rate);
        EXPECT_TRUE(sharpe <= best.sharpe_ratio + 1e-6);
    }
}

TEST_CASE(TestOptimization_LedoitWolf_MatchesReferenceImplementation) {
    // Reference values come from an independent implementation of Ledoit & Wolf (2004),
    // "Honey, I Shrunk the Sample Covariance Matrix", evaluated on this deterministic matrix.
    const int T = 60, N = 3;
    Eigen::MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) =
                0.01 * std::sin(0.37 * t + 1.3 * i) + 0.004 * std::cos(1.7 * t * (i + 1)) + 0.001 * (i - 1);
        }
    }

    double delta = -1.0;
    const auto cov =
        quant::optimization::PortfolioStats::compute_ledoit_wolf_covariance(returns, 252.0, &delta);

    EXPECT_NEAR(delta, 0.0502784214223415, 1e-10);
    EXPECT_NEAR(cov(0, 0), 0.0142072546450127, 1e-10);
    EXPECT_NEAR(cov(1, 1), 0.0146442052815486, 1e-10);
    EXPECT_NEAR(cov(2, 2), 0.0145457604820819, 1e-10);
    EXPECT_NEAR(cov(0, 1), 0.00306145294513901, 1e-10);
    EXPECT_NEAR(cov(0, 2), -0.0100980281934189, 1e-10);
    EXPECT_NEAR(cov(1, 2), 0.00325975827912009, 1e-10);

    EXPECT_TRUE(delta >= 0.0 && delta <= 1.0);
    EXPECT_NEAR(cov(1, 0), cov(0, 1), 1e-15);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    for (int i = 0; i < N; ++i)
        EXPECT_TRUE(solver.eigenvalues()(i) > 0.0);
}

TEST_CASE(TestOptimization_RiskParity_EqualizesRiskContributions) {
    Eigen::VectorXd mu(4);
    mu << 0.10, 0.08, 0.12, 0.05;
    Eigen::MatrixXd cov(4, 4);
    cov << 0.090, 0.030, 0.020, 0.004, 0.030, 0.060, 0.015, 0.003, 0.020, 0.015, 0.160, 0.010, 0.004, 0.003,
        0.010, 0.010;

    const auto rp = quant::optimization::UnconstrainedMarkowitz::risk_parity_portfolio(mu, cov);
    EXPECT_TRUE(rp.converged);
    EXPECT_NEAR(rp.weights.sum(), 1.0, 1e-12);

    // Risk contribution of asset i: w_i (Sigma w)_i / (w' Sigma w). Every asset must carry 1/N.
    const Eigen::VectorXd marginal = cov * rp.weights;
    const double variance = rp.weights.dot(marginal);
    for (Eigen::Index i = 0; i < 4; ++i) {
        EXPECT_TRUE(rp.weights(i) > 0.0);
        EXPECT_NEAR(rp.weights(i) * marginal(i) / variance, 0.25, 1e-10);
    }
}

TEST_CASE(TestOptimization_ConstrainedMaxSharpe_MatchesTangencyWhenBoundsAreSlack) {
    Eigen::VectorXd mu(3);
    mu << 0.12, 0.10, 0.07;
    Eigen::MatrixXd cov(3, 3);
    cov << 0.040, 0.006, 0.004, 0.006, 0.030, 0.005, 0.004, 0.005, 0.020;

    const auto exact = quant::optimization::UnconstrainedMarkowitz::maximum_sharpe_portfolio(mu, cov, 0.02);
    for (Eigen::Index i = 0; i < 3; ++i)
        EXPECT_TRUE(exact.weights(i) > 0.05 && exact.weights(i) < 0.9);

    quant::optimization::ConstrainedQpConfig cfg;
    cfg.min_weight = 0.0;
    cfg.max_weight = 1.0;
    cfg.risk_free_rate = 0.02;
    const auto constrained =
        quant::optimization::ConstrainedQpOptimizer(cfg).maximum_sharpe_portfolio(mu, cov);
    for (Eigen::Index i = 0; i < 3; ++i) {
        EXPECT_NEAR(constrained.weights(i), exact.weights(i), 1e-9);
    }
}
