#include "quant/simulation/monte_carlo.hpp"
#include "quant/simulation/gbm.hpp"
#include "quant/simulation/bootstrap.hpp"
#include "quant/risk/metrics.hpp"
#include <omp.h>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace quant::simulation {

namespace {

double calculate_path_max_drawdown(const std::vector<double>& path) {
    if (path.empty()) return 0.0;
    double peak = path[0];
    double max_dd = 0.0;
    for (double val : path) {
        if (val > peak) {
            peak = val;
        } else if (peak > 0.0) {
            double dd = (peak - val) / peak;
            if (dd > max_dd) max_dd = dd;
        }
    }
    return max_dd;
}

double get_percentile(const std::vector<double>& sorted_data, double p) {
    if (sorted_data.empty()) return 0.0;
    if (p <= 0.0) return sorted_data.front();
    if (p >= 1.0) return sorted_data.back();

    double idx = p * static_cast<double>(sorted_data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(idx));
    size_t upper = static_cast<size_t>(std::ceil(idx));
    double frac = idx - static_cast<double>(lower);

    return sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac;
}

} // namespace

MonteCarloReport MonteCarloEngine::run_simulation(const std::vector<double>& historical_returns) const {
    auto start_time = std::chrono::high_resolution_clock::now();

    const size_t N = config_.num_simulations;
    const size_t H = config_.horizon_days;
    const double S0 = config_.initial_wealth;

    std::vector<double> terminal_wealth(N, 0.0);
    std::vector<double> max_drawdowns(N, 0.0);

    // Compute empirical stats for fallback GBM
    double daily_mu = quant::risk::RiskMetrics::mean(historical_returns);
    double daily_sigma = quant::risk::RiskMetrics::standard_deviation(historical_returns);
    double ann_mu = daily_mu * 252.0;
    double ann_sigma = daily_sigma * std::sqrt(252.0);
    double dt = 1.0 / 252.0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(config_.seed + static_cast<uint64_t>(tid) * 100003ULL);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            std::vector<double> path;
            if (config_.use_bootstrap && !historical_returns.empty()) {
                path = BootstrapSimulator::simulate_portfolio_path(historical_returns, S0, H, rng);
            } else {
                path = GbmSimulator::simulate_single_path(S0, ann_mu, ann_sigma, H, dt, rng);
            }
            terminal_wealth[i] = path.back();
            max_drawdowns[i] = calculate_path_max_drawdown(path);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Sort arrays for quantile evaluation
    std::sort(terminal_wealth.begin(), terminal_wealth.end());
    std::sort(max_drawdowns.begin(), max_drawdowns.end());

    MonteCarloReport rep;
    rep.num_simulations = N;
    rep.horizon_days = H;
    rep.initial_wealth = S0;
    rep.elapsed_ms = elapsed_ms;

    rep.mean_terminal_wealth = std::accumulate(terminal_wealth.begin(), terminal_wealth.end(), 0.0) / static_cast<double>(N);
    rep.median_terminal_wealth = get_percentile(terminal_wealth, 0.50);

    double sum_sq_diff = 0.0;
    size_t loss_count = 0;
    size_t loss_10_count = 0;
    size_t loss_20_count = 0;

    for (double w : terminal_wealth) {
        double d = w - rep.mean_terminal_wealth;
        sum_sq_diff += d * d;
        if (w < S0) loss_count++;
        if (w < S0 * 0.90) loss_10_count++;
        if (w < S0 * 0.80) loss_20_count++;
    }
    rep.std_terminal_wealth = std::sqrt(sum_sq_diff / static_cast<double>(N - 1));

    rep.p01_wealth = get_percentile(terminal_wealth, 0.01);
    rep.p05_wealth = get_percentile(terminal_wealth, 0.05);
    rep.p25_wealth = get_percentile(terminal_wealth, 0.25);
    rep.p50_wealth = get_percentile(terminal_wealth, 0.50);
    rep.p75_wealth = get_percentile(terminal_wealth, 0.75);
    rep.p95_wealth = get_percentile(terminal_wealth, 0.95);
    rep.p99_wealth = get_percentile(terminal_wealth, 0.99);

    rep.mean_max_drawdown = std::accumulate(max_drawdowns.begin(), max_drawdowns.end(), 0.0) / static_cast<double>(N);
    rep.p50_max_drawdown = get_percentile(max_drawdowns, 0.50);
    rep.p95_max_drawdown = get_percentile(max_drawdowns, 0.95);
    rep.p99_max_drawdown = get_percentile(max_drawdowns, 0.99);

    rep.prob_loss = static_cast<double>(loss_count) / static_cast<double>(N);
    rep.prob_loss_gt_10pct = static_cast<double>(loss_10_count) / static_cast<double>(N);
    rep.prob_loss_gt_20pct = static_cast<double>(loss_20_count) / static_cast<double>(N);

    // Terminal VaR & CVaR (relative to initial capital)
    rep.var_95_terminal = (S0 - rep.p05_wealth) / S0;

    size_t cutoff_idx = static_cast<size_t>(0.05 * static_cast<double>(N));
    double sum_tail_loss = 0.0;
    for (size_t i = 0; i <= cutoff_idx; ++i) {
        sum_tail_loss += (S0 - terminal_wealth[i]);
    }
    rep.cvar_95_terminal = (sum_tail_loss / static_cast<double>(cutoff_idx + 1)) / S0;

    return rep;
}

MonteCarloReport MonteCarloEngine::run_gbm_portfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    const Eigen::VectorXd& weights
) const {
    // Portfolio expected return: w^T * mu
    double port_ann_mu = weights.dot(expected_returns);
    // Portfolio daily variance: w^T * Sigma * w
    double port_daily_var = weights.dot(cov_matrix * weights);
    double port_ann_sigma = std::sqrt(std::max(0.0, port_daily_var * 252.0));

    auto start_time = std::chrono::high_resolution_clock::now();

    const size_t N = config_.num_simulations;
    const size_t H = config_.horizon_days;
    const double S0 = config_.initial_wealth;
    const double dt = 1.0 / 252.0;

    std::vector<double> terminal_wealth(N, 0.0);
    std::vector<double> max_drawdowns(N, 0.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(config_.seed + static_cast<uint64_t>(tid) * 100003ULL);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            auto path = GbmSimulator::simulate_single_path(S0, port_ann_mu, port_ann_sigma, H, dt, rng);
            terminal_wealth[i] = path.back();
            max_drawdowns[i] = calculate_path_max_drawdown(path);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::sort(terminal_wealth.begin(), terminal_wealth.end());
    std::sort(max_drawdowns.begin(), max_drawdowns.end());

    MonteCarloReport rep;
    rep.num_simulations = N;
    rep.horizon_days = H;
    rep.initial_wealth = S0;
    rep.elapsed_ms = elapsed_ms;

    rep.mean_terminal_wealth = std::accumulate(terminal_wealth.begin(), terminal_wealth.end(), 0.0) / static_cast<double>(N);
    rep.median_terminal_wealth = get_percentile(terminal_wealth, 0.50);

    double sum_sq_diff = 0.0;
    size_t loss_count = 0;
    size_t loss_10_count = 0;
    size_t loss_20_count = 0;

    for (double w : terminal_wealth) {
        double d = w - rep.mean_terminal_wealth;
        sum_sq_diff += d * d;
        if (w < S0) loss_count++;
        if (w < S0 * 0.90) loss_10_count++;
        if (w < S0 * 0.80) loss_20_count++;
    }
    rep.std_terminal_wealth = std::sqrt(sum_sq_diff / static_cast<double>(N - 1));

    rep.p01_wealth = get_percentile(terminal_wealth, 0.01);
    rep.p05_wealth = get_percentile(terminal_wealth, 0.05);
    rep.p25_wealth = get_percentile(terminal_wealth, 0.25);
    rep.p50_wealth = get_percentile(terminal_wealth, 0.50);
    rep.p75_wealth = get_percentile(terminal_wealth, 0.75);
    rep.p95_wealth = get_percentile(terminal_wealth, 0.95);
    rep.p99_wealth = get_percentile(terminal_wealth, 0.99);

    rep.mean_max_drawdown = std::accumulate(max_drawdowns.begin(), max_drawdowns.end(), 0.0) / static_cast<double>(N);
    rep.p50_max_drawdown = get_percentile(max_drawdowns, 0.50);
    rep.p95_max_drawdown = get_percentile(max_drawdowns, 0.95);
    rep.p99_max_drawdown = get_percentile(max_drawdowns, 0.99);

    rep.prob_loss = static_cast<double>(loss_count) / static_cast<double>(N);
    rep.prob_loss_gt_10pct = static_cast<double>(loss_10_count) / static_cast<double>(N);
    rep.prob_loss_gt_20pct = static_cast<double>(loss_20_count) / static_cast<double>(N);

    rep.var_95_terminal = (S0 - rep.p05_wealth) / S0;

    size_t cutoff_idx = static_cast<size_t>(0.05 * static_cast<double>(N));
    double sum_tail_loss = 0.0;
    for (size_t i = 0; i <= cutoff_idx; ++i) {
        sum_tail_loss += (S0 - terminal_wealth[i]);
    }
    rep.cvar_95_terminal = (sum_tail_loss / static_cast<double>(cutoff_idx + 1)) / S0;

    return rep;
}

std::string MonteCarloEngine::generate_text_report(const MonteCarloReport& r) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "=========================================================================\n";
    ss << "                   MONTE CARLO RISK & TAIL REPORT                        \n";
    ss << "=========================================================================\n";
    ss << "Paths Simulated:     " << r.num_simulations << " paths\n";
    ss << "Horizon:             " << r.horizon_days << " trading days (1 Year)\n";
    ss << "Initial Capital:     $" << r.initial_wealth << "\n";
    ss << "Computation Time:    " << std::setprecision(1) << r.elapsed_ms << " ms (OpenMP Parallelized)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " TERMINAL WEALTH DISTRIBUTION (1 Year)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << std::setprecision(2);
    ss << "  Expected (Mean):     $" << r.mean_terminal_wealth << " (" << ((r.mean_terminal_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  Median (50th %ile):  $" << r.median_terminal_wealth << " (" << ((r.median_terminal_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  Std Deviation:       $" << r.std_terminal_wealth << "\n";
    ss << "  99th Percentile:     $" << r.p99_wealth << " (" << ((r.p99_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  95th Percentile:     $" << r.p95_wealth << " (" << ((r.p95_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  75th Percentile:     $" << r.p75_wealth << " (" << ((r.p75_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  25th Percentile:     $" << r.p25_wealth << " (" << ((r.p25_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  5th Percentile:      $" << r.p05_wealth << " (" << ((r.p05_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "  1st Percentile:      $" << r.p01_wealth << " (" << ((r.p01_wealth / r.initial_wealth - 1.0) * 100.0) << " %)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " TAIL LOSS & DRAWDOWN PROBABILITIES\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Probability of Capital Loss:    " << (r.prob_loss * 100.0) << " %\n";
    ss << "  Probability of > 10% Loss:      " << (r.prob_loss_gt_10pct * 100.0) << " %\n";
    ss << "  Probability of > 20% Loss:      " << (r.prob_loss_gt_20pct * 100.0) << " %\n";
    ss << "  Mean Max Drawdown:              " << (r.mean_max_drawdown * 100.0) << " %\n";
    ss << "  95th %ile Max Drawdown:         " << (r.p95_max_drawdown * 100.0) << " %\n";
    ss << "  99th %ile Max Drawdown:         " << (r.p99_max_drawdown * 100.0) << " %\n";
    ss << "  Terminal 95% 1-Year VaR:        " << (r.var_95_terminal * 100.0) << " %\n";
    ss << "  Terminal 95% 1-Year CVaR:       " << (r.cvar_95_terminal * 100.0) << " %\n";
    ss << "=========================================================================\n";

    return ss.str();
}

} // namespace quant::simulation
