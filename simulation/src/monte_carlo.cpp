#include "quant/simulation/monte_carlo.hpp"
#include "quant/risk/metrics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

namespace quant::simulation {

namespace {

// SplitMix64 finalizer: decorrelates (seed, stream) pairs into independent generator states.
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/**
 * xoshiro256** (Blackman & Vigna) with a portable normal sampler. One instance per path keeps the
 * simulation reproducible independent of how paths are distributed over threads.
 */
class PathRng {
public:
    PathRng(uint64_t seed, uint64_t stream) {
        uint64_t x = splitmix64(seed) ^ splitmix64(stream + 0x632BE59BD9B4E019ULL);
        for (auto& s : s_) {
            x = splitmix64(x);
            s = x;
        }
    }

    uint64_t next() noexcept {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform double in [0, 1) with 53 bits of randomness.
    double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    size_t index(size_t n) noexcept {
        const size_t i = static_cast<size_t>(uniform() * static_cast<double>(n));
        return i < n ? i : n - 1;
    }

    // Marsaglia polar method: exact N(0,1), identical output on every platform.
    double normal() noexcept {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u, v, s;
        do {
            u = 2.0 * uniform() - 1.0;
            v = 2.0 * uniform() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double m = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = v * m;
        has_spare_ = true;
        return u * m;
    }

private:
    static uint64_t rotl(uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4]{};
    double spare_{0.0};
    bool has_spare_{false};
};

inline void track_drawdown(double wealth, double& peak, double& max_dd) noexcept {
    if (wealth > peak) {
        peak = wealth;
    } else if (peak > 0.0) {
        const double dd = (peak - wealth) / peak;
        if (dd > max_dd) max_dd = dd;
    }
}

int resolve_threads(size_t requested) {
#ifdef _OPENMP
    return requested > 0 ? static_cast<int>(requested) : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
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

MonteCarloReport summarize(
    std::vector<double>& terminal_wealth,
    std::vector<double>& max_drawdowns,
    const MonteCarloConfig& cfg,
    double elapsed_ms,
    int threads,
    std::string method
) {
    const size_t N = terminal_wealth.size();
    const double S0 = cfg.initial_wealth;

    std::sort(terminal_wealth.begin(), terminal_wealth.end());
    std::sort(max_drawdowns.begin(), max_drawdowns.end());

    MonteCarloReport rep;
    rep.method = std::move(method);
    rep.num_simulations = N;
    rep.horizon_days = cfg.horizon_days;
    rep.initial_wealth = S0;
    rep.threads_used = threads;
    rep.elapsed_ms = elapsed_ms;
    rep.paths_per_second = elapsed_ms > 0.0 ? static_cast<double>(N) / (elapsed_ms / 1000.0) : 0.0;

    rep.mean_terminal_wealth = std::accumulate(terminal_wealth.begin(), terminal_wealth.end(), 0.0) / static_cast<double>(N);
    rep.median_terminal_wealth = get_percentile(terminal_wealth, 0.50);

    double sum_sq_diff = 0.0;
    size_t loss_count = 0, loss_10_count = 0, loss_20_count = 0;
    for (double w : terminal_wealth) {
        const double d = w - rep.mean_terminal_wealth;
        sum_sq_diff += d * d;
        if (w < S0) ++loss_count;
        if (w < S0 * 0.90) ++loss_10_count;
        if (w < S0 * 0.80) ++loss_20_count;
    }
    rep.std_terminal_wealth = N > 1 ? std::sqrt(sum_sq_diff / static_cast<double>(N - 1)) : 0.0;

    rep.p01_wealth = get_percentile(terminal_wealth, 0.01);
    rep.p05_wealth = get_percentile(terminal_wealth, 0.05);
    rep.p25_wealth = get_percentile(terminal_wealth, 0.25);
    rep.p50_wealth = rep.median_terminal_wealth;
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

    // Terminal VaR / CVaR (Expected Shortfall) as fractions of initial capital.
    rep.var_95_terminal = (S0 - rep.p05_wealth) / S0;
    const size_t tail_count = std::max<size_t>(1, static_cast<size_t>(std::ceil(0.05 * static_cast<double>(N))));
    double sum_tail_loss = 0.0;
    for (size_t i = 0; i < tail_count; ++i) {
        sum_tail_loss += (S0 - terminal_wealth[i]);
    }
    rep.cvar_95_terminal = (sum_tail_loss / static_cast<double>(tail_count)) / S0;

    return rep;
}

} // namespace

MonteCarloReport MonteCarloEngine::run_simulation(const std::vector<double>& historical_returns) const {
    const size_t N = config_.num_simulations;
    const size_t H = config_.horizon_days;
    const double S0 = config_.initial_wealth;
    if (N == 0) throw std::invalid_argument("MonteCarloConfig::num_simulations must be > 0");

    const bool bootstrap = config_.use_bootstrap && !historical_returns.empty();
    const size_t n_hist = historical_returns.size();

    // GBM calibration: E[S_T] = S_0 * exp(mu * T) with mu the annualized arithmetic mean return.
    const double dt = 1.0 / 252.0;
    const double ann_mu = quant::risk::RiskMetrics::mean(historical_returns) * 252.0;
    const double ann_sigma = quant::risk::RiskMetrics::standard_deviation(historical_returns) * std::sqrt(252.0);
    const double drift = (ann_mu - 0.5 * ann_sigma * ann_sigma) * dt;
    const double vol = ann_sigma * std::sqrt(dt);

    std::vector<double> terminal_wealth(N, 0.0);
    std::vector<double> max_drawdowns(N, 0.0);
    const int threads = resolve_threads(config_.num_threads);
    const uint64_t seed = config_.seed;

    const auto start_time = std::chrono::steady_clock::now();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
        PathRng rng(seed, static_cast<uint64_t>(i));
        double wealth = S0, peak = S0, max_dd = 0.0;
        if (bootstrap) {
            for (size_t t = 0; t < H; ++t) {
                wealth *= 1.0 + historical_returns[rng.index(n_hist)];
                if (wealth < 0.0) wealth = 0.0;
                track_drawdown(wealth, peak, max_dd);
            }
        } else {
            for (size_t t = 0; t < H; ++t) {
                wealth *= std::exp(drift + vol * rng.normal());
                track_drawdown(wealth, peak, max_dd);
            }
        }
        terminal_wealth[static_cast<size_t>(i)] = wealth;
        max_drawdowns[static_cast<size_t>(i)] = max_dd;
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    return summarize(terminal_wealth, max_drawdowns, config_, elapsed_ms, threads,
                     bootstrap ? "Empirical bootstrap (i.i.d. resampling)" : "Fitted geometric Brownian motion");
}

MonteCarloReport MonteCarloEngine::run_gbm_portfolio(
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& cov_matrix,
    const Eigen::VectorXd& weights
) const {
    const Eigen::Index n = weights.size();
    if (n == 0 || expected_returns.size() != n || cov_matrix.rows() != n || cov_matrix.cols() != n) {
        throw std::invalid_argument("run_gbm_portfolio: dimension mismatch");
    }
    const size_t N = config_.num_simulations;
    const size_t H = config_.horizon_days;
    const double S0 = config_.initial_wealth;
    if (N == 0) throw std::invalid_argument("MonteCarloConfig::num_simulations must be > 0");

    // Cholesky factor of the daily covariance (with a tiny ridge if it is only semi-definite).
    Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix);
    if (llt.info() != Eigen::Success) {
        const double ridge = 1e-12 * std::max(1e-300, cov_matrix.trace() / static_cast<double>(n));
        llt.compute(cov_matrix + ridge * Eigen::MatrixXd::Identity(n, n));
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("run_gbm_portfolio: covariance matrix is not positive semi-definite");
        }
    }
    const Eigen::MatrixXd L = llt.matrixL();

    // Flatten the lower triangle for a branch-free inner loop, and precompute Ito-corrected drifts.
    const size_t un = static_cast<size_t>(n);
    std::vector<double> L_flat(un * un, 0.0), drift(un), w(un);
    for (size_t i = 0; i < un; ++i) {
        for (size_t j = 0; j <= i; ++j) L_flat[i * un + j] = L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        const auto ii = static_cast<Eigen::Index>(i);
        drift[i] = expected_returns(ii) / 252.0 - 0.5 * cov_matrix(ii, ii);
        w[i] = weights(ii);
    }

    std::vector<double> terminal_wealth(N, 0.0);
    std::vector<double> max_drawdowns(N, 0.0);
    const int threads = resolve_threads(config_.num_threads);
    const uint64_t seed = config_.seed;
    const bool rebalance = config_.rebalance_daily;

    const auto start_time = std::chrono::steady_clock::now();

#ifdef _OPENMP
    #pragma omp parallel num_threads(threads)
#endif
    {
        std::vector<double> z(un), rel_price(un);

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int64_t p = 0; p < static_cast<int64_t>(N); ++p) {
            PathRng rng(seed, static_cast<uint64_t>(p));
            std::fill(rel_price.begin(), rel_price.end(), 1.0);
            double wealth = S0, peak = S0, max_dd = 0.0;

            for (size_t t = 0; t < H; ++t) {
                for (size_t i = 0; i < un; ++i) z[i] = rng.normal();

                double port_growth = 0.0;   // constant-mix: sum_i w_i * gross_i
                double port_value = 0.0;    // buy-and-hold: sum_i w_i * P_i(t) / P_i(0)
                for (size_t i = 0; i < un; ++i) {
                    double shock = 0.0;
                    const double* Li = &L_flat[i * un];
                    for (size_t j = 0; j <= i; ++j) shock += Li[j] * z[j];
                    const double gross = std::exp(drift[i] + shock);
                    if (rebalance) {
                        port_growth += w[i] * gross;
                    } else {
                        rel_price[i] *= gross;
                        port_value += w[i] * rel_price[i];
                    }
                }

                wealth = rebalance ? wealth * port_growth : S0 * port_value;
                if (wealth < 0.0) wealth = 0.0;
                track_drawdown(wealth, peak, max_dd);
            }
            terminal_wealth[static_cast<size_t>(p)] = wealth;
            max_drawdowns[static_cast<size_t>(p)] = max_dd;
        }
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    return summarize(terminal_wealth, max_drawdowns, config_, elapsed_ms, threads,
                     std::string("Correlated multi-asset GBM (Cholesky, ") +
                     (rebalance ? "daily rebalanced)" : "buy-and-hold)"));
}

std::string MonteCarloEngine::generate_text_report(const MonteCarloReport& r) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    const double S0 = r.initial_wealth;
    const double years = static_cast<double>(r.horizon_days) / 252.0;
    auto pct = [S0](double w) { return (w / S0 - 1.0) * 100.0; };

    ss << "=========================================================================\n";
    ss << "                   MONTE CARLO RISK & TAIL REPORT                        \n";
    ss << "=========================================================================\n";
    ss << "Model:               " << r.method << "\n";
    ss << "Paths Simulated:     " << r.num_simulations << " paths\n";
    ss << "Horizon:             " << r.horizon_days << " trading days (" << years << " yrs)\n";
    ss << "Initial Capital:     $" << S0 << "\n";
    ss << "Computation Time:    " << std::setprecision(1) << r.elapsed_ms << " ms on " << r.threads_used
       << (r.threads_used == 1 ? " thread" : " threads")
       << "  (" << std::setprecision(0) << r.paths_per_second << " paths/s)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " TERMINAL WEALTH DISTRIBUTION\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << std::setprecision(2);
    ss << "  Expected (Mean):     $" << r.mean_terminal_wealth << " (" << pct(r.mean_terminal_wealth) << " %)\n";
    ss << "  Median (50th %ile):  $" << r.median_terminal_wealth << " (" << pct(r.median_terminal_wealth) << " %)\n";
    ss << "  Std Deviation:       $" << r.std_terminal_wealth << "\n";
    ss << "  99th Percentile:     $" << r.p99_wealth << " (" << pct(r.p99_wealth) << " %)\n";
    ss << "  95th Percentile:     $" << r.p95_wealth << " (" << pct(r.p95_wealth) << " %)\n";
    ss << "  75th Percentile:     $" << r.p75_wealth << " (" << pct(r.p75_wealth) << " %)\n";
    ss << "  25th Percentile:     $" << r.p25_wealth << " (" << pct(r.p25_wealth) << " %)\n";
    ss << "  5th Percentile:      $" << r.p05_wealth << " (" << pct(r.p05_wealth) << " %)\n";
    ss << "  1st Percentile:      $" << r.p01_wealth << " (" << pct(r.p01_wealth) << " %)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " TAIL LOSS & DRAWDOWN PROBABILITIES\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Probability of Capital Loss:    " << (r.prob_loss * 100.0) << " %\n";
    ss << "  Probability of > 10% Loss:      " << (r.prob_loss_gt_10pct * 100.0) << " %\n";
    ss << "  Probability of > 20% Loss:      " << (r.prob_loss_gt_20pct * 100.0) << " %\n";
    ss << "  Median Max Drawdown:            " << (r.p50_max_drawdown * 100.0) << " %\n";
    ss << "  95th %ile Max Drawdown:         " << (r.p95_max_drawdown * 100.0) << " %\n";
    ss << "  99th %ile Max Drawdown:         " << (r.p99_max_drawdown * 100.0) << " %\n";
    ss << "  Terminal 95% VaR:               " << (r.var_95_terminal * 100.0) << " %\n";
    ss << "  Terminal 95% CVaR (ES):         " << (r.cvar_95_terminal * 100.0) << " %\n";
    ss << "=========================================================================\n";

    return ss.str();
}

} // namespace quant::simulation
