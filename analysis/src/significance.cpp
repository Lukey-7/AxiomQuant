#include "quant/analysis/significance.hpp"
#include "quant/risk/metrics.hpp"
#include "quant/risk/var_cvar.hpp"
#include "quant/simulation/rng.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace quant::analysis {

namespace {

constexpr double kEulerGamma = 0.5772156649015329;

// Annualised Sharpe of the resampled series r[idx[0]], r[idx[1]], ...; same convention as
// RiskMetrics::sharpe_ratio (sample standard deviation, 0 when volatility vanishes).
double resampled_sharpe(const std::vector<double>& r,
                        const std::vector<size_t>& idx,
                        double daily_rf,
                        double ann_factor) {
    const double n = static_cast<double>(idx.size());
    double sum = 0.0;
    for (size_t i : idx)
        sum += r[i];
    const double mean = sum / n;
    double ss = 0.0;
    for (size_t i : idx) {
        const double d = r[i] - mean;
        ss += d * d;
    }
    const double sd = std::sqrt(ss / (n - 1.0));
    if (sd * std::sqrt(ann_factor) <= 1e-9) return 0.0;
    return (mean - daily_rf) / sd * std::sqrt(ann_factor);
}

// Linear interpolation between order statistics (NumPy's default percentile method).
double percentile(const std::vector<double>& sorted, double q) {
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

SharpeInterval summarise(std::vector<double> draws, double estimate, double confidence, bool two_sided) {
    SharpeInterval out;
    out.estimate = estimate;
    size_t extreme = 0;
    for (double d : draws) {
        const double shifted = d - estimate;
        if (two_sided ? std::abs(shifted) >= std::abs(estimate) : shifted >= estimate) ++extreme;
    }
    out.p_value = static_cast<double>(1 + extreme) / static_cast<double>(draws.size() + 1);

    std::sort(draws.begin(), draws.end());
    const double alpha = 1.0 - confidence;
    out.lower = percentile(draws, alpha / 2.0);
    out.upper = percentile(draws, 1.0 - alpha / 2.0);
    return out;
}

}   // namespace

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

double probabilistic_sharpe_ratio(const std::vector<double>& returns,
                                  double benchmark_sharpe,
                                  double risk_free_rate,
                                  double ann_factor) {
    if (returns.size() < 3) return 0.5;
    const double sqrt_ann = std::sqrt(ann_factor);
    const double sr = risk::RiskMetrics::sharpe_ratio(returns, risk_free_rate, ann_factor) / sqrt_ann;
    const double sr_star = benchmark_sharpe / sqrt_ann;
    const double g3 = risk::RiskMetrics::skewness(returns);
    const double g4 = risk::RiskMetrics::excess_kurtosis(returns) + 3.0;

    const double variance_term = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr;
    if (!(variance_term > 0.0)) return sr > sr_star ? 1.0 : 0.0;
    const double z =
        (sr - sr_star) * std::sqrt(static_cast<double>(returns.size()) - 1.0) / std::sqrt(variance_term);
    return normal_cdf(z);
}

double politis_white_block_length(const std::vector<double>& series) {
    const size_t n = series.size();
    if (n < 16) return std::numeric_limits<double>::quiet_NaN();
    const double dn = static_cast<double>(n);

    // A constant series has no dependence structure to estimate; rounding noise in the mean would
    // otherwise leave a tiny non-zero variance and produce a meaningless block length.
    const auto [low, high] = std::minmax_element(series.begin(), series.end());
    if (*high - *low <= 1e-12 * std::max(1.0, std::abs(*high))) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double mean = risk::RiskMetrics::mean(series);
    std::vector<double> centred(n);
    for (size_t i = 0; i < n; ++i)
        centred[i] = series[i] - mean;

    // Autocovariances up to the largest lag the rules below can ask for.
    const auto k_max = static_cast<size_t>(std::max(5.0, std::sqrt(std::log10(dn))));
    const size_t m_max = static_cast<size_t>(std::ceil(std::sqrt(dn))) + k_max;
    const size_t max_lag = std::min(n - 1, 2 * m_max);
    std::vector<double> autocov(max_lag + 1, 0.0);
    for (size_t lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (size_t i = lag; i < n; ++i)
            sum += centred[i] * centred[i - lag];
        autocov[lag] = sum / dn;
    }
    if (!(autocov[0] > 0.0)) return std::numeric_limits<double>::quiet_NaN();

    // Smallest m past which k_max consecutive autocorrelations are insignificant.
    const double threshold = 2.0 * std::sqrt(std::log10(dn) / dn);
    size_t m = 0;
    for (size_t candidate = 1; candidate + k_max <= max_lag; ++candidate) {
        bool all_small = true;
        for (size_t k = 1; k <= k_max && all_small; ++k) {
            all_small = std::abs(autocov[candidate + k] / autocov[0]) < threshold;
        }
        if (all_small) {
            m = candidate;
            break;
        }
    }
    if (m == 0) m = static_cast<size_t>(std::ceil(std::sqrt(dn)));
    const size_t bandwidth = std::max<size_t>(1, std::min(2 * m, std::min(m_max, max_lag)));

    // Flat-top lag window (Politis & Romano, 1995): 1 up to half the bandwidth, then tapering to 0.
    const auto flat_top = [](double x) {
        const double a = std::abs(x);
        if (a <= 0.5) return 1.0;
        if (a <= 1.0) return 2.0 * (1.0 - a);
        return 0.0;
    };

    double g = 0.0, spectrum = autocov[0];
    for (size_t lag = 1; lag <= bandwidth; ++lag) {
        const double weight = flat_top(static_cast<double>(lag) / static_cast<double>(bandwidth));
        g += 2.0 * weight * static_cast<double>(lag) * autocov[lag];   // both signs of the lag
        spectrum += 2.0 * weight * autocov[lag];
    }
    const double d = 2.0 * spectrum * spectrum;   // stationary bootstrap constant
    if (!(d > 0.0) || !std::isfinite(g)) return std::numeric_limits<double>::quiet_NaN();

    const double block = std::cbrt(2.0 * g * g / d) * std::cbrt(dn);
    if (!std::isfinite(block)) return std::numeric_limits<double>::quiet_NaN();
    const double cap = std::min(3.0 * std::sqrt(dn), dn / 3.0);
    return std::clamp(block, 1.0, std::max(1.0, cap));
}

double effective_trials(const std::vector<std::vector<double>>& trial_returns) {
    const size_t trials = trial_returns.size();
    if (trials < 2) return static_cast<double>(std::max<size_t>(1, trials));

    double sum_correlation = 0.0;
    size_t pairs = 0;
    for (size_t i = 0; i < trials; ++i) {
        for (size_t j = i + 1; j < trials; ++j) {
            const auto& a = trial_returns[i];
            const auto& b = trial_returns[j];
            const size_t n = std::min(a.size(), b.size());
            if (n < 3) continue;
            const double mean_a =
                std::accumulate(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n), 0.0) /
                static_cast<double>(n);
            const double mean_b =
                std::accumulate(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(n), 0.0) /
                static_cast<double>(n);
            double cov = 0.0, var_a = 0.0, var_b = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double da = a[k] - mean_a;
                const double db = b[k] - mean_b;
                cov += da * db;
                var_a += da * da;
                var_b += db * db;
            }
            if (!(var_a > 0.0) || !(var_b > 0.0)) continue;
            sum_correlation += cov / std::sqrt(var_a * var_b);
            ++pairs;
        }
    }
    if (pairs == 0) return static_cast<double>(trials);

    // Negative average correlation would inflate the count past N; independence is the ceiling.
    const double mean_correlation = std::clamp(sum_correlation / static_cast<double>(pairs), 0.0, 1.0);
    return std::max(1.0, 1.0 + (static_cast<double>(trials) - 1.0) * (1.0 - mean_correlation));
}

double expected_maximum_sharpe(double trials, double trial_sharpe_std) {
    if (!(trials > 1.0)) return 0.0;
    const double n = trials;
    const double z1 = risk::ValueAtRisk::standard_normal_quantile(1.0 - 1.0 / n);
    const double z2 = risk::ValueAtRisk::standard_normal_quantile(1.0 - 1.0 / (n * std::numbers::e));
    return trial_sharpe_std * ((1.0 - kEulerGamma) * z1 + kEulerGamma * z2);
}

DeflatedSharpe deflated_sharpe_ratio(const std::vector<double>& selected_returns,
                                     const std::vector<double>& trial_sharpes,
                                     double risk_free_rate,
                                     double ann_factor,
                                     double effective_trial_count) {
    DeflatedSharpe out;
    out.observations = selected_returns.size();
    out.trials = trial_sharpes.size();
    out.sharpe = risk::RiskMetrics::sharpe_ratio(selected_returns, risk_free_rate, ann_factor);
    out.trial_sharpe_std =
        trial_sharpes.size() > 1 ? risk::RiskMetrics::standard_deviation(trial_sharpes) : 0.0;
    out.effective_trials = effective_trial_count > 0.0
                               ? std::min(effective_trial_count, static_cast<double>(out.trials))
                               : static_cast<double>(out.trials);
    out.expected_max_sharpe = expected_maximum_sharpe(out.effective_trials, out.trial_sharpe_std);
    out.probabilistic_sharpe = probabilistic_sharpe_ratio(selected_returns, 0.0, risk_free_rate, ann_factor);
    out.deflated_sharpe =
        probabilistic_sharpe_ratio(selected_returns, out.expected_max_sharpe, risk_free_rate, ann_factor);
    return out;
}

BootstrapSharpeResult bootstrap_sharpe(const std::vector<double>& strategy,
                                       const std::vector<double>& benchmark,
                                       const BootstrapConfig& config) {
    const size_t n = strategy.size();
    if (n < 3) throw std::invalid_argument("bootstrap_sharpe: need at least three returns");
    if (!benchmark.empty() && benchmark.size() != n) {
        throw std::invalid_argument("bootstrap_sharpe: benchmark must be empty or match the strategy length");
    }
    if (config.resamples == 0) throw std::invalid_argument("bootstrap_sharpe: resamples must be positive");
    if (!(config.confidence > 0.0 && config.confidence < 1.0)) {
        throw std::invalid_argument("bootstrap_sharpe: confidence must be in (0, 1)");
    }

    BootstrapSharpeResult out;
    out.observations = n;
    out.resamples = config.resamples;
    out.confidence = config.confidence;
    out.has_benchmark = !benchmark.empty();
    // A block length of 0 means "let the data choose"; fall back to T^(1/3) if that fails.
    double block_length = config.mean_block_length;
    if (!(block_length > 0.0)) {
        block_length = politis_white_block_length(strategy);
        if (!std::isfinite(block_length)) block_length = std::cbrt(static_cast<double>(n));
    }
    out.mean_block_length = block_length;
    out.mean_block_length = std::clamp(out.mean_block_length, 1.0, static_cast<double>(n));

    const double daily_rf = config.risk_free_rate / config.ann_factor;
    const double restart_probability = 1.0 / out.mean_block_length;
    const double strategy_sharpe =
        risk::RiskMetrics::sharpe_ratio(strategy, config.risk_free_rate, config.ann_factor);
    const double benchmark_sharpe =
        out.has_benchmark
            ? risk::RiskMetrics::sharpe_ratio(benchmark, config.risk_free_rate, config.ann_factor)
            : 0.0;

    std::vector<double> strategy_draws(config.resamples);
    std::vector<double> benchmark_draws(out.has_benchmark ? config.resamples : 0);

    // Each resample owns its generator and index buffer, so iterations share no mutable state.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < static_cast<int64_t>(config.resamples); ++b) {
        simulation::PathRng rng(config.seed, static_cast<uint64_t>(b));
        std::vector<size_t> idx(n);
        size_t pos = rng.index(n);
        for (size_t t = 0; t < n; ++t) {
            if (t > 0) {
                pos = rng.uniform() < restart_probability ? rng.index(n) : (pos + 1) % n;
            }
            idx[t] = pos;
        }
        const auto slot = static_cast<size_t>(b);
        strategy_draws[slot] = resampled_sharpe(strategy, idx, daily_rf, config.ann_factor);
        if (out.has_benchmark) {
            benchmark_draws[slot] = resampled_sharpe(benchmark, idx, daily_rf, config.ann_factor);
        }
    }

    out.strategy = summarise(strategy_draws, strategy_sharpe, config.confidence, false);
    if (out.has_benchmark) {
        std::vector<double> difference_draws(config.resamples);
        for (size_t b = 0; b < config.resamples; ++b) {
            difference_draws[b] = strategy_draws[b] - benchmark_draws[b];
        }
        out.benchmark = summarise(std::move(benchmark_draws), benchmark_sharpe, config.confidence, false);
        out.difference = summarise(std::move(difference_draws), strategy_sharpe - benchmark_sharpe,
                                   config.confidence, true);
    }
    out.probabilistic_sharpe =
        probabilistic_sharpe_ratio(strategy, 0.0, config.risk_free_rate, config.ann_factor);
    return out;
}

std::string format_bootstrap_report(const BootstrapSharpeResult& r) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    const int level = static_cast<int>(std::lround(r.confidence * 100.0));
    ss << "  Stationary block bootstrap of the stitched out-of-sample returns\n";
    ss << "  " << r.observations << " days, " << r.resamples << " resamples, mean block length "
       << std::setprecision(1) << r.mean_block_length << " days\n";
    ss << std::setprecision(3);
    ss << "                        Sharpe      " << level << "% interval        p-value\n";
    const auto row = [&ss](const char* label, const SharpeInterval& s, const char* hypothesis) {
        ss << "  " << std::left << std::setw(20) << label << std::right << std::setw(8) << s.estimate
           << "   [" << std::setw(7) << s.lower << ", " << std::setw(7) << s.upper << "]" << std::setw(10)
           << s.p_value << "  " << hypothesis << "\n";
    };
    row("SMA (walk-forward)", r.strategy, "(H0: Sharpe <= 0)");
    if (r.has_benchmark) {
        row("Buy & hold", r.benchmark, "(H0: Sharpe <= 0)");
        row("Difference", r.difference, "(H0: no difference)");
    }
    ss << "  Probabilistic Sharpe ratio (P[true Sharpe > 0]): " << r.probabilistic_sharpe << "\n";
    return ss.str();
}

std::string format_deflated_sharpe_report(const DeflatedSharpe& r) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "  Deflated Sharpe ratio of the top pair (" << r.trials << " configurations tried, "
       << std::setprecision(1) << r.effective_trials << " effectively independent, " << r.observations
       << " days)\n"
       << std::setprecision(3);
    ss << "    Sharpe " << r.sharpe << "   spread across the grid (std) " << r.trial_sharpe_std
       << "   best expected from luck alone " << r.expected_max_sharpe << "\n";
    ss << "    P[true Sharpe > 0] = " << r.probabilistic_sharpe
       << "   P[true Sharpe > luck-adjusted hurdle] = " << r.deflated_sharpe << "\n";
    return ss.str();
}

}   // namespace quant::analysis
