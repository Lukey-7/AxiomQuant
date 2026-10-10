#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace quant::analysis {

/**
 * @brief Standard normal cumulative distribution function, Phi(x).
 */
[[nodiscard]] double normal_cdf(double x);

/**
 * @brief Probabilistic Sharpe ratio (Bailey & Lopez de Prado, 2012).
 *
 * The probability that the true Sharpe ratio exceeds `benchmark_sharpe`, given the observed daily
 * returns, correcting the standard error for skewness and kurtosis:
 *
 *     PSR = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) )
 *
 * with SR and SR* per period (the annualised inputs are divided by sqrt(ann_factor)), g3 the sample
 * skewness and g4 the sample (non-excess) kurtosis.
 *
 * @param benchmark_sharpe Annualised Sharpe ratio to test against (0 = "is it positive at all?").
 * @param risk_free_rate   Annualised risk-free rate, as in RiskMetrics::sharpe_ratio.
 * @return A probability in [0, 1]; 0.5 when there are fewer than three observations.
 */
[[nodiscard]] double probabilistic_sharpe_ratio(const std::vector<double>& returns,
                                                double benchmark_sharpe,
                                                double risk_free_rate,
                                                double ann_factor = 252.0);

/**
 * @brief Optimal stationary-bootstrap mean block length (Politis & White, 2004; Patton et al., 2009).
 *
 * Chooses the block length that minimises the asymptotic mean squared error of the bootstrap
 * variance estimate, from a flat-top lag-window estimate of the series' spectrum at zero. Long
 * blocks preserve dependence but leave fewer effectively independent blocks; this picks the balance
 * the data implies instead of a rule of thumb.
 *
 * @return A block length in [1, n], or NaN when the series is too short or degenerate.
 */
[[nodiscard]] double politis_white_block_length(const std::vector<double>& series);

/**
 * @brief Effective number of independent trials behind a set of correlated trial return series.
 *
 *     N_eff = 1 + (N - 1) (1 - rho_bar)
 *
 * with rho_bar the mean pairwise correlation. Counting 41 variations of "mostly long the index" as
 * 41 independent attempts overstates the luck hurdle in the deflated Sharpe ratio; counting them as
 * one understates it. This interpolates between those extremes.
 *
 * @return At least 1.0; the trial count itself when the series are uncorrelated.
 */
[[nodiscard]] double effective_trials(const std::vector<std::vector<double>>& trial_returns);

/**
 * @brief Expected maximum of `trials` independent Sharpe estimates whose true value is zero.
 *
 *     E[max SR] ~ sigma * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))
 *
 * where gamma is the Euler-Mascheroni constant and sigma the standard deviation of the Sharpe
 * estimates across trials. The result is in the same units as `trial_sharpe_std`. Returns 0 for a
 * single trial.
 */
[[nodiscard]] double expected_maximum_sharpe(double trials, double trial_sharpe_std);

struct DeflatedSharpe {
    size_t observations{0};
    size_t trials{0};
    double effective_trials{0.0};       // trials, discounted for how alike they are
    double sharpe{0.0};                 // annualised, of the selected configuration
    double trial_sharpe_std{0.0};       // annualised, across every configuration tried
    double expected_max_sharpe{0.0};    // annualised, best Sharpe expected from luck alone
    double probabilistic_sharpe{0.5};   // PSR against zero
    double deflated_sharpe{0.5};        // PSR against expected_max_sharpe
};

/**
 * @brief Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).
 *
 * The probabilistic Sharpe ratio of the selected configuration, measured against the Sharpe ratio
 * the best of `trial_sharpes.size()` worthless configurations would be expected to reach. Use it on
 * the winner of an in-sample parameter search: a value near 1 means the winner is unlikely to be a
 * product of the search alone.
 *
 * @param selected_returns Daily returns of the configuration that was picked.
 * @param trial_sharpes    Annualised Sharpe ratio of every configuration tried, including the winner.
 * @param effective_trial_count Trials to deflate against, e.g. from effective_trials(); 0 treats
 *                              every trial as an independent attempt.
 */
[[nodiscard]] DeflatedSharpe deflated_sharpe_ratio(const std::vector<double>& selected_returns,
                                                   const std::vector<double>& trial_sharpes,
                                                   double risk_free_rate,
                                                   double ann_factor = 252.0,
                                                   double effective_trial_count = 0.0);

struct BootstrapConfig {
    size_t resamples{10000};
    double mean_block_length{0.0};   // mean block length; 0 = chosen from the data (Politis-White)
    double confidence{0.95};         // two-sided confidence level of the intervals
    uint64_t seed{42};
    double risk_free_rate{0.02};
    double ann_factor{252.0};
};

struct SharpeInterval {
    double estimate{0.0};   // annualised Sharpe on the original sample
    double lower{0.0};      // percentile bootstrap interval
    double upper{0.0};
    double p_value{1.0};   // see bootstrap_sharpe for the hypothesis
};

struct BootstrapSharpeResult {
    size_t observations{0};
    size_t resamples{0};
    double mean_block_length{0.0};
    double confidence{0.0};
    bool has_benchmark{false};
    SharpeInterval strategy;            // H0: Sharpe <= 0 (one-sided)
    SharpeInterval benchmark;           // H0: Sharpe <= 0 (one-sided)
    SharpeInterval difference;          // strategy minus benchmark; H0: difference = 0 (two-sided)
    double probabilistic_sharpe{0.5};   // PSR of the strategy against zero
};

/**
 * @brief Stationary block bootstrap (Politis & Romano, 1994) of the annualised Sharpe ratio.
 *
 * Each resample is built from blocks whose lengths are geometric with mean `mean_block_length`,
 * wrapping around the end of the sample, so short-range autocorrelation and volatility clustering
 * survive resampling. When a benchmark is given, both series are resampled with the same indices,
 * which keeps their dependence and makes the difference test paired.
 *
 * p-values use the bootstrap distribution shifted to the null (Hall & Wilson, 1991):
 * p = (1 + #{SR*_b - SR >= SR}) / (B + 1) for the one-sided tests, and
 * p = (1 + #{|D*_b - D| >= |D|}) / (B + 1) for the difference.
 *
 * Every resample draws from its own generator seeded from (seed, resample index), so the result is
 * identical at any thread count.
 *
 * @param benchmark Same length as `strategy`, or empty to skip the comparison.
 */
[[nodiscard]] BootstrapSharpeResult bootstrap_sharpe(const std::vector<double>& strategy,
                                                     const std::vector<double>& benchmark,
                                                     const BootstrapConfig& config = BootstrapConfig{});

struct SpaConfig {
    size_t resamples{10000};
    double mean_block_length{0.0};   // mean block length; 0 = chosen from the data (Politis-White)
    uint64_t seed{42};
};

struct SpaResult {
    size_t observations{0};
    size_t strategies{0};
    size_t resamples{0};
    double mean_block_length{0.0};
    std::vector<double> mean_excess;       // per strategy: mean daily return minus the benchmark's
    std::vector<double> excess_std;        // per strategy: omega_k, long-run std of the daily excess
    std::vector<double> t_statistics;      // per strategy: sqrt(T) mean_excess / excess_std
    size_t best{0};                        // strategy with the largest t-statistic
    double reality_check_statistic{0.0};   // max_k sqrt(T) mean_excess_k (White, 2000)
    double reality_check_p_value{1.0};
    double spa_statistic{0.0};      // max(0, max_k t_k) (Hansen, 2005)
    double spa_p_lower{1.0};        // liberal: every under-performer recentred at its own mean
    double spa_p_consistent{1.0};   // Hansen's recommended p-value
    double spa_p_upper{1.0};        // conservative: every strategy recentred at zero
};

/**
 * @brief White's (2000) Reality Check and Hansen's (2005) test for Superior Predictive Ability.
 *
 * Asks whether the best of several strategies beats a benchmark once the fact that all of them were
 * tried is taken into account. The null hypothesis is that no strategy has a higher expected daily
 * return than the benchmark: E[r_k - r_bench] <= 0 for every k.
 *
 * The T x K matrix of daily excess returns d_k = r_k - r_bench is resampled jointly with one
 * stationary bootstrap (Politis & Romano, 1994): every strategy uses the same indices in a given
 * resample, so the cross-strategy dependence that drives the best-of-K maximum survives.
 *
 * - Reality Check: V = max_k sqrt(T) dbar_k against the bootstrap distribution of
 *   max_k sqrt(T) (dbar*_k - dbar_k).
 * - SPA: the studentized statistic max(0, max_k sqrt(T) dbar_k / omega_k), with omega_k^2 Hansen's
 *   stationary-bootstrap kernel estimate of the long-run variance. In the bootstrap, strategies
 *   whose t-statistic is below -sqrt(2 log log T) stay centred at their own (negative) mean instead
 *   of zero, so clearly bad strategies cannot dilute the test; this is the consistent p-value. The
 *   lower p-value keeps every under-performer at its own mean, the upper one recentres every
 *   strategy at zero (the least-favourable configuration the Reality Check also uses).
 *
 * p-values are (1 + #{bootstrap statistic >= observed}) / (B + 1), as in bootstrap_sharpe. Every
 * resample draws from its own generator seeded from (seed, resample index), so the result is
 * identical at any thread count. Strategies whose excess return has zero variance (for instance a
 * copy of the benchmark) are left out of the studentized statistic.
 *
 * @param strategy_returns K daily return series, each the length of `benchmark`.
 * @param benchmark        Daily returns of the benchmark (e.g. buy-and-hold).
 */
[[nodiscard]] SpaResult superior_predictive_ability(const std::vector<std::vector<double>>& strategy_returns,
                                                    const std::vector<double>& benchmark,
                                                    const SpaConfig& config = SpaConfig{});

/**
 * @brief Text block for the CLI: one row per strategy plus the Reality Check and SPA p-values.
 *
 * @param names Strategy labels in the order passed to superior_predictive_ability.
 */
[[nodiscard]] std::string format_spa_report(const SpaResult& result,
                                            const std::vector<std::string>& names,
                                            const std::string& benchmark_name);

[[nodiscard]] std::string format_bootstrap_report(const BootstrapSharpeResult& result);

[[nodiscard]] std::string format_deflated_sharpe_report(const DeflatedSharpe& result);

}   // namespace quant::analysis
