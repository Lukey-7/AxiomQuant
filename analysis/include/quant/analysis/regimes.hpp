#pragma once

#include "quant/analysis/walk_forward.hpp"
#include <string>
#include <vector>

namespace quant::analysis {

/// Direction of the benchmark's trailing return. Unknown until enough history exists.
enum class TrendRegime { Unknown, Rising, Falling };

/// Tercile of the benchmark's trailing realized volatility. Unknown until enough history exists.
enum class VolatilityRegime { Unknown, Low, Mid, High };

[[nodiscard]] const char* to_string(TrendRegime regime);
[[nodiscard]] const char* to_string(VolatilityRegime regime);

struct RegimeConfig {
    /// Bars in the trailing return whose sign defines the trend (200 ~ ten months).
    size_t trend_lookback{200};
    /// Daily returns in the trailing realized-volatility estimate (21 ~ one month).
    size_t vol_window{21};
    /// Volatility observations required before tercile cut points are trusted; bars before that
    /// are labelled Unknown rather than classified against a handful of points.
    size_t min_vol_history{252};
    /// Daily returns in each rolling Sharpe window (252 ~ one year).
    size_t rolling_window{252};
};

/**
 * @brief Point-in-time regime labels, one entry per bar of the benchmark's close series.
 *
 * Every entry at bar t is computed from closes[0..t] only, so changing a later bar never changes an
 * earlier label. The label at bar t describes the market as it stood at that close, and is applied
 * to the return earned from bar t to bar t+1.
 */
struct RegimeLabels {
    std::vector<TrendRegime> trend;
    std::vector<VolatilityRegime> volatility;
    std::vector<double> trailing_return;   // closes[t] / closes[t - lookback] - 1; NaN if undefined
    std::vector<double> realized_vol;      // annualized; NaN if undefined
    std::vector<double> low_cut;           // expanding 1/3 quantile of realized_vol[0..t]; NaN if unset
    std::vector<double> high_cut;          // expanding 2/3 quantile of realized_vol[0..t]; NaN if unset

    [[nodiscard]] size_t size() const noexcept { return trend.size(); }
};

/**
 * @brief Classifies every bar by trend and volatility regime without look-ahead.
 *
 * Trend: the sign of the trailing `trend_lookback`-bar return (Rising if > 0, Falling otherwise).
 * Volatility: the sample standard deviation of the last `vol_window` simple returns, annualized,
 * compared with the 1/3 and 2/3 quantiles of every volatility observed up to and including bar t
 * (an expanding window, so the cut points themselves are point-in-time). Low if at or below the
 * lower cut, High if above the upper cut, Mid otherwise. Bars without enough history, or whose
 * window touches a missing (NaN or non-positive) close, are Unknown.
 */
[[nodiscard]] RegimeLabels classify_regimes(const std::vector<double>& closes,
                                            const RegimeConfig& config = {});

/// Performance of one strategy restricted to the days of one regime.
struct RegimePerformance {
    std::string regime;
    size_t days{0};
    double exposure{0.0};            // share of those days that ended with a position open
    double total_return{0.0};        // compounded over the regime's days only
    double annualized_return{0.0};   // (1 + total_return)^(252 / days) - 1
    double sharpe_ratio{0.0};
    double max_drawdown{0.0};   // of the equity curve stitched from the regime's days
};

/**
 * @brief Splits a daily return series by regime.
 *
 * `returns[i]` is the return from bar i to bar i+1 and is attributed to the regime labelled at bar
 * i, which was known before the return was earned. Rows, in order: All, Rising, Falling, Trend
 * unknown, Low vol, Mid vol, High vol, Vol unknown. The trend rows partition the days, as do the
 * volatility rows, so their day counts sum to the All row and their compounded returns multiply
 * back to it.
 *
 * @param invested Optional, same length as `returns`: whether a position was open at the end of
 *                 each return's day. Empty leaves every exposure at zero.
 */
[[nodiscard]] std::vector<RegimePerformance> regime_breakdown(const std::vector<double>& returns,
                                                              const std::vector<bool>& invested,
                                                              const RegimeLabels& labels,
                                                              double risk_free_rate);

/**
 * @brief Annualized Sharpe ratio of each trailing `window`-day block of returns.
 *
 * Entry k covers returns[k .. k + window - 1]. Returns an empty series when there are fewer than
 * `window` returns. A window with zero volatility has a Sharpe of 0, as in RiskMetrics.
 */
[[nodiscard]] std::vector<double> rolling_sharpe(const std::vector<double>& returns,
                                                 size_t window,
                                                 double risk_free_rate);

/// Regime split and rolling Sharpe of one strategy.
struct StrategyRegimeReport {
    std::string strategy_name;
    std::vector<RegimePerformance> regimes;
    std::vector<double> rolling_sharpe;
};

/**
 * @brief Runs regime_breakdown and rolling_sharpe for one strategy's daily returns.
 */
[[nodiscard]] StrategyRegimeReport analyze_regimes(const std::string& strategy_name,
                                                   const std::vector<double>& returns,
                                                   const std::vector<bool>& invested,
                                                   const RegimeLabels& labels,
                                                   const RegimeConfig& config,
                                                   double risk_free_rate);

/**
 * @brief Text report: label counts, the per-regime table for every strategy and a rolling Sharpe
 *        summary. The first report is treated as the benchmark in the rolling comparison.
 */
[[nodiscard]] std::string format_regime_report(const RegimeLabels& labels,
                                               const std::vector<StrategyRegimeReport>& reports,
                                               const RegimeConfig& config,
                                               const std::string& benchmark_ticker);

}   // namespace quant::analysis
