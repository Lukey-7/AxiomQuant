#include "quant/analysis/regimes.hpp"
#include "quant/risk/metrics.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace quant::analysis {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool valid_price(double price) {
    return std::isfinite(price) && price > 0.0;
}

/// Linear-interpolated quantile of an ascending, non-empty sample (NumPy's default method).
double sorted_quantile(const std::vector<double>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

RegimePerformance summarize(const std::string& name,
                            const std::vector<double>& returns,
                            size_t invested_days,
                            double risk_free_rate) {
    const WindowPerformance perf = performance_from_returns(returns, risk_free_rate);
    RegimePerformance row;
    row.regime = name;
    row.days = perf.days;
    row.total_return = perf.total_return;
    row.sharpe_ratio = perf.sharpe_ratio;
    row.max_drawdown = perf.max_drawdown;
    if (row.days > 0) {
        row.exposure = static_cast<double>(invested_days) / static_cast<double>(row.days);
        row.annualized_return =
            perf.total_return > -1.0
                ? std::pow(1.0 + perf.total_return, 252.0 / static_cast<double>(row.days)) - 1.0
                : -1.0;
    }
    return row;
}

double median_of(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

}   // namespace

const char* to_string(TrendRegime regime) {
    switch (regime) {
        case TrendRegime::Rising: return "rising";
        case TrendRegime::Falling: return "falling";
        case TrendRegime::Unknown: break;
    }
    return "unknown";
}

const char* to_string(VolatilityRegime regime) {
    switch (regime) {
        case VolatilityRegime::Low: return "low";
        case VolatilityRegime::Mid: return "mid";
        case VolatilityRegime::High: return "high";
        case VolatilityRegime::Unknown: break;
    }
    return "unknown";
}

RegimeLabels classify_regimes(const std::vector<double>& closes, const RegimeConfig& config) {
    if (config.trend_lookback == 0 || config.vol_window < 2 || config.min_vol_history == 0) {
        throw std::invalid_argument(
            "classify_regimes: need trend_lookback >= 1, vol_window >= 2 and min_vol_history >= 1");
    }

    const size_t n = closes.size();
    RegimeLabels labels;
    labels.trend.assign(n, TrendRegime::Unknown);
    labels.volatility.assign(n, VolatilityRegime::Unknown);
    labels.trailing_return.assign(n, kNaN);
    labels.realized_vol.assign(n, kNaN);
    labels.low_cut.assign(n, kNaN);
    labels.high_cut.assign(n, kNaN);

    // Trend: only closes[t - lookback] and closes[t] are read.
    for (size_t t = config.trend_lookback; t < n; ++t) {
        const double past = closes[t - config.trend_lookback];
        if (!valid_price(past) || !valid_price(closes[t])) continue;
        const double r = closes[t] / past - 1.0;
        labels.trailing_return[t] = r;
        labels.trend[t] = r > 0.0 ? TrendRegime::Rising : TrendRegime::Falling;
    }

    // Volatility: the window ending at t reads closes[t - vol_window .. t]; the cut points read only
    // the volatilities already computed, so everything at bar t is known at bar t's close.
    std::vector<double> window(config.vol_window);
    std::vector<double> history;   // every realized vol so far, kept sorted
    history.reserve(n);
    const double annualize = std::sqrt(252.0);
    for (size_t t = config.vol_window; t < n; ++t) {
        bool complete = true;
        for (size_t k = 0; k < config.vol_window; ++k) {
            const double prev = closes[t - config.vol_window + k];
            const double curr = closes[t - config.vol_window + k + 1];
            if (!valid_price(prev) || !valid_price(curr)) {
                complete = false;
                break;
            }
            window[k] = curr / prev - 1.0;
        }
        if (!complete) continue;

        const double vol = risk::RiskMetrics::standard_deviation(window) * annualize;
        labels.realized_vol[t] = vol;
        history.insert(std::upper_bound(history.begin(), history.end(), vol), vol);
        if (history.size() < config.min_vol_history) continue;

        const double low = sorted_quantile(history, 1.0 / 3.0);
        const double high = sorted_quantile(history, 2.0 / 3.0);
        labels.low_cut[t] = low;
        labels.high_cut[t] = high;
        if (vol <= low) {
            labels.volatility[t] = VolatilityRegime::Low;
        } else if (vol > high) {
            labels.volatility[t] = VolatilityRegime::High;
        } else {
            labels.volatility[t] = VolatilityRegime::Mid;
        }
    }
    return labels;
}

std::vector<RegimePerformance> regime_breakdown(const std::vector<double>& returns,
                                                const std::vector<bool>& invested,
                                                const RegimeLabels& labels,
                                                double risk_free_rate) {
    if (returns.size() > labels.size()) {
        throw std::invalid_argument("regime_breakdown: more returns than labelled bars");
    }
    if (!invested.empty() && invested.size() != returns.size()) {
        throw std::invalid_argument("regime_breakdown: invested flags must match the returns");
    }

    // Bucket order: All, Rising, Falling, Trend unknown, Low, Mid, High, Vol unknown.
    constexpr size_t kBuckets = 8;
    static const char* const kNames[kBuckets] = {"All",     "Rising",  "Falling",  "Trend unknown",
                                                 "Low vol", "Mid vol", "High vol", "Vol unknown"};
    std::vector<std::vector<double>> bucket_returns(kBuckets);
    std::vector<size_t> bucket_invested(kBuckets, 0);

    for (size_t i = 0; i < returns.size(); ++i) {
        size_t trend_bucket = 3;
        switch (labels.trend[i]) {
            case TrendRegime::Rising: trend_bucket = 1; break;
            case TrendRegime::Falling: trend_bucket = 2; break;
            case TrendRegime::Unknown: break;
        }
        size_t vol_bucket = 7;
        switch (labels.volatility[i]) {
            case VolatilityRegime::Low: vol_bucket = 4; break;
            case VolatilityRegime::Mid: vol_bucket = 5; break;
            case VolatilityRegime::High: vol_bucket = 6; break;
            case VolatilityRegime::Unknown: break;
        }
        const bool in_market = !invested.empty() && invested[i];
        for (size_t bucket : {size_t{0}, trend_bucket, vol_bucket}) {
            bucket_returns[bucket].push_back(returns[i]);
            if (in_market) ++bucket_invested[bucket];
        }
    }

    std::vector<RegimePerformance> rows;
    rows.reserve(kBuckets);
    for (size_t b = 0; b < kBuckets; ++b) {
        rows.push_back(summarize(kNames[b], bucket_returns[b], bucket_invested[b], risk_free_rate));
    }
    return rows;
}

std::vector<double> rolling_sharpe(const std::vector<double>& returns, size_t window, double risk_free_rate) {
    if (window < 2) throw std::invalid_argument("rolling_sharpe: window must be at least 2");
    std::vector<double> out;
    if (returns.size() < window) return out;
    out.reserve(returns.size() - window + 1);
    std::vector<double> block(window);
    for (size_t k = 0; k + window <= returns.size(); ++k) {
        std::copy(returns.begin() + static_cast<std::ptrdiff_t>(k),
                  returns.begin() + static_cast<std::ptrdiff_t>(k + window), block.begin());
        out.push_back(risk::RiskMetrics::sharpe_ratio(block, risk_free_rate));
    }
    return out;
}

StrategyRegimeReport analyze_regimes(const std::string& strategy_name,
                                     const std::vector<double>& returns,
                                     const std::vector<bool>& invested,
                                     const RegimeLabels& labels,
                                     const RegimeConfig& config,
                                     double risk_free_rate) {
    StrategyRegimeReport report;
    report.strategy_name = strategy_name;
    report.regimes = regime_breakdown(returns, invested, labels, risk_free_rate);
    report.rolling_sharpe = rolling_sharpe(returns, config.rolling_window, risk_free_rate);
    return report;
}

std::string format_regime_report(const RegimeLabels& labels,
                                 const std::vector<StrategyRegimeReport>& reports,
                                 const RegimeConfig& config,
                                 const std::string& benchmark_ticker) {
    std::ostringstream ss;
    ss << std::fixed;
    ss << "  Regimes from " << benchmark_ticker << ", using only data up to each day's close:\n";
    ss << "    trend = sign of the trailing " << config.trend_lookback << "-day return;\n";
    ss << "    volatility = " << config.vol_window
       << "-day realized vol against expanding-window terciles (after " << config.min_vol_history
       << " observations).\n";
    ss << "  The label at day t is applied to the return from t to t+1.\n";

    size_t counts[3] = {0, 0, 0};
    size_t vol_counts[4] = {0, 0, 0, 0};
    for (size_t t = 0; t < labels.size(); ++t) {
        ++counts[static_cast<size_t>(labels.trend[t])];
        ++vol_counts[static_cast<size_t>(labels.volatility[t])];
    }
    ss << "  Bars: rising " << counts[1] << ", falling " << counts[2] << ", unknown " << counts[0]
       << "  |  low vol " << vol_counts[1] << ", mid " << vol_counts[2] << ", high " << vol_counts[3]
       << ", unknown " << vol_counts[0] << "\n";

    const std::string rule(98, '-');
    ss << "  " << rule << "\n";
    ss << "  " << std::left << std::setw(42) << "Strategy / regime" << std::right << std::setw(7) << "Days"
       << std::setw(10) << "Exposure" << std::setw(10) << "Return" << std::setw(10) << "Ann.ret"
       << std::setw(9) << "Sharpe" << std::setw(9) << "MaxDD" << "\n";
    ss << "  " << rule << "\n";
    for (const auto& report : reports) {
        ss << "  " << report.strategy_name << "\n";
        for (const auto& row : report.regimes) {
            ss << "    " << std::left << std::setw(40) << row.regime << std::right << std::setw(7)
               << row.days;
            if (row.days == 0) {
                ss << std::setw(10) << "-" << std::setw(10) << "-" << std::setw(10) << "-" << std::setw(9)
                   << "-" << std::setw(9) << "-" << "\n";
                continue;
            }
            ss << std::setprecision(1) << std::setw(9) << (row.exposure * 100.0) << "%" << std::setw(9)
               << (row.total_return * 100.0) << "%" << std::setw(9) << (row.annualized_return * 100.0) << "%"
               << std::setprecision(2) << std::setw(9) << row.sharpe_ratio << std::setprecision(1)
               << std::setw(8) << (row.max_drawdown * 100.0) << "%\n";
        }
    }
    ss << "  " << rule << "\n";
    ss << "  Return is compounded over the regime's days only; MaxDD is of that stitched curve.\n";

    ss << "\n  ROLLING " << config.rolling_window << "-DAY SHARPE\n";
    if (reports.empty() || reports.front().rolling_sharpe.empty()) {
        ss << "  Skipped: fewer than " << config.rolling_window << " daily returns.\n";
        return ss.str();
    }
    const auto& benchmark = reports.front().rolling_sharpe;
    ss << "  " << rule << "\n";
    ss << "  " << std::left << std::setw(42) << "Strategy" << std::right << std::setw(8) << "Windows"
       << std::setw(8) << "Min" << std::setw(8) << "Median" << std::setw(8) << "Max" << std::setw(8) << "Last"
       << std::setw(8) << "> 0" << std::setw(8) << "> B&H" << "\n";
    ss << "  " << rule << "\n";
    for (size_t r = 0; r < reports.size(); ++r) {
        const auto& series = reports[r].rolling_sharpe;
        if (series.empty()) continue;
        size_t positive = 0, beat = 0;
        for (size_t k = 0; k < series.size(); ++k) {
            if (series[k] > 0.0) ++positive;
            if (k < benchmark.size() && series[k] > benchmark[k]) ++beat;
        }
        const double windows = static_cast<double>(series.size());
        ss << "  " << std::left << std::setw(42) << reports[r].strategy_name.substr(0, 41) << std::right
           << std::setw(8) << series.size() << std::setprecision(2) << std::setw(8)
           << *std::min_element(series.begin(), series.end()) << std::setw(8) << median_of(series)
           << std::setw(8) << *std::max_element(series.begin(), series.end()) << std::setw(8) << series.back()
           << std::setprecision(0) << std::setw(7) << (100.0 * static_cast<double>(positive) / windows)
           << "%";
        if (r == 0) {
            ss << std::setw(8) << "-" << "\n";
        } else {
            ss << std::setw(7) << (100.0 * static_cast<double>(beat) / windows) << "%\n";
        }
    }
    ss << "  " << rule << "\n";
    ss << "  '> B&H' is the share of windows where the strategy's Sharpe beat the first row's. A window\n";
    ss << "  spent mostly in cash has almost no volatility, so its Sharpe can be extreme.\n";
    return ss.str();
}

}   // namespace quant::analysis
