#include "quant/risk/metrics.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace quant::risk {

double RiskMetrics::cagr(double total_return, double years) {
    if (years <= 0.0) return 0.0;
    if (total_return <= -1.0) return -1.0;
    return std::pow(1.0 + total_return, 1.0 / years) - 1.0;
}

double RiskMetrics::mean(const std::vector<double>& returns) {
    if (returns.empty()) return 0.0;
    double sum = std::accumulate(returns.begin(), returns.end(), 0.0);
    return sum / static_cast<double>(returns.size());
}

double RiskMetrics::variance(const std::vector<double>& returns) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;
    double m = mean(returns);
    double sum_sq = 0.0;
    for (double r : returns) {
        double diff = r - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(n - 1);
}

double RiskMetrics::standard_deviation(const std::vector<double>& returns) {
    return std::sqrt(variance(returns));
}

double RiskMetrics::annualized_volatility(const std::vector<double>& daily_returns, double ann_factor) {
    if (daily_returns.size() < 2) return 0.0;
    return standard_deviation(daily_returns) * std::sqrt(ann_factor);
}

double RiskMetrics::sharpe_ratio(
    const std::vector<double>& daily_returns,
    double risk_free_rate,
    double ann_factor
) {
    if (daily_returns.size() < 2) return 0.0;
    double daily_mean = mean(daily_returns);
    double ann_return = daily_mean * ann_factor;
    double ann_vol = annualized_volatility(daily_returns, ann_factor);
    if (ann_vol <= 1e-9) return 0.0;
    return (ann_return - risk_free_rate) / ann_vol;
}

double RiskMetrics::sortino_ratio(
    const std::vector<double>& daily_returns,
    double mar,
    double ann_factor
) {
    if (daily_returns.size() < 2) return 0.0;
    double daily_mar = mar / ann_factor;
    double daily_mean = mean(daily_returns);
    double ann_return = daily_mean * ann_factor;

    double sum_downside_sq = 0.0;
    for (double r : daily_returns) {
        double underperformance = std::min(0.0, r - daily_mar);
        sum_downside_sq += underperformance * underperformance;
    }

    double downside_var = sum_downside_sq / static_cast<double>(daily_returns.size());
    double downside_dev = std::sqrt(downside_var) * std::sqrt(ann_factor);

    if (downside_dev <= 1e-9) return 0.0;
    return (ann_return - mar) / downside_dev;
}

DrawdownInfo RiskMetrics::calculate_drawdown(const std::vector<double>& equity_curve) {
    DrawdownInfo info;
    if (equity_curve.empty()) return info;

    double peak = equity_curve[0];
    size_t current_peak_idx = 0;
    double max_dd = 0.0;
    size_t max_peak_idx = 0;
    size_t max_trough_idx = 0;
    size_t current_duration = 0;
    size_t max_duration = 0;

    for (size_t i = 0; i < equity_curve.size(); ++i) {
        double eq = equity_curve[i];
        if (eq >= peak) {
            peak = eq;
            current_peak_idx = i;
            current_duration = 0;
        } else {
            current_duration++;
            if (current_duration > max_duration) {
                max_duration = current_duration;
            }
            double dd = (peak - eq) / peak;
            if (dd > max_dd) {
                max_dd = dd;
                max_peak_idx = current_peak_idx;
                max_trough_idx = i;
            }
        }
    }

    // Find recovery index
    size_t recovery_idx = max_trough_idx;
    double peak_val_at_max = equity_curve[max_peak_idx];
    for (size_t i = max_trough_idx + 1; i < equity_curve.size(); ++i) {
        if (equity_curve[i] >= peak_val_at_max) {
            recovery_idx = i;
            break;
        }
    }

    info.max_drawdown = max_dd;
    info.peak_index = max_peak_idx;
    info.trough_index = max_trough_idx;
    info.recovery_index = recovery_idx;
    info.max_duration_bars = max_duration;

    return info;
}

double RiskMetrics::calmar_ratio(double cagr_val, double max_dd) {
    if (std::abs(max_dd) <= 1e-9) return 0.0;
    return cagr_val / std::abs(max_dd);
}

double RiskMetrics::skewness(const std::vector<double>& returns) {
    const size_t n = returns.size();
    if (n < 3) return 0.0;
    double m = mean(returns);
    double s = standard_deviation(returns);
    if (s <= 1e-9) return 0.0;

    double sum_cubed = 0.0;
    for (double r : returns) {
        double z = (r - m) / s;
        sum_cubed += z * z * z;
    }
    double factor = static_cast<double>(n) / (static_cast<double>(n - 1) * static_cast<double>(n - 2));
    return factor * sum_cubed;
}

double RiskMetrics::excess_kurtosis(const std::vector<double>& returns) {
    const size_t n = returns.size();
    if (n < 4) return 0.0;
    double m = mean(returns);
    double s = standard_deviation(returns);
    if (s <= 1e-9) return 0.0;

    double sum_fourth = 0.0;
    for (double r : returns) {
        double z = (r - m) / s;
        sum_fourth += z * z * z * z;
    }
    double dn = static_cast<double>(n);
    double term1 = (dn * (dn + 1.0)) / ((dn - 1.0) * (dn - 2.0) * (dn - 3.0)) * sum_fourth;
    double term2 = (3.0 * (dn - 1.0) * (dn - 1.0)) / ((dn - 2.0) * (dn - 3.0));
    return term1 - term2;
}

} // namespace quant::risk
