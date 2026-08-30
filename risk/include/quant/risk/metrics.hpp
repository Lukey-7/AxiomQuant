#pragma once

#include <vector>
#include <string>
#include <cstddef>

namespace quant::risk {

struct DrawdownInfo {
    double max_drawdown{0.0};      // Maximum percentage drawdown (e.g. 0.15 = 15%)
    size_t peak_index{0};
    size_t trough_index{0};
    size_t recovery_index{0};
    size_t max_duration_bars{0};
};

struct TradeMetrics {
    int total_trades{0};
    int winning_trades{0};
    int losing_trades{0};
    double win_rate{0.0};          // winning_trades / total_trades
    double profit_factor{0.0};     // gross_profit / gross_loss
    double total_realized_pnl{0.0};
    double average_trade_pnl{0.0};
    double largest_win{0.0};
    double largest_loss{0.0};
    double total_commissions{0.0};
};

struct PerformanceSummary {
    double initial_equity{0.0};
    double final_equity{0.0};
    double total_return{0.0};
    double cagr{0.0};
    double annualized_volatility{0.0};
    double daily_volatility{0.0};
    double sharpe_ratio{0.0};
    double sortino_ratio{0.0};
    double max_drawdown{0.0};
    double calmar_ratio{0.0};
    double var_95_historical{0.0};
    double cvar_95_historical{0.0};
    double var_95_parametric{0.0};
    double cvar_95_parametric{0.0};
    double var_95_cornish_fisher{0.0};
    double skewness{0.0};
    double excess_kurtosis{0.0};
    DrawdownInfo drawdown_details;
    TradeMetrics trade_details;
};

class RiskMetrics {
public:
    /**
     * @brief Computes annualized return (CAGR).
     * @param total_return Total cumulative return (e.g. 0.50 = 50%).
     * @param years Number of years duration.
     */
    [[nodiscard]] static double cagr(double total_return, double years);

    /**
     * @brief Computes sample mean of returns.
     */
    [[nodiscard]] static double mean(const std::vector<double>& returns);

    /**
     * @brief Computes sample variance of returns.
     */
    [[nodiscard]] static double variance(const std::vector<double>& returns);

    /**
     * @brief Computes sample standard deviation of returns.
     */
    [[nodiscard]] static double standard_deviation(const std::vector<double>& returns);

    /**
     * @brief Computes annualized volatility: sigma_daily * sqrt(annualization_factor).
     */
    [[nodiscard]] static double annualized_volatility(const std::vector<double>& daily_returns, double ann_factor = 252.0);

    /**
     * @brief Computes annualized Sharpe Ratio: (mean(R) * ann_factor - Rf) / (std(R) * sqrt(ann_factor)).
     * @param daily_returns Vector of daily percentage returns.
     * @param risk_free_rate Annualized risk-free rate (e.g. 0.02 = 2%).
     * @param ann_factor Trading days per year (default 252.0).
     */
    [[nodiscard]] static double sharpe_ratio(
        const std::vector<double>& daily_returns,
        double risk_free_rate = 0.0,
        double ann_factor = 252.0
    );

    /**
     * @brief Computes Sortino Ratio using downside deviation below minimum acceptable return (MAR).
     * @param daily_returns Vector of daily percentage returns.
     * @param mar Annualized minimum acceptable return target (default 0.0).
     * @param ann_factor Trading days per year (default 252.0).
     */
    [[nodiscard]] static double sortino_ratio(
        const std::vector<double>& daily_returns,
        double mar = 0.0,
        double ann_factor = 252.0
    );

    /**
     * @brief Computes maximum drawdown and peak/trough details from an equity curve.
     */
    [[nodiscard]] static DrawdownInfo calculate_drawdown(const std::vector<double>& equity_curve);

    /**
     * @brief Computes Calmar Ratio: CAGR / MaxDrawdown.
     */
    [[nodiscard]] static double calmar_ratio(double cagr_val, double max_dd);

    /**
     * @brief Computes sample skewness of returns.
     */
    [[nodiscard]] static double skewness(const std::vector<double>& returns);

    /**
     * @brief Computes sample excess kurtosis of returns (normal distribution = 0.0).
     */
    [[nodiscard]] static double excess_kurtosis(const std::vector<double>& returns);
};

} // namespace quant::risk
