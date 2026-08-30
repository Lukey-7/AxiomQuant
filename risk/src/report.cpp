#include "quant/risk/report.hpp"
#include "quant/risk/var_cvar.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace quant::risk {

PerformanceSummary RiskReport::evaluate(const backtest::BacktestResult& result, double risk_free_rate) {
    PerformanceSummary s;
    s.initial_equity = result.initial_cash;
    s.final_equity = result.final_equity;
    s.total_return = result.total_return;

    // Estimate duration in years from daily bars (assuming 252 bars/year)
    double years = (result.total_bars > 0) ? (static_cast<double>(result.total_bars) / 252.0) : 1.0;
    s.cagr = RiskMetrics::cagr(s.total_return, years);

    const auto& rets = result.daily_returns;
    if (!rets.empty()) {
        s.daily_volatility = RiskMetrics::standard_deviation(rets);
        s.annualized_volatility = RiskMetrics::annualized_volatility(rets);
        s.sharpe_ratio = RiskMetrics::sharpe_ratio(rets, risk_free_rate);
        s.sortino_ratio = RiskMetrics::sortino_ratio(rets, 0.0);
        s.skewness = RiskMetrics::skewness(rets);
        s.excess_kurtosis = RiskMetrics::excess_kurtosis(rets);

        auto hist_var = ValueAtRisk::historical(rets, 0.95);
        s.var_95_historical = hist_var.var;
        s.cvar_95_historical = hist_var.cvar;

        auto param_var = ValueAtRisk::parametric(rets, 0.95);
        s.var_95_parametric = param_var.var;
        s.cvar_95_parametric = param_var.cvar;

        auto cf_var = ValueAtRisk::cornish_fisher(rets, 0.95);
        s.var_95_cornish_fisher = cf_var.var;
    }

    // Extract equity curve vector
    std::vector<double> eq_series;
    eq_series.reserve(result.equity_curve.size());
    for (const auto& pt : result.equity_curve) {
        eq_series.push_back(pt.equity);
    }

    s.drawdown_details = RiskMetrics::calculate_drawdown(eq_series);
    s.max_drawdown = s.drawdown_details.max_drawdown;
    s.calmar_ratio = RiskMetrics::calmar_ratio(s.cagr, s.max_drawdown);

    // Trade statistics
    s.trade_details.total_trades = static_cast<int>(result.trades.size());
    double gross_profit = 0.0;
    double gross_loss = 0.0;
    double largest_win = 0.0;
    double largest_loss = 0.0;
    double total_comm = 0.0;

    for (const auto& tr : result.trades) {
        total_comm += tr.commission;
        // Approximation of realized PnL per trade
        double pnl = (tr.side == backtest::OrderSide::SELL) ? (tr.price * tr.quantity - tr.commission) : 0.0;
        if (pnl > 0.0) {
            s.trade_details.winning_trades++;
            gross_profit += pnl;
            if (pnl > largest_win) largest_win = pnl;
        } else if (pnl < 0.0) {
            s.trade_details.losing_trades++;
            gross_loss += std::abs(pnl);
            if (pnl < largest_loss) largest_loss = pnl;
        }
    }

    s.trade_details.win_rate = (s.trade_details.total_trades > 0)
        ? (static_cast<double>(s.trade_details.winning_trades) / s.trade_details.total_trades)
        : 0.0;
    s.trade_details.profit_factor = (gross_loss > 0.0) ? (gross_profit / gross_loss) : (gross_profit > 0 ? 999.0 : 0.0);
    s.trade_details.largest_win = largest_win;
    s.trade_details.largest_loss = largest_loss;
    s.trade_details.total_commissions = total_comm;

    return s;
}

std::string RiskReport::generate_text_report(
    const backtest::BacktestResult& result,
    const PerformanceSummary& s
) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "=========================================================================\n";
    ss << "                      QUANT BACKTEST PERFORMANCE REPORT                  \n";
    ss << "=========================================================================\n";
    ss << "Strategy:       " << result.strategy_name << "\n";
    ss << "Period:         " << result.start_date << " -> " << result.end_date << " (" << result.total_bars << " bars)\n";
    ss << "Initial Cash:   $" << s.initial_equity << "\n";
    ss << "Final Equity:   $" << s.final_equity << "\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " RETURN & GROWTH METRICS\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Total Return:                " << (s.total_return * 100.0) << " %\n";
    ss << "  CAGR (Annualized Return):    " << (s.cagr * 100.0) << " %\n";
    ss << "  Annualized Volatility:       " << (s.annualized_volatility * 100.0) << " %\n";
    ss << "  Daily Volatility:            " << (s.daily_volatility * 100.0) << " %\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " RISK-ADJUSTED RETURN METRICS\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Sharpe Ratio (Rf=2%):        " << std::setprecision(3) << s.sharpe_ratio << "\n";
    ss << "  Sortino Ratio (MAR=0%):      " << s.sortino_ratio << "\n";
    ss << "  Calmar Ratio (CAGR/MaxDD):   " << s.calmar_ratio << "\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " DRAWDOWN & TAIL RISK (95% Daily)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << std::setprecision(2);
    ss << "  Max Drawdown:                " << (s.max_drawdown * 100.0) << " %\n";
    ss << "  Historical VaR (95%):        " << (s.var_95_historical * 100.0) << " %\n";
    ss << "  Historical CVaR / ES (95%):  " << (s.cvar_95_historical * 100.0) << " %\n";
    ss << "  Parametric Gaussian VaR (95%):" << (s.var_95_parametric * 100.0) << " %\n";
    ss << "  Cornish-Fisher VaR (95%):    " << (s.var_95_cornish_fisher * 100.0) << " %\n";
    ss << "  Return Skewness:             " << std::setprecision(3) << s.skewness << "\n";
    ss << "  Return Excess Kurtosis:      " << s.excess_kurtosis << "\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " TRADE EXECUTION STATISTICS\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Total Fills Executed:        " << result.total_trades << "\n";
    ss << "  Total Commissions Paid:      $" << std::setprecision(2) << s.trade_details.total_commissions << "\n";
    ss << "=========================================================================\n";

    return ss.str();
}

std::string RiskReport::render_ascii_chart(
    const std::vector<double>& equity_points,
    size_t width,
    size_t height
) {
    if (equity_points.empty() || width == 0 || height == 0) return "";

    double min_val = *std::min_element(equity_points.begin(), equity_points.end());
    double max_val = *std::max_element(equity_points.begin(), equity_points.end());
    if (std::abs(max_val - min_val) < 1e-6) {
        max_val = min_val + 1.0;
    }

    // Sample into `width` bins
    std::vector<double> sampled(width, 0.0);
    double step = static_cast<double>(equity_points.size()) / static_cast<double>(width);
    for (size_t col = 0; col < width; ++col) {
        size_t idx = static_cast<size_t>(std::floor(col * step));
        if (idx >= equity_points.size()) idx = equity_points.size() - 1;
        sampled[col] = equity_points[idx];
    }

    // Grid buffer
    std::vector<std::string> grid(height, std::string(width, ' '));

    for (size_t col = 0; col < width; ++col) {
        double norm = (sampled[col] - min_val) / (max_val - min_val);
        size_t row = static_cast<size_t>(std::floor(norm * (height - 1)));
        if (row >= height) row = height - 1;
        size_t grid_row = (height - 1) - row; // invert for top-down display
        grid[grid_row][col] = '*';
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "\n--- ASCII EQUITY CURVE ---\n";
    for (size_t r = 0; r < height; ++r) {
        double level = max_val - (static_cast<double>(r) / (height - 1)) * (max_val - min_val);
        ss << std::setw(9) << level << " |";
        ss << grid[r] << "\n";
    }
    ss << "          +" << std::string(width, '-') << "\n";
    ss << "          Start: " << std::setw(8) << equity_points.front()
       << std::string(width > 24 ? width - 24 : 1, ' ')
       << "End: " << std::setw(8) << equity_points.back() << "\n\n";

    return ss.str();
}

} // namespace quant::risk
