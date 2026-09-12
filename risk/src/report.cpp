#include "quant/risk/report.hpp"
#include "quant/risk/var_cvar.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

namespace quant::risk {

namespace {

constexpr double kSecondsPerYear = 365.25 * 86400.0;

std::string format_ratio(double v) {
    if (std::isinf(v)) return "inf";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << v;
    return ss.str();
}

} // namespace

PerformanceSummary RiskReport::evaluate(const backtest::BacktestResult& result, double risk_free_rate) {
    PerformanceSummary s;
    s.risk_free_rate = risk_free_rate;
    s.initial_equity = result.initial_cash;
    s.final_equity = result.final_equity;
    s.total_return = result.total_return;

    // Annualize over calendar time when timestamps are available; fall back to 252 bars/year.
    if (result.end_timestamp > result.start_timestamp) {
        s.years = static_cast<double>(result.end_timestamp - result.start_timestamp) / kSecondsPerYear;
    } else if (result.total_bars > 1) {
        s.years = static_cast<double>(result.total_bars - 1) / 252.0;
    }
    s.cagr = RiskMetrics::cagr(s.total_return, s.years);

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

    // Equity curve & market exposure
    std::vector<double> eq_series;
    eq_series.reserve(result.equity_curve.size());
    size_t invested_bars = 0;
    for (const auto& pt : result.equity_curve) {
        eq_series.push_back(pt.equity);
        if (std::abs(pt.positions_value) > 1e-9) ++invested_bars;
    }
    s.exposure = eq_series.empty() ? 0.0 : static_cast<double>(invested_bars) / static_cast<double>(eq_series.size());

    s.drawdown_details = RiskMetrics::calculate_drawdown(eq_series);
    s.max_drawdown = s.drawdown_details.max_drawdown;
    s.calmar_ratio = RiskMetrics::calmar_ratio(s.cagr, s.max_drawdown);

    // Trade statistics computed from PnL actually realized by closing fills.
    auto& td = s.trade_details;
    td.total_trades = static_cast<int>(result.trades.size());
    for (const auto& tr : result.trades) {
        td.total_commissions += tr.commission;
        if (!tr.closes_position) continue;

        ++td.closed_trades;
        const double pnl = tr.realized_pnl;
        td.total_realized_pnl += pnl;
        if (pnl > 0.0) {
            ++td.winning_trades;
            td.gross_profit += pnl;
            td.largest_win = std::max(td.largest_win, pnl);
        } else if (pnl < 0.0) {
            ++td.losing_trades;
            td.gross_loss += -pnl;
            td.largest_loss = std::min(td.largest_loss, pnl);
        }
    }

    if (td.closed_trades > 0) {
        td.win_rate = static_cast<double>(td.winning_trades) / static_cast<double>(td.closed_trades);
        td.average_trade_pnl = td.total_realized_pnl / static_cast<double>(td.closed_trades);
    }
    if (td.gross_loss > 0.0) {
        td.profit_factor = td.gross_profit / td.gross_loss;
    } else if (td.gross_profit > 0.0) {
        td.profit_factor = std::numeric_limits<double>::infinity();
    }

    s.total_transaction_costs = result.total_commissions + result.total_slippage_cost + result.total_spread_cost;

    return s;
}

std::string RiskReport::generate_text_report(
    const backtest::BacktestResult& result,
    const PerformanceSummary& s
) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    const auto& td = s.trade_details;

    ss << "=========================================================================\n";
    ss << "                      QUANT BACKTEST PERFORMANCE REPORT                  \n";
    ss << "=========================================================================\n";
    ss << "Strategy:       " << result.strategy_name << "\n";
    ss << "Period:         " << result.start_date << " -> " << result.end_date
       << " (" << result.total_bars << " bars, " << s.years << " yrs)\n";
    ss << "Initial Cash:   $" << s.initial_equity << "\n";
    ss << "Final Equity:   $" << s.final_equity << "\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " RETURN & GROWTH METRICS\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Total Return:                " << (s.total_return * 100.0) << " %\n";
    ss << "  CAGR (Annualized Return):    " << (s.cagr * 100.0) << " %\n";
    ss << "  Annualized Volatility:       " << (s.annualized_volatility * 100.0) << " %\n";
    ss << "  Daily Volatility:            " << (s.daily_volatility * 100.0) << " %\n";
    ss << "  Market Exposure:             " << (s.exposure * 100.0) << " % of bars\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " RISK-ADJUSTED RETURN METRICS\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << "  Sharpe Ratio (Rf=" << std::setprecision(1) << (s.risk_free_rate * 100.0) << "%):      "
       << std::setprecision(3) << s.sharpe_ratio << "\n";
    ss << "  Sortino Ratio (MAR=0%):      " << s.sortino_ratio << "\n";
    ss << "  Calmar Ratio (CAGR/MaxDD):   " << s.calmar_ratio << "\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " DRAWDOWN & TAIL RISK (95% Daily)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << std::setprecision(2);
    ss << "  Max Drawdown:                " << (s.max_drawdown * 100.0) << " %\n";
    ss << "  Longest Underwater Stretch:  " << s.drawdown_details.max_duration_bars << " bars\n";
    ss << "  Historical VaR (95%):        " << (s.var_95_historical * 100.0) << " %\n";
    ss << "  Historical CVaR / ES (95%):  " << (s.cvar_95_historical * 100.0) << " %\n";
    ss << "  Gaussian VaR (95%):          " << (s.var_95_parametric * 100.0) << " %\n";
    ss << "  Cornish-Fisher VaR (95%):    " << (s.var_95_cornish_fisher * 100.0) << " %\n";
    ss << "  Return Skewness:             " << std::setprecision(3) << s.skewness << "\n";
    ss << "  Return Excess Kurtosis:      " << s.excess_kurtosis << "\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << " TRADE STATISTICS (realized PnL of closing fills)\n";
    ss << "-------------------------------------------------------------------------\n";
    ss << std::setprecision(2);
    ss << "  Fills Executed:              " << result.total_trades
       << "  (" << td.closed_trades << " closing)\n";
    if (td.closed_trades > 0) {
        ss << "  Win Rate:                    " << (td.win_rate * 100.0) << " %\n";
    } else {
        ss << "  Win Rate:                    n/a (no positions were closed)\n";
    }
    ss << "  Profit Factor:               " << format_ratio(td.profit_factor) << "\n";
    ss << "  Avg PnL per Closed Trade:    $" << td.average_trade_pnl << "\n";
    ss << "  Largest Win / Loss:          $" << td.largest_win << " / $" << td.largest_loss << "\n";
    ss << "  Commissions Paid:            $" << td.total_commissions << "\n";
    ss << "  Total Transaction Costs:     $" << s.total_transaction_costs
       << "  (commission + slippage + spread)\n";
    ss << "=========================================================================\n";

    return ss.str();
}

std::string RiskReport::render_ascii_chart(
    const std::vector<double>& equity_points,
    size_t width,
    size_t height
) {
    if (equity_points.empty() || width == 0 || height < 2) return "";

    double min_val = *std::min_element(equity_points.begin(), equity_points.end());
    double max_val = *std::max_element(equity_points.begin(), equity_points.end());
    if (std::abs(max_val - min_val) < 1e-6) {
        max_val = min_val + 1.0;
    }

    // Sample into `width` bins
    std::vector<double> sampled(width, 0.0);
    double step = static_cast<double>(equity_points.size()) / static_cast<double>(width);
    for (size_t col = 0; col < width; ++col) {
        size_t idx = static_cast<size_t>(std::floor(static_cast<double>(col) * step));
        if (idx >= equity_points.size()) idx = equity_points.size() - 1;
        sampled[col] = equity_points[idx];
    }

    std::vector<std::string> grid(height, std::string(width, ' '));
    auto to_row = [&](double v) {
        double norm = (v - min_val) / (max_val - min_val);
        size_t row = static_cast<size_t>(std::floor(norm * static_cast<double>(height - 1) + 0.5));
        if (row >= height) row = height - 1;
        return (height - 1) - row; // invert for top-down display
    };

    // Draw a connected line: fill the vertical gap between consecutive samples.
    size_t prev_row = to_row(sampled[0]);
    for (size_t col = 0; col < width; ++col) {
        size_t row = to_row(sampled[col]);
        size_t lo = std::min(row, prev_row);
        size_t hi = std::max(row, prev_row);
        for (size_t r = lo; r <= hi; ++r) grid[r][col] = (r == row) ? '*' : '|';
        prev_row = row;
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0);
    ss << "\n--- EQUITY CURVE ---\n";
    for (size_t r = 0; r < height; ++r) {
        double level = max_val - (static_cast<double>(r) / static_cast<double>(height - 1)) * (max_val - min_val);
        ss << std::setw(10) << level << " |" << grid[r] << "\n";
    }
    ss << std::string(11, ' ') << "+" << std::string(width, '-') << "\n";
    ss << std::string(12, ' ') << "Start: $" << equity_points.front()
       << std::string(width > 32 ? width - 32 : 1, ' ')
       << "End: $" << equity_points.back() << "\n\n";

    return ss.str();
}

} // namespace quant::risk
