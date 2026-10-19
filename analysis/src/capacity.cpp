#include "quant/analysis/capacity.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace quant::analysis {

namespace {

/// First capital at which `value(point)` falls from above `target` to at or below it, interpolated
/// linearly in log(capital); NaN when it never crosses.
template <typename Value>
double first_crossing(const std::vector<CapacityPoint>& points, double target, Value value) {
    for (size_t i = 1; i < points.size(); ++i) {
        const double before = value(points[i - 1]);
        const double after = value(points[i]);
        if (before > target && after <= target) {
            const double fraction = (before - target) / (before - after);
            const double lo = std::log(points[i - 1].capital);
            const double hi = std::log(points[i].capital);
            return std::exp(lo + fraction * (hi - lo));
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}   // namespace

std::vector<CapacityRow> sweep_capacity(const data::MarketDataUniverse& universe,
                                        const StrategyFactory& make_strategies,
                                        const BacktestSetup& setup,
                                        const std::vector<double>& capitals) {
    if (capitals.empty()) throw std::invalid_argument("sweep_capacity: need at least one account size");
    if (universe.size() < 2) throw std::invalid_argument("sweep_capacity: universe needs at least two bars");

    std::vector<double> levels = capitals;
    std::sort(levels.begin(), levels.end());

    std::vector<CapacityRow> rows;
    for (double capital : levels) {
        if (!(capital > 0.0) || !std::isfinite(capital)) {
            throw std::invalid_argument("sweep_capacity: account sizes must be positive and finite");
        }
        auto strategies = make_strategies();
        if (rows.empty()) rows.resize(strategies.size());
        if (rows.size() != strategies.size()) {
            throw std::invalid_argument(
                "sweep_capacity: the factory must return the same strategies every call");
        }

        for (size_t i = 0; i < strategies.size(); ++i) {
            backtest::BacktestEngine engine(universe, backtest::Portfolio(capital),
                                            backtest::ExecutionModel(setup.execution), setup.engine);
            const auto result = engine.run(*strategies[i]);

            // Same return convention as evaluate_window, so the smallest account reproduces the
            // cost-sensitivity table's 1x column when the sizes match.
            std::vector<double> returns;
            returns.reserve(result.equity_curve.size());
            for (size_t k = 1; k < result.equity_curve.size(); ++k) {
                const double prev = result.equity_curve[k - 1].equity;
                const double curr = result.equity_curve[k].equity;
                returns.push_back(prev > 0.0 ? curr / prev - 1.0 : 0.0);
            }

            CapacityPoint point;
            point.capital = capital;
            point.performance = performance_from_returns(returns, setup.risk_free_rate);
            point.performance.trades = result.trades.size();
            point.transaction_costs =
                result.total_commissions + result.total_slippage_cost + result.total_spread_cost;
            point.capped_fills = result.capped_fills;
            for (const auto& fill : result.trades) {
                point.peak_participation = std::max(point.peak_participation, fill.participation);
            }

            rows[i].strategy_name = strategies[i]->get_name();
            rows[i].points.push_back(point);
        }
    }

    const auto total_return = [](const CapacityPoint& p) { return p.performance.total_return; };
    for (auto& row : rows) {
        const double smallest = row.points.front().performance.total_return;
        row.half_return_capital = smallest > 0.0 ? first_crossing(row.points, 0.5 * smallest, total_return)
                                                 : std::numeric_limits<double>::quiet_NaN();
        row.breakeven_capital = smallest > 0.0 ? first_crossing(row.points, 0.0, total_return)
                                               : std::numeric_limits<double>::quiet_NaN();
    }
    return rows;
}

std::string format_capital(double capital) {
    if (!std::isfinite(capital)) return "-";
    struct Unit {
        double scale;
        const char* suffix;
    };
    static constexpr Unit kUnits[] = {{1e12, "T"}, {1e9, "B"}, {1e6, "M"}, {1e3, "k"}, {1.0, ""}};
    for (const auto& unit : kUnits) {
        if (capital >= unit.scale || unit.scale == 1.0) {
            const double value = capital / unit.scale;
            std::ostringstream ss;
            ss << std::fixed;
            // Whole amounts print bare ($100k); others keep three significant figures ($44.5M).
            const bool whole = std::abs(value - std::round(value)) < 1e-9;
            int decimals = 2;
            if (whole || value >= 100.0) decimals = 0;
            else if (value >= 10.0) decimals = 1;
            ss << std::setprecision(decimals);
            ss << "$" << value << unit.suffix;
            return ss.str();
        }
    }
    return "-";
}

std::string format_capacity_report(const std::vector<CapacityRow>& rows,
                                   const backtest::ExecutionConfig& execution) {
    std::ostringstream ss;
    if (rows.empty()) return ss.str();

    const size_t columns = rows.front().points.size();
    const std::string rule(42 + 9 * columns + 26, '-');
    ss << std::fixed;
    ss << "  CAPACITY - the same strategies and costs at larger account sizes\n";
    ss << "  Impact " << std::setprecision(2) << execution.market_impact_factor
       << " x sqrt(order / bar volume)";
    if (execution.max_participation > 0.0) {
        ss << "; each order takes at most " << std::setprecision(1) << (execution.max_participation * 100.0)
           << "% of a bar's volume, the rest waits";
    }
    ss << "\n";
    ss << "  " << rule << "\n";
    ss << "  " << std::left << std::setw(42) << "Total return" << std::right;
    for (const auto& point : rows.front().points)
        ss << std::setw(9) << format_capital(point.capital);
    ss << std::setw(13) << "Half-return" << std::setw(13) << "Break-even" << "\n";
    ss << "  " << rule << "\n";
    for (const auto& row : rows) {
        ss << "  " << std::left << std::setw(42) << row.strategy_name << std::right << std::setprecision(1);
        for (const auto& point : row.points)
            ss << std::setw(8) << (point.performance.total_return * 100.0) << "%";
        ss << std::setw(13) << format_capital(row.half_return_capital) << std::setw(13)
           << format_capital(row.breakeven_capital) << "\n";
    }
    ss << "  " << rule << "\n";
    ss << "  Costs as % of starting capital\n";
    for (const auto& row : rows) {
        ss << "  " << std::left << std::setw(42) << row.strategy_name << std::right << std::setprecision(1);
        for (const auto& point : row.points)
            ss << std::setw(8) << (point.transaction_costs / point.capital * 100.0) << "%";
        ss << "\n";
    }
    ss << "  Largest fill as % of its bar's volume\n";
    for (const auto& row : rows) {
        ss << "  " << std::left << std::setw(42) << row.strategy_name << std::right << std::setprecision(2);
        for (const auto& point : row.points)
            ss << std::setw(8) << (point.peak_participation * 100.0) << "%";
        ss << "\n";
    }
    ss << "  " << rule << "\n";
    ss << "  Half-return: account size at which the return falls to half the smallest account's. "
          "Break-even:\n";
    ss << "  where it reaches zero. Both interpolate on a log scale; '-' means not reached in this range.\n";
    return ss.str();
}

}   // namespace quant::analysis
