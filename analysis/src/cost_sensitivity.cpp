#include "quant/analysis/cost_sensitivity.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace quant::analysis {

backtest::ExecutionConfig scale_costs(const backtest::ExecutionConfig& base, double multiplier) {
    backtest::ExecutionConfig scaled = base;
    scaled.per_share_commission *= multiplier;
    scaled.min_commission *= multiplier;
    scaled.percentage_commission *= multiplier;
    scaled.fixed_slippage_bps *= multiplier;
    scaled.spread_bps *= multiplier;
    scaled.market_impact_factor *= multiplier;
    return scaled;
}

std::vector<CostSensitivityRow> sweep_costs(const data::MarketDataUniverse& universe,
                                            const StrategyFactory& make_strategies,
                                            const BacktestSetup& setup,
                                            const std::vector<double>& multipliers) {
    if (multipliers.empty()) throw std::invalid_argument("sweep_costs: need at least one cost multiplier");
    if (universe.size() < 2) throw std::invalid_argument("sweep_costs: universe needs at least two bars");

    std::vector<double> levels = multipliers;
    std::sort(levels.begin(), levels.end());

    std::vector<CostSensitivityRow> rows;
    for (double multiplier : levels) {
        if (multiplier < 0.0) throw std::invalid_argument("sweep_costs: multipliers must be non-negative");
        BacktestSetup scaled_setup = setup;
        scaled_setup.execution = scale_costs(setup.execution, multiplier);

        auto strategies = make_strategies();
        if (rows.empty()) rows.resize(strategies.size());
        if (rows.size() != strategies.size()) {
            throw std::invalid_argument("sweep_costs: the factory must return the same strategies every call");
        }
        for (size_t i = 0; i < strategies.size(); ++i) {
            rows[i].strategy_name = strategies[i]->get_name();
            rows[i].points.push_back(
                {multiplier, evaluate_window(universe, *strategies[i], scaled_setup, 0, universe.size(), 0)});
        }
    }

    // Where does each strategy stop making money? Interpolate the first crossing of zero return.
    for (auto& row : rows) {
        row.breakeven_multiplier = std::numeric_limits<double>::quiet_NaN();
        for (size_t i = 1; i < row.points.size(); ++i) {
            const double before = row.points[i - 1].performance.total_return;
            const double after = row.points[i].performance.total_return;
            if (before > 0.0 && after <= 0.0) {
                const double span = before - after;
                const double fraction = span > 0.0 ? before / span : 0.0;
                row.breakeven_multiplier =
                    row.points[i - 1].multiplier +
                    fraction * (row.points[i].multiplier - row.points[i - 1].multiplier);
                break;
            }
        }
    }
    return rows;
}

std::string format_cost_sensitivity_report(const std::vector<CostSensitivityRow>& rows) {
    std::ostringstream ss;
    if (rows.empty()) return ss.str();

    ss << std::fixed;
    ss << "  COST SENSITIVITY - every cost component scaled by the same multiplier\n";
    ss << "  " << std::string(88, '-') << "\n";
    ss << "  " << std::left << std::setw(42) << "Strategy" << std::right;
    for (const auto& point : rows.front().points) {
        std::ostringstream label;
        label << std::fixed << std::setprecision(point.multiplier < 1.0 ? 1 : 0) << point.multiplier << "x";
        ss << std::setw(8) << label.str();
    }
    ss << std::setw(12) << "Break-even" << "\n";
    ss << "  " << std::string(88, '-') << "\n";

    for (const auto& row : rows) {
        ss << "  " << std::left << std::setw(42) << row.strategy_name << std::right << std::setprecision(1);
        for (const auto& point : row.points) {
            ss << std::setw(7) << (point.performance.total_return * 100.0) << "%";
        }
        if (std::isnan(row.breakeven_multiplier)) {
            ss << std::setw(12) << "-";
        } else {
            std::ostringstream breakeven;
            breakeven << std::fixed << std::setprecision(2) << row.breakeven_multiplier << "x";
            ss << std::setw(12) << breakeven.str();
        }
        ss << "\n";
    }
    ss << "  " << std::string(88, '-') << "\n";
    ss << "  Returns at each cost level; break-even is where the total return reaches zero.\n";
    ss << "  A strategy that only works below 1x is not paying for its own trading.\n";
    return ss.str();
}

}   // namespace quant::analysis
