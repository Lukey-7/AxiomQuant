#pragma once

#include "quant/analysis/cost_sensitivity.hpp"
#include "quant/analysis/walk_forward.hpp"
#include "quant/data/universe.hpp"
#include <string>
#include <vector>

namespace quant::analysis {

struct CapacityPoint {
    double capital{0.0};
    WindowPerformance performance;
    double transaction_costs{0.0};    // Commissions + slippage/impact + half-spread, in dollars
    double peak_participation{0.0};   // Largest fill as a share of its bar's volume
    size_t capped_fills{0};           // Fills cut short by the participation cap
};

struct CapacityRow {
    std::string strategy_name;
    std::vector<CapacityPoint> points;
    /// Capital at which the total return falls to half of the smallest account's return, log-linearly
    /// interpolated between the bracketing points. NaN when it never does inside the sweep, or when
    /// the smallest account did not make money to begin with.
    double half_return_capital{0.0};
    /// Capital at which the total return reaches zero, interpolated the same way. NaN when the
    /// strategy stays profitable (or was never profitable) across the sweep.
    double breakeven_capital{0.0};
};

/**
 * @brief Re-runs every strategy over the whole universe at increasing account sizes.
 *
 * Costs that scale with order size relative to the market, the square-root impact term and the
 * participation cap, make a strategy that works for $100k fail for $1B. Running the same rules at
 * several sizes answers the question a desk actually asks: at what size does this stop working?
 *
 * @param capitals Account sizes to test, e.g. {1e5, 1e6, 1e7, 1e8, 1e9}. Sorted ascending on return.
 */
[[nodiscard]] std::vector<CapacityRow> sweep_capacity(const data::MarketDataUniverse& universe,
                                                      const StrategyFactory& make_strategies,
                                                      const BacktestSetup& setup,
                                                      const std::vector<double>& capitals);

/// "$100k", "$2.5M", "$1B": compact dollar labels for the capacity table.
[[nodiscard]] std::string format_capital(double capital);

[[nodiscard]] std::string format_capacity_report(const std::vector<CapacityRow>& rows,
                                                 const backtest::ExecutionConfig& execution);

}   // namespace quant::analysis
