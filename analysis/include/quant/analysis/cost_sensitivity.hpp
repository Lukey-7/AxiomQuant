#pragma once

#include "quant/analysis/walk_forward.hpp"
#include "quant/backtest/engine.hpp"
#include "quant/data/universe.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace quant::analysis {

/**
 * @brief Scales every component of an execution model by the same factor.
 *
 * Commissions, the spread, fixed slippage and the market-impact coefficient are all multiplied, so a
 * multiplier of 2 means "trading is twice as expensive in every respect".
 */
[[nodiscard]] backtest::ExecutionConfig scale_costs(const backtest::ExecutionConfig& base, double multiplier);

struct CostSensitivityPoint {
    double multiplier{1.0};
    WindowPerformance performance;
};

struct CostSensitivityRow {
    std::string strategy_name;
    std::vector<CostSensitivityPoint> points;
    /// Multiplier at which the total return crosses zero, linearly interpolated between the two
    /// bracketing points. NaN when the strategy stays profitable (or unprofitable) across the sweep.
    double breakeven_multiplier{0.0};
};

/// Builds a fresh set of strategies; called once per cost level because the engine consumes them.
using StrategyFactory = std::function<std::vector<std::unique_ptr<backtest::Strategy>>()>;

/**
 * @brief Re-runs every strategy over the whole universe at several cost levels.
 *
 * A backtest at one cost assumption says nothing about how fragile the result is. Running the same
 * strategies from free trading up to several times the modelled cost shows which results are edges
 * and which are artefacts of an optimistic cost model.
 *
 * @param multipliers Cost levels to test, e.g. {0.0, 0.5, 1.0, 2.0, 5.0}. Sorted ascending on return.
 */
[[nodiscard]] std::vector<CostSensitivityRow> sweep_costs(const data::MarketDataUniverse& universe,
                                                          const StrategyFactory& make_strategies,
                                                          const BacktestSetup& setup,
                                                          const std::vector<double>& multipliers);

[[nodiscard]] std::string format_cost_sensitivity_report(const std::vector<CostSensitivityRow>& rows);

}   // namespace quant::analysis
