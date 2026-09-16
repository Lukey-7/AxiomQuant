#pragma once

#include <cstdint>

#include "quant/backtest/portfolio.hpp"
#include "quant/backtest/execution_model.hpp"
#include "quant/backtest/strategy.hpp"
#include "quant/data/universe.hpp"
#include <memory>
#include <string>

namespace quant::backtest {

/**
 * @brief Determines when orders emitted during bar t are executed.
 *
 * NextBarOpen (default) is the realistic, look-ahead-free mode: a signal computed from the
 * close of bar t can only be traded at the open of bar t+1. SameBarClose reproduces the
 * idealised "trade on the signal close" convention and is kept for comparison studies.
 */
enum class FillTiming : std::uint8_t {
    NextBarOpen,
    SameBarClose
};

struct EngineConfig {
    FillTiming fill_timing{FillTiming::NextBarOpen};
    double cash_buffer{0.995};    // Fraction of cash a BUY may consume when an order must be downsized
    size_t trading_start_index{0};// Bars before this index are indicator warm-up only: orders are discarded
};

struct BacktestResult {
    std::string strategy_name;
    std::string start_date;
    std::string end_date;
    int64_t start_timestamp{0};
    int64_t end_timestamp{0};
    double initial_cash{0.0};
    double final_equity{0.0};
    double total_return{0.0};
    size_t total_bars{0};
    size_t total_trades{0};
    size_t rejected_orders{0};     // Orders dropped (no cash, no price, unknown ticker)
    size_t unfilled_orders{0};     // Orders still queued when the data ran out
    double total_commissions{0.0};
    double total_slippage_cost{0.0};
    double total_spread_cost{0.0};
    std::vector<Fill> trades;
    std::vector<quant::data::EquityPointRecord> equity_curve;
    std::vector<double> daily_returns;
};

class BacktestEngine {
public:
    BacktestEngine(
        quant::data::MarketDataUniverse universe,
        Portfolio portfolio = Portfolio(100000.0),
        ExecutionModel execution_model = ExecutionModel{},
        EngineConfig config = EngineConfig{}
    );

    /**
     * @brief Executes backtest of the given strategy across the historical universe.
     */
    BacktestResult run(Strategy& strategy);

    [[nodiscard]] const quant::data::MarketDataUniverse& get_universe() const noexcept { return universe_; }
    [[nodiscard]] const Portfolio& get_portfolio() const noexcept { return portfolio_; }
    [[nodiscard]] const ExecutionModel& get_execution_model() const noexcept { return execution_model_; }
    [[nodiscard]] const EngineConfig& get_config() const noexcept { return config_; }

    void set_portfolio(Portfolio portfolio) { portfolio_ = std::move(portfolio); }
    void set_execution_model(ExecutionModel model) { execution_model_ = model; }
    void set_config(EngineConfig config) { config_ = config; }

private:
    /**
     * @brief Executes a batch of orders against a snapshot. SELLs are processed before BUYs so
     *        that rebalancing frees cash before it is re-deployed.
     */
    void execute_orders(
        std::vector<Order>& orders,
        const quant::data::MarketSnapshot& snapshot,
        bool fill_at_open,
        Strategy& strategy,
        BacktestResult& result
    );

    quant::data::MarketDataUniverse universe_;
    Portfolio portfolio_;
    ExecutionModel execution_model_;
    EngineConfig config_;
    int64_t next_order_id_{1};
    int64_t next_fill_id_{1};
};

} // namespace quant::backtest
