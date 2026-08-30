#pragma once

#include "quant/backtest/portfolio.hpp"
#include "quant/backtest/execution_model.hpp"
#include "quant/backtest/strategy.hpp"
#include "quant/data/universe.hpp"
#include <memory>
#include <string>

namespace quant::backtest {

struct BacktestResult {
    std::string strategy_name;
    std::string start_date;
    std::string end_date;
    double initial_cash{0.0};
    double final_equity{0.0};
    double total_return{0.0};
    size_t total_bars{0};
    size_t total_trades{0};
    std::vector<Fill> trades;
    std::vector<quant::data::EquityPointRecord> equity_curve;
    std::vector<double> daily_returns;
};

class BacktestEngine {
public:
    BacktestEngine(
        quant::data::MarketDataUniverse universe,
        Portfolio portfolio = Portfolio(100000.0),
        ExecutionModel execution_model = ExecutionModel{}
    );

    /**
     * @brief Executes backtest of the given strategy across the historical universe.
     */
    BacktestResult run(Strategy& strategy);

    [[nodiscard]] const quant::data::MarketDataUniverse& get_universe() const noexcept { return universe_; }
    [[nodiscard]] const Portfolio& get_portfolio() const noexcept { return portfolio_; }
    [[nodiscard]] const ExecutionModel& get_execution_model() const noexcept { return execution_model_; }

    void set_portfolio(Portfolio portfolio) { portfolio_ = std::move(portfolio); }
    void set_execution_model(ExecutionModel model) { execution_model_ = model; }

private:
    quant::data::MarketDataUniverse universe_;
    Portfolio portfolio_;
    ExecutionModel execution_model_;
    int64_t next_order_id_{1};
};

} // namespace quant::backtest
