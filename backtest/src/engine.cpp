#include "quant/backtest/engine.hpp"
#include <iostream>

namespace quant::backtest {

BacktestEngine::BacktestEngine(
    quant::data::MarketDataUniverse universe,
    Portfolio portfolio,
    ExecutionModel execution_model
) : universe_(std::move(universe)),
    portfolio_(std::move(portfolio)),
    execution_model_(execution_model) {}

BacktestResult BacktestEngine::run(Strategy& strategy) {
    portfolio_.reset(portfolio_.get_initial_cash());
    strategy.on_start(portfolio_, universe_);

    const size_t total_steps = universe_.size();
    std::vector<Order> pending_orders;

    for (size_t t = 0; t < total_steps; ++t) {
        auto snapshot = universe_.get_snapshot(t);
        pending_orders.clear();

        // 1. Let strategy evaluate current market state and generate signals/orders
        strategy.on_bar(t, snapshot, portfolio_, pending_orders);

        // 2. Process all generated orders with realistic execution model
        for (auto& order : pending_orders) {
            if (order.quantity <= 0.0) continue;
            if (!snapshot.has_ticker(order.ticker)) continue;

            const auto& bar = snapshot.get_bar(order.ticker);
            order.order_id = next_order_id_++;
            order.date = bar.date;
            order.timestamp = bar.timestamp;

            // Execute order
            Fill fill = execution_model_.execute_order(order, bar);

            // Safety check for cash on BUY orders
            if (fill.side == OrderSide::BUY) {
                double required_cash = (fill.quantity * fill.execution_price) + fill.commission;
                if (required_cash > portfolio_.get_cash() && portfolio_.get_cash() > 0.0) {
                    // Downsize order to fit remaining cash
                    double allocatable_cash = portfolio_.get_cash() * 0.99; // 1% buffer
                    fill.quantity = std::floor(allocatable_cash / fill.execution_price);
                    if (fill.quantity <= 0.0) continue;
                    fill = execution_model_.execute_order(order, bar); // recalculate costs
                } else if (portfolio_.get_cash() <= 0.0) {
                    // Insufficient cash
                    continue;
                }
            }

            portfolio_.process_fill(fill);
            strategy.on_order_fill(fill);
        }

        // 3. Mark to market at end of bar
        portfolio_.mark_to_market(snapshot);
    }

    strategy.on_end(portfolio_);

    BacktestResult res;
    res.strategy_name = strategy.get_name();
    if (!universe_.empty()) {
        res.start_date = universe_.get_timeline().front();
        res.end_date = universe_.get_timeline().back();
    }
    res.initial_cash = portfolio_.get_initial_cash();
    res.final_equity = portfolio_.get_total_equity();
    res.total_return = (res.initial_cash > 0.0) ? ((res.final_equity - res.initial_cash) / res.initial_cash) : 0.0;
    res.total_bars = total_steps;
    res.total_trades = portfolio_.get_trade_history().size();
    res.trades = portfolio_.get_trade_history();
    res.equity_curve = portfolio_.get_equity_curve();
    res.daily_returns = portfolio_.get_daily_returns();

    return res;
}

} // namespace quant::backtest
