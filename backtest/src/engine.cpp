#include "quant/backtest/engine.hpp"
#include <algorithm>
#include <cmath>

namespace quant::backtest {

BacktestEngine::BacktestEngine(quant::data::MarketDataUniverse universe,
                               Portfolio portfolio,
                               ExecutionModel execution_model,
                               EngineConfig config)
    : universe_(std::move(universe)),
      portfolio_(std::move(portfolio)),
      execution_model_(execution_model),
      config_(config) {}

void BacktestEngine::execute_orders(std::vector<Order>& orders,
                                    const quant::data::MarketSnapshot& snapshot,
                                    bool fill_at_open,
                                    Strategy& strategy,
                                    BacktestResult& result) {
    // Liquidity first: sells release cash that subsequent buys in the same batch can use.
    std::stable_sort(orders.begin(), orders.end(), [](const Order& a, const Order& b) {
        return a.side == OrderSide::SELL && b.side == OrderSide::BUY;
    });

    for (auto& order : orders) {
        if (order.quantity <= 0.0 || !snapshot.has_ticker(order.ticker)) {
            ++result.rejected_orders;
            continue;
        }

        const auto& bar = snapshot.get_bar(order.ticker);
        const double reference_price = fill_at_open ? bar.open : bar.close;
        if (!(reference_price > 0.0) || !std::isfinite(reference_price)) {
            ++result.rejected_orders;
            continue;
        }

        order.order_id = next_order_id_++;
        order.date = bar.date;
        order.timestamp = bar.timestamp;

        Fill fill = execution_model_.execute_order(order, bar, reference_price);

        if (fill.side == OrderSide::BUY) {
            const double cash = portfolio_.get_cash();
            auto required = [](const Fill& f) { return f.quantity * f.execution_price + f.commission; };

            if (required(fill) > cash) {
                // Downsize to what the account can actually afford, then re-price the smaller order
                // (market impact and commissions depend on size, so the fill must be recomputed).
                Order resized = order;
                resized.quantity = std::floor((cash * config_.cash_buffer) / fill.execution_price);
                if (resized.quantity <= 0.0) {
                    ++result.rejected_orders;
                    continue;
                }
                fill = execution_model_.execute_order(resized, bar, reference_price);
                while (resized.quantity > 0.0 && required(fill) > cash) {
                    resized.quantity -= 1.0;
                    fill = execution_model_.execute_order(resized, bar, reference_price);
                }
                if (resized.quantity <= 0.0) {
                    ++result.rejected_orders;
                    continue;
                }
            }
        }

        fill.fill_id = next_fill_id_++;
        portfolio_.process_fill(fill);

        result.total_commissions += fill.commission;
        result.total_slippage_cost += fill.slippage * fill.quantity;
        result.total_spread_cost += fill.spread_cost;

        strategy.on_order_fill(portfolio_.get_trade_history().back());
    }
}

BacktestResult BacktestEngine::run(Strategy& strategy) {
    portfolio_.reset(portfolio_.get_initial_cash());
    next_order_id_ = 1;
    next_fill_id_ = 1;
    strategy.on_start(portfolio_, universe_);

    BacktestResult res;
    const size_t total_steps = universe_.size();
    const bool next_bar = (config_.fill_timing == FillTiming::NextBarOpen);

    std::vector<Order> queued;       // Orders waiting for the next bar's open
    std::vector<Order> new_orders;   // Orders emitted on the current bar

    for (size_t t = 0; t < total_steps; ++t) {
        const auto& snapshot = universe_.get_snapshot(t);

        // 1. Orders generated from information up to bar t-1 execute at bar t's open.
        if (next_bar && !queued.empty()) {
            execute_orders(queued, snapshot, /*fill_at_open=*/true, strategy, res);
            queued.clear();
        }

        // 2. Strategy observes the completed bar t and emits orders.
        new_orders.clear();
        strategy.on_bar(t, snapshot, portfolio_, new_orders);
        if (t < config_.trading_start_index) {
            new_orders.clear();   // warm-up: the strategy observes data but may not trade yet
        }

        if (next_bar) {
            queued.insert(queued.end(), new_orders.begin(), new_orders.end());
        } else {
            execute_orders(new_orders, snapshot, /*fill_at_open=*/false, strategy, res);
        }

        // 3. Mark to market at the close of bar t.
        portfolio_.mark_to_market(snapshot);
    }

    strategy.on_end(portfolio_);

    res.strategy_name = strategy.get_name();
    if (!universe_.empty()) {
        res.start_date = universe_.get_timeline().front();
        res.end_date = universe_.get_timeline().back();
        res.start_timestamp = universe_.get_snapshot(0).timestamp;
        res.end_timestamp = universe_.get_snapshot(total_steps - 1).timestamp;
    }
    res.initial_cash = portfolio_.get_initial_cash();
    res.final_equity = portfolio_.get_total_equity();
    res.total_return =
        (res.initial_cash > 0.0) ? ((res.final_equity - res.initial_cash) / res.initial_cash) : 0.0;
    res.total_bars = total_steps;
    res.total_trades = portfolio_.get_trade_history().size();
    res.unfilled_orders = queued.size();
    res.trades = portfolio_.get_trade_history();
    res.equity_curve = portfolio_.get_equity_curve();
    res.daily_returns = portfolio_.get_daily_returns();

    return res;
}

}   // namespace quant::backtest
