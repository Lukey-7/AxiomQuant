#pragma once

#include "quant/backtest/strategy.hpp"
#include <cmath>
#include <string>

namespace quant::backtest {
namespace strategies {

/**
 * @brief Passive benchmark: invests `allocation_pct` of equity in a single asset as soon as it can
 *        and holds to the end. Every active strategy should be judged against this.
 *
 * The entry order is re-sent if it never turns into a position (dropped during a warm-up window,
 * or too little cash), but never while a fill is still in flight.
 */
class BuyAndHoldStrategy : public Strategy {
public:
    explicit BuyAndHoldStrategy(std::string ticker, double allocation_pct = 0.99)
        : ticker_(std::move(ticker)), allocation_pct_(allocation_pct) {}

    [[nodiscard]] std::string get_name() const override { return "BuyAndHold(" + ticker_ + ")"; }

    void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& /*universe*/) override {
        order_sent_ = false;
        order_bar_ = 0;
    }

    void on_bar(size_t timeline_index,
                const quant::data::MarketSnapshot& snapshot,
                const Portfolio& portfolio,
                std::vector<Order>& pending_orders) override {
        if (!snapshot.has_ticker(ticker_)) return;
        if (portfolio.get_position_quantity(ticker_) != 0.0) return;
        if (order_sent_ && timeline_index <= order_bar_ + 1) return;   // a fill may still be pending

        const double price = snapshot.get_bar(ticker_).close;
        if (!(price > 0.0)) return;
        const double shares = std::floor(portfolio.get_total_equity() * allocation_pct_ / price);
        if (shares <= 0.0) return;

        Order order;
        order.ticker = ticker_;
        order.side = OrderSide::BUY;
        order.type = OrderType::MARKET;
        order.quantity = shares;
        pending_orders.push_back(order);
        order_sent_ = true;
        order_bar_ = timeline_index;
    }

private:
    std::string ticker_;
    double allocation_pct_;
    bool order_sent_{false};
    size_t order_bar_{0};
};

}   // namespace strategies
}   // namespace quant::backtest
