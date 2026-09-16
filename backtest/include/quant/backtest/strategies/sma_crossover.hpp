#pragma once

#include "quant/backtest/strategy.hpp"
#include "quant/indicators/sma.hpp"
#include <vector>
#include <string>
#include <cmath>

namespace quant::backtest::strategies {

class SmaCrossoverStrategy : public Strategy {
public:
    SmaCrossoverStrategy(std::string ticker,
                         size_t fast_period = 20,
                         size_t slow_period = 50,
                         double allocation_pct = 0.95)
        : ticker_(std::move(ticker)),
          fast_period_(fast_period),
          slow_period_(slow_period),
          allocation_pct_(allocation_pct) {}

    [[nodiscard]] std::string get_name() const override {
        return "SMA_Crossover(" + ticker_ + ", " + std::to_string(fast_period_) + "/" +
               std::to_string(slow_period_) + ")";
    }

    void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& universe) override {
        const auto closes = universe.get_aligned_closes(ticker_);
        fast_sma_ = quant::indicators::SMA::calculate(closes, fast_period_);
        slow_sma_ = quant::indicators::SMA::calculate(closes, slow_period_);
    }

    void on_bar(size_t timeline_index,
                const quant::data::MarketSnapshot& snapshot,
                const Portfolio& portfolio,
                std::vector<Order>& pending_orders) override {
        if (timeline_index == 0 || timeline_index >= fast_sma_.size()) return;
        if (!snapshot.has_ticker(ticker_)) return;

        double prev_fast = fast_sma_[timeline_index - 1];
        double prev_slow = slow_sma_[timeline_index - 1];
        double curr_fast = fast_sma_[timeline_index];
        double curr_slow = slow_sma_[timeline_index];

        if (quant::indicators::is_nan(prev_fast) || quant::indicators::is_nan(prev_slow) ||
            quant::indicators::is_nan(curr_fast) || quant::indicators::is_nan(curr_slow)) {
            return;
        }

        const auto& bar = snapshot.get_bar(ticker_);
        double current_qty = portfolio.get_position_quantity(ticker_);

        // Bullish Crossover: Fast SMA crosses above Slow SMA
        if (prev_fast <= prev_slow && curr_fast > curr_slow) {
            if (current_qty <= 0.0) {
                // Compute shares to buy with allocated cash
                double target_equity = portfolio.get_total_equity() * allocation_pct_;
                double allocatable_cash = std::min(portfolio.get_cash() * 0.98, target_equity);
                if (allocatable_cash > 100.0 && bar.close > 0.0) {
                    double shares = std::floor(allocatable_cash / bar.close);
                    if (shares > 0.0) {
                        Order buy_order;
                        buy_order.ticker = ticker_;
                        buy_order.side = OrderSide::BUY;
                        buy_order.type = OrderType::MARKET;
                        buy_order.quantity = shares;
                        pending_orders.push_back(buy_order);
                    }
                }
            }
        }
        // Bearish Crossover: Fast SMA crosses below Slow SMA
        else if (prev_fast >= prev_slow && curr_fast < curr_slow) {
            if (current_qty > 0.0) {
                // Exit long position
                Order sell_order;
                sell_order.ticker = ticker_;
                sell_order.side = OrderSide::SELL;
                sell_order.type = OrderType::MARKET;
                sell_order.quantity = current_qty;
                pending_orders.push_back(sell_order);
            }
        }
    }

private:
    std::string ticker_;
    size_t fast_period_;
    size_t slow_period_;
    double allocation_pct_;
    std::vector<double> fast_sma_;
    std::vector<double> slow_sma_;
};

}   // namespace quant::backtest::strategies
