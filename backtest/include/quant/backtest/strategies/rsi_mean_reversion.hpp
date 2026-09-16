#pragma once

#include "quant/backtest/strategy.hpp"
#include "quant/indicators/rsi.hpp"
#include <vector>
#include <string>
#include <cmath>

namespace quant::backtest::strategies {

class RsiMeanReversionStrategy : public Strategy {
public:
    RsiMeanReversionStrategy(std::string ticker,
                             size_t period = 14,
                             double oversold = 30.0,
                             double overbought = 70.0,
                             double allocation_pct = 0.95)
        : ticker_(std::move(ticker)),
          period_(period),
          oversold_(oversold),
          overbought_(overbought),
          allocation_pct_(allocation_pct) {}

    [[nodiscard]] std::string get_name() const override {
        return "RSI_MeanReversion(" + ticker_ + ", " + std::to_string(period_) + ")";
    }

    void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& universe) override {
        rsi_ = quant::indicators::RSI::calculate(universe.get_aligned_closes(ticker_), period_);
    }

    void on_bar(size_t timeline_index,
                const quant::data::MarketSnapshot& snapshot,
                const Portfolio& portfolio,
                std::vector<Order>& pending_orders) override {
        if (!snapshot.has_ticker(ticker_) || timeline_index >= rsi_.size()) return;

        double curr_rsi = rsi_[timeline_index];
        if (quant::indicators::is_nan(curr_rsi)) return;

        const auto& bar = snapshot.get_bar(ticker_);
        double current_qty = portfolio.get_position_quantity(ticker_);

        // Buy Oversold
        if (curr_rsi < oversold_ && current_qty <= 0.0) {
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
        // Sell Overbought
        else if (curr_rsi > overbought_ && current_qty > 0.0) {
            Order sell_order;
            sell_order.ticker = ticker_;
            sell_order.side = OrderSide::SELL;
            sell_order.type = OrderType::MARKET;
            sell_order.quantity = current_qty;
            pending_orders.push_back(sell_order);
        }
    }

private:
    std::string ticker_;
    size_t period_;
    double oversold_;
    double overbought_;
    double allocation_pct_;
    std::vector<double> rsi_;
};

}   // namespace quant::backtest::strategies
