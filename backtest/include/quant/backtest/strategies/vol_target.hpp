#pragma once

#include "quant/backtest/strategy.hpp"
#include "quant/indicators/rolling_volatility.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace quant::backtest::strategies {

/**
 * @brief Holds one asset at a weight that targets a constant portfolio volatility.
 *
 * The target weight on bar t is  w_t = target_vol / sigma_t,  where sigma_t is the annualized
 * rolling volatility of log returns over the previous `vol_window` bars, capped at `max_weight`.
 * Volatility is persistent in a way that returns are not, so this sizes down before turbulent
 * periods rather than after them; the weight is deliberately computed from bar t and traded at
 * t+1's open, like every other strategy here.
 *
 * Rebalancing only happens when the target moves by more than `rebalance_threshold` (a fraction of
 * equity), because a strategy that rebalances every bar pays away the benefit in transaction costs.
 */
class VolatilityTargetStrategy : public Strategy {
public:
    VolatilityTargetStrategy(std::string ticker,
                             double target_volatility = 0.10,
                             size_t vol_window = 20,
                             double max_weight = 1.0,
                             double rebalance_threshold = 0.05)
        : ticker_(std::move(ticker)),
          target_volatility_(target_volatility),
          vol_window_(vol_window),
          max_weight_(max_weight),
          rebalance_threshold_(rebalance_threshold) {}

    [[nodiscard]] std::string get_name() const override {
        const long target_pct = std::lround(target_volatility_ * 100.0);
        return "VolTarget(" + ticker_ + ", " + std::to_string(target_pct) + "% vol, " +
               std::to_string(vol_window_) + "d)";
    }

    void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& universe) override {
        volatility_ = quant::indicators::RollingVolatility::calculate(universe.get_aligned_closes(ticker_),
                                                                      vol_window_);
    }

    void on_bar(size_t timeline_index,
                const quant::data::MarketSnapshot& snapshot,
                const Portfolio& portfolio,
                std::vector<Order>& pending_orders) override {
        if (timeline_index >= volatility_.size() || !snapshot.has_ticker(ticker_)) return;
        const double sigma = volatility_[timeline_index];
        if (quant::indicators::is_nan(sigma) || sigma <= 1e-9) return;

        const auto& bar = snapshot.get_bar(ticker_);
        if (bar.close <= 0.0) return;

        const double equity = portfolio.get_total_equity();
        if (equity <= 0.0) return;

        const double weight = std::min(max_weight_, target_volatility_ / sigma);
        const double target_shares = std::floor(weight * equity / bar.close);
        const double current_shares = portfolio.get_position_quantity(ticker_);
        const double delta = target_shares - current_shares;
        if (std::abs(delta) * bar.close < rebalance_threshold_ * equity) return;

        Order order;
        order.ticker = ticker_;
        order.type = OrderType::MARKET;
        if (delta > 0.0) {
            // Leave a little cash for the costs the fill will charge.
            const double affordable = std::floor(portfolio.get_cash() * 0.98 / bar.close);
            order.side = OrderSide::BUY;
            order.quantity = std::min(delta, affordable);
        } else {
            order.side = OrderSide::SELL;
            order.quantity = std::min(-delta, current_shares);
        }
        if (order.quantity > 0.0) pending_orders.push_back(order);
    }

private:
    std::string ticker_;
    double target_volatility_;
    size_t vol_window_;
    double max_weight_;
    double rebalance_threshold_;
    std::vector<double> volatility_;
};

}   // namespace quant::backtest::strategies
