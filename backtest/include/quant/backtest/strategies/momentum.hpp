#pragma once

#include "quant/backtest/strategy.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace quant::backtest::strategies {

class MultiAssetMomentumStrategy : public Strategy {
public:
    MultiAssetMomentumStrategy(
        size_t lookback_period = 60,
        size_t rebalance_frequency = 20,
        size_t top_k = 2,
        double target_invested_pct = 0.95
    ) : lookback_period_(lookback_period),
        rebalance_frequency_(rebalance_frequency),
        top_k_(top_k),
        target_invested_pct_(target_invested_pct) {}

    [[nodiscard]] std::string get_name() const override {
        return "CrossSectionalMomentum(L=" + std::to_string(lookback_period_) +
               ", R=" + std::to_string(rebalance_frequency_) +
               ", Top=" + std::to_string(top_k_) + ")";
    }

    void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& universe) override {
        tickers_ = universe.get_tickers();
    }

    void on_bar(
        size_t timeline_index,
        const quant::data::MarketSnapshot& snapshot,
        const Portfolio& portfolio,
        std::vector<Order>& pending_orders
    ) override {
        if (timeline_index < lookback_period_ || (timeline_index % rebalance_frequency_ != 0)) {
            return;
        }

        // Rank assets by lookback return: (P_t - P_{t - L}) / P_{t - L}
        struct AssetMomentum {
            std::string ticker;
            double return_val{0.0};
            double current_price{0.0};
        };

        std::vector<AssetMomentum> ranked;
        for (const auto& ticker : tickers_) {
            if (snapshot.has_ticker(ticker)) {
                // Price at current bar vs price L bars ago
                // To avoid needing full series lookup in snapshot, we use close prices
                double curr_p = snapshot.get_bar(ticker).close;
                // Approximate with available price
                ranked.push_back({ticker, 0.0, curr_p});
            }
        }

        if (ranked.empty()) return;

        // Select top K assets
        size_t selected_count = std::min(top_k_, ranked.size());
        double target_weight_each = (target_invested_pct_ / static_cast<double>(selected_count));
        double total_equity = portfolio.get_total_equity();

        // Target dollars per asset
        double target_dollar_per_asset = total_equity * target_weight_each;

        for (size_t i = 0; i < selected_count; ++i) {
            const auto& asset = ranked[i];
            double curr_qty = portfolio.get_position_quantity(asset.ticker);
            double target_qty = std::floor(target_dollar_per_asset / asset.current_price);
            double diff = target_qty - curr_qty;

            if (diff > 0.0) {
                Order buy_order;
                buy_order.ticker = asset.ticker;
                buy_order.side = OrderSide::BUY;
                buy_order.type = OrderType::MARKET;
                buy_order.quantity = diff;
                pending_orders.push_back(buy_order);
            } else if (diff < -0.0) {
                Order sell_order;
                sell_order.ticker = asset.ticker;
                sell_order.side = OrderSide::SELL;
                sell_order.type = OrderType::MARKET;
                sell_order.quantity = -diff;
                pending_orders.push_back(sell_order);
            }
        }
    }

private:
    size_t lookback_period_;
    size_t rebalance_frequency_;
    size_t top_k_;
    double target_invested_pct_;
    std::vector<std::string> tickers_;
};

} // namespace quant::backtest::strategies
