#pragma once

#include "quant/backtest/order.hpp"
#include "quant/data/types.hpp"
#include <algorithm>
#include <cmath>

namespace quant::backtest {

struct ExecutionConfig {
    double per_share_commission{0.005};     // $0.005 per share
    double min_commission{1.00};            // $1.00 minimum ticket fee
    double percentage_commission{0.0001};   // 1 basis point (0.01%) of notional
    double fixed_slippage_bps{2.0};         // 2 basis points fixed slippage
    double spread_bps{3.0};                 // 3 basis points total bid-ask spread
    double market_impact_factor{0.1};       // Market impact factor: eta * (qty / volume)^0.5
    bool enable_market_impact{true};
};

class ExecutionModel {
public:
    explicit ExecutionModel(ExecutionConfig config = ExecutionConfig{}) : config_(config) {}

    /**
     * @brief Simulates order execution against a market bar with realistic transaction costs.
     * @param order The order to execute.
     * @param bar Current market OHLCV bar.
     * @return Generated Fill.
     */
    [[nodiscard]] Fill execute_order(const Order& order, const quant::data::Bar& bar) const {
        return execute_order(order, bar, bar.close);
    }

    /**
     * @brief Simulates execution at an explicit reference price (e.g. the bar open for next-bar fills).
     */
    [[nodiscard]] Fill execute_order(const Order& order,
                                     const quant::data::Bar& bar,
                                     double reference_price) const {
        Fill fill;
        fill.order_id = order.order_id;
        fill.ticker = order.ticker;
        fill.side = order.side;
        fill.quantity = order.quantity;
        fill.price = reference_price;
        fill.date = bar.date;
        fill.timestamp = bar.timestamp;

        // 1. Bid-Ask Spread Cost (Half-spread)
        double half_spread_pct = (config_.spread_bps * 0.0001) * 0.5;
        double half_spread_dollar = fill.price * half_spread_pct;
        fill.spread_cost = half_spread_dollar * fill.quantity;

        // 2. Slippage Simulation
        double slippage_pct = (config_.fixed_slippage_bps * 0.0001);
        if (config_.enable_market_impact && bar.volume > 0.0) {
            double participation_rate = fill.quantity / bar.volume;
            // Square-root market impact law: Delta P / P = eta * sqrt(Q / V)
            double impact = config_.market_impact_factor * std::sqrt(std::max(0.0, participation_rate));
            slippage_pct += impact;
        }
        double slippage_dollar = fill.price * slippage_pct;
        fill.slippage = slippage_dollar;

        // 3. Execution Price
        if (order.side == OrderSide::BUY) {
            fill.execution_price = fill.price + half_spread_dollar + slippage_dollar;
        } else {
            fill.execution_price = fill.price - half_spread_dollar - slippage_dollar;
            if (fill.execution_price < 0.0001)
                fill.execution_price = 0.0001;   // Guard against negative prices
        }

        // 4. Commission Calculation
        double share_commission = fill.quantity * config_.per_share_commission;
        double pct_commission = (fill.quantity * fill.execution_price) * config_.percentage_commission;
        double total_comm = std::max(config_.min_commission, share_commission + pct_commission);
        fill.commission = total_comm;

        return fill;
    }

    [[nodiscard]] const ExecutionConfig& get_config() const noexcept { return config_; }
    void set_config(const ExecutionConfig& cfg) noexcept { config_ = cfg; }

private:
    ExecutionConfig config_;
};

}   // namespace quant::backtest
