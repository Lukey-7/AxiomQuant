#pragma once

#include "quant/backtest/order.hpp"
#include <string>
#include <cmath>

namespace quant::backtest {

class Position {
public:
    std::string ticker;
    double quantity{0.0};
    double average_price{0.0};
    double realized_pnl{0.0};
    double total_commission_paid{0.0};

    Position() = default;
    explicit Position(std::string symbol) : ticker(std::move(symbol)) {}

    [[nodiscard]] bool is_open() const noexcept { return std::abs(quantity) > 1e-7; }

    [[nodiscard]] bool is_long() const noexcept { return quantity > 1e-7; }

    [[nodiscard]] bool is_short() const noexcept { return quantity < -1e-7; }

    [[nodiscard]] double market_value(double current_price) const noexcept {
        return quantity * current_price;
    }

    [[nodiscard]] double unrealized_pnl(double current_price) const noexcept {
        if (!is_open()) return 0.0;
        return quantity * (current_price - average_price);
    }

    /**
     * @brief Process an executed fill, updating position size, average price, and realized PnL.
     * @param fill Executed Fill.
     * @return Realized PnL from this fill.
     */
    double update_with_fill(const Fill& fill) {
        total_commission_paid += fill.commission;
        double pnl = 0.0;
        double fill_qty = (fill.side == OrderSide::BUY) ? fill.quantity : -fill.quantity;
        double fill_exec_price = fill.execution_price;

        if (!is_open()) {
            quantity = fill_qty;
            average_price = fill_exec_price;
        } else if ((quantity > 0 && fill_qty > 0) || (quantity < 0 && fill_qty < 0)) {
            // Increasing existing position
            double total_cost = (quantity * average_price) + (fill_qty * fill_exec_price);
            quantity += fill_qty;
            average_price = total_cost / quantity;
        } else {
            // Reducing or closing / flipping existing position
            double closing_qty = std::min(std::abs(quantity), std::abs(fill_qty));
            if (quantity > 0) {
                // Closing long: sell price - avg buy price
                pnl = closing_qty * (fill_exec_price - average_price);
            } else {
                // Closing short: avg short price - buy to cover price
                pnl = closing_qty * (average_price - fill_exec_price);
            }
            pnl -= fill.commission;   // Deduct commission from trade PnL
            realized_pnl += pnl;

            double remaining_qty = quantity + fill_qty;
            if (std::abs(remaining_qty) < 1e-7) {
                quantity = 0.0;
                average_price = 0.0;
            } else if ((quantity > 0 && remaining_qty < 0) || (quantity < 0 && remaining_qty > 0)) {
                // Position flipped direction
                quantity = remaining_qty;
                average_price = fill_exec_price;
            } else {
                quantity = remaining_qty;
            }
        }

        return pnl;
    }
};

}   // namespace quant::backtest
