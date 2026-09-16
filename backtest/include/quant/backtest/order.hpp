#pragma once

#include <string>
#include <cstdint>

namespace quant::backtest {

enum class OrderType : std::uint8_t { MARKET, LIMIT };

enum class OrderSide : std::uint8_t { BUY, SELL };

[[nodiscard]] inline std::string side_to_string(OrderSide side) {
    return side == OrderSide::BUY ? "BUY" : "SELL";
}

struct Order {
    int64_t order_id{0};
    std::string ticker;
    OrderSide side{OrderSide::BUY};
    OrderType type{OrderType::MARKET};
    double quantity{0.0};      // Number of shares
    double limit_price{0.0};   // Relevant for limit orders
    std::string date;
    int64_t timestamp{0};
};

struct Fill {
    int64_t fill_id{0};
    int64_t order_id{0};
    std::string ticker;
    OrderSide side{OrderSide::BUY};
    double quantity{0.0};
    double price{0.0};             // Raw bar price executed at
    double execution_price{0.0};   // Price after slippage and half-spread
    double commission{0.0};
    double slippage{0.0};
    double spread_cost{0.0};
    double realized_pnl{0.0};      // Net PnL realized by this fill (populated by Portfolio)
    bool closes_position{false};   // True if this fill reduced / closed an existing position
    std::string date;
    int64_t timestamp{0};

    [[nodiscard]] double total_transaction_cost() const noexcept {
        return commission + (slippage * quantity) + spread_cost;
    }
};

}   // namespace quant::backtest
