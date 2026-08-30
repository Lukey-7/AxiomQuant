#pragma once

#include "quant/backtest/portfolio.hpp"
#include "quant/backtest/execution_model.hpp"
#include "quant/backtest/order.hpp"
#include "quant/data/types.hpp"
#include "quant/data/universe.hpp"
#include <string>
#include <vector>

namespace quant::backtest {

class Strategy {
public:
    virtual ~Strategy() = default;

    [[nodiscard]] virtual std::string get_name() const = 0;

    /**
     * @brief Called once before the backtest begins.
     */
    virtual void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& /*universe*/) {}

    /**
     * @brief Called at each time step / market snapshot.
     * @param timeline_index Current index into universe timeline.
     * @param snapshot Current market bar snapshot across assets.
     * @param portfolio Current portfolio state.
     * @param pending_orders Queue where the strategy pushes new orders.
     */
    virtual void on_bar(
        size_t timeline_index,
        const quant::data::MarketSnapshot& snapshot,
        const Portfolio& portfolio,
        std::vector<Order>& pending_orders
    ) = 0;

    /**
     * @brief Called whenever an order is filled.
     */
    virtual void on_order_fill(const Fill& /*fill*/) {}

    /**
     * @brief Called after the backtest completes all bars.
     */
    virtual void on_end(Portfolio& /*portfolio*/) {}
};

} // namespace quant::backtest
