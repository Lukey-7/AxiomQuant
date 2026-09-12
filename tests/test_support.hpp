#pragma once

#include "quant/backtest/execution_model.hpp"
#include "quant/backtest/strategy.hpp"
#include "quant/data/universe.hpp"
#include <algorithm>
#include <ctime>
#include <string>
#include <vector>

namespace quant::tests {

/**
 * @brief Synthetic calendar helpers so tests can build universes with real ISO dates.
 */
inline std::string synthetic_date(size_t index) {
    std::time_t stamp = static_cast<std::time_t>(1672531200LL + static_cast<int64_t>(index) * 86400LL);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &stamp);
#else
    gmtime_r(&stamp, &tm);
#endif
    char buffer[16];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &tm);
    return buffer;
}

inline int64_t synthetic_timestamp(size_t index) {
    return 1672531200LL + static_cast<int64_t>(index) * 86400LL;
}

struct SyntheticAsset {
    std::string ticker;
    std::vector<double> close;
    std::vector<double> open; // empty => open == close
};

/**
 * @brief Builds a synchronized universe from explicit price paths.
 */
inline data::MarketDataUniverse make_universe(const std::vector<SyntheticAsset>& assets) {
    data::MarketDataUniverse universe;
    for (const auto& asset : assets) {
        data::TimeSeries series(asset.ticker);
        for (size_t i = 0; i < asset.close.size(); ++i) {
            data::Bar bar;
            bar.date = synthetic_date(i);
            bar.timestamp = synthetic_timestamp(i);
            bar.close = asset.close[i];
            bar.open = asset.open.empty() ? asset.close[i] : asset.open[i];
            bar.high = std::max(bar.open, bar.close) * 1.01;
            bar.low = std::min(bar.open, bar.close) * 0.99;
            bar.adj_close = bar.close;
            bar.volume = 1.0e6;
            series.push_back(bar);
        }
        universe.add_asset(asset.ticker, series);
    }
    universe.synchronize_timeline(true);
    return universe;
}

/**
 * @brief Cost-free execution so tests can assert exact prices and quantities.
 */
inline backtest::ExecutionConfig frictionless_execution() {
    backtest::ExecutionConfig cfg;
    cfg.per_share_commission = 0.0;
    cfg.min_commission = 0.0;
    cfg.percentage_commission = 0.0;
    cfg.fixed_slippage_bps = 0.0;
    cfg.spread_bps = 0.0;
    cfg.market_impact_factor = 0.0;
    cfg.enable_market_impact = false;
    return cfg;
}

/**
 * @brief Emits a fixed script of orders at chosen bars, for deterministic engine tests.
 */
class ScriptedStrategy : public backtest::Strategy {
public:
    struct ScriptedOrder {
        size_t bar{0};
        std::string ticker;
        backtest::OrderSide side{backtest::OrderSide::BUY};
        double quantity{0.0};
    };

    explicit ScriptedStrategy(std::vector<ScriptedOrder> script) : script_(std::move(script)) {}

    [[nodiscard]] std::string get_name() const override { return "Scripted"; }

    void on_bar(
        size_t timeline_index,
        const data::MarketSnapshot& /*snapshot*/,
        const backtest::Portfolio& /*portfolio*/,
        std::vector<backtest::Order>& pending_orders
    ) override {
        for (const auto& entry : script_) {
            if (entry.bar != timeline_index) continue;
            backtest::Order order;
            order.ticker = entry.ticker;
            order.side = entry.side;
            order.type = backtest::OrderType::MARKET;
            order.quantity = entry.quantity;
            pending_orders.push_back(order);
        }
    }

private:
    std::vector<ScriptedOrder> script_;
};

} // namespace quant::tests
