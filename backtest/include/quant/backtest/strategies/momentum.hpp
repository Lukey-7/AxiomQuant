#pragma once

#include "quant/backtest/strategy.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace quant::backtest::strategies {

/**
 * @brief Cross-sectional momentum with an optional absolute-momentum (trend) filter.
 *
 * Every `rebalance_frequency` bars after the warm-up window, assets are ranked by their trailing
 * `lookback_period` total return  R_i = P_t / P_{t-L} - 1. The top-K are held in equal dollar
 * weight and every other position is liquidated. With `require_positive_momentum`, assets whose
 * trailing return is not positive are excluded and the unused capital stays in cash
 * (a "dual momentum" overlay).
 */
class MultiAssetMomentumStrategy : public Strategy {
public:
    struct RankedAsset {
        std::string ticker;
        double trailing_return{0.0};
    };

    MultiAssetMomentumStrategy(size_t lookback_period = 60,
                               size_t rebalance_frequency = 20,
                               size_t top_k = 2,
                               double target_invested_pct = 0.95,
                               bool require_positive_momentum = false)
        : lookback_period_(lookback_period),
          rebalance_frequency_(rebalance_frequency),
          top_k_(top_k),
          target_invested_pct_(target_invested_pct),
          require_positive_momentum_(require_positive_momentum) {}

    [[nodiscard]] std::string get_name() const override {
        return std::string(require_positive_momentum_ ? "DualMomentum" : "CrossSectionalMomentum") +
               "(L=" + std::to_string(lookback_period_) + ", R=" + std::to_string(rebalance_frequency_) +
               ", Top=" + std::to_string(top_k_) + ")";
    }

    void on_start(Portfolio& /*portfolio*/, const quant::data::MarketDataUniverse& universe) override {
        tickers_ = universe.get_tickers();
        closes_.clear();
        for (const auto& ticker : tickers_) {
            closes_[ticker] = universe.get_aligned_closes(ticker);
        }
        last_ranking_.clear();
    }

    void on_bar(size_t timeline_index,
                const quant::data::MarketSnapshot& snapshot,
                const Portfolio& portfolio,
                std::vector<Order>& pending_orders) override {
        if (rebalance_frequency_ == 0 || top_k_ == 0 || timeline_index < lookback_period_) return;
        if ((timeline_index - lookback_period_) % rebalance_frequency_ != 0) return;

        // 1. Rank the universe by trailing return: R_i = P_t / P_{t-L} - 1
        std::vector<RankedAsset> ranked;
        for (const auto& ticker : tickers_) {
            const auto& c = closes_[ticker];
            if (timeline_index >= c.size() || !snapshot.has_ticker(ticker)) continue;
            const double past = c[timeline_index - lookback_period_];
            const double now = c[timeline_index];
            if (!(past > 0.0) || !(now > 0.0)) continue;
            const double r = now / past - 1.0;
            if (require_positive_momentum_ && r <= 0.0) continue;
            ranked.push_back({ticker, r});
        }
        std::sort(ranked.begin(), ranked.end(), [](const RankedAsset& a, const RankedAsset& b) {
            return a.trailing_return != b.trailing_return ? a.trailing_return > b.trailing_return
                                                          : a.ticker < b.ticker;
        });
        last_ranking_ = ranked;

        const size_t selected_count = std::min(top_k_, ranked.size());
        std::unordered_set<std::string> selected;
        for (size_t i = 0; i < selected_count; ++i)
            selected.insert(ranked[i].ticker);

        // 2. Liquidate every open position that fell out of the selection.
        for (const auto& [ticker, pos] : portfolio.get_positions()) {
            if (!pos.is_open() || selected.count(ticker)) continue;
            Order exit;
            exit.ticker = ticker;
            exit.side = pos.is_long() ? OrderSide::SELL : OrderSide::BUY;
            exit.type = OrderType::MARKET;
            exit.quantity = std::abs(pos.quantity);
            pending_orders.push_back(exit);
        }

        if (selected_count == 0) return;

        // 3. Rebalance the winners to equal dollar weight.
        const double target_dollars =
            portfolio.get_total_equity() * target_invested_pct_ / static_cast<double>(selected_count);
        for (size_t i = 0; i < selected_count; ++i) {
            const auto& ticker = ranked[i].ticker;
            const double price = snapshot.get_bar(ticker).close;
            const double target_qty = std::floor(target_dollars / price);
            const double diff = target_qty - portfolio.get_position_quantity(ticker);
            if (std::abs(diff) < 1.0) continue;

            Order order;
            order.ticker = ticker;
            order.side = diff > 0.0 ? OrderSide::BUY : OrderSide::SELL;
            order.type = OrderType::MARKET;
            order.quantity = std::abs(diff);
            pending_orders.push_back(order);
        }
    }

    /**
     * @brief Ranking computed at the most recent rebalance (best first).
     */
    [[nodiscard]] const std::vector<RankedAsset>& last_ranking() const noexcept { return last_ranking_; }

private:
    size_t lookback_period_;
    size_t rebalance_frequency_;
    size_t top_k_;
    double target_invested_pct_;
    bool require_positive_momentum_;
    std::vector<std::string> tickers_;
    std::unordered_map<std::string, std::vector<double>> closes_;
    std::vector<RankedAsset> last_ranking_;
};

}   // namespace quant::backtest::strategies
