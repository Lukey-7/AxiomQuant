#pragma once

#include "quant/backtest/position.hpp"
#include "quant/backtest/order.hpp"
#include "quant/data/types.hpp"
#include "quant/data/sqlite_storage.hpp"
#include <unordered_map>
#include <vector>
#include <string>

namespace quant::backtest {

class Portfolio {
public:
    explicit Portfolio(double initial_cash = 100000.0);

    void reset(double initial_cash = 100000.0);

    /**
     * @brief Process an executed Fill against the portfolio.
     * @return Net PnL realized by the fill (0 for fills that open or add to a position).
     *         The stored trade-history copy carries realized_pnl and closes_position.
     */
    double process_fill(const Fill& fill);

    /**
     * @brief Mark all positions to market using current snapshot prices and record equity history.
     */
    void mark_to_market(const quant::data::MarketSnapshot& snapshot);

    [[nodiscard]] double get_initial_cash() const noexcept { return initial_cash_; }
    [[nodiscard]] double get_cash() const noexcept { return cash_; }
    [[nodiscard]] double get_total_equity() const noexcept { return current_equity_; }
    [[nodiscard]] double get_peak_equity() const noexcept { return peak_equity_; }
    [[nodiscard]] double get_current_drawdown() const noexcept;

    [[nodiscard]] double get_positions_market_value(const quant::data::MarketSnapshot& snapshot) const;
    [[nodiscard]] double get_position_quantity(const std::string& ticker) const;
    [[nodiscard]] const Position& get_position(const std::string& ticker) const;
    [[nodiscard]] bool has_open_position(const std::string& ticker) const;

    [[nodiscard]] const std::unordered_map<std::string, Position>& get_positions() const noexcept { return positions_; }
    [[nodiscard]] const std::vector<Fill>& get_trade_history() const noexcept { return trade_history_; }
    [[nodiscard]] const std::vector<quant::data::EquityPointRecord>& get_equity_curve() const noexcept { return equity_curve_; }

    /**
     * @brief Calculates simple return series from equity curve.
     */
    [[nodiscard]] std::vector<double> get_daily_returns() const;

private:
    double initial_cash_{100000.0};
    double cash_{100000.0};
    double current_equity_{100000.0};
    double peak_equity_{100000.0};

    std::unordered_map<std::string, Position> positions_;
    std::vector<Fill> trade_history_;
    std::vector<quant::data::EquityPointRecord> equity_curve_;

    static const Position empty_position_;
};

} // namespace quant::backtest
