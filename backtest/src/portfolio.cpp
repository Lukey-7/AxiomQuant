#include "quant/backtest/portfolio.hpp"
#include <algorithm>
#include <cmath>

namespace quant::backtest {

const Position Portfolio::empty_position_{""};

Portfolio::Portfolio(double initial_cash)
    : initial_cash_(initial_cash),
      cash_(initial_cash),
      current_equity_(initial_cash),
      peak_equity_(initial_cash) {}

void Portfolio::reset(double initial_cash) {
    initial_cash_ = initial_cash;
    cash_ = initial_cash;
    current_equity_ = initial_cash;
    peak_equity_ = initial_cash;
    positions_.clear();
    trade_history_.clear();
    equity_curve_.clear();
}

double Portfolio::process_fill(const Fill& input_fill) {
    Fill fill = input_fill;
    if (positions_.find(fill.ticker) == positions_.end()) {
        positions_[fill.ticker] = Position(fill.ticker);
    }

    auto& pos = positions_[fill.ticker];
    const bool is_buy = (fill.side == OrderSide::BUY);
    fill.closes_position = pos.is_open() && (pos.is_long() != is_buy);
    fill.realized_pnl = pos.update_with_fill(fill);

    // Cash flow adjustments
    if (fill.side == OrderSide::BUY) {
        cash_ -= (fill.quantity * fill.execution_price) + fill.commission;
    } else {
        cash_ += (fill.quantity * fill.execution_price) - fill.commission;
    }

    trade_history_.push_back(fill);
    return fill.realized_pnl;
}

void Portfolio::mark_to_market(const quant::data::MarketSnapshot& snapshot) {
    double pos_value = 0.0;
    for (const auto& [ticker, pos] : positions_) {
        if (pos.is_open() && snapshot.has_ticker(ticker)) {
            double price = snapshot.get_bar(ticker).close;
            pos_value += pos.market_value(price);
        }
    }

    current_equity_ = cash_ + pos_value;
    if (current_equity_ > peak_equity_) {
        peak_equity_ = current_equity_;
    }

    double dd = get_current_drawdown();

    quant::data::EquityPointRecord point;
    point.date = snapshot.date;
    point.timestamp = snapshot.timestamp;
    point.equity = current_equity_;
    point.cash = cash_;
    point.positions_value = pos_value;
    point.drawdown = dd;

    equity_curve_.push_back(point);
}

double Portfolio::get_current_drawdown() const noexcept {
    if (peak_equity_ <= 0.0) return 0.0;
    return (peak_equity_ - current_equity_) / peak_equity_;
}

double Portfolio::get_positions_market_value(const quant::data::MarketSnapshot& snapshot) const {
    double total = 0.0;
    for (const auto& [ticker, pos] : positions_) {
        if (pos.is_open() && snapshot.has_ticker(ticker)) {
            total += pos.market_value(snapshot.get_bar(ticker).close);
        }
    }
    return total;
}

double Portfolio::get_position_quantity(const std::string& ticker) const {
    auto it = positions_.find(ticker);
    if (it != positions_.end()) {
        return it->second.quantity;
    }
    return 0.0;
}

const Position& Portfolio::get_position(const std::string& ticker) const {
    auto it = positions_.find(ticker);
    if (it != positions_.end()) {
        return it->second;
    }
    return empty_position_;
}

bool Portfolio::has_open_position(const std::string& ticker) const {
    auto it = positions_.find(ticker);
    return it != positions_.end() && it->second.is_open();
}

std::vector<double> Portfolio::get_daily_returns() const {
    if (equity_curve_.size() < 2) return {};
    std::vector<double> rets(equity_curve_.size() - 1);
    for (size_t i = 1; i < equity_curve_.size(); ++i) {
        double prev = equity_curve_[i - 1].equity;
        double curr = equity_curve_[i].equity;
        rets[i - 1] = (prev > 0.0) ? ((curr - prev) / prev) : 0.0;
    }
    return rets;
}

}   // namespace quant::backtest
