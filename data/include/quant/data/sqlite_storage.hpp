#pragma once

#include "quant/data/types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

// Forward declaration of sqlite3 struct
struct sqlite3;

namespace quant::data {

struct BacktestRunRecord {
    std::string run_id;
    std::string strategy_name;
    std::string start_date;
    std::string end_date;
    double initial_cash{0.0};
    double final_equity{0.0};
    double total_return{0.0};
    double cagr{0.0};
    double sharpe_ratio{0.0};
    double sortino_ratio{0.0};
    double max_drawdown{0.0};
    double calmar_ratio{0.0};
    double var_95{0.0};
    double cvar_95{0.0};
    int total_trades{0};
    double win_rate{0.0};
    double profit_factor{0.0};
    std::string created_at;
};

struct TradeRecord {
    int64_t trade_id{0};
    std::string run_id;
    std::string ticker;
    std::string date;
    int64_t timestamp{0};
    std::string side;   // "BUY" or "SELL"
    double quantity{0.0};
    double price{0.0};
    double commission{0.0};
    double slippage{0.0};
    double realized_pnl{0.0};
};

struct EquityPointRecord {
    std::string run_id;
    std::string date;
    int64_t timestamp{0};
    double equity{0.0};
    double cash{0.0};
    double positions_value{0.0};
    double drawdown{0.0};
};

class SqliteStorage {
public:
    explicit SqliteStorage(const std::filesystem::path& db_path);
    ~SqliteStorage();

    // Disable copy, allow move
    SqliteStorage(const SqliteStorage&) = delete;
    SqliteStorage& operator=(const SqliteStorage&) = delete;
    SqliteStorage(SqliteStorage&& other) noexcept;
    SqliteStorage& operator=(SqliteStorage&& other) noexcept;

    /**
     * @brief Initialize tables if not already present.
     */
    void init_tables();

    /**
     * @brief Store or upsert a TimeSeries for a ticker into `market_data`.
     */
    void save_market_data(const std::string& ticker, const TimeSeries& series);

    /**
     * @brief Load a TimeSeries from SQLite for a ticker.
     */
    [[nodiscard]] TimeSeries load_market_data(const std::string& ticker) const;

    /**
     * @brief List all distinct tickers in `market_data`.
     */
    [[nodiscard]] std::vector<std::string> get_stored_tickers() const;

    /**
     * @brief Save backtest run summary.
     */
    void save_backtest_run(const BacktestRunRecord& record);

    /**
     * @brief Save executed trades.
     */
    void save_trades(const std::vector<TradeRecord>& trades);

    /**
     * @brief Save time-series equity curve.
     */
    void save_equity_curve(const std::vector<EquityPointRecord>& equity_curve);

    /**
     * @brief Load backtest run record by ID.
     */
    [[nodiscard]] BacktestRunRecord load_backtest_run(const std::string& run_id) const;

    /**
     * @brief Load all backtest run records.
     */
    [[nodiscard]] std::vector<BacktestRunRecord> load_all_backtest_runs() const;

    /**
     * @brief Load trades for a specific run ID.
     */
    [[nodiscard]] std::vector<TradeRecord> load_trades(const std::string& run_id) const;

    /**
     * @brief Load equity curve for a specific run ID.
     */
    [[nodiscard]] std::vector<EquityPointRecord> load_equity_curve(const std::string& run_id) const;

private:
    sqlite3* db_{nullptr};
    std::string db_path_;

    void execute_sql(const std::string& sql);
};

}   // namespace quant::data
