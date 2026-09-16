#include "quant/data/sqlite_storage.hpp"
#include "sqlite3.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace quant::data {

SqliteStorage::SqliteStorage(const std::filesystem::path& db_path) : db_path_(db_path.string()) {
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string err_msg = db_ ? sqlite3_errmsg(db_) : "Unknown error";
        if (db_) sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("Failed to open SQLite database at " + db_path_ + ": " + err_msg);
    }
    init_tables();
}

SqliteStorage::~SqliteStorage() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

SqliteStorage::SqliteStorage(SqliteStorage&& other) noexcept
    : db_(other.db_), db_path_(std::move(other.db_path_)) {
    other.db_ = nullptr;
}

SqliteStorage& SqliteStorage::operator=(SqliteStorage&& other) noexcept {
    if (this != &other) {
        if (db_) sqlite3_close(db_);
        db_ = other.db_;
        db_path_ = std::move(other.db_path_);
        other.db_ = nullptr;
    }
    return *this;
}

void SqliteStorage::execute_sql(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string err = err_msg ? err_msg : "Unknown error";
        sqlite3_free(err_msg);
        throw std::runtime_error("SQLite error executing: " + sql + " -> " + err);
    }
}

void SqliteStorage::init_tables() {
    execute_sql("PRAGMA journal_mode = WAL;");
    execute_sql("PRAGMA synchronous = NORMAL;");

    const std::string schema = R"(
        CREATE TABLE IF NOT EXISTS market_data (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            adj_close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (ticker, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date);

        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_cash REAL NOT NULL,
            final_equity REAL NOT NULL,
            total_return REAL NOT NULL,
            cagr REAL NOT NULL,
            sharpe_ratio REAL NOT NULL,
            sortino_ratio REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            calmar_ratio REAL NOT NULL,
            var_95 REAL NOT NULL,
            cvar_95 REAL NOT NULL,
            total_trades INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            profit_factor REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            commission REAL NOT NULL,
            slippage REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);

        CREATE TABLE IF NOT EXISTS equity_curve (
            point_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            date TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            drawdown REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_equity_curve_run_id ON equity_curve(run_id);
    )";

    execute_sql(schema);
}

void SqliteStorage::save_market_data(const std::string& ticker, const TimeSeries& series) {
    if (series.empty()) return;

    execute_sql("BEGIN TRANSACTION;");

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT OR REPLACE INTO market_data (ticker, date, timestamp, open, high, low, close, adj_close, "
        "volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        execute_sql("ROLLBACK;");
        throw std::runtime_error("Failed to prepare insert statement for market_data");
    }

    for (size_t i = 0; i < series.size(); ++i) {
        sqlite3_bind_text(stmt, 1, ticker.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, series.dates[i].c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 3, series.timestamps[i]);
        sqlite3_bind_double(stmt, 4, series.open[i]);
        sqlite3_bind_double(stmt, 5, series.high[i]);
        sqlite3_bind_double(stmt, 6, series.low[i]);
        sqlite3_bind_double(stmt, 7, series.close[i]);
        sqlite3_bind_double(stmt, 8, series.adj_close[i]);
        sqlite3_bind_double(stmt, 9, series.volume[i]);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            execute_sql("ROLLBACK;");
            throw std::runtime_error("Failed to execute insert step for market_data");
        }
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    execute_sql("COMMIT;");
}

TimeSeries SqliteStorage::load_market_data(const std::string& ticker) const {
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT date, timestamp, open, high, low, close, adj_close, volume "
        "FROM market_data WHERE ticker = ? ORDER BY timestamp ASC;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare select statement for market_data");
    }

    sqlite3_bind_text(stmt, 1, ticker.c_str(), -1, SQLITE_TRANSIENT);

    TimeSeries ts(ticker);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Bar bar;
        bar.date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        bar.timestamp = sqlite3_column_int64(stmt, 1);
        bar.open = sqlite3_column_double(stmt, 2);
        bar.high = sqlite3_column_double(stmt, 3);
        bar.low = sqlite3_column_double(stmt, 4);
        bar.close = sqlite3_column_double(stmt, 5);
        bar.adj_close = sqlite3_column_double(stmt, 6);
        bar.volume = sqlite3_column_double(stmt, 7);

        ts.push_back(bar);
    }

    sqlite3_finalize(stmt);
    return ts;
}

std::vector<std::string> SqliteStorage::get_stored_tickers() const {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT DISTINCT ticker FROM market_data ORDER BY ticker ASC;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to query distinct tickers");
    }

    std::vector<std::string> tickers;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        tickers.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }

    sqlite3_finalize(stmt);
    return tickers;
}

void SqliteStorage::save_backtest_run(const BacktestRunRecord& r) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT OR REPLACE INTO backtest_runs ("
        "run_id, strategy_name, start_date, end_date, initial_cash, final_equity, "
        "total_return, cagr, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, "
        "var_95, cvar_95, total_trades, win_rate, profit_factor, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement for backtest_runs");
    }

    sqlite3_bind_text(stmt, 1, r.run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, r.strategy_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, r.start_date.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, r.end_date.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 5, r.initial_cash);
    sqlite3_bind_double(stmt, 6, r.final_equity);
    sqlite3_bind_double(stmt, 7, r.total_return);
    sqlite3_bind_double(stmt, 8, r.cagr);
    sqlite3_bind_double(stmt, 9, r.sharpe_ratio);
    sqlite3_bind_double(stmt, 10, r.sortino_ratio);
    sqlite3_bind_double(stmt, 11, r.max_drawdown);
    sqlite3_bind_double(stmt, 12, r.calmar_ratio);
    sqlite3_bind_double(stmt, 13, r.var_95);
    sqlite3_bind_double(stmt, 14, r.cvar_95);
    sqlite3_bind_int(stmt, 15, r.total_trades);
    sqlite3_bind_double(stmt, 16, r.win_rate);
    sqlite3_bind_double(stmt, 17, r.profit_factor);
    sqlite3_bind_text(stmt, 18, r.created_at.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to insert backtest run");
    }
    sqlite3_finalize(stmt);
}

void SqliteStorage::save_trades(const std::vector<TradeRecord>& trades) {
    if (trades.empty()) return;

    execute_sql("BEGIN TRANSACTION;");
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO trades (run_id, ticker, date, timestamp, side, quantity, price, commission, slippage, "
        "realized_pnl) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        execute_sql("ROLLBACK;");
        throw std::runtime_error("Failed to prepare statement for trades");
    }

    for (const auto& t : trades) {
        sqlite3_bind_text(stmt, 1, t.run_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, t.ticker.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, t.date.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 4, t.timestamp);
        sqlite3_bind_text(stmt, 5, t.side.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 6, t.quantity);
        sqlite3_bind_double(stmt, 7, t.price);
        sqlite3_bind_double(stmt, 8, t.commission);
        sqlite3_bind_double(stmt, 9, t.slippage);
        sqlite3_bind_double(stmt, 10, t.realized_pnl);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            execute_sql("ROLLBACK;");
            throw std::runtime_error("Failed to execute insert step for trade");
        }
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    execute_sql("COMMIT;");
}

void SqliteStorage::save_equity_curve(const std::vector<EquityPointRecord>& equity_curve) {
    if (equity_curve.empty()) return;

    execute_sql("BEGIN TRANSACTION;");
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO equity_curve (run_id, date, timestamp, equity, cash, positions_value, drawdown) "
        "VALUES (?, ?, ?, ?, ?, ?, ?);";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        execute_sql("ROLLBACK;");
        throw std::runtime_error("Failed to prepare statement for equity_curve");
    }

    for (const auto& pt : equity_curve) {
        sqlite3_bind_text(stmt, 1, pt.run_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, pt.date.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 3, pt.timestamp);
        sqlite3_bind_double(stmt, 4, pt.equity);
        sqlite3_bind_double(stmt, 5, pt.cash);
        sqlite3_bind_double(stmt, 6, pt.positions_value);
        sqlite3_bind_double(stmt, 7, pt.drawdown);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            execute_sql("ROLLBACK;");
            throw std::runtime_error("Failed to execute insert step for equity curve");
        }
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    execute_sql("COMMIT;");
}

std::vector<BacktestRunRecord> SqliteStorage::load_all_backtest_runs() const {
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT run_id, strategy_name, start_date, end_date, initial_cash, final_equity, "
        "total_return, cagr, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, "
        "var_95, cvar_95, total_trades, win_rate, profit_factor, created_at "
        "FROM backtest_runs ORDER BY created_at DESC;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to query backtest_runs");
    }

    std::vector<BacktestRunRecord> runs;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        BacktestRunRecord r;
        r.run_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        r.strategy_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.start_date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.end_date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        r.initial_cash = sqlite3_column_double(stmt, 4);
        r.final_equity = sqlite3_column_double(stmt, 5);
        r.total_return = sqlite3_column_double(stmt, 6);
        r.cagr = sqlite3_column_double(stmt, 7);
        r.sharpe_ratio = sqlite3_column_double(stmt, 8);
        r.sortino_ratio = sqlite3_column_double(stmt, 9);
        r.max_drawdown = sqlite3_column_double(stmt, 10);
        r.calmar_ratio = sqlite3_column_double(stmt, 11);
        r.var_95 = sqlite3_column_double(stmt, 12);
        r.cvar_95 = sqlite3_column_double(stmt, 13);
        r.total_trades = sqlite3_column_int(stmt, 14);
        r.win_rate = sqlite3_column_double(stmt, 15);
        r.profit_factor = sqlite3_column_double(stmt, 16);
        r.created_at = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 17));

        runs.push_back(r);
    }

    sqlite3_finalize(stmt);
    return runs;
}

std::vector<TradeRecord> SqliteStorage::load_trades(const std::string& run_id) const {
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT trade_id, run_id, ticker, date, timestamp, side, quantity, price, commission, slippage, "
        "realized_pnl "
        "FROM trades WHERE run_id = ? ORDER BY timestamp ASC, trade_id ASC;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to query trades");
    }

    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);

    std::vector<TradeRecord> trades;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        TradeRecord t;
        t.trade_id = sqlite3_column_int64(stmt, 0);
        t.run_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        t.ticker = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        t.date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        t.timestamp = sqlite3_column_int64(stmt, 4);
        t.side = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        t.quantity = sqlite3_column_double(stmt, 6);
        t.price = sqlite3_column_double(stmt, 7);
        t.commission = sqlite3_column_double(stmt, 8);
        t.slippage = sqlite3_column_double(stmt, 9);
        t.realized_pnl = sqlite3_column_double(stmt, 10);

        trades.push_back(t);
    }

    sqlite3_finalize(stmt);
    return trades;
}

std::vector<EquityPointRecord> SqliteStorage::load_equity_curve(const std::string& run_id) const {
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT run_id, date, timestamp, equity, cash, positions_value, drawdown "
        "FROM equity_curve WHERE run_id = ? ORDER BY timestamp ASC;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to query equity curve");
    }

    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);

    std::vector<EquityPointRecord> curve;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        EquityPointRecord pt;
        pt.run_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        pt.date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        pt.timestamp = sqlite3_column_int64(stmt, 2);
        pt.equity = sqlite3_column_double(stmt, 3);
        pt.cash = sqlite3_column_double(stmt, 4);
        pt.positions_value = sqlite3_column_double(stmt, 5);
        pt.drawdown = sqlite3_column_double(stmt, 6);

        curve.push_back(pt);
    }

    sqlite3_finalize(stmt);
    return curve;
}

}   // namespace quant::data
