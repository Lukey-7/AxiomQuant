#include "test_harness.hpp"
#include <string>
#include "quant/data/types.hpp"
#include "quant/data/csv_loader.hpp"
#include "quant/data/universe.hpp"
#include "quant/data/sqlite_storage.hpp"
#include <filesystem>

TEST_CASE(TestData_TimeSeries_Returns) {
    quant::data::TimeSeries ts("TEST");
    ts.push_back({"2023-01-01", 1000, 100.0, 105.0, 95.0, 100.0, 100.0, 1000});
    ts.push_back({"2023-01-02", 2000, 100.0, 115.0, 99.0, 110.0, 110.0, 1500});
    ts.push_back({"2023-01-03", 3000, 110.0, 112.0, 98.0, 99.0, 99.0, 1200});

    EXPECT_EQ(ts.size(), 3);
    auto simple_rets = ts.simple_returns();
    EXPECT_EQ(simple_rets.size(), 2);
    EXPECT_NEAR(simple_rets[0], 0.10, 1e-6);          // (110 - 100)/100 = 0.10
    EXPECT_NEAR(simple_rets[1], -0.10, 1e-6);         // (99 - 110)/110 = -0.10

    auto log_rets = ts.log_returns();
    EXPECT_EQ(log_rets.size(), 2);
    EXPECT_NEAR(log_rets[0], std::log(1.10), 1e-6);
}

TEST_CASE(TestData_Universe_Synchronization) {
    quant::data::TimeSeries ts1("AAPL");
    ts1.push_back({"2023-01-01", 1000, 10.0, 11.0, 9.0, 10.0, 10.0, 100});
    ts1.push_back({"2023-01-02", 2000, 10.0, 12.0, 9.0, 11.0, 11.0, 100});
    ts1.push_back({"2023-01-03", 3000, 11.0, 13.0, 10.0, 12.0, 12.0, 100});

    quant::data::TimeSeries ts2("MSFT");
    ts2.push_back({"2023-01-02", 2000, 20.0, 22.0, 19.0, 21.0, 21.0, 200});
    ts2.push_back({"2023-01-03", 3000, 21.0, 24.0, 20.0, 22.0, 22.0, 200});

    quant::data::MarketDataUniverse uni;
    uni.add_asset("AAPL", ts1);
    uni.add_asset("MSFT", ts2);

    // Require all assets -> intersection is {"2023-01-02", "2023-01-03"}
    uni.synchronize_timeline(true);
    EXPECT_EQ(uni.size(), 2);
    EXPECT_EQ(uni.get_timeline()[0], "2023-01-02");
    EXPECT_EQ(uni.get_timeline()[1], "2023-01-03");

    auto snap = uni.get_snapshot(0);
    EXPECT_EQ(snap.date, "2023-01-02");
    EXPECT_NEAR(snap.get_bar("AAPL").close, 11.0, 1e-6);
    EXPECT_NEAR(snap.get_bar("MSFT").close, 21.0, 1e-6);

    auto close_mat = uni.get_aligned_close_matrix();
    EXPECT_EQ(close_mat.size(), 2);
    EXPECT_EQ(close_mat[0].size(), 2);
}

TEST_CASE(TestData_Sqlite_Persistence_Roundtrip) {
    std::filesystem::path test_db = "test_quant.db";
    std::error_code ec;
    std::filesystem::remove(test_db, ec);

    {
        quant::data::SqliteStorage storage(test_db);

        quant::data::TimeSeries ts("GOOGL");
        ts.push_back({"2023-01-01", 1000, 50.0, 55.0, 48.0, 52.0, 52.0, 5000});
        ts.push_back({"2023-01-02", 2000, 52.0, 54.0, 51.0, 53.5, 53.5, 4500});

        storage.save_market_data("GOOGL", ts);

        auto loaded_ts = storage.load_market_data("GOOGL");
        EXPECT_EQ(loaded_ts.size(), 2);
        EXPECT_EQ(loaded_ts.dates[0], "2023-01-01");
        EXPECT_NEAR(loaded_ts.close[0], 52.0, 1e-6);
        EXPECT_NEAR(loaded_ts.close[1], 53.5, 1e-6);

        quant::data::BacktestRunRecord run_rec;
        run_rec.run_id = "TEST_RUN_1";
        run_rec.strategy_name = "TestStrategy";
        run_rec.start_date = "2023-01-01";
        run_rec.end_date = "2023-01-02";
        run_rec.initial_cash = 10000.0;
        run_rec.final_equity = 12000.0;
        run_rec.total_return = 0.20;
        run_rec.cagr = 0.20;
        run_rec.sharpe_ratio = 1.85;
        run_rec.sortino_ratio = 2.40;
        run_rec.max_drawdown = 0.05;
        run_rec.calmar_ratio = 4.0;
        run_rec.var_95 = 0.02;
        run_rec.cvar_95 = 0.03;
        run_rec.total_trades = 4;
        run_rec.win_rate = 0.75;
        run_rec.profit_factor = 2.5;
        run_rec.created_at = "2026-08-30T00:00:00Z";

        storage.save_backtest_run(run_rec);

        auto loaded_runs = storage.load_all_backtest_runs();
        EXPECT_EQ(loaded_runs.size(), 1);
        EXPECT_EQ(loaded_runs[0].run_id, "TEST_RUN_1");
        EXPECT_NEAR(loaded_runs[0].sharpe_ratio, 1.85, 1e-6);
    }

    std::filesystem::remove(test_db, ec);
}

// --- Regression: indicators must be indexed on the synchronized timeline, not the raw series ---
TEST_CASE(TestData_AlignedClosesFollowTheTimeline) {
    quant::data::TimeSeries ts1("AAPL");
    ts1.push_back({"2023-01-01", 1000, 10.0, 11.0, 9.0, 10.0, 10.0, 100});
    ts1.push_back({"2023-01-02", 2000, 10.0, 12.0, 9.0, 11.0, 11.0, 100});
    ts1.push_back({"2023-01-03", 3000, 11.0, 13.0, 10.0, 12.0, 12.0, 100});

    quant::data::TimeSeries ts2("MSFT");
    ts2.push_back({"2023-01-02", 2000, 20.0, 22.0, 19.0, 21.0, 21.0, 200});
    ts2.push_back({"2023-01-03", 3000, 21.0, 24.0, 20.0, 22.0, 22.0, 200});

    quant::data::MarketDataUniverse uni;
    uni.add_asset("AAPL", ts1);
    uni.add_asset("MSFT", ts2);
    uni.synchronize_timeline(true);

    // The raw AAPL series still has three bars, but the timeline only has two.
    EXPECT_EQ(uni.get_series("AAPL").size(), 3);
    const auto closes = uni.get_aligned_closes("AAPL");
    EXPECT_EQ(closes.size(), 2);
    EXPECT_NEAR(closes[0], 11.0, 1e-9);   // 2023-01-02, not the dropped 2023-01-01 bar
    EXPECT_NEAR(closes[1], 12.0, 1e-9);
}

TEST_CASE(TestData_SliceKeepsWindowOfTimeline) {
    quant::data::TimeSeries ts("AAPL");
    for (int i = 0; i < 10; ++i) {
        const std::string date = "2023-01-" + std::string(i + 1 < 10 ? "0" : "") + std::to_string(i + 1);
        ts.push_back({date, 1000 + i, 10.0, 11.0, 9.0, 10.0 + i, 10.0 + i, 100});
    }
    quant::data::MarketDataUniverse uni;
    uni.add_asset("AAPL", ts);
    uni.synchronize_timeline(true);

    const auto window = uni.slice(3, 7);
    EXPECT_EQ(window.size(), 4);
    EXPECT_EQ(window.get_timeline().front(), uni.get_timeline()[3]);
    EXPECT_EQ(window.get_timeline().back(), uni.get_timeline()[6]);
    EXPECT_NEAR(window.get_snapshot(0).get_bar("AAPL").close, 13.0, 1e-9);
}
