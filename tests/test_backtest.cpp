#include "test_harness.hpp"
#include "test_support.hpp"
#include "quant/backtest/strategies/momentum.hpp"
#include "quant/risk/report.hpp"
#include "quant/backtest/order.hpp"
#include "quant/backtest/position.hpp"
#include "quant/backtest/portfolio.hpp"
#include <cmath>
#include "quant/backtest/execution_model.hpp"
#include "quant/backtest/engine.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/data/types.hpp"
#include "quant/data/universe.hpp"

TEST_CASE(TestBacktest_ExecutionModel_Costs) {
    quant::backtest::ExecutionConfig cfg;
    cfg.per_share_commission = 0.01;
    cfg.min_commission = 1.00;
    cfg.percentage_commission = 0.001;   // 10 bps
    cfg.fixed_slippage_bps = 5.0;        // 5 bps
    cfg.spread_bps = 4.0;                // 4 bps
    cfg.enable_market_impact = false;

    quant::backtest::ExecutionModel exec(cfg);

    quant::data::Bar bar{"2023-01-01", 1000, 100.0, 105.0, 95.0, 100.0, 100.0, 1000000.0};

    quant::backtest::Order buy_order;
    buy_order.ticker = "SPY";
    buy_order.side = quant::backtest::OrderSide::BUY;
    buy_order.quantity = 100.0;

    auto buy_fill = exec.execute_order(buy_order, bar);
    // Half spread = 2 bps = $0.02. Slippage = 5 bps = $0.05.
    // Execution price = 100.0 + 0.02 + 0.05 = 100.07
    EXPECT_NEAR(buy_fill.execution_price, 100.07, 1e-4);
    EXPECT_TRUE(buy_fill.commission > 0.0);

    quant::backtest::Order sell_order;
    sell_order.ticker = "SPY";
    sell_order.side = quant::backtest::OrderSide::SELL;
    sell_order.quantity = 100.0;

    auto sell_fill = exec.execute_order(sell_order, bar);
    // Execution price = 100.0 - 0.02 - 0.05 = 99.93
    EXPECT_NEAR(sell_fill.execution_price, 99.93, 1e-4);
}

TEST_CASE(TestBacktest_Position_PnL) {
    quant::backtest::Position pos("AAPL");
    EXPECT_FALSE(pos.is_open());

    // Buy 100 shares @ $150
    quant::backtest::Fill fill1;
    fill1.ticker = "AAPL";
    fill1.side = quant::backtest::OrderSide::BUY;
    fill1.quantity = 100.0;
    fill1.execution_price = 150.0;
    fill1.commission = 1.0;
    double pnl1 = pos.update_with_fill(fill1);
    EXPECT_NEAR(pnl1, 0.0, 1e-6);
    EXPECT_NEAR(pos.quantity, 100.0, 1e-6);
    EXPECT_NEAR(pos.average_price, 150.0, 1e-6);

    // Buy another 100 shares @ $160 -> average price = $155
    quant::backtest::Fill fill2;
    fill2.ticker = "AAPL";
    fill2.side = quant::backtest::OrderSide::BUY;
    fill2.quantity = 100.0;
    fill2.execution_price = 160.0;
    fill2.commission = 1.0;
    pos.update_with_fill(fill2);
    EXPECT_NEAR(pos.quantity, 200.0, 1e-6);
    EXPECT_NEAR(pos.average_price, 155.0, 1e-6);

    // Sell 100 shares @ $175 -> Realized PnL = 100 * (175 - 155) - 1 commission = 1999.0
    quant::backtest::Fill fill3;
    fill3.ticker = "AAPL";
    fill3.side = quant::backtest::OrderSide::SELL;
    fill3.quantity = 100.0;
    fill3.execution_price = 175.0;
    fill3.commission = 1.0;
    double pnl3 = pos.update_with_fill(fill3);
    EXPECT_NEAR(pnl3, 1999.0, 1e-6);
    EXPECT_NEAR(pos.quantity, 100.0, 1e-6);
    EXPECT_NEAR(pos.average_price, 155.0, 1e-6);   // avg price of remaining shares unchanged
}

TEST_CASE(TestBacktest_Portfolio_Cash_Conservation) {
    quant::backtest::Portfolio port(50000.0);
    EXPECT_NEAR(port.get_cash(), 50000.0, 1e-6);
    EXPECT_NEAR(port.get_total_equity(), 50000.0, 1e-6);

    // Execute buy: 100 shares @ $100 + $2 commission -> Cash = 50000 - 10002 = 39998
    quant::backtest::Fill fill;
    fill.ticker = "MSFT";
    fill.side = quant::backtest::OrderSide::BUY;
    fill.quantity = 100.0;
    fill.execution_price = 100.0;
    fill.commission = 2.0;
    port.process_fill(fill);

    EXPECT_NEAR(port.get_cash(), 39998.0, 1e-6);

    // Mark to market with price = $110
    quant::data::MarketSnapshot snap;
    snap.date = "2023-01-01";
    snap.timestamp = 1000;
    snap.bars["MSFT"] = {"2023-01-01", 1000, 100.0, 112.0, 99.0, 110.0, 110.0, 5000};
    port.mark_to_market(snap);

    // Position value = 100 * 110 = 11000. Equity = 39998 + 11000 = 50998.
    EXPECT_NEAR(port.get_total_equity(), 50998.0, 1e-6);
    EXPECT_EQ(port.get_equity_curve().size(), 1);
}

// --- Regression: orders must not be filled with information from the bar that generated them ---
TEST_CASE(TestBacktest_NextBarOpen_HasNoLookAhead) {
    auto universe = quant::tests::make_universe({{"AAA",
                                                  {100.0, 101.0, 102.0, 103.0, 104.0},      // closes
                                                  {200.0, 201.0, 202.0, 203.0, 204.0}}});   // opens

    quant::tests::ScriptedStrategy strategy({{1, "AAA", quant::backtest::OrderSide::BUY, 10.0}});
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(100000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()),
        quant::backtest::EngineConfig{});

    auto result = engine.run(strategy);
    EXPECT_EQ(result.trades.size(), 1u);
    // The signal is generated on bar 1 but can only be executed at the open of bar 2.
    EXPECT_EQ(result.trades[0].date, quant::tests::synthetic_date(2));
    EXPECT_NEAR(result.trades[0].execution_price, 202.0, 1e-9);
}

TEST_CASE(TestBacktest_SameBarClose_FillTimingIsOptIn) {
    auto universe = quant::tests::make_universe(
        {{"AAA", {100.0, 101.0, 102.0, 103.0, 104.0}, {200.0, 201.0, 202.0, 203.0, 204.0}}});

    quant::tests::ScriptedStrategy strategy({{1, "AAA", quant::backtest::OrderSide::BUY, 10.0}});
    quant::backtest::EngineConfig cfg;
    cfg.fill_timing = quant::backtest::FillTiming::SameBarClose;
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(100000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()), cfg);

    auto result = engine.run(strategy);
    EXPECT_EQ(result.trades.size(), 1u);
    EXPECT_EQ(result.trades[0].date, quant::tests::synthetic_date(1));
    EXPECT_NEAR(result.trades[0].execution_price, 101.0, 1e-9);
}

TEST_CASE(TestBacktest_OrderOnFinalBarCannotBeFilled) {
    auto universe = quant::tests::make_universe({{"AAA", {100.0, 101.0, 102.0}, {}}});
    quant::tests::ScriptedStrategy strategy({{2, "AAA", quant::backtest::OrderSide::BUY, 5.0}});
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(100000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()));

    auto result = engine.run(strategy);
    EXPECT_EQ(result.trades.size(), 0u);
    EXPECT_EQ(result.unfilled_orders, 1u);
}

// --- Regression: an oversized BUY must be downsized, and the downsize must survive re-pricing ---
TEST_CASE(TestBacktest_OversizedBuyIsDownsizedAndCashStaysPositive) {
    auto universe = quant::tests::make_universe({{"AAA", {100.0, 100.0, 100.0, 100.0}, {}}});
    quant::tests::ScriptedStrategy strategy({{0, "AAA", quant::backtest::OrderSide::BUY, 1000.0}});
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(10000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()));

    auto result = engine.run(strategy);
    EXPECT_EQ(result.trades.size(), 1u);
    // cash_buffer 0.995 of $10,000 at $100 per share => 99 shares, not the requested 1,000.
    EXPECT_NEAR(result.trades[0].quantity, 99.0, 1e-9);
    EXPECT_TRUE(engine.get_portfolio().get_cash() >= 0.0);
}

// --- Regression: trade statistics must come from realized PnL, not from sale proceeds ---
TEST_CASE(TestBacktest_RealizedPnlDrivesTradeStatistics) {
    auto universe = quant::tests::make_universe({{"AAA",
                                                  {100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0},
                                                  {100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0}}});

    quant::tests::ScriptedStrategy strategy({
        {1, "AAA", quant::backtest::OrderSide::BUY, 10.0},
        {4, "AAA", quant::backtest::OrderSide::SELL, 10.0},
    });
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(100000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()));
    auto result = engine.run(strategy);

    EXPECT_EQ(result.trades.size(), 2u);
    EXPECT_FALSE(result.trades[0].closes_position);
    EXPECT_TRUE(result.trades[1].closes_position);
    // Bought at the bar-2 open (120), sold at the bar-5 open (150): 10 * 30 = 300.
    EXPECT_NEAR(result.trades[1].realized_pnl, 300.0, 1e-9);

    auto summary = quant::risk::RiskReport::evaluate(result, 0.02);
    EXPECT_EQ(summary.trade_details.total_trades, 2);
    EXPECT_EQ(summary.trade_details.closed_trades, 1);
    EXPECT_EQ(summary.trade_details.winning_trades, 1);
    EXPECT_NEAR(summary.trade_details.win_rate, 1.0, 1e-12);
    EXPECT_NEAR(summary.trade_details.total_realized_pnl, 300.0, 1e-9);
}

// --- Regression: momentum must rank by trailing return and rotate out of losers ---
TEST_CASE(TestBacktest_MomentumRanksAndRotates) {
    const size_t bars = 120;
    std::vector<double> steady(bars), spike(bars), slow(bars);
    for (size_t t = 0; t < bars; ++t) {
        steady[t] = 100.0 * std::pow(1.004, static_cast<double>(t));
        slow[t] = 100.0 * std::pow(1.001, static_cast<double>(t));
        spike[t] = t < 60 ? 100.0 * std::pow(1.02, static_cast<double>(t))
                          : 100.0 * std::pow(1.02, 59.0) * std::pow(0.98, static_cast<double>(t - 59));
    }
    auto universe = quant::tests::make_universe({
        {"STEADY", steady, {}},
        {"SPIKE", spike, {}},
        {"SLOWPOKE", slow, {}},
    });

    quant::backtest::strategies::MultiAssetMomentumStrategy strategy(20, 10, 2, 0.95);
    quant::backtest::BacktestEngine engine(
        universe, quant::backtest::Portfolio(100000.0),
        quant::backtest::ExecutionModel(quant::tests::frictionless_execution()));
    auto result = engine.run(strategy);

    bool bought_spike_early = false, sold_spike_late = false, bought_slowpoke_late = false;
    for (const auto& fill : result.trades) {
        const bool early = fill.date < quant::tests::synthetic_date(60);
        if (fill.ticker == "SPIKE" && fill.side == quant::backtest::OrderSide::BUY && early)
            bought_spike_early = true;
        if (fill.ticker == "SPIKE" && fill.side == quant::backtest::OrderSide::SELL && !early)
            sold_spike_late = true;
        if (fill.ticker == "SLOWPOKE" && fill.side == quant::backtest::OrderSide::BUY && !early)
            bought_slowpoke_late = true;
    }

    EXPECT_TRUE(bought_spike_early);     // the fastest riser is bought while it is rising
    EXPECT_TRUE(sold_spike_late);        // and liquidated once its trailing return turns negative
    EXPECT_TRUE(bought_slowpoke_late);   // the replacement is actually bought
    EXPECT_NEAR(engine.get_portfolio().get_position_quantity("SPIKE"), 0.0, 1e-9);
    EXPECT_TRUE(engine.get_portfolio().get_position_quantity("STEADY") > 0.0);

    const auto& ranking = strategy.last_ranking();
    EXPECT_TRUE(ranking.size() >= 2);
    for (size_t i = 1; i < ranking.size(); ++i) {
        EXPECT_TRUE(ranking[i - 1].trailing_return >= ranking[i].trailing_return);
    }
}
