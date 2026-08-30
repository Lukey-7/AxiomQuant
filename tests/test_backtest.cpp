#include "test_harness.hpp"
#include "quant/backtest/order.hpp"
#include "quant/backtest/position.hpp"
#include "quant/backtest/portfolio.hpp"
#include "quant/backtest/execution_model.hpp"
#include "quant/backtest/engine.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/data/types.hpp"
#include "quant/data/universe.hpp"

TEST_CASE(TestBacktest_ExecutionModel_Costs) {
    quant::backtest::ExecutionConfig cfg;
    cfg.per_share_commission = 0.01;
    cfg.min_commission = 1.00;
    cfg.percentage_commission = 0.001; // 10 bps
    cfg.fixed_slippage_bps = 5.0;      // 5 bps
    cfg.spread_bps = 4.0;              // 4 bps
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
    EXPECT_NEAR(pos.average_price, 155.0, 1e-6); // avg price of remaining shares unchanged
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
