#include "quant/data/csv_loader.hpp"
#include "quant/data/universe.hpp"
#include "quant/data/sqlite_storage.hpp"
#include "quant/indicators/sma.hpp"
#include "quant/indicators/ema.hpp"
#include "quant/indicators/rsi.hpp"
#include "quant/indicators/macd.hpp"
#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/rolling_volatility.hpp"
#include "quant/indicators/atr.hpp"
#include "quant/backtest/engine.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/backtest/strategies/rsi_mean_reversion.hpp"
#include "quant/backtest/strategies/momentum.hpp"
#include "quant/risk/metrics.hpp"
#include "quant/risk/report.hpp"
#include "quant/simulation/monte_carlo.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/constrained_qp.hpp"
#include "quant/optimization/efficient_frontier.hpp"

#include <iostream>
#include <filesystem>
#include <iomanip>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "=========================================================================\n";
    std::cout << "       QUANTITATIVE TRADING & PORTFOLIO OPTIMIZATION ENGINE (C++20)       \n";
    std::cout << "=========================================================================\n\n";

    std::filesystem::path sample_dir = "sample_data";
    if (argc > 1) {
        sample_dir = argv[1];
    } else if (!std::filesystem::exists(sample_dir)) {
        if (std::filesystem::exists("../sample_data")) {
            sample_dir = "../sample_data";
        }
    }

    std::cout << "[1/6] Ingesting multi-asset market data from: " << sample_dir.string() << "\n";
    auto raw_data = quant::data::CsvLoader::load_directory(sample_dir);
    if (raw_data.empty()) {
        std::cerr << "Error: No CSV files found in " << sample_dir.string() << "\n";
        return 1;
    }

    quant::data::MarketDataUniverse universe;
    for (auto& [ticker, series] : raw_data) {
        std::cout << "  - Loaded " << std::setw(6) << std::left << ticker
                  << ": " << series.size() << " bars ("
                  << series.dates.front() << " to " << series.dates.back() << ")\n";
        universe.add_asset(ticker, std::move(series));
    }
    universe.synchronize_timeline(true);
    std::cout << "  Synchronized timeline: " << universe.size() << " trading days across "
              << universe.get_tickers().size() << " assets.\n\n";

    // SQLite Persistence Demonstration
    std::cout << "[2/6] Storing market data into SQLite database (quant_portfolio.db)...\n";
    quant::data::SqliteStorage storage("quant_portfolio.db");
    for (const auto& ticker : universe.get_tickers()) {
        storage.save_market_data(ticker, universe.get_series(ticker));
    }
    std::cout << "  Successfully saved and indexed market data in SQLite.\n\n";

    // Technical Indicators Showcase
    std::cout << "[3/6] Computing technical indicators on benchmark asset (SPY)...\n";
    const auto& spy_series = universe.get_series("SPY");
    auto sma20 = quant::indicators::SMA::calculate(spy_series.close, 20);
    auto sma50 = quant::indicators::SMA::calculate(spy_series.close, 50);
    auto rsi14 = quant::indicators::RSI::calculate(spy_series.close, 14);
    auto macd = quant::indicators::MACD::calculate(spy_series.close, 12, 26, 9);
    auto bb = quant::indicators::BollingerBands::calculate(spy_series.close, 20, 2.0);
    auto vol20 = quant::indicators::RollingVolatility::calculate(spy_series.close, 20);
    auto atr14 = quant::indicators::ATR::calculate(spy_series.high, spy_series.low, spy_series.close, 14);

    size_t last_idx = spy_series.size() - 1;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  SPY Latest Date:  " << spy_series.dates[last_idx] << "\n";
    std::cout << "  SPY Close Price:  $" << spy_series.close[last_idx] << "\n";
    std::cout << "  SMA(20):          $" << sma20[last_idx] << " | SMA(50): $" << sma50[last_idx] << "\n";
    std::cout << "  RSI(14):          " << rsi14[last_idx] << " (Overbought > 70, Oversold < 30)\n";
    std::cout << "  MACD Line:        " << macd.macd_line[last_idx] << " | Signal: " << macd.signal_line[last_idx]
              << " | Hist: " << macd.histogram[last_idx] << "\n";
    std::cout << "  Bollinger Bands:  Lower=$" << bb.lower[last_idx] << " | Mid=$" << bb.middle[last_idx]
              << " | Upper=$" << bb.upper[last_idx] << " (%B=" << bb.percent_b[last_idx] << ")\n";
    std::cout << "  Rolling 20D Vol:  " << (vol20[last_idx] * 100.0) << " % (Annualized)\n";
    std::cout << "  ATR(14):          $" << atr14[last_idx] << "\n\n";

    // Backtesting Engine Simulation
    std::cout << "[4/6] Running Backtests with Realistic Execution Modeling...\n";
    quant::backtest::ExecutionConfig exec_cfg;
    exec_cfg.per_share_commission = 0.005;
    exec_cfg.min_commission = 1.00;
    exec_cfg.percentage_commission = 0.0001; // 1 bp
    exec_cfg.fixed_slippage_bps = 2.0;       // 2 bps
    exec_cfg.spread_bps = 3.0;               // 3 bps
    exec_cfg.enable_market_impact = true;

    quant::backtest::BacktestEngine engine(universe, quant::backtest::Portfolio(100000.0), quant::backtest::ExecutionModel(exec_cfg));

    // Strategy 1: SMA Crossover
    quant::backtest::strategies::SmaCrossoverStrategy sma_strat("SPY", 20, 50, 0.95);
    auto sma_result = engine.run(sma_strat);
    auto sma_summary = quant::risk::RiskReport::evaluate(sma_result, 0.02);

    std::cout << quant::risk::RiskReport::generate_text_report(sma_result, sma_summary);

    // ASCII Equity Curve
    std::vector<double> eq_vals;
    for (const auto& pt : sma_result.equity_curve) eq_vals.push_back(pt.equity);
    std::cout << quant::risk::RiskReport::render_ascii_chart(eq_vals, 60, 12);

    // Save SMA run to SQLite
    quant::data::BacktestRunRecord run_rec;
    run_rec.run_id = "RUN_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    run_rec.strategy_name = sma_result.strategy_name;
    run_rec.start_date = sma_result.start_date;
    run_rec.end_date = sma_result.end_date;
    run_rec.initial_cash = sma_result.initial_cash;
    run_rec.final_equity = sma_result.final_equity;
    run_rec.total_return = sma_result.total_return;
    run_rec.cagr = sma_summary.cagr;
    run_rec.sharpe_ratio = sma_summary.sharpe_ratio;
    run_rec.sortino_ratio = sma_summary.sortino_ratio;
    run_rec.max_drawdown = sma_summary.max_drawdown;
    run_rec.calmar_ratio = sma_summary.calmar_ratio;
    run_rec.var_95 = sma_summary.var_95_historical;
    run_rec.cvar_95 = sma_summary.cvar_95_historical;
    run_rec.total_trades = static_cast<int>(sma_result.trades.size());
    run_rec.win_rate = sma_summary.trade_details.win_rate;
    run_rec.profit_factor = sma_summary.trade_details.profit_factor;
    run_rec.created_at = "2026-08-30T12:00:00Z";
    storage.save_backtest_run(run_rec);

    std::vector<quant::data::TradeRecord> trade_records;
    for (const auto& tr : sma_result.trades) {
        quant::data::TradeRecord r;
        r.run_id = run_rec.run_id;
        r.ticker = tr.ticker;
        r.date = tr.date;
        r.timestamp = tr.timestamp;
        r.side = quant::backtest::side_to_string(tr.side);
        r.quantity = tr.quantity;
        r.price = tr.execution_price;
        r.commission = tr.commission;
        r.slippage = tr.slippage;
        r.realized_pnl = 0.0;
        trade_records.push_back(r);
    }
    storage.save_trades(trade_records);

    std::vector<quant::data::EquityPointRecord> eq_records;
    for (auto pt : sma_result.equity_curve) {
        pt.run_id = run_rec.run_id;
        eq_records.push_back(pt);
    }
    storage.save_equity_curve(eq_records);

    // Strategy 2: Multi-Asset Momentum
    std::cout << "\nRunning Multi-Asset Momentum Strategy...\n";
    quant::backtest::strategies::MultiAssetMomentumStrategy mom_strat(60, 20, 2, 0.95);
    auto mom_result = engine.run(mom_strat);
    auto mom_summary = quant::risk::RiskReport::evaluate(mom_result, 0.02);
    std::cout << "  Multi-Asset Momentum: Return=" << (mom_summary.total_return * 100.0) << " %, Sharpe="
              << mom_summary.sharpe_ratio << ", MaxDD=" << (mom_summary.max_drawdown * 100.0) << " %\n\n";

    // Markowitz Portfolio Optimization
    std::cout << "[5/6] Markowitz Mean-Variance Portfolio Optimization...\n";
    auto ret_mat_std = universe.get_aligned_returns_matrix();
    const size_t rows = ret_mat_std.size();
    const size_t cols = universe.get_tickers().size();

    Eigen::MatrixXd ret_mat(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            ret_mat(r, c) = ret_mat_std[r][c];
        }
    }

    Eigen::VectorXd mu = quant::optimization::PortfolioStats::compute_expected_returns(ret_mat, 252.0);
    Eigen::MatrixXd cov_shrunk = quant::optimization::PortfolioStats::compute_ledoit_wolf_covariance(ret_mat, 252.0);

    auto gmv_uncon = quant::optimization::UnconstrainedMarkowitz::global_minimum_variance(mu, cov_shrunk, 0.02);
    auto tan_uncon = quant::optimization::UnconstrainedMarkowitz::maximum_sharpe_portfolio(mu, cov_shrunk, 0.02);

    quant::optimization::ConstrainedQpConfig qp_cfg;
    qp_cfg.min_weight = 0.0;  // Long-only (no shorting)
    qp_cfg.max_weight = 0.40; // 40% maximum concentration per asset
    qp_cfg.risk_free_rate = 0.02;

    quant::optimization::ConstrainedQpOptimizer qp_opt(qp_cfg);
    auto gmv_con = qp_opt.global_minimum_variance(mu, cov_shrunk);
    auto tan_con = qp_opt.maximum_sharpe_portfolio(mu, cov_shrunk);
    auto risk_parity = quant::optimization::UnconstrainedMarkowitz::risk_parity_portfolio(mu, cov_shrunk, 0.02);

    std::cout << quant::optimization::EfficientFrontier::generate_portfolio_report(
        universe.get_tickers(), gmv_uncon, tan_uncon, gmv_con, tan_con, risk_parity
    );

    auto frontier_pts = quant::optimization::EfficientFrontier::compute_constrained_frontier(mu, cov_shrunk, qp_cfg, 35);
    quant::optimization::FrontierPoint gmv_pt{gmv_con.expected_return, gmv_con.volatility, gmv_con.sharpe_ratio, gmv_con.weights};
    quant::optimization::FrontierPoint tan_pt{tan_con.expected_return, tan_con.volatility, tan_con.sharpe_ratio, tan_con.weights};
    std::cout << quant::optimization::EfficientFrontier::render_ascii_frontier(frontier_pts, gmv_pt, tan_pt, 60, 12);

    // Monte Carlo Simulation with OpenMP
    std::cout << "[6/6] Running 50,000-Path Monte Carlo Simulation (OpenMP Parallel)...\n";
    quant::simulation::MonteCarloConfig mc_cfg;
    mc_cfg.num_simulations = 50000;
    mc_cfg.horizon_days = 252;
    mc_cfg.initial_wealth = 100000.0;
    mc_cfg.seed = 42;
    mc_cfg.use_bootstrap = true;

    quant::simulation::MonteCarloEngine mc_engine(mc_cfg);

    // Run 1: Bootstrapped Strategy Returns
    auto mc_strat_rep = mc_engine.run_simulation(sma_result.daily_returns);
    std::cout << "  [A] Bootstrapped SMA Crossover Strategy Simulation (50,000 paths):\n";
    std::cout << quant::simulation::MonteCarloEngine::generate_text_report(mc_strat_rep);

    // Run 2: Correlated GBM for Optimal Markowitz Portfolio
    std::cout << "\n  [B] Correlated GBM Simulation for Tangency Portfolio (50,000 paths):\n";
    auto mc_markowitz_rep = mc_engine.run_gbm_portfolio(mu, cov_shrunk / 252.0, tan_con.weights);
    std::cout << quant::simulation::MonteCarloEngine::generate_text_report(mc_markowitz_rep);

    std::cout << "\n=========================================================================\n";
    std::cout << "                 ALL SIMULATIONS & DEMONSTRATIONS COMPLETE               \n";
    std::cout << "=========================================================================\n";

    return 0;
}
