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
#include "quant/backtest/strategies/vol_target.hpp"
#include "quant/backtest/strategies/buy_and_hold.hpp"
#include "quant/risk/metrics.hpp"
#include "quant/risk/report.hpp"
#include "quant/simulation/monte_carlo.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include "quant/optimization/unconstrained.hpp"
#include "quant/optimization/constrained_qp.hpp"
#include "quant/optimization/efficient_frontier.hpp"
#include "quant/analysis/cost_sensitivity.hpp"
#include "quant/analysis/significance.hpp"
#include "quant/analysis/walk_forward.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace data = quant::data;
namespace backtest = quant::backtest;
namespace risk = quant::risk;
namespace simulation = quant::simulation;
namespace optimization = quant::optimization;
namespace analysis = quant::analysis;

namespace {

constexpr int kTotalSteps = 8;

struct Options {
    fs::path data_dir;
    std::string ticker{"SPY"};
    double capital{100000.0};
    double risk_free{0.02};
    double max_weight{0.40};
    size_t paths{50000};
    size_t horizon{252};
    uint64_t seed{42};
    size_t bootstrap{10000};
    double block{1.0};
    size_t wf_train{504};
    size_t wf_test{126};
    fs::path db_path{"axiomquant.db"};
    bool use_db{true};
    bool has_export{false};
    fs::path export_dir;
    backtest::FillTiming fill{backtest::FillTiming::NextBarOpen};
    bool show_help{false};
};

void print_usage(std::ostream& os) {
    os << "AxiomQuant - backtesting, risk, optimization and Monte Carlo engine\n\n"
          "Usage: axiomquant [options]\n\n"
          "Options:\n"
          "  --data <dir>          Directory of <TICKER>.csv OHLCV files (default: sample_data)\n"
          "  --ticker <symbol>     Asset for single-name strategies and indicators (default: SPY)\n"
          "  --capital <usd>       Starting capital (default: 100000)\n"
          "  --rf <rate>           Annual risk-free rate used in Sharpe (default: 0.02)\n"
          "  --fill <open|close>   Fill on the next bar open (default) or the signal bar close\n"
          "  --max-weight <w>      Per-asset cap for the constrained optimizer (default: 0.40)\n"
          "  --paths <n>           Monte Carlo paths (default: 50000)\n"
          "  --horizon <days>      Monte Carlo horizon in trading days (default: 252)\n"
          "  --seed <n>            Monte Carlo and bootstrap seed (default: 42)\n"
          "  --bootstrap <n>       Bootstrap resamples for the significance test (default: 10000)\n"
          "  --block <days>        Monte Carlo bootstrap mean block length (default: 1 = i.i.d.)\n"
          "  --wf-train <bars>     Walk-forward training window (default: 504)\n"
          "  --wf-test <bars>      Walk-forward test window (default: 126)\n"
          "  --db <path>           SQLite database file (default: axiomquant.db)\n"
          "  --no-db               Do not touch SQLite\n"
          "  --export-dir <dir>    Write equity curves, trades, weights and sweeps as CSV\n"
          "  -h, --help            Show this help and exit\n";
}

double parse_double(const std::string& flag, const std::string& value) {
    try {
        size_t pos = 0;
        const double parsed = std::stod(value, &pos);
        if (pos == value.size()) return parsed;
    } catch (const std::exception&) {
        // fall through to the shared error below
    }
    throw std::invalid_argument("invalid number for " + flag + ": '" + value + "'");
}

uint64_t parse_uint(const std::string& flag, const std::string& value) {
    if (!value.empty() && value[0] != '-') {
        try {
            size_t pos = 0;
            const unsigned long long parsed = std::stoull(value, &pos);
            if (pos == value.size()) return static_cast<uint64_t>(parsed);
        } catch (const std::exception&) {
            // fall through
        }
    }
    throw std::invalid_argument("invalid positive integer for " + flag + ": '" + value + "'");
}

Options parse_args(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") opt.show_help = true;
        else if (arg == "--data") opt.data_dir = value();
        else if (arg == "--ticker") opt.ticker = value();
        else if (arg == "--capital") opt.capital = parse_double(arg, value());
        else if (arg == "--rf") opt.risk_free = parse_double(arg, value());
        else if (arg == "--max-weight") opt.max_weight = parse_double(arg, value());
        else if (arg == "--paths") opt.paths = static_cast<size_t>(parse_uint(arg, value()));
        else if (arg == "--horizon") opt.horizon = static_cast<size_t>(parse_uint(arg, value()));
        else if (arg == "--seed") opt.seed = parse_uint(arg, value());
        else if (arg == "--bootstrap") opt.bootstrap = static_cast<size_t>(parse_uint(arg, value()));
        else if (arg == "--block") opt.block = parse_double(arg, value());
        else if (arg == "--wf-train") opt.wf_train = static_cast<size_t>(parse_uint(arg, value()));
        else if (arg == "--wf-test") opt.wf_test = static_cast<size_t>(parse_uint(arg, value()));
        else if (arg == "--db") {
            opt.db_path = value();
            opt.use_db = true;
        } else if (arg == "--no-db") opt.use_db = false;
        else if (arg == "--export-dir") {
            opt.export_dir = value();
            opt.has_export = true;
        } else if (arg == "--fill") {
            const std::string mode = value();
            if (mode == "open") opt.fill = backtest::FillTiming::NextBarOpen;
            else if (mode == "close") opt.fill = backtest::FillTiming::SameBarClose;
            else throw std::invalid_argument("--fill expects 'open' or 'close', got '" + mode + "'");
        } else if (!arg.empty() && arg[0] != '-' && opt.data_dir.empty()) opt.data_dir = arg;
        else throw std::invalid_argument("unknown option: " + arg);
    }

    if (opt.data_dir.empty()) {
        opt.data_dir = "sample_data";
        if (!fs::exists(opt.data_dir) && fs::exists("../sample_data")) opt.data_dir = "../sample_data";
    }
    if (!(opt.capital > 0.0)) throw std::invalid_argument("--capital must be positive");
    if (opt.paths == 0) throw std::invalid_argument("--paths must be positive");
    if (opt.horizon == 0) throw std::invalid_argument("--horizon must be positive");
    if (opt.bootstrap == 0) throw std::invalid_argument("--bootstrap must be positive");
    if (!(opt.block >= 1.0)) throw std::invalid_argument("--block must be at least 1");
    if (!(opt.max_weight > 0.0) || opt.max_weight > 1.0)
        throw std::invalid_argument("--max-weight must be in (0, 1]");
    if (opt.wf_train < 2 || opt.wf_test < 2)
        throw std::invalid_argument("--wf-train and --wf-test must be at least 2");
    return opt;
}

std::string utc_now_iso8601() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

void section(int step, const std::string& title) {
    std::cout << "\n[" << step << "/" << kTotalSteps << "] " << title << "\n";
}

struct StrategyRun {
    backtest::BacktestResult result;
    risk::PerformanceSummary summary;
};

std::string csv_field(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    return escaped + "\"";
}

std::ofstream open_csv(const fs::path& dir, const std::string& name) {
    std::ofstream out(dir / name);
    if (!out) throw std::runtime_error("cannot write " + (dir / name).string());
    out << std::fixed << std::setprecision(6);
    return out;
}

void print_tournament(const std::vector<StrategyRun>& runs) {
    const std::string rule(115, '-');
    std::cout << rule << "\n";
    std::cout << std::left << std::setw(42) << "Strategy" << std::right << std::setw(9) << "Return"
              << std::setw(8) << "CAGR" << std::setw(8) << "Vol" << std::setw(8) << "Sharpe" << std::setw(9)
              << "Sortino" << std::setw(9) << "MaxDD" << std::setw(8) << "Fills" << std::setw(9) << "WinRate"
              << std::setw(10) << "Costs$" << "\n";
    std::cout << rule << "\n";
    for (const auto& run : runs) {
        const auto& s = run.summary;
        std::cout << std::left << std::setw(42) << run.result.strategy_name << std::right << std::fixed
                  << std::setprecision(1) << std::setw(8) << (s.total_return * 100.0) << "%" << std::setw(7)
                  << (s.cagr * 100.0) << "%" << std::setw(7) << (s.annualized_volatility * 100.0) << "%"
                  << std::setprecision(2) << std::setw(8) << s.sharpe_ratio << std::setw(9) << s.sortino_ratio
                  << std::setprecision(1) << std::setw(8) << (s.max_drawdown * 100.0) << "%" << std::setw(8)
                  << run.result.total_trades;
        if (s.trade_details.closed_trades > 0) {
            std::cout << std::setw(8) << (s.trade_details.win_rate * 100.0) << "%";
        } else {
            std::cout << std::setw(9) << "n/a";   // never closed a position, so there is no win rate
        }
        std::cout << std::setprecision(0) << std::setw(10) << s.total_transaction_costs << "\n";
    }
    std::cout << rule << "\n";
}

void persist_run(data::SqliteStorage& storage,
                 const StrategyRun& run,
                 const std::string& run_id,
                 const std::string& created_at) {
    data::BacktestRunRecord record;
    record.run_id = run_id;
    record.strategy_name = run.result.strategy_name;
    record.start_date = run.result.start_date;
    record.end_date = run.result.end_date;
    record.initial_cash = run.result.initial_cash;
    record.final_equity = run.result.final_equity;
    record.total_return = run.result.total_return;
    record.cagr = run.summary.cagr;
    record.sharpe_ratio = run.summary.sharpe_ratio;
    record.sortino_ratio = run.summary.sortino_ratio;
    record.max_drawdown = run.summary.max_drawdown;
    record.calmar_ratio = run.summary.calmar_ratio;
    record.var_95 = run.summary.var_95_historical;
    record.cvar_95 = run.summary.cvar_95_historical;
    record.total_trades = static_cast<int>(run.result.trades.size());
    record.win_rate = run.summary.trade_details.win_rate;
    record.profit_factor =
        std::isinf(run.summary.trade_details.profit_factor) ? 0.0 : run.summary.trade_details.profit_factor;
    record.created_at = created_at;
    storage.save_backtest_run(record);

    std::vector<data::TradeRecord> trades;
    trades.reserve(run.result.trades.size());
    for (const auto& fill : run.result.trades) {
        data::TradeRecord row;
        row.run_id = run_id;
        row.ticker = fill.ticker;
        row.date = fill.date;
        row.timestamp = fill.timestamp;
        row.side = backtest::side_to_string(fill.side);
        row.quantity = fill.quantity;
        row.price = fill.execution_price;
        row.commission = fill.commission;
        row.slippage = fill.slippage;
        row.realized_pnl = fill.realized_pnl;
        trades.push_back(row);
    }
    storage.save_trades(trades);

    std::vector<data::EquityPointRecord> curve;
    curve.reserve(run.result.equity_curve.size());
    for (auto point : run.result.equity_curve) {
        point.run_id = run_id;
        curve.push_back(point);
    }
    storage.save_equity_curve(curve);
}

int run_pipeline(const Options& opt) {
    std::cout << "=========================================================================\n";
    std::cout << "   AXIOMQUANT - BACKTESTING, RISK, OPTIMIZATION & MONTE CARLO ENGINE      \n";
    std::cout << "=========================================================================\n";
#ifdef _OPENMP
    std::cout << "Parallelism: OpenMP, " << omp_get_max_threads() << " threads\n";
#else
    std::cout << "Parallelism: serial build (compiled without OpenMP)\n";
#endif

    // ------------------------------------------------------------------ data
    section(1, "Ingesting market data from: " + opt.data_dir.string());
    auto raw_data = data::CsvLoader::load_directory(opt.data_dir);
    if (raw_data.empty()) throw std::runtime_error("no CSV files found in " + opt.data_dir.string());

    std::vector<std::string> symbols;
    symbols.reserve(raw_data.size());
    for (const auto& entry : raw_data)
        symbols.push_back(entry.first);
    std::sort(symbols.begin(), symbols.end());

    data::MarketDataUniverse universe;
    for (const auto& symbol : symbols) {
        auto& series = raw_data[symbol];
        std::cout << "  - " << std::setw(6) << std::left << symbol << std::right << ": " << series.size()
                  << " bars (" << series.dates.front() << " to " << series.dates.back() << ")\n";
        universe.add_asset(symbol, std::move(series));
    }
    universe.synchronize_timeline(true);
    if (universe.size() < 60) throw std::runtime_error("need at least 60 aligned bars to run the pipeline");

    const auto& tickers = universe.get_tickers();
    std::string ticker = opt.ticker;
    if (std::find(tickers.begin(), tickers.end(), ticker) == tickers.end()) {
        std::cout << "  note: '" << ticker << "' is not in this universe, using " << tickers.front()
                  << " instead\n";
        ticker = tickers.front();
    }
    std::cout << "  Synchronized timeline: " << universe.size() << " trading days across " << tickers.size()
              << " assets (" << universe.get_timeline().front() << " to " << universe.get_timeline().back()
              << ")\n";

    // ---------------------------------------------------------------- sqlite
    section(2, "SQLite persistence");
    std::unique_ptr<data::SqliteStorage> storage;
    if (opt.use_db) {
        storage = std::make_unique<data::SqliteStorage>(opt.db_path);
        for (const auto& symbol : tickers)
            storage->save_market_data(symbol, universe.get_series(symbol));
        std::cout << "  Stored " << tickers.size() << " price histories in " << opt.db_path.string() << "\n";
    } else {
        std::cout << "  Skipped (--no-db)\n";
    }

    // ------------------------------------------------------------ indicators
    section(3, "Technical indicators on " + ticker);
    const auto& series = universe.get_series(ticker);
    const auto closes = universe.get_aligned_closes(ticker);
    const auto sma20 = quant::indicators::SMA::calculate(closes, 20);
    const auto sma50 = quant::indicators::SMA::calculate(closes, 50);
    const auto rsi14 = quant::indicators::RSI::calculate(closes, 14);
    const auto macd = quant::indicators::MACD::calculate(closes, 12, 26, 9);
    const auto bb = quant::indicators::BollingerBands::calculate(closes, 20, 2.0);
    const auto vol20 = quant::indicators::RollingVolatility::calculate(closes, 20);
    const auto atr14 = quant::indicators::ATR::calculate(series.high, series.low, series.close, 14);

    const size_t last = closes.size() - 1;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Last bar:         " << universe.get_timeline()[last] << "   close $" << closes[last]
              << "\n";
    std::cout << "  SMA(20)/SMA(50):  $" << sma20[last] << " / $" << sma50[last] << "\n";
    std::cout << "  RSI(14):          " << rsi14[last] << "   (overbought > 70, oversold < 30)\n";
    std::cout << "  MACD(12,26,9):    line " << macd.macd_line[last] << ", signal " << macd.signal_line[last]
              << ", hist " << macd.histogram[last] << "\n";
    std::cout << "  Bollinger(20,2):  $" << bb.lower[last] << " / $" << bb.middle[last] << " / $"
              << bb.upper[last] << "   (%B " << bb.percent_b[last] << ")\n";
    std::cout << "  Realized vol 20d: " << (vol20[last] * 100.0) << " % annualized\n";
    std::cout << "  ATR(14):          $" << atr14[series.size() - 1] << "\n";

    // ------------------------------------------------------------ tournament
    section(4, "Strategy tournament (identical costs, capital and fill rules)");
    backtest::ExecutionConfig exec_cfg;
    exec_cfg.per_share_commission = 0.005;
    exec_cfg.min_commission = 1.00;
    exec_cfg.percentage_commission = 0.0001;
    exec_cfg.fixed_slippage_bps = 2.0;
    exec_cfg.spread_bps = 3.0;
    exec_cfg.enable_market_impact = true;

    backtest::EngineConfig engine_cfg;
    engine_cfg.fill_timing = opt.fill;

    std::cout << "  Costs: $0.005/share (min $1.00) + 1 bp notional, 3 bp spread, 2 bp slippage, sqrt market "
                 "impact\n";
    std::cout << "  Fills: "
              << (opt.fill == backtest::FillTiming::NextBarOpen ? "next bar open (no look-ahead)"
                                                                : "signal bar close")
              << "   Capital: $" << std::setprecision(0) << opt.capital << std::setprecision(2) << "\n";

    backtest::BacktestEngine engine(universe, backtest::Portfolio(opt.capital),
                                    backtest::ExecutionModel(exec_cfg), engine_cfg);

    // Strategies are stateful, so each pass over the data needs a fresh set.
    auto make_strategies = [&ticker]() {
        std::vector<std::unique_ptr<backtest::Strategy>> built;
        built.push_back(std::make_unique<backtest::strategies::BuyAndHoldStrategy>(ticker, 0.99));
        built.push_back(std::make_unique<backtest::strategies::SmaCrossoverStrategy>(ticker, 20, 50, 0.95));
        built.push_back(
            std::make_unique<backtest::strategies::RsiMeanReversionStrategy>(ticker, 14, 30.0, 70.0, 0.95));
        built.push_back(std::make_unique<backtest::strategies::MultiAssetMomentumStrategy>(60, 20, 2, 0.95));
        built.push_back(std::make_unique<backtest::strategies::VolatilityTargetStrategy>(ticker, 0.10, 20));
        return built;
    };
    auto strategies = make_strategies();

    std::vector<StrategyRun> runs;
    const auto tournament_start = std::chrono::steady_clock::now();
    for (auto& strategy : strategies) {
        StrategyRun run;
        run.result = engine.run(*strategy);
        run.summary = risk::RiskReport::evaluate(run.result, opt.risk_free);
        runs.push_back(std::move(run));
    }
    const double tournament_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tournament_start)
            .count();

    print_tournament(runs);
    std::cout << "  " << runs.size() << " backtests over " << universe.size() << " bars in "
              << std::setprecision(1) << tournament_ms << " ms\n";

    size_t best_index = 1;
    for (size_t i = 2; i < runs.size(); ++i) {
        if (runs[i].summary.sharpe_ratio > runs[best_index].summary.sharpe_ratio) best_index = i;
    }
    const auto& best = runs[best_index];
    std::cout << "\n  Best active strategy by Sharpe: " << best.result.strategy_name << "\n\n";
    std::cout << risk::RiskReport::generate_text_report(best.result, best.summary);

    std::vector<double> best_equity;
    best_equity.reserve(best.result.equity_curve.size());
    for (const auto& point : best.result.equity_curve)
        best_equity.push_back(point.equity);
    std::cout << risk::RiskReport::render_ascii_chart(best_equity, 60, 12);

    if (storage) {
        const std::string created_at = utc_now_iso8601();
        const auto stamp = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count());
        for (size_t i = 0; i < runs.size(); ++i) {
            persist_run(*storage, runs[i], "RUN_" + stamp + "_" + std::to_string(i), created_at);
        }
        std::cout << "  Persisted " << runs.size() << " runs (summary, fills, equity curves) at "
                  << created_at << "\n";
    }

    // --------------------------------------------------- cost of look-ahead
    // Same strategies, same costs, same data - only the fill convention changes. The gap is the
    // return a backtest invents when it trades on the close that produced the signal.
    {
        backtest::EngineConfig biased_cfg = engine_cfg;
        biased_cfg.fill_timing = (engine_cfg.fill_timing == backtest::FillTiming::NextBarOpen)
                                     ? backtest::FillTiming::SameBarClose
                                     : backtest::FillTiming::NextBarOpen;
        backtest::BacktestEngine biased_engine(universe, backtest::Portfolio(opt.capital),
                                               backtest::ExecutionModel(exec_cfg), biased_cfg);

        std::cout << "\n  COST OF LOOK-AHEAD - next-bar-open fills vs trading the signal bar's close\n";
        std::cout << "  " << std::string(88, '-') << "\n";
        std::cout << "  " << std::left << std::setw(42) << "Strategy" << std::right << std::setw(11)
                  << "Sharpe" << std::setw(13) << "Sharpe(LA)" << std::setw(9) << "delta" << std::setw(11)
                  << "Return" << std::setw(13) << "Return(LA)" << "\n";
        std::cout << "  " << std::string(88, '-') << "\n";

        auto biased_strategies = make_strategies();
        for (size_t i = 0; i < biased_strategies.size(); ++i) {
            const auto biased_result = biased_engine.run(*biased_strategies[i]);
            const auto biased_summary = risk::RiskReport::evaluate(biased_result, opt.risk_free);
            std::cout << "  " << std::left << std::setw(42) << runs[i].result.strategy_name << std::right
                      << std::fixed << std::setprecision(2) << std::setw(11) << runs[i].summary.sharpe_ratio
                      << std::setw(13) << biased_summary.sharpe_ratio << std::setw(9)
                      << (biased_summary.sharpe_ratio - runs[i].summary.sharpe_ratio) << std::setprecision(1)
                      << std::setw(10) << (runs[i].summary.total_return * 100.0) << "%" << std::setw(12)
                      << (biased_summary.total_return * 100.0) << "%\n";
        }
        std::cout << "  " << std::string(88, '-') << "\n";
        std::cout
            << "  (LA) fills at the close that generated the signal - information no live trader has.\n";
    }

    // ------------------------------------------------- cost sensitivity
    {
        analysis::BacktestSetup cost_setup;
        cost_setup.initial_cash = opt.capital;
        cost_setup.execution = exec_cfg;
        cost_setup.engine = engine_cfg;
        cost_setup.risk_free_rate = opt.risk_free;
        const auto cost_rows =
            analysis::sweep_costs(universe, make_strategies, cost_setup, {0.0, 0.5, 1.0, 2.0, 5.0, 10.0});
        std::cout << "\n" << analysis::format_cost_sensitivity_report(cost_rows);
    }

    // ----------------------------------------------------------------- sweep
    section(5, "In-sample parameter sweep: SMA crossover on " + ticker);
    analysis::BacktestSetup setup;
    setup.initial_cash = opt.capital;
    setup.execution = exec_cfg;
    setup.engine = engine_cfg;
    setup.risk_free_rate = opt.risk_free;

    const std::vector<size_t> fast_grid{5, 10, 15, 20, 30, 40, 50};
    const std::vector<size_t> slow_grid{50, 75, 100, 125, 150, 200};
    const auto sweep_start = std::chrono::steady_clock::now();
    const auto sweep =
        analysis::sweep_sma_parameters(universe, ticker, fast_grid, slow_grid, setup, 0, universe.size(), 0);
    const double sweep_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sweep_start).count();

    std::cout << "  " << sweep.size() << " parameter pairs backtested in " << std::setprecision(1) << sweep_ms
              << " ms\n";
    std::cout << "  Rank  Fast/Slow    Sharpe   Return    MaxDD  Fills\n";
    for (size_t i = 0; i < std::min<size_t>(5, sweep.size()); ++i) {
        const auto& p = sweep[i];
        const std::string params = std::to_string(p.fast_period) + "/" + std::to_string(p.slow_period);
        std::cout << std::setw(6) << (i + 1) << "  " << std::left << std::setw(9) << params << std::right
                  << std::setprecision(2) << std::setw(8) << p.performance.sharpe_ratio
                  << std::setprecision(1) << std::setw(8) << (p.performance.total_return * 100.0) << "%"
                  << std::setw(8) << (p.performance.max_drawdown * 100.0) << "%" << std::setw(7)
                  << p.performance.trades << "\n";
    }
    if (!sweep.empty()) {
        std::vector<double> sharpes;
        sharpes.reserve(sweep.size());
        for (const auto& p : sweep)
            sharpes.push_back(p.performance.sharpe_ratio);
        std::sort(sharpes.begin(), sharpes.end());
        const double median = sharpes[sharpes.size() / 2];
        const double benchmark_sharpe = runs.front().summary.sharpe_ratio;
        const auto beating = std::count_if(sharpes.begin(), sharpes.end(),
                                           [benchmark_sharpe](double s) { return s > benchmark_sharpe; });
        std::cout << std::setprecision(2);
        std::cout << "  Sharpe across the grid: best " << sharpes.back() << ", median " << median
                  << ", worst " << sharpes.front() << "\n";
        std::cout << "  Pairs beating buy & hold (Sharpe " << benchmark_sharpe << "): " << beating << " / "
                  << sweep.size() << "\n";
        std::cout << "  NOTE: every number above is in-sample. The top pair is the luckiest pair on this\n";
        std::cout << "        sample, not a forecast - the walk-forward test below is the honest one.\n";

        // Deflate the winner's Sharpe for the number of pairs the search tried.
        const auto& best = sweep.front();
        backtest::strategies::SmaCrossoverStrategy best_strategy(ticker, best.fast_period, best.slow_period,
                                                                 0.95);
        std::vector<double> best_returns;
        (void)analysis::evaluate_window(universe, best_strategy, setup, 0, universe.size(), 0, &best_returns);
        std::vector<double> trial_sharpes;
        trial_sharpes.reserve(sweep.size());
        for (const auto& p : sweep)
            trial_sharpes.push_back(p.performance.sharpe_ratio);
        const auto deflated = analysis::deflated_sharpe_ratio(best_returns, trial_sharpes, opt.risk_free);
        std::cout << "\n" << analysis::format_deflated_sharpe_report(deflated);
    }

    // ----------------------------------------------------------- walk-forward
    section(6, "Walk-forward out-of-sample evaluation");
    analysis::WalkForwardConfig wf_cfg;
    wf_cfg.train_bars = opt.wf_train;
    wf_cfg.test_bars = opt.wf_test;
    analysis::WalkForwardResult wf;
    if (universe.size() >= opt.wf_train + opt.wf_test) {
        wf = analysis::run_sma_walk_forward(universe, ticker, wf_cfg, setup);
        std::cout << analysis::format_walk_forward_report(wf, ticker);
        if (wf.oos_returns.size() >= 3) {
            analysis::BootstrapConfig boot_cfg;
            boot_cfg.resamples = opt.bootstrap;
            boot_cfg.seed = opt.seed;
            boot_cfg.risk_free_rate = opt.risk_free;
            const auto boot = analysis::bootstrap_sharpe(wf.oos_returns, wf.benchmark_returns, boot_cfg);
            std::cout << "\n  IS THE OUT-OF-SAMPLE RESULT DISTINGUISHABLE FROM LUCK?\n"
                      << analysis::format_bootstrap_report(boot);
        }
    } else {
        std::cout << "  Skipped: " << universe.size() << " bars is less than train (" << opt.wf_train
                  << ") + test (" << opt.wf_test << ")\n";
    }

    // ----------------------------------------------------------- optimization
    section(7, "Markowitz mean-variance optimization");
    const auto returns_std = universe.get_aligned_returns_matrix();
    const auto rows = static_cast<Eigen::Index>(returns_std.size());
    const auto cols = static_cast<Eigen::Index>(tickers.size());
    Eigen::MatrixXd returns(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r) {
        for (Eigen::Index c = 0; c < cols; ++c) {
            returns(r, c) = returns_std[static_cast<size_t>(r)][static_cast<size_t>(c)];
        }
    }

    double shrinkage = 0.0;
    const Eigen::VectorXd mu = optimization::PortfolioStats::compute_expected_returns(returns, 252.0);
    const Eigen::MatrixXd cov =
        optimization::PortfolioStats::compute_ledoit_wolf_covariance(returns, 252.0, &shrinkage);
    std::cout << "  Ledoit-Wolf shrinkage towards constant correlation: delta = " << std::setprecision(4)
              << shrinkage << "\n";

    const auto gmv_uncon =
        optimization::UnconstrainedMarkowitz::global_minimum_variance(mu, cov, opt.risk_free);
    const auto tan_uncon =
        optimization::UnconstrainedMarkowitz::maximum_sharpe_portfolio(mu, cov, opt.risk_free);
    const auto risk_parity =
        optimization::UnconstrainedMarkowitz::risk_parity_portfolio(mu, cov, opt.risk_free);

    optimization::ConstrainedQpConfig qp_cfg;
    qp_cfg.min_weight = 0.0;
    qp_cfg.max_weight = opt.max_weight;
    qp_cfg.risk_free_rate = opt.risk_free;
    const optimization::ConstrainedQpOptimizer qp(qp_cfg);
    const auto gmv_con = qp.global_minimum_variance(mu, cov);
    const auto tan_con = qp.maximum_sharpe_portfolio(mu, cov);

    std::cout << optimization::EfficientFrontier::generate_portfolio_report(tickers, gmv_uncon, tan_uncon,
                                                                            gmv_con, tan_con, risk_parity);

    const auto frontier = optimization::EfficientFrontier::compute_constrained_frontier(mu, cov, qp_cfg, 35);
    const optimization::FrontierPoint gmv_pt{gmv_con.expected_return, gmv_con.volatility,
                                             gmv_con.sharpe_ratio, gmv_con.weights};
    const optimization::FrontierPoint tan_pt{tan_con.expected_return, tan_con.volatility,
                                             tan_con.sharpe_ratio, tan_con.weights};
    std::cout << optimization::EfficientFrontier::render_ascii_frontier(frontier, gmv_pt, tan_pt, 60, 12);

    // ------------------------------------------------------------ monte carlo
    section(8, "Monte Carlo simulation");
    simulation::MonteCarloConfig mc_cfg;
    mc_cfg.num_simulations = opt.paths;
    mc_cfg.horizon_days = opt.horizon;
    mc_cfg.initial_wealth = opt.capital;
    mc_cfg.seed = opt.seed;
    mc_cfg.use_bootstrap = true;
    mc_cfg.block_length = opt.block;
    const simulation::MonteCarloEngine mc(mc_cfg);

    std::cout << "  [A] Bootstrapped daily returns of " << best.result.strategy_name << "\n";
    const auto mc_strategy = mc.run_simulation(best.result.daily_returns);
    std::cout << simulation::MonteCarloEngine::generate_text_report(mc_strategy);

    std::cout << "\n  [B] Correlated GBM of the constrained max-Sharpe portfolio\n";
    const auto mc_portfolio = mc.run_gbm_portfolio(mu, cov / 252.0, tan_con.weights);
    std::cout << simulation::MonteCarloEngine::generate_text_report(mc_portfolio);

    // ---------------------------------------------------------------- exports
    if (opt.has_export) {
        fs::create_directories(opt.export_dir);

        auto summary_csv = open_csv(opt.export_dir, "strategy_summary.csv");
        summary_csv << "strategy,total_return,cagr,annual_vol,sharpe,sortino,calmar,max_drawdown,"
                       "var95,cvar95,fills,closed_trades,win_rate,transaction_costs\n";
        for (const auto& run : runs) {
            const auto& s = run.summary;
            summary_csv << csv_field(run.result.strategy_name) << ',' << s.total_return << ',' << s.cagr
                        << ',' << s.annualized_volatility << ',' << s.sharpe_ratio << ',' << s.sortino_ratio
                        << ',' << s.calmar_ratio << ',' << s.max_drawdown << ',' << s.var_95_historical << ','
                        << s.cvar_95_historical << ',' << run.result.total_trades << ','
                        << s.trade_details.closed_trades << ',' << s.trade_details.win_rate << ','
                        << s.total_transaction_costs << '\n';
        }

        auto equity_csv = open_csv(opt.export_dir, "equity_curves.csv");
        equity_csv << "date";
        for (const auto& run : runs)
            equity_csv << ',' << csv_field(run.result.strategy_name);
        equity_csv << '\n';
        for (size_t t = 0; t < universe.size(); ++t) {
            equity_csv << universe.get_timeline()[t];
            for (const auto& run : runs) {
                equity_csv << ','
                           << (t < run.result.equity_curve.size() ? run.result.equity_curve[t].equity : 0.0);
            }
            equity_csv << '\n';
        }

        auto trades_csv = open_csv(opt.export_dir, "trades.csv");
        trades_csv << "strategy,date,ticker,side,quantity,execution_price,commission,slippage,realized_pnl,"
                      "closes_position\n";
        for (const auto& run : runs) {
            for (const auto& fill : run.result.trades) {
                trades_csv << csv_field(run.result.strategy_name) << ',' << fill.date << ',' << fill.ticker
                           << ',' << backtest::side_to_string(fill.side) << ',' << fill.quantity << ','
                           << fill.execution_price << ',' << fill.commission << ',' << fill.slippage << ','
                           << fill.realized_pnl << ',' << (fill.closes_position ? 1 : 0) << '\n';
            }
        }

        auto weights_csv = open_csv(opt.export_dir, "portfolio_weights.csv");
        weights_csv << "ticker,gmv_unconstrained,tangency_unconstrained,gmv_constrained,max_sharpe_"
                       "constrained,risk_parity\n";
        for (size_t i = 0; i < tickers.size(); ++i) {
            const auto idx = static_cast<Eigen::Index>(i);
            weights_csv << tickers[i] << ',' << gmv_uncon.weights(idx) << ',' << tan_uncon.weights(idx) << ','
                        << gmv_con.weights(idx) << ',' << tan_con.weights(idx) << ','
                        << risk_parity.weights(idx) << '\n';
        }

        auto frontier_csv = open_csv(opt.export_dir, "efficient_frontier.csv");
        frontier_csv << "expected_return,volatility,sharpe\n";
        for (const auto& point : frontier) {
            frontier_csv << point.expected_return << ',' << point.volatility << ',' << point.sharpe_ratio
                         << '\n';
        }

        auto sweep_csv = open_csv(opt.export_dir, "parameter_sweep.csv");
        sweep_csv << "fast,slow,sharpe,total_return,max_drawdown,fills\n";
        for (const auto& p : sweep) {
            sweep_csv << p.fast_period << ',' << p.slow_period << ',' << p.performance.sharpe_ratio << ','
                      << p.performance.total_return << ',' << p.performance.max_drawdown << ','
                      << p.performance.trades << '\n';
        }

        auto wf_csv = open_csv(opt.export_dir, "walk_forward.csv");
        wf_csv << "fold,train_start,train_end,test_start,test_end,fast,slow,is_sharpe,oos_sharpe,"
                  "oos_return,oos_max_drawdown,benchmark_return\n";
        for (size_t i = 0; i < wf.folds.size(); ++i) {
            const auto& f = wf.folds[i];
            wf_csv << (i + 1) << ',' << f.train_start << ',' << f.train_end << ',' << f.test_start << ','
                   << f.test_end << ',' << f.fast_period << ',' << f.slow_period << ','
                   << f.in_sample.sharpe_ratio << ',' << f.out_of_sample.sharpe_ratio << ','
                   << f.out_of_sample.total_return << ',' << f.out_of_sample.max_drawdown << ','
                   << f.benchmark.total_return << '\n';
        }

        std::cout << "\nExported CSV results to " << opt.export_dir.string() << "\n";
    }

    std::cout << "\n=========================================================================\n";
    std::cout << "                          PIPELINE COMPLETE                              \n";
    std::cout << "=========================================================================\n";
    return 0;
}

}   // namespace

int main(int argc, char* argv[]) {
    Options opt;
    try {
        opt = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return 2;
    }
    if (opt.show_help) {
        print_usage(std::cout);
        return 0;
    }
    try {
        return run_pipeline(opt);
    } catch (const std::exception& e) {
        std::cerr << "\nerror: " << e.what() << "\n";
        return 1;
    }
}
