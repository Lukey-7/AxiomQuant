// Python bindings for AxiomQuant.
//
// The API is deliberately small: load a universe from a CSV directory, describe a strategy, run a
// backtest with a cost model, and read the results back as NumPy arrays and plain numbers. The
// walk-forward study and the significance tests are exposed on top of that. Everything long-running
// releases the GIL, and all heavy lifting happens in the same C++ code the CLI uses, so results are
// identical to the command line for the same inputs.

#include "quant/analysis/significance.hpp"
#include "quant/analysis/walk_forward.hpp"
#include "quant/backtest/engine.hpp"
#include "quant/backtest/strategies/buy_and_hold.hpp"
#include "quant/backtest/strategies/momentum.hpp"
#include "quant/backtest/strategies/rsi_mean_reversion.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/backtest/strategies/vol_target.hpp"
#include "quant/data/csv_loader.hpp"
#include "quant/data/membership.hpp"
#include "quant/data/universe.hpp"
#include "quant/risk/metrics.hpp"
#include "quant/risk/report.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace fs = std::filesystem;
namespace data = quant::data;
namespace backtest = quant::backtest;
namespace strategies = quant::backtest::strategies;
namespace risk = quant::risk;
namespace analysis = quant::analysis;

using namespace py::literals;

namespace {

// ----------------------------------------------------------------------------- helpers

/// Copies a vector into a new one-dimensional float64 NumPy array.
py::array_t<double> to_array(const std::vector<double>& values) {
    py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

/// Copies a row-major matrix (all rows the same length) into a two-dimensional NumPy array.
py::array_t<double> to_matrix(const std::vector<std::vector<double>>& rows) {
    const size_t n_rows = rows.size();
    const size_t n_cols = n_rows == 0 ? 0 : rows.front().size();
    py::array_t<double> out({static_cast<py::ssize_t>(n_rows), static_cast<py::ssize_t>(n_cols)});
    double* dst = out.mutable_data();
    for (const auto& row : rows) {
        if (row.size() != n_cols) throw std::runtime_error("ragged matrix");
        dst = std::copy(row.begin(), row.end(), dst);
    }
    return out;
}

backtest::FillTiming parse_fill(const std::string& fill) {
    if (fill == "open") return backtest::FillTiming::NextBarOpen;
    if (fill == "close") return backtest::FillTiming::SameBarClose;
    throw py::value_error("fill must be 'open' (next bar open, the default) or 'close', got '" + fill + "'");
}

/// Shared engine settings, built from the keyword arguments every backtest entry point accepts.
analysis::BacktestSetup make_setup(double capital,
                                   const std::optional<backtest::ExecutionConfig>& costs,
                                   const std::string& fill,
                                   double risk_free) {
    if (!(capital > 0.0)) throw py::value_error("capital must be positive");
    analysis::BacktestSetup setup;
    setup.initial_cash = capital;
    setup.execution = costs.value_or(backtest::ExecutionConfig{});
    setup.engine.fill_timing = parse_fill(fill);
    setup.risk_free_rate = risk_free;
    return setup;
}

void require_ticker(const data::MarketDataUniverse& universe, const std::string& ticker) {
    const auto& tickers = universe.get_tickers();
    if (std::find(tickers.begin(), tickers.end(), ticker) == tickers.end()) {
        throw py::key_error("ticker '" + ticker + "' is not in the universe");
    }
}

// ----------------------------------------------------------------------------- data

/// Loads every <TICKER>.csv in a directory and aligns them on a common timeline, as the CLI does.
data::MarketDataUniverse load_universe(const fs::path& directory,
                                       const std::string& align,
                                       const std::optional<std::vector<std::string>>& tickers) {
    bool require_all = true;
    if (align == "union") require_all = false;
    else if (align != "intersection")
        throw py::value_error("align must be 'intersection' or 'union', got '" + align + "'");

    auto raw = data::CsvLoader::load_directory(directory);
    if (raw.empty()) throw py::value_error("no CSV files found in " + directory.string());

    std::vector<std::string> symbols;
    if (tickers) {
        symbols = *tickers;
        for (const auto& symbol : symbols) {
            if (raw.find(symbol) == raw.end())
                throw py::key_error("no " + symbol + ".csv in " + directory.string());
        }
    } else {
        for (const auto& entry : raw)
            symbols.push_back(entry.first);
    }
    std::sort(symbols.begin(), symbols.end());
    symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());

    data::MarketDataUniverse universe;
    for (const auto& symbol : symbols)
        universe.add_asset(symbol, std::move(raw[symbol]));
    universe.synchronize_timeline(require_all);
    return universe;
}

// ----------------------------------------------------------------------------- strategies

/// A strategy description. Strategies carry state, so each backtest builds a fresh instance from
/// this recipe; the same object can therefore be reused across runs.
struct StrategySpec {
    std::function<std::unique_ptr<backtest::Strategy>()> make;
    std::string repr;

    [[nodiscard]] std::string name() const { return make()->get_name(); }
};

// ----------------------------------------------------------------------------- backtest

/// A finished backtest together with its risk summary.
struct BacktestRun {
    backtest::BacktestResult result;
    risk::PerformanceSummary metrics;

    [[nodiscard]] std::vector<double> equity() const {
        std::vector<double> out;
        out.reserve(result.equity_curve.size());
        for (const auto& point : result.equity_curve)
            out.push_back(point.equity);
        return out;
    }

    [[nodiscard]] std::vector<double> drawdown() const {
        std::vector<double> out;
        out.reserve(result.equity_curve.size());
        for (const auto& point : result.equity_curve)
            out.push_back(point.drawdown);
        return out;
    }

    [[nodiscard]] std::vector<std::string> dates() const {
        std::vector<std::string> out;
        out.reserve(result.equity_curve.size());
        for (const auto& point : result.equity_curve)
            out.push_back(point.date);
        return out;
    }
};

BacktestRun run_backtest(const data::MarketDataUniverse& universe,
                         const StrategySpec& spec,
                         const analysis::BacktestSetup& setup) {
    if (universe.empty()) throw py::value_error("the universe is empty");
    auto strategy = spec.make();
    py::gil_scoped_release release;
    backtest::BacktestEngine engine(universe, backtest::Portfolio(setup.initial_cash),
                                    backtest::ExecutionModel(setup.execution), setup.engine);
    BacktestRun run;
    run.result = engine.run(*strategy);
    run.metrics = risk::RiskReport::evaluate(run.result, setup.risk_free_rate);
    return run;
}

py::dict metrics_to_dict(const risk::PerformanceSummary& s) {
    return py::dict(
        "total_return"_a = s.total_return, "cagr"_a = s.cagr,
        "annualized_volatility"_a = s.annualized_volatility, "sharpe_ratio"_a = s.sharpe_ratio,
        "sortino_ratio"_a = s.sortino_ratio, "calmar_ratio"_a = s.calmar_ratio,
        "max_drawdown"_a = s.max_drawdown, "max_drawdown_duration"_a = s.drawdown_details.max_duration_bars,
        "var_95"_a = s.var_95_historical, "cvar_95"_a = s.cvar_95_historical,
        "var_95_parametric"_a = s.var_95_parametric, "var_95_cornish_fisher"_a = s.var_95_cornish_fisher,
        "skewness"_a = s.skewness, "excess_kurtosis"_a = s.excess_kurtosis, "exposure"_a = s.exposure,
        "years"_a = s.years, "transaction_costs"_a = s.total_transaction_costs,
        "closed_trades"_a = s.trade_details.closed_trades, "win_rate"_a = s.trade_details.win_rate,
        "profit_factor"_a = s.trade_details.profit_factor);
}

py::list fills_to_list(const std::vector<backtest::Fill>& fills) {
    py::list out;
    for (const auto& f : fills) {
        out.append(py::dict(
            "date"_a = f.date, "ticker"_a = f.ticker, "side"_a = backtest::side_to_string(f.side),
            "quantity"_a = f.quantity, "price"_a = f.price, "execution_price"_a = f.execution_price,
            "commission"_a = f.commission, "slippage"_a = f.slippage, "spread_cost"_a = f.spread_cost,
            "participation"_a = f.participation, "realized_pnl"_a = f.realized_pnl));
    }
    return out;
}

std::string format_performance(const analysis::WindowPerformance& p) {
    return "Performance(total_return=" + std::to_string(p.total_return) +
           ", sharpe_ratio=" + std::to_string(p.sharpe_ratio) +
           ", max_drawdown=" + std::to_string(p.max_drawdown) + ", trades=" + std::to_string(p.trades) +
           ", days=" + std::to_string(p.days) + ")";
}

}   // namespace

PYBIND11_MODULE(axiomquant, m) {
    m.doc() =
        "AxiomQuant: backtesting, risk metrics, walk-forward analysis and significance tests.\n\n"
        "Typical use::\n\n"
        "    import axiomquant as aq\n"
        "    universe = aq.load_universe(\"sample_data\")\n"
        "    result = aq.backtest(universe, aq.sma_crossover(\"SPY\", fast=20, slow=50))\n"
        "    result.equity            # NumPy array, one value per bar\n"
        "    result.metrics.sharpe_ratio\n";

    // ------------------------------------------------------------------ data
    py::class_<data::MarketDataUniverse>(m, "Universe",
                                         "Price histories aligned on a common timeline of trading days.")
        .def_property_readonly(
            "tickers", [](const data::MarketDataUniverse& u) { return u.get_tickers(); },
            "Symbols in the universe, sorted.")
        .def_property_readonly(
            "dates", [](const data::MarketDataUniverse& u) { return u.get_timeline(); },
            "ISO-8601 dates of the aligned timeline.")
        .def("__len__", &data::MarketDataUniverse::size)
        .def(
            "closes",
            [](const data::MarketDataUniverse& u, const std::string& ticker) {
                require_ticker(u, ticker);
                return to_array(u.get_aligned_closes(ticker));
            },
            "ticker"_a, "Close prices of one ticker on the aligned timeline (NaN where missing).")
        .def(
            "close_matrix",
            [](const data::MarketDataUniverse& u) { return to_matrix(u.get_aligned_close_matrix()); },
            "Close prices as a (bars, tickers) array, columns in `tickers` order.")
        .def(
            "returns_matrix",
            [](const data::MarketDataUniverse& u) { return to_matrix(u.get_aligned_returns_matrix()); },
            "Daily simple returns as a (bars - 1, tickers) array.")
        .def(
            "slice",
            [](const data::MarketDataUniverse& u, size_t begin, size_t end) {
                if (begin >= end || end > u.size()) throw py::index_error("slice bounds out of range");
                return u.slice(begin, end);
            },
            "begin"_a, "end"_a, "Sub-universe covering timeline positions [begin, end).")
        .def("__repr__", [](const data::MarketDataUniverse& u) {
            std::string tickers;
            for (const auto& t : u.get_tickers())
                tickers += (tickers.empty() ? "" : ", ") + t;
            std::string span =
                u.empty() ? "empty" : u.get_timeline().front() + " to " + u.get_timeline().back();
            return "Universe([" + tickers + "], " + std::to_string(u.size()) + " bars, " + span + ")";
        });

    m.def("load_universe", &load_universe, "directory"_a, "align"_a = "intersection",
          "tickers"_a = py::none(),
          "Load every <TICKER>.csv in `directory` and align them on one timeline.\n\n"
          "align: 'intersection' keeps dates where every symbol trades; 'union' keeps all dates and\n"
          "forward-fills gaps. tickers: optional subset of symbols to load.");

    // ------------------------------------------------------------------ costs
    py::class_<backtest::ExecutionConfig>(
        m, "CostModel",
        "Transaction costs: commission, bid-ask spread, slippage and square-root "
        "market impact. Defaults match the CLI.")
        .def(py::init([](double per_share_commission, double min_commission, double percentage_commission,
                         double fixed_slippage_bps, double spread_bps, double market_impact_factor,
                         bool enable_market_impact, double max_participation, double short_borrow_rate) {
                 backtest::ExecutionConfig c;
                 c.per_share_commission = per_share_commission;
                 c.min_commission = min_commission;
                 c.percentage_commission = percentage_commission;
                 c.fixed_slippage_bps = fixed_slippage_bps;
                 c.spread_bps = spread_bps;
                 c.market_impact_factor = market_impact_factor;
                 c.enable_market_impact = enable_market_impact;
                 if (max_participation < 0.0 || max_participation > 1.0)
                     throw py::value_error("max_participation must be in [0, 1]");
                 if (short_borrow_rate < 0.0) throw py::value_error("short_borrow_rate must be non-negative");
                 c.max_participation = max_participation;
                 c.short_borrow_rate = short_borrow_rate;
                 return c;
             }),
             py::kw_only(), "per_share_commission"_a = 0.005, "min_commission"_a = 1.0,
             "percentage_commission"_a = 0.0001, "fixed_slippage_bps"_a = 2.0, "spread_bps"_a = 3.0,
             "market_impact_factor"_a = 0.1, "enable_market_impact"_a = true, "max_participation"_a = 0.0,
             "short_borrow_rate"_a = 0.0)
        .def_static(
            "free",
            []() {
                backtest::ExecutionConfig c;
                c.per_share_commission = 0.0;
                c.min_commission = 0.0;
                c.percentage_commission = 0.0;
                c.fixed_slippage_bps = 0.0;
                c.spread_bps = 0.0;
                c.enable_market_impact = false;
                return c;
            },
            "A cost model that charges nothing, for frictionless comparisons.")
        .def_readwrite("per_share_commission", &backtest::ExecutionConfig::per_share_commission)
        .def_readwrite("min_commission", &backtest::ExecutionConfig::min_commission)
        .def_readwrite("percentage_commission", &backtest::ExecutionConfig::percentage_commission)
        .def_readwrite("fixed_slippage_bps", &backtest::ExecutionConfig::fixed_slippage_bps)
        .def_readwrite("spread_bps", &backtest::ExecutionConfig::spread_bps)
        .def_readwrite("market_impact_factor", &backtest::ExecutionConfig::market_impact_factor)
        .def_readwrite("enable_market_impact", &backtest::ExecutionConfig::enable_market_impact)
        .def_readwrite("max_participation", &backtest::ExecutionConfig::max_participation,
                       "Largest share of a bar's volume one order may take (0 = no cap); the rest is "
                       "carried to the next bar.")
        .def_readwrite("short_borrow_rate", &backtest::ExecutionConfig::short_borrow_rate,
                       "Annual fee on the market value of short positions.")
        .def("__repr__", [](const backtest::ExecutionConfig& c) {
            return "CostModel(per_share_commission=" + std::to_string(c.per_share_commission) +
                   ", min_commission=" + std::to_string(c.min_commission) +
                   ", percentage_commission=" + std::to_string(c.percentage_commission) +
                   ", fixed_slippage_bps=" + std::to_string(c.fixed_slippage_bps) +
                   ", spread_bps=" + std::to_string(c.spread_bps) +
                   ", market_impact_factor=" + std::to_string(c.market_impact_factor) +
                   ", enable_market_impact=" + (c.enable_market_impact ? "True" : "False") +
                   ", max_participation=" + std::to_string(c.max_participation) +
                   ", short_borrow_rate=" + std::to_string(c.short_borrow_rate) + ")";
        });

    // ------------------------------------------------------------------ strategies
    py::class_<StrategySpec>(m, "Strategy",
                             "A built-in strategy and its parameters. Create one with buy_and_hold(), "
                             "sma_crossover(), rsi_mean_reversion(), momentum() or vol_target().")
        .def_property_readonly("name", &StrategySpec::name, "Name as it appears in CLI reports.")
        .def("__repr__", [](const StrategySpec& s) { return s.repr; });

    m.def(
        "buy_and_hold",
        [](const std::string& ticker, double allocation) {
            return StrategySpec{
                [=]() { return std::make_unique<strategies::BuyAndHoldStrategy>(ticker, allocation); },
                "buy_and_hold('" + ticker + "', allocation=" + std::to_string(allocation) + ")"};
        },
        "ticker"_a, "allocation"_a = 0.99,
        "Buy `ticker` with `allocation` of equity on the first bar and hold.");

    m.def(
        "sma_crossover",
        [](const std::string& ticker, size_t fast, size_t slow, double allocation) {
            if (fast == 0 || fast >= slow) throw py::value_error("need 0 < fast < slow");
            return StrategySpec{[=]() {
                                    return std::make_unique<strategies::SmaCrossoverStrategy>(
                                        ticker, fast, slow, allocation);
                                },
                                "sma_crossover('" + ticker + "', fast=" + std::to_string(fast) +
                                    ", slow=" + std::to_string(slow) + ")"};
        },
        "ticker"_a, "fast"_a = 20, "slow"_a = 50, "allocation"_a = 0.95,
        "Long `ticker` while the fast moving average is above the slow one, flat otherwise.");

    m.def(
        "rsi_mean_reversion",
        [](const std::string& ticker, size_t period, double oversold, double overbought, double allocation) {
            if (!(oversold < overbought)) throw py::value_error("need oversold < overbought");
            return StrategySpec{
                [=]() {
                    return std::make_unique<strategies::RsiMeanReversionStrategy>(ticker, period, oversold,
                                                                                  overbought, allocation);
                },
                "rsi_mean_reversion('" + ticker + "', period=" + std::to_string(period) + ")"};
        },
        "ticker"_a, "period"_a = 14, "oversold"_a = 30.0, "overbought"_a = 70.0, "allocation"_a = 0.95,
        "Buy `ticker` when RSI falls below `oversold`, sell when it rises above `overbought`.");

    m.def(
        "momentum",
        [](size_t lookback, size_t rebalance, size_t top_k, double invested, bool require_positive,
           const std::optional<fs::path>& members) {
            if (lookback == 0 || rebalance == 0 || top_k == 0)
                throw py::value_error("lookback, rebalance and top_k must be positive");
            data::MembershipCalendar calendar;
            if (members) calendar = data::MembershipCalendar::load(*members);
            return StrategySpec{[=]() {
                                    auto s = std::make_unique<strategies::MultiAssetMomentumStrategy>(
                                        lookback, rebalance, top_k, invested, require_positive);
                                    s->set_membership(calendar);
                                    return s;
                                },
                                "momentum(lookback=" + std::to_string(lookback) + ", rebalance=" +
                                    std::to_string(rebalance) + ", top_k=" + std::to_string(top_k) + ")"};
        },
        "lookback"_a = 60, "rebalance"_a = 20, "top_k"_a = 2, "invested"_a = 0.95,
        "require_positive"_a = false, "members"_a = py::none(),
        "Every `rebalance` bars, hold the `top_k` symbols with the best `lookback`-bar return.\n\n"
        "members: optional point-in-time membership CSV (ticker,start_date,end_date); only symbols\n"
        "that were members on the rebalance date are ranked.");

    m.def(
        "vol_target",
        [](const std::string& ticker, double target_volatility, size_t window, double max_weight,
           double rebalance_threshold) {
            if (!(target_volatility > 0.0)) throw py::value_error("target_volatility must be positive");
            return StrategySpec{[=]() {
                                    return std::make_unique<strategies::VolatilityTargetStrategy>(
                                        ticker, target_volatility, window, max_weight, rebalance_threshold);
                                },
                                "vol_target('" + ticker +
                                    "', target_volatility=" + std::to_string(target_volatility) +
                                    ", window=" + std::to_string(window) + ")"};
        },
        "ticker"_a, "target_volatility"_a = 0.10, "window"_a = 20, "max_weight"_a = 1.0,
        "rebalance_threshold"_a = 0.05,
        "Scale exposure to `ticker` so trailing volatility matches `target_volatility` (annualised).");

    // ------------------------------------------------------------------ results
    py::class_<risk::PerformanceSummary>(m, "Metrics", "Risk and return statistics of a backtest.")
        .def_readonly("total_return", &risk::PerformanceSummary::total_return)
        .def_readonly("cagr", &risk::PerformanceSummary::cagr)
        .def_readonly("annualized_volatility", &risk::PerformanceSummary::annualized_volatility)
        .def_readonly("sharpe_ratio", &risk::PerformanceSummary::sharpe_ratio)
        .def_readonly("sortino_ratio", &risk::PerformanceSummary::sortino_ratio)
        .def_readonly("calmar_ratio", &risk::PerformanceSummary::calmar_ratio)
        .def_readonly("max_drawdown", &risk::PerformanceSummary::max_drawdown)
        .def_readonly("var_95", &risk::PerformanceSummary::var_95_historical)
        .def_readonly("cvar_95", &risk::PerformanceSummary::cvar_95_historical)
        .def_readonly("skewness", &risk::PerformanceSummary::skewness)
        .def_readonly("excess_kurtosis", &risk::PerformanceSummary::excess_kurtosis)
        .def_readonly("exposure", &risk::PerformanceSummary::exposure)
        .def_readonly("years", &risk::PerformanceSummary::years)
        .def_property_readonly("win_rate",
                               [](const risk::PerformanceSummary& s) { return s.trade_details.win_rate; })
        .def_property_readonly(
            "profit_factor", [](const risk::PerformanceSummary& s) { return s.trade_details.profit_factor; })
        .def("to_dict", &metrics_to_dict, "All metrics as a plain dict.")
        .def("__repr__", [](const risk::PerformanceSummary& s) {
            return "Metrics(total_return=" + std::to_string(s.total_return) +
                   ", sharpe_ratio=" + std::to_string(s.sharpe_ratio) +
                   ", max_drawdown=" + std::to_string(s.max_drawdown) + ")";
        });

    py::class_<BacktestRun>(m, "BacktestResult", "Output of backtest().")
        .def_property_readonly("strategy", [](const BacktestRun& r) { return r.result.strategy_name; })
        .def_property_readonly("start_date", [](const BacktestRun& r) { return r.result.start_date; })
        .def_property_readonly("end_date", [](const BacktestRun& r) { return r.result.end_date; })
        .def_property_readonly("initial_cash", [](const BacktestRun& r) { return r.result.initial_cash; })
        .def_property_readonly("final_equity", [](const BacktestRun& r) { return r.result.final_equity; })
        .def_property_readonly("total_return", [](const BacktestRun& r) { return r.result.total_return; })
        .def_property_readonly(
            "num_fills", [](const BacktestRun& r) { return r.result.total_trades; },
            "Number of executed fills.")
        .def_property_readonly(
            "capped_fills", [](const BacktestRun& r) { return r.result.capped_fills; },
            "Fills cut short by the CostModel's max_participation cap.")
        .def_property_readonly(
            "borrow_cost", [](const BacktestRun& r) { return r.result.total_borrow_cost; },
            "Total fee paid on short positions.")
        .def_property_readonly(
            "equity", [](const BacktestRun& r) { return to_array(r.equity()); },
            "Portfolio equity at each bar's close (NumPy array).")
        .def_property_readonly(
            "drawdown", [](const BacktestRun& r) { return to_array(r.drawdown()); },
            "Drawdown from the running peak at each bar (NumPy array, 0.1 = 10%).")
        .def_property_readonly(
            "returns", [](const BacktestRun& r) { return to_array(r.result.daily_returns); },
            "Daily simple returns of the equity curve (NumPy array, one shorter than `equity`).")
        .def_property_readonly("dates", &BacktestRun::dates, "Date of each equity point.")
        .def_property_readonly(
            "fills", [](const BacktestRun& r) { return fills_to_list(r.result.trades); },
            "Executed fills as a list of dicts (ready for pandas.DataFrame).")
        .def_readonly("metrics", &BacktestRun::metrics)
        .def(
            "report",
            [](const BacktestRun& r) { return risk::RiskReport::generate_text_report(r.result, r.metrics); },
            "The CLI's text report for this run.")
        .def("__repr__", [](const BacktestRun& r) {
            return "BacktestResult(" + r.result.strategy_name + ", " + r.result.start_date + " to " +
                   r.result.end_date + ", total_return=" + std::to_string(r.result.total_return) +
                   ", sharpe_ratio=" + std::to_string(r.metrics.sharpe_ratio) + ")";
        });

    m.def(
        "backtest",
        [](const data::MarketDataUniverse& universe, const StrategySpec& strategy, double capital,
           const std::optional<backtest::ExecutionConfig>& costs, const std::string& fill, double risk_free) {
            return run_backtest(universe, strategy, make_setup(capital, costs, fill, risk_free));
        },
        "universe"_a, "strategy"_a, py::kw_only(), "capital"_a = 100000.0, "costs"_a = py::none(),
        "fill"_a = "open", "risk_free"_a = 0.02,
        "Run `strategy` over `universe` and return a BacktestResult.\n\n"
        "costs: a CostModel (default: CostModel()). fill: 'open' fills orders at the next bar's open\n"
        "(no look-ahead, the default); 'close' fills at the signal bar's close. risk_free: annual rate\n"
        "used for the Sharpe ratio.");

    // ------------------------------------------------------------------ risk functions
    m.def("sharpe_ratio", &risk::RiskMetrics::sharpe_ratio, "returns"_a, "risk_free"_a = 0.0,
          "ann_factor"_a = 252.0, "Annualised Sharpe ratio of daily simple returns.");
    m.def("sortino_ratio", &risk::RiskMetrics::sortino_ratio, "returns"_a, "mar"_a = 0.0,
          "ann_factor"_a = 252.0, "Annualised Sortino ratio of daily simple returns.");
    m.def("annualized_volatility", &risk::RiskMetrics::annualized_volatility, "returns"_a,
          "ann_factor"_a = 252.0, "Annualised volatility of daily simple returns.");
    m.def(
        "max_drawdown",
        [](const std::vector<double>& equity) {
            return risk::RiskMetrics::calculate_drawdown(equity).max_drawdown;
        },
        "equity"_a, "Largest peak-to-trough decline of an equity curve (0.2 = 20%).");

    // ------------------------------------------------------------------ walk-forward
    py::class_<analysis::WindowPerformance>(m, "Performance", "Summary statistics of one return window.")
        .def_readonly("total_return", &analysis::WindowPerformance::total_return)
        .def_readonly("sharpe_ratio", &analysis::WindowPerformance::sharpe_ratio)
        .def_readonly("max_drawdown", &analysis::WindowPerformance::max_drawdown)
        .def_readonly("trades", &analysis::WindowPerformance::trades)
        .def_readonly("days", &analysis::WindowPerformance::days)
        .def("__repr__", &format_performance);

    py::class_<analysis::SmaSweepPoint>(m, "SweepPoint", "One (fast, slow) pair of an SMA parameter sweep.")
        .def_readonly("fast", &analysis::SmaSweepPoint::fast_period)
        .def_readonly("slow", &analysis::SmaSweepPoint::slow_period)
        .def_readonly("performance", &analysis::SmaSweepPoint::performance)
        .def("__repr__", [](const analysis::SmaSweepPoint& p) {
            return "SweepPoint(fast=" + std::to_string(p.fast_period) +
                   ", slow=" + std::to_string(p.slow_period) +
                   ", sharpe_ratio=" + std::to_string(p.performance.sharpe_ratio) + ")";
        });

    py::class_<analysis::WalkForwardFold>(m, "WalkForwardFold",
                                          "One train/test fold of a walk-forward study.")
        .def_readonly("train_start", &analysis::WalkForwardFold::train_start)
        .def_readonly("train_end", &analysis::WalkForwardFold::train_end)
        .def_readonly("test_start", &analysis::WalkForwardFold::test_start)
        .def_readonly("test_end", &analysis::WalkForwardFold::test_end)
        .def_readonly("fast", &analysis::WalkForwardFold::fast_period)
        .def_readonly("slow", &analysis::WalkForwardFold::slow_period)
        .def_readonly("in_sample", &analysis::WalkForwardFold::in_sample)
        .def_readonly("out_of_sample", &analysis::WalkForwardFold::out_of_sample)
        .def_readonly("benchmark", &analysis::WalkForwardFold::benchmark)
        .def("__repr__", [](const analysis::WalkForwardFold& f) {
            return "WalkForwardFold(test " + f.test_start + " to " + f.test_end + ", " +
                   std::to_string(f.fast_period) + "/" + std::to_string(f.slow_period) +
                   ", oos_sharpe=" + std::to_string(f.out_of_sample.sharpe_ratio) + ")";
        });

    py::class_<analysis::WalkForwardResult>(m, "WalkForwardResult", "Output of walk_forward().")
        .def_readonly("folds", &analysis::WalkForwardResult::folds)
        .def_property_readonly(
            "oos_returns", [](const analysis::WalkForwardResult& r) { return to_array(r.oos_returns); },
            "Stitched out-of-sample strategy returns (NumPy array).")
        .def_property_readonly(
            "benchmark_returns",
            [](const analysis::WalkForwardResult& r) { return to_array(r.benchmark_returns); },
            "Buy-and-hold returns over the same out-of-sample days (NumPy array).")
        .def_readonly("strategy", &analysis::WalkForwardResult::stitched_strategy)
        .def_readonly("benchmark", &analysis::WalkForwardResult::stitched_benchmark)
        .def_readonly("mean_in_sample_sharpe", &analysis::WalkForwardResult::mean_in_sample_sharpe)
        .def_readonly("mean_out_of_sample_sharpe", &analysis::WalkForwardResult::mean_out_of_sample_sharpe)
        .def_readonly("folds_beating_benchmark", &analysis::WalkForwardResult::folds_beating_benchmark)
        .def("__repr__", [](const analysis::WalkForwardResult& r) {
            return "WalkForwardResult(" + std::to_string(r.folds.size()) +
                   " folds, oos_sharpe=" + std::to_string(r.stitched_strategy.sharpe_ratio) +
                   ", benchmark_sharpe=" + std::to_string(r.stitched_benchmark.sharpe_ratio) + ")";
        });

    m.def(
        "sma_sweep",
        [](const data::MarketDataUniverse& universe, const std::string& ticker,
           const std::vector<size_t>& fast_grid, const std::vector<size_t>& slow_grid, double capital,
           const std::optional<backtest::ExecutionConfig>& costs, const std::string& fill, double risk_free) {
            require_ticker(universe, ticker);
            const auto setup = make_setup(capital, costs, fill, risk_free);
            py::gil_scoped_release release;
            return analysis::sweep_sma_parameters(universe, ticker, fast_grid, slow_grid, setup, 0,
                                                  universe.size(), 0);
        },
        "universe"_a, "ticker"_a, "fast_grid"_a = std::vector<size_t>{5, 10, 15, 20, 30, 40, 50},
        "slow_grid"_a = std::vector<size_t>{50, 75, 100, 125, 150, 200}, py::kw_only(),
        "capital"_a = 100000.0, "costs"_a = py::none(), "fill"_a = "open", "risk_free"_a = 0.02,
        "Backtest an SMA crossover for every fast < slow pair over the whole universe, best Sharpe first.");

    m.def(
        "walk_forward",
        [](const data::MarketDataUniverse& universe, const std::string& ticker, size_t train_bars,
           size_t test_bars, const std::vector<size_t>& fast_grid, const std::vector<size_t>& slow_grid,
           bool carry_position, double capital, const std::optional<backtest::ExecutionConfig>& costs,
           const std::string& fill, double risk_free) {
            require_ticker(universe, ticker);
            if (train_bars < 2 || test_bars < 2)
                throw py::value_error("train_bars and test_bars must be >= 2");
            const auto setup = make_setup(capital, costs, fill, risk_free);
            analysis::WalkForwardConfig config;
            config.train_bars = train_bars;
            config.test_bars = test_bars;
            config.fast_grid = fast_grid;
            config.slow_grid = slow_grid;
            config.carry_position = carry_position;
            py::gil_scoped_release release;
            return analysis::run_sma_walk_forward(universe, ticker, config, setup);
        },
        "universe"_a, "ticker"_a, py::kw_only(), "train_bars"_a = 504, "test_bars"_a = 126,
        "fast_grid"_a = std::vector<size_t>{5, 10, 20, 30, 50},
        "slow_grid"_a = std::vector<size_t>{50, 100, 150, 200}, "carry_position"_a = false,
        "capital"_a = 100000.0, "costs"_a = py::none(), "fill"_a = "open", "risk_free"_a = 0.02,
        "Rolling walk-forward optimisation of the SMA crossover on `ticker`.\n\n"
        "Each fold picks the best (fast, slow) pair by in-sample Sharpe on `train_bars` bars, then\n"
        "trades it unchanged on the next `test_bars` bars. carry_position lets each test window\n"
        "inherit the rule's position instead of starting flat.");

    m.def(
        "walk_forward_report",
        [](const analysis::WalkForwardResult& result, const std::string& ticker) {
            return analysis::format_walk_forward_report(result, ticker);
        },
        "result"_a, "ticker"_a, "The CLI's text report for a walk-forward result.");

    // ------------------------------------------------------------------ significance
    py::class_<analysis::SharpeInterval>(m, "SharpeInterval",
                                         "A Sharpe estimate with its bootstrap interval.")
        .def_readonly("estimate", &analysis::SharpeInterval::estimate)
        .def_readonly("lower", &analysis::SharpeInterval::lower)
        .def_readonly("upper", &analysis::SharpeInterval::upper)
        .def_readonly("p_value", &analysis::SharpeInterval::p_value)
        .def("__repr__", [](const analysis::SharpeInterval& s) {
            return "SharpeInterval(estimate=" + std::to_string(s.estimate) +
                   ", lower=" + std::to_string(s.lower) + ", upper=" + std::to_string(s.upper) +
                   ", p_value=" + std::to_string(s.p_value) + ")";
        });

    py::class_<analysis::BootstrapSharpeResult>(m, "BootstrapResult", "Output of bootstrap_sharpe().")
        .def_readonly("observations", &analysis::BootstrapSharpeResult::observations)
        .def_readonly("resamples", &analysis::BootstrapSharpeResult::resamples)
        .def_readonly("mean_block_length", &analysis::BootstrapSharpeResult::mean_block_length)
        .def_readonly("confidence", &analysis::BootstrapSharpeResult::confidence)
        .def_readonly("has_benchmark", &analysis::BootstrapSharpeResult::has_benchmark)
        .def_readonly("strategy", &analysis::BootstrapSharpeResult::strategy)
        .def_readonly("benchmark", &analysis::BootstrapSharpeResult::benchmark)
        .def_readonly("difference", &analysis::BootstrapSharpeResult::difference)
        .def_readonly("probabilistic_sharpe", &analysis::BootstrapSharpeResult::probabilistic_sharpe)
        .def("report", &analysis::format_bootstrap_report, "The CLI's text report for this result.");

    m.def(
        "bootstrap_sharpe",
        [](const std::vector<double>& returns, const std::optional<std::vector<double>>& benchmark,
           size_t resamples, double block_length, double confidence, uint64_t seed, double risk_free,
           double ann_factor) {
            if (benchmark && benchmark->size() != returns.size())
                throw py::value_error("benchmark must have the same length as returns");
            analysis::BootstrapConfig config;
            config.resamples = resamples;
            config.mean_block_length = block_length;
            config.confidence = confidence;
            config.seed = seed;
            config.risk_free_rate = risk_free;
            config.ann_factor = ann_factor;
            const std::vector<double> bench = benchmark.value_or(std::vector<double>{});
            py::gil_scoped_release release;
            return analysis::bootstrap_sharpe(returns, bench, config);
        },
        "returns"_a, "benchmark"_a = py::none(), py::kw_only(), "resamples"_a = 10000, "block_length"_a = 0.0,
        "confidence"_a = 0.95, "seed"_a = 42, "risk_free"_a = 0.02, "ann_factor"_a = 252.0,
        "Stationary block bootstrap of the annualised Sharpe ratio.\n\n"
        "With a benchmark, both series are resampled with the same indices and the difference is\n"
        "tested too. block_length 0 picks the Politis-White length from the data. Results are\n"
        "identical at any thread count for a given seed.");

    m.def("probabilistic_sharpe_ratio", &analysis::probabilistic_sharpe_ratio, "returns"_a,
          "benchmark_sharpe"_a = 0.0, "risk_free"_a = 0.02, "ann_factor"_a = 252.0,
          "Probability that the true annualised Sharpe ratio exceeds `benchmark_sharpe`.");

    py::class_<analysis::DeflatedSharpe>(m, "DeflatedSharpe", "Output of deflated_sharpe_ratio().")
        .def_readonly("observations", &analysis::DeflatedSharpe::observations)
        .def_readonly("trials", &analysis::DeflatedSharpe::trials)
        .def_readonly("effective_trials", &analysis::DeflatedSharpe::effective_trials)
        .def_readonly("sharpe", &analysis::DeflatedSharpe::sharpe)
        .def_readonly("trial_sharpe_std", &analysis::DeflatedSharpe::trial_sharpe_std)
        .def_readonly("expected_max_sharpe", &analysis::DeflatedSharpe::expected_max_sharpe)
        .def_readonly("probabilistic_sharpe", &analysis::DeflatedSharpe::probabilistic_sharpe)
        .def_readonly("deflated_sharpe", &analysis::DeflatedSharpe::deflated_sharpe)
        .def("report", &analysis::format_deflated_sharpe_report, "The CLI's text report for this result.");

    m.def("deflated_sharpe_ratio", &analysis::deflated_sharpe_ratio, "returns"_a, "trial_sharpes"_a,
          "risk_free"_a = 0.02, "ann_factor"_a = 252.0, "effective_trials"_a = 0.0,
          "Deflated Sharpe ratio of the configuration selected from `trial_sharpes` (annualised).");

    m.def("politis_white_block_length", &analysis::politis_white_block_length, "series"_a,
          "Optimal stationary-bootstrap mean block length (NaN if the series is too short).");

    m.def("effective_trials", &analysis::effective_trials, "trial_returns"_a,
          "Effective number of independent trials behind correlated trial return series.");
}
