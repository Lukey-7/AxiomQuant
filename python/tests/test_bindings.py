"""Tests for the axiomquant Python module (stdlib unittest; NumPy is the only dependency).

The module is found on PYTHONPATH, or in <build>/python where <build> is $AXIOMQUANT_BUILD_DIR
(default: build/ at the repository root). The CLI cross-check uses $AXIOMQUANT_CLI, or
<build>/bin/axiomquant, and is skipped when neither exists.

    cmake -S . -B build -DAXIOM_BUILD_PYTHON=ON && cmake --build build --parallel
    python -m unittest discover -s python/tests -v
"""

import csv
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
BUILD = pathlib.Path(os.environ.get("AXIOMQUANT_BUILD_DIR", REPO / "build"))
DATA = pathlib.Path(os.environ.get("AXIOMQUANT_DATA", REPO / "sample_data"))

if (BUILD / "python").is_dir():
    sys.path.insert(0, str(BUILD / "python"))

import numpy as np  # noqa: E402

import axiomquant as aq  # noqa: E402


def find_cli():
    """Path of the axiomquant command-line tool, or None when it was not built."""
    if os.environ.get("AXIOMQUANT_CLI"):
        return pathlib.Path(os.environ["AXIOMQUANT_CLI"])
    for candidate in ("bin/axiomquant", "bin/axiomquant.exe", "bin/Release/axiomquant.exe"):
        path = BUILD / candidate
        if path.is_file():
            return path
    return None


# The CLI's strategy tournament, rebuilt through the Python API with the same parameters.
def cli_strategies(ticker="SPY"):
    return [
        aq.buy_and_hold(ticker, allocation=0.99),
        aq.sma_crossover(ticker, fast=20, slow=50, allocation=0.95),
        aq.rsi_mean_reversion(ticker, period=14, oversold=30.0, overbought=70.0, allocation=0.95),
        aq.momentum(lookback=60, rebalance=20, top_k=2, invested=0.95),
        aq.vol_target(ticker, target_volatility=0.10, window=20),
    ]


class UniverseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = aq.load_universe(DATA)

    def test_loads_sample_data(self):
        u = self.universe
        self.assertEqual(u.tickers, ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY"])
        self.assertEqual(len(u), len(u.dates))
        self.assertGreater(len(u), 1000)
        closes = u.closes("SPY")
        self.assertIsInstance(closes, np.ndarray)
        self.assertEqual(closes.shape, (len(u),))
        self.assertEqual(u.close_matrix().shape, (len(u), 5))
        np.testing.assert_array_equal(u.close_matrix()[:, 4], closes)

    def test_slice_and_errors(self):
        part = self.universe.slice(10, 110)
        self.assertEqual(len(part), 100)
        self.assertEqual(part.dates[0], self.universe.dates[10])
        with self.assertRaises(KeyError):
            self.universe.closes("NOPE")
        with self.assertRaises(ValueError):
            aq.load_universe(DATA, align="sideways")

    def test_ticker_subset(self):
        u = aq.load_universe(DATA, tickers=["SPY", "AAPL"])
        self.assertEqual(u.tickers, ["AAPL", "SPY"])


class BacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = aq.load_universe(DATA)
        cls.result = aq.backtest(cls.universe, aq.sma_crossover("SPY", fast=20, slow=50))

    def test_equity_curve_is_numpy(self):
        r = self.result
        self.assertIsInstance(r.equity, np.ndarray)
        self.assertEqual(r.equity.dtype, np.float64)
        self.assertEqual(len(r.equity), len(self.universe))
        self.assertEqual(len(r.returns), len(r.equity) - 1)
        self.assertEqual(len(r.dates), len(r.equity))
        self.assertAlmostEqual(r.equity[0], 100000.0)
        self.assertAlmostEqual(r.equity[-1], r.final_equity)
        self.assertAlmostEqual(r.total_return, r.final_equity / 100000.0 - 1.0)

    def test_metrics_agree_with_the_curve(self):
        r = self.result
        self.assertAlmostEqual(r.metrics.max_drawdown, aq.max_drawdown(r.equity), places=12)
        self.assertAlmostEqual(r.metrics.sharpe_ratio, aq.sharpe_ratio(r.returns, 0.02), places=12)
        self.assertAlmostEqual(r.metrics.to_dict()["total_return"], r.total_return, places=12)
        self.assertEqual(r.num_fills, len(r.fills))
        self.assertIn(r.fills[0]["side"], ("BUY", "SELL"))

    def test_strategy_spec_is_reusable(self):
        spec = aq.sma_crossover("SPY", fast=20, slow=50)
        first = aq.backtest(self.universe, spec)
        second = aq.backtest(self.universe, spec)
        np.testing.assert_array_equal(first.equity, second.equity)

    def test_costs_and_fill_timing(self):
        spec = aq.sma_crossover("SPY")
        free = aq.backtest(self.universe, spec, costs=aq.CostModel.free())
        dear = aq.backtest(self.universe, spec, costs=aq.CostModel(spread_bps=50.0))
        self.assertGreater(free.final_equity, self.result.final_equity)
        self.assertLess(dear.final_equity, self.result.final_equity)
        close = aq.backtest(self.universe, spec, fill="close")
        self.assertNotEqual(close.final_equity, self.result.final_equity)
        with self.assertRaises(ValueError):
            aq.backtest(self.universe, spec, fill="later")
        with self.assertRaises(ValueError):
            aq.sma_crossover("SPY", fast=50, slow=20)


class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = aq.load_universe(DATA)

    def test_walk_forward(self):
        wf = aq.walk_forward(self.universe, "SPY", train_bars=504, test_bars=126)
        self.assertGreater(len(wf.folds), 0)
        self.assertEqual(len(wf.oos_returns), len(wf.benchmark_returns))
        self.assertEqual(len(wf.oos_returns), sum(f.out_of_sample.days for f in wf.folds))
        for fold in wf.folds:
            self.assertLess(fold.fast, fold.slow)
        self.assertIn("SPY", aq.walk_forward_report(wf, "SPY"))

    def test_sweep_sorted_by_sharpe(self):
        sweep = aq.sma_sweep(self.universe, "SPY", [10, 20], [50, 100])
        self.assertEqual(len(sweep), 4)
        sharpes = [p.performance.sharpe_ratio for p in sweep]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))

    def test_significance(self):
        r = aq.backtest(self.universe, aq.sma_crossover("SPY")).returns
        b = aq.backtest(self.universe, aq.buy_and_hold("SPY")).returns
        boot = aq.bootstrap_sharpe(r, b, resamples=300, seed=7)
        again = aq.bootstrap_sharpe(r, b, resamples=300, seed=7)
        self.assertTrue(boot.has_benchmark)
        self.assertEqual(boot.strategy.p_value, again.strategy.p_value)
        self.assertLessEqual(boot.strategy.lower, boot.strategy.estimate)
        self.assertGreaterEqual(boot.strategy.upper, boot.strategy.estimate)
        self.assertAlmostEqual(boot.strategy.estimate, aq.sharpe_ratio(r, 0.02), places=10)
        psr = aq.probabilistic_sharpe_ratio(r)
        self.assertTrue(0.0 <= psr <= 1.0)
        dsr = aq.deflated_sharpe_ratio(r, [0.1, 0.3, -0.2, 0.27])
        self.assertLessEqual(dsr.deflated_sharpe, dsr.probabilistic_sharpe)
        self.assertGreaterEqual(aq.effective_trials([list(r), list(b)]), 1.0)
        self.assertTrue(math.isfinite(aq.politis_white_block_length(r)))


@unittest.skipIf(find_cli() is None, "axiomquant CLI not built")
class MatchesCliTest(unittest.TestCase):
    """The same strategies through Python and through the CLI must give the same numbers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = pathlib.Path(cls.tmp.name)
        subprocess.run(
            [str(find_cli()), "--data", str(DATA), "--no-db", "--export-dir", str(out),
             "--paths", "200", "--bootstrap", "200"],
            check=True, stdout=subprocess.DEVNULL, cwd=out,
        )
        with open(out / "strategy_summary.csv", newline="") as f:
            cls.cli = {row["strategy"]: row for row in csv.DictReader(f)}
        with open(out / "equity_curves.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        cls.cli_equity = {name: np.array([float(r[name]) for r in rows]) for name in rows[0] if name != "date"}
        universe = aq.load_universe(DATA)
        cls.runs = [aq.backtest(universe, s, capital=100000.0, risk_free=0.02) for s in cli_strategies()]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_strategies(self):
        self.assertEqual(sorted(self.cli), sorted(r.strategy for r in self.runs))

    def test_summary_metrics_match(self):
        # The CLI writes six decimals, so agree to rounding.
        for run in self.runs:
            with self.subTest(strategy=run.strategy):
                row = self.cli[run.strategy]
                m = run.metrics
                self.assertAlmostEqual(m.total_return, float(row["total_return"]), delta=1e-6)
                self.assertAlmostEqual(m.cagr, float(row["cagr"]), delta=1e-6)
                self.assertAlmostEqual(m.sharpe_ratio, float(row["sharpe"]), delta=1e-6)
                self.assertAlmostEqual(m.sortino_ratio, float(row["sortino"]), delta=1e-6)
                self.assertAlmostEqual(m.max_drawdown, float(row["max_drawdown"]), delta=1e-6)
                self.assertEqual(run.num_fills, int(row["fills"]))

    def test_equity_curves_match(self):
        for run in self.runs:
            with self.subTest(strategy=run.strategy):
                np.testing.assert_allclose(run.equity, self.cli_equity[run.strategy], rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
