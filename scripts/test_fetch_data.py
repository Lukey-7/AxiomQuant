#!/usr/bin/env python3
"""Offline tests for scripts/fetch_data.py. No network access.

    python3 -m unittest discover -s scripts -p "test_*.py" -v
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data  # noqa: E402


def run(argv: list[str]) -> int:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fetch_data.main(argv)


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class MakeBarTests(unittest.TestCase):
    def test_adjustment_scales_every_price(self) -> None:
        bar = fetch_data.make_bar("2020-01-02", 10.0, 12.0, 9.0, 11.0, 100.0, adj_close=5.5)
        self.assertEqual(bar, ("2020-01-02", 5.0, 6.0, 4.5, 5.5, 100.0))

    def test_rejects_bad_rows(self) -> None:
        self.assertIsNone(fetch_data.make_bar("2020-01-02", 1.0, 2.0, 1.0, None, 0.0))
        self.assertIsNone(fetch_data.make_bar("2020-01-02", 1.0, 2.0, 1.0, -3.0, 0.0))
        self.assertIsNone(fetch_data.make_bar("2020-01-02", 1.0, 1.0, 2.0, 1.5, 0.0))

    def test_finalise_sorts_and_deduplicates(self) -> None:
        a = ("2020-01-03", 1.0, 1.0, 1.0, 1.0, 0.0)
        b = ("2020-01-02", 2.0, 2.0, 2.0, 2.0, 0.0)
        c = ("2020-01-03", 3.0, 3.0, 3.0, 3.0, 0.0)
        self.assertEqual(fetch_data.finalise([a, b, c]), [b, c])


class YahooTests(unittest.TestCase):
    def payload(self) -> str:
        return json.dumps({"chart": {"error": None, "result": [{
            "meta": {"gmtoffset": -14400},
            "timestamp": [1577975400, 1578061800, 1578321000],
            "indicators": {
                "quote": [{"open": [10.0, None, 12.0], "high": [11.0, None, 13.0],
                           "low": [9.0, None, 11.0], "close": [10.0, None, 12.5],
                           "volume": [100, None, 300]}],
                "adjclose": [{"adjclose": [5.0, None, 12.5]}],
            },
        }]}})

    def test_parses_and_adjusts(self) -> None:
        bars = fetch_data.parse_yahoo(self.payload())
        self.assertEqual([b[0] for b in bars], ["2020-01-02", "2020-01-06"])
        self.assertAlmostEqual(bars[0][1], 5.0)
        self.assertAlmostEqual(bars[0][4], 5.0)
        self.assertAlmostEqual(bars[1][4], 12.5)

    def test_html_challenge_is_an_error(self) -> None:
        with self.assertRaises(fetch_data.SourceError):
            fetch_data.parse_yahoo("<!DOCTYPE html><html><body>verify</body></html>")

    def test_reported_error_is_an_error(self) -> None:
        with self.assertRaises(fetch_data.SourceError):
            fetch_data.parse_yahoo(json.dumps({"chart": {"result": None,
                                                         "error": {"description": "No data found"}}}))


class TiingoTests(unittest.TestCase):
    def test_uses_adjusted_fields(self) -> None:
        payload = json.dumps([{"date": "2020-01-02T00:00:00.000Z", "adjOpen": 1.0, "adjHigh": 2.0,
                               "adjLow": 0.5, "adjClose": 1.5, "adjVolume": 10}])
        self.assertEqual(fetch_data.parse_tiingo(payload), [("2020-01-02", 1.0, 2.0, 0.5, 1.5, 10)])


class ImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_long_format_kaggle_file(self) -> None:
        # Layout of the widely used Kaggle "S&P 500 stock data" all_stocks_5yr.csv.
        src = self.write("all_stocks_5yr.csv",
                         "date,open,high,low,close,volume,Name\n"
                         "2013-02-08,15.07,15.12,14.63,14.75,8407500,AAL\n"
                         "2013-02-11,14.89,15.01,14.26,14.46,8882000,AAL\n"
                         "2013-02-08,67.7142,68.4014,66.8928,67.8542,158168416,AAPL\n"
                         "2013-02-11,68.0714,69.2771,67.6071,68.5614,129029425,AAPL\n")
        out = self.root / "out"
        code = run(["import", str(src), "--out", str(out), "--min-rows", "2", "--tickers", "AAPL"])
        self.assertEqual(code, 0)
        self.assertFalse((out / "AAL.csv").exists())
        rows = read_rows(out / "AAPL.csv")
        self.assertEqual(rows[0], fetch_data.HEADER)
        self.assertEqual(rows[1], ["2013-02-08", "67.714200", "68.401400", "66.892800",
                                   "67.854200", "67.854200", "158168416"])

    def test_one_file_per_symbol_with_adj_close_and_us_dates(self) -> None:
        folder = self.root / "in"
        folder.mkdir()
        (folder / "msft.csv").write_text(
            "﻿Date,Open,High,Low,Close,Adj Close,Volume\n"
            "01/03/2020,10,11,9,10,5,1000\n"
            "01/02/2020,10,11,9,10,5,1000\n"
            "garbage,,,,,,\n", encoding="utf-8")
        out = self.root / "out"
        self.assertEqual(run(["import", str(folder), "--out", str(out), "--min-rows", "2"]), 0)
        rows = read_rows(out / "MSFT.csv")
        self.assertEqual([r[0] for r in rows[1:]], ["2020-01-02", "2020-01-03"])
        self.assertEqual(rows[1][4], "5.000000")
        self.assertEqual(rows[1][1], "5.000000")

    def test_nasdaq_style_dollar_prices(self) -> None:
        src = self.write("QQQ.csv",
                         "Date,Close/Last,Volume,Open,High,Low\n"
                         "09/12/2025,$586.66,1000,$585.00,$588.00,$584.00\n"
                         "09/11/2025,$584.00,1000,$580.00,$585.00,$579.00\n")
        out = self.root / "out"
        self.assertEqual(run(["import", str(src), "--out", str(out), "--min-rows", "2"]), 0)
        self.assertEqual(read_rows(out / "QQQ.csv")[2][4], "586.660000")

    def test_too_few_rows_fails(self) -> None:
        src = self.write("SPY.csv", "Date,Close\n2020-01-02,1\n")
        self.assertEqual(run(["import", str(src), "--out", str(self.root / "o")]), 1)

    def test_missing_columns_fails(self) -> None:
        src = self.write("X.csv", "when,value\n2020-01-02,1\n")
        self.assertEqual(run(["import", str(src), "--out", str(self.root / "o"), "--min-rows", "1"]), 1)


class SyntheticTests(unittest.TestCase):
    def test_same_seed_same_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (Path(tmp) / n for n in ("a", "b", "c"))
            base = ["synthetic", "--tickers", "AAA", "BBB", "--start", "2020-01-01", "--end", "2021-12-31"]
            self.assertEqual(run(base + ["--out", str(a), "--seed", "7"]), 0)
            self.assertEqual(run(base + ["--out", str(b), "--seed", "7"]), 0)
            self.assertEqual(run(base + ["--out", str(c), "--seed", "8"]), 0)
            self.assertEqual((a / "AAA.csv").read_bytes(), (b / "AAA.csv").read_bytes())
            self.assertNotEqual((a / "AAA.csv").read_bytes(), (c / "AAA.csv").read_bytes())
            rows = read_rows(a / "BBB.csv")[1:]
            self.assertEqual(len(rows), 523)  # weekdays in 2020 (262) and 2021 (261)
            for row in rows:
                o, h, l, cl = map(float, row[1:5])
                self.assertTrue(l <= min(o, cl) and h >= max(o, cl) and l > 0)


class CompatibilityTests(unittest.TestCase):
    def test_flag_only_invocation_means_live(self) -> None:
        args = fetch_data.build_parser().parse_args(["live", "--out", "x"])
        self.assertEqual(args.func, fetch_data.command_live)
        # main() inserts "live" when the first argument is a flag.
        with tempfile.TemporaryDirectory() as tmp:
            original = fetch_data.PROVIDERS["yahoo"]
            fetch_data.PROVIDERS["yahoo"] = lambda t, s, e: [
                (f"2020-01-{d:02d}", 1.0, 1.0, 1.0, 1.0, 0.0) for d in range(1, 4)]
            try:
                code = run(["--out", tmp, "--tickers", "SPY", "--min-rows", "3", "--provider", "yahoo"])
            finally:
                fetch_data.PROVIDERS["yahoo"] = original
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "SPY.csv").exists())


if __name__ == "__main__":
    unittest.main()
