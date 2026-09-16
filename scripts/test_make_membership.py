#!/usr/bin/env python3
"""Offline tests for scripts/make_membership.py.

    python3 -m unittest discover -s scripts -p "test_*.py" -v
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_membership  # noqa: E402


def _run(argv: list[str]) -> int:
    """Runs the script with the given arguments, keeping its output out of the test log."""
    original = sys.argv
    sys.argv = ["make_membership.py", *argv]
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return make_membership.main()
    finally:
        sys.argv = original


def read(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class ParsingTests(unittest.TestCase):
    def test_parses_common_date_formats(self) -> None:
        for text in ["2008-09-15", "09/15/2008", "09-15-2008", "2008/09/15", "20080915"]:
            self.assertEqual(make_membership.parse_date(text), "2008-09-15")
        self.assertEqual(make_membership.parse_date("2008-09-15 00:00:00"), "2008-09-15")
        with self.assertRaises(make_membership.InputError):
            make_membership.parse_date("not a date")

    def test_splits_symbol_lists(self) -> None:
        self.assertEqual(make_membership.split_symbols('"AAPL,msft; ibm|T"'), ["AAPL", "MSFT", "IBM", "T"])
        self.assertEqual(make_membership.split_symbols(""), [])


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_snapshots_become_spells(self) -> None:
        src = self.root / "snapshots.csv"
        src.write_text(
            'date,tickers\n'
            '2000-01-03,"AAPL,LEH,NFLX"\n'
            '2004-01-02,"AAPL,LEH"\n'          # NFLX leaves
            '2011-01-03,"AAPL,NFLX"\n'         # LEH leaves, NFLX returns
            , encoding="utf-8")
        out = self.root / "members.csv"
        self.assertEqual(_run([str(src), "--out", str(out)]), 0)

        rows = read(out)
        self.assertEqual(rows[0], ["ticker", "start_date", "end_date"])
        self.assertEqual(rows[1:], [
            ["AAPL", "2000-01-03", ""],
            ["LEH", "2000-01-03", "2011-01-02"],
            ["NFLX", "2000-01-03", "2004-01-01"],
            ["NFLX", "2011-01-03", ""],
        ])

    def test_changes_become_spells(self) -> None:
        src = self.root / "changes.csv"
        src.write_text(
            "date,added,removed\n"
            "2008-09-15,,LEH\n"     # removed without ever being added: member from before the record
            "2010-12-20,NFLX,\n",
            encoding="utf-8")
        out = self.root / "members.csv"
        self.assertEqual(_run([str(src), "--out", str(out)]), 0)
        self.assertEqual(read(out)[1:], [["LEH", "", "2008-09-14"], ["NFLX", "2010-12-20", ""]])

    def test_ticker_filter_and_unsorted_input(self) -> None:
        src = self.root / "snapshots.csv"
        src.write_text('date,tickers\n2011-01-03,"AAPL NFLX"\n2000-01-03,"AAPL LEH NFLX"\n', encoding="utf-8")
        out = self.root / "members.csv"
        self.assertEqual(_run([str(src), "--out", str(out), "--tickers", "leh"]), 0)
        self.assertEqual(read(out)[1:], [["LEH", "2000-01-03", "2011-01-02"]])

    def test_unusable_input_fails(self) -> None:
        src = self.root / "bad.csv"
        src.write_text("when,what\n2000-01-03,AAPL\n", encoding="utf-8")
        self.assertEqual(_run([str(src), "--out", str(self.root / "o.csv")]), 1)


if __name__ == "__main__":
    unittest.main()
