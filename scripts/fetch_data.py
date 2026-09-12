#!/usr/bin/env python3
"""Download real daily OHLCV bars into the CSV layout AxiomQuant expects.

Source: Stooq (https://stooq.com), which serves split-adjusted daily history as plain CSV with
no API key and no account. Only the Python standard library is used.

    python3 scripts/fetch_data.py --out real_data --tickers SPY AAPL MSFT GOOGL AMZN --start 2015-01-01
    ./build/bin/axiomquant --data real_data

Each file is written as <TICKER>.csv with the header

    Date,Open,High,Low,Close,Adj Close,Volume

Stooq's daily series is already adjusted for splits and dividends, so `Adj Close` repeats `Close`;
the loader accepts either column. Rows with non-positive or unparseable prices are dropped, and the
result is sorted by date. The script exits non-zero if any requested symbol yields no usable rows,
so a CI job fails loudly rather than silently producing an empty universe.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STOOQ_URL = "https://stooq.com/q/d/l/"
USER_AGENT = "AxiomQuant/1.0 (+https://github.com/Lukey-7/AxiomQuant)"
TIMEOUT_SECONDS = 60


def stooq_symbol(ticker: str) -> str:
    """Map a plain ticker to a Stooq symbol (US equities and ETFs carry a .us suffix)."""
    ticker = ticker.strip().lower()
    return ticker if "." in ticker else f"{ticker}.us"


def fetch_csv(ticker: str, start: str, end: str) -> str:
    query = urllib.parse.urlencode(
        {
            "s": stooq_symbol(ticker),
            "d1": start.replace("-", ""),
            "d2": end.replace("-", ""),
            "i": "d",
        }
    )
    request = urllib.request.Request(f"{STOOQ_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_rows(payload: str) -> list[list[str]]:
    """Validate and normalise Stooq's Date,Open,High,Low,Close,Volume rows."""
    reader = csv.DictReader(io.StringIO(payload))
    if not reader.fieldnames or "Date" not in reader.fieldnames or "Close" not in reader.fieldnames:
        return []

    rows: list[list[str]] = []
    for row in reader:
        try:
            date = row["Date"].strip()
            close = float(row["Close"])
            open_ = float(row.get("Open") or close)
            high = float(row.get("High") or max(open_, close))
            low = float(row.get("Low") or min(open_, close))
            volume = float(row.get("Volume") or 0.0)
        except (TypeError, ValueError):
            continue
        if len(date) < 10 or min(open_, high, low, close) <= 0.0 or high < low:
            continue
        rows.append([date, f"{open_:.6f}", f"{high:.6f}", f"{low:.6f}",
                     f"{close:.6f}", f"{close:.6f}", f"{volume:.0f}"])

    rows.sort(key=lambda r: r[0])
    return rows


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
                        help="symbols to download (default: SPY AAPL MSFT GOOGL AMZN)")
    parser.add_argument("--start", default="2015-01-01", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", default="2100-01-01", help="last date, YYYY-MM-DD")
    parser.add_argument("--out", default="real_data", help="output directory")
    parser.add_argument("--min-rows", type=int, default=250,
                        help="fail if a symbol returns fewer usable rows than this")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for ticker in args.tickers:
        try:
            payload = fetch_csv(ticker, args.start, args.end)
        except (urllib.error.URLError, TimeoutError) as exc:
            failures.append(f"{ticker}: download failed ({exc})")
            continue

        rows = parse_rows(payload)
        if len(rows) < args.min_rows:
            failures.append(f"{ticker}: only {len(rows)} usable rows (minimum {args.min_rows})")
            continue

        destination = out_dir / f"{ticker.upper()}.csv"
        write_csv(destination, rows)
        print(f"{ticker.upper():<6} {len(rows):>6} bars  {rows[0][0]} to {rows[-1][0]}  -> {destination}")

    if failures:
        print("\nFailed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\nWrote {len(args.tickers)} files to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
