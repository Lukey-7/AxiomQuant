#!/usr/bin/env python3
"""Build a directory of <TICKER>.csv files in the layout AxiomQuant expects, from any of three kinds
of source. Only the Python standard library is used.

    Date,Open,High,Low,Close,Adj Close,Volume

Sources (subcommands):

  live       Download daily bars from a live provider.
               yahoo   Yahoo Finance chart API. No key. Default.
               tiingo  Tiingo end-of-day API. Needs TIINGO_API_KEY (free account).
               auto    yahoo, then tiingo if TIINGO_API_KEY is set.
  import     Normalise CSVs you already have: Kaggle downloads, broker exports, spreadsheets.
             Accepts one file per symbol or a single "long" file with a symbol column.
  kaggle     Download a Kaggle dataset with the official `kaggle` CLI, then run `import` on it.
             Needs `pip install kaggle` and KAGGLE_API_TOKEN (or KAGGLE_USERNAME / KAGGLE_KEY).
  synthetic  Generate a correlated geometric-Brownian-motion universe from a seed. Deterministic.

Examples:

    python3 scripts/fetch_data.py live --out real_data --tickers SPY AAPL MSFT --start 2015-01-01
    python3 scripts/fetch_data.py import ~/Downloads/all_stocks_5yr.csv --out kaggle_data --tickers AAPL MSFT
    python3 scripts/fetch_data.py kaggle camnugent/sandp500 --out kaggle_data --tickers AAPL MSFT AMZN
    python3 scripts/fetch_data.py synthetic --out synth_data --tickers AAA BBB CCC --seed 7
    ./build/bin/axiomquant --data real_data

Calling the script without a subcommand behaves like `live`, so older invocations keep working.

Every source goes through the same validation: unparseable or non-positive prices and inverted
high/low ranges are dropped, duplicate dates keep the last row, and rows are sorted by date. When a
source provides an adjusted close that differs from the raw close, Open/High/Low/Close are all scaled
by the same adjustment factor so that intraday and close-to-close returns stay consistent. The script
exits non-zero if any requested symbol yields fewer than --min-rows rows, so a CI job fails loudly
instead of silently producing an empty universe.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "AxiomQuant/1.0 (+https://github.com/Lukey-7/AxiomQuant)"
TIMEOUT_SECONDS = 60
HEADER = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]

# A bar is (date "YYYY-MM-DD", open, high, low, close, volume); close is already adjusted.
Bar = tuple[str, float, float, float, float, float]


class SourceError(RuntimeError):
    """A source answered, but not with usable data."""


# --------------------------------------------------------------------------------------------------
# Shared validation and output
# --------------------------------------------------------------------------------------------------

def make_bar(date: str, open_: float | None, high: float | None, low: float | None,
             close: float | None, volume: float | None, adj_close: float | None = None) -> Bar | None:
    """Validate one row and apply the adjustment factor. Returns None for unusable rows."""
    if close is None or not math.isfinite(close) or close <= 0.0:
        return None
    factor = 1.0
    if adj_close is not None and math.isfinite(adj_close) and adj_close > 0.0:
        factor = adj_close / close
    open_ = close if open_ is None or not math.isfinite(open_) else open_
    high = max(open_, close) if high is None or not math.isfinite(high) else high
    low = min(open_, close) if low is None or not math.isfinite(low) else low
    volume = 0.0 if volume is None or not math.isfinite(volume) or volume < 0.0 else volume
    if min(open_, high, low) <= 0.0 or high < low:
        return None
    return (date, open_ * factor, high * factor, low * factor, close * factor, volume)


def finalise(bars: list[Bar]) -> list[Bar]:
    """Sort by date and keep the last row for each duplicated date."""
    by_date = {bar[0]: bar for bar in bars}
    return [by_date[d] for d in sorted(by_date)]


def write_csv(path: Path, bars: list[Bar]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for date, open_, high, low, close, volume in bars:
            writer.writerow([date, f"{open_:.6f}", f"{high:.6f}", f"{low:.6f}",
                             f"{close:.6f}", f"{close:.6f}", f"{volume:.0f}"])


def emit(out_dir: Path, universe: dict[str, list[Bar]], requested: list[str] | None,
         min_rows: int, allow_missing: bool = False) -> int:
    """Write every symbol with enough rows; report and fail on the rest.

    With `allow_missing` the run succeeds as long as at least one symbol worked, which is what a
    point-in-time universe needs: symbols delisted years ago have no downloadable history left.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    names = requested if requested else sorted(universe)
    if not names:
        failures.append("no symbols found in the input")
    written = 0
    for name in names:
        bars = finalise(universe.get(name.upper(), []))
        if len(bars) < min_rows:
            failures.append(f"{name.upper()}: only {len(bars)} usable rows (minimum {min_rows})")
            continue
        destination = out_dir / f"{safe_filename(name)}.csv"
        write_csv(destination, bars)
        written += 1
        print(f"{name.upper():<8} {len(bars):>6} bars  {bars[0][0]} to {bars[-1][0]}  -> {destination}")
    return report(failures, written, out_dir, allow_missing)


def report(failures: list[str], written: int, out_dir: Path, allow_missing: bool = False) -> int:
    tolerated = allow_missing and written > 0
    if failures:
        print("\nSkipped:" if tolerated else "\nFailed:", file=sys.stderr)
        for failure in failures[:40]:
            print(f"  {failure}", file=sys.stderr)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more", file=sys.stderr)
        if not tolerated:
            return 1
    suffix = f" ({len(failures)} symbols had no usable history)" if failures else ""
    print(f"\nWrote {written} files to {out_dir}/{suffix}")
    return 0


def safe_filename(ticker: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in ticker.strip().upper())
    return cleaned or "UNKNOWN"


def parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"null", "nan", "na", "n/a", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------------
# Live providers
# --------------------------------------------------------------------------------------------------

def http_get(url: str, headers: dict[str, str] | None = None, attempts: int = 3) -> str:
    """GET with retries on rate limiting and transient server errors."""
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=merged)
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise AssertionError("unreachable")


def looks_like_html(payload: str) -> bool:
    head = payload.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def to_epoch(date: str) -> int:
    return int(dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_yahoo(ticker: str, start: str, end: str) -> list[Bar]:
    query = urllib.parse.urlencode({
        "period1": to_epoch(start),
        "period2": to_epoch(end) + 86400,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker.upper())}?{query}"
    return parse_yahoo(http_get(url))


def parse_yahoo(payload: str) -> list[Bar]:
    if looks_like_html(payload):
        raise SourceError("Yahoo returned an HTML page instead of JSON")
    try:
        chart = json.loads(payload)["chart"]
    except (ValueError, KeyError) as exc:
        raise SourceError(f"unexpected Yahoo response ({exc})") from exc
    if chart.get("error"):
        raise SourceError(f"Yahoo error: {chart['error'].get('description') or chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise SourceError("Yahoo returned no result")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    offset = int(result.get("meta", {}).get("gmtoffset") or 0)

    def column(name: str) -> list:
        values = quote.get(name) or []
        return values if len(values) == len(timestamps) else [None] * len(timestamps)

    opens, highs, lows, closes, volumes = (column(n) for n in ("open", "high", "low", "close", "volume"))
    if adjusted is None or len(adjusted) != len(timestamps):
        adjusted = [None] * len(timestamps)

    bars: list[Bar] = []
    for i, stamp in enumerate(timestamps):
        date = dt.datetime.fromtimestamp(stamp + offset, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        bar = make_bar(date, opens[i], highs[i], lows[i], closes[i], volumes[i], adjusted[i])
        if bar:
            bars.append(bar)
    return bars


def fetch_tiingo(ticker: str, start: str, end: str) -> list[Bar]:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key:
        raise SourceError("TIINGO_API_KEY is not set")
    query = urllib.parse.urlencode({"startDate": start, "endDate": end, "format": "json"})
    url = f"https://api.tiingo.com/tiingo/daily/{urllib.parse.quote(ticker.lower())}/prices?{query}"
    return parse_tiingo(http_get(url, headers={"Authorization": f"Token {key}"}))


def parse_tiingo(payload: str) -> list[Bar]:
    try:
        rows = json.loads(payload)
    except ValueError as exc:
        raise SourceError(f"unexpected Tiingo response ({exc})") from exc
    if isinstance(rows, dict):
        raise SourceError(f"Tiingo error: {rows.get('detail') or rows}")
    bars: list[Bar] = []
    for row in rows:
        # Tiingo's adjOpen/adjHigh/adjLow/adjClose are already mutually consistent.
        bar = make_bar(str(row.get("date", ""))[:10], row.get("adjOpen"), row.get("adjHigh"),
                       row.get("adjLow"), row.get("adjClose"), row.get("adjVolume"))
        if bar:
            bars.append(bar)
    return bars


PROVIDERS = {"yahoo": fetch_yahoo, "tiingo": fetch_tiingo}


def command_live(args: argparse.Namespace) -> int:
    if args.provider == "auto":
        order = ["yahoo"] + (["tiingo"] if os.environ.get("TIINGO_API_KEY") else [])
    else:
        order = [args.provider]
    end = args.end or dt.date.today().isoformat()

    universe: dict[str, list[Bar]] = {}
    errors: list[str] = []
    for ticker in args.tickers:
        for provider in order:
            try:
                bars = PROVIDERS[provider](ticker, args.start, end)
            except (urllib.error.URLError, TimeoutError, SourceError) as exc:
                errors.append(f"{ticker.upper()} via {provider}: {exc}")
                continue
            if len(bars) >= args.min_rows:
                universe[ticker.upper()] = bars
                break
            errors.append(f"{ticker.upper()} via {provider}: only {len(bars)} usable rows")
    if not args.allow_missing:
        for error in errors:
            print(f"note: {error}", file=sys.stderr)
    return emit(Path(args.out), universe, args.tickers, args.min_rows, args.allow_missing)


# --------------------------------------------------------------------------------------------------
# Import (local CSVs, Kaggle datasets)
# --------------------------------------------------------------------------------------------------

DATE_ALIASES = ["date", "timestamp", "datetime", "time", "day", "trade_date", "tradedate"]
OPEN_ALIASES = ["open", "open_price", "adj_open", "adjopen"]
HIGH_ALIASES = ["high", "high_price", "adj_high", "adjhigh"]
LOW_ALIASES = ["low", "low_price", "adj_low", "adjlow"]
CLOSE_ALIASES = ["close", "close_price", "last", "price", "close/last"]
ADJ_ALIASES = ["adj close", "adj_close", "adjclose", "adjusted_close", "adjusted close", "adj. close"]
VOLUME_ALIASES = ["volume", "vol", "adj_volume", "adjvolume"]
SYMBOL_ALIASES = ["symbol", "ticker", "name", "stock", "code", "sym"]

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%m/%d/%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y"]


def normalise_header(value: str | None) -> str:
    return (value or "").strip().lstrip("﻿").strip().lower()


def find_column(headers: list[str], aliases: list[str], override: str | None = None) -> str | None:
    lookup = {normalise_header(h): h for h in headers}
    if override:
        key = normalise_header(override)
        if key not in lookup:
            raise SourceError(f"column '{override}' not found (have: {', '.join(headers)})")
        return lookup[key]
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def parse_date(text: str, date_format: str | None) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if date_format:
        try:
            return dt.datetime.strptime(text, date_format).strftime("%Y-%m-%d")
        except ValueError:
            return None
    if text.isdigit() and len(text) >= 9:  # Unix seconds or milliseconds
        seconds = int(text) / (1000 if len(text) >= 12 else 1)
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    candidate = text[:10] if len(text) > 10 and text[4:5] in "-/" else text
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def import_csv(path: Path, args: argparse.Namespace) -> dict[str, list[Bar]]:
    """Read one CSV. Returns bars keyed by upper-case symbol."""
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        headers = [h for h in (reader.fieldnames or []) if h is not None]
        date_col = find_column(headers, DATE_ALIASES, args.date_column)
        close_col = find_column(headers, CLOSE_ALIASES, args.close_column)
        adj_col = find_column(headers, ADJ_ALIASES)
        if close_col is None and adj_col is not None:
            close_col, adj_col = adj_col, None
        if date_col is None or close_col is None:
            raise SourceError(f"{path}: needs a date and a close column (have: {', '.join(headers)})")
        if args.no_adjust:
            adj_col = None
        open_col = find_column(headers, OPEN_ALIASES)
        high_col = find_column(headers, HIGH_ALIASES)
        low_col = find_column(headers, LOW_ALIASES)
        volume_col = find_column(headers, VOLUME_ALIASES)
        symbol_col = find_column(headers, SYMBOL_ALIASES, args.symbol_column)
        file_symbol = path.stem.upper()

        universe: dict[str, list[Bar]] = {}
        for row in reader:
            symbol = (row.get(symbol_col) or "").strip().upper() if symbol_col else file_symbol
            if not symbol:
                continue
            date = parse_date(row.get(date_col) or "", args.date_format)
            if date is None or (args.start and date < args.start) or (args.end and date > args.end):
                continue
            bar = make_bar(
                date,
                parse_float(row.get(open_col)) if open_col else None,
                parse_float(row.get(high_col)) if high_col else None,
                parse_float(row.get(low_col)) if low_col else None,
                parse_float(row.get(close_col)),
                parse_float(row.get(volume_col)) if volume_col else None,
                parse_float(row.get(adj_col)) if adj_col else None,
            )
            if bar:
                universe.setdefault(symbol, []).append(bar)
    return universe


def collect_csv_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in (".csv", ".txt")))
        elif path.is_file():
            files.append(path)
        else:
            raise SourceError(f"{item}: no such file or directory")
    return files


def import_paths(inputs: list[str], args: argparse.Namespace) -> int:
    universe: dict[str, list[Bar]] = {}
    files = collect_csv_files(inputs)
    wanted = {t.upper() for t in args.tickers} if args.tickers else None
    for path in files:
        try:
            found = import_csv(path, args)
        except SourceError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            continue
        for symbol, bars in found.items():
            if wanted is None or symbol in wanted:
                universe.setdefault(symbol, []).extend(bars)
    return emit(Path(args.out), universe, args.tickers, args.min_rows, args.allow_missing)


def command_import(args: argparse.Namespace) -> int:
    try:
        return import_paths(args.inputs, args)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def command_kaggle(args: argparse.Namespace) -> int:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("error: the Kaggle CLI is not installed (pip install kaggle)", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="axiom-kaggle-") as tmp:
        command = [kaggle, "datasets", "download", "-d", args.dataset, "-p", tmp, "--unzip"]
        if args.file:
            command += ["-f", args.file]
        print("$ " + " ".join(command[1:]), file=sys.stderr)
        if subprocess.run(command).returncode != 0:
            print("error: Kaggle download failed (check KAGGLE_API_TOKEN, or KAGGLE_USERNAME / KAGGLE_KEY)", file=sys.stderr)
            return 1
        # `-f` downloads a single file, sometimes zipped even with --unzip.
        for archive in Path(tmp).glob("*.zip"):
            shutil.unpack_archive(str(archive), tmp)
            archive.unlink()
        try:
            return import_paths([tmp], args)
        except SourceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1


# --------------------------------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------------------------------

def business_days(start: str, end: str) -> list[str]:
    day = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    days: list[str] = []
    while day <= last:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return days


def generate_synthetic(tickers: list[str], start: str, end: str, seed: int, drift: float,
                       vol: float, correlation: float, start_price: float) -> dict[str, list[Bar]]:
    """One-factor correlated GBM. Python's Mersenne Twister is specified, so a seed reproduces exactly."""
    rng = random.Random(seed)
    days = business_days(start, end)
    dt_year = 1.0 / 252.0
    loading = math.sqrt(max(0.0, min(1.0, correlation)))
    idio = math.sqrt(1.0 - loading * loading)

    params = []
    for _ in tickers:
        asset_vol = vol * rng.uniform(0.7, 1.5)
        asset_drift = drift + rng.uniform(-0.04, 0.06)
        params.append((asset_drift, asset_vol, start_price * rng.uniform(0.3, 3.0)))

    universe: dict[str, list[Bar]] = {t.upper(): [] for t in tickers}
    prices = [p[2] for p in params]
    for date in days:
        market = rng.gauss(0.0, 1.0)
        for i, ticker in enumerate(tickers):
            mu, sigma, _ = params[i]
            shock = loading * market + idio * rng.gauss(0.0, 1.0)
            prev = prices[i]
            close = prev * math.exp((mu - 0.5 * sigma * sigma) * dt_year + sigma * math.sqrt(dt_year) * shock)
            open_ = prev * math.exp(sigma * math.sqrt(dt_year) * 0.25 * rng.gauss(0.0, 1.0))
            span = sigma * math.sqrt(dt_year) * abs(rng.gauss(0.0, 0.6))
            high = max(open_, close) * math.exp(span)
            low = min(open_, close) * math.exp(-span)
            volume = 1e6 * math.exp(rng.gauss(0.0, 0.35))
            bar = make_bar(date, open_, high, low, close, volume)
            if bar:
                universe[ticker.upper()].append(bar)
            prices[i] = close
    return universe


def command_synthetic(args: argparse.Namespace) -> int:
    end = args.end or dt.date.today().isoformat()
    universe = generate_synthetic(args.tickers, args.start, end, args.seed, args.drift, args.vol,
                                  args.correlation, args.start_price)
    return emit(Path(args.out), universe, args.tickers, args.min_rows, args.allow_missing)


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, tickers_required: bool) -> None:
        p.add_argument("--out", default="real_data", help="output directory (default: real_data)")
        p.add_argument("--min-rows", type=int, default=250,
                       help="fail if a symbol has fewer usable rows than this (default: 250)")
        p.add_argument("--allow-missing", action="store_true",
                       help="skip symbols with no usable history instead of failing the run")
        if tickers_required:
            p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                           help="symbols (default: SPY AAPL MSFT GOOGL AMZN)")
        else:
            p.add_argument("--tickers", nargs="+", help="keep only these symbols (default: all found)")

    live = sub.add_parser("live", help="download from a live provider")
    common(live, tickers_required=True)
    live.add_argument("--provider", choices=["auto", "yahoo", "tiingo"], default="auto")
    live.add_argument("--start", default="2015-01-01", help="first date, YYYY-MM-DD")
    live.add_argument("--end", help="last date, YYYY-MM-DD (default: today)")
    live.set_defaults(func=command_live)

    def import_options(p: argparse.ArgumentParser) -> None:
        common(p, tickers_required=False)
        p.add_argument("--start", help="drop rows before this date, YYYY-MM-DD")
        p.add_argument("--end", help="drop rows after this date, YYYY-MM-DD")
        p.add_argument("--date-column", help="name of the date column (default: detected)")
        p.add_argument("--close-column", help="name of the close column (default: detected)")
        p.add_argument("--symbol-column", help="name of the symbol column in a long-format file")
        p.add_argument("--date-format", help="strptime format, e.g. %%d/%%m/%%Y (default: detected)")
        p.add_argument("--no-adjust", action="store_true", help="ignore an Adj Close column")

    imp = sub.add_parser("import", help="normalise local CSV files or directories (e.g. a Kaggle download)")
    imp.add_argument("inputs", nargs="+", help="CSV files and/or directories")
    import_options(imp)
    imp.set_defaults(func=command_import)

    kag = sub.add_parser("kaggle", help="download a Kaggle dataset and import it")
    kag.add_argument("dataset", help="owner/dataset-slug, e.g. camnugent/sandp500")
    kag.add_argument("--file", help="download only this file from the dataset")
    import_options(kag)
    kag.set_defaults(func=command_kaggle)

    syn = sub.add_parser("synthetic", help="generate a seeded synthetic universe")
    common(syn, tickers_required=True)
    syn.add_argument("--start", default="2020-01-01", help="first date, YYYY-MM-DD")
    syn.add_argument("--end", default="2024-12-31", help="last date, YYYY-MM-DD")
    syn.add_argument("--seed", type=int, default=42)
    syn.add_argument("--drift", type=float, default=0.07, help="mean annual drift (default: 0.07)")
    syn.add_argument("--vol", type=float, default=0.22, help="mean annual volatility (default: 0.22)")
    syn.add_argument("--correlation", type=float, default=0.5, help="pairwise correlation in [0, 1]")
    syn.add_argument("--start-price", type=float, default=100.0)
    syn.set_defaults(func=command_synthetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"live", "import", "kaggle", "synthetic", "-h", "--help"}
    if not argv or argv[0] not in commands:
        argv.insert(0, "live")  # backwards compatible with the original flag-only interface
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
