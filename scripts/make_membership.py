#!/usr/bin/env python3
"""Turn a historical index-constituents file into the membership calendar AxiomQuant reads.

    python3 scripts/make_membership.py snapshots.csv --out members.csv
    ./build/bin/axiomquant --data real_data --members members.csv

Input (either shape, detected from the header; only the standard library is used):

  snapshots   date,tickers          one row per date, `tickers` a separated list of the members
                                    that day (the shape published as "S&P 500 historical components")
  changes     date,added,removed    one row per change, with the symbols added and removed that day

Output is one row per membership spell, which is what `MembershipCalendar` expects:

    ticker,start_date,end_date
    AAPL,1982-11-30,
    LEH,1994-09-08,2008-09-15

Both dates are inclusive; an empty `end_date` means the symbol was still a member in the last row of
the input. A symbol that leaves and rejoins gets one row per spell, so a backtest ranking the index
only ever sees the names that were actually in it on the day — the point of the exercise.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

DATE_COLUMNS = ["date", "day", "asof", "as_of", "effective_date"]
TICKERS_COLUMNS = ["tickers", "constituents", "members", "symbols", "components"]
ADDED_COLUMNS = ["added", "added_ticker", "add", "additions", "ticker_added"]
REMOVED_COLUMNS = ["removed", "removed_ticker", "remove", "removals", "ticker_removed"]
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y", "%d-%b-%Y", "%b %d, %Y", "%Y%m%d"]


class InputError(RuntimeError):
    """The file is not in a shape this script understands."""


def normalise(name: str | None) -> str:
    return (name or "").strip().lstrip("﻿").strip().lower()


def find_column(header: list[str], names: list[str]) -> str | None:
    lookup = {normalise(h): h for h in header}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def parse_date(text: str) -> str:
    text = (text or "").strip().strip('"')
    if not text:
        raise InputError("a row has an empty date")
    candidate = text[:10] if len(text) > 10 and text[4:5] in "-/" else text
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise InputError(f"unrecognised date: {text!r}")


def split_symbols(text: str) -> list[str]:
    """Symbols may be separated by commas, semicolons, spaces or pipes, and quoted as one field."""
    if not text:
        return []
    parts = re.split(r"[,;|\s]+", text.strip().strip('"').strip("'"))
    return [p.strip().upper() for p in parts if p.strip()]


def previous_day(date: str) -> str:
    return (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()


def spells_from_snapshots(rows: list[tuple[str, set[str]]]) -> list[tuple[str, str, str]]:
    """Membership spells from dated snapshots of the full member list."""
    spells: list[tuple[str, str, str]] = []
    open_since: dict[str, str] = {}
    previous: set[str] = set()
    for date, members in rows:
        for ticker in sorted(members - previous):
            open_since[ticker] = date
        for ticker in sorted(previous - members):
            # Present in the previous snapshot, gone in this one: the spell ended the day before.
            spells.append((ticker, open_since.pop(ticker, date), previous_day(date)))
        previous = members
    for ticker, start in sorted(open_since.items()):
        spells.append((ticker, start, ""))
    return spells


def spells_from_changes(rows: list[tuple[str, list[str], list[str]]]) -> list[tuple[str, str, str]]:
    """Membership spells from a log of additions and removals.

    Symbols removed without ever being added are treated as members from before the record begins,
    which is what an index-changes file implies: it only lists what changed.
    """
    spells: list[tuple[str, str, str]] = []
    open_since: dict[str, str] = {}
    for date, added, removed in rows:
        for ticker in removed:
            spells.append((ticker, open_since.pop(ticker, ""), previous_day(date)))
        for ticker in added:
            open_since.setdefault(ticker, date)
    for ticker, start in sorted(open_since.items()):
        spells.append((ticker, start, ""))
    return spells


def read_input(path: Path) -> list[tuple[str, str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        header = [h for h in (reader.fieldnames or []) if h is not None]
        date_col = find_column(header, DATE_COLUMNS)
        tickers_col = find_column(header, TICKERS_COLUMNS)
        added_col = find_column(header, ADDED_COLUMNS)
        removed_col = find_column(header, REMOVED_COLUMNS)
        if date_col is None:
            raise InputError(f"{path}: no date column (have: {', '.join(header)})")

        if tickers_col is not None:
            snapshots = [(parse_date(row[date_col]), set(split_symbols(row.get(tickers_col, "")))) for row in reader]
            snapshots.sort(key=lambda item: item[0])
            return spells_from_snapshots(snapshots)

        if added_col is None and removed_col is None:
            raise InputError(f"{path}: needs either a member list or added/removed columns "
                             f"(have: {', '.join(header)})")
        changes = [(parse_date(row[date_col]),
                    split_symbols(row.get(added_col, "") if added_col else ""),
                    split_symbols(row.get(removed_col, "") if removed_col else ""))
                   for row in reader]
        changes.sort(key=lambda item: item[0])
        return spells_from_changes(changes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="constituents file (snapshots or changes)")
    parser.add_argument("--out", default="members.csv", help="output path (default: members.csv)")
    parser.add_argument("--tickers", nargs="+", help="keep only these symbols")
    args = parser.parse_args()

    try:
        spells = read_input(Path(args.input).expanduser())
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        spells = [s for s in spells if s[0] in wanted]
    spells.sort(key=lambda s: (s[0], s[1]))
    if not spells:
        print("error: no membership spells found", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ticker", "start_date", "end_date"])
        writer.writerows(spells)

    symbols = {s[0] for s in spells}
    still_in = sum(1 for s in spells if not s[2])
    print(f"{len(spells)} spells for {len(symbols)} symbols ({still_in} open) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
