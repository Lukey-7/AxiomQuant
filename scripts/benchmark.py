#!/usr/bin/env python3
"""Benchmark the C++ engine against vectorised NumPy implementations of the same workloads.

    cmake --build build --target axiom_bench
    python3 scripts/benchmark.py --binary build/bin/axiom_bench --data sample_data --output bench-out

Workloads (identical inputs, identical statistics computed):

  mc_bootstrap            50,000 paths x 252 days, i.i.d. resampling of the first symbol's returns,
                          terminal wealth and max drawdown per path, then percentiles
  mc_gbm_portfolio        50,000 paths x 252 days, correlated GBM of an equal-weight buy-and-hold
                          portfolio of every symbol (Cholesky of the daily covariance)
  block_bootstrap_sharpe  10,000 stationary-bootstrap resamples of two return series, annualised
                          Sharpe of each, percentile intervals and p-values

The C++ side runs at each thread count; NumPy runs in one process, vectorised in chunks of paths so
memory stays bounded, which is how the same task would normally be written in NumPy. Each timing is
the fastest of --repeats runs. The two implementations use different random number generators, so
their summary statistics agree to Monte Carlo error rather than exactly; the report shows both.

Writes benchmark.csv, benchmark.md and benchmark.svg to --output.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_charts import BG, FG, GRID, MUTED, SERIES, text  # noqa: E402

PATHS, HORIZON, RESAMPLES, CHUNK = 50_000, 252, 10_000, 5_000
WEALTH = 100_000.0


def load_returns(data_dir: Path) -> np.ndarray:
    """Aligned simple returns, columns in alphabetical symbol order (as the C++ loader sorts them)."""
    series = {}
    for path in sorted(data_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            series[path.stem.upper()] = {row["Date"]: float(row["Close"]) for row in csv.DictReader(handle)}
    dates = sorted(set.intersection(*(set(s) for s in series.values())))
    prices = np.array([[series[name][d] for name in sorted(series)] for d in dates])
    return prices[1:] / prices[:-1] - 1.0


def summarise_paths(terminal: np.ndarray, drawdown: np.ndarray) -> float:
    np.percentile(terminal, [1, 5, 25, 50, 75, 95, 99])
    np.percentile(drawdown, [50, 95, 99])
    return float(terminal.mean())


def max_drawdown(wealth: np.ndarray) -> np.ndarray:
    start = np.full((wealth.shape[0], 1), WEALTH)
    path = np.hstack([start, wealth])
    peak = np.maximum.accumulate(path, axis=1)
    return ((peak - path) / peak).max(axis=1)


def numpy_mc_bootstrap(returns: np.ndarray, rng: np.random.Generator) -> float:
    hist = returns[:, 0]
    terminal, drawdown = np.empty(PATHS), np.empty(PATHS)
    for lo in range(0, PATHS, CHUNK):
        hi = min(PATHS, lo + CHUNK)
        wealth = WEALTH * np.cumprod(1.0 + hist[rng.integers(0, hist.size, (hi - lo, HORIZON))], axis=1)
        terminal[lo:hi], drawdown[lo:hi] = wealth[:, -1], max_drawdown(wealth)
    return summarise_paths(terminal, drawdown)


def numpy_mc_gbm(returns: np.ndarray, rng: np.random.Generator) -> float:
    n = returns.shape[1]
    cov = np.cov(returns, rowvar=False)
    chol = np.linalg.cholesky(cov)
    drift = returns.mean(axis=0) - 0.5 * np.diag(cov)
    weights = np.full(n, 1.0 / n)
    terminal, drawdown = np.empty(PATHS), np.empty(PATHS)
    for lo in range(0, PATHS, CHUNK):
        hi = min(PATHS, lo + CHUNK)
        shocks = rng.standard_normal((hi - lo, HORIZON, n)) @ chol.T
        relative = np.exp(np.cumsum(drift + shocks, axis=1))
        wealth = WEALTH * (relative @ weights)
        terminal[lo:hi], drawdown[lo:hi] = wealth[:, -1], max_drawdown(wealth)
    return summarise_paths(terminal, drawdown)


def annualised_sharpe(x: np.ndarray, rf: float = 0.02) -> np.ndarray:
    sd = x.std(axis=-1, ddof=1)
    return np.where(sd * np.sqrt(252.0) > 1e-9, (x.mean(axis=-1) - rf / 252.0) / sd * np.sqrt(252.0), 0.0)


def numpy_block_bootstrap(returns: np.ndarray, rng: np.random.Generator) -> float:
    a, b = returns[:, 0], returns[:, 1]
    n = a.size
    restart_p = 1.0 / np.cbrt(n)
    steps = np.arange(n)
    draws_a, draws_b = np.empty(RESAMPLES), np.empty(RESAMPLES)
    for lo in range(0, RESAMPLES, CHUNK // 5):
        hi = min(RESAMPLES, lo + CHUNK // 5)
        m = hi - lo
        restart = rng.random((m, n)) < restart_p
        restart[:, 0] = True
        starts = rng.integers(0, n, (m, n))
        block_begin = np.maximum.accumulate(np.where(restart, steps, 0), axis=1)
        idx = (np.take_along_axis(starts, block_begin, axis=1) + steps - block_begin) % n
        draws_a[lo:hi], draws_b[lo:hi] = annualised_sharpe(a[idx]), annualised_sharpe(b[idx])
    est_a, est_b = float(annualised_sharpe(a)), float(annualised_sharpe(b))
    diff = draws_a - draws_b
    for draws, est in ((draws_a, est_a), (draws_b, est_b), (diff, est_a - est_b)):
        np.percentile(draws, [2.5, 97.5])
        np.mean(np.abs(draws - est) >= abs(est))
    return float(np.percentile(draws_a, 2.5))


NUMPY_WORKLOADS = {
    "mc_bootstrap": numpy_mc_bootstrap,
    "mc_gbm_portfolio": numpy_mc_gbm,
    "block_bootstrap_sharpe": numpy_block_bootstrap,
}
LABELS = {
    "mc_bootstrap": ("Monte Carlo, bootstrap", "50,000 paths x 252 days", "paths/s", PATHS),
    "mc_gbm_portfolio": ("Monte Carlo, correlated GBM portfolio", "50,000 paths x 252 days", "paths/s", PATHS),
    "block_bootstrap_sharpe": ("Stationary block bootstrap of Sharpe", "10,000 resamples", "resamples/s", RESAMPLES),
}


def time_numpy(returns: np.ndarray, repeats: int) -> list[dict]:
    rows = []
    for name, work in NUMPY_WORKLOADS.items():
        best, check = float("inf"), 0.0
        for r in range(repeats):
            rng = np.random.default_rng(42 + r)
            start = time.perf_counter()
            check = work(returns, rng)
            best = min(best, time.perf_counter() - start)
        rows.append({"workload": name, "implementation": "numpy", "threads": 1, "seconds": best, "check": check})
    return rows


def time_cpp(binary: Path, data_dir: Path, threads: list[int], repeats: int) -> list[dict]:
    command = ([sys.executable, str(binary)] if binary.suffix == ".py" else [str(binary)]) + ["--data", str(data_dir), "--threads", ",".join(map(str, threads)), "--repeats", str(repeats)]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return [{"workload": r["workload"], "implementation": "cpp", "threads": int(r["threads"]),
             "seconds": float(r["seconds"]), "check": float(r["check"])} for r in csv.DictReader(io.StringIO(out))]


def write_markdown(rows: list[dict], threads: list[int], path: Path, environment: str) -> None:
    lines = [f"Environment: {environment}", "",
             "| Workload | Size | NumPy | " + " | ".join(f"C++ {t} thread{'s' if t > 1 else ''}" for t in threads) + " |",
             "|---|---|---|" + "---|" * len(threads)]
    checks = []
    for name, (label, size, unit, count) in LABELS.items():
        numpy_row = next(r for r in rows if r["workload"] == name and r["implementation"] == "numpy")
        cells = [f"{numpy_row['seconds'] * 1000:.0f} ms"]
        for t in threads:
            row = next(r for r in rows if r["workload"] == name and r["implementation"] == "cpp" and r["threads"] == t)
            cells.append(f"{row['seconds'] * 1000:.0f} ms ({numpy_row['seconds'] / row['seconds']:.1f}x)")
        lines.append(f"| {label} | {size} | " + " | ".join(cells) + " |")
        cpp_row = next(r for r in rows if r["workload"] == name and r["implementation"] == "cpp")
        checks.append(f"- {label}: C++ {cpp_row['check']:.4f}, NumPy {numpy_row['check']:.4f}")
    what = "mean terminal wealth for the Monte Carlo rows, lower 95% bound of the first series' Sharpe for the bootstrap"
    lines += ["", "Speed-up over NumPy in parentheses. Sanity check that both computed the same statistic "
              f"({what}; different random streams, so equal up to sampling error):", *checks, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg(rows: list[dict], threads: list[int], path: Path, environment: str) -> None:
    width, height = 900, 420
    left, right, top, bottom = 250, 40, 70, 60
    plot_w = width - left - right
    names = list(LABELS)
    series = [("NumPy (vectorised)", None)] + [(f"C++ {t} thread{'s' if t > 1 else ''}", t) for t in threads]
    speedups = {}
    for name in names:
        base = next(r for r in rows if r["workload"] == name and r["implementation"] == "numpy")["seconds"]
        speedups[name] = [1.0] + [base / next(r for r in rows if r["workload"] == name and r["implementation"] == "cpp"
                                              and r["threads"] == t)["seconds"] for t in threads]
    top_value = max(max(v) for v in speedups.values())
    scale_max = 10 ** np.ceil(np.log10(top_value))
    x_of = lambda v: left + plot_w * np.log10(max(v, 1e-3) / 0.1) / np.log10(scale_max / 0.1)  # noqa: E731

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f'<rect width="{width}" height="{height}" fill="{BG}"/>',
           text(24, 28, "Throughput relative to vectorised NumPy (log scale, higher is faster)", size=17, weight="600"),
           text(24, 48, environment, size=12, fill=MUTED)]
    tick = 0.1
    while tick <= scale_max * 1.0001:
        x = x_of(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="{GRID}"/>')
        svg.append(text(x, height - bottom + 18, f"{tick:g}x", size=11, fill=MUTED, anchor="middle"))
        tick *= 10
    group_h = (height - top - bottom) / len(names)
    bar_h = min(16.0, (group_h - 14) / len(series))
    for g, name in enumerate(names):
        y0 = top + g * group_h + (group_h - bar_h * len(series)) / 2
        svg.append(text(left - 12, y0 + bar_h * len(series) / 2 + 4, LABELS[name][0], size=12, anchor="end"))
        for s, value in enumerate(speedups[name]):
            y = y0 + s * bar_h
            x1 = x_of(value)
            color = MUTED if s == 0 else SERIES[(s - 1) % len(SERIES)]
            svg.append(f'<rect x="{left}" y="{y:.1f}" width="{max(1.0, x1 - left):.1f}" height="{bar_h - 3:.1f}" fill="{color}"/>')
            svg.append(text(x1 + 6, y + bar_h - 5, f"{value:.1f}x", size=10, fill=FG))
    lx = left
    for s, (label, _) in enumerate(series):
        color = MUTED if s == 0 else SERIES[(s - 1) % len(SERIES)]
        svg.append(f'<rect x="{lx}" y="{height - 26}" width="12" height="12" fill="{color}"/>')
        svg.append(text(lx + 18, height - 16, label, size=12))
        lx += 26 + 8 * len(label)
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--data", default="sample_data")
    parser.add_argument("--output", default="bench-out")
    parser.add_argument("--threads", help="comma-separated C++ thread counts (default: 1, 2, 4, ... up to the CPU count)")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cpus = os.cpu_count() or 1
    threads = [int(t) for t in args.threads.split(",")] if args.threads else \
        [t for t in (1, 2, 4, 8, 16, 32) if t <= cpus] or [1]
    data_dir, out_dir = Path(args.data), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    returns = load_returns(data_dir)
    environment = (f"{platform.system()} {platform.machine()}, {cpus} logical CPUs, Python {platform.python_version()}, "
                   f"NumPy {np.__version__}; {returns.shape[0]} days x {returns.shape[1]} symbols; best of {args.repeats}")
    rows = time_cpp(Path(args.binary), data_dir, threads, args.repeats) + time_numpy(returns, args.repeats)

    with (out_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["workload", "implementation", "threads", "seconds", "check"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, threads, out_dir / "benchmark.md", environment)
    write_svg(rows, threads, out_dir / "benchmark.svg", environment)
    print((out_dir / "benchmark.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
