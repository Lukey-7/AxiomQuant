#!/usr/bin/env python3
"""Render the CLI's CSV exports as SVG charts for the README.

    ./build/bin/axiomquant --data sample_data --no-db --export-dir out
    python3 scripts/make_charts.py --input out --output docs/images

Standard library only: the charts are written as hand-rolled SVG, so there is nothing to install
and the output is deterministic (byte-identical for identical inputs). Each chart paints its own
light background so it stays readable in GitHub's dark theme.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

W, H = 900, 460
PAD_L, PAD_R, PAD_T, PAD_B = 78, 26, 52, 62
PLOT_W, PLOT_H = W - PAD_L - PAD_R, H - PAD_T - PAD_B

BG, FG, MUTED, GRID = "#ffffff", "#1f2328", "#656d76", "#e4e8ec"
SERIES = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
POS, NEG = "#16a34a", "#dc2626"


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=13, fill=FG, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def header(title, subtitle):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" fill="{BG}"/>',
             text(PAD_L - 48, 26, title, size=17, weight="600")]
    if subtitle:
        parts.append(text(PAD_L - 48, 44, subtitle, size=12, fill=MUTED))
    return parts


def y_axis(lo, hi, fmt, ticks=6):
    """Horizontal gridlines plus left-hand labels; returns (svg_parts, scale_fn)."""
    span = (hi - lo) or 1.0
    parts = []
    for i in range(ticks + 1):
        value = lo + span * i / ticks
        y = PAD_T + PLOT_H - PLOT_H * (value - lo) / span
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(text(PAD_L - 10, y + 4, fmt(value), size=12, fill=MUTED, anchor="end"))
    return parts, lambda v: PAD_T + PLOT_H - PLOT_H * (v - lo) / span


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def legend(entries, y=H - 18):
    parts, x = [], PAD_L
    for label, colour in entries:
        parts.append(f'<rect x="{x}" y="{y - 9}" width="11" height="11" rx="2" fill="{colour}"/>')
        parts.append(text(x + 17, y, label, size=12))
        x += 20 + 7.0 * len(label)
    return parts


def chart_equity(rows: list[dict], out: Path) -> None:
    names = [c for c in rows[0].keys() if c != "date"]
    series = {n: [float(r[n]) for r in rows] for n in names}
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad

    svg = header("Equity curves", f"{len(rows)} bars, identical costs and fill rules, $100,000 start")
    axis, ys = y_axis(lo, hi, lambda v: f"${v/1000:,.0f}k")
    svg += axis

    for i, name in enumerate(names):
        values = series[name]
        step = PLOT_W / max(1, len(values) - 1)
        points = " ".join(f"{PAD_L + j * step:.1f},{ys(v):.1f}" for j, v in enumerate(values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{SERIES[i % len(SERIES)]}" '
                   f'stroke-width="1.8" stroke-linejoin="round"/>')

    svg.append(text(PAD_L, H - 38, rows[0]["date"], size=12, fill=MUTED))
    svg.append(text(PAD_L + PLOT_W, H - 38, rows[-1]["date"], size=12, fill=MUTED, anchor="end"))
    svg += legend([(n.split("(")[0], SERIES[i % len(SERIES)]) for i, n in enumerate(names)])
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def chart_walk_forward(rows: list[dict], out: Path) -> None:
    is_sharpe = [float(r["is_sharpe"]) for r in rows]
    oos_sharpe = [float(r["oos_sharpe"]) for r in rows]
    lo = min(0.0, min(oos_sharpe), min(is_sharpe)) - 0.4
    hi = max(0.0, max(oos_sharpe), max(is_sharpe)) + 0.4

    mean_is = sum(is_sharpe) / len(is_sharpe)
    mean_oos = sum(oos_sharpe) / len(oos_sharpe)
    svg = header("Walk-forward: in-sample vs out-of-sample Sharpe",
                 f"mean in-sample {mean_is:+.2f} vs out-of-sample {mean_oos:+.2f} "
                 f"(decay {mean_is - mean_oos:.2f}) - the cost of fitting parameters")
    axis, ys = y_axis(lo, hi, lambda v: f"{v:+.1f}")
    svg += axis
    svg.append(f'<line x1="{PAD_L}" y1="{ys(0):.1f}" x2="{PAD_L + PLOT_W}" y2="{ys(0):.1f}" stroke="{MUTED}"/>')

    group = PLOT_W / len(rows)
    bar = group * 0.30
    for i, row in enumerate(rows):
        cx = PAD_L + group * (i + 0.5)
        for offset, value, colour in ((-bar * 1.05, is_sharpe[i], "#93c5fd"), (bar * 0.05, oos_sharpe[i], SERIES[0])):
            top, bottom = min(ys(value), ys(0)), max(ys(value), ys(0))
            svg.append(f'<rect x="{cx + offset:.1f}" y="{top:.1f}" width="{bar:.1f}" '
                       f'height="{max(1.0, bottom - top):.1f}" fill="{colour}" rx="2"/>')
        svg.append(text(cx, H - 38, f"fold {i + 1}", size=12, fill=MUTED, anchor="middle"))
        svg.append(text(cx, H - 24, f'{row["fast"]}/{row["slow"]}', size=11, fill=MUTED, anchor="middle"))

    svg += legend([("in-sample (fitted)", "#93c5fd"), ("out-of-sample (unseen)", SERIES[0])])
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def chart_sweep(rows: list[dict], out: Path) -> None:
    fasts = sorted({int(r["fast"]) for r in rows})
    slows = sorted({int(r["slow"]) for r in rows})
    table = {(int(r["fast"]), int(r["slow"])): float(r["sharpe"]) for r in rows}
    values = list(table.values())
    best, worst = max(values), min(values)
    limit = max(abs(best), abs(worst)) or 1.0

    def colour(v: float) -> str:
        # diverging scale: red for negative Sharpe, green for positive, white at zero
        t = max(-1.0, min(1.0, v / limit))
        if t >= 0:
            r, g, b = int(255 - 235 * t), int(255 - 93 * t), int(255 - 217 * t)
        else:
            r, g, b = int(255 - 35 * -t), int(255 - 217 * -t), int(255 - 217 * -t)
        return f"rgb({r},{g},{b})"

    median = sorted(values)[len(values) // 2]
    svg = header("In-sample SMA parameter sweep",
                 f"{len(values)} pairs: best {best:+.2f}, median {median:+.2f}, worst {worst:+.2f} "
                 f"- the winner is the luckiest pair, not a forecast")

    cw, ch = PLOT_W / len(slows), PLOT_H / len(fasts)
    for i, fast in enumerate(fasts):
        for j, slow in enumerate(slows):
            if (fast, slow) not in table:
                continue
            v = table[(fast, slow)]
            x, y = PAD_L + j * cw, PAD_T + i * ch
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" '
                       f'fill="{colour(v)}" stroke="{GRID}" rx="3"/>')
            svg.append(text(x + cw / 2 - 1, y + ch / 2 + 4, f"{v:+.2f}", size=12, anchor="middle"))
        svg.append(text(PAD_L - 10, PAD_T + i * ch + ch / 2 + 4, f"fast {fast}", size=12, fill=MUTED, anchor="end"))
    for j, slow in enumerate(slows):
        svg.append(text(PAD_L + j * cw + cw / 2, H - 38, f"slow {slow}", size=12, fill=MUTED, anchor="middle"))

    svg.append(text(PAD_L, H - 16, "green = positive Sharpe in sample, red = negative", size=12, fill=MUTED))
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def chart_frontier(rows: list[dict], weights_rows: list[dict], out: Path) -> None:
    vols = [float(r["volatility"]) for r in rows]
    rets = [float(r["expected_return"]) for r in rows]
    sharpes = [float(r["sharpe"]) for r in rows]
    vpad, rpad = (max(vols) - min(vols)) * 0.12 or 0.01, (max(rets) - min(rets)) * 0.12 or 0.01

    svg = header("Constrained efficient frontier",
                 "long-only, 40% per-asset cap, Ledoit-Wolf covariance, FISTA with exact simplex projection")
    axis, ys = y_axis(min(rets) - rpad, max(rets) + rpad, lambda v: f"{v*100:.1f}%")
    svg += axis

    lo_v, hi_v = min(vols) - vpad, max(vols) + vpad
    xs = lambda v: PAD_L + PLOT_W * (v - lo_v) / ((hi_v - lo_v) or 1.0)

    path = " ".join(f"{'M' if i == 0 else 'L'}{xs(v):.1f},{ys(r):.1f}" for i, (v, r) in enumerate(zip(vols, rets)))
    svg.append(f'<path d="{path}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>')
    for v, r in zip(vols, rets):
        svg.append(f'<circle cx="{xs(v):.1f}" cy="{ys(r):.1f}" r="3" fill="{SERIES[0]}"/>')

    best = max(range(len(sharpes)), key=lambda i: sharpes[i])
    svg.append(f'<circle cx="{xs(vols[best]):.1f}" cy="{ys(rets[best]):.1f}" r="7" fill="none" '
               f'stroke="{POS}" stroke-width="2.5"/>')
    svg.append(text(xs(vols[best]) + 12, ys(rets[best]) + 4, f"max Sharpe {sharpes[best]:.2f}", size=12, fill=POS))
    svg.append(f'<circle cx="{xs(vols[0]):.1f}" cy="{ys(rets[0]):.1f}" r="7" fill="none" '
               f'stroke="{MUTED}" stroke-width="2.5"/>')
    svg.append(text(xs(vols[0]) + 12, ys(rets[0]) + 4, "min variance", size=12, fill=MUTED))

    holdings = ", ".join(f'{r["ticker"]} {float(r["max_sharpe_constrained"])*100:.0f}%'
                         for r in weights_rows if float(r["max_sharpe_constrained"]) > 0.005)
    svg.append(text(PAD_L, H - 38, f"annualized volatility {min(vols)*100:.1f}% to {max(vols)*100:.1f}%",
                    size=12, fill=MUTED))
    svg.append(text(PAD_L, H - 16, f"max-Sharpe holdings: {holdings}", size=12, fill=MUTED))
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="out", help="directory written by --export-dir")
    parser.add_argument("--output", default="docs/images", help="directory for the SVG files")
    args = parser.parse_args()

    src, dst = Path(args.input), Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    charts = {
        "equity_curves.svg": lambda: chart_equity(read_csv(src / "equity_curves.csv"), dst / "equity_curves.svg"),
        "walk_forward.svg": lambda: chart_walk_forward(read_csv(src / "walk_forward.csv"), dst / "walk_forward.svg"),
        "parameter_sweep.svg": lambda: chart_sweep(read_csv(src / "parameter_sweep.csv"), dst / "parameter_sweep.svg"),
        "efficient_frontier.svg": lambda: chart_frontier(read_csv(src / "efficient_frontier.csv"),
                                                         read_csv(src / "portfolio_weights.csv"),
                                                         dst / "efficient_frontier.svg"),
    }

    written = 0
    for name, build in charts.items():
        try:
            build()
        except FileNotFoundError as exc:
            print(f"skipped {name}: {exc.filename} not found")
            continue
        print(f"wrote {dst / name}")
        written += 1

    if written == 0:
        print("no charts written - is --input pointing at an --export-dir?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
