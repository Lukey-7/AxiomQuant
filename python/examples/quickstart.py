"""Load the sample data, backtest an SMA crossover against buy-and-hold, and plot both equity curves.

    PYTHONPATH=build/python python python/examples/quickstart.py [path/to/csv_dir]

matplotlib is optional: without it the script prints the summary and skips the chart.
"""

import pathlib
import sys

import axiomquant as aq

data_dir = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "sample_data"

universe = aq.load_universe(data_dir)
print(universe)

runs = [
    aq.backtest(universe, aq.sma_crossover("SPY", fast=20, slow=50), costs=aq.CostModel()),
    aq.backtest(universe, aq.buy_and_hold("SPY")),
]
for run in runs:
    m = run.metrics
    print(f"{run.strategy:28s} return {m.total_return:7.1%}  Sharpe {m.sharpe_ratio:5.2f}  "
          f"max drawdown {m.max_drawdown:6.1%}  fills {run.num_fills}")

# Is the crossover's Sharpe distinguishable from buy-and-hold's? Paired block bootstrap.
boot = aq.bootstrap_sharpe(runs[0].returns, runs[1].returns, resamples=2000)
print(f"Sharpe difference {boot.difference.estimate:+.2f}, p = {boot.difference.p_value:.2f}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; skipping the chart")
    sys.exit(0)

import numpy as np  # noqa: E402

dates = np.array(runs[0].dates, dtype="datetime64[D]")
fig, ax = plt.subplots(figsize=(10, 4))
for run in runs:
    ax.plot(dates, run.equity, label=run.strategy)
ax.set_ylabel("Equity ($)")
ax.set_title(f"{data_dir}: {universe.dates[0]} to {universe.dates[-1]}")
ax.legend()
fig.tight_layout()
plt.show()
