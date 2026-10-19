# AxiomQuant for Python

A [pybind11](https://github.com/pybind/pybind11) module, `axiomquant`, that exposes the engine to
Python: load a universe from a directory of CSVs, run the built-in strategies with the same cost model
and fill rules as the CLI, and get equity curves back as NumPy arrays together with the risk metrics,
the walk-forward study and the significance tests. Every number is computed by the same C++ code the
CLI runs, and the test suite checks that the two agree.

## Ten lines

```python
import axiomquant as aq

universe = aq.load_universe("sample_data")                      # every <TICKER>.csv, aligned
strategy = aq.sma_crossover("SPY", fast=20, slow=50)
result = aq.backtest(universe, strategy, capital=100_000, costs=aq.CostModel(), fill="open")

result.equity                    # numpy.ndarray, one value per bar; result.dates has the dates
result.metrics.sharpe_ratio      # also cagr, sortino_ratio, max_drawdown, var_95, ... or .to_dict()

benchmark = aq.backtest(universe, aq.buy_and_hold("SPY"))
print(aq.bootstrap_sharpe(result.returns, benchmark.returns).difference)   # paired test, p-value
```

[`examples/quickstart.py`](examples/quickstart.py) does this end to end and plots both equity curves
(matplotlib is optional; without it the script prints the summary only).

## Build

The module is off by default so the core build keeps no dependencies. It needs Python 3.9+ with its
development headers, NumPy, and pybind11:

```bash
python -m pip install pybind11 numpy
```

CMake finds pybind11 through `python -m pybind11 --cmakedir`, using the interpreter given by
`-DPython_EXECUTABLE` (or the first one it finds). The module lands in `build/python/`.

**Linux and macOS**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAXIOM_BUILD_PYTHON=ON -DPython_EXECUTABLE="$(which python3)"
cmake --build build --parallel
ctest --test-dir build --output-on-failure          # includes the python_bindings test
PYTHONPATH=build/python python3 python/examples/quickstart.py
```

On Linux the interpreter's headers come from `python3-dev` (Debian/Ubuntu) or `python3-devel`
(Fedora); a venv or a `setup-python`/pyenv/conda interpreter already has them. On macOS Apple Clang
has no OpenMP, which only makes the parameter sweeps serial.

**Windows (MSVC, python.org Python)**

```bat
py -m pip install pybind11 numpy
cmake -B build -DAXIOM_BUILD_PYTHON=ON -DPython_EXECUTABLE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
set PYTHONPATH=build\python
py python\examples\quickstart.py
```

Use whichever `python.exe` you installed pybind11 into (`py -c "import sys; print(sys.executable)"`
prints it). The module is always written to `build\python`, without a `Release\` subfolder.

**Windows (MSYS2 / MinGW GCC)** also works against the python.org interpreter: configure with
`-G Ninja -DCMAKE_BUILD_TYPE=Release` as above. CPython does not search `PATH` for an extension's
dependencies, so the build links the C++ runtime statically and copies `libgomp-1.dll` and
`libwinpthread-1.dll` next to the module.

To use the module from anywhere, add `build/python` to `PYTHONPATH` (or drop a `.pth` file
containing that path into your environment's `site-packages`).

## API

| Call | Returns |
| --- | --- |
| `load_universe(dir, align="intersection", tickers=None)` | `Universe`: `.tickers`, `.dates`, `len()`, `.closes(t)`, `.close_matrix()`, `.returns_matrix()`, `.slice(b, e)` |
| `buy_and_hold(t)`, `sma_crossover(t, fast, slow)`, `rsi_mean_reversion(t, period, oversold, overbought)`, `momentum(lookback, rebalance, top_k, members=None)`, `vol_target(t, target_volatility, window)` | `Strategy`, a reusable recipe; each backtest builds a fresh instance |
| `CostModel(per_share_commission=0.005, min_commission=1.0, percentage_commission=0.0001, fixed_slippage_bps=2.0, spread_bps=3.0, market_impact_factor=0.1, enable_market_impact=True, max_participation=0.0, short_borrow_rate=0.0)`, `CostModel.free()` | transaction costs; the defaults are the CLI's. `max_participation` caps each order at that share of the bar's volume and carries the rest |
| `backtest(universe, strategy, *, capital, costs, fill="open", risk_free=0.02)` | `BacktestResult`: `.equity`, `.returns`, `.drawdown` (NumPy), `.dates`, `.fills` (list of dicts), `.capped_fills`, `.borrow_cost`, `.metrics`, `.report()` |
| `sharpe_ratio`, `sortino_ratio`, `annualized_volatility`, `max_drawdown` | floats, from returns or an equity curve |
| `sma_sweep(universe, t, fast_grid, slow_grid)` | `SweepPoint`s, best Sharpe first |
| `walk_forward(universe, t, *, train_bars=504, test_bars=126, carry_position=False, ...)` | `WalkForwardResult`: `.folds`, `.oos_returns`, `.benchmark_returns`, `.strategy`, `.benchmark` |
| `bootstrap_sharpe(returns, benchmark=None, *, resamples, block_length=0, seed)` | `BootstrapResult` with Sharpe intervals and p-values; `.report()` |
| `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `politis_white_block_length`, `effective_trials` | as in `analysis/significance.hpp` |

Every function has a docstring (`help(aq.walk_forward)`). Long computations release the GIL.

## Tests

```bash
python -m unittest discover -s python/tests -v
```

The suite uses only the standard library and NumPy. It looks for the module in `build/python` (or
`$AXIOMQUANT_BUILD_DIR/python`), runs the CLI on `sample_data` and checks that the five tournament
strategies give the same returns, Sharpe, Sortino, drawdown, fill counts and equity curves through
both front ends. The CLI comparison is skipped if the CLI was not built.
