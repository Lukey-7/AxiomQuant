# Project notes

Read this first when picking the project up cold. It records the constraints, the verification loop,
what is already done, and what is worth doing next.

Repo: `github.com/Lukey-7/AxiomQuant` (public) · owner account: **Lukey-7**

---

## Hard constraints

1. **No tool or vendor attribution anywhere in the repository.** No commit trailers, no
   `Co-Authored-By`, no mentions in the README, RESEARCH, code comments or docs. This file stays
   about the project itself.
2. **Build and test locally before pushing.** MSYS2/UCRT64 provides GCC, CMake, Ninja, clang-format
   and clang-tidy at `C:\msys64\ucrt64\bin` (add it to PATH). `cmake -S . -B build -G Ninja
   -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel` takes about a minute;
   `./build/bin/quant_tests.exe` runs the suite. clang-format is pinned via pip (21.1.2) so local and
   CI formatting agree - run the pip binary, not the MSYS2 one, which is a different major version.
   Pushing a broken commit emails the owner on every failed run, so verify locally first.
3. **Every number in the README and RESEARCH.md must come from an actual CI run**, and the run and
   runner must be named. Nothing invented, nothing remembered from a previous run. If a figure cannot
   be traced to a run, remove it.
4. **Verification loop:** push to a branch (`improvements`, never straight to `main`) →
   `gh run watch <id> --exit-status` → `gh run view <id> --log-failed` → fix → repeat until green on
   all jobs (four build jobs plus Data tooling) → merge to `main`.
5. Small, focused commits with clear messages. Match the surrounding code style (4 spaces, `snake_case`
   members with a trailing underscore, `[[nodiscard]]` on pure accessors, doc comments on public APIs).

## Machine gotchas

- `gh` has two accounts logged in and **the active one keeps reverting to VD0WQ, which cannot push**.
  Before pushing, check `gh auth status`; if VD0WQ is active, the user must run
  `gh auth switch --user Lukey-7` by hand, then push immediately after.
- Long heredocs through the Bash tool fail with `ENAMETOOLONG`, and backslash escapes inside them get
  mangled. Use the file-writing tool for anything sizeable, and the editing tool for precise patches.
- Files on disk are CRLF (git `autocrlf`); normalise before string matching in scripts.
- Python 3.14 with NumPy is available locally. Useful for computing independent reference values for
  tests — that is how the Ledoit-Wolf test was pinned.

---

## Current state (all green)

`main` is green on Linux/GCC, macOS/Apple Clang, Windows/MSVC and a Clang ASan+UBSan build:
**64 tests** plus an end-to-end CLI run, plus the *Data tooling*, *Optimizer vs reference solver*,
*Benchmark vs NumPy*, *clang-format and clang-tidy* and *Test coverage* (80% line floor, currently
86%) jobs.

Branch `feat/realism-capacity-tests` (19 Oct 2026, not yet merged) brings the suite to **89 tests**
plus three more CTest entries (CLI end to end, golden output, fuzz replay) and adds the *Fuzz the
CSV loader* job; it was green locally (MSYS2 GCC) before its first push, CI still to confirm. Reference run for every
number in RESEARCH.md and the README charts: **35057670453** (workflow *Real data*, job *Fetch and
analyse real prices*, `ubuntu-latest`, 4 threads; Yahoo bars 2015-01-02 to 2026-09-14). The README's
*Sample output* section still quotes synthetic-data run **34671037291** (job *Linux (GCC)*).

### Layout

```
data/         CSV loader (load_file / load_stream, strict row validation), columnar TimeSeries,
              MarketDataUniverse (synchronize, aligned closes, slice)
indicators/   SMA EMA RSI MACD Bollinger rolling-vol ATR (header-only)
backtest/     engine (fill timing, cash guard, optional volume cap with carried remainders, short
              borrow), execution/cost model, portfolio, strategies
risk/         metrics, VaR/CVaR (historical, Gaussian, Cornish-Fisher), text reports, ASCII charts
simulation/   deterministic per-path RNG (rng.hpp, shared with analysis), bootstrap, correlated GBM
optimization/ covariance + Ledoit-Wolf, GMV/tangency, constrained QP (FISTA), frontier, risk parity
analysis/     evaluate_window, sweep_sma_parameters, run_sma_walk_forward, sweep_costs,
              sweep_capacity ($100k-$1B, half-return and break-even size); significance: stationary
              block bootstrap of Sharpe (paired), probabilistic and deflated Sharpe ratio
              regimes: point-in-time trend/volatility labels, per-regime metrics, rolling Sharpe
cli/          axiomquant: 9-stage pipeline, flags, CSV export, cost-of-look-ahead study;
              axiom_optimizer_dump and axiom_bench feed the two cross-check/benchmark CI jobs
python/       optional pybind11 module axiomquant (-DAXIOM_BUILD_PYTHON=ON): universe, backtest,
              metrics, walk-forward, significance; unittest suite cross-checks the CLI; quickstart
tests/        89 tests; test_support.hpp builds synthetic universes and scripted strategies;
              golden/ pins the CLI output on sample_data (golden_diff, update_golden target);
              fuzz/ holds the libFuzzer target, a seeded replay driver and the seed corpus
scripts/      fetch_data.py (stdlib only: live Yahoo/Tiingo, import incl. Kaggle, synthetic),
              test_fetch_data.py (offline unittest), make_charts.py (dependency-free SVG),
              check_optimizer.py (cvxpy cross-check), benchmark.py (NumPy benchmark + SVG)
docs/         guide/quant-from-zero (beginner guide + PDF + build_pdf.py),
              guide/study-guide/STUDY_GUIDE.md, guide/reports/SP500_STUDY.md, images/*.svg
```

### Design decisions worth preserving

- Orders emitted on bar *t* fill at bar *t+1*'s **open**. `FillTiming::SameBarClose` exists only so the
  bias can be measured; the CLI prints that comparison.
- Trade statistics come from realized PnL on closing fills, never from sale proceeds.
- Monte Carlo seeds **per path**, not per thread, so results are identical at any thread count. Keep
  the custom xoshiro256\*\* + polar sampler; `std::normal_distribution` is not portable across
  standard libraries and would break reproducibility.
- The bounded-simplex projection is an exact O(N log N) breakpoint sweep. Do not reintroduce
  renormalisation after clamping — it can violate the box constraints.
- Walk-forward windows start flat; warm-up bars are replayed with `trading_start_index` blocking orders.
- OpenMP is optional everywhere; guard every `omp.h` include and pragma with `#ifdef _OPENMP`.
- The participation cap and borrow fee default to **off**, so every published figure stays valid.
  A carried remainder is cancelled when the strategy sends a newer order for that ticker (the
  strategy sized it from the position it really holds); a buy cut down for cash carries nothing.
- The golden test compares integers exactly and decimals to one unit in the last printed digit.
  Any change that moves numbers on purpose must regenerate `tests/golden/expected`
  (`cmake --build build --target update_golden`) in the same commit, with the diff reviewed.

### Known truths about the data

`sample_data/` is **synthetic**: its "SPY" rises through the March 2020 crash, ends 2024 at 381
(real ≈ 586), starts at exactly 320.0000, and all pairwise correlations sit in 0.539–0.578. That last
point is why Ledoit-Wolf δ clamps to 1.0 on this data — verified against an independent NumPy
implementation, not a bug. On real prices δ = 0.1463. RESEARCH.md says all of this plainly; keep it
that way.

Data secrets on the repo: `KAGGLE_API_TOKEN` (set). `TIINGO_API_KEY` is optional and not set. Yahoo
needs no key but is an unofficial API. The Kaggle path was verified with `camnugent/sandp500`
(`all_stocks_5yr.csv`) in run 34927316003.

---

## Next steps, highest value first

1. ~~Rerun the analysis on real data~~ — done. Follow-ups it exposed: the five-name universe is
   survivorship-biased (use a point-in-time universe, e.g. a Kaggle S&P 500 constituents dataset), and
   flat-start walk-forward biases slow crossovers to cash (10 of 19 folds never traded; add a variant
   that carries signal state across windows).
2. ~~Statistical significance~~ — done (`analysis/significance.*`, RESEARCH.md section 4). OOS Sharpe
   0.185 has p = 0.27; possible follow-ups: Politis-White automatic block length, and an effective
   number of trials for the deflated Sharpe ratio (the 41 SMA pairs are highly correlated).
   Yahoo re-serves slightly different adjusted closes on each download, so always quote one run.
3. ~~Cross-validate the optimizer~~ — done (`scripts/check_optimizer.py`, CI job *Optimizer vs
   reference solver*). It found and fixed a risk-parity bug; solver accuracy is now ~1e-11.
4. ~~Benchmark against vectorised NumPy~~ — done (`scripts/benchmark.py`, `cli/src/bench.cpp`, CI job
   *Benchmark vs NumPy*, README Performance). 1.4-3.7x single-threaded, 3.3-10.8x on 4 threads.
5. ~~Polish~~ - done: clang-format + clang-tidy + coverage jobs, block bootstrap (`--block`),
   cost-sensitivity sweep, volatility targeting (best real-data finding: same Sharpe as buy-and-hold
   with 13.2% drawdown instead of 33.4%).
6. ~~Point-in-time universe~~ - done: the *S&P 500 study* workflow runs the pipeline on the real
   membership record (fja05680/sp500), 771 member symbols since 2015 of which 605 still have prices.
   Momentum falls from +957.7% to +106.6% (Sharpe 0.88 -> 0.31, MaxDD 38.9% -> 75.6%).
   See docs/guide/reports/SP500_STUDY.md, run **35059578040**. RESEARCH.md still uses the five-symbol universe for
   its single-name strategies, which the study confirms are unaffected.
7. ~~Fairer walk-forward~~ - done (`carry_position`, `sweep_walk_forward`; RESEARCH.md section 3).
8. ~~Data-driven significance settings~~ - done (Politis-White block length, `effective_trials`).
9. ~~Execution realism, capacity, golden tests, fuzzing~~ - done on `feat/realism-capacity-tests`
   (backlog items 5, 6, 9, 10 below). Follow-up: quote the capacity table and capacity.svg from a
   named *Real data* run in RESEARCH.md, and try the participation cap on the S&P 500 study.
10. Remaining ideas: cluster-based effective trials, a longer history (from 2000) covering a decade
   that was bad for buy-and-hold, and intraday or higher-frequency data.

## Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel   # CI only, not locally
ctest --test-dir build --output-on-failure -V
./build/bin/axiomquant --data sample_data --no-db --export-dir out
python3 scripts/fetch_data.py --out real_data --start 2015-01-01
python3 scripts/make_charts.py --input out --output docs/images
gh run list --branch main --limit 3
gh run watch <run-id> --exit-status && gh run view <run-id> --log-failed
gh workflow run "Real data"
```

---

## Backlog: ten improvements, with enough detail to start cold

Ordered by value. Each entry says what to build, why it matters, where it goes, and how to prove it
worked. Items 1, 2, 7 and 8 are done (10 Oct 2026); 5, 6, 9 and 10 are done on
`feat/realism-capacity-tests` (19 Oct 2026); 3 and 4 are not started.

### Research credibility

**1. ~~Reality Check / SPA test for strategy selection~~** — done: `superior_predictive_ability` and
`format_spa_report` in `analysis/significance.*` (White 2000 RC + Hansen 2005 SPA, lower/consistent/
upper p-values, shared-index stationary bootstrap), printed by the CLI after the walk-forward
significance block for every tournament strategy against buy-and-hold; tests in
`tests/test_significance.cpp`, pinned against `scripts/spa_reference.py` (NumPy). RESEARCH.md
section 4 describes the method; the real-data p-value still needs quoting from a named CI run.

**2. ~~Regime breakdown and rolling performance.~~** Done: `analysis/regimes.*`, CLI step 7,
`regime_breakdown.csv` / `rolling_sharpe.csv` / `regime_labels.csv`, `rolling_sharpe.svg`, method in
RESEARCH.md section 3. Results still need quoting from a named real-data run.
*What:* split every result by market regime (rising/falling, high/low volatility, e.g. by trailing
200-day return sign and volatility tercile) and report a rolling 252-day Sharpe rather than one
number for 11.7 years. *Why:* "no edge on average" may hide "works only when volatility is high",
which is a more interesting and more defensible finding. *Where:* new `analysis/regimes.*`, printed
after the walk-forward section; chart via `scripts/make_charts.py`. *Proof:* the trend rule should
show visibly better results in falling regimes, which is exactly what folds 4, 7 and 11 hinted at.

**3. Recover some of the 166 missing delisted symbols.**
*What:* the S&P 500 study covers 605 of 771 point-in-time members because free providers drop
delisted tickers. Try a second source (Tiingo with the existing `TIINGO_API_KEY` path, or a Kaggle
dataset of delisted US equities) and merge per symbol. *Why:* the missing names are exactly the ones
that failed, so their absence still flatters every result; this is the last big honesty gap.
*Where:* `scripts/fetch_data.py` (a `--fallback-source` that fills gaps), then rerun the *S&P 500
study* workflow. *Proof:* coverage rises above 605/771 and the momentum result moves (expect it to
get worse again).

**4. Repeat the study on another market.**
*What:* run the same pipeline on European or Japanese equities, or on crypto, with the same
point-in-time discipline where membership data exists. *Why:* "no edge after costs" on one index in
one decade is an anecdote; the same answer across markets is a result. *Where:* no new code needed
for prices (`fetch_data.py live` takes any symbols); membership for other indices needs a new source
for `make_membership.py`. *Proof:* a second report next to `docs/guide/reports/SP500_STUDY.md`.

### Production realism

**5. ~~Execution realism: partial fills, volume caps, borrow costs.~~** Done: `ExecutionConfig::
max_participation` / `short_borrow_rate` (CLI `--max-participation`, `--borrow-rate`; Python
`CostModel(...)`), remainders carried with the order id, `capped_fills`, `cancelled_remainders`,
`total_borrow_cost` and per-fill `participation`; tests in `tests/test_backtest.cpp`. Off by default.
Original entry:
*What:* cap each order at a fraction of the bar's volume (e.g. 10%), carry the remainder to the next
bar, and charge a borrow rate on short positions. *Why:* every order currently fills completely at
one price, which quietly assumes infinite liquidity. *Where:* `backtest/src/engine.cpp` and
`execution_model.hpp`; the `Order` type needs a remaining-quantity field. *Proof:* momentum's fill
count and costs should rise and its return fall; the cost-sensitivity table gains a column.

**6. ~~Capacity analysis.~~** Done: `analysis/capacity.*`, CLI table after cost sensitivity,
`capacity.csv`, `capacity.svg` in `make_charts.py`; the sweep uses a 10% participation cap unless
`--max-participation` is set. Results still need quoting from a named real-data run. Original entry:
*What:* rerun a strategy at increasing account sizes ($100k to $1B) and plot return against capital.
*Why:* answers "at what size does this stop working?", which is the question a desk actually asks,
and the square-root impact model already in `execution_model.hpp` makes it nearly free to compute.
*Where:* a new `analysis/capacity.*` mirroring `cost_sensitivity.*` (same shape: sweep a parameter,
report where profit dies). *Proof:* a capacity curve in the CLI output and a chart.

**7. ~~Python bindings.~~** Done: `python/` (pybind11 module `axiomquant`, built with
`-DAXIOM_BUILD_PYTHON=ON`), `python/README.md`, `python/examples/quickstart.py`; CI job *Python
bindings* runs `python/tests`, which checks every tournament strategy against the CLI's exports.
*What:* expose `MarketDataUniverse`, `BacktestEngine` and the analysis functions through pybind11.
*Why:* makes the engine usable from a notebook, which is how most quant work is actually done, and
far easier to demo live than a CLI. *Where:* new `python/` module plus a CMake target; keep it
optional so the core build stays dependency-free. *Proof:* a notebook that loads sample data, runs a
strategy and plots the equity curve in ten lines.

### Engineering

**8. ~~Speed up the correlated-GBM Monte Carlo.~~** Done: `run_gbm_portfolio` runs paths in blocks of 8
with a hand-tiled packed-Cholesky kernel, summing in the original order so results stay bit-identical
(pinned by `TestSimulation_CorrelatedGbm_MatchesReferenceAlgorithm`). Normal draws and `exp` now
dominate; a vectorised exp/log would be faster but would break bit-identity. Quote speedups only from a
CI run. Original entry:
*What:* on 606 assets it took 296 seconds (169 paths/s) while everything else in the pipeline runs in
seconds. The inner loop multiplies a Cholesky factor by a vector per day per path with poor memory
access. *Why:* it is the one clear performance defect, and it is measurable. *Where:*
`simulation/src/monte_carlo.cpp` `run_gbm_portfolio`; batch paths so the Cholesky product becomes a
matrix-matrix operation, or generate all shocks for a path block at once. *Proof:* `scripts/
benchmark.py` already times this workload — report before/after in the README performance table.

**9. ~~Golden-output regression tests.~~** Done: `tests/golden` (stdout plus seven CSV exports,
`golden_diff`, `update_golden`). A 0.1 bp change to the spread turns it red. Original entry:
*What:* pin the full CLI output on `sample_data` (with fixed seeds and paths) as a checked-in file,
and fail CI on any diff. *Why:* results currently change silently when numerics change; this catches
it. *Where:* `tests/` plus a CI step; store the expected output under `tests/golden/`. *Proof:* a
deliberate one-line change to a formula turns CI red.

**10. ~~Fuzz the CSV loader.~~** Done: `tests/fuzz` and the *Fuzz the CSV loader* CI job. It found
duplicate dates, non-finite prices, negative volume, impossible dates and numbers with trailing
text getting through; all are rejected now. Original entry:
*What:* libFuzzer or AFL++ over `CsvLoader::load_file` with a seed corpus of the sample CSVs.
*Why:* it parses untrusted text and is the most likely place for a crash; the ASan build already in
CI makes findings actionable. *Where:* `tests/fuzz/` with a `-DAXIOM_BUILD_FUZZERS=ON` target, run
for a fixed time in CI. *Proof:* it survives a few minutes of fuzzing without a crash, or it does not
and you fix what it finds.

### If you only do three

1 (SPA test) strengthens the headline claim, 8 (Monte Carlo speed) is a visible, measurable
engineering win with the benchmark harness already in place, and 7 (Python bindings) makes the whole
project demoable in a notebook.
