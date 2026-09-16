# Working context for AxiomQuant

Read this first when picking the project up cold. It records the constraints, the verification loop,
what is already done, and what is worth doing next.

Repo: `github.com/Lukey-7/AxiomQuant` (public) · owner account: **Lukey-7**

---

## Hard constraints

1. **Nothing in the repository may attribute the work to an AI assistant.** No commit trailers, no
   `Co-Authored-By`, no mentions in the README, RESEARCH, code comments or docs. This file is working
   context only; keep its content about the project.
2. **There is no C++ compiler or CMake on the development machine.** Do not try to build locally.
   Every change is verified by GitHub Actions. Expect the first push of any large change to fail to
   compile, and budget a fix round.
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
  `gh auth switch --user Lukey-7` (the assistant is not permitted to run it). Push immediately after.
- Long heredocs through the Bash tool fail with `ENAMETOOLONG`, and backslash escapes inside them get
  mangled. Use the file-writing tool for anything sizeable, and the editing tool for precise patches.
- Files on disk are CRLF (git `autocrlf`); normalise before string matching in scripts.
- Python 3.14 with NumPy is available locally. Useful for computing independent reference values for
  tests — that is how the Ledoit-Wolf test was pinned.

---

## Current state (all green)

`main` is green on Linux/GCC, macOS/Apple Clang, Windows/MSVC and a Clang ASan+UBSan build:
**53 tests** plus an end-to-end CLI run, plus the *Data tooling*, *Optimizer vs reference solver* and
*Benchmark vs NumPy* jobs. Reference run for every
number in RESEARCH.md and the README charts: **34930622057** (workflow *Real data*, job *Fetch and
analyse real prices*, `ubuntu-latest`, 4 threads; Yahoo bars 2015-01-02 to 2026-09-14). The README's
*Sample output* section still quotes synthetic-data run **34671037291** (job *Linux (GCC)*).

### Layout

```
data/         CSV loader, columnar TimeSeries, MarketDataUniverse (synchronize, aligned closes, slice)
indicators/   SMA EMA RSI MACD Bollinger rolling-vol ATR (header-only)
backtest/     engine (fill timing, cash guard), execution/cost model, portfolio, strategies
risk/         metrics, VaR/CVaR (historical, Gaussian, Cornish-Fisher), text reports, ASCII charts
simulation/   deterministic per-path RNG (rng.hpp, shared with analysis), bootstrap, correlated GBM
optimization/ covariance + Ledoit-Wolf, GMV/tangency, constrained QP (FISTA), frontier, risk parity
analysis/     evaluate_window, sweep_sma_parameters, run_sma_walk_forward; significance: stationary
              block bootstrap of Sharpe (paired), probabilistic and deflated Sharpe ratio
cli/          axiomquant: 8-stage pipeline, flags, CSV export, cost-of-look-ahead study;
              axiom_optimizer_dump and axiom_bench feed the two cross-check/benchmark CI jobs
tests/        53 tests; test_support.hpp builds synthetic universes and scripted strategies
scripts/      fetch_data.py (stdlib only: live Yahoo/Tiingo, import incl. Kaggle, synthetic),
              test_fetch_data.py (offline unittest), make_charts.py (dependency-free SVG),
              check_optimizer.py (cvxpy cross-check), benchmark.py (NumPy benchmark + SVG)
docs/         STUDY_GUIDE.md, images/*.svg
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
5. **Polish:** `clang-format` + `clang-tidy` in CI, coverage reporting, block bootstrap in the
   simulation module, cost-sensitivity sweep, volatility-targeted position sizing.

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
