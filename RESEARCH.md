# Do simple technical strategies beat buy-and-hold after costs, out of sample?

**Short answer: no.** On the bundled dataset, walk-forward SMA crossover lost **-22.1%** over 750
out-of-sample trading days with a Sharpe of **-1.40**. It lost *less* than buy-and-hold (-32.6%,
Sharpe -0.75) and drew down less (26.2% vs 39.4%), but "lost money more slowly" is not an edge.
The same parameter search looked profitable in sample — that gap is the entire point of this note.

Every number below comes from a CI run of this repository, not from a local machine:
[run 34671037291](https://github.com/Lukey-7/AxiomQuant/actions/runs/34671037291), job *Linux (GCC)*,
Release build, `axiomquant --data sample_data --no-db --export-dir cli-output`. The raw log and the
CSV exports are attached to that run as artifacts.

---

## 1. Methodology

**Universe.** 5 symbols (AAPL, AMZN, GOOGL, MSFT, SPY), daily OHLCV bars, 1,304 dates common to all
five, labelled 2020-01-02 to 2024-12-31.

**Costs.** Identical for every strategy, charged on every fill:

| Component | Setting |
|---|---|
| Commission | $0.005/share, $1.00 minimum ticket, plus 1 bp of notional |
| Bid-ask spread | 3 bp total, charged as a 1.5 bp half-spread |
| Slippage | 2 bp fixed |
| Market impact | square-root law, `ΔP/P = 0.1 · √(Q/V)` |

**Execution.** A signal computed from bar *t*'s close is executed at bar *t+1*'s **open**. This is the
single most important methodological choice here: filling at the close that generated the signal
leaks information that no live trader has.

**Benchmark.** Buy-and-hold SPY, 99% of capital, charged the same costs.

**Out-of-sample protocol.** Rolling walk-forward: train on 504 bars (~2 years), pick the (fast, slow)
SMA pair with the best in-sample Sharpe from a 19-pair grid, then trade that pair unchanged on the
next 126 bars (~6 months), which the search never saw. Roll forward by 126 bars and repeat. Each test
window starts flat; indicator warm-up is replayed without trading, so no position is inherited from
the training period. Test windows tile the data without overlap.

---

## 2. In-sample results (the tempting ones)

Full-sample backtest, 1,304 bars:

| Strategy | Return | CAGR | Vol | Sharpe | Sortino | MaxDD | Fills | Win rate | Costs |
|---|---|---|---|---|---|---|---|---|---|
| BuyAndHold(SPY) | 18.3% | 3.4% | 18.3% | 0.16 | 0.39 | 46.8% | 1 | n/a | $67 |
| SMA_Crossover(SPY, 20/50) | 26.4% | 4.8% | 12.2% | **0.27** | 0.66 | 26.0% | 27 | 38.5% | $2,177 |
| RSI_MeanReversion(SPY, 14) | -4.5% | -0.9% | 11.1% | -0.20 | -0.03 | 26.8% | 10 | 60.0% | $606 |
| CrossSectionalMomentum(L=60, R=20, Top=2) | 13.0% | 2.5% | 20.7% | 0.12 | 0.31 | 52.0% | 165 | 61.4% | $4,577 |

A 41-pair sweep over the same full sample finds SMA 40/50 with Sharpe **0.44** — nearly three times
the benchmark. Note the dispersion: best 0.44, **median -0.56**, worst -1.04, and only **5 of 41**
pairs beat buy-and-hold. A grid whose median member loses money is a grid whose winner is mostly
luck.

Two details worth flagging:

- RSI mean reversion has the *highest* win rate (60%) and the *worst* Sharpe. Win rate says nothing
  about profitability when losers are larger than winners.
- Momentum pays $4,577 in costs, 68× the benchmark's $67, for a worse risk-adjusted result.

---

## 3. Out-of-sample results (the honest ones)

| Fold | Test window | Params | IS Sharpe | OOS Sharpe | OOS return | Buy & hold |
|---|---|---|---|---|---|---|
| 1 | 2021-12-08 → 2022-06-01 | 30/50 | 1.25 | 0.12 | +1.3% | +2.7% |
| 2 | 2022-06-02 → 2022-11-24 | 30/100 | 0.26 | 0.00 | 0.0% | -10.8% |
| 3 | 2022-11-25 → 2023-05-19 | 30/100 | 0.31 | 0.00 | 0.0% | -12.7% |
| 4 | 2023-05-22 → 2023-11-13 | 30/100 | 0.03 | -2.34 | -12.0% | -0.9% |
| 5 | 2023-11-14 → 2024-05-07 | 20/100 | -0.27 | 0.00 | 0.0% | -3.0% |
| 6 | 2024-05-08 → 2024-10-30 | 20/50 | -0.28 | -3.50 | -12.6% | -12.3% |

Stitched across all 750 out-of-sample days:

| | Return | Sharpe | Max drawdown |
|---|---|---|---|
| SMA, walk-forward | **-22.1%** | **-1.40** | 26.2% |
| Buy & hold | -32.6% | -0.75 | 39.4% |

**Mean in-sample Sharpe 0.22 → mean out-of-sample Sharpe -0.95, a decay of 1.17.** The optimizer's
fold-1 pick looked spectacular in training (IS Sharpe 1.25) and delivered 0.12 live.

Three folds returned exactly 0.00%. That is not a bug: each test window starts flat and the strategy
only enters on a bullish crossover, which never arrived inside those six-month windows. Sitting in
cash is why the strategy beat buy-and-hold in folds 2, 3 and 5 — it dodged a falling market rather
than predicting anything. It beat the benchmark in 3 of 6 folds and still finished deeply negative.

**Conclusion.** These strategies show no out-of-sample edge on this data after realistic costs. The
in-sample sweep is a mirage: its best Sharpe (0.44) and the walk-forward reality (-0.95) differ by
more than a full unit. The defensive behaviour (lower drawdown, cash during downtrends) is the only
durable-looking property, and it comes from being out of the market, not from timing it.

---

## 4. Portfolio optimization findings

Ledoit-Wolf shrinkage returns **δ = 1.0000** — full shrinkage to the constant-correlation target.
That is a real result, not a clamp bug: all ten pairwise correlations lie in **0.539–0.578** around
r̄ = 0.557, so the target is almost exactly right, ‖F − S‖² ≈ 3.0e-10, and the optimal intensity
κ/T = 2.29 clamps to 1. An independent implementation of Ledoit & Wolf (2004) in NumPy reproduces
δ = 1.0000 on the same inputs, and the unit test pins the estimator against that reference on a
separate matrix where δ = 0.0503 (interior, so the formula is genuinely exercised).

Uniform correlations like these are a fingerprint of synthetic data; real large-cap correlations are
far more dispersed, and δ would normally land well inside (0, 1).

The constrained optimizer (long-only, 40% cap) produces the expected shape: unconstrained tangency
demands 250% AAPL and -95% MSFT, while the constrained max-Sharpe portfolio holds 40/40/20 across
AAPL, GOOGL and SPY with Sharpe 0.49 versus 0.25 for global minimum variance.

---

## 5. About the sample data

**The bundled `sample_data/` is synthetic, and the results above should be read as a demonstration of
the machinery, not as a claim about real markets.** The evidence:

- Its "SPY" *rises* through the February–March 2020 crash: 304.63 on 2020-02-19 to 340.15 on
  2020-03-23, where the real ETF fell roughly a third.
- It ends 2024 at 381.03; real SPY closed 2024 near $586.
- The first bar is exactly 320.0000.
- All pairwise correlations sit in a 0.04-wide band, which does not happen with real equities.

The repository ships it so the CLI runs out of the box with no network access. To draw conclusions
about real markets, point `--data` at a directory of real daily OHLCV CSVs (`Date,Open,High,Low,
Close,Adj Close,Volume`) and re-run; the loader accepts any column order. No data-fetching script is
included, because nothing in this repository can reach the network in CI and I will not ship code
this project has never executed.

---

## 6. Limitations

- Five symbols, one asset class, five years of daily bars — far too small for a real conclusion.
- The universe is fixed and survivorship-biased by construction.
- Costs are a simple parametric model. Real slippage depends on order type, venue and volatility.
- No shorting in the tested strategies, no leverage, no position-level risk limits, no intraday data.
- Walk-forward uses one train/test split (504/126). A robustness study over several window sizes
  would be the natural next step.
- Sharpe ratios over 126-bar windows are noisy; single-fold numbers should not be over-read.
