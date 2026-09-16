# Do simple technical strategies beat buy-and-hold after costs, out of sample?

**Short answer: no.** On 11.7 years of real daily prices for SPY, walk-forward SMA crossover returned
**+32.3%** over 2,375 out-of-sample trading days with a Sharpe of **0.19**. Buying and holding SPY over
the same days returned **+254.6%**, Sharpe **0.73**. The strategy drew down less (18.0% vs 33.3%),
but only because it spent most of a strong bull market in cash. The same parameter search looked
competitive in sample, with a mean in-sample Sharpe of 1.03; out of sample that fell to 0.38.

**And the out-of-sample Sharpe of 0.19 is statistically indistinguishable from zero** (block-bootstrap
95% interval −0.38 to 0.78, p = 0.27), while buy-and-hold's 0.73 is
clearly positive (p = 0.017). See section 4.

Every number below comes from one CI run of this repository, not from a local machine:
[run 35057670453](https://github.com/Lukey-7/AxiomQuant/actions/runs/35057670453), workflow *Real
data*, job *Fetch and analyse real prices*, `ubuntu-latest`, OpenMP on 4 threads, commit `919db29`.
It ran

```bash
python3 scripts/fetch_data.py live --out real_data --provider auto --start 2015-01-01
./build/bin/axiomquant --data real_data --no-db --export-dir real-output
```

The downloaded prices, the full log and the CSV exports are attached to that run as artifacts, so the
exact inputs can be re-used even after the live feed moves on.

---

## 1. Methodology

**Data.** Daily bars from the Yahoo Finance chart API for AAPL, AMZN, GOOGL, MSFT and SPY: 2,941
trading days common to all five, 2015-01-02 to 2026-09-14. Prices are adjusted for splits and
dividends; the adjustment factor is applied to open, high, low and close alike, so intraday and
close-to-close returns are consistent. Because of the adjustment, historical levels are below the
prices printed at the time (the last SPY close in the file is $760.88).

**Costs.** Identical for every strategy, charged on every fill:

| Component | Setting |
|---|---|
| Commission | $0.005/share, $1.00 minimum ticket, plus 1 bp of notional |
| Bid-ask spread | 3 bp total, charged as a 1.5 bp half-spread |
| Slippage | 2 bp fixed |
| Market impact | square-root law, `ΔP/P = 0.1 · √(Q/V)` |

**Execution.** A signal computed from bar *t*'s close is executed at bar *t+1*'s **open**. Filling at
the close that generated the signal leaks information no live trader has; section 2 measures how much.

**Benchmark.** Buy-and-hold SPY, 99% of $100,000 starting capital, charged the same costs.

**Out-of-sample protocol.** Rolling walk-forward: train on 504 bars (~2 years), pick the (fast, slow)
SMA pair with the best in-sample Sharpe from the grid, then trade that pair unchanged on the next 126
bars (~6 months), which the search never saw. Roll forward by 126 bars and repeat, giving 19 folds.
Each test window starts flat; indicator warm-up is replayed without trading, so no position is
inherited from the training period. Test windows tile the data without overlap.

---

## 2. In-sample results (the tempting ones)

Full-sample backtest, 2,941 bars. Sharpe uses a 2% risk-free rate.

| Strategy | Return | CAGR | Vol | Sharpe | Sortino | MaxDD | Fills | Win rate | Costs |
|---|---|---|---|---|---|---|---|---|---|
| BuyAndHold(SPY) | 343.2% | 13.6% | 17.4% | 0.71 | 1.16 | 33.4% | 1 | n/a | $65 |
| SMA_Crossover(SPY, 20/50) | 142.7% | 7.9% | 10.6% | 0.58 | 1.05 | 27.0% | 51 | 60.0% | $5,237 |
| RSI_MeanReversion(SPY, 14) | 130.3% | 7.4% | 13.3% | 0.45 | 0.88 | 27.5% | 24 | 91.7% | $2,374 |
| CrossSectionalMomentum(L=60, R=20, Top=2) | **957.7%** | 22.3% | 24.1% | **0.88** | 1.39 | 38.9% | 374 | 79.5% | $43,709 |
| VolTarget(SPY, 10% vol, 20d) | 172.2% | 8.9% | **10.2%** | 0.69 | 1.22 | **13.2%** | 425 | 81.0% | $3,513 |

On SPY alone, neither timing rule beats doing nothing. Both give up more than half of the benchmark's
return for a lower Sharpe. RSI mean reversion wins 91.7% of its closed trades and still has the worst
Sharpe of the four: a high win rate says nothing when the strategy is out of the market for most of
a rising decade.

**The momentum result is not evidence of an edge.** It rotates between four stocks that were chosen
*today*, knowing they became the largest companies in the world, plus SPY. A universe picked with
hindsight is survivorship bias in its purest form; any rule that tends to hold the recent winners of
that list will look spectacular. It also paid $43,709 in costs, 672× the benchmark's $65. A fair test
needs a point-in-time universe, which the engine now supports: `MembershipCalendar` (`--members`)
restricts the ranking to symbols that were index members on the rebalance date, so a run with a
constituents file cannot pick a name the strategy could not have held. The results here still use the
fixed five-name universe, so this caveat stands for every number in this section.

**Volatility targeting is the one genuinely useful result.** Holding SPY at a weight of
`10% / trailing volatility` delivers a Sharpe of 0.69 against buy-and-hold's 0.71 — no better
risk-adjusted return — but it does so with **13.2% maximum drawdown instead of 33.4%**, and realised
volatility of 10.2% against the 10% target. That is not market timing; it is the well-documented
persistence of volatility. The comparison to keep in mind is a static stock/cash mix, which would
also cut drawdown; the advantage here is that the exposure adapts as volatility changes.

A 41-pair SMA sweep over the full sample finds 5/125 with Sharpe **0.74**, only just above the
benchmark's 0.71. The grid's median is 0.51, its worst 0.34, and **3 of 41** pairs beat buy-and-hold.

**How fragile is any of this to the cost model?** Every strategy is re-run with all cost components
scaled together, from free trading to ten times the modelled cost:

| Strategy | 0x | 1x | 2x | 5x | 10x |
|---|---|---|---|---|---|
| BuyAndHold(SPY) | 343.2% | 343.2% | 343.1% | 342.9% | 342.6% |
| SMA_Crossover(SPY, 20/50) | 151.0% | 142.7% | 134.7% | 112.3% | 80.2% |
| RSI_MeanReversion(SPY, 14) | 133.8% | 130.3% | 126.8% | 116.9% | 101.3% |
| CrossSectionalMomentum | 1,056.6% | 957.7% | 869.3% | 653.1% | 408.8% |
| VolTarget(SPY, 10% vol, 20d) | 177.5% | 172.2% | 166.9% | 152.5% | 127.3% |

Nothing here breaks even inside 10x, because the underlying assets rose enough to absorb the costs.
That is a statement about a bull market in mega-caps, not about the strategies: the ranking is
unchanged at every cost level, and buy-and-hold leads at all of them.

**Cost of look-ahead.** Re-running each strategy with fills at the signal bar's close changes Sharpe
by at most 0.03 (SMA 0.58 → 0.56, RSI 0.45 → 0.42, momentum 0.88 → 0.89). With daily signals on
liquid large caps, the overnight gap is a small part of the result; the protocol matters more.

---

## 3. Out-of-sample results (the honest ones)

| Fold | Test window | Params | IS Sharpe | OOS Sharpe | OOS return | Buy & hold |
|---|---|---|---|---|---|---|
| 1 | 2017-01-03 → 2017-07-03 | 5/200 | 0.24 | 0.00 | 0.00% | +8.20% |
| 2 | 2017-07-05 → 2018-01-02 | 5/200 | 0.74 | 0.00 | 0.00% | +11.99% |
| 3 | 2018-01-03 → 2018-07-03 | 5/200 | 1.77 | 0.00 | 0.00% | +0.66% |
| 4 | 2018-07-05 → 2019-01-03 | 20/50 | 1.21 | 0.00 | 0.00% | -9.63% |
| 5 | 2019-01-04 → 2019-07-05 | 20/50 | 0.88 | 1.04 | +5.36% | +18.95% |
| 6 | 2019-07-08 → 2020-01-03 | 20/50 | 0.94 | 1.87 | +7.49% | +9.90% |
| 7 | 2020-01-06 → 2020-07-06 | 10/100 | 0.83 | 0.73 | +4.94% | -0.90% |
| 8 | 2020-07-07 → 2021-01-04 | 10/50 | 0.63 | -0.64 | -1.95% | +17.96% |
| 9 | 2021-01-05 → 2021-07-06 | 5/150 | 1.10 | 0.00 | 0.00% | +17.54% |
| 10 | 2021-07-07 → 2022-01-03 | 30/150 | 1.46 | 0.00 | 0.00% | +11.78% |
| 11 | 2022-01-04 → 2022-07-06 | 30/150 | 1.70 | 0.00 | 0.00% | -18.91% |
| 12 | 2022-07-07 → 2023-01-04 | 20/50 | 0.41 | -1.39 | -8.68% | -0.11% |
| 13 | 2023-01-05 → 2023-07-07 | 30/200 | 0.00 | 1.02 | +6.83% | +15.35% |
| 14 | 2023-07-10 → 2024-01-05 | 50/150 | 0.76 | 0.98 | +2.88% | +6.90% |
| 15 | 2024-01-08 → 2024-07-09 | 50/150 | 1.01 | 0.00 | 0.00% | +18.11% |
| 16 | 2024-07-10 → 2025-01-07 | 50/150 | 1.73 | 0.00 | 0.00% | +5.38% |
| 17 | 2025-01-08 → 2025-07-11 | 20/200 | 1.53 | 2.28 | +6.39% | +6.87% |
| 18 | 2025-07-14 → 2026-01-09 | 5/150 | 1.40 | 0.00 | 0.00% | +11.10% |
| 19 | 2026-01-12 → 2026-07-14 | 20/200 | 1.17 | 1.26 | +6.36% | +8.52% |

Stitched across all 2,375 out-of-sample days:

| | Return | Sharpe | Max drawdown |
|---|---|---|---|
| SMA, walk-forward | **+32.3%** | **0.19** | 18.0% |
| Buy & hold | +254.6% | 0.73 | 33.3% |

**Mean in-sample Sharpe 1.03 → mean out-of-sample Sharpe 0.38, a decay of 0.65.** The strategy
out-returned buy-and-hold in only **3 of 19** folds.

**Ten folds returned exactly 0.00%.** That is the protocol, not a bug: each test window starts flat and
the strategy only enters on a fresh bullish crossover, which never arrived inside those six months.
The selected pairs were mostly slow (slow SMA of 150–200 bars), so a crossover inside a 126-bar
window is rare. On synthetic data that drifted down, sitting in cash looked defensive; on real data
that mostly rose, it is the main reason the strategy lagged. All three folds where it beat the
benchmark (4, 7 and 11) are periods when SPY fell or went nowhere; in two of them (4 and 11) the
strategy simply never entered.

**Does the conclusion depend on how the test was built?** The protocol above is one choice among
many, so the engine repeats it over several window sizes and in two start modes: `flat` restarts each
test window in cash (above), while `carried` lets the selected pair keep the position it already
implied.

| Train/test | Window start | Folds | OOS return | Buy & hold | OOS Sharpe | B&H Sharpe | Beat B&H |
|---|---|---|---|---|---|---|---|
| 504/126 | flat | 19 | +32.3% | +254.6% | 0.18 | 0.73 | 3/19 |
| 504/126 | carried | 19 | +66.8% | +254.6% | 0.44 | 0.73 | 3/19 |
| 252/63 | flat | 42 | +0.2% | +279.4% | -0.31 | 0.71 | 7/42 |
| 252/63 | carried | 42 | +18.3% | +279.4% | -0.00 | 0.71 | 8/42 |
| 756/189 | flat | 11 | +40.2% | +149.9% | 0.32 | 0.58 | 3/11 |
| 756/189 | carried | 11 | +59.6% | +149.9% | 0.47 | 0.58 | 2/11 |

Carrying the position doubles the strategy's return at 504/126 (+66.8% against +32.3%) and lifts its
Sharpe from 0.18 to 0.44, which confirms that the flat start was costing it real exposure. It does
not change the answer: buy-and-hold wins every row, on both return and Sharpe, and the shorter
252/63 windows are the worst of all. The result is a property of the strategy, not of one split.

**Conclusion.** These strategies show no out-of-sample edge on real prices after realistic costs.
In-sample selection overstated risk-adjusted performance by a factor of about 2.7 (1.03 vs 0.38),
and even the in-sample best pair only matched the benchmark. The lower drawdown is real, but it is
bought by being out of the market, not by timing it: a static stock/cash mix with the same average
exposure would, in principle, keep the benchmark's Sharpe without the turnover.

The walk-forward design itself deserves scrutiny before blaming the idea completely. Starting every
test window flat biases a trend-following rule towards cash; a variant that carries the signal state
across windows (still with no look-ahead) is a fairer test and is listed under limitations.

---

## 4. Is any of this statistically significant?

A difference in Sharpe ratios means little without an error bar. Two tests, both printed by the CLI.

**Out of sample: stationary block bootstrap.** The 2,375 stitched out-of-sample daily returns of the
strategy and of buy-and-hold are resampled together, 10,000 times, in blocks of random length
(geometric, mean **3.1 days**) so that volatility clustering and short-range autocorrelation survive
(Politis & Romano, 1994). The block length is not a rule of thumb: it is estimated from the series'
own dependence by the Politis-White (2004) rule, which minimises the asymptotic error of the
bootstrap variance. Using the same blocks for both series keeps the comparison paired.

| | Sharpe | 95% interval | p-value | Null hypothesis |
|---|---|---|---|---|
| SMA, walk-forward | 0.185 | [−0.415, 0.816] | 0.277 | Sharpe ≤ 0 |
| Buy & hold | 0.726 | [0.130, 1.361] | **0.013** | Sharpe ≤ 0 |
| Difference | −0.541 | [−1.240, 0.143] | 0.124 | no difference |

The probabilistic Sharpe ratio (Bailey & López de Prado, 2012), which corrects the standard error for
skewness and fat tails instead of resampling, puts the probability that the strategy's true Sharpe is
above zero at 0.713: better than a coin flip, far from convincing.

Read together: **there is no evidence the strategy makes money at all**, and good evidence that simply
holding the index does. The evidence that the strategy is *worse* than buy-and-hold is suggestive but
not conclusive at the 5% level (p = 0.12). Nearly 12 years of daily data is not enough to pin a Sharpe
ratio down to better than about ±0.6, which is worth remembering whenever a backtest reports one to
two decimal places.

**In sample: deflated Sharpe ratio.** The best of 41 SMA pairs has Sharpe 0.74. Counting those as 41
independent attempts would overstate the search, because they are variations of one rule: their daily
returns are highly correlated, and the engine discounts them to **8.1 effectively independent trials**
(`effective_trials`). If every trial were worthless, the best of 8.1 noisy estimates with the grid's
observed spread would be expected to reach about 0.19 by luck alone (Bailey & López de Prado, 2014).
The probability that the winner's true Sharpe clears that hurdle is **0.973**.

That can look like a contradiction with the out-of-sample result. It is not. The deflated Sharpe ratio
asks whether the winner beats *zero skill*, and over 2015–2026 almost any rule that is long SPY most
of the time does. It does not ask whether the winner beats *buy-and-hold*, which had Sharpe 0.71 over
the same period without any search. The grid's spread is small because the 41 pairs are all variations
of "mostly long the index", so the luck hurdle is low and the test passes. The walk-forward then shows
that choosing *which* variation, before seeing the data, adds nothing.

---

## 5. Portfolio optimization findings

**Ledoit-Wolf shrinkage returns δ = 0.1462**, well inside (0, 1): the sample covariance gets roughly
15% weight on the constant-correlation target. On the synthetic sample data the same estimator
returned δ = 1.0000, because that data's pairwise correlations were almost perfectly uniform (section
7). Real stocks are not, and the estimator responds exactly as the theory says it should.

| Asset | GMV (unconstrained) | Tangency (unconstrained) | GMV (long-only) | Max Sharpe (long-only) | Risk parity |
|---|---|---|---|---|---|
| AAPL | -9.48% | 58.62% | 18.85% | 33.03% | 18.73% |
| AMZN | -4.40% | 31.44% | 2.77% | 20.58% | 16.30% |
| GOOGL | -4.16% | 40.85% | 17.10% | 23.08% | 18.12% |
| MSFT | -7.93% | 45.80% | 21.27% | 23.30% | 18.64% |
| SPY | 125.97% | -76.71% | 40.00% | 0.00% | 28.21% |
| **Expected return** | 11.18% | 36.17% | 21.48% | 26.81% | 23.31% |
| **Volatility** | 17.17% | 33.13% | 20.86% | 24.69% | 22.04% |
| **Sharpe (2%)** | 0.534 | 1.031 | 0.934 | 1.005 | 0.967 |

The unconstrained tangency portfolio shorts SPY by 77% to fund 177% in the four stocks. That is the
optimizer exploiting the fact that SPY is largely made of those same stocks: short the index, keep
their excess return. It is a textbook example of why unconstrained mean-variance weights are unusable.
With long-only weights and a 40% cap per asset, the max-Sharpe portfolio drops SPY entirely and the
minimum-variance portfolio puts the full 40% cap into it.

Risk parity gives every asset the same share of portfolio variance, which is why SPY — the least
volatile column — carries the largest weight (28.2%) and the four single stocks sit near 17–19%.

Every portfolio in this table is checked against an independent solver on each push: closed-form
linear algebra for the unconstrained columns and cvxpy/Clarabel for the constrained ones, agreeing to
1e-8 (CI job *Optimizer vs reference solver*). That check is what caught the earlier risk parity
implementation, which normalised inside its iteration and left risk contributions up to 14 percentage
points apart.

**These expected returns are in-sample means of the stocks that won the decade.** A 26.9% expected
return is a description of the past, not a forecast. The weights show what the machinery does;
they are not an allocation recommendation.

---

## 6. Monte Carlo

Resampling the momentum strategy's daily returns over a one-year horizon (50,000 paths) gives a
median outcome of +22.6%, a 20.0% probability of losing money and a 95% terminal VaR of 17.7%.
Simulating the long-only max-Sharpe portfolio as correlated GBM gives a median of +26.6% and a 17.0%
probability of loss. Resampling can also draw contiguous blocks rather than single days (`--block`),
which preserves the volatility clustering that i.i.d. resampling destroys. Both models inherit every in-sample bias described above: i.i.d. resampling and
constant-parameter GBM assume the next year looks like an average year of 2015–2026 for these five
names. They are useful for the shape of the risk, not for the level of the return.

---

## 7. About the bundled sample data

The repository ships a synthetic dataset in `sample_data/` so the CLI runs with no network access,
and CI runs on it. It should not be read as market evidence:

- Its "SPY" *rises* through the February–March 2020 crash (304.63 on 2020-02-19 to 340.15 on
  2020-03-23), where the real ETF fell roughly a third.
- Its first bar is exactly 320.0000.
- All ten pairwise correlations sit in a 0.04-wide band (0.539–0.578), which drives Ledoit-Wolf δ to
  exactly 1. An independent NumPy implementation of Ledoit & Wolf (2004) reproduces that value, and
  the unit test pins the estimator on a separate matrix with an interior δ = 0.0503.

An earlier version of this note drew its conclusions from that dataset
([run 34671037291](https://github.com/Lukey-7/AxiomQuant/actions/runs/34671037291)). The headline
answer (no out-of-sample edge) survived the move to real prices; the details did not. On synthetic
data, the strategy beat a falling benchmark by holding cash; on real data the benchmark rose and cash
became the cost.

To reproduce this study or run it on other data, see *Using other data* in the README:
`scripts/fetch_data.py` supports live prices, Kaggle datasets and local CSVs, and synthetic universes.

---

## 8. Limitations

- **Survivorship and selection bias.** Five symbols chosen in 2026, four of them the decade's biggest
  winners. This inflates every long-only and momentum result and is the single biggest caveat here.
  [docs/SP500_STUDY.md](docs/SP500_STUDY.md) now measures exactly this: on a point-in-time S&P 500
  universe (771 member symbols, 605 with usable history), cross-sectional momentum falls from +957.7%
  to +106.6% and its worst drawdown rises from 38.9% to 75.6%. The single-symbol strategies in this
  note are unaffected, but the momentum row above should be read as the biased number it is.
- **One market regime, mostly up.** 2015–2026 contains sharp drawdowns (2018, 2020, 2022) but a
  strongly positive trend overall. A trend-following rule would be judged differently over 2000–2012.
- **The test protocol itself matters** (section 3): carrying the position instead of restarting flat
  doubles the strategy's out-of-sample return without changing the conclusion. Both modes and three
  window sizes are reported rather than one.
- **Wide error bars.** Even with 2,375 out-of-sample days, the Sharpe interval spans about ±0.6
  (section 4). The block length and the effective trial count are now estimated from the data rather
  than assumed, but both are approximations: the effective-trials formula uses a single average
  correlation, where a clustering approach would be sharper.
- **Adjusted prices get revised.** Re-downloading the same dates from Yahoo a few hours apart changed
  the adjusted closes of four of the five symbols in the last decimal places, enough to move the
  in-sample sweep's top Sharpe from 0.73 to 0.74. The artifacts of the reference run pin the exact
  inputs; reproduce from those rather than from a fresh download.
- **Costs are a simple parametric model.** Real slippage depends on order type, venue and volatility.
- **No shorting, no leverage, no position-level risk limits, no intraday data.**
- **Data from an unofficial API.** Yahoo's chart endpoint has no published terms of service for this
  use and can change without notice. The run's artifacts preserve the exact prices used.
