# What survivorship bias was worth: the same strategies on a point-in-time S&P 500

**Cross-sectional momentum returned +957.7% on five symbols chosen in 2026, and +106.6% on the index
as it actually was.** Same code, same costs, same dates, same eleven and a half years. The only
change is that the strategy may rank a stock only on days when that stock was really in the S&P 500.
Its Sharpe falls from 0.88 to 0.31 and its worst drawdown rises from 38.9% to **75.6%**.

That gap is the value of hindsight, and it is the main result of this study.

Every number here comes from CI run
[35059578040](https://github.com/Lukey-7/AxiomQuant/actions/runs/35059578040), workflow *S&P 500
study*, job *Point-in-time S&P 500*, `ubuntu-latest`, OpenMP on 4 threads, commit `697c4c4`. The
comparison run on the five hindsight-chosen symbols is
[35057670453](https://github.com/Lukey-7/AxiomQuant/actions/runs/35057670453), the reference run for
[RESEARCH.md](../RESEARCH.md).

---

## 1. What "point-in-time" means here

**Membership.** The public record of S&P 500 components and changes maintained at
[fja05680/sp500](https://github.com/fja05680/sp500) lists, for every symbol, the spells during which
it belonged to the index (`ticker,start_date,end_date`) — exactly the shape `MembershipCalendar`
reads. **771 symbols** were members at some point between 2015-01-02 and 2026-09-15: today's roughly
500 names, plus everything that was dropped, acquired or delisted along the way.

**Prices.** Daily bars for all 771 from the usual provider, plus SPY as the benchmark. **605 had
usable history; 166 did not** — AABA (Yahoo's remains), ABC, ABMD, AET and 162 others that were taken
over or delisted and no longer answer. Those 166 are the residue of survivorship this study cannot
remove: their absence still flatters any strategy that would have held them into a collapse.

**The rule.** Strategies may rank a symbol only on dates when the calendar says it was a member. A
stock joins the investable set on the day it joins the index, and leaves it the day it leaves.
Without the calendar, a 2015 backtest can hold a company that did not join until 2021 — precisely the
bias this measures.

**Alignment.** The universe changes over time, so dates are unioned rather than intersected
(`--align union`): a symbol simply does not exist before its first bar, and its last price is carried
after it stops trading. 2,942 trading days across 606 assets.

```bash
curl -fsSL https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv -o members.csv
python3 scripts/fetch_data.py live --out sp500_data --start 2015-01-01 --allow-missing --tickers <771 symbols> SPY
./build/bin/axiomquant --data sp500_data --members members.csv --align union --ticker SPY --no-db
```

---

## 2. The result

Identical costs, capital and fill rules in both columns; 2,942 bars, 2015-01-02 to 2026-09-15.

| Strategy | Universe | Return | CAGR | Vol | Sharpe | MaxDD | Fills | Costs |
|---|---|---|---|---|---|---|---|---|
| CrossSectionalMomentum (L=60, R=20, Top=2) | 5 symbols picked in 2026 | **+957.7%** | 22.3% | 24.1% | **0.88** | 38.9% | 374 | $43,709 |
| CrossSectionalMomentum (L=60, R=20, Top=2) | point-in-time S&P 500 | **+106.6%** | 6.4% | 39.8% | **0.31** | **75.6%** | 502 | $32,102 |
| BuyAndHold(SPY) | — | +343.2% | 13.6% | 17.4% | 0.71 | 33.4% | 1 | $65 |
| VolTarget(SPY, 10% vol, 20d) | — | +172.2% | 8.9% | 10.2% | 0.69 | 13.2% | 425 | $3,513 |

Three things stand out.

**Momentum stops beating the index.** On the hindsight universe it looked like the best strategy in
the engine by Sharpe (0.88 against buy-and-hold's 0.71). On the real universe it earns less than a
third of the index's return while carrying more than twice its volatility. Rotating into the top two
trailing performers out of ~500 real names means holding whatever just spiked — and on the real index
that includes names on their way to being deleted.

**Risk arrives where the bias was hiding.** A 75.6% peak-to-trough drawdown is not a smaller version
of 38.9%; it is a different strategy. The four mega-caps in the old universe could not all collapse
together, because they were selected for not having collapsed.

**The best strategy is now the boring one.** With momentum's mirage gone, the highest Sharpe among
the active strategies is volatility targeting (0.69), which does not pick stocks at all — it sizes a
single index position by recent volatility. Its Sharpe still does not beat buy-and-hold's 0.71; what
it buys is a 13.2% maximum drawdown instead of 33.4%.

The SPY-only strategies (buy-and-hold, SMA crossover, RSI, volatility targeting) are unaffected by the
universe change, as they should be — they trade one symbol. Their numbers match the reference run
exactly, which is a useful check that nothing else moved.

---

## 3. Costs bite where the edge was thin

Every cost component scaled together, from free trading to ten times the modelled cost:

| Strategy | 0x | 1x | 2x | 5x | 10x | Break-even |
|---|---|---|---|---|---|---|
| BuyAndHold(SPY) | 343.2% | 343.2% | 343.1% | 342.9% | 342.6% | — |
| SMA_Crossover(SPY, 20/50) | 151.0% | 142.7% | 134.7% | 112.3% | 80.2% | — |
| RSI_MeanReversion(SPY, 14) | 133.8% | 130.3% | 126.8% | 116.9% | 101.3% | — |
| **CrossSectionalMomentum (point-in-time)** | 198.5% | 106.6% | 47.7% | **-36.7%** | -80.9% | **3.70x** |
| VolTarget(SPY, 10% vol, 20d) | 177.5% | 172.2% | 166.9% | 152.5% | 127.3% | — |

On the hindsight universe, momentum survived 10x costs with +408.8% left. On the real one it is
underwater by 5x and breaks even at **3.7x** the modelled cost. A strategy that needs today's cost
assumptions to be accurate within a factor of four is not a robust strategy — and this is before any
borrow cost, market-on-close slippage or capacity constraint.

---

## 4. What 606 assets do to the optimizer

The Markowitz section is worth reading as a warning rather than a recommendation. With 606 assets and
2,941 observations, the sample covariance matrix is nearly singular, and the unconstrained solutions
become nonsense:

| | GMV (uncon.) | Tangency (uncon.) | GMV (long-only) | Max Sharpe (long-only) | Risk parity |
|---|---|---|---|---|---|
| Expected return | 0.75% | **-29.86%** | 1.47% | 33.22% | 11.86% |
| Volatility | 1.14% | 5.74% | 1.97% | 15.96% | 13.08% |
| Sharpe (2%) | -1.100 | **-5.547** | -0.267 | **1.957** | 0.754 |

The unconstrained tangency portfolio reports a Sharpe of **-5.5**: the optimizer is fitting noise in
a 606x606 matrix estimated from 12 years of data. Ledoit-Wolf shrinkage drops to **δ = 0.0389** (it
was 0.146 on five assets), because with this many assets the constant-correlation target is a poor
description of the data and the estimator leans on the sample.

The long-only, 40%-capped max-Sharpe portfolio reports an in-sample Sharpe of 1.957. That number is
not a forecast either — it is the best combination *after the fact*, across 606 assets, which is the
largest parameter search anywhere in this repository. Treat it as an illustration of what
optimization does with a wide universe, not as an allocation.

---

## 5. What this study does not fix

- **166 missing symbols.** Delisted names without downloadable history are simply absent, so some
  survivorship remains. A paid point-in-time price database would close this; a free one will not.
- **Index membership is not the investable universe.** Real constraints (borrow, liquidity at the
  open, index-add price pops, corporate actions mid-spell) are not modelled.
- **One index, one regime.** 2015-2026 was kind to US large caps. The same study over 2000-2012 is
  the obvious next test.
- **Momentum's parameters were never tuned here** (L=60, R=20, Top=2 throughout). A search over them
  on the real universe would need the deflated Sharpe ratio from
  [RESEARCH.md](../RESEARCH.md) section 4 to stay honest.

---

## 6. Reproducing it

The *S&P 500 study* workflow does all of the above and uploads the prices, the membership file, the
full log and the CSV exports as artifacts:

```bash
gh workflow run "S&P 500 study" --ref main -f start=2015-01-01
```

It takes about 11 minutes, most of it downloading 771 price histories. `max_symbols` caps the
universe for a quicker run; `start` moves the window.
