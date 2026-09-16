# AxiomQuant — Interview Study Guide

Study notes on the concepts, design decisions and results behind this engine.

Repo: `github.com/Lukey-7/AxiomQuant` · Reference runs: **35057670453** (real prices) and
**35059578040** (point-in-time S&P 500) · 64 tests on 4 platforms, 9 CI jobs.

**New to all of this?** Read [Quant From Zero](../quant-from-zero/QUANT_FROM_ZERO.md) first — it
assumes no finance background and walks through the same ideas with runnable experiments. This
document is the interview-preparation companion: denser, and organised around questions you may be
asked.

**Numbers in this guide** come from the runs named above. The results on the bundled synthetic data
(which several sections still quote as a worked example) are labelled as such; every headline figure
lives in [RESEARCH.md](../../../RESEARCH.md) and
[SP500_STUDY.md](../reports/SP500_STUDY.md).

---

## Part 1 — Every concept, in plain English

### 1. Look-ahead bias
**Plain English.** Using information in a backtest that you could not have had at the moment you
traded. The classic version: your rule uses today's closing price, and your backtest also *buys* at
today's close. Nobody can do that — at the close, the close is already gone.

**Why it matters.** It manufactures returns that vanish live. It is the single most common reason a
backtest looks good and a live account does not.

**In the code.** `backtest/src/engine.cpp:94-118`. Orders emitted on bar *t* go into a `queued` list
and are executed at bar *t+1*'s open. `FillTiming` is at `backtest/include/quant/backtest/engine.hpp:19`.
Regression test: `tests/test_backtest.cpp` → `TestBacktest_NextBarOpen_HasNoLookAhead`.

### 2. Transaction costs and market impact
**Plain English.** Four separate things get charged: commission (per share and per dollar, with a
minimum), the bid-ask spread (you buy at the ask, sell at the bid — count half the spread each way),
slippage (the price moves between decision and execution), and market impact (your own order pushes
the price against you).

**The formula.** Impact follows a square-root law: ΔP/P = η·√(Q/V), where Q is your quantity and V is
the bar's volume. Square root, not linear — impact grows sublinearly with size. This is an empirical
regularity found across markets.

**In the code.** `backtest/include/quant/backtest/execution_model.hpp`, `execute_order`.

**The talking point.** On the sample data, momentum pays $4,577 in costs versus buy-and-hold's $67.
Turnover is not free, and a cost model is what makes that visible.

### 3. Sharpe, Sortino, Calmar
- **Sharpe** = (annualized return − risk-free) / annualized volatility. Return per unit of *total*
  risk. Annualize daily numbers with ×252 for the mean and ×√252 for volatility.
- **Sortino** = same numerator, but the denominator only counts *downside* deviation (returns below a
  minimum acceptable return). Rationale: upside volatility isn't risk. Sortino > Sharpe whenever the
  return distribution is right-skewed.
- **Calmar** = CAGR / maximum drawdown. Return per unit of worst-case pain. Favoured by people who
  have to explain a drawdown to investors.

**In the code.** `risk/src/metrics.cpp:55` (Sortino), `:129` (Calmar), `:78` (drawdown).

**Gotcha to mention.** Sharpe assumes roughly normal returns. With fat tails or skew it flatters the
strategy, which is why this project also reports skewness, excess kurtosis and CVaR.

### 4. VaR vs CVaR
- **VaR at 95%** = the loss you exceed 5% of the time. "On the worst 1 day in 20, I lose *at least*
  this much."
- **CVaR / Expected Shortfall at 95%** = the *average* loss on those worst 5% of days.

**Why CVaR is better.** VaR tells you where the cliff edge is, not how far down the rocks are. VaR is
also not sub-additive (a portfolio's VaR can exceed the sum of its parts' VaR), so it's not a coherent
risk measure; CVaR is.

**Three estimators, all implemented:** historical (`risk/src/var_cvar.cpp:66`, empirical quantile, no
distributional assumption), Gaussian parametric (`:90`, assumes normality, understates fat tails),
Cornish-Fisher (`:109`).

### 5. Cornish-Fisher expansion
**Plain English.** Real returns are skewed and fat-tailed, so the normal quantile z is wrong. C-F
corrects it using the sample's skewness S and excess kurtosis K:

```
z_cf = z + (S/6)(z² − 1) + (K/24)(z³ − 3z) − (S²/36)(2z³ − 5z)
```

Then VaR = −μ + z_cf·σ. **In the code:** `risk/src/var_cvar.cpp:121`.

### 6. Geometric Brownian motion and Itô's drift correction
**Plain English.** Prices are modelled as growing at rate μ with random shocks proportional to
volatility σ. In discrete steps: S_{t+1} = S_t · exp((μ − σ²/2)Δt + σ√Δt · z).

**Why the −σ²/2.** This is the part interviewers probe. Log returns and simple returns are not the
same thing: because exp() is convex, E[exp(X)] > exp(E[X]). If you want E[S_T] = S_0·e^{μT}, the
*log* drift must be μ − σ²/2. Without that correction, volatility alone would inflate your expected
price. It is also why the median path of a GBM sits below its mean, and why a volatile asset can have
positive expected return but a median that loses money (volatility drag).

**In the code.** `simulation/src/monte_carlo.cpp:265` for the multi-asset drift.

### 7. Cholesky decomposition for correlated assets
**Plain English.** To simulate assets that move together, you need correlated random shocks. Take the
covariance matrix Σ, factor it as Σ = L·Lᵀ with L lower-triangular (Cholesky), draw independent
normals z, and L·z has exactly covariance Σ.

**In the code.** `simulation/src/monte_carlo.cpp:257`, inner loop at `:300`.

**Gotcha.** Cholesky needs Σ positive-definite. The code retries with a tiny ridge on the diagonal if
the factorization fails — a semi-definite matrix (perfectly collinear assets) would otherwise throw.

### 8. Bootstrapping
**Plain English.** Instead of assuming a distribution, resample actual historical returns with
replacement. This preserves the real fat tails and skew that a normal model erases.

**Trade-off.** i.i.d. resampling destroys autocorrelation and volatility clustering — it says "these
are the days that happen" but not "bad days cluster". Block bootstrap would fix that and is the
natural extension to mention.

**In the code.** `simulation/include/quant/simulation/bootstrap.hpp`; used in the CLI to simulate the
best strategy's forward distribution.

### 9. Markowitz, GMV and tangency
**Plain English.** Given expected returns μ and covariance Σ, mean-variance optimization finds the
weights with the least variance for a given return.

- **GMV (global minimum variance):** w = Σ⁻¹1 / (1ᵀΣ⁻¹1). Notice it uses *only* Σ — no expected
  returns, which is why it's more robust in practice.
- **Tangency (max Sharpe):** w ∝ Σ⁻¹(μ − r_f·1), normalized to sum to 1. Uses μ, and μ is the hardest
  thing in finance to estimate.

**In the code.** `optimization/src/unconstrained.cpp:35` (GMV), `:100-108` (tangency).

**The critique to volunteer.** Unconstrained Markowitz is an error-maximizer: it puts huge weight on
whichever asset has the most overstated expected return. In this project's output the unconstrained
tangency portfolio wants 250% AAPL and −95% MSFT. That's why the long-only, 40%-capped version
exists.

### 10. Ledoit-Wolf shrinkage
**Plain English.** The sample covariance matrix is noisy, especially with many assets and few
observations. Shrink it towards a simple, structured target:

```
Σ_LW = δF + (1−δ)S
```

Here S is the sample covariance and F is the **constant-correlation target**: keep each asset's own
variance, but replace every pairwise correlation with the average correlation r̄. The optimal δ is
estimated from the data:

```
δ = clamp((π̂ − ρ̂) / (T·γ̂), 0, 1)
```

- π̂ = total variance of the sample covariance entries (how noisy S is)
- ρ̂ = covariance between the errors in S and in F (the term that is easy to omit — the original code
  did omit it)
- γ̂ = ‖F − S‖²_F (how wrong the target is)

Intuition: shrink hard when S is noisy (π̂ large) and the target is nearly right (γ̂ small).

**In the code.** `optimization/src/portfolio_stats.cpp:63-97`.

**Your headline result.** On the sample data δ = 1.0 exactly — full shrinkage — because all ten
pairwise correlations lie in 0.539–0.578, so the constant-correlation target is almost exactly right
and γ̂ ≈ 3e-10. Verified against an independent NumPy implementation. That uniformity is itself
evidence the data is synthetic.

### 11. FISTA and projection onto the simplex
**Plain English.** Adding constraints (long-only, 40% cap) kills the closed-form solution, so you
solve numerically. Projected gradient descent: take a gradient step, then project back onto the
feasible set. FISTA (Nesterov acceleration) adds a momentum term that improves convergence from
O(1/k) to O(1/k²).

**The projection.** Projecting onto {w : Σw = 1, lo ≤ w_i ≤ hi} has a known structure: the answer is
w_i = clamp(v_i − θ, lo, hi) for a single scalar θ that makes the weights sum to 1. Finding θ: the
sum is a piecewise-linear, non-increasing function of θ with 2N breakpoints (each coordinate leaves
the upper bound at v_i − hi and hits the lower bound at v_i − lo). Sort the breakpoints and sweep,
maintaining the active set — **O(N log N), exact**. The old code bisected to a tolerance instead.

**In the code.** `optimization/src/constrained_qp.cpp:10-95` (projection), `:112-125` (FISTA loop),
`:180` (golden-section refinement of max-Sharpe).

### 12. Risk parity
**Plain English.** Instead of equalizing dollars, equalize *risk contributions*: each asset
contributes 1/N of portfolio variance. Asset i's risk contribution is w_i·(Σw)_i. Solved here by
cyclical coordinate descent. It ignores expected returns entirely, which is a feature — μ is the
noisiest input.

**In the code.** `optimization/src/unconstrained.cpp:129`.

### 13. Walk-forward analysis and overfitting
**Plain English.** If you test 41 parameter pairs on one dataset and report the best, you've reported
the luckiest, not the best. Walk-forward fixes the protocol: choose parameters on a training window,
then trade them unchanged on the *next* window, which the search never saw. Roll forward and repeat,
then stitch the out-of-sample segments into one track record.

**The number that matters.** In-sample-to-out-of-sample Sharpe *decay*. Here: mean IS 0.217 → mean OOS
−0.954, a decay of 1.17. That gap is the overfitting, quantified.

**In the code.** `analysis/src/walk_forward.cpp:106` (walk-forward), `:69` (parallel sweep).
Each test window starts flat, with warm-up bars replayed but trading disabled
(`EngineConfig::trading_start_index`, applied at `backtest/src/engine.cpp:109`).

**Related concept to name-drop:** multiple-testing / data-snooping bias; deflated Sharpe ratio
(Bailey & López de Prado) adjusts a Sharpe for the number of configurations tried.

---

## Part 2 — Bugs fixed, and why each mattered

Say these as "I found and fixed", with the consequence first — that's what makes it a story.

| # | Bug | Consequence | Fix |
|---|---|---|---|
| 1 | Orders filled at the same bar's close that generated the signal | Classic look-ahead bias; every backtest result was inflated by information no trader has | Queue orders, fill at next bar's open; old behaviour kept as an explicit opt-in (`engine.cpp:94-118`) |
| 2 | Oversized BUY: code set `fill.quantity`, then re-executed the **original** order | The downsize was silently discarded, so the portfolio could spend cash it didn't have — a backtest trading on margin it never had | Resize `order.quantity`, then re-price, looping until commissions and impact fit (`engine.cpp:53-70`) |
| 3 | Momentum strategy never computed momentum | It pushed every asset with `return_val = 0.0`, bought the first K tickers alphabetically, and never sold. It was not a momentum strategy at all | Rank by trailing return P_t/P_{t−L} − 1, hold top K, liquidate names that drop out (`momentum.hpp:66-88`) |
| 4 | Win rate / profit factor counted sale **proceeds** as profit | Every sale looked like a win, so win rate was meaningless and profit factor was noise | Track realized PnL per fill in `Portfolio::process_fill`, compute stats from closing fills only |
| 5 | Monte Carlo seeded per thread | Results changed with thread count and machine — a "5% tail probability" was not reproducible | One xoshiro256\*\* stream per path seeded from (seed, path index); identical at 1 or 64 threads (`monte_carlo.cpp:30-75`) |
| 6 | `run_gbm_portfolio` collapsed everything into one GBM | The README claimed correlated Cholesky simulation; the code simulated a single aggregate asset, understating tail risk from correlation | Real multi-asset simulation through the Cholesky factor (`monte_carlo.cpp:257-305`) |
| 7 | Ledoit-Wolf used a **diagonal** target and dropped ρ̂ | Not the documented estimator; wrong shrinkage intensity | Constant-correlation target with the full ρ̂ term, checked against an independent NumPy reference |
| 8 | Simplex projection was 45 steps of bisection, then renormalized | Approximate, and the renormalization could push weights back outside their box constraints | Exact O(N log N) breakpoint sweep, no renormalization |
| 9 | Indicators indexed the raw series by timeline position | After date-intersection dropped a date, every indicator silently shifted relative to the bar being traded | `get_aligned_closes()` returns a timeline-aligned series |
| 10 | CAGR from bar count (bars/252), hard-coded `created_at` | Wrong annualization (5.17 vs 5.00 years here) and fake timestamps in the database | Calendar span from real timestamps; real UTC time |

---

## Part 3 — 25 interview questions with answers

**1. What does this project do?**
A C++20 research engine: it loads daily OHLCV data, backtests strategies with realistic costs and no
look-ahead, measures risk, optimizes portfolios, runs parallel Monte Carlo, and — the part I care
about — evaluates strategies walk-forward out of sample. On the bundled data it reports honestly that
the strategies have no edge.

**2. Why should I trust your backtest?**
Because it's built to disappoint me. Fills happen at the next bar's open, costs are charged on every
fill, trade statistics come from realized PnL, and the headline result is walk-forward, not in-sample.
Each of those choices lowers the reported return. The in-sample sweep finds Sharpe 0.44; the
walk-forward says −0.95, and I report the second one as the answer.

**3. What is look-ahead bias, and how do you prevent it?**
Using data you couldn't have had at decision time. Here, a signal from bar t's close can only fill at
bar t+1's open — orders are queued, not executed inline. I kept the optimistic mode behind a flag so
the difference can be measured rather than argued about.

**4. Walk me through your cost model.**
Per-share commission with a minimum ticket plus a percentage of notional; half the bid-ask spread in
each direction; fixed slippage in basis points; and square-root market impact, ΔP/P = η√(Q/V), so
large orders relative to bar volume pay more. On the sample data this makes momentum pay 68× the
benchmark's costs, which is the whole point.

**5. Why square-root market impact rather than linear?**
It's the empirical regularity — impact grows roughly with the square root of participation rate, so
doubling size costs about 1.4× not 2×. Linear would overpenalize large orders; a constant would
underpenalize them.

**6. Sharpe vs Sortino vs Calmar — when do you use which?**
Sharpe for a like-for-like default. Sortino when the distribution is skewed and you don't want to
punish upside volatility. Calmar when the binding constraint is drawdown — which it usually is, in
practice, because drawdown is what makes people redeem.

**7. What's wrong with Sharpe?**
It assumes returns are roughly normal and that volatility is the risk. A strategy that sells options
has a lovely Sharpe until it doesn't. That's why I also report skew, excess kurtosis, max drawdown
and CVaR.

**8. VaR vs CVaR?**
VaR is a quantile — the loss you exceed 5% of the time. CVaR is the mean loss beyond it. CVaR is
coherent (sub-additive), VaR isn't, so VaR can tell you a diversified portfolio is riskier than its
parts. I implement historical, Gaussian and Cornish-Fisher VaR so the model risk is visible.

**9. Why Cornish-Fisher?**
Gaussian VaR understates tails for skewed, fat-tailed returns. C-F adjusts the normal quantile using
the sample's skewness and excess kurtosis — a cheap third/fourth-moment correction that doesn't
require committing to a new distribution.

**10. Explain the −σ²/2 in your GBM.**
It's the Itô correction. exp() is convex, so E[exp(X)] > exp(E[X]); to have E[S_T] = S_0e^{μT} with
lognormal prices, the log drift must be μ − σ²/2. It's also volatility drag: the median path sits
below the mean, so a high-volatility asset can have positive expected return and a median that loses.

**11. How do you simulate correlated assets?**
Cholesky: Σ = L·Lᵀ, draw independent standard normals z, and L·z has covariance Σ. I factor the daily
covariance once and reuse L in the inner loop, with a diagonal ridge fallback if Σ is only
semi-definite.

**12. Your Monte Carlo is parallel — how do you keep it reproducible?**
Each path gets its own xoshiro256\*\* stream seeded from (seed, path index) via SplitMix64, and I use a
Marsaglia polar normal sampler rather than `std::normal_distribution`, which differs between standard
libraries. So the output is bit-identical at 1 thread or 64, on any compiler. That's the fix for a
real bug: the original seeded per thread, so results moved with the machine. There's a test that runs
the same config at 1 and 4 threads and asserts exact equality.

**13. Bootstrap or GBM — which and why?**
Both. Bootstrap resamples actual returns, so it keeps the real fat tails; GBM is parametric and
smooth, good for a portfolio with an assumed covariance. Bootstrap's weakness is that i.i.d.
resampling destroys volatility clustering — a block bootstrap would be the upgrade.

**14. Why shrink the covariance matrix?**
Sample covariance is noisy, and mean-variance optimization amplifies exactly that noise into extreme
weights. Shrinkage trades a little bias for a big variance reduction. Ledoit-Wolf gives the optimal
intensity in closed form rather than a guessed constant.

**15. Why the constant-correlation target rather than a diagonal one?**
Equities are genuinely correlated; a diagonal target assumes they aren't, which is a badly wrong
prior. Constant correlation keeps each asset's variance and imposes one average correlation — wrong
in detail, but far closer to the truth, so the estimator can shrink harder.

**16. Your δ came out at exactly 1.0 — isn't that a bug?**
I checked, and no. All ten pairwise correlations are between 0.539 and 0.578, so the target is nearly
exactly right, γ̂ = ‖F−S‖² is about 3e-10, and κ/T = 2.29 clamps to 1. An independent NumPy
implementation reproduces it, and the unit test pins the estimator on a different matrix where δ =
0.0503 — interior, so the formula is genuinely exercised. Incidentally, correlations that uniform are
a fingerprint of synthetic data.

**17. Why can't you solve the constrained portfolio in closed form?**
Inequality constraints. With only the budget constraint you get a Lagrangian solution; add w_i ≥ 0 and
w_i ≤ 0.4 and you have a QP with KKT conditions and an unknown active set. I solve it with FISTA —
gradient step, then exact projection onto the bounded simplex.

**18. Explain the projection.**
The optimal solution has the form w_i = clamp(v_i − θ, lo, hi) for one scalar θ chosen so the weights
sum to 1. The sum is piecewise-linear and non-increasing in θ with 2N breakpoints, so I sort them and
sweep, tracking how many coordinates are at each bound. O(N log N) and exact — the previous version
bisected to a tolerance and then renormalized, which could violate the box constraints.

**19. How do you find the constrained max-Sharpe portfolio?**
Sharpe isn't concave in w, but the constrained frontier is parameterized by risk aversion γ, so I
scan γ on a log grid, then refine with golden-section search around the best point. I also include
γ = 0 (the GMV end) as a candidate.

**20. What is walk-forward analysis, and what did it tell you?**
Choose parameters on a training window, trade them unchanged on the next unseen window, roll forward.
Here: mean in-sample Sharpe 0.217 versus mean out-of-sample −0.954, a decay of 1.17, and a stitched
out-of-sample return of −22.1% over 750 days. The in-sample sweep's best Sharpe of 0.44 is noise.

**21. Three of your folds returned exactly 0.00%. Bug?**
No — each test window starts flat and the SMA strategy only enters on a bullish crossover, which never
arrived inside those six-month windows, all in the 2022 drawdown. Sitting in cash is also why it beat
buy-and-hold in those folds: it dodged a falling market rather than predicting anything. I call that
out explicitly rather than claiming skill.

**22. Your strategy beat buy-and-hold in 3 of 6 folds. Isn't that good?**
Not really. It lost 22% out of sample with a Sharpe of −1.40; buy-and-hold lost more. "Loses less"
isn't an edge, and with six folds that 3-of-6 split is indistinguishable from a coin flip. The
defensible claim is a drawdown profile of 26% versus 39%, and that comes from being out of the market
half the time.

**23. How would you improve the results?**
Real data first — the bundled sample is synthetic. Then: a wider universe, block bootstrap for
volatility clustering, multiple train/test window sizes to check robustness, a deflated Sharpe ratio
to account for the number of configurations tried, and transaction costs calibrated to a real venue.

**24. What's the hardest bug you found?**
The cash-downsizing one. When a BUY exceeded available cash the engine computed a smaller quantity,
wrote it to the fill, then called the execution model again with the *original* order — silently
throwing the downsize away. It never crashed and never produced an obviously wrong number; it just let
the portfolio spend money it didn't have. It needed reading the ordering of the re-pricing call, not
just the arithmetic. The regression test asserts the filled quantity is 99 shares and cash stays
non-negative.

**25. How is it tested and how do you know it's correct?**
64 tests on Linux/GCC, macOS/Clang, Windows/MSVC and an ASan+UBSan build, every push, plus four
jobs that tests alone cannot cover: the optimizer against cvxpy, performance against NumPy,
clang-format/clang-tidy, and an 80% coverage floor. The valuable
ones check against outside truth rather than against the code: the two-asset GMV against its
closed-form solution, Ledoit-Wolf against an independent NumPy implementation, the new exact
projection against the bisection it replaced, correlated GBM against the analytic expected terminal
wealth, historical VaR/CVaR against hand-computed quantiles, and the constrained max-Sharpe against
3,000 randomly sampled feasible portfolios, none of which may beat it.

---

## Part 4 — Things to be ready for

- **"Why C++?"** Backtest sweeps and Monte Carlo are compute-bound and embarrassingly parallel: 50,000
  paths × 252 days in 18 ms, a 41-pair sweep in 13 ms. Deterministic control over the RNG and memory
  layout (columnar time series) matters at that scale. I'd still prototype a signal in Python.
- **"What would you do differently?"** Put walk-forward in from day one; it changes which results you
  even bother reporting. And I'd separate the strategy interface from the engine's order queue earlier
  so intraday fills could slot in.
- **Don't oversell.** The honest summary is: solid machinery, synthetic data, no edge found. That is a
  *better* interview story than a fake Sharpe of 2 — and if you claim the latter, the first question
  will be about your fill assumptions.
- **Know your own numbers.** 2,942 bars, 64 tests, 2,375 out-of-sample days, IS→OOS decay of
  1.17, δ = 1.0, costs $67 vs $4,577.

---

## Part 5 — What changed since the first version of this guide

The engine below is the same; the evidence around it is much stronger. If you are asked "how do you
know your results are real?", these are the answers, newest first.

**Point-in-time universe.** Cross-sectional momentum was tested on five symbols chosen today, and
again on the S&P 500 as it actually was (771 member symbols since 2015, 605 with usable prices).
+957.7% became +106.6%, Sharpe 0.88 became 0.31, drawdown 38.9% became 75.6%. That difference is
survivorship bias, measured rather than asserted. See [SP500_STUDY.md](../reports/SP500_STUDY.md).

**Error bars on every conclusion.** A stationary block bootstrap (block length chosen from the data
by the Politis-White rule) gives the out-of-sample Sharpe of 0.185 a 95% interval of [-0.415, 0.816],
p = 0.277 — indistinguishable from zero, while buy-and-hold's 0.726 is significant at p = 0.013.

**Deflation for the parameter search.** The best of 41 SMA pairs is judged against what the best of N
worthless trials would reach by luck; because the 41 pairs are variations of one rule, they count as
8.1 effectively independent trials rather than 41.

**Cross-validation against an independent solver.** Every optimizer is checked against cvxpy/Clarabel
and closed-form solutions to 1e-8 on each push. This caught a real bug: risk parity was normalising
inside its coordinate descent and converging to a portfolio whose risk contributions differed by up
to 14 percentage points.

**Costs stress-tested.** Every strategy is re-run from 0x to 10x the modelled cost. On the real
universe, momentum breaks even at 3.7x — its apparent profit depends on the cost model being right
within a factor of four.

**Benchmarks against a reference implementation.** The Monte Carlo and bootstrap workloads are timed
against vectorised NumPy doing the same job: 1.4-3.7x faster single-threaded, up to 10.8x on four
threads. Timings in isolation mean nothing; ratios against a known implementation mean something.

**A useful positive result, finally.** Volatility targeting — sizing one index position at
`target vol / trailing vol` — matches buy-and-hold's Sharpe (0.69 vs 0.71) with a 13.2% maximum
drawdown instead of 33.4%. No forecasting: volatility persists, returns do not.
