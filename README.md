# AxiomQuant

[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Build](https://img.shields.io/badge/Build-CMake%20%7C%20Ninja-brightgreen.svg)]()
[![Parallelism](https://img.shields.io/badge/Parallelism-OpenMP%205.0-orange.svg)]()
[![Linear Algebra](https://img.shields.io/badge/Dependencies-Eigen%203.4-informational.svg)](https://eigen.tuxfamily.org/)
[![Tests](https://img.shields.io/badge/Tests-23%2F23%20Passed-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, high-performance quantitative finance engine written in modern C++20 for strategy backtesting, statistical risk evaluation, OpenMP-parallelized Monte Carlo simulations, and constrained Markowitz portfolio optimization.

---

## Overview

AxiomQuant provides a zero-external-finance-dependency framework for quantitative research, algorithmic trading strategy evaluation, and portfolio construction. All optimization solvers, risk estimators, and indicators are implemented from first mathematical principles using **Eigen** solely for linear algebra.

```
                    +------------------------------------------+
                    |           Historical OHLCV Data          |
                    |           (CSV / SQLite Engine)          |
                    +--------------------+---------------------+
                                         |
                                         v
                         +---------------+---------------+
                         |   MarketDataUniverse Matrix   |
                         +---------------+---------------+
                                         |
            +----------------------------+----------------------------+
            |                            |                            |
            v                            v                            v
+-----------------------+    +-----------------------+    +-----------------------+
| Technical Indicators  |    |  Event-Driven Engine  |    | Portfolio Optimizer   |
| SMA, EMA, RSI, MACD,  |    | Slippage, Spread, Comm|    | Markowitz, Ledoit-    |
| Bollinger, Vol, ATR   |    | Order Execution Model |    | Wolf, Constrained QP  |
+-----------+-----------+    +-----------+-----------+    +-----------+-----------+
            |                            |                            |
            +----------------------------+----------------------------+
                                         |
                                         v
                         +---------------+---------------+
                         |    Risk & Tail Metrics        |
                         | Sharpe, Sortino, VaR/CVaR, MDD|
                         +---------------+---------------+
                                         |
                                         v
                         +---------------+---------------+
                         | OpenMP Monte Carlo Simulation |
                         | 50,000+ paths in < 40ms       |
                         +-------------------------------+
```

---

## Modules

### 1. Data Ingestion & Time-Series (`quant::data`)
- **`TimeSeries`**: Columnar in-memory container holding synchronized OHLCV and volume series.
- **`CsvLoader`**: Robust parser handling arbitrary header orders, ISO dates, and directory batch imports.
- **`MarketDataUniverse`**: Multi-asset calendar alignment with intersection/union forward-filling and aligned matrix extraction.
- **`SqliteStorage`**: Relational persistence for price history, backtest runs, trade logs, and equity curves.

### 2. Technical Indicators (`quant::indicators`)
- **Simple Moving Average (SMA)**: $O(1)$-step sliding window accumulator.
- **Exponential Moving Average (EMA)**: Vectorized recursive exponential weighting ($\alpha = \frac{2}{N + 1}$).
- **Relative Strength Index (RSI)**: Wilder's smoothing on gains and losses.
- **MACD**: Fast/Slow EMA spread, signal line, and histogram.
- **Bollinger Bands**: Rolling mean $\pm k \cdot \sigma$, bandwidth, and $\%B$ position indicator.
- **Rolling Volatility**: Annualized standard deviation of log returns ($\sigma_{\text{ann}} = \sqrt{252} \cdot \operatorname{std}(\ln(P_t / P_{t-1}))$).
- **Average True Range (ATR)**: Wilder's average of True Range $\max(H-L, |H-C_{prev}|, |L-C_{prev}|)$.

### 3. Backtesting Engine (`quant::backtest`)
- **Event-Driven Execution Loop**: Chronological step-by-step market snapshot evaluation.
- **Execution Cost Modeling**:
  - Fixed ticket fees and per-share / percentage commissions.
  - Bid-ask half-spread penalties.
  - Linear slippage and square-root market impact ($\Delta P / P = \eta \sqrt{Q / V}$).
- **Position & Portfolio Accounting**: Accurate average cost basis, realized/unrealized PnL, cash conservation, and drawdown tracking.
- **Built-in Strategies**: SMA Crossover, RSI Mean Reversion, Cross-Sectional Momentum.

### 4. Risk Analytics & Tail Loss (`quant::risk`)
- **Performance Ratios**: Annualized Return (CAGR), Annualized Volatility, Sharpe Ratio ($R_f$ adjusted), Sortino Ratio (downside semi-variance below MAR), Calmar Ratio.
- **Drawdown Analysis**: Max Drawdown, peak/trough indices, duration, and recovery tracking.
- **Higher Moments**: Sample Skewness and Excess Kurtosis.
- **Value at Risk (VaR) & Conditional VaR (CVaR / Expected Shortfall)**:
  - Historical (Empirical quantile)
  - Parametric (Gaussian distribution)
  - Cornish-Fisher Expansion (Adjusted for skewness and fat tails)

### 5. OpenMP Monte Carlo Simulation (`quant::simulation`)
- **Geometric Brownian Motion (GBM)**: Correlated multi-asset path generation via Cholesky decomposition ($\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T$).
- **Empirical Bootstrapping**: Historical return vector resampling with replacement to preserve empirical correlation and kurtosis.
- **Performance**: Simulates **50,000 paths of 252 steps in ~35 ms** utilizing CPU multi-threading.
- **Distribution Analysis**: Terminal wealth quantiles (1st, 5th, 25th, 50th, 75th, 95th, 99th percentiles), drawdown distribution, and probability of capital loss.

### 6. Markowitz Portfolio Optimization (`quant::optimization`)
- **Covariance Estimation**: Sample covariance and Ledoit-Wolf constant correlation shrinkage for ill-conditioned return matrices.
- **Analytical Lagrangian Solvers**:
  - Global Minimum Variance (GMV): $\mathbf{w}_{\text{GMV}} = \frac{\mathbf{\Sigma}^{-1}\mathbf{1}}{\mathbf{1}^T \mathbf{\Sigma}^{-1}\mathbf{1}}$
  - Tangency / Maximum Sharpe: $\mathbf{w}_{\text{tan}} = \frac{\mathbf{\Sigma}^{-1}(\boldsymbol{\mu} - R_f \mathbf{1})}{\mathbf{1}^T \mathbf{\Sigma}^{-1}(\boldsymbol{\mu} - R_f \mathbf{1})}$
- **Constrained Quadratic Programming (QP)**:
  - First-principles Nesterov-accelerated projected gradient descent (FISTA).
  - Exact $O(N \log N)$ bounded simplex projection: $\Pi_{\Delta}(\mathbf{v}) = \operatorname{argmin}_{\mathbf{w}} \|\mathbf{w} - \mathbf{v}\|_2^2 \text{ s.t. } \mathbf{1}^T\mathbf{w} = 1, \ 0 \le w_i \le w_{\max}$.
  - Supports long-only constraints ($w_i \ge 0$) and maximum weight concentration caps ($w_i \le w_{\max}$).
- **Equal Risk Contribution (Risk Parity)**: Cyclical coordinate descent solver.
- **Efficient Frontier Generation**: Sweeps risk-aversion parameters with ASCII curve visualization.

---

## Directory Structure

```
.
├── CMakeLists.txt
├── .gitignore
├── data/
│   ├── include/quant/data/     # TimeSeries, CsvLoader, Universe, SqliteStorage
│   └── src/
├── indicators/
│   ├── include/quant/indicators/ # SMA, EMA, RSI, MACD, Bollinger, ATR, Volatility
│   └── src/
├── backtest/
│   ├── include/quant/backtest/ # Engine, Portfolio, Position, ExecutionModel, Strategy
│   └── src/
├── risk/
│   ├── include/quant/risk/     # RiskMetrics, ValueAtRisk, RiskReport
│   └── src/
├── simulation/
│   ├── include/quant/simulation/ # GbmSimulator, BootstrapSimulator, MonteCarloEngine
│   └── src/
├── optimization/
│   ├── include/quant/optimization/ # PortfolioStats, Unconstrained, ConstrainedQP, Frontier
│   └── src/
├── cli/
│   └── src/main.cpp            # End-to-end interactive CLI runner
├── sample_data/                # 5-asset 5-year OHLCV sample universe
├── tests/                      # Unit test suite (23 test cases)
└── third_party/                # SQLite3 C amalgamation and Eigen headers
```

---

## Build Instructions

### Prerequisites
- C++20 compliant compiler (GCC 14+, Clang 17+, or MSVC 2022)
- CMake 3.20+
- Ninja (recommended) or Make
- OpenMP runtime

### Build with CMake
```bash
# Clone the repository
git clone https://github.com/Lukey-7/AxiomQuant.git
cd AxiomQuant

# Configure and compile
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Testing & Verification

Run the automated test suite verifying indicator math, cost accounting, risk metrics, optimizer analytical benchmarks, and Monte Carlo convergence:

```bash
./build/tests/quant_tests
```

```
=========================================================================
                        RUNNING QUANT ENGINE TESTS                       
=========================================================================
 [PASS] TestOptimization_TwoAsset_Analytical_GMV
 [PASS] TestOptimization_Simplex_Projection
 [PASS] TestOptimization_Constrained_LongOnly_And_BoxCaps
 [PASS] TestOptimization_LedoitWolf_PositiveDefinite
 [PASS] TestSimulation_GBM_Theoretical_Convergence
 [PASS] TestSimulation_Bootstrap_Bounds
 [PASS] TestRisk_CAGR
 [PASS] TestRisk_Sharpe_Sortino
 [PASS] TestRisk_MaxDrawdown
 [PASS] TestRisk_VaR_CVaR
 [PASS] TestBacktest_ExecutionModel_Costs
 [PASS] TestBacktest_Position_PnL
 [PASS] TestBacktest_Portfolio_Cash_Conservation
 [PASS] TestIndicators_SMA
 [PASS] TestIndicators_EMA
 [PASS] TestIndicators_RSI_Monotonic
 [PASS] TestIndicators_MACD
 [PASS] TestIndicators_BollingerBands
 [PASS] TestIndicators_RollingVolatility
 [PASS] TestIndicators_ATR
 [PASS] TestData_TimeSeries_Returns
 [PASS] TestData_Universe_Synchronization
 [PASS] TestData_Sqlite_Persistence_Roundtrip
-------------------------------------------------------------------------
Summary: 23 Passed, 0 Failed, Total: 23
=========================================================================
```

---

## Running the CLI Showcase

Execute the end-to-end quantitative demonstration on the sample multi-asset dataset:

```bash
./build/cli/quant_engine_cli sample_data
```

### Sample Output

#### 1. Markowitz Portfolio Optimization Matrix
```
========================================================================================================
                               MARKOWITZ PORTFOLIO OPTIMIZATION REPORT                                  
========================================================================================================
Asset              GMV (Uncon)  Tangency (Uncon)   GMV (Long-Only) Max Sharpe (Long)       Risk Parity
--------------------------------------------------------------------------------------------------------
AAPL                     8.47%           243.24%            17.92%            40.00%            26.34%
AMZN                    -3.44%           -81.65%             1.11%             0.00%            18.62%
GOOGL                    8.97%            75.13%            17.04%            40.00%            17.93%
MSFT                    12.65%          -100.76%            23.93%             0.00%            16.63%
SPY                     73.35%           -35.95%            40.00%            20.00%            20.48%
--------------------------------------------------------------------------------------------------------
Exp Return               6.49%            43.06%             7.86%            12.30%             8.76%
Volatility              17.75%            53.71%            18.56%            20.82%            20.08%
Sharpe (2%)              0.253             0.764             0.316             0.494             0.336
========================================================================================================

--- ASCII EFFICIENT FRONTIER ---
Return (%)
  12.5% |                                                            
  12.1% |                                                        T   
  11.6% |                                            .               
  11.2% |                     .     .                                
  10.7% |                  .                                         
  10.3% |                                                            
   9.9% |          .                                                 
   9.4% |      .                                                     
   9.0% |    .                                                       
   8.5% |   .                                                        
   8.1% |  ..                                                        
   7.6% |  G                                                         
       +------------------------------------------------------------
          18.5%                                          20.9% Volatility
Legend: [.] Frontier   [G] Global Min Variance   [T] Max Sharpe Tangency
```

#### 2. OpenMP Monte Carlo Risk Report (50,000 Paths)
```
=========================================================================
                   MONTE CARLO RISK & TAIL REPORT                        
=========================================================================
Paths Simulated:     50000 paths
Horizon:             252 trading days (1 Year)
Initial Capital:     $100000.00
Computation Time:    35.1 ms (OpenMP Parallelized)
-------------------------------------------------------------------------
 TERMINAL WEALTH DISTRIBUTION (1 Year)
-------------------------------------------------------------------------
  Expected (Mean):     $104912.00 (4.91 %)
  Median (50th %ile):  $104094.31 (4.09 %)
  Std Deviation:       $12820.49
  99th Percentile:     $138971.68 (38.97 %)
  95th Percentile:     $127348.77 (27.35 %)
  75th Percentile:     $112895.83 (12.90 %)
  25th Percentile:     $95928.27 (-4.07 %)
  5th Percentile:      $85370.68 (-14.63 %)
  1st Percentile:      $78674.81 (-21.33 %)
-------------------------------------------------------------------------
 TAIL LOSS & DRAWDOWN PROBABILITIES
-------------------------------------------------------------------------
  Probability of Capital Loss:    37.02 %
  Probability of > 10% Loss:      11.32 %
  Probability of > 20% Loss:      1.47 %
  Mean Max Drawdown:              11.42 %
  95th %ile Max Drawdown:         20.51 %
  99th %ile Max Drawdown:         25.36 %
  Terminal 95% 1-Year VaR:        14.63 %
  Terminal 95% 1-Year CVaR:       18.78 %
=========================================================================
```

---

## C++ Code Snippets

### Mean-Variance Portfolio Optimization
```cpp
#include "quant/optimization/portfolio_stats.hpp"
#include "quant/optimization/constrained_qp.hpp"

// Compute annualized expected returns and Ledoit-Wolf shrunk covariance matrix
Eigen::VectorXd mu = quant::optimization::PortfolioStats::compute_expected_returns(return_matrix);
Eigen::MatrixXd cov = quant::optimization::PortfolioStats::compute_ledoit_wolf_covariance(return_matrix);

// Long-only constrained optimization with 40% maximum allocation per asset
quant::optimization::ConstrainedQpConfig cfg;
cfg.min_weight = 0.0;
cfg.max_weight = 0.40;

quant::optimization::ConstrainedQpOptimizer optimizer(cfg);
auto max_sharpe_portfolio = optimizer.maximum_sharpe_portfolio(mu, cov);

std::cout << "Optimal Weights:\n" << max_sharpe_portfolio.weights << "\n";
std::cout << "Expected Return: " << max_sharpe_portfolio.expected_return << "\n";
std::cout << "Volatility:      " << max_sharpe_portfolio.volatility << "\n";
```

### Event-Driven Strategy Backtesting
```cpp
#include "quant/backtest/engine.hpp"
#include "quant/backtest/strategies/sma_crossover.hpp"
#include "quant/risk/report.hpp"

quant::backtest::ExecutionConfig exec_cfg;
exec_cfg.fixed_slippage_bps = 2.0;
exec_cfg.spread_bps = 3.0;
exec_cfg.per_share_commission = 0.005;

quant::backtest::BacktestEngine engine(universe, quant::backtest::Portfolio(100000.0), quant::backtest::ExecutionModel(exec_cfg));

quant::backtest::strategies::SmaCrossoverStrategy strategy("SPY", 20, 50);
auto result = engine.run(strategy);

auto summary = quant::risk::RiskReport::evaluate(result, 0.02);
std::cout << quant::risk::RiskReport::generate_text_report(result, summary);
```

---

## Performance Benchmarks

| Benchmark Operation | Dataset Size | Duration |
|---|---|---|
| **Monte Carlo Simulation (50k paths, 252 steps)** | 50,000 paths | **35.1 ms** (OpenMP) |
| **Constrained QP Optimizer (5 assets, FISTA)** | 2,000 iterations | **0.4 ms** |
| **5-Year Multi-Asset Event-Driven Backtest** | 1,304 daily bars | **2.8 ms** |
| **Ledoit-Wolf Covariance Matrix Shrinkage** | 1,304 x 5 matrix | **0.6 ms** |
| **RSI / MACD Vectorized Indicators (1,304 bars)** | 1,304 bars | **< 0.1 ms** |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
