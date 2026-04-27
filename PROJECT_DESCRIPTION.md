# RAMPA — Project Description

**Full Name:** Regime-Aware Multi-Agent Portfolio Allocator  
**Author:** Felipe Cardozo  
**Institution:** Emory University — CS 534  
**Version:** 1.0.0  
**License:** MIT

---

## 1. Overview

RAMPA is a hierarchical machine learning pipeline for portfolio allocation that replaces static mean–variance optimization (MVO) with a modular, regime-aware system. The core argument is that standard MVO is fragile because it assumes stationary return moments and relies on covariance estimates that are unstable in finite samples. In practice, markets exhibit time-varying volatility, structural breaks, and heavy tails that systematically violate those assumptions.

The system organizes portfolio construction into five sequential phases, each addressing a distinct statistical subproblem and exposing well-defined artifacts to the next stage. The five phases are: feature engineering and dimensionality reduction, hidden regime detection and classification, cross-sectional alpha signal generation, rough volatility calibration, and reinforcement learning for final weight allocation. All phases are decoupled — each can be retrained or swapped independently without breaking the overall pipeline.

The end product is a walk-forward backtested portfolio allocator that, in reported experiments, outperforms a 60/40 benchmark, an equal-weight portfolio, and a classical rolling-window MVO on Sharpe Ratio, Sortino Ratio, Calmar Ratio, and maximum drawdown.

---

## 2. Problem Statement

Classical mean–variance portfolio optimization has three well-documented failure modes:

1. **Estimation error amplification.** The optimizer concentrates weight in assets with the highest estimated expected return, which are also the most uncertain estimates. Small input errors produce extreme and unstable output weights.
2. **Stationarity assumption.** MVO treats return moments as fixed. In reality, volatility regimes, correlations, and expected returns all shift over the business cycle.
3. **Single-period myopia.** Standard MVO ignores path-dependence, regime transitions, and higher-order predictive features available in market data.

RAMPA addresses all three by conditioning allocation decisions on latent market regimes, generating alpha signals from high-dimensional features, and framing final weight selection as a sequential decision problem solved with reinforcement learning.

---

## 3. Dataset

| Property | Details |
|---|---|
| **Asset universe** | SPY (U.S. large-cap equity), QQQ (U.S. growth/tech equity), IEF (intermediate Treasuries), GLD (gold), SHV (short-duration cash-like Treasuries) |
| **Sample period** | January 2010 – December 2024 (daily frequency) |
| **Data sources** | Equity/ETF prices via Polygon.io; macroeconomic indicators via FRED (St. Louis Fed); options/IV data via Polygon.io |
| **Raw targets** | Daily adjusted closing prices |
| **Derived series** | Log returns, realized volatility, cross-sectional spreads, yield-curve tenors, VIX-derived implied vol measures, momentum indicators |
| **Feature count** | 100+ raw engineered features; compressed to ~15 principal components via PCA |
| **Stress events covered** | COVID-19 crash (Feb–Apr 2020), 2022 Federal Reserve rate-hike cycle |
| **Storage format** | Apache Parquet (features, regime labels, alpha signals, vol features, IV surfaces) |

The five-ETF universe was chosen to ensure data availability and clean backtesting. SPY and QQQ provide equity exposure; IEF and SHV serve as diversifiers/defensive assets; GLD provides a crisis hedge with low equity correlation. The low correlation of GLD with equities and the historically negative correlation of IEF with SPY during stress periods are central to the portfolio construction logic.

---

## 4. Pipeline Architecture

```
RAW DATA  (prices, macro, options)
    │
    ▼
Phase 1 — Feature Engineering & Dimensionality Reduction
    PCA (Eigen-Portfolios) → Lasso/Ridge Baseline → Online SGD
    │
    ▼
Phase 2 — Regime Classification
    HMM Generative Labeling → Naive Bayes Baseline → RBF-SVM (production)
    │
    ▼
Phase 3 — Alpha Signal Generation
    Decision Tree (depth=5) → Random Forest (200–500 trees) → LightGBM (leaf-wise boosting)
    │
    ▼
Phase 4 — Volatility Oracle
    GARCH(1,1) Baseline → IV Surface Construction → Deep Vol Net (rBergomi Calibration)
    │
    ▼
Phase 5 — RL Execution Engine
    PortfolioEnv (Gymnasium) → PPO Agent → Regime-Aware Portfolio Weights
    │
    ▼
Walk-Forward Backtest → Performance Report vs. Benchmarks
```

---

## 5. Methods

### 5.1 Phase 1 — Feature Engineering and Dimensionality Reduction

**Goal:** Transform raw price, macro, and derivative time series into a compressed, stationary feature matrix.

**Steps:**
- Compute log returns, realized volatility, and momentum signals per asset.
- Augment with macroeconomic indicators from FRED (yield-curve levels, spreads, industrial production) and option-implied features (VIX level, term structure).
- Apply fractional differencing at the minimum stationary order `d*` to preserve long-range dependence that first-differencing discards, while satisfying the stationarity requirements of downstream classifiers.
- Apply PCA to the standardized 100+ feature matrix to extract orthogonal latent factors (eigen-portfolios). Fewer than 15 components typically explain 90% of variance.
- Train linear regression baselines (Lasso, Ridge, online SGD) on PCA factors as return forecasting benchmarks.

**Key outputs:** `data/processed/features.parquet` (compressed feature matrix).

### 5.2 Phase 2 — Regime Classification

**Goal:** Assign each trading day to one of four latent market regimes.

**Regimes defined:**
| Regime | Description |
|---|---|
| Trending Bull | Persistent positive momentum, low volatility |
| Choppy / Sideways | No clear direction, moderate volatility |
| High-Vol Stress | Elevated volatility, declining equity trend |
| Crisis | Extreme volatility, sharp drawdown |

**Steps:**
1. Fit a Hidden Markov Model (HMM) to macro and volatility-sensitive features. The HMM captures regime persistence (high self-transition probabilities) and smooth posterior state probabilities.
2. Decode the Viterbi path to generate historical regime labels.
3. Train discriminative classifiers on these labels for out-of-sample prediction:
   - **Naive Bayes** (baseline): assumes feature independence.
   - **RBF-SVM** (production): flexible kernel decision boundaries; meaningfully higher accuracy across all four classes.
4. Out-of-sample confusion matrices show the most common misclassification is between Trending Bull and Choppy — adjacent regimes in feature space.

**Key artifacts:** `data/processed/regime_labels.parquet`, `models/svm_regime.pkl`.

### 5.3 Phase 3 — Alpha Signal Generation

**Goal:** Produce a continuous cross-sectional signal predicting next-day SPY log return, conditioned on current regime.

**Model progression:**

| Model | Mean IC (Trending Bull) | Notes |
|---|---|---|
| Decision Tree (depth=5) | 0.018 | Fully interpretable; IC floor |
| Random Forest (200–500 trees) | 0.044 | +144% vs. Decision Tree via variance reduction |
| LightGBM (leaf-wise boosting) | 0.082 | +86% vs. Random Forest; 10× faster than depth-wise |

- **Target:** `r_{t+1}` (next-day SPY log return).
- **Signal:** `s_t = f_LGB(x_t, z_t) ∈ [−1, 1]` where `z_t` is the regime indicator.
- **Interpretation:** `s_t > 0.5` → net long equity; `s_t < 0.3` → defensive tilt (IEF/SHV/GLD).
- `regime_id` ranks 4th among all features by LightGBM gain, confirming that regime conditioning adds predictive content beyond technical indicators alone.
- Out-of-sample IC (Spearman correlation of signal vs. realized return) ranges from 0.082 in Trending Bull to near zero in Choppy — demonstrating that regime conditioning is necessary, not optional.

**Key artifacts:** `data/processed/alpha_signals.parquet`, `models/lgbm_alpha.pkl`.

### 5.4 Phase 4 — Volatility Oracle

**Goal:** Estimate time-varying rough volatility parameters to provide a forward-looking risk signal.

**Motivation:** Gatheral, Jaisson & Rosenbaum (2018) showed that realized volatility exhibits rough fractional dynamics with Hurst exponent H < 0.5, inconsistent with standard Brownian-based GARCH models.

**Steps:**
1. Fit GARCH(1,1) to SPY returns as a baseline conditional volatility estimate.
2. Construct daily implied volatility surfaces from SPY option chains (moneyness × expiry grids).
3. Train a convolutional deep neural network on synthetic Monte Carlo data generated under the rough Bergomi (rBergomi) model to learn the mapping: `IV surface → (H, η, ξ₀)` where H is the Hurst exponent, η the vol-of-vol, and ξ₀ the initial variance.
4. Apply the trained network to historical surfaces to produce time-series of rough vol parameters.

**Key finding:** Estimated H values consistently below 0.5, confirming rough volatility empirically. Episodic spikes toward H = 0.3–0.4 during stress periods indicate temporarily less rough behavior, consistent with higher volatility persistence during crises.

**Key artifacts:** `data/processed/vol_features.parquet`, `data/processed/iv_surfaces.parquet`, `models/deep_vol_net.pt`.

### 5.5 Phase 5 — Reinforcement Learning Execution Engine

**Goal:** Learn a dynamic portfolio allocation policy that maps current observations to portfolio weights, subject to leverage and drawdown constraints.

**Environment:**
- **Framework:** Custom Gymnasium-compatible `PortfolioEnv`.
- **Action space:** Continuous vector of portfolio weights over the 5-ETF universe (simplex-constrained).
- **Observation space:** Current engineered features, regime posterior probabilities, rough volatility parameters.
- **Constraints:** Leverage (long-only, sum-to-one), max turnover, 10% drawdown hard floor.

**Reward function:** Markovian step-level proxy combining:
- Risk-adjusted instantaneous return.
- Drawdown penalty.
- Turnover cost penalty.

The Sharpe ratio is non-Markovian (depends on full path history), so a Markovian proxy is required for PPO. The proxy is calibrated so that policies maximizing cumulative discounted reward also achieve high episode-level Sharpe ratios.

**Algorithm:** Proximal Policy Optimization (PPO) — chosen for robustness to hyperparameter choice and compatibility with continuous action spaces via clipped surrogate loss.

**Training:** ~1 million environment timesteps to convergence; 50-episode rolling reward monotonically improves above the random agent baseline.

**Key learned behavior:** The agent shifts toward IEF and SHV during the 2020 COVID crash and 2022 rate-hike period, then returns to equity-heavy positioning during the 2023 recovery — demonstrating learned regime-conditional risk management.

**Key artifacts:** `models/ppo_agent.zip`.

---

## 6. Backtest Design and Validation

### 6.1 Walk-Forward Backtest

- **Window:** Expanding window starting 2015; models trained on all data up to time `t`, evaluated on the next out-of-sample segment.
- **Benchmarks:** (1) Static 60/40 (SPY/IEF), (2) Equal-weight (all 5 ETFs), (3) Classical rolling-window MVO.
- **Rebalancing:** Monthly (end-of-month), with proportional transaction costs applied.
- **Transaction cost model:** Proportional spread (10bps default) + optional square-root market impact term.

### 6.2 Model Validation — Purged Cross-Validation

Regime and alpha models are validated using purged cross-validation with embargo periods (Lopez de Prado, 2018):
- Observations temporally adjacent to test folds are removed from training.
- An embargo window is applied around test periods to prevent leakage via overlapping labels or autocorrelated features.
- This prevents the inflated out-of-sample estimates produced by standard time-series train/test splits.

All hyperparameters are fixed before the final backtest. No re-tuning is performed on test-period information.

### 6.3 Performance Results

| Metric | RAMPA | 60/40 | Equal-Weight | MVO |
|---|---|---|---|---|
| Sharpe Ratio | Highest | — | — | — |
| Sortino Ratio | Highest | — | — | — |
| Calmar Ratio | Highest | — | — | — |
| Max Drawdown | Shallowest (≤10% hard floor) | >20% in 2022 | — | — |
| Terminal Value | Highest | — | — | — |

Key observations:
- RAMPA's Sortino improvement exceeds its Sharpe improvement, indicating excess returns come from upside capture rather than additional downside risk.
- RAMPA outperforms most consistently during regime transitions, where regime-conditional policy provides informational advantage over static allocation.
- The 10% maximum drawdown constraint imposed on the RL agent is visible as a hard floor in the RAMPA drawdown curve.

---

## 7. The `quantopt` Python Package

The repository includes a custom Python library (`quantopt/`) implementing classical portfolio optimization components used throughout the project. It is installed as an editable package (`pip install -e .`).

### 7.1 Returns Estimators (`quantopt/returns/`)

| Class | Method |
|---|---|
| `MeanHistoricalReturn` | Arithmetic or geometric annualized mean; optional exponential weighting |
| `CAPMReturn` | OLS Beta against market; `E[R_i] = Rf + β_i × (E[R_m] - Rf)` |
| `BlackLittermanReturn` | Full Bayesian posterior blending market equilibrium implied returns with investor views via pick matrix P and view vector Q |

### 7.2 Covariance Estimators (`quantopt/risk/`)

| Class | Method |
|---|---|
| `SampleCovariance` | Standard sample covariance, annualized; condition number monitoring; PSD projection |
| `EWMCovariance` | Exponentially weighted moving covariance with configurable decay span; captures volatility clustering |
| `LedoitWolfCovariance` | Oracle Approximating Shrinkage (OAS); blends sample covariance toward structured target to minimize MSE |
| `FactorModelCovariance` | Barra-style PCA factor model; decomposes covariance into systematic (K-factor) + idiosyncratic diagonal; K selected by variance threshold |

All estimators expose `.covariance()`, `.correlation()`, `.std()` and enforce positive semi-definiteness via eigenvalue projection.

### 7.3 Optimizers (`quantopt/optimization/`)

| Class | Objective |
|---|---|
| `EfficientFrontier` | Max Sharpe (SLSQP, 5 random restarts), Min Volatility, Efficient Return (target return), Efficient Risk (target vol), full frontier trace |
| `CVaROptimizer` | Minimizes Conditional Value-at-Risk via Rockafellar-Uryasev smooth approximation; supports optional mean return constraint |
| `RiskParity` | Equal Risk Contribution / generalized risk budgeting; minimizes squared deviation of actual vs. target risk contributions; 15 random restarts |

`EfficientFrontier` supports L2 regularization (`l2_gamma`) to penalize extreme weight concentration, and a `ConstraintSet` API for custom bounds and linear/inequality constraints.

### 7.4 Analytics (`quantopt/analytics/`)

Performance metrics implemented:

| Function | Description |
|---|---|
| `annualized_return` | Geometric or arithmetic annualization |
| `annualized_volatility` | Annualized standard deviation |
| `sharpe_ratio` | (Ann. Return − Rf) / Ann. Vol |
| `sortino_ratio` | Downside deviation denominator |
| `calmar_ratio` | Ann. Return / \|Max Drawdown\| |
| `omega_ratio` | Ratio of upside to downside returns relative to threshold |
| `value_at_risk_historical` | Empirical VaR at configurable confidence |
| `cvar_historical` | Expected Shortfall (CVaR) |
| `max_drawdown` | Peak-to-trough drawdown |
| `max_drawdown_duration` | Periods from trough to recovery (None if not recovered) |
| `factor_attribution` | OLS regression of portfolio returns on factor returns; reports alpha, betas, t-stats, p-values |
| `rolling_metrics` | Rolling Sharpe, volatility, max drawdown over configurable window |
| `performance_summary` | Full tear sheet including all above metrics; optional benchmark comparison with Beta, Alpha, Tracking Error, Information Ratio |

### 7.5 Backtesting Engine (`quantopt/backtest/`)

`WalkForwardBacktester` implements a production-grade backtester:
- Mark-to-market weight drift between rebalances (no look-ahead).
- Configurable rebalancing frequency (monthly, quarterly, annual).
- Expanding lookback window with configurable lookback period.
- `TransactionCostModel`: proportional spread + fixed fee + square-root market impact.
- Turnover constraint via proportional weight shrinkage toward prior position.
- Graceful fallback: if optimizer fails on a rebalance date, holds current drifted position.
- `run_comparison()`: runs multiple strategies on the same dataset/config for fair comparison.
- Outputs: `BacktestResult` dataclass with portfolio returns, gross returns, weight history, realized weights, turnover history, transaction costs, cumulative returns, dollar value, and full performance summary.

---

## 8. Technologies and Languages

### Language

| Item | Version |
|---|---|
| Python | 3.11+ |

### Core Scientific Stack

| Library | Role |
|---|---|
| NumPy | Numerical arrays, linear algebra |
| Pandas | Time-series data handling, DataFrames |
| SciPy | Optimization (SLSQP solver), statistics |
| Matplotlib / Seaborn | Visualization and figures |

### Machine Learning

| Library | Role | Version |
|---|---|---|
| scikit-learn | PCA, OAS shrinkage, HMM regime baseline, SVM classifier | 1.4+ |
| LightGBM | Gradient-boosted trees for alpha signal generation | 4.0+ |
| PyTorch | Deep neural network for rBergomi volatility calibration | 2.1+ |
| Stable-Baselines3 | PPO reinforcement learning algorithm | 2.2+ |
| Gymnasium | Portfolio environment interface for RL | — |

### Infrastructure

| Library | Role | Version |
|---|---|---|
| vectorbt | Walk-forward backtesting utilities | 0.26+ |
| MLflow | Experiment tracking, model registry | 2.9+ |
| uv | Package manager and virtual environment | 0.1+ |

### Code Quality

| Tool | Role |
|---|---|
| pytest | Unit and integration testing (9 test modules) |
| Black | Code formatting (line length 88) |
| mypy | Static type checking (Python 3.11 target) |

### Data Formats

| Format | Usage |
|---|---|
| Apache Parquet | All processed datasets (features, labels, signals, surfaces) |
| YAML | Pipeline configuration files for all phases |
| `.pt` (PyTorch) | Serialized deep vol net weights |
| `.pkl` (pickle) | Serialized SVM regime classifier and LightGBM model |
| `.zip` | Serialized PPO agent (Stable-Baselines3 format) |

---

## 9. Repository Structure

```
quantopt/                  # Installable Python package
├── returns/
│   ├── estimators.py      # MeanHistoricalReturn, CAPMReturn, BlackLittermanReturn
│   └── preprocessing.py
├── risk/
│   ├── covariance.py      # SampleCovariance, EWMCovariance, LedoitWolfCovariance, FactorModelCovariance
│   └── metrics.py         # VaR, CVaR, risk metrics
├── optimization/
│   ├── base.py            # BaseOptimizer ABC
│   ├── constraints.py     # ConstraintSet builder
│   ├── efficient_frontier.py  # EfficientFrontier (max Sharpe, min vol, efficient return/risk)
│   ├── cvar_optimizer.py  # CVaROptimizer (Rockafellar-Uryasev)
│   ├── risk_parity.py     # RiskParity (ERC, generalized budgets)
│   └── factory.py
├── analytics/
│   └── performance.py     # Full tear sheet: Sharpe, Sortino, Calmar, Omega, VaR, CVaR, attribution
├── backtest/
│   └── engine.py          # WalkForwardBacktester, BacktestResult, TransactionCostModel, BacktestConfig
├── plotting/
│   └── charts.py
└── utils/
    └── validation.py      # Input validation, PSD projection

src/                       # Pipeline phase scripts (not packaged)
├── data/                  # Data fetching scripts
├── features/              # Feature engineering
├── regime/                # HMM labeling + SVM training
├── alpha/                 # LightGBM alpha model training
├── volatility/            # GARCH + deep vol net training
├── rl/                    # PPO agent training and environment
└── backtest/              # Walk-forward report generation

tests/                     # pytest test suite (9 modules)
notebooks/                 # Jupyter notebooks for exploration and visualization
docs/figures/              # Generated figures (Phases 1–3, 15 figures)
docs/figures2/             # Generated figures (Phases 4–5 + tables, 16 images)
Presentation/              # Slide prompts and PPTX files
```

---

## 10. Test Suite

Nine pytest modules covering all main components:

| Module | Coverage |
|---|---|
| `test_estimators.py` | MeanHistoricalReturn, CAPMReturn, BlackLittermanReturn |
| `test_covariance.py` | SampleCovariance, EWMCovariance, LedoitWolfCovariance, FactorModelCovariance |
| `test_efficient_frontier.py` | Max Sharpe, Min Vol, Efficient Return, Efficient Risk, frontier trace |
| `test_cvar.py` | CVaROptimizer convergence, decomposition, VaR at confidence |
| `test_risk_parity.py` | ERC convergence, risk contribution check, concentration check |
| `test_metrics.py` | VaR, CVaR, individual risk metrics |
| `test_performance.py` | Sharpe, Sortino, Calmar, Omega, drawdown, rolling metrics, factor attribution |
| `test_preprocessing.py` | Feature validation and return preprocessing |
| `test_backtest.py` | WalkForwardBacktester end-to-end, transaction costs, turnover constraint |

---

## 11. Limitations

1. **Options data quality.** The deep vol net calibration requires complete daily IV surfaces. Free-tier Polygon.io data provides delayed, potentially incomplete chains for short-dated or deep OTM strikes. Production deployment requires OptionMetrics or CBOE DataShop data.
2. **Transaction cost model.** The backtest uses proportional L1-turnover costs. Real market impact is non-linear and depends on order size, venue, and intraday timing. Costs for large rebalances are likely understated.
3. **Narrow asset universe.** The five-ETF universe ensures data availability. Extension to individual equities, international markets, or alternatives requires recalibrating PCA dimensionality, HMM state count, and RL reward coefficients.
4. **HMM lookahead in labeling.** The HMM is trained on the full historical sample, so regime labels for early dates are influenced by future data. This constitutes mild lookahead bias in the labeling step, which is difficult to eliminate without online HMM estimation.
5. **RL sample efficiency.** PPO requires ~1 million environment steps to converge on the 5-asset universe. Scaling to larger universes or intraday rebalancing would require substantially more compute or more sample-efficient algorithms (SAC, model-based RL).
6. **Synthetic DNN training data.** The rBergomi calibration DNN is trained on synthetic Monte Carlo data using a Bergomi-Guyon approximation. If true market dynamics deviate from the rBergomi parameterization, calibrated H and η values may be biased (model risk).

---

## 12. Key References

| # | Citation |
|---|---|
| [1] | Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91. |
| [2] | Hamilton, J. D. (1989). A new approach to nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384. |
| [3] | Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187. |
| [4] | Michaud, R. O. (1989). The Markowitz optimization enigma. *Financial Analysts Journal*, 45(1), 31–42. |
| [5] | Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933–949. |
| [6] | Bayer, C., Friz, P. & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887–904. |
| [7] | Horvath, B., Jacquier, A. & Muguruza, C. (2021). Deep learning volatility. *Applied Mathematical Finance*, 28(6), 499–521. |
| [8] | Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. |
| [9] | Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273. |
| [10] | Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347. |
| [11] | Liang, Z. et al. (2018). Adversarial deep reinforcement learning in portfolio management. arXiv:1808.09940. |
| [12] | Nystrup, P. et al. (2020). Dynamic portfolio optimization across hidden market regimes. *Quantitative Finance*, 20(1), 83–95. |
