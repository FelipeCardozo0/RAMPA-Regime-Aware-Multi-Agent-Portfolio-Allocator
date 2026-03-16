<p align="center">
  <img src="https://img.shields.io/badge/RAMPA-v1.0.0-0d1117?style=for-the-badge&labelColor=0d1117&color=58a6ff" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="pytorch"/>
  <img src="https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=for-the-badge" alt="lightgbm"/>
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-blue?style=for-the-badge" alt="sb3"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="license"/>
</p>

<h1 align="center">RAMPA</h1>
<h4 align="center">Regime-Aware Multi-Agent Portfolio Allocator</h4>

<p align="center">
  <em>Hierarchical ML pipeline — from regime detection to reinforcement learning execution.</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#installation">Installation</a> •
  <a href="#methods">Methods</a> •
  <a href="#backtest-results">Results</a> •
  <a href="#references">References</a>
</p>

---

## Overview

**RAMPA** is a regime-aware portfolio allocation framework that integrates regime detection, alpha generation, rough volatility calibration, and reinforcement learning into a single hierarchical pipeline. It addresses the fragility of static mean–variance optimization by conditioning allocation decisions on hidden market regimes and high-dimensional predictive features.

| Phase | Method | Output |
|:------|:-------|:-------|
| **Feature Engineering** | PCA eigen-portfolios, fractional differencing, Lasso/Ridge baselines | Compressed feature matrix |
| **Regime Classification** | HMM labeling → Naive Bayes baseline → RBF-SVM | 4-state regime labels |
| **Alpha Generation** | Decision Tree → Random Forest → LightGBM ensemble | Cross-sectional alpha signals |
| **Volatility Oracle** | GARCH(1,1) + Deep Neural Net calibrated to rBergomi model | Time-varying $(H, \eta, \xi_0)$ |
| **RL Execution** | PPO agent in custom Gymnasium `PortfolioEnv` | Regime-aware portfolio weights |

The core design principle is that each technique is applied to the subproblem for which it has the strongest theoretical justification: HMMs where latent state dynamics are central, gradient-boosted trees where non-linear interactions dominate, rough volatility models where fractional dynamics are empirically supported, and PPO where policy gradients under continuous actions and risk constraints are required.

| Component | Technology | Version |
|:----------|:-----------|:--------|
| ML Framework | scikit-learn | 1.4+ |
| Boosting | LightGBM | 4.0+ |
| Deep Learning | PyTorch | 2.1+ |
| RL Framework | Stable-Baselines3 + Gymnasium | 2.2+ |
| Backtesting | vectorbt | 0.26+ |
| Experiment Tracking | MLflow | 2.9+ |
| Package Manager | uv | 0.1+ |

---

## Pipeline

```text
RAW DATA
    │
    ▼
Phase 1: Feature Engineering
    PCA (Eigen-Portfolios) → Lasso/Ridge Baseline → Online SGD
    │
    ▼
Phase 2: Regime Classification
    HMM Labeling → Naive Bayes Baseline → RBF-SVM
    │
    ▼
Phase 3: Alpha Generation
    Decision Tree → Random Forest → LightGBM → Ensemble Signal
    │
    ▼
Phase 4: Volatility Oracle
    GARCH(1,1) Baseline → IV Surface → Deep Vol Net (rBergomi Calibration)
    │
    ▼
Phase 5: RL Execution Engine
    PortfolioEnv (Gymnasium) → PPO Agent → Portfolio Weights
    │
    ▼
Walk-Forward Backtest → Performance Report
```

---

## Data and Features

The investable universe consists of five highly liquid U.S.-listed ETFs: **SPY** (large-cap equity), **QQQ** (growth/technology), **IEF** (intermediate Treasuries), **GLD** (gold), and **SHV** (cash-like Treasuries). From 2010 to 2024, daily prices are transformed into log returns, realized volatility measures, and cross-sectional spreads, augmented with macroeconomic indicators and option-implied features.

<p align="center">
  <img src="reports/figures/fig_01_asset_prices.png" width="100%" alt="Asset price history"/>
</p>

<sub>Figure 1. Normalized price indices (base = 100) for the five-asset universe. Gray bands denote the COVID-19 crash and 2022 rate-hike stress period. The diversification benefit of IEF and GLD is evident during the 2020 drawdown.</sub>

<p align="center">
  <img src="reports/figures/fig_02_correlation_matrix.png" width="100%" alt="Correlation matrix"/>
</p>

<sub>Figure 2. Average pairwise return correlations over the full sample. Low GLD–equity correlation and negative IEF–SPY correlation during stress motivate their inclusion as diversifiers.</sub>

<p align="center">
  <img src="reports/figures/fig_03_return_distributions.png" width="100%" alt="Return distributions"/>
</p>

<sub>Figure 3. Empirical return distributions. Negative skewness and excess kurtosis across all series confirm that Gaussian assumptions underlying classical MVO are violated.</sub>

---

## Methods

### Dimensionality Reduction and Baseline Regression

The raw feature set spans 100+ dimensions. PCA extracts orthogonal latent factors (eigen-portfolios) that summarize co-movements across assets and macro drivers. Linear models (Lasso, Ridge, online SGD) trained on these factors provide transparent benchmarks against which non-linear models are evaluated.

<p align="center">
  <img src="reports/figures/fig_04_pca_variance.png" width="100%" alt="PCA explained variance"/>
</p>

<sub>Figure 4. Individual and cumulative explained variance by PCA component. Fewer than 15 components explain 90% of variance, achieving substantial compression from 100+ raw features.</sub>

<p align="center">
  <img src="reports/figures/fig_05_factor_loadings.png" width="100%" alt="Factor loadings"/>
</p>

<sub>Figure 5. PCA factor loadings for the top five components. Factor 1 loads on yield-curve tenors (level factor); Factor 2 loads on equity volatility features (risk appetite factor).</sub>

<p align="center">
  <img src="reports/figures/fig_06_fractional_diff.png" width="100%" alt="Fractional differentiation"/>
</p>

<sub>Figure 6. Raw SPY price, standard log returns (d = 1.0), and fractionally differenced series at minimum stationary order d*. Fractional differencing preserves long-range dependence while satisfying stationarity requirements.</sub>

---

### Regime Classification

Hidden Markov models infer latent market regimes from macro and volatility-sensitive features, capturing persistence and transition dynamics invisible in raw returns. The HMM labels historical periods into four interpretable regimes: **Trending Bull**, **Choppy**, **High-Vol Stress**, and **Crisis**. Discriminative classifiers (Naive Bayes baseline, production RBF-SVM) then predict regimes out-of-sample.

<p align="center">
  <img src="reports/figures/fig_07_regime_timeline.png" width="100%" alt="Regime timeline"/>
</p>

<sub>Figure 7. SPY price (log scale) with HMM-decoded regime overlay. The model correctly identifies COVID as a Crisis episode and 2022 as High-Volatility Stress.</sub>

<p align="center">
  <img src="reports/figures/fig_08_regime_posterior.png" width="100%" alt="Regime posteriors"/>
</p>

<sub>Figure 8. Posterior probabilities for each regime over time. Rapid transitions during dislocations confirm the HMM is responsive to structural breaks.</sub>

<p align="center">
  <img src="reports/figures/fig_09_regime_transition_matrix.png" width="100%" alt="Transition matrix"/>
</p>

<sub>Figure 9. HMM transition probability matrix. High diagonal values confirm regime persistence. The Crisis regime has the lowest self-transition probability, consistent with the episodic nature of crises.</sub>

<p align="center">
  <img src="reports/figures/fig_10_regime_return_stats.png" width="100%" alt="Return stats by regime"/>
</p>

<sub>Figure 10. Return distributions and annualized volatility by regime. Crisis exhibits the widest distribution and highest volatility; Trending Bull produces the most consistent positive returns.</sub>

<p align="center">
  <img src="reports/figures/fig_11_svm_confusion_matrix.png" width="100%" alt="SVM confusion matrices"/>
</p>

<sub>Figure 11. Out-of-sample confusion matrices for Naive Bayes and RBF-SVM. The SVM achieves higher accuracy across all four classes. Most common misclassification: Choppy ↔ Trending Bull.</sub>

---

### Alpha Signal Generation

Cross-sectional alpha signals are generated using tree-based ensembles trained on engineered features and regime labels. LightGBM is selected for its ability to model non-linear interactions and handle heterogeneous feature scales. The target is next-day SPY return; model outputs become continuous long–short signals that are explicitly **regime-conditional**.

<p align="center">
  <img src="reports/figures/fig_12_feature_importance.png" width="100%" alt="Feature importance"/>
</p>

<sub>Figure 12. LightGBM feature importances (gain) for the top 25 features. Volatility and momentum features dominate (Gu, Kelly & Xiu, 2020). The regime_id feature ranks top 10, confirming regime conditioning adds predictive content.</sub>

<p align="center">
  <img src="reports/figures/fig_13_rolling_ic.png" width="100%" alt="Rolling IC"/>
</p>

<sub>Figure 13. 63-day rolling Information Coefficient (Spearman rank correlation). Positive IC during trending markets, near-zero during choppy regimes — consistent with regime-conditional signal efficacy.</sub>

<p align="center">
  <img src="reports/figures/fig_14_alpha_signal_heatmap.png" width="100%" alt="Alpha signal heatmap"/>
</p>

<sub>Figure 14. Monthly average ensemble alpha signal. Values above 0.5 indicate net long bias. Strong long signals during 2013–2014 and 2019–2020 Q1; correct defensive shift during March 2020.</sub>

---

### Volatility Oracle

The volatility oracle combines a GARCH(1,1) baseline with a deep neural network calibrated to the rough Bergomi (rBergomi) model. Historical SPY option chains are transformed into implied volatility surfaces, which a CNN maps to underlying rBergomi parameters $(H, \eta, \xi_0)$. This produces a time-varying rough volatility signal for both risk management and RL policy conditioning.

<p align="center">
  <img src="reports/figures/fig_15_garch_conditional_vol.png" width="100%" alt="GARCH volatility"/>
</p>

<sub>Figure 15. GARCH(1,1) conditional volatility vs. 21-day realized volatility. GARCH tracks calm periods well but underestimates the speed of volatility spikes during dislocations.</sub>

<p align="center">
  <img src="reports/figures/fig_16_iv_surface.png" width="100%" alt="IV surface"/>
</p>

<sub>Figure 16. SPY implied volatility surface. The characteristic downward slope (skew) across moneyness and upward term structure are clearly visible.</sub>

<p align="center">
  <img src="reports/figures/fig_17_dnn_calibration.png" width="100%" alt="DNN calibration"/>
</p>

<sub>Figure 17. True vs. predicted rBergomi parameters on the synthetic test set. Tight clustering around the diagonal confirms accurate calibration. Hurst exponent H achieves the highest R², as it most strongly affects IV surface skew.</sub>

<p align="center">
  <img src="reports/figures/fig_18_hurst_over_time.png" width="100%" alt="Hurst exponent"/>
</p>

<sub>Figure 18. Time series of estimated Hurst exponent H. Values consistently below 0.5 confirm rough volatility (Gatheral et al., 2018). Spikes toward H = 0.3–0.4 during stress indicate temporarily less rough behaviour.</sub>

---

### Reinforcement Learning Execution Engine

The execution engine is a custom Gymnasium environment (`PortfolioEnv`) exposing continuous portfolio weights over the five-ETF universe. The agent observes features, regime indicators, and volatility oracle outputs, then chooses allocations subject to leverage and turnover constraints. PPO is used for its robustness and ability to handle continuous actions with clipped updates. The reward function balances risk-adjusted returns against drawdown and turnover penalties.

<p align="center">
  <img src="reports/figures/fig_19_training_reward_curve.png" width="100%" alt="Training reward"/>
</p>

<sub>Figure 19. PPO training reward over timesteps. Rolling mean improves monotonically and stabilises above the random baseline. Reward plateau at ~700K steps suggests convergence.</sub>

<p align="center">
  <img src="reports/figures/fig_20_weight_allocation.png" width="100%" alt="Weight allocation"/>
</p>

<sub>Figure 20. Portfolio weight allocation over the backtest. The agent shifts toward IEF and SHV during the 2020 crash and 2022 rate-hike period, then returns to equity-heavy positioning during 2023 recovery.</sub>

<p align="center">
  <img src="reports/figures/fig_21_weight_regime_heatmap.png" width="100%" alt="Weights by regime"/>
</p>

<sub>Figure 21. Mean portfolio weights by regime. Majority SPY/QQQ during Trending Bull; substantial rotation into IEF/SHV during Crisis. This regime-conditional allocation is the core behavioural contribution over static MVO.</sub>

---

## Backtest Results

Expanding-window walk-forward backtest from 2015 onward, all hyperparameters fixed prior to evaluation. RAMPA is compared against a static 60/40 portfolio (SPY/IEF), an equal-weight portfolio, and a rolling-window MVO portfolio.

<p align="center">
  <img src="reports/figures/fig_22_cumulative_returns.png" width="100%" alt="Cumulative returns"/>
</p>

<sub>Figure 22. Cumulative portfolio value for RAMPA and three benchmarks. RAMPA achieves higher terminal value with shallower drawdowns during both major stress episodes.</sub>

<p align="center">
  <img src="reports/figures/fig_23_drawdown.png" width="100%" alt="Drawdown"/>
</p>

<sub>Figure 23. Drawdown profiles. The 10% maximum drawdown constraint on the RL agent is visible as a hard floor on RAMPA. The 60/40 benchmark breaches 20% drawdown during 2022.</sub>

<p align="center">
  <img src="reports/figures/fig_24_rolling_sharpe.png" width="100%" alt="Rolling Sharpe"/>
</p>

<sub>Figure 24. 63-day rolling Sharpe ratio. RAMPA outperforms most consistently during regime transitions, where its state-conditional policy provides an informational advantage.</sub>

<p align="center">
  <img src="reports/figures/fig_25_metrics_comparison.png" width="100%" alt="Metrics comparison"/>
</p>

<sub>Figure 25. Risk-adjusted performance metrics. RAMPA leads on Sharpe, Sortino, and Calmar ratios. The Sortino improvement over Sharpe indicates excess return comes from upside capture, not increased downside risk.</sub>

<p align="center">
  <img src="reports/figures/fig_26_monthly_returns_heatmap.png" width="100%" alt="Monthly returns"/>
</p>

<sub>Figure 26. Monthly return heatmaps for RAMPA (top) and 60/40 (bottom). RAMPA produces fewer extreme negative months and a more consistent positive return profile across calendar years.</sub>

---

## Model Validation

Regime and alpha models are validated using **purged cross-validation with embargo** (Lopez de Prado, 2018). Observations temporally adjacent to the test set are removed from training folds, and an embargo window prevents information leakage via overlapping labels or serial correlation — superior to standard time-series splits.

The walk-forward backtest uses an **expanding window**: train on data up to time $t$, evaluate on the next out-of-sample segment, extend, repeat. All hyperparameters are fixed prior to the final backtest with no re-tuning on test-period information.

For RL, the reward is a **Markovian step-level proxy** (not terminal Sharpe, which is non-Markovian) combining risk-adjusted instantaneous return, drawdown penalties, and turnover costs. This aligns RL learning signals with traditional portfolio metrics without violating the Markov property required by PPO.

---

## Repository Structure

```text
.
├── config/                          # YAML configs for all pipeline parameters
│   ├── data/ features/ regimes/ alpha/ volatility/ rl/
├── data/
│   ├── raw/                         # Fetched market data
│   └── processed/                   # Parquet artefacts (features, labels, signals)
├── src/                             # Source modules by pipeline phase
│   ├── data/ features/ regime/ alpha/ volatility/ rl/ backtest/
├── notebooks/                       # 8 Jupyter notebooks (exploration → visualization)
├── models/                          # Serialized models (.pkl, .pt, .zip)
├── tests/                           # Pytest suite
├── reports/
│   ├── backtest_summary.csv
│   └── figures/                     # 26 auto-generated figures
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Installation

```bash
# Clone and install
git clone https://github.com/FelipeCardozo0/RAMPA-Regime-Aware-Multi-Agent-Portfolio-Allocator.git
cd RAMPA-Regime-Aware-Multi-Agent-Portfolio-Allocator

# Install uv and create environment
pip install uv
uv venv
source .venv/bin/activate          # Linux / macOS
uv pip install -e .

# Verify
python -c "import lightgbm, torch, stable_baselines3; print('Installation verified.')"

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

| Key | Source | Cost |
|:----|:-------|:-----|
| `FRED_API_KEY` | fred.stlouisfed.org | Free |
| `POLYGON_API_KEY` | polygon.io | Free tier |
| `NASDAQ_API_KEY` | data.nasdaq.com | Free |

---

## Running the Pipeline

```bash
# Data
python data/scripts/fetch_equity.py
python data/scripts/fetch_macro.py
python data/scripts/fetch_options.py
python data/scripts/build_dataset.py

# Training (steps 3–6 can run independently after step 2)
python src/regime/train_regime.py
python src/alpha/train_alpha.py
python src/volatility/deep_vol_net.py
python src/rl/ppo_agent.py

# Evaluation
python src/backtest/generate_report.py
jupyter nbconvert --to notebook --execute notebooks/08_visualizations.ipynb

# Optional — MLflow UI
mlflow ui    # http://localhost:5000
```

> Once `data/processed/` is populated, training steps load from Parquet artefacts rather than in-memory state — run them in any order for experimentation and hyperparameter sweeps.

---

## Limitations

<details>
<summary><strong>Options Data Availability</strong></summary>

The DNN volatility calibration requires a complete daily IV surface. Free-tier Polygon.io provides delayed chains that may be incomplete for short-dated or deep OTM strikes. Production deployment should use OptionMetrics or CBOE DataShop.
</details>

<details>
<summary><strong>Transaction Cost Modeling</strong></summary>

The backtest uses proportional costs ($\kappa \times L_1$ turnover). Market impact is non-linear in practice and depends on order size, venue, and timing. Current model likely understates costs for large allocations.
</details>

<details>
<summary><strong>Asset Universe Scope</strong></summary>

The five-ETF universe is intentionally narrow for clean backtesting. Extension to individual equities, international markets, or alternatives requires recalibration of PCA dimensionality, HMM state count, and RL reward coefficients.
</details>

<details>
<summary><strong>Regime Label Stationarity</strong></summary>

The HMM is trained on the full historical sample, so early-period labels are influenced by future data — a mild form of lookahead bias that is difficult to eliminate without online HMM estimation.
</details>

<details>
<summary><strong>RL Sample Efficiency</strong></summary>

PPO requires ~1M environment steps for the five-asset universe. Larger universes or higher rebalancing frequencies would need substantially more compute and potentially SAC or model-based RL.
</details>

<details>
<summary><strong>Synthetic DNN Training Data</strong></summary>

The rBergomi calibration DNN is trained on synthetic Monte Carlo data using a Bergomi-Guyon approximation. If true market dynamics deviate from the rBergomi parameterization, calibrated $H$ and $\eta$ values may be biased.
</details>

---

## Roadmap

- [ ] **Hierarchical Risk Parity** integration as alternative to MVO benchmark
- [ ] **Online HMM estimation** to eliminate regime labeling lookahead bias
- [ ] **SAC / model-based RL** for improved sample efficiency at scale
- [ ] **Individual equity universe** with sector-level regime conditioning
- [ ] **Almgren-Chriss market impact** model replacing proportional costs
- [ ] **Live paper-trading** integration via Interactive Brokers API

---

## References

<details>
<summary>Expand full reference list</summary>

- Markowitz, H. M. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2), 357–384.
- Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.
- Michaud, R. O. (1989). The Markowitz optimization enigma. *Financial Analysts Journal*, 45(1), 31–42.
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933–949.
- Bayer, C., Friz, P. & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887–904.
- Horvath, B., Jacquier, A. & Muguruza, C. (2021). Deep learning volatility. *Applied Mathematical Finance*, 28(6), 499–521.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Liang, Z. et al. (2018). Adversarial deep reinforcement learning in portfolio management. arXiv:1808.09940.
- Nystrup, P. et al. (2020). Dynamic portfolio optimization across hidden market regimes. *Quantitative Finance*, 20(1), 83–95.

</details>

---

## License and Citation

This project is released under the **MIT License**.

```text
Cardozo, F. (2024). RAMPA: Regime-Aware Multi-Agent Portfolio Allocator.
Emory University. https://github.com/FelipeCardozo0/RAMPA-Regime-Aware-Multi-Agent-Portfolio-Allocator
```

---

<p align="center">
  <sub>Built by <a href="https://github.com/FelipeCardozo0">Felipe Cardozo</a> · Emory University</sub>
</p>
