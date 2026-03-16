# RAMPA — Regime-Aware Multi-Agent Portfolio Allocator

**Version:** 1.0.0  
**Author:** Felipe Cardozo  
**Institution:** Emory University  
**Status:** Research / Portfolio Project  
**License:** MIT

| Component        | Technology                         | Version  |
|------------------|------------------------------------|----------|
| Language         | Python                             | 3.11+    |
| ML Framework     | scikit-learn                       | 1.4+     |
| Boosting         | LightGBM                           | 4.0+     |
| Deep Learning    | PyTorch                            | 2.1+     |
| RL Framework     | Stable-Baselines3 + Gymnasium      | 2.2+     |
| Backtesting      | vectorbt                           | 0.26+    |
| Experiment Track | MLflow                             | 2.9+     |
| Package Manager  | uv                                 | 0.1+     |
| Config Format    | YAML                               | —        |
| Data Format      | Apache Parquet                     | —        |

## Abstract

RAMPA is a regime-aware portfolio allocation framework that integrates regime detection, alpha generation, rough volatility calibration, and reinforcement learning into a single hierarchical pipeline. The system addresses the fragility of static mean–variance optimization by conditioning allocation decisions on hidden market regimes and high-dimensional predictive features. Methodologically, RAMPA combines hidden Markov models for regime labeling, gradient-boosted trees for cross-sectional alpha signals, a deep neural network calibrated to the rough Bergomi model for implied volatility surfaces, and a PPO agent trained in a custom portfolio environment. In a walk-forward backtest, RAMPA outperforms a 60/40 benchmark and a classical MVO portfolio on risk-adjusted metrics while maintaining strictly controlled drawdowns.

## Table of Contents

- [Abstract](#abstract)
- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Methods](#methods)
  - [8.1 Data and Features](#81-data-and-features)
  - [8.2 Dimensionality Reduction and Baseline Regression](#82-dimensionality-reduction-and-baseline-regression)
  - [8.3 Regime Classification](#83-regime-classification)
  - [8.4 Alpha Signal Generation](#84-alpha-signal-generation)
  - [8.5 Volatility Oracle](#85-volatility-oracle)
  - [8.6 Reinforcement Learning Execution Engine](#86-reinforcement-learning-execution-engine)
  - [8.7 Backtest Results](#87-backtest-results)
- [Model Validation](#model-validation)
- [Limitations](#limitations)
- [References](#references)
- [License and Citation](#license-and-citation)

## Project Overview

Static mean–variance optimization (MVO) assumes that asset return moments are stationary and can be estimated reliably from finite samples. In practice, asset return distributions exhibit time-varying volatility, structural breaks, and heavy tails, which cause covariance matrix estimates to be unstable and amplify estimation error in optimized portfolios. Moreover, single-period MVO is myopic: it ignores regime shifts, path-dependence, and higher-order features that are critical for robust portfolio construction in real markets.

RAMPA addresses these limitations by organizing the portfolio construction process into a hierarchical machine learning pipeline. Each phase solves a narrowly defined subproblem and exposes well-defined artefacts (features, labels, signals, parameters, and policies) to the subsequent phase. Regime detection, alpha generation, volatility modelling, and reinforcement learning are coordinated but decoupled, allowing each component to be selected, tuned, and validated according to its own statistical and economic assumptions. This modular design makes the system extensible while preserving a coherent end-to-end workflow.

The pipeline is structured into five phases. Phase 1 performs feature engineering and dimensionality reduction, transforming raw price, macro, and derivative data into a compressed representation that preserves predictive structure. Phase 2 learns hidden market regimes using a hidden Markov model and trains discriminative classifiers that can assign regime labels out-of-sample. Phase 3 estimates cross-sectional alpha signals using tree-based ensemble models, explicitly conditioning on the inferred regimes. Phase 4 constructs a volatility oracle by combining a GARCH baseline with a deep neural network that calibrates an rBergomi model to the implied volatility surface, producing time-varying rough volatility parameters. Phase 5 embeds these components into a Gymnasium-compatible portfolio environment and trains a PPO agent that outputs regime-aware portfolio weights, which are evaluated in a walk-forward backtest.

The core design principle is that each technique is applied to the subproblem for which it has the strongest theoretical justification. HMMs are used where latent state dynamics and persistence are central; gradient-boosted trees are used where non-linear interactions in high-dimensional feature spaces dominate; rough volatility models are used where empirical evidence supports fractional dynamics in volatility; and PPO is used where policy gradients under continuous actions and risk constraints are required. The result is a portfolio allocator that is both statistically grounded and operationally implementable.

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

## Repository Structure

```text
.
├── config/
│   ├── data/
│   ├── features/
│   ├── regimes/
│   ├── alpha/
│   ├── volatility/
│   └── rl/
├── data/
│   ├── raw/
│   └── processed/
│       ├── features.parquet
│       ├── regime_labels.parquet
│       ├── alpha_signals.parquet
│       ├── vol_features.parquet
│       └── iv_surfaces.parquet
├── models/
│   ├── svm_regime.pkl
│   ├── lgbm_alpha.pkl
│   ├── deep_vol_net.pt
│   └── ppo_agent.zip
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_regime_modeling.ipynb
│   ├── 04_alpha_modeling.ipynb
│   ├── 05_volatility_oracle.ipynb
│   ├── 06_rl_training.ipynb
│   ├── 07_backtest_analysis.ipynb
│   └── 08_visualizations.ipynb
├── reports/
│   ├── backtest_summary.csv
│   └── figures/
│       ├── fig_01_asset_prices.png
│       ├── fig_02_correlation_matrix.png
│       ├── fig_03_return_distributions.png
│       ├── fig_04_pca_variance.png
│       ├── fig_05_factor_loadings.png
│       ├── fig_06_fractional_diff.png
│       ├── fig_07_regime_timeline.png
│       ├── fig_08_regime_posterior.png
│       ├── fig_09_regime_transition_matrix.png
│       ├── fig_10_regime_return_stats.png
│       ├── fig_11_svm_confusion_matrix.png
│       ├── fig_12_feature_importance.png
│       ├── fig_13_rolling_ic.png
│       ├── fig_14_alpha_signal_heatmap.png
│       ├── fig_15_garch_conditional_vol.png
│       ├── fig_16_iv_surface.png
│       ├── fig_17_dnn_calibration.png
│       ├── fig_18_hurst_over_time.png
│       ├── fig_19_training_reward_curve.png
│       ├── fig_20_weight_allocation.png
│       ├── fig_21_weight_regime_heatmap.png
│       ├── fig_22_cumulative_returns.png
│       ├── fig_23_drawdown.png
│       ├── fig_24_rolling_sharpe.png
│       ├── fig_25_metrics_comparison.png
│       └── fig_26_monthly_returns_heatmap.png
├── src/
│   ├── data/
│   ├── features/
│   ├── regime/
│   ├── alpha/
│   ├── volatility/
│   ├── rl/
│   └── backtest/
├── tests/
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

| Directory    | Description                                          |
|--------------|------------------------------------------------------|
| `config/`    | YAML configuration files for all pipeline parameters |
| `data/`      | Raw fetched data and processed Parquet artefacts     |
| `src/`       | Source modules organized by pipeline phase           |
| `notebooks/` | Exploratory and visualization Jupyter notebooks      |
| `models/`    | Serialized trained model files                       |
| `tests/`     | Pytest test suite for all modules                    |
| `reports/`   | Backtest summary CSV and generated figures           |

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<username>/rampa.git
cd rampa

# 2. Install uv (if not already installed)
pip install uv

# 3. Create a virtual environment and install all dependencies
uv venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows (PowerShell)
uv pip install -e .

# 4. Verify the installation
python -c "import lightgbm, torch, stable_baselines3; print('Installation verified.')"

# 5. Copy the environment template and populate API keys
cp .env.example .env
# Edit .env and insert your FRED_API_KEY, POLYGON_API_KEY, NASDAQ_API_KEY
```

| Key                | Source              | Cost                |
|--------------------|---------------------|---------------------|
| `FRED_API_KEY`     | fred.stlouisfed.org | Free                |
| `POLYGON_API_KEY`  | polygon.io          | Free tier available |
| `NASDAQ_API_KEY`   | data.nasdaq.com     | Free                |

## Running the Pipeline

```bash
# Step 1 — Fetch raw data
python data/scripts/fetch_equity.py
python data/scripts/fetch_macro.py
python data/scripts/fetch_options.py

# Step 2 — Build feature matrix
python data/scripts/build_dataset.py

# Step 3 — Train regime classifier
python src/regime/train_regime.py

# Step 4 — Train alpha models
python src/alpha/train_alpha.py

# Step 5 — Train volatility oracle
python src/volatility/deep_vol_net.py

# Step 6 — Train RL agent
python src/rl/ppo_agent.py

# Step 7 — Run walk-forward backtest
python src/backtest/generate_report.py

# Step 8 — Generate all visualizations
jupyter nbconvert --to notebook --execute notebooks/08_visualizations.ipynb

# Optional — launch MLflow UI
mlflow ui  # opens at http://localhost:5000
```

Once step 2 has completed and the processed datasets exist in `data/processed/`, steps 3 through 6 can be run independently and in any order. Each training script loads its inputs from the Parquet artefacts rather than relying on in-memory state from previous steps, which simplifies experimentation and hyperparameter sweeps.

## Methods

### 8.1 Data and Features

The investable universe consists of five highly liquid U.S.-listed ETFs: SPY (U.S. large-cap equity), QQQ (U.S. growth/technology equity), IEF (intermediate Treasuries), GLD (gold), and SHV (cash-like Treasuries). From 2010 to 2024, daily prices are transformed into log returns, realized volatility measures, and cross-sectional spreads; these are augmented with macroeconomic indicators and option-implied features to form a unified feature matrix. The resulting dataset combines slow-moving macro structure with fast-moving market microstructure, enabling models to detect both regime-level shifts and short-horizon alpha. All engineered features are stored in `data/processed/features.parquet` for reproducible downstream use.

![Asset Price History](reports/figures/fig_01_asset_prices.png)  
*Figure 1. Normalized price indices (base = 100) for the five-asset universe. Gray bands denote the COVID-19 crash (Feb–Apr 2020) and the 2022 rate-hike stress period. The diversification benefit of including IEF and GLD is evident during the 2020 drawdown.*

![Return Correlation Matrix](reports/figures/fig_02_correlation_matrix.png)  
*Figure 2. Average pairwise return correlations over the full 2010–2024 sample. The low correlation of GLD with equities and the negative correlation of IEF with SPY during stress periods motivate their inclusion as diversifiers.*

![Return Distributions](reports/figures/fig_03_return_distributions.png)  
*Figure 3. Empirical return distributions for all five assets. Negative skewness and excess kurtosis (leptokurtosis) across all series confirm that Gaussian assumptions underlying classical MVO are violated.*

### 8.2 Dimensionality Reduction and Baseline Regression

The raw feature set spans more than one hundred dimensions, including overlapping transformations of macro series, realized measures, and implied volatilities. Principal component analysis (PCA) is applied to this standardized feature matrix to extract orthogonal latent factors, which act as eigen-portfolios summarizing co-movements across assets and macro drivers. Linear models such as Lasso, Ridge, and online stochastic gradient descent are trained on these factors as baselines for return forecasting, providing a transparent benchmark against which non-linear models are evaluated. The PCA transformation also mitigates multicollinearity and reduces the effective degrees of freedom in subsequent models.

![PCA Explained Variance](reports/figures/fig_04_pca_variance.png)  
*Figure 4. Individual and cumulative explained variance by PCA component. The red dashed line marks the 90% threshold. Fewer than 15 components typically explain 90% of variance in the macro + return feature space, achieving substantial compression from over 100 raw features.*

![Factor Loadings](reports/figures/fig_05_factor_loadings.png)  
*Figure 5. PCA factor loadings for the top five components against the fifteen highest-variance input features. Factor 1 loads heavily on yield-curve tenors, consistent with its interpretation as a level factor. Factor 2 loads on equity volatility features, consistent with a risk appetite factor.*

![Fractional Differentiation](reports/figures/fig_06_fractional_diff.png)  
*Figure 6. Comparison of the raw SPY price series, standard log returns (d = 1.0), and the fractionally differenced series at the minimum stationary order d*. Fractional differencing preserves long-range dependence that first-differencing discards while satisfying the stationarity requirement of downstream classifiers.*

### 8.3 Regime Classification

Hidden Markov models (HMMs) are employed to infer latent market regimes from macro and volatility-sensitive features, capturing persistence and transition dynamics that are not visible in raw returns alone. The HMM is first used generatively to label historical periods into four interpretable regimes: Trending Bull, Choppy, High-Vol Stress, and Crisis. These labels are then used to train discriminative classifiers, including a Naive Bayes baseline and a production RBF-SVM, which can predict regimes out-of-sample. This two-stage process combines the time-series structure of HMMs with the flexible decision boundaries of kernel methods.

![Regime Timeline](reports/figures/fig_07_regime_timeline.png)  
*Figure 7. SPY price (log scale) with HMM-decoded regime overlay. The model correctly identifies the COVID crash as a Crisis episode and the 2022 drawdown as a High-Volatility Stress episode. The predominance of the Trending Bull regime during 2013–2019 is consistent with the post-GFC bull market.*

![Regime Posteriors](reports/figures/fig_08_regime_posterior.png)  
*Figure 8. Posterior probabilities for each regime over time. Rapid transitions in the posterior during market dislocations confirm that the HMM is responsive to structural breaks rather than smoothing them away.*

![Transition Matrix](reports/figures/fig_09_regime_transition_matrix.png)  
*Figure 9. HMM transition probability matrix. The high diagonal values confirm that regimes are persistent — the system does not churn between states at daily frequency. The Crisis regime has the lowest self-transition probability, consistent with the episodic nature of market crises.*

![Return Stats by Regime](reports/figures/fig_10_regime_return_stats.png)  
*Figure 10. Return distribution and annualized volatility by regime. The Crisis regime exhibits the widest return distribution and highest volatility. The Trending Bull regime produces the most consistent positive returns.*

![SVM Confusion Matrices](reports/figures/fig_11_svm_confusion_matrix.png)  
*Figure 11. Out-of-sample confusion matrices for the Naive Bayes baseline and the RBF-SVM production classifier. The SVM achieves meaningfully higher accuracy across all four regime classes. The most common misclassification is between the Choppy and Trending Bull regimes, which are adjacent in feature space.*

### 8.4 Alpha Signal Generation

Cross-sectional alpha signals are generated using tree-based ensemble methods trained on the engineered features and regime labels. A sequence of increasingly expressive models—decision trees, random forests, and finally LightGBM—are compared, with LightGBM selected for its ability to model non-linear interactions and handle heterogeneous feature scales. The target is the next-day SPY return, and model outputs are converted into continuous long–short signals that serve as inputs to the RL agent and benchmark allocation rules. The resulting alpha signal is explicitly regime-conditional, exploiting the observation that trend-following and mean-reversion signals have different efficacy across regimes.

![Feature Importance](reports/figures/fig_12_feature_importance.png)  
*Figure 12. LightGBM feature importances by gain for the top 25 features. Volatility-based features and momentum features dominate, consistent with the established empirical asset pricing literature (Gu, Kelly & Xiu, 2020). The regime_id feature ranks in the top 10, confirming that regime conditioning adds predictive content beyond technical indicators alone.*

![Rolling IC](reports/figures/fig_13_rolling_ic.png)  
*Figure 13. Sixty-three-day rolling Information Coefficient (Spearman correlation between predicted signal and realized next-day return). Positive IC during trending markets and near-zero IC during choppy regimes is consistent with the signal being regime-conditional. The dotted line at IC = 0.02 marks the practical significance threshold.*

![Alpha Signal Heatmap](reports/figures/fig_14_alpha_signal_heatmap.png)  
*Figure 14. Monthly average ensemble alpha signal. A value above 0.5 indicates a net long bias for that month. Strong long signals during 2013–2014 and 2019–2020 Q1 correspond to documented bull-market periods. The signal correctly shifts defensive during March 2020.*

### 8.5 Volatility Oracle

To capture the empirically observed roughness of volatility, the volatility oracle combines a classical GARCH(1,1) baseline with a deep neural network calibrated to the rough Bergomi (rBergomi) model. Historical SPY option chains are transformed into implied volatility surfaces, which are then mapped to underlying rBergomi parameters \`(H, \eta, \xi_0)\` by a convolutional neural network trained on synthetic data. This approach leverages the structural realism of rBergomi while avoiding the computational expense of direct likelihood-based calibration. The resulting time series of rough volatility parameters provides a rich, forward-looking signal for both risk management and RL policy conditioning.

![GARCH Volatility](reports/figures/fig_15_garch_conditional_vol.png)  
*Figure 15. GARCH(1,1) conditional volatility versus 21-day realized volatility for SPY. The GARCH model tracks realized volatility closely during calm periods but underestimates the speed of volatility spikes during dislocations (volatility clustering).*

![IV Surface](reports/figures/fig_16_iv_surface.png)  
*Figure 16. SPY implied volatility surface for a representative date. The characteristic downward slope across moneyness (volatility skew) and upward term structure are clearly visible. The deep neural network calibration module learns to map surfaces of this form to the rBergomi parameters (H, eta, xi0) that reproduce them.*

![DNN Calibration](reports/figures/fig_17_dnn_calibration.png)  
*Figure 17. Scatter plots of true versus predicted rBergomi parameters on the synthetic test set. Points clustered tightly around the diagonal (y = x) confirm accurate calibration. The Hurst exponent H achieves the highest R-squared, as it has the strongest effect on the IV surface skew and is therefore most identifiable from the surface shape.*

![Hurst Over Time](reports/figures/fig_18_hurst_over_time.png)  
*Figure 18. Time series of the estimated Hurst exponent H_t. Values consistently below 0.5 confirm rough volatility (as established by Gatheral et al., 2018). Episodic spikes toward H = 0.3–0.4 during stress periods indicate temporarily less rough behaviour, consistent with volatility persistence increasing during crises.*

### 8.6 Reinforcement Learning Execution Engine

The execution engine is implemented as a custom Gymnasium environment, \`PortfolioEnv\`, which exposes continuous portfolio weights over the five-ETF universe as actions. At each step, the agent observes current features, regime indicators, and volatility oracle outputs, and chooses a new allocation subject to leverage and turnover constraints. Proximal Policy Optimization (PPO) is used as the policy-gradient algorithm due to its robustness to hyperparameter choices and its ability to handle continuous action spaces with clipped updates. The reward function is a Markovian step-level proxy that balances risk-adjusted returns against drawdown and turnover penalties, ensuring that long-horizon objectives are aligned with step-level learning signals.

![Training Reward Curve](reports/figures/fig_19_training_reward_curve.png)  
*Figure 19. PPO agent training reward over timesteps. The 50-episode rolling mean (solid line) monotonically improves and stabilises above the random agent baseline, confirming that the agent learns a non-trivial allocation policy. The reward plateau at approximately 700,000 timesteps suggests convergence.*

![Weight Allocation](reports/figures/fig_20_weight_allocation.png)  
*Figure 20. RAMPA portfolio weight allocation over the full backtest period. The agent shifts toward IEF and SHV (cash) during the 2020 COVID crash and the 2022 rate-hike period, demonstrating that it has learned to respond to regime signals by de-risking. The return to equity-heavy positioning during the 2023 recovery is consistent with rational risk-taking behaviour.*

![Weights by Regime](reports/figures/fig_21_weight_regime_heatmap.png)  
*Figure 21. Mean portfolio weights by market regime. The agent allocates the majority of capital to SPY and QQQ during the Trending Bull regime and rotates substantially into IEF and SHV during the Crisis regime. This regime-conditional allocation is the core behavioural contribution of the RL framework over static MVO.*

### 8.7 Backtest Results

The full RAMPA pipeline is evaluated using an expanding-window walk-forward backtest from 2015 onward, with all hyperparameters fixed prior to evaluation. RAMPA is compared against three benchmarks: a static 60/40 portfolio (SPY/IEF), an equal-weight portfolio across the five ETFs, and a classic MVO portfolio optimized on a rolling window. Performance is assessed using cumulative returns, drawdown profiles, rolling Sharpe ratios, and risk-adjusted summary statistics. The results demonstrate that RAMPA achieves higher terminal wealth and superior downside risk control relative to all benchmarks.

![Cumulative Returns](reports/figures/fig_22_cumulative_returns.png)  
*Figure 22. Cumulative portfolio value for RAMPA and three benchmarks over the walk-forward backtest period. RAMPA achieves a higher terminal value with a shallower drawdown profile during the two major stress episodes highlighted.*

![Drawdown](reports/figures/fig_23_drawdown.png)  
*Figure 23. Drawdown profiles for all four strategies. The 10% maximum drawdown constraint imposed on the RL agent is visible as a hard floor on the RAMPA drawdown curve. The 60/40 benchmark breaches 20% drawdown during the 2022 rate-hike stress.*

![Rolling Sharpe](reports/figures/fig_24_rolling_sharpe.png)  
*Figure 24. Sixty-three-day rolling Sharpe ratio for RAMPA versus the 60/40 benchmark. RAMPA outperforms most consistently during regime transitions, where its state-conditional policy provides an informational advantage over the static 60/40 allocation.*

![Metrics Comparison](reports/figures/fig_25_metrics_comparison.png)  
*Figure 25. Grouped bar chart of risk-adjusted performance metrics. RAMPA leads on Sharpe Ratio, Sortino Ratio, and Calmar Ratio. The improvement in Sortino Ratio relative to Sharpe Ratio is particularly notable, indicating that RAMPA's excess return comes predominantly from upside capture rather than increased downside risk.*

![Monthly Returns Heatmap](reports/figures/fig_26_monthly_returns_heatmap.png)  
*Figure 26. Monthly return heatmaps for RAMPA (top) and the 60/40 benchmark (bottom). RAMPA produces fewer extreme negative months and a more consistent positive return profile across calendar years.*

## Model Validation

Regime and alpha models are validated using purged cross-validation with an embargo period, as proposed by Lopez de Prado. In this scheme, observations that are temporally adjacent to the test set are removed from the training folds, and an additional embargo window is applied around test periods to prevent information leakage via overlapping labels or features. This approach is superior to standard time-series splits because it explicitly accounts for label overlap and serial correlation, which would otherwise inflate out-of-sample performance estimates.

The walk-forward backtest uses an expanding window design: models are trained on data up to time \(t\), then evaluated on the next out-of-sample segment, after which the training window is extended to include this segment, and the process is repeated. This procedure mimics the information flow available to a real-world practitioner and avoids in-sample evaluation bias. All hyperparameters are fixed prior to the final backtest, and no re-tuning is performed using test-period information.

For the reinforcement learning component, the reward cannot be the terminal Sharpe ratio because Sharpe is non-Markovian: it depends on the entire path of returns and their variance, not solely on the current state and action. Instead, a Markovian step-level proxy is used that combines risk-adjusted instantaneous return, drawdown penalties, and turnover costs. This proxy is designed such that policies which maximize the cumulative discounted reward tend to exhibit higher episode-level Sharpe ratios, thereby aligning the RL objective with the traditional portfolio evaluation metric without violating the Markov property required by PPO.

## Limitations

- **Options Data Availability.** The DNN volatility calibration module requires a complete, daily implied volatility surface. The free-tier Polygon.io data provides delayed chains which may be incomplete for short-dated or deep out-of-the-money strikes. Production deployment should use OptionMetrics or CBOE DataShop data.

- **Transaction Cost Modeling.** The backtest uses a simplified proportional transaction cost model (kappa * L1 turnover). In practice, market impact is non-linear and depends on order size, venue, and intraday timing. The current model likely understates costs for large allocations.

- **Asset Universe Scope.** The five-ETF universe is intentionally narrow to ensure data availability and clean backtesting. Extension to individual equities, international markets, or alternative asset classes requires recalibration of the PCA dimensionality, HMM state count, and RL reward coefficients.

- **Regime Label Stationarity.** The HMM is trained on the full historical sample, meaning regime labels for early dates are influenced by data that would not have been available at that time. This constitutes a mild form of lookahead bias in the labeling step that is difficult to eliminate without online HMM estimation.

- **RL Sample Efficiency.** PPO requires approximately one million environment steps to converge on the five-asset universe. Scaling to larger universes or higher rebalancing frequencies (intraday) would require substantially more compute and potentially more sample-efficient algorithms such as SAC or model-based RL.

- **Synthetic Training Data for DNN.** The rBergomi calibration DNN is trained on synthetic Monte Carlo data using a simplified Bergomi-Guyon approximation. This approximation introduces model risk: if the true market dynamics deviate from the rBergomi parameterization, calibrated H and eta values may be biased.

## References

[1] H. Markowitz, "Portfolio selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77–91, 1952.

[2] J. D. Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," *Econometrica*, vol. 57, no. 2, pp. 357–384, 1989.

[3] A. Ang and G. Bekaert, "International asset allocation with regime shifts," *The Review of Financial Studies*, vol. 15, no. 4, pp. 1137–1187, 2002.

[4] R. O. Michaud, "The Markowitz optimization enigma: Is optimized optimal?" *Financial Analysts Journal*, vol. 45, no. 1, pp. 31–42, 1989.

[5] J. Gatheral, T. Jaisson, and M. Rosenbaum, "Volatility is rough," *Quantitative Finance*, vol. 18, no. 6, pp. 933–949, 2018.

[6] C. Bayer, P. Friz, and J. Gatheral, "Pricing under rough volatility," *Quantitative Finance*, vol. 16, no. 6, pp. 887–904, 2016.

[7] B. Horvath, A. Jacquier, and C. Muguruza, "Deep learning volatility," *Applied Mathematical Finance*, vol. 28, no. 6, pp. 499–521, 2021. arXiv:1901.09647.

[8] M. Lopez de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ: Wiley, 2018.

[9] S. Gu, B. Kelly, and D. Xiu, "Empirical asset pricing via machine learning," *The Review of Financial Studies*, vol. 33, no. 5, pp. 2223–2273, 2020.

[10] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

[11] Z. Liang, H. Chen, J. Zhu, K. Jiang, and Y. Li, "Adversarial deep reinforcement learning in portfolio management," arXiv:1808.09940, 2018.

[12] P. Nystrup, B. W. Hansen, H. O. Larsen, H. Madsen, and E. Lindstrom, "Dynamic portfolio optimization across hidden market regimes," *Quantitative Finance*, vol. 20, no. 1, pp. 83–95, 2020.

## License and Citation

This project is released under the MIT License. See `LICENSE` for details.

If you reference this project in academic or professional work, please cite:

```text
Cardozo, F. (2024). RAMPA: Regime-Aware Multi-Agent Portfolio Allocator.
Emory University. Available at: https://github.com/<username>/rampa
```

