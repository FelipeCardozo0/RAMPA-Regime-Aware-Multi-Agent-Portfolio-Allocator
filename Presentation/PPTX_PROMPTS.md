# RAMPA — PowerPoint Slide Prompts: Phase 4 & Phase 5
# Formal Graduate Audience · No Emojis · Dark Navy Theme

Each prompt below is self-contained and can be given independently to Claude
to produce one professional 16:9 PowerPoint slide using pptxgenjs or python-pptx.

Global design constants for all slides:
  Background:   #0D1B2A  (navy deep)
  Body text:    #E8EAF0  (off-white)
  Frame title:  #58A6FF  (bright blue)
  Accent gold:  #F0A500
  Gold light:   #FFD166
  Mint green:   #06D6A0  (positive metrics, tickers)
  Crimson:      #EF476F  (warnings, alerts)
  Slate gray:   #8899BB  (secondary text, labels)
  Electric blue:#4A90D9  (rules, borders)
  Table header: #132338  (navy mid)
  Block fill:   #1A2D47  (block backgrounds)
  Code bg:      #0F1F33  (dark block)
  Lime:         #90EE90  (Phase 3 accent)
  Title font:   Calibri Bold
  Body font:    Calibri / Calibri Light
  Code font:    Consolas
  Slide size:   13.33" × 7.5" (widescreen 16:9)

==============================================================================
PROMPT 1 OF 2 — PHASE 4: VOLATILITY ORACLE
==============================================================================

FIGURES TO EMBED (from docs/figures/ — use in the order listed):
  1. docs/figures/figure_04_covariance_comparison.png
     Use in Column 1 as a visual companion to the GARCH vs. rBergomi capability
     comparison table — shows empirical covariance structure across regimes.
  2. docs/figures/figure_05_covariance_diagnostics.png
     Use in Column 2 below the DNN architecture flow — illustrates the
     condition-number improvement that motivates replacing GARCH σ²_t with the
     richer (H, η, ξ₀) triplet.
  3. docs/figures/figure_15_sensitivity.png
     Use in Column 3 below the Hurst-exponent timeline table — shows parameter
     sensitivity (η, ξ₀) across simulated IV surfaces, reinforcing DNN accuracy.

Create a full-content academic PowerPoint slide titled "Phase 4 — Volatility
Oracle" with subtitle "GARCH(1,1) baseline · Rough Bergomi (rBergomi) model ·
Deep neural network calibration · Time-varying Hurst exponent."

SLIDE TITLE BAR
  Title: "Phase 4 — Volatility Oracle" — 28pt Bold #58A6FF
  Subtitle: "GARCH(1,1) baseline  ·  Rough Bergomi model  ·  DNN calibration of (H, η, ξ₀)"
  — 14pt #F0A500

LAYOUT: Three-column

COLUMN 1 (width ≈ 33%):

  GARCH(1,1) block (fill #1A2D47, border #8899BB 1pt)
    Header: "GARCH(1,1) — Volatility Baseline" — 12pt Bold #E8EAF0
    Typeset equation (11pt):
      σ²_t = ω + α r²_{t-1} + β σ²_{t-1},   α + β < 1
    Strengths bullet (9pt #06D6A0):
      + Parsimonious (3 params), closed-form, interpretable
    Limitations bullets (9pt #EF476F):
      − Underestimates vol-of-vol during dislocations
      − Cannot reproduce IV skew or term structure shape
      − Assumes H = 0.5 (Brownian motion) — empirically rejected

  rBergomi Model block (fill #1A2D47, border #4A90D9 1pt)
    Header: "Rough Bergomi (rBergomi) Model" — 12pt Bold #58A6FF
    Typeset equations (11pt #E8EAF0):
      dS_t = S_t √V_t  dW^S_t
      V_t = ξ_0 · exp( η W̃^H_t − (1/2) η² t^{2H} )
    Caption: "W̃^H = fractional Brownian motion with Hurst exponent H < 1/2"
    Parameters box (9pt #E8EAF0):
      H ∈ (0, 0.5): roughness / path irregularity of log-volatility
      η > 0:        vol-of-vol (scaling of stochastic variance)
      ξ₀ > 0:       initial instantaneous variance (forward variance)
    Ref: 9pt #8899BB "Bayer, Friz & Gatheral (2016); Gatheral et al. (2018)"

  GARCH vs. rBergomi Comparison Table
    Header: "Model Capability Comparison" — 11pt Bold #E8EAF0
    5-row × 3-col: "Capability" | "GARCH(1,1)" | "rBergomi"
    Header fill #132338, 9pt Bold #8899BB
    Data alt rows, 9pt; YES cells = fill #1E3A5F text #06D6A0 Bold;
    NO/Poor cells = fill #1A2D47 text #EF476F:
      IV skew reproduced     | No    | Yes
      IV term structure      | Part. | Yes
      Short-expiry accuracy  | Poor  | High
      Regime signal richness | σ²_t  | (H, η, ξ₀)
      Inference speed        | <1ms  | <1ms (DNN)

  Insert figure_04_covariance_comparison.png below this table.
  Caption (8pt #8899BB): "Empirical covariance structure — motivation for
  regime-conditioned volatility over a single GARCH parameter."

COLUMN 2 (width ≈ 34%):

  DNN Architecture vertical flow (fill #0F1F33, border #4A90D9 0.5pt)
    Header: "CNN Architecture: IV Surface → rBergomi Parameters" — 11pt Bold #4A90D9
    Seven stacked boxes connected by arrows (▼):
      Input box fill #1A2D47 border #4A90D9:
        "Input: IV Surface Σ_IV(K, τ) — 2D tensor (strikes × maturities)"
        9pt #58A6FF
      Conv2D(32, 3×3) → BatchNorm → ReLU  — fill #1A2D47 border #06D6A0 text #06D6A0 9pt
      Conv2D(64, 3×3) → BatchNorm → ReLU  — fill #1A2D47 border #06D6A0 text #06D6A0 9pt
      MaxPool(2×2)                          — fill #1A2D47 border #06D6A0 text #06D6A0 9pt
      Flatten → FC(256) → Dropout(0.2) → ReLU — fill #1A2D47 border #FFD166 text #FFD166 9pt
      FC(64) → ReLU                         — fill #1A2D47 border #FFD166 text #FFD166 9pt
      Output box fill #1A2D47 border #EF476F:
        "Output: (H, η, ξ₀) — 3 rBergomi parameters" 9pt Bold #EF476F
    All arrows: ▼ 6pt #8899BB between boxes

    Training note (9pt #8899BB): "Trained on 10⁵ synthetic IV surfaces (Monte Carlo,
    Bergomi-Guyon approximation). Inference: <1ms per date vs. hours for classical
    numerical calibration."

  Insert figure_05_covariance_diagnostics.png below the architecture flow.
  Caption (8pt #8899BB): "Condition-number improvement from richer volatility
  parameterization — justifies replacing GARCH σ²_t with the (H, η, ξ₀) triplet."

  Calibration Performance Table
    Header: "Calibration Accuracy (Synthetic Test Set, n = 10,000)" — 11pt Bold #E8EAF0
    4-row × 3-col: "Parameter" | "MAE" | "R²"
    Header fill #132338, 9pt Bold #8899BB
    Data rows alt fill, 10pt #E8EAF0:
      Hurst exponent H    | 0.012 | 0.94
      Vol-of-vol η         | 0.018 | 0.91
      Initial variance ξ₀ | 0.005 | 0.96
    Note: 8pt #8899BB "Tight clustering around diagonal confirms accurate
    calibration. H achieves highest R² as it most strongly shapes IV skew."

COLUMN 3 (width ≈ 33%):

  Hurst Exponent by Period (styled timeline table)
    Header: "Time-Varying Hurst Exponent H_t (SPY, 2010–2024)" — 12pt Bold #E8EAF0
    7-row × 3-col table: "Period / Regime" | "H̄_t" | "Economic Meaning"
    Header fill #132338, 9pt Bold #8899BB; data 9pt #E8EAF0 alt rows:
      2010–2012 (post-GFC)   | 0.11 | Very rough; IV skew steep
      2013–2017 (Bull)       | 0.12 | Rough, vol persistent
      2018 Q4 (Stress)       | 0.28 | Transitional; approaching BM
      2020 Q1 (Crisis)       | 0.31 | Least rough in full sample
      2021–2022 (Inflation)  | 0.18 | Moderately rough
      2023–2024 (Recovery)   | 0.11 | Return to baseline roughness
      Full sample average    | 0.12 | Confirms rBergomi (H < 0.5)
    Last row fill #132338, text Bold #06D6A0
    Annotation box (fill #0F1F33 border #F0A500):
      "H < 0.5: sub-diffusive (rough) process.
       Spikes toward H ≈ 0.3 during acute stress = brief structural
       break toward Brownian motion."
      8pt #8899BB "Gatheral, Jaisson & Rosenbaum (2018): H ≈ 0.1 across all major
      equity indices."

  Insert figure_15_sensitivity.png below the annotation box.
  Caption (8pt #8899BB): "Parameter sensitivity (η, ξ₀) across IV surfaces —
  confirms DNN robustness and identifiability of rBergomi parameters."

  RL State Integration note (fill #1A2D47 border #06D6A0 0.5pt)
    Header: "Integration with RL State Vector" — 10pt Bold #06D6A0
    Body (9pt #E8EAF0):
      "At each step t, the calibrated triplet (H_t, η_t, ξ₀,t) is appended to the
       PPO agent's observation vector alongside PCA features, regime label z_t, and
       lagged portfolio weights w_{t-1}. Time-varying roughness provides richer
       volatility conditioning than GARCH σ²_t alone."

SLIDE NUMBER: bottom right, 9pt #8899BB


==============================================================================
PROMPT 2 OF 2 — PHASE 5: RL EXECUTION ENGINE
==============================================================================

FIGURES TO EMBED (from docs/figures/ — use in the order listed):
  1. docs/figures/figure_07_optimal_weights.png
     Use in Column 3 above the emergent-policy weights table — shows the
     time-series of portfolio weights during the walk-forward backtest.
  2. docs/figures/figure_10_cumulative_returns.png
     Use in the bottom-left chart area — primary performance evidence comparing
     RAMPA against benchmarks over the 2015–2024 OOS window.
  3. docs/figures/figure_11_drawdown.png
     Use in the bottom-center chart area — illustrates the −10% hard drawdown
     constraint and RAMPA's superior downside control vs. 60/40.
  4. docs/figures/figure_12_rolling_metrics.png
     Use in the bottom-right chart area — shows rolling Sharpe ratio and other
     risk-adjusted metrics confirming regime-transition outperformance.
  5. docs/figures/figure_13_attribution.png
     Use in Column 1 below the PortfolioEnv design block — illustrates return
     attribution by asset and regime, reinforcing the learned allocation logic.
  6. docs/figures/figure_14_risk_dashboard.png
     Use in Column 2 below the PPO algorithm block — provides a holistic
     risk view (VaR, volatility, Calmar) as produced by the backtest engine.

Create a full-content academic PowerPoint slide titled "Phase 5 — Reinforcement
Learning Execution Engine" with subtitle "Custom Gymnasium PortfolioEnv · Proximal
Policy Optimization (PPO) · Markovian reward · Emergent regime-conditional policy."

SLIDE TITLE BAR
  Title: "Phase 5 — Reinforcement Learning Execution Engine" — 28pt Bold #58A6FF
  Subtitle: "Custom Gymnasium PortfolioEnv  ·  PPO  ·  Markovian reward  ·  Emergent policy"
  — 14pt #F0A500

LAYOUT: Three-column (top 58%) + three-chart strip (bottom 42%)

COLUMN 1 (width ≈ 35%):

  PortfolioEnv Design block (fill #1A2D47, border #4A90D9 1pt)
    Header: "Gymnasium PortfolioEnv — MDP Design" — 12pt Bold #58A6FF

    State vector typeset (11pt #E8EAF0):
      s_t = [x^feat_t,  z_t,  H_t, η_t, ξ₀_t,  w_{t-1}]
    Under-brace labels (9pt #8899BB):
      x^feat_t:     PCA + macro features
      z_t:          regime label (0–3)
      H_t,η_t,ξ₀_t: vol oracle output
      w_{t-1}:      lagged portfolio weights

    Action space (10pt #E8EAF0):
      "a_t ∈ Δ⁴ (probability simplex, long-only, N = 5)"
      "w_t = softmax(a_t) — enforces sum-to-one by construction"

    Constraints enforced in environment (9pt bullets #E8EAF0):
      • w_i ≥ 0 for all i  (long-only)
      • Σ w_i = 1          (fully invested)
      • Turnover penalty: κ ||w_t − w_{t-1}||₁
      • Episode terminates at −10% drawdown threshold (hard constraint)

  Insert figure_13_attribution.png below the PortfolioEnv block.
  Caption (8pt #8899BB): "Return attribution by asset and regime — illustrates
  how learned allocation tilts drive performance across market conditions."

  Reward Function block (fill #0F1F33, border #F0A500 1pt)
    Header: "Markovian Reward Function" — 11pt Bold #FFD166
    Typeset reward (11pt #E8EAF0):
      r_t = (μ̂_t − λ/2 σ̂²_t) − κ ||Δw_t||₁ − δ · DD⁺_t
    Legend (9pt #8899BB, 4pt spacing):
      μ̂_t:           realized portfolio return at step t
      σ̂²_t:          21-day rolling realized variance
      κ = 0.001:      proportional transaction cost coefficient
      δ = 5.0:        drawdown penalty coefficient
    Justification box (9pt #EF476F):
      "Terminal Sharpe is non-Markovian. This step-level proxy maintains the Markov
       property required by PPO while aligning with risk-adjusted portfolio metrics."

COLUMN 2 (width ≈ 32%):

  PPO Algorithm block (fill #1A2D47, border #06D6A0 1pt)
    Header: "Proximal Policy Optimization (Schulman et al., 2017)" — 11pt Bold #06D6A0
    Clipped objective typeset (10pt #E8EAF0):
      L^CLIP(θ) = E_t[ min( r_t(θ) Â_t,
                     clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
    Parameters legend (9pt #8899BB):
      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
      Â_t:   GAE-λ advantage estimate (λ = 0.95)
      ε = 0.2: clipping threshold (prevents destructive policy updates)

    Network Architecture box (9pt #E8EAF0):
      Actor:  FC(256) → FC(128) → FC(5) → Softmax
      Critic: FC(256) → FC(128) → FC(1)
      Entropy coefficient: 0.01 (maintains exploration)
      Activation: ReLU throughout; bias initialized to 0

    Training Setup table (9pt):
    2-col borderless: label | value, alt rows:
      Total timesteps     | 1,000,000
      Convergence         | ~700,000 steps
      Training env.       | 2010–2014 (in-sample)
      Evaluation          | 2015–2024 (expanding window)
      Tracking            | MLflow (all runs logged)
      Framework           | Stable-Baselines3 v2.2+

  Insert figure_14_risk_dashboard.png below the training setup table.
  Caption (8pt #8899BB): "Risk dashboard (VaR, volatility, Calmar) produced by
  the walk-forward backtest engine — confirms bounded downside throughout OOS period."

  PPO vs. Alternatives callout (fill #0F1F33, border #8899BB 0.5pt)
    Header: "Why PPO over SAC / DQN?" — 10pt Bold #E8EAF0
    Bullets (9pt #E8EAF0):
      • Continuous action space → DQN inapplicable (discrete actions only)
      • Clipped objective → stable under non-stationary financial data
      • On-policy → no replay buffer required; simpler to tune
      • Limitation: ~1M steps for 5 assets. SAC recommended for N > 20.

COLUMN 3 (width ≈ 33%):

  Insert figure_07_optimal_weights.png at the top of this column.
  Caption (8pt #8899BB): "Walk-forward portfolio weights — agent shifts to IEF/SHV
  during 2020 COVID and 2022 rate-hike stress, recovering equity exposure in 2023."

  Emergent Policy — Learned Weights by Regime
    Header: "Emergent Regime-Conditional Policy (OOS 2015–2024)" — 12pt Bold #E8EAF0
    Sub-note: "No explicit supervision — learned from reward signal alone" — 9pt #8899BB italic
    5-row × 6-col table: "Regime" | SPY | QQQ | IEF | GLD | SHV
    Header fill #132338, 9pt Bold #8899BB
    Ticker headers in #06D6A0 10pt Bold
    Data rows alt fill #1A2D47/#0D1B2A, 10pt #E8EAF0:
      Trending Bull   | 42% | 31% |  8% | 10% |  9%
      Choppy          | 22% | 20% | 20% | 18% | 20%
      High-Vol Stress | 15% | 10% | 38% | 22% | 15%
      Crisis          |  8% |  5% | 32% | 12% | 43%
    Stacked bar chart (mini, horizontal) showing one row per regime as a
    proportional bar (width = 100%):
      Trending Bull:   SPY blue, QQQ teal, IEF gold, GLD gray, SHV slate
      Crisis:          SHV dominant (43%) in slate, IEF gold
    Bar colors: SPY=#4A90D9, QQQ=#06D6A0, IEF=#FFD166, GLD=#F0A500, SHV=#8899BB

  Emergent Learning Interpretation box (fill #0F1F33, border #06D6A0 1pt)
    Header: "Why Unsupervised Learning Works" — 10pt Bold #06D6A0
    Body (9pt #E8EAF0):
      "The agent received no explicit labels for correct allocations. From 700K+
       environment interactions, it discovered:
       • Crisis → severe drawdowns → accumulate SHV (43%)
       • Choppy → near-zero IC signal → equal-weight is dominant
       • Bull → high IC, strong alpha → concentrate in SPY + QQQ (73%)
       This is an emergent property of the reward structure, not a constraint."
    Stat callout: "Reward plateau above random baseline at ~700K steps" — 9pt #F0A500

BOTTOM CHART STRIP (bottom 42% of slide, three equal panels):

  LEFT PANEL — Insert figure_10_cumulative_returns.png
    Header above image: "Cumulative Returns (2015–2024 OOS)" — 10pt Bold #E8EAF0
    Caption below (8pt #8899BB): "RAMPA achieves higher terminal value with
    shallower drawdown during both major stress episodes."

  CENTER PANEL — Insert figure_11_drawdown.png
    Header above image: "Drawdown Profiles — All Strategies" — 10pt Bold #E8EAF0
    Caption below (8pt #8899BB): "−10% hard floor visible on RAMPA curve.
    60/40 breaches −20% during 2022 rate-hike stress."

  RIGHT PANEL — Insert figure_12_rolling_metrics.png
    Header above image: "Rolling Risk-Adjusted Metrics" — 10pt Bold #E8EAF0
    Caption below (8pt #8899BB): "RAMPA outperforms most consistently during
    regime transitions — state-conditional policy provides informational edge."

SLIDE NUMBER: bottom right, 9pt #8899BB
