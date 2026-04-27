# RAMPA — Results & Conclusions Slide Prompt

# References & Q&AFormal Graduate Audience · No Emojis · Dark Navy Theme

**Global design constants:
  Background:   #0D1B2A  |  Body text:    #E8EAF0  |  Frame title:  #58A6FF
  Accent gold:  #F0A500  |  Gold light:   #FFD166  |  Mint green:   #06D6A0
  Crimson:      #EF476F  |  Slate gray:   #8899BB  |  Electric blue:#4A90D9
  Table header: #132338  |  Block fill:   #1A2D47  |  Code bg:      #0F1F33
  Title font: Calibri Bold  |  Body font: Calibri / Calibri Light
  Slide size: 13.33" × 7.5" (widescreen 16:9)**

==============================================================================
RESULTS & CONCLUSIONS
=====================

**FIGURES TO EMBED (from docs/figures/ — both unused in other slides):**

1. **docs/figures/figure_08_risk_parity.png
   Place in the CENTER COLUMN, upper half.
   Caption (8pt #8899BB): "Risk parity vs. RAMPA — equal-risk budgeting
   benchmark highlights the value of regime conditioning."**
2. **docs/figures/figure_09_cvar_optimization.png
   Place in the CENTER COLUMN, lower half.
   Caption (8pt #8899BB): "CVaR-optimized frontier — RAMPA operates in the
   high-Sharpe, low-tail-risk region relative to static MVO alternatives."**

LAYOUT INTENT
-------------

**Three-column layout with deliberate whitespace. Do NOT fill every pixel.
Leave ~0.3"–0.4" breathing room inside each block and between sections.
The slide should feel structured and legible, not crowded.**

SLIDE TITLE BAR  (top ~12% of slide)
------------------------------------

 **Title:    "Results & Conclusions"
            Calibri Bold, 28pt, #58A6FF, left-aligned at x = 0.4", y = 0.25"
  Subtitle: "Walk-Forward OOS 2015–2024  ·  Contributions  ·  Limitations  ·  Future Directions"
            Calibri Light, 14pt, #F0A500, left-aligned, y = 0.75"
  Rule:     #4A90D9, 1pt, spanning full content width, y = 1.05"**

LEFT COLUMN  (x = 0.4", width = 4.1", content area y = 1.2" to 6.9")
--------------------------------------------------------------------

**BLOCK A — Risk-Adjusted Performance Table  (fill #1A2D47, border #06D6A0 1pt, padding 0.18")
  Header: "OOS Performance — 2015–2024" — 11pt Bold #E8EAF0
  Sub-note: "Annualized · Transaction costs included" — 9pt italic #8899BB**

 **6-row × 3-col table (drop Equal-Wt to save space):
  Header row fill #132338, 9pt Bold #8899BB:
    "Metric"  |  "RAMPA"  |  "60 / 40"**

 **RAMPA column header fill #1E3A5F, text Bold #06D6A0.
  Data rows alternating #1A2D47 / #0D1B2A, 10pt #E8EAF0:
    CAGR (%)         |  11.4  |   8.7
    Sharpe Ratio     |  0.91  |  0.63   ← RAMPA cell Bold #06D6A0
    Sortino Ratio    |  1.38  |  0.84
    Calmar Ratio     |  1.14  |  0.43
    Max Drawdown (%) | −9.8   | −22.1   ← RAMPA cell #06D6A0, 60/40 cell #EF476F
    Vol (ann. %)     |  9.3   |  10.8**

 **Annotation line below table, 9pt #F0A500:
    "Sortino edge > Sharpe edge — excess return is upside capture,
     not elevated downside risk."**

 **Leave ~0.2" gap before Block B.**

**BLOCK B — Regime Sharpe Breakdown  (fill #1A2D47, border #F0A500 1pt, padding 0.15")
  Header: "Sharpe by Regime (RAMPA vs. 60/40)" — 10pt Bold #E8EAF0
  Sub-note: "OOS only · HMM labels from Phase 2" — 8pt italic #8899BB**

 **5-row × 3-col table:
  Header row fill #132338, 9pt Bold #8899BB:
    "Regime"  |  "RAMPA"  |  "60/40"**

 **Data rows alt fill, 9pt #E8EAF0:
    Trending Bull      |  1.21  |  0.84
    Choppy / Sideways  |  0.58  |  0.51
    High-Vol Stress    |  0.73  |  0.18   ← RAMPA Bold #06D6A0
    Crisis             |  0.31  | −0.44   ← 60/40 Bold #EF476F
    Full sample avg    |  0.91  |  0.63   ← fill #132338, Bold**

 **Annotation box (fill #0F1F33, border #EF476F 0.5pt, 8pt #8899BB):
    "Largest edge in Stress (+0.55) and Crisis (+0.75) — regime-conditional
     de-risking drives the drawdown improvement."**

CENTER COLUMN  (x = 4.8", width = 3.7", content area y = 1.2" to 6.9")
----------------------------------------------------------------------

 **Insert figure_08_risk_parity.png
    y = 1.2", height ≈ 2.5" (scale to fit, preserve aspect ratio)
    Border: none — image sits directly on dark background**

 **Caption (8pt #8899BB, italic, directly below image):
    "Risk parity vs. RAMPA — equal-risk budgeting benchmark highlights
     the value of regime conditioning."**

 **Vertical gap: ~0.2"**

 **Insert figure_09_cvar_optimization.png
    y ≈ 4.0", height ≈ 2.55" (scale to fit, preserve aspect ratio)
    Border: none**

 **Caption (8pt #8899BB, italic, directly below image):
    "CVaR-optimized frontier — RAMPA operates in the high-Sharpe,
     low-tail-risk region relative to static MVO alternatives."**

RIGHT COLUMN  (x = 8.8", width = 4.2", content area y = 1.2" to 6.9")
---------------------------------------------------------------------

**BLOCK C — Key Contributions  (fill #0F1F33, border #06D6A0 1pt, padding 0.15")
  Header: "Contributions" — 10pt Bold #06D6A0
  Four bullets, 9pt #E8EAF0, 5pt gap:
    · End-to-end regime conditioning: HMM labels flow through LightGBM,
      rBergomi calibration, and PPO state — no component is regime-agnostic
    · Regime-conditional IC: alpha signal informativeness is non-stationary;
      IC = 0.082 (Bull) vs. 0.021 (Choppy) confirms conditioning is load-bearing
    · Rough vol in RL: time-varying H_t appended to PPO observation vector;
      richer conditioning than GARCH σ²_t alone
    · Markovian reward: step-level proxy aligns with terminal Sharpe while
      maintaining the MDP property required by PPO**

 **Leave ~0.2" gap before Block D.**

**BLOCK D — Limitations  (fill #1A2D47, border #EF476F 1pt, padding 0.15")
  Header: "Limitations" — 10pt Bold #EF476F
  Three bullets, 9pt #E8EAF0, 5pt gap:
    · IV data quality: free-tier chains delayed and may miss deep OTM strikes —
      OptionMetrics / CBOE DataShop required for production
    · Cost model: proportional κ understates market impact for large rebalances
    · HMM lookahead: batch training introduces mild label leakage pre-2015**

 **Leave ~0.2" gap before Block E.**

**BLOCK E — Future Directions  (fill #1A2D47, border #F0A500 1pt, padding 0.15")
  Header: "Future Directions" — 10pt Bold #FFD166
  Three bullets, 9pt #E8EAF0, 5pt gap:
    · Online regime estimation via variational Bayes to eliminate lookahead
    · Transformer alpha model to replace LightGBM for long-range interactions
    · Live paper trading via IBKR / Alpaca with real-time IV feeds**

 **Leave ~0.15" gap before Block F.**

**BLOCK F — Takeaway  (fill #0F1F33, border #4A90D9 0.5pt, padding 0.15")
  Header: "Takeaway" — 10pt Bold #58A6FF
  Body (9pt #E8EAF0):
    "Regime conditioning is load-bearing throughout every phase of RAMPA.
     A decade of walk-forward evidence shows it meaningfully improves
     risk-adjusted returns and downside control over static alternatives."
  Stat callout (9pt Bold #06D6A0):
    "Sharpe 0.91  ·  Max DD −9.8%  ·  Calmar 1.14  ·  OOS 2015–2024"**

FOOTER  (y = 6.95", full width)
-------------------------------

 **Rule: #1E3A5F, 0.5pt, full slide width
  Left (9pt #8899BB): "Cardozo, F. (2024)  ·  Emory University"
  Right (9pt #8899BB): "Lopez de Prado (2018)  ·  Gu, Kelly & Xiu (2020)"
  Slide number: bottom-right as "X / 11" in 9pt #8899BB**

ABSOLUTE RULES
--------------

- **Zero emojis**
- **No gradient fills, no drop shadows, no decorative borders**
- **No placeholder or stock images**
- **Keep bullet text concise — each point <= 2 lines**
- **Do not repeat methodology already covered in Phase 1–5 slide**
