# RAMPA — Phase 3 Slide Prompt
# Formal Graduate Audience · No Emojis · Dark Navy Theme

Global design constants:
  Background:   #0D1B2A  |  Body text:    #E8EAF0  |  Frame title:  #58A6FF
  Accent gold:  #F0A500  |  Gold light:   #FFD166  |  Mint green:   #06D6A0
  Crimson:      #EF476F  |  Slate gray:   #8899BB  |  Electric blue:#4A90D9
  Table header: #132338  |  Block fill:   #1A2D47  |  Code bg:      #0F1F33
  Title font: Calibri Bold  |  Body font: Calibri / Calibri Light
  Slide size: 13.33" × 7.5" (widescreen 16:9)

==============================================================================
PHASE 3 — ALPHA SIGNAL GENERATION
==============================================================================

FIGURES TO EMBED (from docs/figures/ — both are unused in other slides):

  1. docs/figures/figure_03_return_estimators.png
     Place in the RIGHT COLUMN, upper half.
     Caption (8pt #8899BB): "Return estimator comparison — progressive model
     hierarchy from linear baseline to LightGBM ensemble."

  2. docs/figures/figure_06_efficient_frontier.png
     Place in the RIGHT COLUMN, lower half.
     Caption (8pt #8899BB): "Efficient frontier conditioned on alpha signal —
     regime-conditional expected returns shift the frontier meaningfully."

------------------------------------------------------------------------------
LAYOUT INTENT
------------------------------------------------------------------------------
Two-column layout with deliberate whitespace. Do NOT fill every pixel.
Leave ~0.3"–0.4" breathing room inside each block and between sections.
The slide should feel structured and legible, not crowded.

------------------------------------------------------------------------------
SLIDE TITLE BAR  (top ~12% of slide)
------------------------------------------------------------------------------
  Title:    "Phase 3 — Alpha Signal Generation"
            Calibri Bold, 28pt, #58A6FF, left-aligned at x = 0.4", y = 0.25"
  Subtitle: "Decision Tree  ·  Random Forest  ·  LightGBM  ·  Regime-Conditional IC"
            Calibri Light, 14pt, #F0A500, left-aligned, y = 0.75"
  Rule:     #4A90D9, 1pt, spanning full content width, y = 1.05"

------------------------------------------------------------------------------
LEFT COLUMN  (x = 0.4", width = 5.8", content area y = 1.2" to 6.9")
------------------------------------------------------------------------------

BLOCK A — Model Progression  (fill #1A2D47, border #4A90D9 1pt, padding 0.2")
  Header: "Progressive Model Hierarchy" — 12pt Bold #58A6FF
  Three concise bullet items with badge numbers, 10pt #E8EAF0, 6pt line gap:

    [1] Decision Tree (depth = 5)
        · Fully interpretable; all rules extractable as text
        · Establishes IC floor: Mean IC = 0.018 in Trending Bull

    [2] Random Forest (200–500 trees, bootstrap bagging)
        · Variance reduction via ensemble averaging
        · IC = 0.044 in Trending Bull (+144% vs. Decision Tree)

    [3] LightGBM — leaf-wise boosting, histogram binning  ← Bold #06D6A0
        · Captures non-linear higher-order feature interactions
        · IC = 0.082 in Trending Bull (+86% vs. Random Forest)
        · 10× faster training than depth-wise alternatives

  Badge colors: [1] #8899BB  [2] #4A90D9  [3] #06D6A0
  Leave ~0.25" gap below this block before Block B.

BLOCK B — IC by Regime Table  (fill #1A2D47, border #F0A500 1pt, padding 0.18")
  Header: "Out-of-Sample IC by Market Regime" — 11pt Bold #E8EAF0
  Sub-note: "IC = Spearman(signal, next-day return)  ·  63-day rolling window"
            9pt italic #8899BB

  5-row × 3-col table, no outer border beyond block border:
  Header row fill #132338, 9pt Bold #8899BB:
    "Regime"  |  "Mean IC"  |  "IC > 0 (%)"

  Data rows alternating #1A2D47 / #0D1B2A, 10pt #E8EAF0:
    Trending Bull        |  0.082  |  68%
    High-Vol Stress      |  0.055  |  61%
    Crisis               |  0.041  |  58%
    Choppy / Sideways    |  0.021  |  53%
    Regime-averaged      |  0.052  |  60%   ← fill #132338, text Bold #06D6A0

  Single annotation line below table, 9pt #8899BB:
    "Signal collapses near zero in Choppy — regime conditioning is not optional."

BLOCK C — Signal Construction  (fill #0F1F33, border #F0A500 1pt, padding 0.18")
  Header: "Alpha Signal" — 11pt Bold #FFD166
  Three concise lines, 10pt #E8EAF0:
    Target:   r_{t+1}  (next-day SPY log return)
    Signal:   s_t = f_LGB(x_t, z_t)  ∈  [−1, 1]
    Rule:     s_t > 0.5 → net long equity  ·  s_t < 0.3 → defensive tilt (IEF / SHV / GLD)
  Note, 8pt #8899BB: "regime_id (z_t) ranks 4th of all features by LightGBM gain."

------------------------------------------------------------------------------
RIGHT COLUMN  (x = 6.5", width = 6.5", content area y = 1.2" to 6.9")
------------------------------------------------------------------------------

  Insert figure_03_return_estimators.png
    y = 1.2", height ≈ 2.5" (scale to fit, preserve aspect ratio)
    Border: none — image sits directly on dark background

  Caption (8pt #8899BB, italic, directly below image):
    "Return estimator comparison — progressive model hierarchy from linear
     baseline to LightGBM ensemble."

  Vertical gap: ~0.2"

  Insert figure_06_efficient_frontier.png
    y ≈ 4.0", height ≈ 2.55" (scale to fit, preserve aspect ratio)
    Border: none

  Caption (8pt #8899BB, italic, directly below image):
    "Efficient frontier conditioned on alpha signal — regime-conditional
     expected returns shift the frontier meaningfully."

------------------------------------------------------------------------------
FOOTER  (y = 6.95", full width)
------------------------------------------------------------------------------
  Rule: #1E3A5F, 0.5pt, full slide width
  Left (9pt #8899BB): "Cardozo, F. (2024)  ·  Emory University"
  Right (9pt #8899BB): "Ref: Gu, Kelly & Xiu (2020)  ·  Lopez de Prado (2018)"
  Slide number: omit or place bottom-right as "X / 11" in 9pt #8899BB

------------------------------------------------------------------------------
ABSOLUTE RULES
------------------------------------------------------------------------------
  - Zero emojis
  - No gradient fills, no drop shadows, no decorative borders
  - No placeholder or stock images
  - Keep bullet text concise — each point ≤ 2 lines
  - Do not repeat content already covered in Phase 4 or Phase 5 slides
