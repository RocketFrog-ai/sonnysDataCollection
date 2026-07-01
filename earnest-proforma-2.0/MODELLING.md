# Proforma Modelling — How It Works

How the cold-start car-wash proforma turns a dropped pin into a 5-year P&L. Wash count comes
**strictly from the model**; ASP/OPEX/CAPEX are dashboard knobs applied on top.

Code: [`streamlits/coldstart_model.py`](streamlits/coldstart_model.py) (the forecaster),
[`streamlits/app.py`](streamlits/app.py) (UI + P&L), `data/main-ds.csv` (panel).

---

## 1. The core equation

Drop a pin (lat/lon, optionally an operator) → a monthly trajectory for 60 months:

```
trajectory(month) = PLATEAU_LEVEL  ×  RAMP_SHAPE(month)
```

- **PLATEAU_LEVEL** — the mature monthly wash count (how big the site gets).
- **RAMP_SHAPE** — the normalized lifecycle curve (how fast it gets there), = 1.0 at maturity.

Membership and retail are forecast separately (a share model splits the total), because they ramp
and monetise differently.

---

## 2. PLATEAU_LEVEL — the mature wash count

A machine-learning model on **location + local-market features**: neighbour counts/levels at 5/10/20 km,
nearest-site distance & level, cluster size, region/state, and the operator's track record (`brand_loo`).

Trained on sites with a real mature window (≥4 months in the 18–30-mo band) and ≥24 months of history.
Two engines:
- **LightGBM** quantile regression (q10/q50/q90) → gives the P10–P90 uncertainty band.
- **ExtraTrees** → the central level (more accurate on this small/noisy cold-start set); band widths
  borrowed from the LightGBM quantiles.

Two corrections make the level reflect the **local market** rather than the global average:

1. **Local-mature anchor.** With no operator given, the model's strongest feature would be blank, so we
   fill it with the **median mature wash count of the pin's matured neighbours** (≥3 sites within 20 km,
   ≥24 mo). Backtested: WAPE 46.6%→40.2%.
2. **Anchor-weight calibration (`ANCHOR_CALIB_W = 0.50`).** The level model still regresses *strong*
   clusters toward the global mean (LOSO: strong-cluster bias 0.92, and the top quartile fell below its
   neighbour floor 21% of the time and *never* over-predicted). So we blend the predicted level toward the
   neighbour median in log space, only for pins that have a real anchor:
   `level = expm1( w·log1p(anchor) + (1−w)·log1p(model) )`.
   Backtest @ w=0.5: bias 0.912→0.924, in-band (neighbour P25–P75) 60.6%→70.0%, WAPE +0.4pp; strong-tercile
   bias 0.92→~0.96 with no overshoot. Capped ≤0.6 (higher erodes weak markets). Touches the ~40% of pins
   that have an anchor; the rest pass through.

### The 4 dashboard models (differ ONLY in how the level is set)

| Model | Level engine | Local anchor | Operator known | Use |
|-------|-------------|:---:|:---:|-----|
| ~~1~~ | LightGBM | ✗ | ✗ | **Dropped** — under-predicts (bias 0.72, 29% below neighbour floor) |
| 2 | LightGBM | ✓ | ✗ | LightGBM cross-check |
| 3 | ExtraTrees | ✓ | ✗ | **Default** — best cold-start (operator unknown) |
| 4 | ExtraTrees | ✓ | ✓ | Best overall — only when a known operator is selected |

LOO WAPE: M2 43.6 · M3 40.2 · M4 ~34.1 (operator known).

---

## 3. RAMP_SHAPE — the lifecycle curve

Each historical site's monthly washes are normalized to its own months-18–30 mean (its plateau = 1.0),
then the **median normalized curve** is taken at each age-month. Membership ramps up over ~7 months;
retail is roughly flat from month 1.

Built hierarchically (global → region → cluster) so a pin can use its local market's curve, shrinking to
the parent when the local sample is thin. The **cleaned pool** drops the COVID-distorted **2020 cohort**
and requires ≥30 months of history — validated by temporal holdout (predicting unseen 2022/2023 cohorts
improved). Default selection is **region-pooled** (cluster-anchoring the shape didn't beat it in backtest;
it's available as an opt-in). The data-driven part runs to month 42, then holds (a gentle slope extension
is available as an opt-in).

A share model assigns each month's **membership vs retail** split.

---

## 4. Neighbours — cannibalization

Every existing site within the radius is projected flat and docked a **retail cannibalization** that
decays with distance, `a·exp(−d/L)`. The `(a, L)` curve is **learned from a diff-in-diff event study**:
for each real opening, each incumbent's pre→post retail change is compared to the market-wide mature-site
trend over the same calendar window (the control removes the secular retail-down/membership-up drift),
then `−a·exp(−d/L)` is fit to the (distance, excess-change) pairs. Pooled + per-region, with a safe
fallback.

---

## 5. P&L layer — washes → dollars

In [`app.py`](streamlits/app.py):

- **Revenue** = `membership_purchases × $/purchase + retail_washes × $/wash`, where membership washes are
  converted to purchases via the cluster purchases-per-wash ratio. The **ASPs are cluster-derived** (the
  ≤radius neighbours' last 12 months) and editable per year in the UI.
  - **Corruption filter.** ~90 sites across 9 operators have a data-feed fault where revenue decays to ~0
    while wash counts continue. `_drop_corrupt_asp_rows` removes site-months with ≥200 washes but an
    implied $/wash below a floor ($4 membership / $5 retail) **before** pooling the cluster ASP — otherwise
    a couple of bad neighbours halve the $/wash and the forecast revenue. Wash-weighted, so clean markets
    are unchanged; falls back to a global healthy ASP if every neighbour is corrupt.
- **OPEX** = a fitted opex-%-of-revenue curve (hot launch ~55–62%, easing to a mature ~45%), re-scaled to
  the per-year targets in the UI.
- **CAPEX** = build cost from tunnel length (learned from real builds), spread across the 5 years for the
  income-statement view.
- **Breakeven** = **cash payback**: months until cumulative *operating* profit (sales − OPEX) recovers the
  **full upfront** CAPEX. (It does *not* use the amortized-CAPEX net, which would report payback far too
  early.)

---

## 6. LLM summaries (context, not the model)

Separate from the numbers — `app/pnl_analysis/insights/`. The explore-markets "Analysis" dropdown offers:
- **Key Insights** — grounded in this market's actual data (LangGraph pipeline).
- **Direct** — location-only world-knowledge read (no data sent).
- **Pollinated** — fuses location commentary × data insights × the competitive landscape; data wins ties.
- **Competitive landscape** — the LLM estimates the *full* competitive set in the trade area (total + express
  tunnels, typed competitor table, intensity, pricing, expansion, headroom) vs the **client's own portfolio**,
  yielding a "coverage scale" multiple (e.g. the client runs 3 of an est. 8–12 express tunnels → scale
  competitive pressure ~3×). These are estimates/context; they never alter the modelled numbers.

---

## 7. Running / retraining

- **App:** `venv/bin/streamlit run earnest-proforma-2.0/streamlits/app.py`
- **Refit the cold-start artifact** (after changing the level model / ramp): run
  `coldstart_model.py` in the conda `sonnysDataCollection` env (sklearn 1.6.1) so the FastAPI backend can
  unpickle it — `coldstart_artifacts.joblib`. The anchor calibration and ASP/breakeven logic are
  inference-time and need **no** refit.

---

## 8. Key facts established by backtest

- Wash count is **mildly conservative**, worst in **strong clusters** (mean-regression) → fixed by the
  anchor calibration.
- The yr-4–5 "decline" in raw data is a **2020 COVID cohort artifact**, not aging → excluded from the ramp.
- The big revenue shortfall some markets showed was an **ASP-data corruption** problem (revenue feed →0,
  washes intact), not the model → fixed by the corruption filter.
- Breakeven was reported ~30 months too early (amortized CAPEX) → fixed to true cash payback.
