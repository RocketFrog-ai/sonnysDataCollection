# New-site forecasting — methodology, accuracy, P&L & campaign

This documents the **Drop-a-pin forecast** ("Forecasting for a new site") tab in
`earnest-proforma-2.0/streamlits/app.py` + `coldstart_model.py`: the 5-year wash forecast, its
accuracy, the P&L (revenue & opex), and the data-driven **campaign** feature (verdict, effect,
"eating the market" chart, and the real-campaign evidence panel).

Data sources: `data/main-ds.csv` (monthly washes/revenue per site) and `data/opex-data.csv`
(monthly P&L: income, cogs, expenses). site_key = `client_id::site_id` (matches across both).

---

## In plain terms (TL;DR)

A brand-new site has **no history**, so we predict it entirely from **where it is** (and who runs
it), learning the patterns from ~1,300 existing sites.

**How the forecast is built:**
1. **Read the location** — sites within 5/10/20 km, nearest site's size, density, region/state, brand.
2. **Predict the plateau** — the steady-state monthly washes it settles at (~1.5–2 yrs), as a best
   estimate **P50** plus a low–high band (**P10–P90**).
3. **Split membership vs retail** — a second model predicts the membership %.
4. **Apply a ramp** — a learned growth curve so it climbs from ~0 to the plateau over ~24 months.
5. **Drift years 3–5** — a gentle saturating trend (no infinite boom) + optional sliders.
6. **P&L on top** — revenue = washes × ASP (price/wash from nearby sites), opex = learned $/wash ×
   washes × learned cost ramp, net = revenue − opex.

**Dry run (Wildwood, FL):** saw 3 neighbours ≤20 km, nearest 7.6 km at ~6,935/mo, region South,
no operator → predicted plateau **6,287 washes/mo** (band 3,377–10,726), 51% members; opens
≈2,860/mo → ≈5,700 by month 12 → settles ≈6,200/mo.

**Accuracy & how to read it:** tested by hiding real sites, predicting them, comparing to actual
(1,272 sites). **Median error ≈ 27.5%** (typical miss ≈ 2,811 washes/mo). Read it as: *half the time
the prediction is within ~27.5% of the truth* — so 6,287/mo means a typical actual ~4,560–8,000/mo.
It's the error on the monthly **rate**, measured once, so it **does not compound** — annual and
5-year % error stay ~27.5%. It's a **median across sites** (a typical site), and it's the
**unknown-operator** case — picking the operator tightens it. The plateau *height* is the hard part;
the ramp *shape* is learned much more reliably.

---

## 1. How the wash forecast works (location → 5-year trajectory)

Cold-start model: a new site has no history, so the forecast is built from **location** alone
(plus operator/brand if known).

1. **Features from the dropped point** (`_point_features`): neighbour counts within **5/10/20 km**,
   mean/log-sum of those neighbours' recent washes, **distance to nearest site** and its level,
   local **cluster size**, **region/state** (from the nearest site), and **brand** if selected
   (strongest predictor).
2. **Plateau model** — LightGBM **quantile** regressors (q10/q50/q90) on `log1p(mature washes/mo)`,
   *mature* = each training site's average over **months 18–30** → mature washes/mo + **P10–P90 band**.
3. **Membership-share model** — separate LightGBM predicting mature membership share
   (`mat_mem/mat_total`) → splits the plateau into membership vs retail.
4. **Ramp curve** (`_select_ramp`) — hierarchical life-cycle (**cluster → region → global**), shapes
   the climb to plateau over ~24 months (region-pooled usually; per-cluster overfits).
5. **Post-maturity drift** — a **saturating** trend (~2-yr time constant) + the growth sliders.
   Output = **60 monthly points** (total/membership/retail) with bands.

> The membership share shown for the site is read **off the realized trajectory** (predicted
> membership ÷ total washes over plateau months 36–60), not the raw share-model parameter — the
> trajectory is *built from* that parameter, but membership and retail have different ramp curves so
> the realized split differs slightly.

---

## 2. Dry run — a pin near Wildwood, FL (28.86, −82.04)

**Features the model saw:** 3 neighbours ≤20 km · nearest 7.6 km at ~6,935 washes/mo ·
region South / FL · no brand.

| Output | Value |
|---|---|
| Plateau washes/mo (q10 / **q50** / q90) | 3,377 / **6,287** / 10,726 |
| Membership share | 51% |
| Ramp source | `region: South` |

Trajectory: month 0 ≈ **2,860** → month 12 ≈ **5,700** → plateau ≈ **6,200/mo** (mem ~3,400, ret ~2,800).

---

## 3. Accuracy (measured backtest)

Out-of-sample backtest (`evaluate_trajectory`, 5-fold across **1,272** labeled sites — predict each
held-out site's plateau from location/brand):

| Metric | Value |
|---|---|
| Median absolute % error (plateau washes/mo) | **≈ 27.5%** |
| Mean absolute error | **≈ 2,811 washes/mo** |

For the Wildwood pin (q50 = 6,287/mo), a typical actual lands within ±27.5% → ~**4,560–8,000/mo**;
the wider P10–P90 (3,377–10,726) brackets more to cover the spread.

> **This does NOT compound over time.** The 27.5% is the error on the site's *level* (its mature
> monthly wash **rate**), measured once per site — not a per-month error that accumulates. A % error
> on a rate carries straight through: if the monthly rate is ~27.5% high, the **annual** total
> (rate × 12) and the **5-year** total are also ~27.5% off — the *same* percentage, not 12× or 60×
> it. It's also a **median across sites**, a measure of typical site-level accuracy, not a temporal
> quantity.

Caveats: (1) error is on the **plateau level** — the ramp *shape* is much better learned. (2) The
27.5% is the **unknown-operator** case; a known operator tightens the band.

---

## 4. The P&L — revenue vs operating expense

**Revenue = washes × ASP.** ASP defaults to the local **cluster blended $/wash**: from the ≤20 km
neighbours' last 12 months we take membership $/wash and retail $/wash (`main-ds.csv`), blended by
the site's membership share. The ASP slider scales price while keeping the membership/retail split.
`revenue = mem × asp_mem + ret × asp_ret`.

**Opex is LEARNED from `opex-data.csv`** (`load_pnl_monthly`, opex = `total_expenses`):

- **Mature level** = `opex_per_wash() × predicted plateau washes`. `opex_per_wash` = median **$/wash**
  of mature (age 18–30) sites in the pin's **state → region → all** scope.
- **Ramp shape** (`opex_ramp`) = each site's `opex(age) ÷ its own mature opex`, median by age,
  region-scoped — new sites run **hot early (~1.5×)** and settle to ~1× by year 1.
- **Beyond ~33 months** (the P&L horizon) opex is driven by the **forecast wash volume**
  (opex ≈ $/wash × forecast washes), not flat-lined.
- Optional **Opex cost growth %/yr** slider on top.

`opex_base = mature_opex × ramp × growth`; **net = revenue − opex**.

> Note: base opex uses the `total_expenses` column; the campaign opex-spike curve (§5) uses
> `true_opex = cogs + expenses`. Both from `opex-data.csv`, slightly different definitions.

---

## 5. Campaign — when to run a promotion, and what it does

### The honest signal (event study on `opex-data.csv`, see `notebooks/book_v4.ipynb`)
A "campaign" = a promotional OPEX spike. The **raw** event study looks great (membership washes
+20–40%, revenue +20–30% persisting 12 months, break-even ~month 1, ROI 3.8× by month 6) — but
that is **confounded by the site's own ramp** (~half of campaign sites are <1 yr old and growing
anyway). Once each site's trend is removed (Theil-Sen counterfactual + difference-in-differences),
the **clean incremental effect is short (~1–6 months)** (~+5% revenue at launch, fading by ~month
2–3) and **trend-adjusted ROI is roughly break-even** — only ~21% of promos actually win share.
The `opex>1.2×` trigger also conflates marketing with capex/renovation. So: treat as a planning
guide, not a guaranteed payoff.

**What IS real & actionable** — a campaign is a **retail→membership conversion**:
- biggest where there's **retail headroom** (low membership share);
- **best ROI in dense markets** (it steals share from competitors);
- book_v4 spillover (M+1..M+3): focal membership washes **+19%**, focal retail **−7%**, **neighbour
  retail −13.6%**; but **77% of promos gained no share**, ~21% gained *and kept* it (**+6.4pp**).

### The verdict — "should this site run a promotion?"
Three metrics drive a recommendation:
- **Neighbours' membership share** = **median** of each established incumbent's own recent-12-mo
  share (every site counts equally). Does membership work in this market?
- **Established incumbents** = count of sites within **20 km** with **≥24 months** of history
  (proof the market is viable/competitive).
- **This site's predicted membership** = realized plateau share from the trajectory (§1).

Rule: 🟢 **Recommended** if ≥2 incumbents **and** neighbours' share in **[0.45, 0.82)**;
🔴 **Not recommended** if <2 incumbents (unproven market) or share <0.45 (membership doesn't stick);
🟠 **Marginal** if share ≥0.82 (saturated — little retail left to convert).

### The effect model (`campaign_effect`)
A short retail→membership conversion over a **6-month window**:
- **lift** = `campaign_conv_pct(predicted_membership)` × intensity, where headroom sets the size:
  share <0.65 → **30%**, <0.78 → **14%**, else **7%** membership-wash lift.
- ramps in over ~2 months, holds through the window, then fades (members partly stick, 12-mo half-life);
  retail gives up ~half as many washes as membership gains.
- **opex bump** = `CAMP_OPEX_TAIL = [1.33, 1.17, 1.10, 1.05, 1.03, 1.02]` × intensity (the promo spend,
  hot launch month then taper — from the event study).

Applied to the P&L: `rev = mem·mem_mult·asp_mem + ret·ret_mult·asp_ret`, `opex = opex_base·opex_mult`.
The launch month + intensity are user sliders; before the launch month the campaign lines are
identical to baseline (the earlier rise is the site's own ramp).

---

## 6. "Eating the market" chart — your site vs each incumbent

Washes/month vs months since opening; new site + **each incumbent as its own dotted "expected"
line**; on **Apply campaign** they **drift** (yours up, theirs down).

- **Your site (expected)** = the §1 trajectory (`mem + ret`), green dotted, ramping to plateau.
- **Each incumbent (expected)** = its **own time-series forecast forward 5 years** (not flat). For
  each established incumbent (≥24 mo, ≤20 km) we project its monthly total-wash history 60 months
  with `forecast_series` (robust Theil-Sen trend, starts at last actual, saturates):
  `yb = [last_actual] + forecast_series(history, 60)`. The largest **6** are drawn individually.
- **On Apply campaign:** each incumbent drifts **down** (`yb · (1 − steal)`) as the promo steals its
  retail; your site drifts **up**. `steal` scales with density (≈ up to 6% in dense markets, ~0 when
  isolated; grounded in the −13.6% neighbour-retail spillover). The gap each incumbent's solid line
  opens below its dotted line is the share you take from it.

---

## 7. Real-campaign evidence panel

"Real campaigns in this local market" plots actual nearby sites (≤20 km) — membership share /
membership washes / retail washes — with **real detected promo OPEX spikes marked** as dotted
verticals. Spike detection (`campaign_months_by_site`): per site, `true_opex = cogs + expenses`
flagged where it exceeds **median + 3·MAD AND 1.3× the trailing-6-month median** (interior months
only). It's the real-world evidence behind the model's conversion story.

---

## 8. Reproduce

- Forecast: `cm.predict_site(lat, lon, brand=None)` → `(traj, info)`.
- Accuracy: `cm.evaluate_trajectory(n_folds=5)` → `{n, mape, mae}`.
- Campaign signal: reproduce the event study in `notebooks/book_v4.ipynb` on `data/opex-data.csv`.
- Run from `earnest-proforma-2.0/streamlits/` with the project venv.
