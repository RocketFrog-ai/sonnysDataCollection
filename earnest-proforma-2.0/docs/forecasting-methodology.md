# New-site forecasting — methodology, models, plots, accuracy

This documents every **forecasting plot** in the **"Forecasting for a new site"** (📍 Drop-a-pin) tab of
`earnest-proforma-2.0/streamlits/app.py` + `coldstart_model.py`: how each is modelled, which model
produces it, the data behind it, a worked **dry run** with real numbers, and a measured **accuracy /
evaluation**.

There are **four forecasting plots** in the tab, all built off one cold-start model:
1. **Total local-market wash count** — history + 5-yr forecast of the whole market, with vs without the new site.
2. **Predicted 5-year trajectory** — the new site's own monthly washes (total / membership / retail) with a band.
3. **Monthly P&L** — revenue vs operating expense vs net, 5-yr, with an optional campaign overlay.
4. **Eating the market** — the new site vs each incumbent, each forecast forward 5 yr, drifting on campaign.

Data sources: `data/main-ds.csv` (monthly washes/revenue per site, the *operational* panel) and
`data/opex-data.csv` (monthly P&L: income, cogs, expenses). site_key = `client_id::site_id` (joins both).
Panel spans ~2,000 sites, Jan 2020 – mid 2026; 1,272 of them are old enough to train the plateau model.

> **No operator input.** The forecast tab takes **only a dropped pin (lat/lon)** — `brand=None` always
> (`app.py:694`). Operator selection exists only on the Explore-markets map as a footprint highlight, not in
> forecasting. So every number below is the **location-only** prediction; the model *can* take a brand (it's the
> strongest feature) but the tab doesn't expose one.

---

## In plain terms (TL;DR)

A brand-new site has **no history**, so a normal time-series model (ARIMA/MOIRAI) can't run — there's nothing
to extend. We predict it entirely from **where it is**, learning patterns from ~1,300 existing sites.

**How the new-site forecast is built (model = `cm.predict_site`):**
1. **Read the location** (`_point_features`) — sites within 5/10/20 km, nearest site's size & distance, local
   density/cluster size, region/state (inherited from the nearest site).
2. **Predict the plateau** — the steady-state monthly washes it settles at (~1.5–2 yr), as a best estimate
   **P50** plus a low–high band (**P10–P90**). (LightGBM quantile regression.)
3. **Split membership vs retail** — a second LightGBM predicts the mature membership share.
4. **Apply a ramp** — a learned growth curve so it climbs from ~0 to the plateau over ~24 months.
5. **Drift years 3–5** — the **local market's own measured trend** (per-neighbour Theil-Sen slope) applied after
   maturity, *saturating* (no infinite boom), with the band fanning out at the trend's confidence interval.
   Sliders add extra drift on top.

**How the market forecast is built (`forecast_series` + `market_trend`):** each existing neighbour's history is
carried forward at the market's robust trend; the new site's trajectory is added on top, minus a learned
distance-decaying retail **cannibalization**.

**The P&L:** revenue = washes × ASP (price/wash from nearby sites); opex = learned $/wash × washes × learned
cost-ramp; net = revenue − opex. A **campaign** overlay models a short retail→membership conversion.

**Dry run (a pin near Wildwood, FL — 28.86, −82.04):** 3 neighbours ≤20 km, nearest at ~6,935/mo, region South,
no operator → plateau **6,287 washes/mo** (band 3,377–10,726), **51% members**; opens ≈2,860/mo → ≈5,700 by
month 12 → settles ≈6,200/mo. Local trend (from 3 neighbours): membership shrinks to **flat** (0%/yr, band
−9.5…+12.3), retail **−7.3%/yr** (band −21.4…+2.8) — a textbook shrink-to-flat on a thin, noisy market.

**Accuracy:** tested by hiding each real site, predicting it from location, comparing to actual (5-fold over
**1,272** sites). **Median plateau error ≈ 27.5%** (typical miss ≈ 2,811 washes/mo). Read it as: *half the time
the prediction is within ~27.5% of the truth.* It's the error on the monthly **rate**, measured once, so it
**does not compound** — annual and 5-year % error stay ~27.5%. The plateau *height* is the hard part; the ramp
*shape* is learned far more reliably (~17% on held-out openings).

---

## 1. The cold-start model (location → 5-year new-site trajectory)

`cm.predict_site(lat, lon)` returns 60 monthly points (total/mem/ret, each with lo/med/hi). It's a product of
four learned pieces:

```
washes(month) = Plateau(location)  ×  Ramp(month)  ×  Drift(month, saturating)
```

**1a. Features from the pin** (`_point_features`, ≈14 features) — neighbour counts within **5/10/20 km**,
their mean & log-sum recent washes, their membership share, **distance to nearest site** and its level, **local
cluster size**, **region/state** (from the nearest site), and a **brand** slot (unused in the tab). A brand
feature, when supplied, is the strongest predictor; without it the model leans on local density + region.

**1b. Plateau model** — **LightGBM quantile regression** on `log1p(mature washes/mo)`, where *mature* = each
training site's mean total washes over **months 18–30** of its life (old enough to be settled, recent enough to
matter). Three quantile regressors (**q10 / q50 / q90**) → P50 best estimate + **P10–P90 band**. A separate
LightGBM predicts the **membership share** (`mat_mem / mat_total`), which splits the plateau into mem vs retail.
A **plateau-override** sidebar box just rescales all three quantiles to a typed number.

**1c. Ramp curve** (`_select_ramp`) — each real site's monthly washes ÷ its own mature level → a normalized 0→1
climb; take the **median across sites**, separately for membership and retail. The data shows membership fills
to ~80% in ~7 months (settled by ~18–24), retail is ~flat from month 1. The curve follows real data to ~3.5 yr
then holds. **Region-pooled** (~1,000 sites), *not* per-cluster — backtested, per-cluster ramps overfit (~21%
WAPE vs ~17% region/global; a ramp from 2–3 cluster siblings is just noise). Thin regions shrink toward global.

**1d. Post-maturity drift** — applied only **after ~month 24**, and it **saturates**:
`eff = τ·(1 − e^(−years/τ))` with **τ = 2** (`_sat_years`). Full strength early, then levels off — a booming
area ramps then plateaus instead of compounding for 5 straight years (real sites plateau by ~2 yr). The drift
**rate** is the local market's measured trend (§2), not a constant; the **band fans out** because the lo/hi
trajectories drift at the trend's lo/hi confidence rates. Two sidebar sliders add **extra** drift on top
(*"Yr 3–5 membership/retail — extra on top of per-site trend, %/yr"*).

> The membership share shown for the site is read **off the realized trajectory** (predicted membership ÷ total
> over plateau months 36–60), not the raw share-model parameter — membership and retail ramp differently, so the
> realized split differs slightly from the parameter (51% param → 55% realized in the dry run).

---

## 2. The local-market trend (`market_trend`) — drives plots 1 & 2

Both the market forecast and the new site's post-maturity drift use the **same** measured trend, computed from
the ≤20 km neighbours' own histories. This is the most important non-obvious mechanism in the tab.

**Per-neighbour, composition-robust** (`market_trend` → `_robust_slope` → `_shrink_annualize`):
- For **each** neighbour, fit a **Theil-Sen** slope (robust median slope, ignores one-off spikes) to
  `log(washes)` over the last ~30 months of a 6-month-smoothed series. Then take the **median slope across
  neighbours.** *Why per-site, not a pooled total ÷ active-sites:* a lump average **lurches** every time a site
  opens or closes, which faked big trends in 1–2-site markets.
- **Membership and retail get separate trends** — they genuinely differ (membership rising, retail flat-to-down).
- **Confidence band:** Theil-Sen gives a slope CI; we **widen** it ×√6 for the smoothing autocorrelation, and
  **shrink the central line toward flat** by its signal-to-noise `t²/(1+t²)` (`t` = slope ÷ SE). A trend not
  clearly different from zero → central ≈ 0 but a **wide band**. A loose ±40%/yr rail only stops a degenerate
  series exploding — it is **not** the old hand-set [−5%, +8%] clamp.

So noisy/thin markets **self-widen** instead of being capped, and a clear local trend carries through.

---

## 3. Plot 1 — Total local-market wash count (history + 5-yr forecast)

The headline forecasting chart (`app.py:774`). Shows the **whole local market** (every site ≤20 km, summed)
with and without the new entrant.

`forecast_series(history, H, g)` builds each line: it **starts exactly at the last actual value**, blends over
~a quarter (`w = e^(−(t−1)/3)`) into a smooth trend line at the recent **deseasonalized level**, growing at the
robust rate `g` from §2, **saturating** (`_sat_years`). It does **not** photocopy seasonality forward — month-by-
month seasonal wiggle 5 yr out isn't meaningful; the real history keeps its seasonality, the forecast is smooth.

The chart draws:
- **market total — actual history** (solid blue): summed neighbour washes to today.
- **market without the new site** (grey dotted): membership + retail each carried forward by `forecast_series`.
- **market total — forecast (with new site)** (blue dotted): the above **− cannibalization** (learned
  `a·exp(−d/L)` retail loss, phased in over 12 mo) **+ the new entrant's own trajectory**.
- **🆕 new entrant — its own journey** (red): the §1 P50 total.
- **forecast band (trend CI)**: re-forecast at the trend's lo/hi rates — fans out with trend uncertainty.

**Cannibalization** (learned, `_fit_cannibalization`): a diff-in-diff event study over **2,791** real
incumbent-opening pairs. For every real opening, each ≤20 km incumbent's retail change pre (−6…−1) vs post
(7…12) **minus the market-wide retail trend** (the control) = the entrant's effect; fit `a·exp(−d/L)` to the
per-distance-bin **median** drop. Result: **hyper-local** (`a≈0.244, L≈2.70 km` pooled; South `a≈0.255, L≈2.94`)
— ~−12% at 2 km, ~−4% at 5 km, ~−1% at 10 km, ~0 by 20 km. Hits **retail** (members are sticky).

> **Guard:** if the pin is **outside any cluster** (no clustered site within 20 km), the tab **refuses to
> forecast** and asks you to move the pin — a forecast with no local grounding would just be a region prior.

---

## 4. Plot 2 — Predicted 5-year trajectory (the new site alone)

Directly the §1 model output: **total / membership / retail** lines + a shaded **P10–P90** band that fans out
with the trend CI. This is the cleanest read on the new site itself, independent of the surrounding market.

---

## 5. Dry run — a pin near Wildwood, FL (28.86, −82.04)

**Features the model saw:** 3 neighbours ≤20 km · nearest 7.6 km at ~6,935 washes/mo · region South / FL · no
brand. **Local trend** (3 neighbours, §2): membership **0.0%/yr** (band −9.5…+12.3) — shrunk to flat;
retail **−7.3%/yr** (band −21.4…+2.8).

| Output | Value |
|---|---|
| Plateau washes/mo (q10 / **q50** / q90) | 3,377 / **6,287** / 10,726 |
| Membership share (model parameter) | 51% |
| Membership share (realized, m36–60) | 55% |
| Ramp source | `region: South` |

**Trajectory (P50 total = mem + ret):**

| Month | Total | Membership | Retail | Band (lo–hi) |
|---:|---:|---:|---:|---:|
| 0  | **2,860** | 691   | 2,169 | 1,536 – 4,879 |
| 6  | 5,293 | 2,339 | 2,954 | 2,844 – 9,032 |
| 12 | **5,700** | 2,707 | 2,993 | 3,062 – 9,725 |
| 24 | 6,119 | 3,140 | 2,979 | 3,287 – 10,440 |
| 36 | 6,085 | 3,288 | 2,797 | 3,269 – 10,382 |
| 48 | 6,191 | 3,394 | 2,798 | 3,326 – 10,564 |
| 60 | **6,191** | 3,394 | 2,798 | 3,326 – 10,564 |

**How to read it:** opens hot at ~2,860 (retail ~flat from day 1, membership still filling), climbs to ~6,100 by
month 24, then **flattens** — membership keeps drifting up (its trend shrank to flat so it just rides the ramp
tail), retail eases with its −7.3%/yr drift, net ~level. The band widens further out as trend uncertainty stacks
on plateau uncertainty.

---

## 6. Accuracy & evaluation (measured backtest)

**Plateau level — leave-one-site-out, 5-fold** (`cm.evaluate_trajectory`, predict each held-out site's plateau
from location alone, no operator):

| Metric | Value | Reading |
|---|---|---|
| Sites tested | **1,272** | every labelled site, held out |
| Median absolute % error | **≈ 27.5%** | half of sites are predicted within ±27.5% |
| Mean absolute error (MAE) | **≈ 2,811 washes/mo** | on a median site of ~7,744/mo |
| Median signed error | **−0.9%** | essentially **unbiased** — not systematically high/low |
| Pearson r (log) | **0.58** | predicted size tracks actual size |
| R² | **0.33** (log) / **0.38** (raw) | captures ~⅓ of the variance in site size |

**Expected vs actual — how many sites land within X%:**

| Within | Share of sites |
|---|---|
| ±10% | **22%** |
| ±20% | **38%** |
| ±30% | **53%** |
| ±50% | **72%** |

So ~half of sites land within ±30% and ~3 in 4 within ±50%. It's an **unbiased ballpark, not a precise number** —
which is why the tool always shows a band, not a single line. For the Wildwood pin (q50 = 6,287/mo) a typical
actual lands within ±27.5% → ~**4,560–8,000/mo**.

> **Band calibration (a known gap).** The P10–P90 band is *meant* to contain the truth ~80% of the time; measured
> out-of-sample it contains it only **~57%** (20% of sites fall below P10, 22% above P90). The quantile regressors
> are **overconfident** — real site-to-site spread is wider than the model's band. Read P10–P90 as a *likely*
> range, not an 80% interval, and widen your own read at the extremes.

| Piece | Test | Result |
|---|---|---|
| **Plateau level** | leave-one-site-out (5-fold, 1,272 sites) | **~27.5% median error**, MAE ~2,811/mo; r=0.58, R²≈0.33; unbiased (−0.9%) |
| **Ramp shape** | leave-one-opening-out (predict first ~3 yr) | **~17% WAPE**; region/global ≈ tie, per-cluster worse (~21%) |
| **Cannibalization** | diff-in-diff event study, **2,791** real pairs | clean distance gradient; matches an independent counterfactual |

> **This does NOT compound over time.** The 27.5% is the error on the site's *level* (its mature monthly wash
> **rate**), measured once per site — not a per-month error that accumulates. A % error on a rate carries
> straight through: if the rate is ~27.5% high, the **annual** (rate × 12) and **5-year** totals are also
> ~27.5% off — the *same* percentage, not 12× or 60× it. It's a **median across sites** (a typical site).

**What this means in practice:**
- **Years 0–2 (ramp + plateau)** = the validated part — trust the middle line most here.
- **Years 3–5** = a **scenario with a range**, not a precise prediction (no 5-yr-ahead history to score against),
  so it saturates and shows the band. The sliders test other long-run assumptions.
- **Biggest error driver** is missing demand data (population, income, traffic), not the method. The tab is
  location-only; supplying an operator (the model's strongest feature) would tighten the plateau materially.
- **Worst case:** a fresh area with no neighbours → no local grounding → the tab refuses (§3 guard) rather than
  emit a national prior dressed up as a forecast.

---

## 7. Plot 3 — P&L (revenue vs operating expense)

**Revenue = washes × ASP.** ASP defaults to the local **cluster blended $/wash**: from the ≤20 km neighbours'
last 12 months we compute membership $/wash and retail $/wash (`main-ds.csv`), blended by the site's membership
share. The ASP slider scales price while keeping the mem/retail split.
`revenue = mem × asp_mem + ret × asp_ret`.

**Opex is LEARNED from `opex-data.csv`** (`load_pnl_monthly`, opex = `total_expenses`):
- **Mature level** = `opex_per_wash() × predicted plateau washes`. `opex_per_wash` = median **$/wash** of mature
  (age 18–30) sites in the pin's **state → region → all** scope.
- **Ramp shape** (`opex_ramp`) = each site's `opex(age) ÷ its own mature opex`, median by age, region-scoped —
  new sites run **hot early (~1.5×)** and settle to ~1× by year 1.
- **Beyond the ~33-mo P&L horizon** opex is driven by the **forecast wash volume** (opex ≈ $/wash × forecast
  washes), not flat-lined — so years 3–5 track the volume drift.
- Optional **Opex cost growth %/yr** slider on top (defaults flat; the raw historical opex trend ≈ −10…−16%/yr
  is a noisy reporting artifact, shown not used).

`opex = mature_opex × ramp × growth`; **net = revenue − opex**. Because opex is high early while revenue is still
ramping, **net is negative for ~1–2 yr then turns positive** — the realistic shape of opening a wash.

> Base opex uses `total_expenses`; the campaign opex-spike detection (§8) uses `true_opex = cogs + expenses`.
> Both from `opex-data.csv`, slightly different definitions.

---

## 8. Campaign — when to run a promotion, and its effect

### The honest signal (event study on `opex-data.csv`, see `notebooks/book_v4.ipynb`)
A "campaign" = a promotional OPEX spike. The **raw** event study looks great (membership washes +20–40%, revenue
+20–30% persisting 12 mo, ROI 3.8× by month 6) — but it's **confounded by the site's own ramp** (~half of
campaign sites are <1 yr old, growing anyway). Once each site's trend is removed (Theil-Sen counterfactual +
diff-in-diff), the **clean incremental effect is short (~1–6 months)** (~+5% revenue at launch, fading by ~month
2–3) and **trend-adjusted ROI is ~break-even** — only ~21% of promos actually win share. Treat as a planning
guide, not a guaranteed payoff.

**What IS real & actionable** — a campaign is a **retail→membership conversion**: biggest where there's **retail
headroom** (low membership share); **best ROI in dense markets** (steals share from competitors). book_v4
spillover (M+1..M+3): focal membership **+19%**, focal retail **−7%**, **neighbour retail −13.6%**; but **77% of
promos gained no share**, ~21% gained *and kept* it (**+6.4pp**).

### The verdict — "should this site run a promotion?"
Three metrics drive it:
- **Neighbours' membership share** = **median** of each established incumbent's own recent-12-mo share (every
  site counts equally). Does membership work in this market?
- **Established incumbents** = count of sites within **20 km** with **≥24 months** of history (`n_obs ≥ 24`).
- **This site's predicted membership** = realized plateau share from the trajectory (§1).

Rule: 🟢 **Recommended** if ≥2 incumbents **and** neighbours' share in **[0.45, 0.82)**;
🔴 **Not recommended** if <2 incumbents (unproven market) or share <0.45 (membership doesn't stick);
🟠 **Marginal** if share ≥0.82 (saturated — little retail left to convert).

### The effect model (`campaign_effect`)
A short retail→membership conversion over a **6-month window**:
- **lift** = `campaign_conv_pct(predicted_membership)` × intensity, headroom-scaled: share <0.65 → **30%**, <0.78
  → **14%**, else **7%** membership-wash lift.
- ramps in over ~2 months, holds through the window, then fades (members partly stick, 12-mo half-life); retail
  gives up ~half as many washes as membership gains.
- **opex bump** = `CAMP_OPEX_TAIL = [1.33, 1.17, 1.10, 1.05, 1.03, 1.02]` × intensity (hot launch month, taper).

Applied to the P&L: `rev = mem·mem_mult·asp_mem + ret·ret_mult·asp_ret`, `opex = opex_base·opex_mult`. Launch
month + intensity are sliders; before launch the campaign lines equal baseline (any earlier rise is the ramp).

---

## 9. Plot 4 — "Eating the market" (your site vs each incumbent)

Washes/month vs months since opening; new site + **each incumbent as its own dotted "expected" line**; on
**Apply campaign** they **drift** (yours up, theirs down).

- **Your site (expected)** = the §1 trajectory (`mem + ret`), green dotted.
- **Each incumbent (expected)** = its **own `forecast_series` forward 60 months** (not flat), starting at its
  last actual. The largest **6** incumbents (≥24 mo, ≤20 km) are drawn individually.
- **On Apply campaign:** each incumbent drifts **down** (`yb·(1 − steal)`) as the promo steals retail; your site
  drifts **up**. `steal` scales with density (`0.06 × min(1, n_inc/4) × intensity`, ≈ up to 6% in dense markets,
  ~0 when isolated; grounded in the −13.6% neighbour-retail spillover) and recovers after the window. The gap an
  incumbent's solid line opens below its dotted line is the share you take from it.

**Real-campaign evidence panel** (`campaign_cluster_panel`): plots actual nearby sites (≤20 km) — membership
share / membership washes / retail washes — with **real detected promo OPEX spikes** marked. Spike detection
(`campaign_months_by_site`): per site, `true_opex = cogs + expenses` flagged where it exceeds **median + 3·MAD
AND 1.3× the trailing-6-month median** (interior months only).

---

## 10. What's "from the data" vs "our one choice"
- **From the data:** plateau level, membership share, ramp shape, the local trend direction/size + its band
  width, the cannibalization curve, the opex ramp & $/wash, the campaign conversion lift & persistence.
- **Our single structural choice:** post-maturity growth **saturates over ~2 years** (τ = 2). It's a *timescale*
  (not a rate cap), picked because real sites plateau by ~2 yr; without it, extrapolating a local boom for 5
  straight years doubled the forecast unrealistically.

## 11. Things tried and dropped
- **MOIRAI / deep time-series** — can't forecast a no-history site (cold-start); on existing sites only ties a
  seasonal-naive baseline.
- **Per-cluster ramp curves** — noisier and worse than region-pooled (~21% vs ~17%).
- **A hard ±% growth clamp** — replaced by shrink-to-flat + confidence band + saturation (§2, §1d).
- **Pooled total ÷ active-sites trend** — lurches on site entry/exit; replaced by per-neighbour median slope.

## 12. Reproduce
- New-site forecast: `cm.predict_site(lat, lon, brand=None)` → `(traj, info)`.
- Accuracy: `cm.evaluate_trajectory(n_folds=5)` → `{n, mape, mae}`.
- Market trend: `market_trend(date×site pivot)` in `app.py` → `(g, g_lo, g_hi)`.
- Campaign signal: reproduce the event study in `notebooks/book_v4.ipynb` on `data/opex-data.csv`.
- Companion deep-dive: `streamlits/PINPOINT_FORECAST.md`. Run from `earnest-proforma-2.0/streamlits/` with the
  project venv (conda `sonnysDataCollection`).
