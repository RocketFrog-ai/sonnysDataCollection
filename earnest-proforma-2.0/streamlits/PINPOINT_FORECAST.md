# Drop-a-Pin Forecast — How It Works

*A detailed but plain-English guide to the two charts in the **📍 Drop-a-pin** mode.*
Code: [`coldstart_model.py`](coldstart_model.py) (the model) + [`app.py`](app.py) (the charts & trend logic).

---

## 1. The problem & the idea

You drop a pin on a spot that has **no car wash yet**, pick the operator, and ask: *how many washes/month will it do for the next 5 years, and what does it do to the neighbours?*

There's no history to extend (that's what makes it **"cold-start"**). A normal time-series model (ARIMA, MOIRAI, etc.) needs the site's own past — it has none. So we **borrow patterns** from sites that already exist: the same operator's other sites, and the sites physically nearby.

The whole forecast is one sentence:

> **A new site rises to a ceiling, on a known climb shape, then drifts gently with its local area (leveling off) — and it quietly steals a little retail from very close neighbours.**

Four learned pieces make that sentence numeric: **Plateau, Ramp, Trend, Cannibalization.**

---

## 2. The data it learns from

- Monthly panel: ~2,000 sites (each = `client_id + site_id`), Jan 2020 – mid 2026.
- Per site/month: **membership** wash count & revenue, **retail** wash count & revenue, lat/lon, operator, region/state, and **operational start date** (so we know each site's *age*).
- "Mature" volume for a site = its **average total washes over months 18–30** of its life (old enough to be settled, young enough to be recent). This is what the plateau model is trained to predict.

---

## 3. The "local market" (clustering) — and the trick

Several pieces need *"the neighbours."* The obvious approach — **one fixed 20 km circle** (plain DBSCAN) — fails badly in two opposite ways:
- **Dense metros:** every site has dozens of neighbours within 20 km, so the whole city fuses into one giant market.
- **Chaining:** even spread-out sites daisy-chain. If A–B–C–D are each 20 km apart, A links B, B links C, C links D… and you get one **60 km blob** even though A and D are nowhere near each other. On this data plain DBSCAN produced markets up to **138 km wide**.

So we use a **density-aware adaptive** scheme with three tricks:

**Trick 1 — adaptive reach (finer where it's dense).**
Each site gets its own radius: **10 km if it's in a dense spot** (≥5 sites within 10 km) **else 20 km**. Dense areas get carved finely; sparse rural sites still reach far enough to find a partner.

**Trick 2 — mutual `min(rᵢ, rⱼ)` linking (the key trick).**
Two sites link **only if the distance is within *both* of their reaches** — i.e. `distance ≤ min(rᵢ, rⱼ)`. This stops a lone rural site's *big* 20 km reach from reaching **into** a dense cluster and dragging it out:

```
   dense site D (reach 10km) ───15km─── rural site R (reach 20km)
   R would "see" D (15 ≤ 20), but D would not see R (15 > 10).
   min(10, 20) = 10 < 15  →  NO link.  The dense cluster stays clean.
```

Both sites have to "agree" they're close. That single rule is what prevents dense markets from absorbing their sparse surroundings.

**Trick 3 — diameter cap + re-split (anti-chaining).**
After forming connected groups, any group **wider than 25 km end-to-end** is broken up again with **complete-linkage** clustering (which caps the *maximum* pairwise distance, not the average — so no chaining can survive). Lone sites become **standalone** (orange/grey on the map, not forced into a market).

**Why this one won (bake-off vs fixed-20 km DBSCAN & HDBSCAN):** same coverage (~80% of sites placed) but markets that actually **move together** more (within-cluster co-movement **0.59 vs 0.52**), **bounded size** (90th-percentile diameter **21 km vs 50 km** — chaining gone), and **finer markets** (387 vs 235). It's implemented once in `assign_clusters(...)` and used by both the app and the model.

> The **trend** and **cannibalization** pieces don't use the cluster boundaries at all — they use a simple **"every site ≤ 20 km from the pin"** list, so they're unaffected by where a cluster line happens to fall.

---

## 4. Building block 1 — Plateau level (the ceiling)

**Goal:** the site's *mature* monthly washes, before any ramp or drift.

**Method:** **LightGBM gradient-boosted trees**, trained as **quantile regression** so we get a low/middle/high instead of a single number:
- three models at the **10th / 50th / 90th percentile** → `P10 / P50 / P90`,
- plus a separate model for the **membership share** (what fraction of washes are members).

**Target:** `log(mature total washes)` (log so big and small sites are treated proportionally).

**Features it uses** (≈14):
| Group | Examples | Why |
|---|---|---|
| **Operator** | leave-one-out avg mature volume for that brand, # of that brand's sites | **Strongest signal** — a big national operator ≠ a single-site local |
| **Neighbour demand** | # sites within 5/10/20 km, their avg & summed recent volume, their membership share | A busy area supports a busier site |
| **Location** | lat, lon, region, state, distance to nearest site, that site's volume | Geographic/market context |

> **Leave-one-out** on the operator feature means: when scoring a site, the brand average is computed **without that site**, so it can't peek at its own answer. Same trick used in evaluation.

**Output for a pin:** `P50` (the middle ceiling), `P10–P90` (the range), and the membership share. If you type a **plateau override**, it just rescales these to your number.

---

## 5. Building block 2 — Ramp (the climb)

**Goal:** the *shape* of getting from 0 to the ceiling.

**Method:** take every real site, divide its monthly counts by **its own** mature level → a normalized curve that starts near 0 and rises to ~1. Then take the **median curve across sites**, separately for membership and retail.

What the data shows:
- **Membership** reaches ~**80% of mature in ~7 months**, settled by ~**18–24 months**.
- **Retail** is ~**flat from month 1** (no long ramp).
- The curve follows real data out to ~**3.5 years**, then holds flat.

**Which sites' curve do we use?** A **region-pooled** curve (~1,000 sites), not the pin's own cluster. We tested per-cluster curves and they were **worse** (~21% error vs ~17%) — a ramp built from 2–3 cluster siblings is just noisy. Thin regions fall back toward the global curve (shrinkage: weight = `n / (n + k)`).

---

## 6. Building block 3 — Trend (post-maturity drift)

**Goal:** after the site matures, does its area drift up or down — and by how much?

This is the part we reworked most, so here's the full logic:

1. **Per-site slopes, not a lump average.** For **each** neighbour we fit a **Theil-Sen** slope (a robust line that ignores one-off spikes) to `log(volume)` over the last ~30 months, on a 6-month-smoothed series. Then take the **median slope across neighbours.**
   - *Why per-site:* a lump-sum "total ÷ active sites" average **lurches** every time a site opens or closes, which faked big growth/decline in 1–2-site markets. Each site's own slope doesn't have that problem.
2. **Membership and retail get separate trends** — they genuinely differ (membership rising, retail flat-to-down).
3. **Honesty about noise (the confidence band).** Theil-Sen also gives a **confidence interval** on the slope. We:
   - **widen** it for the smoothing (6-month averaging makes neighbouring points correlated → fewer truly-independent points → ×√6 on the error), and
   - **shrink the middle line toward flat** when the trend isn't clearly different from zero (a 1-site market just bouncing on noise → central ≈ flat, but a **wide band**). Formula: keep a fraction `t²/(1+t²)` of the slope, where `t` = slope ÷ its error.
   - So **noisy/thin markets self-widen** instead of being forced to a hand-picked cap.
4. **Saturation (the key fix).** The drift is applied **only after ~month 24**, and it **decelerates**: instead of compounding `(1+g)` every year, it uses *saturating* years
   `eff = τ · (1 − e^(−years/τ))`, with **τ = 2**.
   - Effect: full strength early, then **levels off** — so even a booming area ramps then plateaus, instead of doubling over 5 years. (Real sites are observed to plateau by ~2 years; this matches that.)

There used to be a hard `[−5%, +8%]/yr` clamp here — it's **gone**, replaced by *(shrink-to-flat + confidence band + saturation)*, all driven by the data's own spread.

---

## 7. Building block 4 — Cannibalization (stealing from neighbours)

**Goal:** how much **retail** the new site pulls from existing ones, by distance.

**Method — a before/after "diff-in-diff" event study** (measured from the data, not assumed):
- For **every real opening** in the history, look at each already-open incumbent within 20 km.
- Compare that incumbent's **retail in the 6 months before** vs the **7–12 months after** the opening…
- …**minus** the market-wide retail trend over the same months (the "control" — so we don't blame the new site for a general up/down swing).
- That leftover drop = the new site's effect. Fit a curve `a · e^(−distance/L)` to the **median** drop per distance bin (median, because a few incumbents boom and would skew an average).

**What the data says** (from 2,791 incumbent-opening pairs):
| Distance | Retail lost |
|---|---|
| 2 km | **−12%** |
| 5 km | **−4%** |
| 10 km | **~−1%** |
| 20 km | **~0%** |

So it's **hyper-local** (`a≈0.24, L≈2.7 km`) — much shorter-range than the old fixed assumption. It hits **retail** (members are sticky) and phases in over the first year. Where a region has enough openings we fit it **per-region** (e.g. South: `a≈0.26, L≈2.9 km`); otherwise the pooled curve is used.

---

## 8. Putting it together — the two charts

**Chart 1 — the new site's 5-year trajectory:**
```
washes(month) = Plateau  ×  Ramp(month)  ×  Drift(month, saturating)
```
computed for **low / middle / high** plateau **and** low / middle / high trend → the shaded range that **fans out** over time. Membership and retail are computed separately and added.

**Chart 2 — the whole market, with vs without the new site:**
```
WITHOUT new site = every existing neighbour carried forward at the local trend (saturating)
WITH new site    = WITHOUT  −  cannibalization(retail, phased in over 12 mo)  +  the new site's trajectory
```
- **Solid line** = real history. **Dotted** = forecast. **Shaded band** = trend confidence interval.
- **Grey dotted** = market without the new site. **Red** = the new site's own contribution.
- The caption reports the **net** market change at year 5 and its band.

---

## 9. How accurate is it? (honest)

| Piece | Test | Result |
|---|---|---|
| **Plateau level** | Leave-one-site-out (hide a real site, predict it) | **~27–28% median error** with a known operator (≈**40–46%** for naive guesses); R²≈0.33 |
| **Ramp shape** | Leave-one-opening-out (predict a real opening's first 3 yr) | **~17% error**; region vs cluster vs global ≈ no difference (so we use the simplest, region) |
| **Cannibalization** | Before/after event study, 2,791 real pairs | Clean distance gradient above; direction & size match an independent counterfactual check |

**What this means in practice:**
- **Years 0–2 (ramp + plateau)** = the validated part. Trust the middle line more here.
- **Years 3–5** = a **scenario with a range**, not a precise prediction — there's no 5-year-ahead history to score it against, so we let it saturate and show the band rather than pretend precision. The growth sliders let you test other long-run assumptions.
- **Biggest error driver** isn't the method — it's missing demand data (population, income, traffic). With those, the plateau would tighten; without them, **picking the operator** is what helps most.
- **Worst case:** a fresh area with no neighbours **and** no operator selected → leans on a broad regional average → wide band (by design).

---

## 10. Dry-run example (real numbers)

**Pin:** operator *Wash Rite*, a spot with **3 neighbours within 20 km**, region South.

**Step 1 — Plateau.** Model → mature **≈ 10,975 washes/mo** (range **8,827 – 12,167**), **88% membership**.
**Step 2 — Ramp.** Region-South curve: membership fills over ~7 months, retail flat from month 1.
**Step 3 — Trend.** From the 3 neighbours' own slopes: membership **+4%/yr**, retail **+7%/yr** — applied after month 24, **saturating** over ~2 years.
**Step 4 — Cannibalization (Chart 2).** Closest neighbour 4.4 km → loses ~**6%** retail; the 8–9 km ones lose ~**1%**.

**Resulting trajectory (Chart 1):**

| Month | Total | = Membership | + Retail | Range (low–high) |
|---:|---:|---:|---:|---:|
| 3 | 7,060 | 5,826 | 1,234 | 5,679 – 7,828 |
| 6 | 8,265 | 7,026 | 1,240 | 6,648 – 9,164 |
| 12 | 9,388 | 8,132 | 1,256 | 7,551 – 10,409 |
| 24 | 10,682 | 9,432 | 1,250 | 8,591 – 11,843 |
| 36 | 11,440 | 10,198 | 1,242 | 8,828 – 13,538 |
| 48 | 12,019 | 10,734 | 1,285 | 9,044 – 14,796 |
| 60 | 12,173 | 10,861 | 1,312 | 9,021 – 15,350 |

**How to read it:** climbs hard for ~2 years (7,060 → 10,682), then **flattens** — only **+14% across the last 3 years**, which is the saturation working. Membership dominates (88% operator). The range **widens** further out because the long run is genuinely less certain.

---

## 11. What's "from the data" vs "our one choice"

- **From the data:** plateau level, membership share, ramp shape, trend direction & size, the confidence band width, and the cannibalization curve.
- **Our single structural choice:** growth **saturates over ~2 years** (τ = 2). It's a *timescale* (not a rate cap), picked because real sites are observed to plateau by ~2 years. Without it, extrapolating the local membership boom for 5 straight years doubled the forecast unrealistically.

---

## 12. Things we tried and dropped (so they're not re-tried)

- **MOIRAI / deep time-series** — can't forecast a site with no history (cold-start), and on existing sites only ties a seasonal-naive baseline. Removed.
- **Per-cluster ramp curves** — noisier and *worse* than region-pooled.
- **Age-split / cohort / ridge-regression** variants on the plateau — all worse than the plateau×ramp + saturating-trend setup.
- **The hard ±% growth clamp** — replaced by shrink-to-flat + confidence band + saturation (Section 6).

---

## 13. Reading the bands (quick reference)

- **Wide band** → noisy or thin neighbourhood, or no operator picked. Middle line is a loose guess.
- **Narrow band** → lots of consistent nearby history + known operator. More trustworthy.
- The band stacks **two** uncertainties: *how big the site gets* (plateau P10–P90) **and** *which way the area drifts* (trend interval).

---

## 14. P&L — revenue, operating expense & promotions

The 💰 section turns the wash forecast into dollars: a **monthly revenue vs operating-expense** chart over the 5 years, with a **net** line. Everything is learned from the two data files (`main-ds.csv` for ASP, `pnl_operational.xlsx` for opex).

### Revenue
`revenue(month) = predicted washes(month) × ASP`. One **ASP slider** (blended $/wash) defaults to the **cluster average** — computed from the **dense operational data** (the ≤20 km neighbours' last 12 months in `main-ds`), not the sparse P&L. The membership/retail price split is kept from the data and scaled together by the one slider.

### Operating expense — a *learned new-site ramp* (not flat, not volume-tracking)
We measured how opex actually moves for sites that **opened during the data window** (aligned to each site's age, ÷ its own mature opex). New sites run opex **hot early** — opening/setup/marketing/ramp staffing — then settle:

| months since open | 0 | 1 | 6 | 12 | 24 | 36+ |
|---|---|---|---|---|---|---|
| opex ÷ mature | **1.65×** | 1.61× | 1.24× | 1.13× | 1.12× | 1.00× |

`opex(month) = mature_opex × ramp(month)`, where **mature_opex = $/wash × predicted plateau washes**.
- **Ramp shape** = **pooled across all sites** (only ~63 sites have both early-age and mature data, so it can't be split by region without going noisy).
- **$/wash level** = **scope-aware** — the pin's **state** if it has ≥5 P&L sites (e.g. TX $3.98, FL $6.57), else its **region** (South $4.60), else **all sites** ($4.58). (The P&L is ~90% Texas/South, so most pins land on state-TX or region-South.)
- An **"opex cost growth %/yr"** slider escalates it; it **defaults to flat** — the raw historical opex trend is ≈ −10 to −16%/yr but that's a noisy reporting artifact (recent years under-report), so it's shown, not used.

Because opex is high early while revenue is still ramping, **Net is negative for the first ~1–2 years then turns positive** — the realistic shape of opening a wash.

### Promotions (optional)
A promo window (start, duration, ASP discount %, extra opex %) — **grounded in the data**: across 65 real ASP-dip events, a ~13% dip drove **+8.6% volume / +10% membership** over the next quarter (~¾ of the time), so volume lifts at **elasticity ≈ 0.64** and **persists ~3 months** (decaying half-life) after the window. Revenue dips slightly *during* the discount, then rises *after*. The opex bump during a promo is a **user input** (the data showed no automatic one).

### What's learned vs your input (P&L)
- **Learned:** cluster ASP (main-ds), the opex ramp shape & mature $/wash (P&L), the promo volume-lift elasticity & persistence (P&L).
- **Your input / scenario:** the ASP slider, opex cost-growth (default flat), and any promotion.
