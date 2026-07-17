# old-proforma-analysis — context

## What this is

A backtest of Sonny's **legacy proforma site-selection documents** against what the built
sites actually did. Each proforma is a pre-build projection for one prospective site: ~10
site-selection factor choices (pay stations, vacuums, corner/light, visibility, competition,
…), 4 demographic components, a traffic count, target capture scores, and a 5-year projected
wash volume. We address-match each proforma to the operating panel
(`proforma/data/panel/main-data-v2-stitched.csv`) and ask two questions:

1. **Calibration** — how far off were the projected volumes, aligned by operating year?
2. **Signal** — do the factor choices/scores actually correlate with real wash volume, or
   are they box-ticking? Any real factor *combinations*?

The projection formula is an identity (verified r=0.997):
`vol_yearly_yN = traffic_count × target_score_yN × ~300 operating days` — so the backtest
decomposes into "was the capture target right" + "does linear-in-traffic hold".

Standalone experiment: nothing outside `experiments/` imports it; it only **reads** the panel.

## Files

```
old-proforma-data/                  179 source docs (119 xlsx + 2 xls parsed; 58 PDFs IGNORED)
code/extract_proformas.py           xlsx/xls -> one row per proforma (2 layouts: input_form=112,
                                    legacy_sheet1/xls=9). PDFs never extracted.
code/match_panel.py                 address-match to the panel + attach actual_* aggregates.
                                    Also: handoff stitching, zip zero-pad, actuals_suspect flag.
                                    Rewrites old-proforma-combined.csv IN PLACE (idempotent).
code/build_monthly.py               expand each matched proforma to 1 row / panel month
                                    (imports union_months from match_panel).
code/make_notebook.py               generates proforma_backtest.ipynb (all analysis cells).
old-proforma-combined.csv           summary: 121 rows (one per proforma) + match_* + actual_*.
old-proforma-combined-monthly.csv   monthly: 4,200 rows = 4,194 site-months + 6 blank rows
                                    for the 6 no_match proformas.
proforma_backtest.ipynb             the executed analysis (regenerate, don't hand-edit).
```

### Rebuild pipeline (conda `sonnys`, from this folder)

```bash
python code/extract_proformas.py        # only if source xlsx change
python code/match_panel.py              # rewrites the summary CSV in place
python code/build_monthly.py
python code/make_notebook.py && jupyter nbconvert --to notebook --execute --inplace proforma_backtest.ipynb
```

## Matching semantics (match_panel.py)

- Normalized street tokens must appear as a contiguous run in the proforma address, then ZIP
  (preferred) or state must agree; two fuzzy fallbacks recover typos ("Dempdter", dropped
  directionals/suffixes). `match_status`: 102 matched_zip, 5 zip_multi, 5 fuzzy, 1 state,
  1 fuzzy_state, 1 street_only (Canadian, no ZIP), 6 no_match (verified genuinely absent).
- **Handoff stitching**: 5 matched sites exist in the panel as TWO client_ids (operator
  rebrand) whose windows overlap at the seam month — the panel's own stitcher deliberately
  skips overlapping windows, so we stitch here: actuals = union of all same-street+ZIP
  segments, seam month resolved by keeping the row with the larger wash_count (every kept row
  is a real panel row); `match_operational_start` = earliest segment start;
  `match_secondary_segments` = "client_id:site_id;…". Monthly rows carry
  `panel_client_id`/`panel_site_id` provenance.
- **`actuals_suspect`** (3 sites): matched record averages <200 washes/mo over its history —
  dead panel records (1–108/mo vs next-lowest real site 462/mo). Exclude from all analysis.
- Panel postal codes lose leading zeros ("1835" = 01835 MA); match_panel zero-pads before
  comparing.

## Analysis conventions (the notebook)

- **Site key is `client_id + site_id`** (repo-wide rule). One proforma = one prospective site;
  no two proformas share a panel record (asserted).
- Only non-imputed panel months are used.
- **`mature_wash`** = median of last 12 clean months, only for sites **≥24 months old**
  (n=70). Without the age gate the mature ratio reads 0.62 instead of 0.71 — ramping sites
  masquerade as weak mature sites. Never quote the ungated number.
- **`y2_wash`** = median over operating months 12–23 (uniform-age outcome, n=78).
- Operating years are aligned to `match_operational_start`; a site only enters op-year
  comparisons if its open date is **inside** the panel window (month-precise; panel is
  left-censored at 2020-01) and the year has ≥10 observed months.
- Choice labels must be normalized before grouping (case variants, "MUTIPLE" typo, "2.0").
- Significance = permutation p + Benjamini–Hochberg FDR across the 16 tested drivers; combo
  claims use max-statistic permutation over the whole searched set.

## Findings (as of 2026-07, n=70 mature / 78 year-2)

- **Mature volume over-projected ~30%**: actual/projected median 0.71 (IQR 0.41–1.10), 73%
  of sites below projection. Ramp years nearly unbiased (Y1 0.87, Y2 0.93). The miss is the
  capture target: assumed 1.43% of daily traffic vs realized 0.94%. MdAPE 30–49%/year.
- **Error grows with traffic** (r=−0.35, FDR q=0.04): linear-in-traffic fails; capture
  saturates. Traffic count alone doesn't even predict volume (oos R²≈0).
- **Real factors (FDR-surviving, replicate on both outcomes, hold Express-only and after
  controlling traffic):** pay_stations (r=0.41), free_vacuum_slots (0.34), type_of_site
  (0.33), cumulative_site_score (0.32). Monotone ladders: pay 1/2/3+ → 3.3k/6.1k/9.4k
  washes/mo; vacuums <12/12-20/>20 → 3.6k/6.3k/9.1k; **traffic light beats corner**
  (corner+light 9.8k > inside+light 5.9k > corner-no-light 4.8k > inside-no-light 3.3k —
  the template scores corner-no-light too high).
- **Dead on this sample:** nearest_competition (sign even flips), visibility, accessibility,
  traffic_speed, area_profile, all 4 demographics. weekly_hours is constant across all 115
  proformas (untestable). ~Half the scorecard carries no signal.
- **No combo synergy**: best pair (pay≥2 & vac-high, +3.7k/mo) fails max-stat permutation
  (p≈0.10) and adds nothing beyond its strongest single lever. Additive main effects only.
- **Ranking ≠ sizing**: best out-of-sample R² from any template inputs ≈ 0.10 (score+traffic);
  the proforma's own projection ≈ 0.03; all-9-factors ridge is negative (overfit). Inputs can
  screen sites, not size volumes — supports comparable-site anchoring (coldstart).
- ASP: realized blended ASP ($16.04 median) ≈ menu-1 price ($10) × 1.6; menu price carries
  no cross-site signal for realized ASP (r=−0.15, ns) — blended ASP is a membership-mix
  variable.

## proforma_backtest.ipynb — in-depth walkthrough

Generated by `code/make_notebook.py` (32 cells; edit the generator, then regenerate + execute
— see pipeline above). Runs top-to-bottom in conda `sonnys` in ~1–2 min (the permutation
tests dominate). All randomness comes from `np.random.default_rng(0)`, so re-execution on
unchanged CSVs reproduces every number exactly. Data loading auto-detects cwd (`BASE = '.'`
if the CSVs are beside the notebook, else `'..'`), so it executes both from this folder and
from `code/`. Every output cell is followed by an **Insights** markdown cell whose numbers are
hard-coded from the last verified run — if you regenerate after a *data* change, re-check
those cells against the printed output (the generator is the place to fix them).

### Setup cell — load + projection-identity check

Loads both CSVs, prints population (121 proformas / 115 matched / 3 suspects), then verifies
the projection identity on all 115: `log(traffic × y1_target)` vs `log(vol_yearly_y1)` gives
r=0.997 with implied operating days locked at median 300 (IQR 300–300). This is the license
for §7's decomposition: projected volume has exactly three ingredients — traffic count,
capture target, day count.

### §1 — site-level analysis table (the denominators everything else uses)

One row per proforma. From clean (non-imputed) monthly actuals:

| Column | Definition | Gate | n |
|---|---|---|---|
| `mature_wash/rev/asp` | median of the **last 12 clean months** | `mature_ok`: age ≥24 mo at last obs, ≥6 months in window, >0 | 70 |
| `y2_wash` | median over operating months 12–23 | `y2_ok`: observed open, ≥6 mo in window, >0 | 78 |
| `mean_y1..y5`, `size_y1..y5` | mean washes + observed-month count per operating year | used with `size_yN ≥ 10` in §2 | 88→12 |
| `open_observed` | `open_idx > 2020-01` **and** `open_idx ≥ first panel row − 1` | month-precise left-censor guard | 96 |

Operating month 0 = the `match_operational_start` month (earliest segment for stitched
sites — this is why the handoff fix upstream matters: Seaford/Jupiter would otherwise get
op-years counted from the *second* operator). Choice labels are normalized here
(`*_choice_n` columns); `FS` = the 9 live factor scores (weekly_hours constant → dropped);
`DEMO` = composites + 4 demographic values + traffic.

### §2 — projected vs actual (year-aligned, plotly)

Three outputs:
1. **Year-aligned plotly grid** (2×3, interactive; hover = site name/address/ratio): panels
   Y1..Y5 plot actual op-year-N monthly washes vs `vol_monthly_yN` (observed opens, ≥10 obs
   months); the sixth panel is the age-gated **mature actual vs the Y5 target** — Y5 *is* the
   template's mature column, so that panel mixes ages by design while the year panels are
   like-for-like. Ratio medians / spearman per panel: Y1 0.87/ρ=0.25 (n=88), Y2 0.93/0.37
   (n=60), Y3 0.85/0.30 (n=46), Y4 0.85/0.38 (n=21, p≈.09), Y5 0.82/0.22 (n=12, ns);
   mature **0.71**/0.31, IQR [0.41, 1.10], 73% over-projected. Prints the ungated ratio
   (0.62, n=108) purely as a bias demonstration — never quote it. (Plotly renderer is
   `plotly_mimetype+notebook`, which embeds plotly.js and grows the .ipynb to ~5.5 MB;
   the remaining figures are matplotlib.)
2. **Op-year accuracy table** (same filters, + per-year spearman column): over-projected
   share rises 60%→75%; MdAPE 30–49% every year.
3. **Capture + cohort**: actual mature capture = `mature_wash × 12 / (traffic × 300)` —
   median **0.94%** vs assumed mature target **1.43%** (ratio 0.70, i.e. the whole mature
   miss). Cohort split: 2020–21 openers 0.61 vs 2022+ 0.76.

### §3 — which inputs carry signal (the fluke-proofing section)

Helpers defined here and reused by §7:
- `perm_spearman(x, y)`: Spearman via standardized ranks, **20k permutations**, two-sided,
  p floored at 1/20k.
- `bh(p)`: Benjamini–Hochberg FDR across the 16 tested drivers (9 factors + 7 DEMO).
- **partial r**: residualize *both* the outcome ranks and the driver on `log(traffic)`
  (linear fit), then Spearman the residuals — "does the driver add signal beyond traffic?".

Run twice — outcome `log(mature_wash)` (n=70) and `log(y2_wash)` (n=78; independent age
definition, so agreement ≈ replication). Results table (mature): pay_stations .414/q=.005,
free_vacuum_slots .336/q=.025, type_of_site .332/q=.025, cumulative_site_score .320/q=.028
— all keep partial r ≈ .30–.38; traffic_count .205/q=.28 fails; all demographics |r|≤.154,
q≥.53; nearest_competition is *negative* (−.075). Year-2 replicates pay (.363/q=.017) and
vacuums (.329/q=.028); entrance_stack_up hits q=.099 there (see §4 trap).
The bar chart stars FDR q<0.05 (not raw p). A robustness cell reruns the three headline
factors Express-only (r=.37–.41, n=66) and checks the era confound
(mature-window calendar year vs outcome: r=.10, p=.41 — not an era artifact).

### §4 — choice-level medians (washes/mo per ticked box)

Groups `mature_wash` by each normalized choice, cells n≥5, with a **permutation
Kruskal–Wallis** (5k shuffles) per factor. Signal: pay_stations p=.018 (1/2/3+ →
3,304/6,142/9,429), free_vacuum_slots p≈.015 (<12 / 12–20 / >20 → 3,576/6,332/9,117),
type_of_site p≈.03 (corner+light 9,766 > inside+light 5,939 > corner-no-light 4,832 >
inside-no-light 3,281 — light dominates corner). Noise: competition .86, visibility .79,
area .51, speed .87. **entrance_stack_up p=.007 is the documented trap**: non-monotone
(">20 vehicles" is the *worst* cell at 3,500) and its top cell is the double-ticked
extraction artifact "20 - 15 / LESS THAN 10" (n=6).

### §5 — combinations (the "certain combos or fluke?" answer)

Six binary levers (`pay2p`, `vac_hi`, `corner`, `light`, `stack_hi`, `comp_multi2`), all 15
pairwise ANDs with ≥8 sites in and out of the cell. Best combo pay2p & vac_hi: 7,006 in vs
3,348 out (+3,657). Honesty device: **max-statistic permutation** — shuffle the outcome
3,000×, rerun the *entire 15-combo search* each time, count how often the best chance combo
beats the best real one → **p=0.094**. Combined with single levers already reaching +3,625
(vac_hi), the verdict is additive main effects, no proven synergy.

### §6 — weightage + honest predictive power

Left chart: template's realized weight per factor (mean |score| share of
`cumulative_site_score`) vs the data's signal share (|Spearman| from §3, normalized).
Right chart + prints: **out-of-sample R²** (RepeatedKFold 5×40, Ridge α=1, features
z-scored on each train fold, R² vs the train-mean baseline) on `log(mature_wash)`, complete
cases n=70: traffic-only −0.006, site-score-only +0.065, score+log traffic **+0.098**,
capacity trio −0.090, all-9-factors −0.242, the proforma's own projection
(log vol_monthly_y5) +0.035. This is the "ranking ≠ sizing" evidence and the anti-overfit
warning for re-weighting factors on 70 sites.

### §7 — error decomposition

`log_err = log(mature_wash / vol_monthly_y5)` (median −0.35 = ratio 0.71), same
perm+FDR machinery as §3. Only traffic_count survives (r=−0.353, q=0.044): over-projection
grows with traffic. cumulative_site_score is next (+0.301, q=0.092): high-scored sites
under-promise. The scatter (log traffic vs log_err, with fit line) is the visual for "the
linear-in-traffic assumption fails". If the projection formula were calibrated, this
section would be all noise — it isn't.

### §8 — ASP reality check

Mature-window blended ASP (`revenue/washes`, trimmed to (0,100)) vs `pkg_menu1_price`
(n=65; legacy-layout proformas lack package prices). Realized median **$16.04** vs menu-1
median **$10.00**; spearman −0.15 (p=0.22) → the menu price is a floor, not a forecast;
blended ASP is a membership-mix variable. Inherits the panel's known revenue-corruption
caveat on a minority of operators.

### §9 — findings + scorecard

Narrative findings (the four numbered conclusions) and a printed scorecard cell that
recomputes the headline numbers live: 112/70 population, 0.71 ratio, 1.43% vs 0.94%
capture, the four FDR-significant drivers with r's, combo p≈0.10, oos R² ceiling ~0.10.
If a data fix changes anything upstream, this cell is the quickest diff surface.

## ensemble/ — coldstart benchmark + the stacked "super" ensemble (2026-07)

`ensemble/` holds the pipeline that (a) benchmarked the production cold-start Model 3
(plateau × ramp, `model_kind="et"`, local anchor) against the old proformas, and (b) built the
stacked ensemble that beats both. All scripts assume conda `sonnys`; result CSVs/JSONs live in
`ensemble/results/` (paths inside the scripts point there).

- `eval_coldstart.py` / `panel_eval.py` — run `predict_site` at each site's pin with
  **leave-one-out artifact surgery** (the site, its handoff twin and anything ≤0.2 km removed
  from `art["sites_rl"]`, so neighbour features + the local-mature anchor can't read the
  site's own actuals; audited: the only residual leaks are O(1/n) shape-only ramp influence
  and one training row in 1,223). 112 proforma sites / 862 panel-eligible sites
  (opened 2021–24, ≥24 clean months, mature >200/mo).
- `compare_forecasters.py` — head-to-head vs the proforma on 227 identical site-years:
  **tied on magnitude** (MdAPE 43 vs 41%), **model ~2× on ranking** (ρ 0.58 vs 0.36), model
  better at maturity (46 vs 58%). Leaky variant (no surgery) only reaches MdAPE 38% — most
  site-year error is irreducible.
- `error_budget.py` — the level-is-everything decomposition: oracle mature level × model ramp
  gives Y2 MdAPE 13% (vs 39%); persistence Y2→Y3 is 9.3%; panel YoY noise floor 7.3%;
  capacity factors explain the model's residuals (pay r=+0.31).
- `ensemble_features.py` / `ensemble_eval.py` — the stacked ensemble under a **fully nested
  LOSO protocol** (level model, median debiases c_mat/c_y, blend weights all re-fit per fold;
  independently audited; shuffled-target negative controls: ridge skill collapses to ρ=0.02 →
  fitted, not leaked). Results (227 site-years / 70 mature):
  **blend** (coldstart⊕proforma level + per-year debias): site-year MdAPE **37.0%** vs
  proforma 40.9%, wins 62.6% (p=0.0002, survives ×5 Bonferroni); **ridge** (+pay/vac/corner-
  light/traffic/cohort): mature MdAPE **37.2%** vs coldstart 45.7% / proforma 58.5%.
  The era-aware anchor was a dud (41/112 coverage, adds nothing) — kept as a documented
  negative result.
- `panel_analysis.py` — 862-site validation: the raw model is **unbiased at scale**
  (ratio 0.99–1.05, mature MdAPE 32.9%, Y1–Y5 28–34%, ρ 0.46–0.52); calibration adds only
  ~1pt; local-median naive baseline is 36.3% — the ~30% MdAPE information ceiling is real.
  The 0.78-ratio over-prediction seen on the proforma cohort is that cohort's oddity.

Deployable recipe ("super"): ridge level when capacity factors are known (else panel plateau),
× Model-3 ramp × per-year median debias; report P10–P90. **Shipped as an artifact:**
`proforma/artifacts/ensemble_super (model 5)/super_ensemble_v1.joblib` (model card in that folder's
README), loaded by `proforma/models/super_ensemble.py` (`predict_site_super`), fitted by
`ensemble/fit_super_artifact.py`. Runtime inputs: lat/lon + open year required; pay stations,
vacuum slots, lot type, traffic count optional (the 4 inputs buy mature MdAPE 33.4→29.6 and
within±20% 34→39, nested-LOSO ablation, n=70). smoke.sh passes (1e-9) after adding the module.
Exposed in the Streamlit drop-pin panel as **"Model 5 — SUPER"** (radio + the 4 optional site
inputs; manual plateau override still wins; verified via AppTest drive — all paths exception-free).
Model cards for every model live under proforma/artifacts/<model>/README.md.

Ceiling-hunt verdict (6-agent campaign, 2026-07): Council covariates (StreetLight traffic,
competitors, demographics; 99.3% join) add NOTHING at 862 scale — zero of 25 covariates
survive FDR on plateau residuals; the plateau already absorbs them. Outliers are not the
lever (only ~14% of the worst decile are reporting artifacts; robust losses buy ~1pp via
bias correction; a 4.9% data-quality screen buys ~2pp of denominator, labeled as such).
Rich location features: only 20–40km market aggregates carry small signal (~1pp, blend-only).
**Post-open Bayesian updater is the real lever: mature MdAPE 33→28 with 3 observed months,
→23 at 6 months (within±20% 32→43%), Y2-from-6-months MdAPE 20.4% (ρ 0.83); τ≈5 — six real
months ≈ the whole ex-ante model.** Literature check: no published ex-ante new-site benchmark
beats ~30% (best: gravity-model holdouts ~10% within an established chain; Kaggle new-restaurant
~35–45%; industry capture heuristics span ±2x) — 32% ex-ante is at best-practice level.

Benchmark artifact (charts): https://claude.ai/code/artifact/f75cc9a5-8755-468f-82c7-9d450ffd8911

## Gotchas

- `entrance_stack_up`'s low KW p is a trap: non-monotone (deepest stack is *worst*) and its
  top cell is a double-ticked extraction artifact ("20 - 15 / Less Than 10", n=6).
- Small top cells: 3+ pay stations and >20 vacuums are n=6 each — direction solid, magnitudes
  anecdotal.
- Causality untested: operators may add pay stations where they expect volume.
- 2020–21 openers backtest worse (0.61 vs 0.76) — COVID era + older template vintages;
  treat open-cohort as a confounder.
- 58 PDFs in old-proforma-data/ were never parsed — a future extraction could grow n by ~50%.
- The notebook is generated: edit `code/make_notebook.py`, never the .ipynb. Every output
  cell must be followed by an Insights markdown cell (repo skill
  `.claude/skills/output-insights`).
