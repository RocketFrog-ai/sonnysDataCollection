# Huff Model Backtest — context for whoever (human or agent) picks this up next

## What this is

A first-pass backtest of a Huff gravity/spatial-choice model for predicting car-wash site
volume, run against 69 real Sonny's sites where we have both (a) an old underwriting proforma
and (b) real, actual operating history. It also benchmarks the Huff model head-to-head against
the old proforma's own formula (already ported to `app/pnl_analysis/modelling/bosch_forecast.py`
in a separate, earlier piece of work — see `experiments/bosch-prediction-api/`).

**Bottom line up front**: Huff-lite modestly outperforms the old proforma formula (median error
~48% vs ~58%), but neither is production-ready. The advantage is concentrated in
competitor-dense markets; for sparse-competition sites the old formula is roughly as good or
slightly better. Full results and caveats below — read the whole thing before trusting a number
out of context, several of these numbers looked much better or worse before a second look.

## How this came to exist — the chain, in order

1. A meeting asked for an API reproducing Rafal's proforma Excel formula ("Bosch prediction").
   That work is in `experiments/bosch-prediction-api/` — fully separate, already shipped as
   `POST /v1/pnl_analysis/bosch-forecast`. Not this folder's concern, but this folder reuses its
   formula knowledge (`SITE_FACTOR_WEIGHTS`, the site-score bounds) and its parsing code lineage.
2. A doc (`ss1.docx`, workspace root) records a prompt-and-response session proposing a **Huff
   gravity model** instead: `P(i→j) = (A_j^α / D_ij^β) / Σ_k(...)`, calibrated against known
   vendor-site volumes, to answer questions the old formula structurally cannot (share capture,
   competitor pull, promo/membership dynamics). A companion `.jsx`
   (`carwash-701-w-ridge-rochester.jsx`) is Fable 5's from-scratch implementation of that idea
   for one prospect site (Rochester, NY), by hand-collecting competitor data.
3. We proved the old proforma formula and the Huff model are **not mathematically related** —
   the old formula has no distance term, no competitor sensitivity (`∂Volume/∂A_competitor ≡ 0`),
   and no market-share normalization. They share only an input vocabulary (visibility,
   accessibility, demographics, traffic), not a functional form.
4. To actually backtest Huff, we needed real historical sites with (their original proforma
   inputs) + (what they actually did). That required joining the old-proforma dataset to the real
   operating panel — the **"join key problem."** Investigation found no ID-based join is possible
   (`salesforce_id` has no counterpart on the panel side), but an **address-fuzzy-match pipeline
   already exists** at `experiments/old-proforma-analysis/code/match_panel.py` and had already
   been run once against the full 179-file `old-proforma-data/` set.
5. The user redirected to a **different, curated folder**:
   `experiments/old-proforma-analysis/n70-final-considered/` — 70 proforma workbooks plus a
   `MANIFEST.csv` (dated notably more recent than the workbooks) that **already contains**, per
   site: `match_client_name`, `match_state`, real `actual_mature_wash_mo` (ground truth),
   `model5_forecast_mo` (an existing forecast of unconfirmed provenance — not used in this
   backtest, deliberately; see Limitations), `n_neighbours_20km`, and a `grounding` tier
   (`grounded`/`thin`/`isolated`) already derived from neighbor density.
6. `match_client_name`/`match_state` isn't a precise key (name collisions possible for chains).
   **This folder's step 1** resolves it to an exact `client_id::site_id` against the real panel
   (`proforma.pnl.data.load_panel()`) by normalized-name + state matching: **69 of 70 resolved
   uniquely**; 1 (`Seaford Car Wash, NY`) is genuinely ambiguous — two panel entries at the
   *identical* lat/lon with different `client_id`s, different `op_start`, different `n_obs` —
   almost certainly a duplicate/ownership-change data-quality issue in the panel itself, not
   resolvable from the address alone. Left unresolved; excluded from the backtest (n=69, not 70).

## The model (`02_huff_backtest.py`)

Single demand-centroid simplification (`ss1.docx` explicitly names this as an acceptable
starting point, to later be replaced by real SiteWise block-group zones):

```
A_j = 1 + 9 * (S_j - S_min) / (S_max - S_min)      # site's own proforma score -> [1,10]
       where S_min=-0.425, S_max=1.5 are the theoretical bounds of the 10-factor weight sum
A_unknown = median(A_j over the 69 known sites)     # real competitors with NO proforma on file
own_term  = A_j^alpha / d0^beta                     # d0 = 0.1 mi nominal self-distance
comp_term = sum over real neighbours within 20km( A_k^alpha / dist_km^beta )
Share_j   = own_term / (own_term + comp_term)
Volume_j  = k * traffic_j * Share_j
```

`(alpha, beta, k)` calibrated by nonlinear least squares in **log-space** (wash volumes span
41–24,086/mo — plain least squares would be dominated by the largest sites) against each site's
real `actual_mature_wash_mo`.

**Where each input comes from:**
- `S_j` (site score), `traffic_j`, and the old-proforma comparison prediction (`vol_monthly_y5`)
  are extracted directly from each site's own `.xlsx` in `n70-final-considered/`, by **reusing**
  `experiments/old-proforma-analysis/code/extract_proformas.py`'s `process_file()` — not
  reimplemented.
- Real neighbours (who's actually within 20km, and their own `A_k` if they're also one of the 69
  known sites) come from `proforma.pnl.data.load_panel()`'s site table + `haversine_km`.
- Unknown neighbours (real competitors near a site, but with no proforma on file) get a **flat
  placeholder**: the median `A_j` across the 69 known sites. This is the single biggest modeling
  simplification here — see Limitations.

## Results (as of this run — re-run to refresh if the underlying data changes)

| | n | MAPE | **MdAPE** | R² | median(pred/actual) |
|---|---|---|---|---|---|
| Huff-lite | 69 | 275.0% | **47.5%** | −0.044 | 0.90 |
| Old proforma (Year 5) | 69 | 468.7% | **58.1%** | −0.266 | 1.41 |

By grounding tier (MdAPE, the reliable number here — see below for why MAPE is misleading):

| Tier | Huff-lite MdAPE | Old-proforma MdAPE | Winner |
|---|---|---|---|
| grounded (n=39) | 50.8% | 72.2% | Huff-lite, clearly |
| thin (n=16) | 51.9% | 42.0% | **old proforma**, slightly |
| isolated (n=14) | 24.7% | 33.9% | Huff-lite, but margin is small |

**Read MdAPE, not MAPE, as the headline number.** MAPE (mean) is wildly distorted by a handful
of outliers in a sample this small and skewed — it initially made the `isolated` tier look like a
catastrophic model failure (889% MAPE) when the *median* isolated site is actually the
**best**-predicted group (24.7% MdAPE). One pathological site (see next section) was responsible
for most of that gap.

## A real data-quality finding, not just a model-accuracy one

The worst single miss (predicted ~4,500–5,000/mo, actual 41/mo) is `splashcarwash_000318::1`.
Pulling its full monthly history from the real panel (`tot_wash_count`) shows it **used to do
5,000–10,000+ washes/month** (all-time average 5,545, peak 10,121) and **collapsed to near-zero
only very recently** (last 3 months averaging 0.3/mo). `MANIFEST.csv`'s `actual_mature_wash_mo`
captured this site's *current, dying* state, not its historical mature performance — and its
`actuals_suspect` flag is `False`, so it wasn't caught.

**This means the ground truth for at least one site is misleading for backtesting a
site-selection model** — no site-selection formula, old or new, should be expected to predict a
subsequent business collapse. **The other 68 sites have not been checked for the same issue.**
A spot-check of 5 sites (see chat history / re-derive similarly) found the other 4 matched a
trailing-window average of real `tot_wash_count` reasonably well, so this is likely not
widespread, but it hasn't been systematically verified.

## Limitations — read before trusting any of the above further

1. **`alpha` does not converge.** Left unconstrained, it runs to whatever numeric bound is set
   (tested up to 12; kept climbing). It is now hard-bounded to `[0.5, 2.5]` — the range `ss1.docx`
   itself cites for convenience retail — as a deliberate methodological choice, not a real fit.
   `beta≈1.0` and `k≈0.24` are stable regardless of where `alpha` lands; those two are the
   actually-identified parameters. With one crude attractiveness feature and 69 points, 3 free
   parameters is likely over-parameterized.
2. **Real sites' attractiveness only spans `[5.91, 9.30]` of the theoretical `[1,10]` range** —
   built sites are self-selected to not be terrible, so the signal available to separate "good"
   from "bad" sites is compressed. This is *why* alpha can't be pinned down (point 1).
3. **Unknown-neighbour attractiveness is a single flat median placeholder** for every real
   competitor without a proforma on file. No sensitivity analysis has been run on how much this
   specific assumption drives the result — that was proposed as necessary (vary it across the
   empirical range, see how much predictions move) but not yet executed.
4. **No true market size `M`.** The model uses each site's own `traffic` count directly, scaled
   by a single global fit constant `k`, instead of the household/vehicle/wash-frequency market
   model `ss1.docx` describes — because total households per trade area isn't available from the
   proforma (only *average household size*, not a total count) and no Census data was pulled in.
5. **`model5_forecast_mo` in `MANIFEST.csv` was deliberately not used** — its provenance was
   asked about and never confirmed, so this backtest computes its own old-proforma comparison
   figure (`vol_monthly_y5`, from the same extraction) instead of trusting an unidentified column.
6. **Ground-truth data quality has not been systematically checked** — see the section above.
   `actuals_suspect` in `MANIFEST.csv` cannot be assumed reliable (it missed the collapsed-site
   case). Anyone extending this should pull each site's full monthly trend and look for similar
   late collapses, acquisitions, or reporting gaps before trusting per-site errors.
7. **Single demand-centroid simplification** — each site is treated as one point, not zones of
   real demand around it (`ss1.docx`'s own stated next step, never done here).
8. **All data-file access in this pipeline is read-only against `n70-final-considered/` and the
   real panel** — nothing here writes back to `MANIFEST.csv`, `old-proforma-combined.csv`, or
   the panel itself. Every script in this folder is safe to re-run.

## How to run this

Needs `pandas`, `numpy`, `scipy`, `openpyxl`, `xlrd`, `plotly`, `kaleido` — this repo's `sonnys`
conda env has all of these (`environment.yml`); this work was actually developed/tested in an ad
hoc venv on a machine without that env available, not the project's own env — re-verify in `sonnys`
before trusting new numbers.

```bash
cd sonnysDataCollection   # repo root
python experiments/huff-model-backtest/01_resolve_site_keys.py   # -> resolved_sites.json
python experiments/huff-model-backtest/02_huff_backtest.py       # -> backtest_results.csv
python experiments/huff-model-backtest/03_plot_backtest.py       # -> backtest_scatter.png
```

## Suggested next steps, roughly in priority order

1. Systematically check all 69 sites' monthly trend for the same late-collapse issue found in
   `splashcarwash_000318::1` — this could materially change which sites are fair backtest cases.
2. Resolve the `Seaford Car Wash, NY` ambiguity (two panel entries, identical coordinates) — this
   is a real panel data-quality question, not a script bug.
3. Run the unknown-neighbour attractiveness sensitivity analysis proposed but never executed
   (sweep the placeholder value across the empirical range, see how much predictions move).
4. Investigate whether a richer attractiveness feature (beyond the single proforma site-score)
   would let `alpha` actually converge — e.g. separating the 10 site factors into distinct terms
   rather than one collapsed score.
5. Find out what `model5_forecast_mo` in `MANIFEST.csv` actually is — it may be a useful third
   comparison point, or may reveal this backtest infrastructure already exists elsewhere.
