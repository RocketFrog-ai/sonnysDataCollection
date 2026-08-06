# Conclusions — the proforma evidence pack

Built for a CIO review. **One section per conclusion**, in one Streamlit app and one notebook.

```
conclusion/
  demo/app.py              thin entry — page config, shared CSS, section switch
  demo/ui.py               palette, Plotly layout, CSS, table + callout helpers
  demo/section_tunnel.py   ① Streamlit rendering
  demo/section_proforma.py ② Streamlit rendering
  demo/section_campaign.py ③ Streamlit rendering
  demo/section_demographics.py ④ Streamlit rendering
  demo/section_competition.py  ⑤ Streamlit rendering
  demo/section_cluster.py  ⑥ Streamlit rendering
  demo/tunnel_data.py      ① the maths — Streamlit-free
  demo/proforma_data.py    ② the maths — Streamlit-free
  demo/campaign_data.py    ③ the maths — Streamlit-free
  demo/demographics_data.py    ④ the maths — Streamlit-free
  demo/competition_data.py     ⑤ the maths — Streamlit-free
  demo/cluster_data.py     ⑥ the maths — Streamlit-free
  notebook/conclusions.ipynb    sections ① ②, static plots + written insights
  notebook/book_v4_revolt.ipynb section ③'s working notebook — the source `campaign_data.py`
                                was ported from
  data/                    section ① ⑤ inputs
```

The app and the notebook import the **same** analysis modules, so they cannot report different
numbers. Adding a section is: one `*_data.py`, one `section_*.py`, one line in `app.py`'s `SECTIONS`,
and one block appended to the notebook. (The notebook currently carries ① and ②; ③, ④, ⑤ and ⑥ are
app-only so far.)

## Run it

```bash
conda activate sonnys
streamlit run conclusion/demo/app.py          # from the repo root
```

The notebook runs in the same env. There is no `jupyter` in `sonnys` — open it through the VS Code
notebook UI. Charts are Plotly, not matplotlib, because `sonnys` has plotly and does **not** have
matplotlib.

---

## Section ① — Tunnel length: are we over-building?

**One input file**, `conclusion/data/tunnel_length_with_wash.csv`, and **only the 39 sites with three
or more years of trading** (of 78 in the file). Nothing is joined in from anywhere else.

Two things that are normally assumed are **measured** here: the peak throughput is a real hourly car
count, and the capacity rule is **Sonny's own** — `tunnel feet = year-5 peak hourly volume` (one foot
per car/hour, no +20).

| | |
|---|---|
| Year 1 → year 2 ramp | **59% → 98%** of eventual volume, then flat |
| Calling year 5 from year 1 | within **19%** using the ramp, vs **41%** taking year 1 at face value |
| Tunnel length vs washes | +1,337 washes/ft, **R² = 0.24** — real but weak |
| Capacity used, typical day | **24%** |
| Capacity used, busiest hour ever recorded | **64%** median; estate max **98%** — **no site has ever exceeded its rating** |
| Sites under half their rating | **28%** (11 of 39), median **72 ft (22 m)** spare = **58% of the tunnel** |

**Units matter.** `tunnel_length_actual_ft` is whole **metres × 3.2**; the true factor is 3.28084, so
that column understates length by ~2.5%. We recover the metres and convert properly.

**One row per operating year, not per calendar year.** A site that opened in October has two
calendar rows — the stub of its opening year and most of the next — that both fall inside operating
year 1. Left alone that draws the same site twice in one facet at two different heights and makes
its "path" look like it moved. `cohort_peaks()` folds the fragments coverage-weighted
(`rate = Σ washes ÷ Σ coverage`) before anything else. 20 of 306 site-years needed it; 326 rows → 301.

**Facets are derived, not fixed.** The old `["Year 1", "Year 2", "Year 3", "Year 4+"]` buried
operating years 4–9 in one panel — exactly the range where "does under-use close with age?" is
answered. Years now get their own facet while they hold ≥10 sites (**Year 1–6**), and the thin tail
folds into **Year 7+**, where each site appears once at its most mature year.

**The site key is `client_id + site_id`.** The picker maps label → `site_key` via
`td.site_picker()`. An earlier version keyed the map on the site's *name*, so two sites sharing a
name would silently collapse. 78 rows carry 78 names but only **64 client_ids** — one client runs up
to six sites — so name-uniqueness was an accident, not a property.

**No CAPEX.** This file carries no cost data, so the finding stops at feet.

## Section ② — Proforma backtest: how close did the projection land?

68 of the 70 mature matched sites from `conclusion/data/n70_backtest_dataset.csv`
(2 collapsed sites excluded — closures, not forecast misses).

| | |
|---|---|
| Proforma error at maturity | **58.5%** median absolute error |
| Proforma bias | **1.37×** actual; over-projects on **72%** of sites; **4.33×** at p90 |
| Does it improve with age? | No — **45.9%** in year 1, **54.8%** by year 5 |
| Model 5 on the same sites | **29.4%** error at **0.98×** bias — roughly halves the miss |
| Is that luck? | Closer on **49 of 68** sites, sign-test **p = 3.6e-04** (leave-one-site-out) |
| Which proforma inputs predict? | **4 of 14** survive FDR — pay stations, type of site, free vacuum slots, and the site score they drive. All capacity; demographics and competition are flat-to-negative |

| Recommended vs built tunnel | The formula length tracks the proforma's own volume promise at **0.997** but the tunnel actually built at **0.22**; median **29 m** recommended vs **38 m** built, **68%** built longer |

You cannot fix this with a haircut: the proforma is off-centre **and** 2.5× wider than the model, so
a flat correction that fixes the median still leaves the p90 site 3× over.

### Traffic — where that error actually comes from

The old sheet is, arithmetically, **one multiplication**: traffic count × a near-fixed capture rate.

| | |
|---|---|
| Share of the proforma's own projection explained by traffic alone | **90%** (98% unique to traffic once the scores are in); the nine scored boxes + the whole demographic block explain **0.9%** |
| Share of **actual** volume explained by traffic alone | **2.2%**; rho **+0.22, p = 0.07** — not significant |
| Elasticity of washes to traffic — proforma | **0.95** → doubling traffic buys **+93%** |
| Elasticity — **actual** | **0.25** [95% CI −0.14, +0.65] → doubling buys **+19%**. Slope = 1 rejected (**p = 0.0005**); slope = 0 **not** rejected (p = 0.23). Robust variants (Theil–Sen 0.34, outliers dropped 0.37, model-free quartile ratio 0.36) put the range at **0.25–0.37**; the drawn line is the cautious end and the app says so |
| Elasticity — Model 5 / cold-start | 0.33 / 0.15 — both far closer to reality than the sheet |
| Capture rate assumed vs achieved | assumed **1.17%** median, p90÷p10 = **1.4×**; achieved **0.79%** median, p90÷p10 = **8.2×**. Only **19 of 68** sites beat their own assumption; median site hit **73%** of it |
| Over-projection by traffic quartile | **0.93× → 1.42× → 1.61× → 2.03×** (rho +0.37, p = 0.0022). On a quiet road the sheet is right; on a busy one it projects double |
| Traffic **speed** | nothing. Kruskal across 4 bands **p = 0.38**; the sheet's speed points correlate **+0.06 (p = 0.61)** with volume. Holding traffic constant, +10 mph is worth −19% but **p = 0.20**. Directionally agrees with the sheet, statistically indistinguishable from noise on 68 sites |

**The 58% headline error has an address**: it is concentrated on busy roads and caused by the fixed
capture assumption. The slope has to change, not the level — cutting every projection by a third
would start under-projecting the quiet sites, which are already at 0.93×.

**A chart deliberately not shown.** Bucketing sites by traffic and plotting the median capture rate
is circular — capture is washes ÷ traffic, so it must fall as traffic rises whether or not anything
real is happening. The elasticity, and the per-site assumed-vs-achieved comparison, replace it.

**The two sections deliberately do not share data.** Each reads exactly one file and they are never
merged — the 11 sites that appear in both carry tunnel lengths that disagree by up to 7 m, so any
join would silently pick a winner.

Reference working notebook: `experiments/old-proforma-analysis/proforma_backtest.ipynb`.

## Section ③ — Campaigns: do promotions work, or is it the opening ramp?

One input file, `proforma/data/opex/opex-data.csv` — the monthly P&L panel, 162 sites. Ported from
`conclusion/notebook/book_v4_revolt.ipynb`; only the cells that actually execute there are
reproduced (the commented-out ones are not).

**There is no campaign table in this business.** A campaign is *inferred*: a month where
`cogs + expenses` exceeds the site's own trailing 6-month mean by 1.2×, with consecutive such months
merged. 264 spike months across 97 of 162 sites → **184 campaigns**. That inference is the weakest
link in the section and §4.3 re-runs the estimate on a different trigger to price it.

The section is built as an argument that turns on itself. Sections 1–3 are the raw event studies:

| | |
|---|---|
| Revenue, months +1 to +3 vs the site's own past | **+16.7%** median |
| Membership washes / retail washes, same window | **+19.4%** / **−7.0%** — reads as retail→member conversion |
| Neighbours within 20 km, retail washes | **−13.6%**; 39% of neighbours see revenue fall |
| Focal share of its own 20 km market | **28.5% → 30.4%**, but median per-event gain ≈ **0 pp** |
| Median incremental campaign spend | **\$29,894** (77% of campaigns last one month) |

Section 4 rebuilds all of it as a difference-in-differences against matched sites that ran **no**
campaign over the same calendar months, matched on census region and site age:

| | |
|---|---|
| Seasonality — the objection that was raised | **rejected**: accounts for **−0.7 pp** of the lift |
| Pre-campaign gap, all campaigns | **+5.9%** [+1.4, +11.1] — a clean design must read **zero** here |
| Placebo in time (campaign date moved back 9 months) | **fails**: reports a **+18.6%** "lift" on a date where nothing happened |
| The actual confound | the **opening ramp** — the pre-trend drains monotonically as young sites are excluded, which seasonality or market effects could not do |
| Headline effect, sites 18+ months old | **+7.1%** [+0.2, +13.4] on 17 events — the lowest age bar whose placebo passes |
| New membership sign-ups | **−0.8%** [−5.5, +4.3] — **no effect**, against a naive +14.9%. The conversion story does not survive |
| "Stealing" from neighbours | neighbours **−24.8%** retail washes, but unstealable sites 100+ km away **−13.8%** — only **−12.7%** is proximity |

Section 5 is the picture that explains it: every site re-indexed to months since opening. Revenue
and market share climb steadily through the first 18 months with no campaign required, and most
campaigns in this data are run inside that window.

The honest summary is that the direction survives every filter and the magnitude does not: ≈ **+7%**,
on an interval that only barely clears zero, not the +21%–34% the naive sections report. What would
settle it is **staggered rollouts** — one operator promoting at some sites but not others in the
same market and month.

**Section ③ shares no data with ① or ②.** It reads its own single file, as they do.

---

## Section ④ — Demographics: does the market explain the wash?

The premise every site-selection proforma rests on — score the trade area, and the score tells you
what the wash will do — tested against **1,263 sites that traded all twelve months of 2025**, in 54
states, on **31 market measures** each.

Cohort: `historical_data_5yrs_monthly.csv` joined to `historical_data_sitewise.csv` on
**`client_id_1 + site_id`** (the number-first client id is the correct key; the name-first column in
the same file matches more rows only because the monthly panel carries both styles). The twelve-month
gate means every annual total is a real sum — no site looks small for having opened in August.

| | |
|---|---|
| Strongest of 31 measures vs total washes | **+0.11** (population growth) — explains **1.3%** of the differences between sites |
| Measures under ±0.10 | **30 of 31** |
| Within one operator's own portfolio | every measure collapses toward zero; several **flip sign** |
| Model given all 31 measures, scored on **unseen states** | **R² = −0.02 to −0.09** — no better than quoting the estate median, on either a boosted tree or a ridge |
| Highest vs lowest fifth, **median across all 31 measures** | **1.10×**; 15 of 31 move volume under 10% either way; the widest in the set (mass merchants nearby) is **1.25×** |
| Highest vs lowest fifth on **membership customers** | **4.7×**, monotonic every step |
| Who operates the site | **39%** of the differences (leave-one-out, 82 operators with 3+ sites) |
| Median of the 10 nearest neighbours | **5.2%** — small, but positive, which demographics are not |
| Across the 30 rankable states, population vs volume | **+0.00**; membership share vs volume **+0.41** (p = 0.026) |

### The correlation grid, and the one place it is not flat

31 measures × 3 wash types = 93 cells. Strongest is **0.157**; **84%** sit under 0.10. But the
columns differ in a way that makes sense: the typical measure correlates **0.07** with retail washes
and only **0.02** with membership washes. The $150–250k income bands run **+0.15** with retail and
**−0.02** with membership — the sign flips. Only 74% of measures agree on direction between the two.
**If the market matters at all, it matters to the drive-up half — the half that is shrinking.**

Split by census region (retail washes), the national flat line turns out to hide something:

| Region | Sites | Strongest | Clearing FDR | Typical \|rho\| | …within state | Independent factors | Held-out R² |
|---|---|---|---|---|---|---|---|
| South | 823 | grocery +0.12 | 7 of 31 | 0.05 | 0.05 | 9 | **−0.163** |
| West | 190 | grocery +0.34 | 16 of 31 | 0.17 | 0.17 | 9 | **−0.188** |
| Midwest | 144 | 1-vehicle HH +0.25 | 8 of 31 | 0.18 | 0.09 | 8 | **−0.363** |
| Northeast | 95 | $100–125k HH +0.36 | 15 of 31 | 0.22 | 0.24 | 7 | too few states |

**The regions are wildly unequal (823 vs 95), so the raw grid is not comparable.** Corrected three
ways — permuting wash counts inside each region (destroys the relationship, keeps the 31 measures as
collinear as they really are), subtracting the resulting floor, and cutting every region to 95 sites:

| Region | Sites | Typical \|rho\| | Noise floor at that n | Excess | Balanced to n=95 | Permutation p |
|---|---|---|---|---|---|---|
| South | 820 | 0.055 | 0.025 | **0.030** | 0.083 | 0.030 |
| West | 186 | 0.168 | 0.052 | **0.116** | 0.172 | 0.003 |
| Midwest | 142 | 0.170 | 0.056 | **0.114** | 0.165 | 0.000 |
| Northeast | 95 | 0.220 | 0.070 | **0.150** | 0.220 | 0.000 |

Small regions really do start **3× further from zero for free** — but it does not explain the gap.
The excess is still 4–5× larger outside the South, and equalising the samples preserves the ordering
(the South's measured correlation *rises* from 0.055 to 0.083 when cut to 95 sites, which is the
noise inflation itself). The permutation p is an omnibus test on the whole grid, so it sidesteps the
"31 measures are really 7–9 things" problem; all four regions beat their own noise.

Four qualifications, all in the app:
1. Outside the South the relationship is **real, survives holding the state fixed, and survives the
   sample-size correction**. The section says so rather than burying it.
2. The South — biggest sample, best measured — has **almost none of it** (excess 0.03). The regions
   where something appears are the three where we hold the fewest sites.
3. "15 of 31 cleared significance" overstates it: the 31 measures are only **7–9 independent
   things** (~40% of their variance in one component). It is *market size*, counted many times.
4. **None of it forecasts.** A model trained inside one region and scored on states it has not seen
   is negative in every region.

**The modelling read-through:** anchor a new-site forecast on **how comparable nearby sites actually
trade** and on **who will run it**, not on a trade-area score. This is not a claim that markets are
irrelevant to a car wash — it is a claim that these measures, at this resolution, cannot rank two
candidate sites. Every site in the panel was already chosen by someone who believed in these
measures, which compresses the range; that is also exactly the range a new site is picked from.

Zero-population and zero-traffic cells (14 and 19 sites) are read as **missing**, not as real
deserts — they are failed geocodes, and would otherwise anchor the bottom of every ranking.

**Section ④ shares no data with ①, ② or ③.**

---

## Section ⑤ — Competition: somebody opens nearby. What happens?

One input file, `conclusion/data/historical_data_5yrs_monthly.csv` — the monthly wash panel, 2,103
sites, Jan 2020 → Jun 2026, each with `operational_start`, coordinates and state. The unit is an
**event**: a site opens, and every already-trading site within the radius is a neighbour exposed to
it. **751 openings → 2,239 (entrant, neighbour) pairs**, against the archive's single-nearest-
neighbour run (`archive/hypothesis-testing/interaction_outputs_nochem_v2/`, 85 pairs, no
counterfactual).

**Every number is measured against a counterfactual**, because §③ is the cautionary tale in this
same pack: the same before/after change is computed for every *untouched* site — no opening within
the radius anywhere in the window — and the neighbour is scored against the median of untouched
sites in its own census region, **age bracket** and calendar months. 99% of pairs match at that
tightest level.

| | |
|---|---|
| Typical neighbour, raw before/after | **−1.6%** washes |
| Untouched sites over the same months | **+3.2%** |
| **Neighbour vs the counterfactual** | **−3.2%**; **60%** of neighbours lose |
| Within 1 mile / 1–2 mi / 2–3 mi / 3–5 mi / 5–10 mi | **−9.4% · −8.5% · −3.6% · −2.9% · −3.0%** |
| Shape of it | a **level step at month 0**, flat for the 12 months before, no recovery in the 12 after |
| The pair together (neighbour + entrant) | **+59%** on what the neighbour alone was doing |
| Pure cannibalisation (neighbour loses, pair does not grow) | **9%** of pairs; market expansion **57%** |

**Two-body distance is not within-market distance.** 465 openings land on two qualifying neighbours
at once, which holds the market, the calendar and the entrant fixed and varies only *which one is
closer*. The nearer neighbour comes out **+0.2 pp** from the further one and is the worse of the two
in **49%** of events — a coin flip; restricting to the 85 events where the nearest is under two
miles does not change it (**+0.1 pp**), and there the further neighbour, a median 4.3 miles away, is
down **−8.1%** — as much as the nearer one. So the two-body gradient is substantially *which markets
get built into*, not *how many metres away*. The exposure is the market, and every site in it is
exposed.

**The membership moat does not survive the counterfactual.** Raw, retail washes at a close
neighbour fall **−9.5%** and membership washes only **−1.0%**. But untouched sites *grew* membership
**+5.0%** over the same months against **−1.6%** retail — so against the counterfactual the
membership book is down **−6.0%** to retail's **−7.9%**. It is hit almost as hard; the loss just
arrives as growth that never happened.

**The new site pays more than the neighbour does.** An entrant with nobody inside three miles opens
at **4,971** washes a month; one with two or more opens at **3,176** — about **36% less** — and the
gap is still there at months 12–24 (**7,671** vs **5,717**, −25%). The neighbour gives up single
digits. On 42 crowded entrants of which 18 have matured, so the direction is the finding and the
size is not.

Four exclusions, each removing a way to manufacture a fake effect, together worth ~0.2 pp on the
headline: pairs under **0.2 mi** (operator handoffs — the same wash under a new `client_id`),
neighbours under **500 washes/month** (stubs; a few site-months are negative), neighbours with **two
or more months under 5%** of their own past (they closed — the series hits a literal zero and stays,
verified by eye), and entrants under **250 washes/month** (openings that never opened; 153 excluded).

`operational_start` equals the site's first panel month for **every** site, so the 348 sites stamped
2020-01 are "open by then", not "opened then" — they are used as neighbours, never as entrants.

The **case explorer** tab puts all 2,239 pairs in front of the reviewer individually: filter by
state, distance, outcome; then both sites' actual monthly wash counts with the opening month and
both windows marked, the relative geometry with 1/3/5/10-mile rings, and both addresses, coordinates
and opening dates. There is no basemap on purpose — at two miles apart, a country-scale Scattergeo
is a blank field with two dots, and this app has no tile layer. It defaults to the **closest** pair,
not the biggest loss: the extreme tail is where sites that were quietly winding down anyway live,
and no filter separates a gradual slide to near zero from competition without also deleting the
finding. A single case is not an attribution.

**Section ⑤ shares no data with ②, ③ or ④** — it reads the same monthly panel as ⓪, which is the
panel, not a section-specific extract.

---

## Section ⑥ — Operator clusters: pick a multi-site operator, get their whole story

**One input**, the monthly panel (`conclusion/data/historical_data_5yrs_monthly.csv`, byte-identical
to `proforma/data/panel/main-data-v2-stitched.csv`). Nothing is joined in.

The reader picks a **company**, not a place — and only companies that **actually clustered**. Two
filters, applied in this order and defaulting to 15 miles / 3 sites / 1 complete year, cut the 221
multi-site operators to **42**:

1. a site needs **whole calendar years** behind it, or it has no year-on-year to compare;
2. a **location** needs **3+ sites** in it, or it is a pair, not a cluster — and markets below the
   threshold are dropped whole, along with operators left with none.

The year filter runs *first*, so a market that only reaches three sites by counting one that opened
last month does not qualify.

The reader then picks **one of that operator's localities** — there is deliberately **no
whole-estate roll-up**, because a national view answers "how big are they" and this section exists
to answer "what did they do to one town". Everything shown is that operator in that town: their
sites on a zoomed tile map with **3-mile trade-area circles**, a sitewise table whose addresses link
straight to Google Maps, washes **month on month** (total, plus a site × month heatmap) and **year
on year** (total, plus one small-multiple panel per site on a shared scale), and then **who else
washes cars there**. Distances are in **miles** throughout, including the grouping slider.

### Section 6 — the competition in the same locality

Every other operator in the panel with a site within a slider distance (default 5 miles) of any of
this operator's sites, drawn on the same map in **orange against the operator's blue**, with the
same 3-mile circles. Then their monthly trajectory on **one axis** beside the operator's — not an
index, so the heights compare directly — and what the already-settled rivals did in the six months
either side of each of this operator's openings, against **those same rival companies' sites outside
the locality** over the identical months.

Of 67 localities, 48 have at least one visible rival and 27 support a measurable before/after.
Gate Express in Jacksonville is the clearest: as it added sites 6 → 16, the nearby rivals' response
walked from **+8.1pp to −11.4pp**, median **−8.3pp**, 67% of openings leaving the neighbours worse
off than their own other branches.

**The panel is Sonny's customers only**, which bounds this in one direction and is stated on the
page. A rival we do not sell to is invisible, so a locality can look emptier than it is and the
effect is measured on the *visible* competition — biased toward zero if unseen rivals absorbed the
hit. What is not bounded is the other direction: where a rival is in the panel, these are its real
monthly wash counts, over the same months, in the same few square miles.

**The 3-mile circle is the point.** `historical_data_sitewise.csv` — the same vendor trade-area pull
behind §④ — supplies population, income, traffic and competitors for the catchment around each site,
and those figures sit in the map's hover box next to the wash counts they are meant to explain. The
file records **no radius**, so the 3-mile circle is a stated convention, not something read out of
the data. Where two of one operator's circles overlap, both sites are credited with the same
households. `overlap_fraction()` is the exact circle-circle lens area over circle area — pairwise
against the nearest sibling, not a union across all siblings (no `shapely` in `sonnys`), so it
**understates** where three or more circles pile up.

It joins on **`client_id`**, not `client_id_1`. The file carries ids in two styles split across
those columns and the panel uses both: `client_id` matches 1,988 of 2,077 panel sites, `client_id_1`
only 1,914, coalescing adds nothing, and the two agree wherever both resolve. §④ keys on
`client_id_1` and so drops BlueWave and the other name-first operators from its cohort — left alone
on purpose, since §④'s published numbers hang off that cohort.

| Trade-area double-counting, at a 3-mile radius | |
|---|---|
| Across the 42 qualifying operators (261 sites) | **34%** of counted trade-area population is claimed by two sites at once — 7.5M of 21.7M people |
| The median operator | **67%** of its sites overlap a sibling's catchment |
| Luvcarwash (37 sites, 9 markets) | **92%** of sites overlap; median pair shares **34%**, worst **81%**; **36%** of its 3.8M counted people double-counted |

A competitor "distance" under 0.05 miles is the **site itself** in its own competitor list — true of
76% of the nearest-competitor rows — so the nearest genuine competitor is the first ranked distance
that clears that threshold.

| Estate-wide, at 25 km, 3+ sites | |
|---|---|
| Clustered markets | **110**, across **62 operators** |
| Sites inside one | **424** — 22% of placeable sites, **27% of all washing** |
| Typical site's nearest sibling | **6.0 km (3.8 mi)** |
| Typical market's build-out, first opening → last | **18 months** |
| Neighbours' washes when the operator opens another, vs its sites elsewhere | **−3.0pp** median over 100 openings; 62% negative |

**The catchment overlap is far larger than the wash loss** — sites routinely share 30–90% of a
3-mile circle for a measured hit of a few points. Either the real trade area is wider than 3 miles,
or a second site brings enough new demand to cover most of what it takes; both readings argue
against sizing a site on its circle alone.

**The basemap is CARTO/OpenStreetMap, not Google.** Google's tiles cannot be used as a raster tile
source outside their own Maps JavaScript API, so a Plotly chart cannot legally draw a Google
basemap. Every address in the sitewise table instead links to that exact coordinate in Google Maps,
which needs no API key.

**A coordinate defect this section had to work around, stated in its method tab rather than hidden.**
**100 sites across 27 coordinate points carry a placeholder lat/lon** — one coordinate shared by
several sites of the same operator whose street addresses all differ. BlueWave stamps 21
Houston-area sites on a single point; Buckeye stamps 10 sites spread over six Ohio towns on another.
Their wash data is real, their location is not, and all 100 (5.1% of washes) are dropped before
clustering. Sites sharing a coordinate **and** an address are a different thing — a second tunnel or
an operator handoff — and are kept at a true distance of 0; there are 95.

Section 5's before/after is **descriptive, not identified** — openings inside one cluster are months
apart so their windows overlap. Two guards are load-bearing anyway: an incumbent needs 12 months of
trading (§⓪ puts a new wash at ~98% of eventual volume only by year 2), and both the neighbour and
control sums are balanced across the two windows with the control held to the same settled test.
Without them the control pool reads **+13%** growth that is only the operator's own new sites
elsewhere ramping — which would be charged to the new neighbour as cannibalization. §⑤ is where
entry is estimated properly.

**Section ⑥ shares no data with ①–⑤** — it reads the same monthly panel as ⓪, which is the panel,
not a section-specific extract.

---

## One environment landmine

`st.dataframe` / `st.table` **segfault the Streamlit server on the second script run** in this env
(pyarrow 25.0.0 + pandas 3.0.2 + streamlit 1.58.0). A three-row toy frame reproduces it; the process
dies the moment anyone moves a slider. Two mitigations are in place and both are load-bearing:

- every table is hand-rendered HTML (`html_table` in `ui.py`);
- `app.py` sets `pd.set_option("mode.string_storage", "python")` before any DataFrame exists, which
  takes the crashing `string_arrow._from_sequence` path out of play.

Verified over repeated reruns, section switches and live slider changes with no crash. The real fix
is an environment pin, deliberately not done here because `sonnys` is pinned and `scripts/smoke.sh`
baselines hang off it.
