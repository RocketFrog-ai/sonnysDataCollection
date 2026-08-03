# Conclusions — the proforma evidence pack

Built for a CIO review. **One section per conclusion**, in one Streamlit app and one notebook.

```
conclusion/
  demo/app.py              thin entry — page config, shared CSS, section switch
  demo/ui.py               palette, Plotly layout, CSS, table + callout helpers
  demo/section_tunnel.py   ① Streamlit rendering
  demo/section_proforma.py ② Streamlit rendering
  demo/section_campaign.py ③ Streamlit rendering
  demo/tunnel_data.py      ① the maths — Streamlit-free
  demo/proforma_data.py    ② the maths — Streamlit-free
  demo/campaign_data.py    ③ the maths — Streamlit-free
  notebook/conclusions.ipynb    sections ① ②, static plots + written insights
  notebook/book_v4_revolt.ipynb section ③'s working notebook — the source `campaign_data.py`
                                was ported from
  data/                    section ① inputs
```

The app and the notebook import the **same** analysis modules, so they cannot report different
numbers. Adding section ④ is: one `*_data.py`, one `section_*.py`, one line in `app.py`'s `SECTIONS`,
and one block appended to the notebook.

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
