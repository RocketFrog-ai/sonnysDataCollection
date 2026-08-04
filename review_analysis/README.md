# Review Analysis

A local Streamlit app for exploring customer reviews from `data/final_reviews.csv`
(6,087 reviews across 25 physical locations of a single car-wash chain — "location" means the
`site` column, not `businessName`, which is constant; see `docs/dataset_schema.md`).

The UI is built to match the product dashboard: fixed top bar and left icon rail, an
"All Sites" tab strip, a location/period filter row, and a 3×3 grid of KPI tiles. Every tile
drills through to a per-site breakdown, where expanding a site reveals its individual reviews,
sorted however you ask — including by raw sentiment score.

## Install

```bash
cd review_analysis
python3 -m venv .venv        # optional but recommended
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies (see `requirements.txt`): `streamlit`, `pandas`, `numpy`,
`plotly`, `vaderSentiment`.

## Launch

```bash
streamlit run app/Home.py
```

Opens at `http://localhost:8501`. All data is read from `data/final_reviews.csv` inside this
project folder — no external network calls, no files outside `review_analysis/`.

`.streamlit/config.toml` pins the **light** theme. Without it Streamlit follows the viewer's
OS color-scheme preference and renders the dashboard dark, which fights every color the UI
sets. Don't delete it.

## Pages

### `app/Home.py` — dashboard
Nine tiles, each showing a value, a period-over-period delta, two sub-stats and a mini chart:

| Tile | Reads |
|---|---|
| New Reviews | volume in the period, locations reporting, reviews/day, rating mix |
| Average Rating | period mean, **chain rating across all reviews**, 5-star share |
| **Review Sentiment** | net sentiment (%positive − %negative), positive/negative counts, mean VADER score |
| Positive Reviews | count, share of reviews with text, rating-only count |
| Negative Reviews | count (green when falling), 1–2 star count |
| Owner Response Rate | reply rate, median time to reply, replies outstanding |
| Review Volume Trend | 8-month volume, best month |
| Best Sentiment Location | leader on net sentiment, with sample size |
| Needs Attention | laggard on net sentiment, with negative count |

The **Location** pill filters sites; the **🗓 period** pill switches between current month,
last 3, last 12, and all time. The prior window for the delta is always the same number of
months immediately before the current one. The tab strip switches between the three pages.

**Every tile is a click target across its whole surface**, and each opens the view that
explains it — already sorted and filtered, with the dashboard's period carried across:

| Tile | Opens | Review lens |
|---|---|---|
| New Reviews · Review Volume Trend | Insights, latest period expanded | all, most recent |
| Average Rating | Site Breakdown's **average-rating-by-location chart** | no reviews opened |
| Review Sentiment | Site Breakdown, worst net sentiment first | all, most negative |
| Positive Reviews · Best Sentiment Location | Site Breakdown (that site open) | **positive**, most positive |
| Negative Reviews · Needs Attention | Site Breakdown, worst site open | **negative**, most negative |
| Owner Response Rate | Site Breakdown, slowest repliers first | **awaiting owner reply**, most recent |

The lens is page-level, not per-site: open any *other* location from a negative-reviews drill-down
and you still see only its negative reviews. The single **Sort / Search / Show** row above the
table carries it and is where you change it.

The drill-down also **shows only the columns that tile is about** — the negative tile's table is
Location / Negative (#) / Negative % / Reviews (#), not all nine metrics — and hides locations
with none of what you asked for (4 of 25 locations have a negative review this month, so it lists
4 rows).

## AI insights

Every tile carries a **✨ AI insights** popover, and each drill-down table has one in its header.
It asks gpt-4o (Azure OpenAI) a question about *exactly the reviews that tile counted* — the
negative tile opens with "What are the main concerns customers raise in these negative reviews?",
and the question is editable. Nothing is sent until you press **Generate**; answers are cached for
an hour per selection.

The model only ever reads review text and is instructed to name recurring themes, say how many of
the shown reviews mention each, and quote a short phrase as evidence. **It never produces a
number on the page** — every figure still comes from `metrics.py`, so a wrong summary can't
corrupt a metric.

Credentials live in `.streamlit/secrets.toml`, which is **gitignored**:

```toml
azure_openai_api_key = "..."
azure_openai_endpoint = "https://<resource>.openai.azure.com"
azure_openai_deployment = "gpt-4o"
```

`AZURE_OPENAI_API_KEY` etc. work as environment variables too. Without a key the popover says so
and the rest of the app is unaffected.

### `app/pages/2_📊_Insights.py` — period drill-down
The three-level view. A chart card (**Monthly / Quarterly / Yearly** selector) over grouped
bars with dashed trend lines, then a table whose rows are calendar periods:

```
Jul-26            355 reviews   4.79★   223 pos   22 neu   8 neg   85.0% net   45.4% replied
  └ Big Dan's Kissimmee OBT    32       4.88★    15       2        1       77.8%      0.0%
      └ the 32 reviews themselves, sorted by sentiment score / date / rating
```

Expanding a period lists the locations that reported in it; expanding a location opens its
reviews for that period with the same sort / search / sentiment-filter controls as below.
A period row and a site row are the same `metrics._row_stats` over a different slice, so a
month's totals are exactly what its locations add up to.

### `app/pages/1_📍_Location_Detail.py` — site breakdown
- **Sentiment & Rating By Month** — grouped bars with dashed least-squares trend lines;
  toggle between the positive/negative split and volume-vs-rating.
- **Reviews By Site** — one row per site (reviews, avg rating, Google rating, positive /
  neutral / negative counts, net sentiment, response rate). Every column header sorts;
  clicking it again reverses.
- **Expanding a row** shows that site's reviews with three controls:
  - **Sort by** — sentiment score most positive, sentiment score most negative, most recent,
    oldest, rating highest, rating lowest;
  - **Search text** — substring match on review text;
  - **Show** — all / positive / neutral / negative / rating-only / 1–2 star / awaiting owner reply.

  Reviews render 15 at a time with a "Show more" button, each carrying its stars, date,
  sentiment chip, compound score, text, and the owner's reply.

## Two things about the numbers

**Average rating.** `businessAvgRating` (Google's published per-site rating) was documented as a
float but never actually cast, so it stayed a string and no code could use it. It is now cast,
and the chain-level figure is Google's rating **weighted by each site's own review count** —
not the mean over scraped rows, which weights sites by how many reviews we happened to collect
(Rome contributes 912 rows, Muscle Shoals 17). Both numbers are exposed:
`get_kpis()["avg_rating"]` is the sample mean, `["avg_rating_weighted"]` the real one, and
`["avg_rating_display"]` picks the best available. The per-site table shows both alongside the
capture rate, so a site's rating can be read next to how much of it we actually have.

**Sentiment denominators.** 28% of reviews are rating-only. VADER scores an empty string as
0.0 — indistinguishable from a genuinely even-handed review — so those get a dedicated
`no_text` label and are excluded from every sentiment share. `n_scored` carries that
denominator so it can't be divided by the wrong number.

## Project structure

```
review_analysis/
├── .streamlit/config.toml               # pins the light theme
├── .streamlit/secrets.toml              # Azure OpenAI key (gitignored)
├── app/
│   ├── Home.py                          # entrypoint — dashboard tiles
│   ├── pages/
│   │   ├── 1_📍_Location_Detail.py     # site table -> reviews
│   │   └── 2_📊_Insights.py            # period -> site -> reviews
│   └── utils/
│       ├── ai.py                       # Azure OpenAI summaries of a review selection
│       ├── data_loader.py              # CSV load/clean/filter/KPIs (the only CSV reader)
│       ├── metrics.py                  # row stats, period/site rollups, review sorts
│       ├── reviews_ui.py               # shared table rows + expanded review list
│       ├── sentiment.py                # VADER score + label
│       ├── theme.py                    # CSS shell, KPI cards, mini charts
│       └── time_utils.py               # generic month/day aggregation helpers
├── code/                                # exploratory notebook (source of the VADER logic)
├── data/                                # final_reviews.csv (source data)
├── docs/                                # architecture, dataset schema, QA checklist
├── requirements.txt
└── README.md
```

The entrypoint is `Home.py`, not `app.py`, on purpose: `app/` is also a Python package, and
`streamlit run app/app.py` registers the script as module `app`, shadowing the package so
`from app.utils... import` fails.

## Notes for whoever touches this next

- **Per-element CSS is attached through container keys.** `st.container(key="foo")` emits a
  div with class `st-key-foo`; `theme.py` targets those (`[class*="st-key-qtile_"]`). That is
  also how a whole KPI card becomes one click target — an opacity-0 button stretched over it.
- **Text columns are kept on object dtype** (`pd.set_option("future.infer_string", False)` in
  `data_loader.py`). With pandas ≥3's Arrow-backed strings, the hash kernels behind
  `groupby`/`value_counts` run inside libarrow, whose mimalloc allocator segfaults on
  macOS/arm64 when first touched from a Streamlit script-runner worker thread — i.e. on every
  rerun. Removing that line brings the crash back.
- Sentiment is scored once for the whole file (~0.25s) and cached in
  `metrics.load_scored_data()`, so a review's score never depends on which filter surfaced it.
