# Architecture — Review Analysis Streamlit MVP

Owner: Agent 1 (Project Architect). This is the contract the other six agents'
deliverables must fit into. A human does final integration — nothing here
assumes agents merge each other's code directly.

## 0. Data reality check (read this before designing filters)

Verified by loading the actual CSV with pandas (not just `head`):

- **6,209 real data rows**, not 7,897 — `reviewText`/`ownerResponseText` contain
  embedded newlines inside quoted CSV fields, which inflates a raw `wc -l`
  count. Always parse with `pandas.read_csv`, never split on `\n`.
- **`businessName` is constant** (`"Big Dan's Car Wash"`) — this dataset is one
  car-wash chain, not multiple businesses. The real location dimension is
  **`site`** (25 unique values, e.g. `"Big Dan's Rome"`, `"Big Dan's Woodstock 2"`),
  which maps ~1:1 to `address`/`city`. **Location filters and drilldowns must
  key off `site` (or `site`+`address`), not `businessName`.** This is the single
  most important correction to the columns list in everyone's brief.
- **`state` is inconsistently formatted**: mixes full names (`Georgia`,
  `Florida`, `South Carolina`) and abbreviations (`AL`, `FL`, `SC`) for the
  same states. `data_loader.py` must normalize to one representation (full
  name recommended) before anything filters or groups by state.
- **124 duplicate `reviewId` values** — dedupe on `reviewId` during cleaning
  (drop_duplicates, keep first).
- Nulls to handle defensively: `reviewText` (1740 null — rating-only reviews,
  sentiment functions must treat as neutral/skip), `ownerResponseText`/
  `ownerResponseDate` (697 null — most reviews have no owner reply),
  `isLocalGuide`/`reviewerReviewCount`/`likesCount`/`reviewUrl` (497 null),
  `placeId`/`businessAvgRating`/`businessReviewCount` (377 null), `language`
  (411 null), `reviewId` (125 null — dedupe logic must not crash on NaN keys).
- `isLocalGuide` loads as Python `True`/`False`/`NaN` (object dtype) — cast to
  nullable boolean, don't assume clean bool.
- `rating` is a clean int 1–5, no nulls.
- `reviewDate` is ISO‑8601 UTC (`2026-07-17T15:45:58.308Z`) — parse with
  `pd.to_datetime(..., utc=True)`.

## 1. Architecture overview

Single-user, local/demo Streamlit app. Flow:

```
CSV (data/final_reviews.csv)
   │  data_loader.load_data()          [st.cache_data]
   ▼
cleaned DataFrame  (parsed dates, deduped, normalized state, cast dtypes)
   │  data_loader.filter_data()  +  time_utils.apply_date_range()
   ▼
filtered DataFrame  (scoped to current session_state["filters"])
   │  time_utils.resample_timeseries() / data_loader.get_kpis()
   │  sentiment.add_sentiment_columns() (only where sentiment is shown)
   ▼
aggregated tables / KPI dicts
   │
   ▼
Plotly charts + st.metric tiles (per plotting_guidelines.md — Plotly for all
new charts, not matplotlib)
```

Everything downstream of the CSV works off **one shared cleaned DataFrame**
loaded once per session and cached. Pages never re-read the CSV themselves —
they call `data_loader.load_data()`, which is cheap after the first call
because of `st.cache_data`.

## 2. Final file/folder structure

```
review_analysis/
├── data/
│   └── final_reviews.csv
├── code/
│   └── reviews_analysis (1).ipynb        # VADER logic source, reused by Agent 6
├── app/
│   ├── app.py                            # entrypoint = landing/home page
│   ├── pages/
│   │   ├── 1_📈_Trends.py                # from _drafts/trends_page.py
│   │   └── 2_📍_Location_Drilldown.py    # from _drafts/location_drilldown_page.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py                # Agent 2
│       ├── time_utils.py                 # Agent 4
│       └── sentiment.py                  # Agent 6
├── _drafts/                              # working area, merged into app/ by integrator
│   ├── landing_page.py                   # Agent 3
│   ├── trends_page.py                    # Agent 4
│   └── location_drilldown_page.py        # Agent 5
├── docs/
│   ├── architecture.md                   # this file
│   ├── dataset_schema.md                 # Agent 2
│   └── testing_checklist.md              # Agent 7
├── requirements.txt                      # Agent 7
└── README.md                             # Agent 7
```

Streamlit's native multipage convention: `app/app.py` is the entrypoint (run
via `streamlit run app/app.py`), and any `.py` file in `app/pages/` becomes an
extra page automatically, ordered by the leading number and labeled from the
filename. The landing/home content (KPI tiles) lives directly in `app.py`,
not in a `pages/0_...py` file — that's why Agent 3's draft is merged into
`app.py` rather than becoming a page.

## 3. Shared utility interfaces

These are the exact signatures the other modules must expose. Downstream
pages should be able to import and call these without knowing anything about
the CSV's quirks — all cleaning/normalization happens inside `data_loader.py`.

### `app/utils/data_loader.py` (Agent 2)

```python
import pandas as pd
import streamlit as st
from datetime import date

CSV_PATH = "data/final_reviews.csv"  # resolved relative to project root

@st.cache_data(show_spinner="Loading reviews...")
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """Read raw CSV, parse reviewDate/ownerResponseDate to UTC datetime,
    cast rating to int, isLocalGuide to nullable bool, normalize `state`
    to full names, dedupe on reviewId (drop rows with null reviewId's
    dupes too). Returns one clean DataFrame — the single source of truth
    for every page."""

def get_filter_options(df: pd.DataFrame) -> dict:
    """Return choices for filter widgets, e.g.:
    {
      "sites": sorted(df.site.unique()),        # location dimension
      "cities": sorted(df.city.unique()),
      "states": sorted(df.state.unique()),
      "categories": sorted(df.category.unique()),
      "date_min": date, "date_max": date,
      "rating_min": 1, "rating_max": 5,
    }"""

def filter_data(
    df: pd.DataFrame,
    date_range: tuple[date, date] | None = None,
    sites: list[str] | None = None,
    cities: list[str] | None = None,
    states: list[str] | None = None,
    categories: list[str] | None = None,
    rating_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Apply all given filters (None/empty = no-op for that filter).
    Returns a filtered copy; never mutates df in place."""

def get_kpis(df: pd.DataFrame) -> dict:
    """Summary numbers for landing-page tiles, e.g.:
    {
      "total_reviews": int, "avg_rating": float, "total_sites": int,
      "pct_5_star": float, "pct_with_owner_response": float,
      "total_cities": int,
    }"""
```

### `app/utils/time_utils.py` (Agent 4)

```python
import pandas as pd
from datetime import date

def add_time_columns(df: pd.DataFrame, date_col: str = "reviewDate") -> pd.DataFrame:
    """Add derived columns: review_date (date), review_month (period-start
    Timestamp), review_week, review_year, day_of_week. Returns a copy."""

def get_date_bounds(df: pd.DataFrame, date_col: str = "reviewDate") -> tuple[date, date]:
    """(min_date, max_date) — used to set st.date_input defaults."""

def apply_date_range(
    df: pd.DataFrame, start: date, end: date, date_col: str = "reviewDate"
) -> pd.DataFrame:
    """Inclusive filter on date_col. Returns a copy."""

def resample_timeseries(
    df: pd.DataFrame,
    freq: str = "M",              # 'D' | 'W' | 'M' | 'Q' | 'Y'
    date_col: str = "reviewDate",
    value_col: str = "rating",
    agg: str = "mean",            # 'mean' | 'count' | 'sum'
) -> pd.DataFrame:
    """Group by period, return columns [period, value] — ready to hand
    straight to px.line()."""
```

### `app/utils/sentiment.py` (Agent 6 — reuses VADER logic from the notebook)

```python
import pandas as pd
import streamlit as st

@st.cache_resource
def get_analyzer():
    """Singleton SentimentIntensityAnalyzer (vaderSentiment). Cached as a
    resource, not data, since it's a model/object, not serializable output."""

def score_text(text: str) -> dict:
    """VADER polarity_scores for one string: {neg, neu, pos, compound}.
    None/NaN/empty text -> {"neg":0,"neu":1,"pos":0,"compound":0} (neutral),
    never raises."""

@st.cache_data(show_spinner="Scoring sentiment...")
def add_sentiment_columns(df: pd.DataFrame, text_col: str = "reviewText") -> pd.DataFrame:
    """Adds sentiment_compound (float) and sentiment_label
    ('positive'/'neutral'/'negative', thresholds compound >= 0.05 / <= -0.05)
    columns. Returns a copy. Rows with null reviewText get the neutral
    defaults from score_text, not NaN — downstream aggregation must not
    special-case missing text."""

def get_sentiment_summary(df: pd.DataFrame) -> dict:
    """{"pct_positive":.., "pct_neutral":.., "pct_negative":.., "avg_compound":..}
    — for sentiment KPI tiles/pages."""
```

Note for Agent 6: `add_sentiment_columns` is the only heavy VADER call and it
is `st.cache_data`-cached, so callers should call it on the **filtered**
DataFrame right before charting, not on the full 6,209-row dataset on every
page load, unless a page genuinely needs sentiment for the whole dataset.

## 4. Navigation & session-state design

**Global filter state** — one canonical dict in `st.session_state["filters"]`,
shape matching `filter_data()`'s kwargs:

```python
st.session_state.setdefault("filters", {
    "date_range": None, "sites": [], "cities": [], "states": [],
    "categories": [], "rating_range": None,
})
```

Every page reads/writes this same key so filters chosen on one page persist
when the user navigates to another (via the sidebar or a tile click). Filter
widgets should be initialized from `st.session_state["filters"]` and write
back to it on change (`st.session_state["filters"]["cities"] = ...`), not use
ad hoc local variables.

**Tile-click navigation (landing → detail pages)** — use `st.session_state` +
`st.switch_page()`, not query params, as the primary mechanism (this is a
single-process local app, not a shareable multi-tenant URL product — query
params are unnecessary complexity for the MVP):

1. On the landing page, each tile is a `st.button()`. On click, the handler
   sets a *pending pre-filter*, distinct from the general filter state:
   ```python
   st.session_state["nav_prefill"] = {"sites": ["Big Dan's Rome"]}  # example
   st.switch_page("pages/2_📍_Location_Drilldown.py")
   ```
2. The target page, on load, checks for `nav_prefill`:
   ```python
   if "nav_prefill" in st.session_state:
       st.session_state["filters"].update(st.session_state.pop("nav_prefill"))
   ```
   then builds its widgets from `st.session_state["filters"]` as normal. This
   keeps "landing tile sets a filter" and "the filter state itself" as two
   separate concerns so pages don't need special-case logic beyond one
   `if`/`pop` at the top.
3. `st.query_params` is an acceptable *optional* stretch add (e.g. for
   deep-linking a specific site) but is not required for MVP functionality
   and should not block integration if skipped.

**Caching strategy**

- `load_data()` — `st.cache_data`, keyed on `path`; this is the expensive
  step (CSV parse + cleaning) and should run once per session.
- `filter_data()` / `apply_date_range()` — cheap at 6,209 rows; caching is
  optional (Streamlit hashes DataFrame args by content, which is fine at
  this size, but not necessary to add `@st.cache_data` here — don't over-cache
  trivial pandas filtering).
- `add_sentiment_columns()` — `st.cache_data`, since VADER scoring is the
  next most expensive step and is re-run whenever the filtered set changes.
- `get_analyzer()` — `st.cache_resource` (it's a model/singleton object, not
  data).
- Nothing needs a TTL — the CSV is static for the MVP; if it later becomes
  refreshable, add `ttl=` to `load_data`.

## 5. Integration plan (for the human integrator)

Merge order — utilities first, then pages, then cross-cutting navigation:

1. **Scaffold**: create `app/`, `app/pages/`, `app/utils/__init__.py` (empty).
   Move/confirm Agents 2/4/6 already dropped `data_loader.py`, `time_utils.py`,
   `sentiment.py` into `app/utils/` per their assigned paths — no file moves
   should be needed if each agent respected their scoped path.
2. **Smoke-test utils in isolation**: `python -c "from app.utils import
   data_loader; df = data_loader.load_data(); print(df.shape, df.dtypes)"` —
   confirm the signatures above match what got built, especially that `site`
   (not `businessName`) is exposed as the location filter dimension, and
   `state` values are normalized.
3. **`app.py`**: create the entrypoint, `st.set_page_config(...)`, sidebar
   filter widgets bound to `st.session_state["filters"]`, then fold in
   `_drafts/landing_page.py`'s tile/KPI rendering, wiring each tile's button
   to set `nav_prefill` + `st.switch_page(...)` as described in §4.
4. **`app/pages/1_📈_Trends.py`**: fold in `_drafts/trends_page.py`, replacing
   any standalone CSV loading/filtering it did with calls to `data_loader`/
   `time_utils`. Confirm it reads `st.session_state["filters"]` and consumes
   `nav_prefill` if present.
5. **`app/pages/2_📍_Location_Drilldown.py`**: same merge for
   `_drafts/location_drilldown_page.py`. Pay special attention here — this is
   where the `site`-vs-`businessName` correction from §0 matters most, since
   "location" in this dataset means `site`.
6. **Sentiment wiring**: wherever Trends/Drilldown show sentiment, confirm
   they call `sentiment.add_sentiment_columns()` on the already-filtered
   frame (small), not the full dataset, per the caching note in §4.
7. **`requirements.txt`**: verify it covers `streamlit`, `pandas`, `plotly`,
   `vaderSentiment` (or whatever VADER package the notebook used), and pin
   a Streamlit version ≥1.30 (needed for `st.switch_page`/multipage-as-code).
8. **Run + walk the testing checklist** (`docs/testing_checklist.md`,
   Agent 7): load app, exercise every filter, click every landing tile,
   confirm prefill lands correctly on the target page, confirm charts are
   Plotly (per house style) and render in both light/dark if applicable.
9. **README.md** (Agent 7): finalize run instructions
   (`streamlit run app/app.py`) once the above is confirmed working end to end.

If any agent's actual function signature drifted from §3, the integrator
should treat this doc as the source of truth and adapt the caller (page)
side rather than threading special cases through `data_loader`/`time_utils`/
`sentiment` — keeping those three modules' public surface exactly as
specified here is what makes steps 3–6 mechanical instead of exploratory.
