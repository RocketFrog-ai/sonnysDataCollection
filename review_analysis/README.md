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

## Run it in Docker

Everything needed is in this folder — the review CSV, the precomputed AI
answers, the theme. The only thing from outside is the Azure key, and the app
runs without one (only the AI button reports it has nothing to call).

```bash
cd review_analysis
docker build -t review-analysis .
docker run -p 8501:8501 -e AZURE_OPENAI_API_KEY=... review-analysis
```

or, with the key in a local `.env` (copy `.env.example`):

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

- The key is **never** baked into an image: `.streamlit/secrets.toml` and `.env`
  are in `.dockerignore`, and `app/utils/ai.py` falls back to the
  `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT`
  environment variables.
- The image runs as a non-root user and exposes Streamlit's `/_stcore/health`
  endpoint as its `HEALTHCHECK`.
- The one file written at runtime is `data/ai_insights_cache.json` (new AI
  answers). `docker-compose.yml` mounts it so those survive a restart; without
  the mount the container still ships with the 18 precomputed ones.

## Deploy to the Azure Web App

The existing `proforma-demo-2` app (RG `son_eastus2_proforma_rg02`, Linux
container, pulling from `proformaacr.azurecr.io`) can run this instead:

```bash
cd review_analysis
export AZURE_OPENAI_API_KEY=...        # optional; enables the AI button
./scripts/deploy_azure.sh
```

It builds `proformaacr.azurecr.io/review-analysis:<timestamp>`, pushes it,
repoints the web app, sets the app settings and restarts. It pushes a **new
repository** rather than overwriting `proforma-demo-2:latest`, so the image
that app runs today is untouched and rollback is one command (printed at the
end of the run).

Four things App Service needs that the script handles, and that are the usual
causes of a container that "deploys" but never serves:

| Setting | Why |
|---|---|
| `--platform linux/amd64` | App Service is x86_64; an arm64 image built on an Apple-silicon Mac starts and dies with an exec-format error visible only in the container log |
| `WEBSITES_PORT=8501` | Without it the platform probes 8080 and returns 504 while the app is healthy |
| Web sockets enabled | Streamlit drives the page over a websocket; without it the page loads and sits on "connecting" |
| `healthCheckPath=/_stcore/health` | The app currently shows Health Check "Not Configured" |

The image also reads `PORT` if the platform injects one, so the same image runs
unchanged on Cloud Run or Fly.

Secrets go in as **app settings**, never in the image: `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`. `REVIEW_ANALYSIS_DEMO_LOCKED`
defaults to `1` (locked demo view); set it to `0` for the full build.

Watch the rollout:

```bash
az webapp log tail --name proforma-demo-2 --resource-group son_eastus2_proforma_rg02
```

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

Every tile carries a **✨ AI** button in its top-right corner, and each drill-down table has one
in its header. One press summarises *exactly the reviews that tile counted* — no second click —
and opens a dialog laid out as:

- a **headline** answering the tile's question in one sentence;
- **Key points** — each with the number of reviews that mention it and verbatim quotes as
  evidence;
- **Recommendations** — 2–4 concrete actions, each tied back to a key point.

Long selections are handled with map-reduce: reviews are chunked 45 at a time, each chunk is
summarised into themes, and a second call merges them. A 400-review selection is ~10 calls. The
model returns JSON, so the page lays the answer out rather than rendering a markdown blob.

**Answers are saved.** Each is keyed by a hash of (question, scope, the exact reviews) and written
to `data/ai_insights_cache.json`, so the same selection is never paid for twice — across restarts,
not just reruns.

```bash
python scripts/precompute_insights.py                    # 9 tiles x 2 windows
python scripts/precompute_insights.py --window "Current month"
python scripts/precompute_insights.py --force            # regenerate
```

That has been run for **Current month** and **Last 3 months**, so those tiles open in ~0.3s off
the cache. Other windows generate on demand (a few seconds to ~90s depending on size) and are
cached from then on.

The model only ever reads review text. **It never produces a number on the page** — every figure
still comes from `metrics.py`, so a wrong summary cannot corrupt a metric. Transient Azure 429/5xx
responses are retried three times with backoff.

Credentials live in `.streamlit/secrets.toml`, which is **gitignored**:

```toml
azure_openai_api_key = "..."
azure_openai_endpoint = "https://<resource>.openai.azure.com"
azure_openai_deployment = "gpt-4o"
```

`AZURE_OPENAI_API_KEY` etc. work as environment variables too. Without a key the button reports
that and the rest of the app is unaffected.

## Project structure

```
review_analysis/
├── Dockerfile / docker-compose.yml      # self-contained deployment
├── .dockerignore / .env.example         # keeps the key out of the image
├── .streamlit/config.toml               # pins the light theme
├── .streamlit/secrets.toml              # Azure OpenAI key (gitignored)
├── app/
│   ├── Home.py                          # entrypoint — dashboard tiles
│   ├── pages/
│   │   ├── 1_📍_Location_Detail.py     # site table -> reviews
│   │   └── 2_📊_Insights.py            # period -> site -> reviews
│   └── utils/
│       ├── ai.py                       # Azure OpenAI insights (map-reduce + disk cache)
│       ├── data_loader.py              # CSV load/clean/filter/KPIs (the only CSV reader)
│       ├── metrics.py                  # row stats, period/site rollups, review sorts
│       ├── reviews_ui.py               # shared table rows + expanded review list
│       ├── sentiment.py                # VADER score + label
│       ├── theme.py                    # CSS shell, KPI cards, mini charts
│       └── time_utils.py               # generic month/day aggregation helpers
├── scripts/precompute_insights.py       # fills the AI cache for the default windows
├── code/                                # exploratory notebook (source of the VADER logic)
├── data/                                # final_reviews.csv + ai_insights_cache.json
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
- **The review lens lives in plain session keys, not widget keys.** Streamlit discards the state
  of any widget it did not render, so when every row was collapsed (no controls on screen) a lens
  handed over by a tile was thrown away, and re-expanding rebuilt the controls at their first
  option — which is how a negative-reviews tile ended up showing five-star reviews. The widgets
  are keyed separately and seeded with `index=` from the canonical values.
