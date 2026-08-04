# Review Analysis MVP

A local Streamlit app for exploring and analyzing customer reviews from `data/final_reviews.csv`
(6,208 reviews across 25 physical locations of a single car-wash chain — "location" means the
`site` column, not `businessName`, which is constant; see `docs/dataset_schema.md`). The app
gives an at-a-glance dashboard of all locations, a trend chart with month/day toggle, a
per-location drilldown, and searchable reviews split by VADER sentiment.

## Install

```bash
cd review_analysis
python3 -m venv .venv        # optional but recommended
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies (see `requirements.txt`): `streamlit`, `pandas`, `plotly`,
`vaderSentiment`.

## Launch

```bash
streamlit run app/Home.py
```

This opens the app in your browser (default `http://localhost:8501`). All data is read from
`data/final_reviews.csv` inside this project folder — no external network calls or files
outside `review_analysis/` are required.

## Features

- **Landing dashboard** (`app/Home.py`) — KPI tiles (total reviews, avg rating, location
  count, 5-star share, owner-response rate), a rating-distribution chart, a review-count-by-
  location chart, and a clickable tile grid — one tile per location — that jumps to its detail
  page.
- **Location Detail page** (`app/pages/1_📍_Location_Detail.py`) — site/city filters, a
  month/day trend toggle with growth/decline deltas and a Plotly count+rating chart, a
  keyword search over review text, a recency/rating sort control, and Positive/Negative/Neutral
  tabs (VADER sentiment) with each review's text and any owner response.

## Project structure

```
review_analysis/
├── app/
│   ├── Home.py                          # entrypoint — landing dashboard
│   ├── pages/
│   │   └── 1_📍_Location_Detail.py     # trends + review drilldown + sentiment
│   └── utils/
│       ├── data_loader.py              # CSV load/clean/filter/KPIs
│       ├── time_utils.py               # month/day aggregation, growth indicators
│       └── sentiment.py                # VADER classification
├── code/                                # exploratory notebook (source of the VADER logic)
├── data/                                # final_reviews.csv (source data)
├── docs/                                # architecture, dataset schema, QA checklist
├── requirements.txt
└── README.md
```
