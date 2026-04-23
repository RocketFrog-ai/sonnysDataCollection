# Backend structure for site profiling: quantile data → LLM narratives

This doc recommends how to structure the backend so that:

1. **User inputs address** → you fetch weather (and other features), run quantile-based prediction (v3), then feed that **structured quantile data** to **LLM agents** to produce per-feature summaries, business impact, impact classification, and overall insight/observation/conclusion.

---

## 1. End-to-end flow

```
Address
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Feature fetch (existing)                                           │
│  Geocode → get_climate, get_nearby_gas_stations, get_nearby_retailers,       │
│  get_nearby_competitors, get_nearby_stores_data, etc.                        │
│  Output: raw feature dict (API-shaped keys → values)                          │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: Quantile prediction (v3)                                           │
│  Map API feature dict → internal keys → QuantilePredictorV4.analyze()          │
│  Output: predicted_wash_quantile, predicted_wash_range, feature_analysis     │
│  (per feature: value, label, raw_percentile, wash_correlated_q,               │
│   wash_q_q4_median, quantile_boundaries, importance, etc.)                   │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Display shaping (new)                                               │
│  Map v3 feature_analysis → UI metrics (e.g. 4 weather metrics).               │
│  For each metric: value, unit, scale (min/max), category (Poor/Fair/Good/      │
│  Strong), percentile. Optional: volume band (“150k–350k washes”) from v3.    │
│  Output: list of { metric_key, value, unit, category, min, max, percentile,   │
│           volume_band? }                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: LLM agents (new)                                                   │
│  • Per-feature agent: (metric_key, value, unit, category, min, max,            │
│    percentile, volume_band) → summary, business_impact, impact_classification │
│  • Overall agent: all feature narratives + weather score / overall score      │
│    → insight, observation, conclusion                                         │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
API response: metrics with value/unit/scale/category + narratives + overall text
```

---

## 2. Implemented folder layout

Single fetch in one place; narratives in `modelling/ai`; feature data from `features/active`:

```
app/
├── server/
│   ├── config.py          # Already exists: Redis, DIMENSIONS, WEATHER_METRIC_CONFIG
│   ├── models.py          # Request/response models
│   └── routes.py          # Thin: validate, call pipeline, return response
│
├── pipeline/                      # NEW: orchestration only
│   ├── __init__.py
│   ├── feature_fetch.py           # Address → lat,lon + fetch all features (weather, gas, retailers, competitors)
│   ├── feature_mapping.py         # API feature dict ↔ v3 predictor feature names
│   └── run_site_profiling.py      # run(address) → feature dict → v3.analyze() → raw v3 result
│
├── modelling/
│   └── ds/                        # Existing: scorer, prediction (wash tiers), dimension_summary
│       └── ...
│
├── narrative/                     # NEW: quantile data → text
│   ├── __init__.py
│   ├── config.py                  # Metric key → display name, scale bounds (Poor/Fair/Good/Strong)
│   ├── shape_for_ui.py            # v3 feature_analysis → list of { value, unit, category, min, max, ... }
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── feature_summary_agent.py   # One feature’s quantile data → summary, business_impact, impact_classification
│   │   └── overall_summary_agent.py   # All feature narratives + score → insight, observation, conclusion
│   └── prompts.py                 # (Optional) centralise prompt templates
│
└── utils/
    └── llm/                       # Existing: local_llm.get_llm_response
        └── ...
```

- **app/site_analysis/server**: HTTP only; no business logic beyond calling analysis.
- **app/site_analysis/modelling/site_analysis**: One orchestration: address → geocode → fetch all features once (using `features/active`) → build `feature_values` and v3 `location_features` → quantile prediction. Same fetched data reused for API and quantile.
- **app/site_analysis/modelling/ds**: Already holds v3 and scoring; no changes to its responsibilities.
- **app/site_analysis/modelling/ai**: Narrative generation: per-feature (summary, business_impact, impact_classification) and overall (insight, observation, conclusion). Stub implementations; wire LLM agents here.

---

## 3. Data contracts (recommended)

### 3.1 After Layer 2 (v3 output you already have)

From `QuantilePredictorV4.analyze()` you get, per feature in `feature_analysis`:

- `value`, `label`, `raw_percentile`, `adjusted_percentile`
- `wash_correlated_q`, `wash_correlated_exceeds_q4`, `wash_q_q4_median`
- `quantile_boundaries` (scale min/max), `importance`
- Plus `predicted_wash_quantile`, `predicted_wash_range`, `quantile_probabilities`, etc.

### 3.2 After Layer 3 (display-ready metric)

One object per UI metric (e.g. the 4 weather cards):

```python
{
    "metric_key": "dirt-trigger-days",
    "display_name": "Dirt Creation Days",
    "subtitle": "Rainy Days + Snowy Days",
    "value": 120,
    "unit": "days/year",
    "category": "Good",           # Poor | Fair | Good | Strong (from percentile/quantile or config)
    "min": 0,
    "max": 200,
    "percentile": 65.0,
    "volume_band": "150k–350k",   # optional: from v3 wash range or banding
}
```

You can add `quantile_score` (0–100) and `scale_bands` (e.g. Poor 0–50, Fair 50–100, …) here if the UI needs them.

### 3.3 After Layer 4 – per-feature LLM

```python
{
    "metric_key": "dirt-trigger-days",
    "summary": "90% of sites generating 150k–350k washes per year have a Dirt Trigger Window of 100–150 days. This site falls within that range.",
    "business_impact": "Frequent dirt triggers create strong recurring wash demand.",
    "impact_classification": "Strong · 100–150 days",
}
```

### 3.4 After Layer 4 – overall LLM

```python
{
    "insight": "Weather contributes ~20% to the site potential. With a Weather Impact Score of 70%, supported by balanced dirt triggers, strong comfortable washing days, and low shutdown risk, the location shows favorable conditions for stable wash demand.",
    "observation": "Site ABC benefits from a well-rounded weather profile. ...",
    "conclusion": "The weather profile supports consistent year-round wash activity with moderate seasonal variation, making this a favorable location from a weather-driven demand perspective.",
}
```

---

## 4. Where config lives

- **app/site_analysis/server/config.py**: API-level config (e.g. `WEATHER_METRIC_CONFIG`: metric_key → climate key, unit, description). Add any route-level defaults here.
- **app/narrative/config.py**: Narrative/UI config, e.g.:
  - Metric key → display name, subtitle, scale bounds (min/max per category if fixed).
  - Mapping from percentile or wash_q to category (Poor/Fair/Good/Strong) for each metric.

That way “what the API exposes” stays in `server`, and “how we describe it and which scales we use for narratives” stays in `narrative`.

---

## 5. API shape suggestion

- **Option A – single “full profile” endpoint**  
  `POST /api/v1/site-profile` body: `{ "address": "..." }`.  
  Returns: feature metrics (value, unit, scale, category) + per-feature narratives (summary, business_impact, impact_classification) + overall (insight, observation, conclusion).  
  Backend: run_site_analysis(address) → quantile result → modelling/ai narratives (per-feature + overall) → return one JSON.

- **Option B – stepwise**  
  - `POST /api/v1/site-profile/quantile` → raw v3 result (and optionally shaped metrics without LLM).  
  - `POST /api/v1/site-profile/narratives` body: `{ "quantile_result": {...} }` or a `profile_id` → run only narrative agents and return narratives.  
  Useful for debugging and for optional “narratives on demand.”

Start with **Option A**; add Option B later if you need to re-run only LLM or cache quantile result.

---

## 6. Summary

| Layer        | Responsibility                    | Location              |
|-------------|------------------------------------|------------------------|
| Feature fetch | Geocode, get weather/gas/retail/competitors (single fetch) | `app/site_analysis/modelling/site_analysis.py` (fetch_all_features) |
| Feature mapping | API keys ↔ v3 internal keys       | `app/site_analysis/modelling/site_analysis.py` (build_feature_values_and_v3_input) |
| Quantile prediction | v4.analyze(location_features)     | `app/site_analysis/modelling/ds/prediction` |
| Per-feature narrative | One metric’s quantile data → summary, business_impact, impact_classification | `app/site_analysis/modelling/ai/narratives.py` (get_feature_narratives) |
| Overall narrative | All narratives + score → insight, observation, conclusion | `app/site_analysis/modelling/ai/narratives.py` (get_overall_narrative) |
| HTTP         | Validate, call analysis, return JSON | `app/site_analysis/server/routes.py` |

This keeps a clear path from **address → single fetch (features/active) → feature_values + quantile result (v3) → modelling/ai narratives → summaries and overall text**, with config in `server` and no business logic in routes.
