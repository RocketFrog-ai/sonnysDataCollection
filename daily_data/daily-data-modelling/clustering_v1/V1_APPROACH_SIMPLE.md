# Clustering V1: Step-by-Step (Simple + Clear)

This folder is for the **V1 projection flow**.  
Goal: for a new site, estimate wash potential and forward projection using local market clusters.

---

## 1) Why clustering exists in this system

Clustering is used for **localization**.

Idea:
- Sites near each other usually share demand patterns (traffic, weather, demographics, retail mix).
- So first we place the new site inside a nearby local market cluster.

In V1:
- Clusters are built using **latitude/longitude** only (DBSCAN).
- Radius options: `12km` or `18km`.

So clustering gives:
- **which local market group** the site belongs to,
- and **historical local range** for that group.

---

## 2) End-to-end modelling flow (V1)

For each cohort (`more_than_2yrs`, `less_than_2yrs`) the flow is:

1. Input site (`address` or `lat/lon`).
2. Geocode address if needed.
3. Find nearest cluster centroid (haversine distance).
4. If too far from all centroids (outside radius gate), mark unassigned.
5. If assigned:
   - Run Ridge prediction (`predicted_wash_count_ridge`).
   - Fetch cluster historical stats (`min/p10/median/p90/max` from train data).
   - Build 24-month projection using `arima`, `holt_winters`, or `blend`.
6. Return both cohorts in one JSON response.
7. Plot bar chart for 6/12/18/24 cumulative projection.

---

## 3) Ridge model: what it uses

Ridge is the **point prediction model** (single wash estimate).

Important:
- Ridge does **not** simply average other sites.
- Ridge uses learned weights on features.

Feature scope in V1 (typical):
- time/lag signals (`day_number`, `month_number`, `prev_wash_count`, etc.)
- weather summary features
- competition/retail/gas proximity and ratings
- location (`latitude`, `longitude`)
- cluster id (e.g. `dbscan_cluster_12km`)

So locality is introduced mainly through:
- cluster assignment,
- cluster id feature,
- local historical range block.

---

## 4) Projection model: what it uses

Projection is a separate step from Ridge.

It uses cluster history to project forward monthly values:
- `arima`
- `holt_winters`
- `blend` = average of both

Then it returns:
- monthly forecast next 24 months,
- cumulative totals for horizons 6/12/18/24,
- bar-graph-ready data.

---

## 5) Output structure (important keys)

Top level:
- `radius`
- `method`
- `more_than_2yrs`
- `less_than_2yrs`

Each cohort block:
- `assigned_cluster_id`
- `distance_to_cluster_km`
- `predicted_wash_count_ridge`
- `historical_daily_wash_count` (`min/p10/median/p90/max`)
- `projection`
  - `horizons_months`: `[6, 12, 18, 24]`
  - `cumulative_wash_count`
  - `avg_monthly_wash_count`
  - `bar_graph_data`
  - `monthly_forecast_next_24`

---

## 6) Run command

From repo root:

```bash
python daily_data/daily-data-modelling/clustering_v1/run_projection_demo.py \
  --address "5360 Laurel Springs Pkwy, Suwanee, GA 30024" \
  --radius 12km \
  --method blend \
  --base-url http://localhost:8001
```

Outputs go to:
- `daily_data/daily-data-modelling/clustering_v1/results/projection_demo/`

---

## 7) Dry run example (simple)

Input:
- Address: `5360 Laurel Springs Pkwy, Suwanee, GA 30024`
- Radius: `12km`
- Method: `blend`

What happens:
1. System geocodes the address.
2. Finds closest cluster in `more_than_2yrs` and `less_than_2yrs` datasets.
3. Runs Ridge for each cohort.
4. Pulls each cohort’s cluster historical range.
5. Builds each cohort’s 24-month projection.
6. Saves:
   - one JSON with both cohorts
   - one PNG with two bar charts (one per cohort)

If output is healthy, each cohort has:
- a valid `assigned_cluster_id`,
- non-null `predicted_wash_count_ridge`,
- increasing cumulative bars (6 -> 12 -> 18 -> 24).
