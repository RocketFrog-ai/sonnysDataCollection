# Quantile Prediction — Flow & Strategy

## Data source

- **Car wash counts:** `scoringmetric/approach2/v2/temp_extrapolated.csv`
  - Columns used: `street`, `city`, `zip`, `current_count`, `location_id`, and optionally `client_id` for control-site filtering.
  - One row per site; `current_count` = annual car wash volume (extrapolated where needed).
- **Features:** `Proforma-v2-data-final (1).xlsx` (19 environmental/location features).
- **Merge:** Inner join on normalized **street + 5-digit ZIP** so only sites present in both datasets are used for training and quartile definitions.

---

## New quantile strategy: equal count of sites

**Goal:** Business-explainable tiers: “bottom 25% of sites”, “top 25% of sites”, etc.

**Method:**

1. Take all sites that have both features and `current_count` (after merge).
2. Sort sites by `current_count` (ascending).
3. Define quartiles by **equal count of sites**, not by equal value ranges:
   - **Q1** = bottom 25% of sites (e.g. bottom ~121 of 483) → lowest volume.
   - **Q2** = next 25% of sites.
   - **Q3** = next 25% of sites.
   - **Q4** = top 25% of sites → highest volume.

**Implementation:** Use percentile boundaries `[0, 25, 50, 75, 100]` on `current_count`. That yields four value ranges such that when each site is assigned to a bin, **each quartile has (approximately) the same number of sites**. Value ranges (e.g. Q1: 47–56k vs Q4: 164k–596k) will be non-uniform; the important part is that **each tier = same share of sites**.

**Rationale:** “You are in the top 25% of car wash sites by volume” is clear. “You are in the tier with 164k–596k cars/yr” is the supporting detail.

---

## Control sites and model rebuild

- **Control sites:** The model will be rebuilt using **only control sites** (e.g. exclude internal/test sites such as “Controls Training Classroom”) when a suitable flag or `client_id` filter is available.
- **Missing features:** Additional features will be added as they become available; the pipeline supports more than the current 19.
- **Accuracy target:** **>70% within-1-quantile accuracy** (prediction is either exact quartile or one tier adjacent). Exact quartile accuracy remains a secondary metric.

---

## End-to-end flow

| Step | What happens |
|------|-------------------------------|
| 1. Load | Read `temp_extrapolated.csv` and Proforma Excel. |
| 2. Merge | Join on street + 5-digit ZIP; keep only rows with both features and `current_count`. |
| 3. (Optional) Filter | If “control sites only” is enabled, drop rows where `client_id` (or similar) indicates non-control/test sites. |
| 4. Equal-count quartiles | Compute 0/25/50/75/100 percentiles of `current_count`; assign each site to Q1–Q4. Each quartile has ~same number of sites. |
| 5. Feature quartiles | For each of the 19 features, compute 0/25/50/75/100 percentiles over the same training set (equal-count per feature bin). |
| 6. Wash-correlated Q (WashQ) | For each feature, compute median value within each **car wash quartile** (Q1–Q4). For a new location, map each feature value to the wash quartile whose median it’s closest to (direction-adjusted). |
| 7. Train model | Random Forest classifier: 19 features → wash quartile (Q1–Q4). Median imputation for missing features. |
| 8. For a new location | Predict wash quartile + probabilities; compute WashQ and percentiles per feature; run quantile-shift analysis; build narrative. |
| 9. Report | Prediction, feature table (Value, WashQ, Pctile, Q4 median, Importance), profile comparison, **Strengths & Weaknesses** (investment-report style), optional narrative. |

---

## Investment-report framing: Strengths & Weaknesses

- **Strengths:** Factors that look like high-performing (Q4) sites: features where the location’s value **matches or exceeds** the Q4 group median (WashQ = Q4 or “exceeds Q4”).
- **Weaknesses:** Factors that look like low-performing (Q1) or below-median sites: features where WashQ = Q1 or Q2, or value is below the Q4 benchmark.

The report presents these as short, human-readable bullets so the output reads like an investment summary: key positives and key concerns, with quartiles and benchmarks as backing.

---

## Methodology reference

See `distribution_plots/QUANTILE_ANALYSIS_METHODOLOGY.txt` for:

- Feature direction (higher vs lower is better).
- WashQ and Q4 median interpretation.
- Model and accuracy metrics.
