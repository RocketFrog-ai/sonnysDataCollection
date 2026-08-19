# Customer profiling — membership book

Member-level profiling on `hurricane_customer_profille_data.csv`: an event log of **466 customers /
587 vehicles** across **two sites**, **2025-09-13 → 2026-08-10**, carrying both what members pay and
what they consume. That combination is what makes it modellable — a payments-only export tells you
when someone left; this one tells you *why*, early enough to act.

Standalone per `CLAUDE.md`: `experiments/` is off the import path, and nothing here is imported by
the proforma model or the API.

```bash
conda activate sonnys
jupyter lab experiments/customer-profiling/customer_profiling.ipynb   # the analysis
streamlit run experiments/customer-profiling/app.py                   # the demo
```

## Files

| File | What it is |
|------|-----------|
| `profiling.py` | All the logic — preprocessing, features, churn model, segmentation, CLV. Streamlit-free. |
| `viz.py` | Palette and Plotly layout (the validated palette, both modes selected). |
| `customer_profiling.ipynb` | The analysis, 9 sections, every output followed by its insights. |
| `app.py` | Five-tab Streamlit demo. Imports `profiling.py`, so it cannot disagree with the notebook. |

## Two things about the file's shape

Both are handled in `load_events` / `payments`, and getting either backwards wrecks every number
downstream.

1. **Payments are fanned out across the household's vehicles.** A three-car household paying once
   appears as three identical rows. Naive `amount.sum()` gives **\$111,285**; the correct figure is
   **\$70,229** — a 58.5% overstatement, concentrated on the *best* customers.
2. **Washes are per vehicle and must not be collapsed.** The only duplicates are one `vehicle_id`
   under two spellings of its plate (249 rows of 7,741).

Also: `vehicle_vin` / `vehicle_year` / `vehicle_model` are >96% null and dropped; `$0` renewals are
comped months on live accounts (kept); negatives are refunds (netted, not counted as cycles); one
customer has washes but no payment row at all.

## What the analysis found

| Finding | Number | Evidence |
|---|---|---|
| **Dormancy is the biggest churn lever** | zero-wash cycles renew at 87.0% vs 94.7% | χ² p=4e-09, n=2,027, monotone dose-response |
| **The intro promo is a cliff, not a discount** | step-up cycles renew 90.2% vs 95.0% | p=0.0009; \$10 → \$32 (3.2×) at month 1 for 333/351 members |
| **Multi-car households are the stickiest thing in the book** | 1 car 92.0% vs 2+ 96.4% | p=0.0019, gradient continues to 4+ cars |
| **Promo joiners are not worse customers** | 92.3% vs 94.0%, p=0.51 | null result, but underpowered — 91.4% of signups took the promo |
| **Revenue is far more concentrated than headcount** | 16% of members = 35% of revenue | four k-means personas |
| **29.2% of paid membership months involved no wash** | \$8,685 collected, 14.5% of cycle revenue | simultaneously the highest-margin and most fragile revenue |

**Churn model** — discrete-time renewal hazard, one row per paid membership month, features known
at the moment of the charge. Logistic regression: **holdout AUC 0.713** on a time-ordered split,
**2.71× top-decile lift**, and calibrated (riskiest quintile predicts 15.8%, delivers 15.3%). A
LightGBM check scores 0.662, so nothing non-linear is being missed. Live book: **~22.8 of 334 active
members** expected to churn next cycle, **\$564 of \$9,726 MRR**.

**Personas** (k=4, silhouette 0.304):

| Persona | Members | Revenue | ARPU | Washes/mo | Churned | CLV |
|---|---|---|---|---|---|---|
| Power household | 73 (16%) | 35% | \$47 | 5.1 | 10% | \$586 |
| Core regular | 221 (48%) | 52% | \$27 | 2.5 | 5% | \$329 |
| Never activated | 93 (20%) | 11% | \$19 | 0.7 | 58% | \$304 |
| Promo flipper | 78 (17%) | 2% | \$10 | 3.1 | 74% | \$56 |

## What it cannot answer

- **Anything causal.** No experiment in the file. The two biggest effects both have live alternative
  explanations: reverse causation for dormancy (deciding to quit → stopping coming), and near-perfect
  collinearity with month 1 for the price step. An A/B holdout is the only fix.
- **Retail (non-member) behaviour**, seasonality (11 months of a ramping book), site comparison
  (site 3 is 108 members with a different cohort mix), demographics (82.7% Kentucky plates).
- **Unit economics rest on an assumed \$2.25 variable cost per wash** — the export has no cost side.
  The sign of every conclusion holds from \$1 to \$5; the count of unprofitable members (10 → 80)
  does not.

## Environment notes

Two real fault lines in this environment, both worked around in `app.py` with comments:

- `st.dataframe` / `st.table` **segfault** the server on the second script run (pyarrow 25 +
  pandas 3.0.2 + streamlit 1.58). Every table is rendered as HTML instead — same conclusion the
  conclusions app reached (`conclusion/demo/ui.py`).
- Same fault line, different trigger: pandas 3.0 backs its default `str` dtype with Arrow, and a
  boolean mask over such a column calls pyarrow's `compute.take`, which segfaults on Streamlit's
  script thread during a re-run. `app.py` sets `pd.set_option("mode.string_storage", "python")`
  before building any frame, and caches with `cache_resource` (not `cache_data`, which would
  re-introduce Arrow backing through its serialisation).
