"""Central configuration for the neighbour-based spatial prediction model.

Single source of truth for paths, geographic params, target KPIs and cleaning
options. Mirrors the conventions established in notebooks/cluster_pattern.ipynb.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]          # final_modelling/
DATA_PATH = ROOT / "data" / "monthly_kpi.csv"
# Per-package file (one row per package_name) that also carries the
# membership/retail share columns — full export covering all sites.
PCT_DATA_PATH = ROOT / "data" / "monthly_kpi_agg.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
BBOX_PATH = ARTIFACTS_DIR / "cluster_bboxes.json"
EVAL_PATH = ARTIFACTS_DIR / "eval_results.csv"

# --- Data filtering (matches the notebook) ---
PACKAGE = "Monthly Recurring Plan"
# In monthly_withpackage.csv the recurring-plan category lives in `package_plan`
# (one row per package_name underneath it), so the share file filters on that.
PCT_PACKAGE_PLAN = "Monthly Recurring Plan"

# --- Geography ---
EARTH_KM = 6371.0088          # mean Earth radius for haversine (notebook constant)
BUFFER_KM = 20.0              # cluster / neighbourhood radius proven to co-move
MIN_SITES = 3                 # min sites to call a geographic cluster (for bboxes)

# --- Neighbour search ---
MIN_NEIGHBOURS = 3            # below this within BUFFER_KM -> KNN fallback
KNN_FALLBACK_K = 5           # k nearest sites used when the radius is too sparse
IDW_EPS_KM = 1e-3            # avoids div-by-zero for a coincident neighbour

# --- Cohort / evaluation ---
COHORT_MIN_MONTHS = 12       # sites with >=12 months drive validation
MIN_OVERLAP_MONTHS = 3       # min overlapping months to score a (site, KPI) cell

# --- Target KPIs (the 5 metrics we predict) ---
TARGET_KPIS = [
    "membership_purchased_count",
    "membership_wash_count",
    "asp_per_membership",
    "retail_wash_count",
    "asp_per_retail_wash",
]

# ASP metrics are ratios (price per wash / per membership), not volumes.
RATIO_KPIS = {"asp_per_membership", "asp_per_retail_wash"}

KPI_LABELS = {
    "membership_purchased_count": "Memberships sold",
    "membership_wash_count": "Membership washes",
    "asp_per_membership": "ASP / membership ($)",
    "retail_wash_count": "Retail washes",
    "asp_per_retail_wash": "ASP / retail wash ($)",
}

# --- Membership vs retail share (%) ---
# Monthly percentage-share metrics, recomputed at site-month level from the
# underlying amounts/counts (see data_loader.load_pct_panels). Each pair sums to
# 100: sales share splits membership vs retail dollars, wash share splits washes.
PCT_KPIS = [
    "membership_pct_sales",
    "retail_pct_sales",
    "membership_pct_wash",
    "retail_pct_wash",
]

PCT_KPI_LABELS = {
    "membership_pct_sales": "Membership % of sales",
    "retail_pct_sales": "Retail % of sales",
    "membership_pct_wash": "Membership % of washes",
    "retail_pct_wash": "Retail % of washes",
}

# --- Data cleaning ---
WINSORIZE = True             # clip each KPI to WINSOR_PCT to tame impossible spikes
WINSOR_PCT = (0.01, 0.99)    # 1st / 99th percentile caps (computed on the MRP set)
DROP_FLATLINE = True         # exclude sites whose whole history is 0 for a KPI

# Defensive rename: a past version of the CSV repeated the
# `membership_wash_count` header, so pandas loaded the 9th column as
# `membership_wash_count.1` when it was really asp_per_membership.
LEGACY_RENAME = {"membership_wash_count.1": "asp_per_membership"}
