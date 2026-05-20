"""Central configuration for the model_3 cold-start forecasting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
ART_DIR = OUT_DIR / "artifacts"
MET_DIR = OUT_DIR / "metrics"
PLOT_DIR = OUT_DIR / "plots"
FORECAST_DIR = OUT_DIR / "forecasts"
SUMMARY_DIR = OUT_DIR / "summaries"

for d in (ART_DIR, MET_DIR, PLOT_DIR, FORECAST_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)

LT_CSV = DATA_DIR / "less_than-2yrs_monthly.csv"
MT_CSV = DATA_DIR / "more_than-2yrs_monthly.csv"
CBSA_SHP = DATA_DIR / "cb_2018_us_cbsa_500k.shp"

# Reference "now": the most recent month with reliable observations.
# Anything strictly after this is treated as future / projection and dropped.
# LT year_number=2 rows are relabelled from 2026 -> 2025 in data_prep before
# this cutoff is applied, so this should be 2025-12.
OBSERVED_CUTOFF_YM = "2025-12"

# CBSA assignment fallback distances (km) when a point falls outside every polygon.
NEAREST_CBSA_KM_MAX = 50.0

# Minimum peers required for a CBSA to be considered "self-sufficient" as a market.
MIN_PEERS_PER_MARKET = 3
# Sparse CBSAs roll up to a "state cohort" then to a "region cohort" then to national.
PEER_ROLLUP_LEVELS = ["cbsa", "state", "region", "national"]

# Maturity definition (months since site opened). Mature window is used for
# computing market reference levels and lifecycle ramp targets.
MATURITY_MONTHS = 24

# Horizon for forecasts (5 years monthly).
FORECAST_HORIZON_MONTHS = 60

# HIT/MISS threshold for annual totals (washes).
HIT_BAND = 20_000

# Random seed used for splits and model training.
SEED = 7

# Temporal split: rows with year_month <= TRAIN_END_YM go to train (used as
# "known history" for site-level features and as model training rows).
# Rows strictly after go to the temporal-eval test set. Matches the
# 2024-train / 2025+test methodology of model_1 and model_2.
TRAIN_END_YM = "2024-12"

# Fraction of sites held out for the cold-start (never-seen) evaluation.
COLD_START_HOLDOUT_FRAC = 0.15

# H3 hex resolution used for sub-CBSA "local submarket" assignment.
# r=8 ~ 0.7 km² hexes (~750m edge); r=7 is ~5 km²; r=6 is ~36 km².
# r=8 is the recommended neighbourhood layer when paired with CBSA macro.
H3_RES = 8
# Minimum peers in the local H3 disk for the local layer to be used. Below
# this we fall back to CBSA macro alone.
MIN_PEERS_PER_H3 = 2
# How many rings to expand around the candidate hex when collecting local
# peers. With r=8 each ring is ~1 km of additional radius.
H3_DISK_RINGS = 1
