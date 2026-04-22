"""Training panel paths for clustering_v2 (greenfield / Streamlit / peer overlays).

These are the same inputs used by ``build_v2.py`` / ``build_quantile_v2.py``:
``daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv`` and
``daily_data/daily-data-modelling/master_more_than-2yrs.csv`` next to this package.
"""

from __future__ import annotations

from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
MODELLING_DATA_DIR = V2_DIR.parent
LESS_THAN_CLUSTERING_READY_CSV = MODELLING_DATA_DIR / "less_than-2yrs-clustering-ready.csv"
MASTER_MORE_THAN_2YRS_CSV = MODELLING_DATA_DIR / "master_more_than-2yrs.csv"
