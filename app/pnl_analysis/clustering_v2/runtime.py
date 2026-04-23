from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[3]
_V2_DIR = _REPO_ROOT / "daily_data" / "daily-data-modelling" / "clustering_v2"


def _ensure_v2_on_syspath() -> None:
    p = str(_V2_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


@lru_cache(maxsize=1)
def _project_site_module():
    _ensure_v2_on_syspath()
    mod = importlib.import_module("project_site")
    try:
        mod_dir = Path(getattr(mod, "__file__", "")).resolve().parent
    except Exception:
        mod_dir = None
    if mod_dir != _V2_DIR:
        raise RuntimeError(f"Loaded unexpected project_site module from {mod_dir} (expected {_V2_DIR})")
    return mod


@lru_cache(maxsize=1)
def _project_site_quantile_module():
    _ensure_v2_on_syspath()
    mod = importlib.import_module("project_site_quantile")
    try:
        mod_dir = Path(getattr(mod, "__file__", "")).resolve().parent
    except Exception:
        mod_dir = None
    if mod_dir != _V2_DIR:
        raise RuntimeError(f"Loaded unexpected project_site_quantile module from {mod_dir} (expected {_V2_DIR})")
    return mod


def run_projection(
    *,
    address: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    method: str,
    use_opening_prefix_for_mature_forecast: bool = True,
    bridge_opening_to_mature_when_prefix: bool = True,
    allow_nearest_cluster_beyond_distance_cap: bool = False,
    level_model: str = "ridge",
) -> Dict[str, Any]:
    ps = _project_site_module()
    try:
        return ps.run_projection(
            address,
            lat,
            lon,
            method,
            use_opening_prefix_for_mature_forecast=use_opening_prefix_for_mature_forecast,
            bridge_opening_to_mature_when_prefix=bridge_opening_to_mature_when_prefix,
            allow_nearest_cluster_beyond_distance_cap=allow_nearest_cluster_beyond_distance_cap,
            level_model=level_model,
        )
    except SystemExit as e:
        # project_site uses SystemExit for CLI validation; map to a normal exception for Celery/FastAPI.
        raise ValueError(str(e)) from e


def run_quantile_projection(
    *,
    lat: float,
    lon: float,
    method: str,
    address: Optional[str],
    use_opening_prefix_for_mature_forecast: bool = True,
    bridge_opening_to_mature_when_prefix: bool = True,
    allow_nearest_cluster_beyond_distance_cap: bool = False,
) -> Dict[str, Any]:
    psq = _project_site_quantile_module()
    return psq.build_quantile_projection_response(
        float(lat),
        float(lon),
        str(method),
        address,
        use_opening_prefix_for_mature_forecast=use_opening_prefix_for_mature_forecast,
        bridge_opening_to_mature_when_prefix=bridge_opening_to_mature_when_prefix,
        allow_nearest_cluster_beyond_distance_cap=allow_nearest_cluster_beyond_distance_cap,
    )


def build_monthly_wash_projection_48mo(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the clustering_v2 response into a single 48-month timeline (months 1–24 + 25–48)."""
    lt = resp.get("less_than_2yrs") or {}
    mt = resp.get("more_than_2yrs") or {}

    lt_mp = lt.get("monthly_projection") or []
    mt_mp = mt.get("monthly_projection_mature_25_48") or []

    out: List[Dict[str, Any]] = []
    for row in lt_mp:
        try:
            out.append(
                {
                    "month": str(row["month"]),
                    "wash_count": float(row["wash_count"]),
                    "operational_month_index": int(row.get("operational_month_index") or len(out) + 1),
                }
            )
        except Exception:
            continue

    # Mature months 25–48 are returned separately (without operational index) in the v2 response.
    start = 25
    for ii, row in enumerate(mt_mp, start=start):
        try:
            out.append({"month": str(row["month"]), "wash_count": float(row["wash_count"]), "operational_month_index": ii})
        except Exception:
            continue

    out.sort(key=lambda r: int(r.get("operational_month_index") or 0))
    return out
