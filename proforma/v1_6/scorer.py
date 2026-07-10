"""
Ground-truth scorer — grades the council's call for a focal build against what the site ACTUALLY did.

The focal site's own rows for `date >= T` are the outcome (never shown to the council). We reduce them to:
  • realized_mature_washes — mean total washes over the site's own months 18–30 (the model's label window),
  • realized_ramp          — trailing-3mo vs first-3mo growth, and months_open.
and define a hindsight "good build" (data-relative floor + healthy ramp + reached maturity). Per seat we
then score projection error (APE vs realized) and whether the go/no-go matched the realized outcome.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from proforma.v1_6 import data_1_6 as D


def realized_outcome(focal_key: str, as_of: pd.Timestamp, *, df: Optional[pd.DataFrame] = None,
                     site: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """The focal site's realized post-T trajectory reduced to grading inputs. Pure lookup on the 1.6 CSV."""
    if df is None or site is None:
        df, site = D.load_panel_1_6()
    f = df[df.site_key == focal_key].sort_values("date")
    arr = f.loc[f.rel >= 0, "tot_wash_count"].dropna()            # months from opening onward = the actuals
    mat = f.loc[(f.rel >= D.MAT_LO) & (f.rel <= D.MAT_HI), "tot_wash_count"].dropna()
    realized_mature = float(mat.mean()) if len(mat) >= 4 else np.nan
    first3 = float(arr.iloc[:3].mean()) if len(arr) >= 1 else np.nan
    current3 = float(arr.iloc[-3:].mean()) if len(arr) >= 1 else np.nan
    ramp = (current3 - first3) / first3 if np.isfinite(first3) and first3 else np.nan
    months_open = int(len(arr))

    floor = D.mature_floor(site)
    good = bool(np.isfinite(realized_mature) and realized_mature >= floor
                and np.isfinite(ramp) and ramp > 0 and months_open >= 24)
    return {
        "focal_key": focal_key,
        "realized_mature_washes": realized_mature if np.isfinite(realized_mature) else None,
        "realized_ramp": float(ramp) if np.isfinite(ramp) else None,
        "months_open": months_open,
        "first3_per_month": first3 if np.isfinite(first3) else None,
        "current3_per_month": current3 if np.isfinite(current3) else None,
        "mature_floor": floor,
        "realized_good_build": good,
    }


def ape(predicted: Optional[float], realized: Optional[float]) -> Optional[float]:
    """Absolute percentage error |pred - actual| / actual. None when either side is missing/zero."""
    if predicted is None or realized is None:
        return None
    try:
        predicted, realized = float(predicted), float(realized)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(predicted) or not np.isfinite(realized) or realized == 0:
        return None
    return abs(predicted - realized) / abs(realized)


# go/no-go: a Build/Conditional lean is a "yes". Conditional counts as yes for the 2-class rate but is
# tracked separately by the harness so we can report it on its own.
_YES = {"Build", "Conditional"}


def gonogo_correct(lean: Optional[str], realized_good_build: bool) -> Optional[bool]:
    """Did the seat's go/no-go match reality? None when the seat abstained (no lean)."""
    if lean is None:
        return None
    said_yes = str(lean) in _YES
    return said_yes == bool(realized_good_build)
