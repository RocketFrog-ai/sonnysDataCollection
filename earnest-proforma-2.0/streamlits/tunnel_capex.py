"""
Tunnel-length → CAPEX model, learned from the enriched proforma data
(earnest-proforma-final-1.6/data/old-excel-proforma-data-enriched.csv).

187 real builds carry BOTH a tunnel length and a non-zero total investment, and CAPEX climbs clearly
with tunnel length (corr ~0.53, ~$98k per metre). When the user picks an expected tunnel length for a
new forecast, we set the build CAPEX from the median of real builds at that length.

Public surface (pure functions — the Streamlit app wraps these in st.cache_data):
  • BANDS                 — the 5-m length bands offered as options.
  • load_builds()         — cleaned per-build (tlen, capex) points with capex>0 (empty df if CSV absent).
  • capex_band_table()    — per-band {band, lo, hi, n, median, mean}; medians made non-decreasing so a
                            longer tunnel never shows a lower CAPEX (raw points stay honest in the scatter).
  • capex_for_band(label) — the band's (monotonic) median CAPEX → the auto-set build CAPEX.
  • fit()                 — (slope $/m, intercept $, corr, n) of CAPEX ~ length for the caption.

Tunnel length is on the data's own scale (median actual ≈ 38, consistent with the metres used by the
explore-markets tunnel-length proxy). If the CSV can't be read, baked-in band medians keep the picker
working (the scatter simply notes the data file is unavailable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 5-metre bands: (label, lo inclusive, hi exclusive); the last band is open-ended.
BANDS = [("<25 m", 0, 25), ("25–30 m", 25, 30), ("30–35 m", 30, 35),
         ("35–40 m", 35, 40), ("40–45 m", 40, 45), ("45 m+", 45, 1e9)]

# Baked fallback medians (from the 187 real builds) used only if the CSV can't be read.
_FALLBACK = {"<25 m": 2_200_000, "25–30 m": 2_365_000, "30–35 m": 3_950_000,
             "35–40 m": 4_200_000, "40–45 m": 5_462_538, "45 m+": 6_438_682}

_C_TLA = "tunnel_length_actual"
_C_TLP = "tunnel_length_predicted"
_C_CAP = "project_cost_total_investment[car_wash_acquisition_budget]"

# Candidate CSV locations (repo-root-relative); first that exists wins.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSV_CANDIDATES = [
    _REPO_ROOT / "earnest-proforma-final-1.6" / "data" / "old-excel-proforma-data-enriched.csv",
    Path(__file__).resolve().parent.parent / "data" / "old-excel-proforma-data-enriched.csv",
]


def csv_path() -> Path | None:
    for p in _CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce a possibly $-/comma-formatted column to float."""
    if col not in df.columns:
        return pd.Series([np.nan] * len(df))
    return pd.to_numeric(df[col].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")


def load_builds() -> pd.DataFrame:
    """Per-build (tlen, capex) for real builds with capex>0; uses actual length, falls back to predicted."""
    p = csv_path()
    if p is None:
        return pd.DataFrame(columns=["tlen", "capex"])
    df = pd.read_csv(p, low_memory=False)
    tlen = _num(df, _C_TLA).fillna(_num(df, _C_TLP))
    out = pd.DataFrame({"tlen": tlen, "capex": _num(df, _C_CAP)}).dropna()
    return out[(out.capex > 0) & (out.tlen > 0)].reset_index(drop=True)


def capex_band_table() -> pd.DataFrame:
    """Per-band CAPEX summary with non-decreasing medians (monotonic in length)."""
    b = load_builds()
    rows, running = [], 0.0
    for lbl, lo, hi in BANDS:
        if len(b):
            seg = b[(b.tlen >= lo) & (b.tlen < hi)]
            med = float(seg.capex.median()) if len(seg) else np.nan
            mean = float(seg.capex.mean()) if len(seg) else np.nan
            n = int(len(seg))
        else:
            med, mean, n = np.nan, np.nan, 0
        if not np.isfinite(med):
            med = float(_FALLBACK[lbl])
        med = max(med, running)                       # enforce non-decreasing across bands
        running = med
        rows.append({"band": lbl, "lo": lo, "hi": hi, "n": n, "median": med, "mean": mean})
    return pd.DataFrame(rows)


def capex_for_band(label: str) -> float:
    t = capex_band_table()
    r = t[t.band == label]
    return float(r["median"].iloc[0]) if len(r) else float(_FALLBACK.get(label, 0.0))


def band_center(label: str) -> float:
    for lbl, lo, hi in BANDS:
        if lbl == label:
            return (lo + min(hi, lo + 10)) / 2.0      # cap the open-ended top band's "center" sensibly
    return float("nan")


def fit():
    """(slope $/m, intercept $, corr, n) of CAPEX ~ tunnel length, or (None, None, None, n) if too few."""
    b = load_builds()
    if len(b) < 10:
        return (None, None, None, len(b))
    slope, intercept = np.polyfit(b.tlen, b.capex, 1)
    corr = float(np.corrcoef(b.tlen, b.capex)[0, 1])
    return (float(slope), float(intercept), corr, int(len(b)))
