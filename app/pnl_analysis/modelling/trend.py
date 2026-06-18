"""
Composition-robust trend & smooth-forecast helpers — ported verbatim from streamlits/app.py.

These turn a site's (or a market's) noisy monthly history into:
  • robust_growth(arr)  — one series' annual growth + a data-based CI band (Theil-Sen, shrunk to flat).
  • market_trend(piv)   — a date×site pivot's pooled per-site trend (immune to sites entering/leaving).
  • forecast_series(s)  — a smooth 5-yr expected-trend forecast that starts at the last actual and
                          blends into a trend line growing at a SATURATING annual rate (booms decelerate).

No hand-set ±growth clamp: noisy/thin markets self-widen via the confidence band instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAU_Y = 2.0   # post-maturity growth saturates over ~2 yr (sites plateau ~24 mo empirically)


def sat_years(years):
    """Saturating 'effective years' of drift: ≈ linear near 0, asymptotes to TAU_Y (booms decelerate)."""
    return TAU_Y * (1.0 - np.exp(-np.asarray(years, dtype=float) / TAU_Y))


def _robust_slope(arr):
    """Per-month log-growth slope (Theil-Sen on 6-mo-smoothed log level, last ~30 mo) + its SE.
    SE inflated by √6 for the smoothing autocorrelation. Returns (slope, se) or None if too short."""
    arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
    if len(arr) < 18:
        return None
    sm = pd.Series(arr).rolling(6, min_periods=3).mean().dropna().to_numpy()
    K = min(30, len(sm))
    if K < 8:
        return None
    yv = np.log(np.clip(sm[-K:], 1.0, None)); xv = np.arange(K, dtype=float)
    try:
        from scipy.stats import theilslopes
        sl, _, lo, hi = theilslopes(yv, xv)
    except Exception:
        sl = float(np.polyfit(xv, yv, 1)[0]); lo = hi = sl
    se = max((hi - lo) / (2 * 1.96), 1e-9) * np.sqrt(6.0)
    return float(sl), float(se)


def _shrink_annualize(sl, se):
    """Shrink a per-month slope toward 0 by its signal-to-noise t²/(1+t²), annualize → (g, g_lo, g_hi).
    Loose ±40%/yr sanity rail only stops a degenerate series exploding — NOT the old [-5%,+8%] clamp."""
    SANE = lambda r: float(np.clip(r, -0.40, 0.40))
    t = abs(sl) / se
    sl_c = sl * (t * t / (1.0 + t * t))
    return SANE(np.exp(sl_c * 12) - 1), SANE(np.exp((sl - 1.96 * se) * 12) - 1), SANE(np.exp((sl + 1.96 * se) * 12) - 1)


def robust_growth(arr):
    """Annual growth (central + data-based CI band) for ONE series — no hand-set ±clamp."""
    g = _robust_slope(arr)
    return _shrink_annualize(*g) if g else (0.0, 0.0, 0.0)


def market_trend(piv):
    """Composition-robust market trend from a date×site pivot: each site's OWN robust slope, pooled by
    MEDIAN; pooled SE from the between-site spread and within-site error. Returns (g, g_lo, g_hi)."""
    slopes, ses = [], []
    for col in getattr(piv, "columns", []):
        r = _robust_slope(piv[col].dropna().to_numpy())
        if r:
            slopes.append(r[0]); ses.append(r[1])
    if not slopes:
        return 0.0, 0.0, 0.0
    slopes = np.asarray(slopes); ses = np.asarray(ses); n = len(slopes)
    sl = float(np.median(slopes))
    between = float(np.std(slopes, ddof=1)) if n >= 2 else 0.0
    se = max(np.hypot(between, float(np.median(ses))) / np.sqrt(n), 1e-9)
    return _shrink_annualize(sl, se)


def forecast_series(s, H, g=None):
    """SMOOTH 5-yr expected-trend forecast (no repeating seasonal photocopy). Starts exactly at the last
    actual value and blends over ~a quarter into a trend line at the recent deseasonalized LEVEL, growing
    at a ROBUST annual rate (`robust_growth`), SATURATING over ~2 yr so a boom decelerates."""
    s = pd.Series(s).astype(float).dropna()
    n = len(s)
    if n == 0:
        return np.zeros(H)
    arr = s.to_numpy()
    last = float(arr[-1])
    level = float(arr[-min(12, n):].mean())
    if g is None:
        g = robust_growth(arr)[0]
    t = np.arange(1, H + 1)
    trend = level * (1 + g) ** sat_years(t / 12.0)
    w = np.exp(-(t - 1) / 3.0)
    return np.clip(last * w + trend * (1 - w), 0, None)
