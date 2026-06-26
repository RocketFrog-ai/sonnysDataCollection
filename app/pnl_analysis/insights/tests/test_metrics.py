"""
Unit tests for the pure metrics node — no LLM, no network, hand-computable numbers.

Synthetic market: 2 constant incumbents (A, B) + 1 ramping entrant (C, opens 2023-06). Every
site prices retail at $20/wash and membership at $30/purchase, so the revenue-weighted market ASPs
are exactly $20 / $30 and easy to assert (this also guards the mem_revenue/mem_purchase_count ASP
definition the chart uses).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, so `app.*` imports
from app.pnl_analysis.insights.metrics import compute_metrics

MS = "MS"


def _incumbent_rows(site_key, months):
    return [dict(site_key=site_key, date=d, mem_wash_count=1000, ret_wash_count=500,
                 mem_revenue=30000, ret_revenue=10000, mem_purchase_count=1000,
                 tot_wash_count=1500, tot_revenue=40000) for d in months]


def _entrant_rows(site_key, months):
    rows = []
    for i, d in enumerate(months):
        mw, rw = 100 + 50 * i, 50
        mrev, rrev = mw * 30, rw * 20
        rows.append(dict(site_key=site_key, date=d, mem_wash_count=mw, ret_wash_count=rw,
                         mem_revenue=mrev, ret_revenue=rrev, mem_purchase_count=mw,
                         tot_wash_count=mw + rw, tot_revenue=mrev + rrev))
    return rows


def _meta(rows):
    return pd.DataFrame(rows)


def _full_market():
    full = pd.date_range("2020-01-01", "2024-12-01", freq=MS)
    cmonths = pd.date_range("2023-06-01", "2024-12-01", freq=MS)  # i = 0..18 -> 19 rows
    panel = pd.DataFrame(_incumbent_rows("A", full) + _incumbent_rows("B", full) + _entrant_rows("C", cmonths))
    meta = _meta([
        dict(site_key="A", name="Site 1", op_start=pd.Timestamp("2019-06-01"), dist_km=1.0, is_entrant=False, left_censored=True),
        dict(site_key="B", name="Site 2", op_start=pd.Timestamp("2019-06-01"), dist_km=2.0, is_entrant=False, left_censored=True),
        dict(site_key="C", name="Site 3", op_start=pd.Timestamp("2023-06-01"), dist_km=0.5, is_entrant=True, left_censored=False),
    ])
    return panel, meta


def test_main_market():
    panel, meta = _full_market()
    m = compute_metrics(panel, meta, "C", last_n_months=12)

    assert m["meta"]["n_sites"] == 3
    assert m["meta"]["n_entrants"] == 1
    assert m["meta"]["has_entrant"] is True

    # weighted market ASPs are exactly $20 / $30 (guards the mem_purchase_count definition)
    assert abs(m["asps"]["retail"]["current"] - 20.0) < 1e-9
    assert abs(m["asps"]["membership"]["current"] - 30.0) < 1e-9

    # membership share of washes (3-mo averaged near the end) ~0.738
    assert 0.72 < m["washes"]["membership_share"]["current"] < 0.75

    # total washes YoY (3-mo-averaged, de-spiked): ~ +17-18%
    assert 0.14 < m["washes"]["total"]["yoy"] < 0.20

    ee = m["washes"]["entry_effect"]
    assert ee is not None
    assert abs(ee["pre_per_month"] - 3000) < 1e-9
    assert ee["months_pre"] == 12 and ee["months_post"] == 12
    assert ee["change"] > 0

    fr = m["washes"]["focal_ramp"]
    assert fr is not None
    assert fr["months_open"] == 19
    assert abs(fr["ramp"] - 4.25) < 1e-6        # 1050/200 - 1
    assert fr["short_history"] is False

    cb = m["washes"]["cannibalization"]
    assert cb is not None
    assert abs(cb["retail_change"]) < 1e-9       # incumbent retail is constant -> no cannibalization

    assert m["asps"]["focal_gap"] is not None
    assert m["revenue"]["focal_contribution"] is not None

    json.dumps(m)  # whole dict is JSON-serializable (no numpy / NaN leaks)


def test_single_site():
    full = pd.date_range("2020-01-01", "2024-12-01", freq=MS)
    panel = pd.DataFrame(_incumbent_rows("A", full))
    meta = _meta([dict(site_key="A", name="Site 1", op_start=pd.Timestamp("2019-06-01"),
                       dist_km=1.0, is_entrant=False, left_censored=True)])
    m = compute_metrics(panel, meta, "A")
    assert m["flags"]["single_site"] is True
    assert m["flags"]["no_entrant"] is True
    assert m["washes"]["cannibalization"] is None
    assert m["washes"]["entry_effect"] is None
    assert m["asps"]["focal_gap"] is None
    json.dumps(m)


def test_no_entrant():
    full = pd.date_range("2020-01-01", "2024-12-01", freq=MS)
    panel = pd.DataFrame(_incumbent_rows("A", full) + _incumbent_rows("B", full))
    meta = _meta([
        dict(site_key="A", name="Site 1", op_start=pd.Timestamp("2019-06-01"), dist_km=1.0, is_entrant=False, left_censored=True),
        dict(site_key="B", name="Site 2", op_start=pd.Timestamp("2019-06-01"), dist_km=2.0, is_entrant=False, left_censored=True),
    ])
    m = compute_metrics(panel, meta, "A")  # focal falls back to nearest incumbent
    assert m["flags"]["no_entrant"] is True
    assert m["washes"]["entry_effect"] is None
    assert m["washes"]["focal_ramp"] is None
    assert m["revenue"]["focal_contribution"] is None


def test_short_history_yoy_none():
    full = pd.date_range("2024-01-01", "2024-12-01", freq=MS)  # only 12 months
    panel = pd.DataFrame(_incumbent_rows("A", full) + _incumbent_rows("B", full))
    meta = _meta([
        dict(site_key="A", name="Site 1", op_start=pd.Timestamp("2019-06-01"), dist_km=1.0, is_entrant=False, left_censored=True),
        dict(site_key="B", name="Site 2", op_start=pd.Timestamp("2019-06-01"), dist_km=2.0, is_entrant=False, left_censored=True),
    ])
    m = compute_metrics(panel, meta, "A", last_n_months=12)
    assert m["washes"]["total"]["yoy"] is None   # no month 12 prior in the window
    json.dumps(m)


def test_left_censored_focal():
    full = pd.date_range("2020-01-01", "2024-12-01", freq=MS)
    cmonths = pd.date_range("2023-06-01", "2024-12-01", freq=MS)
    panel = pd.DataFrame(_incumbent_rows("A", full) + _incumbent_rows("B", full) + _entrant_rows("C", cmonths))
    meta = _meta([
        dict(site_key="A", name="Site 1", op_start=pd.Timestamp("2019-06-01"), dist_km=1.0, is_entrant=False, left_censored=True),
        dict(site_key="B", name="Site 2", op_start=pd.Timestamp("2019-06-01"), dist_km=2.0, is_entrant=False, left_censored=True),
        dict(site_key="C", name="Site 3", op_start=pd.Timestamp("2023-06-01"), dist_km=0.5, is_entrant=True, left_censored=True),
    ])
    m = compute_metrics(panel, meta, "C")
    assert m["flags"]["focal_left_censored"] is True
    assert m["washes"]["entry_effect"] is None    # no clean entry window
    assert m["washes"]["focal_ramp"] is None


if __name__ == "__main__":
    test_main_market()
    test_single_site()
    test_no_entrant()
    test_short_history_yoy_none()
    test_left_censored_focal()
    print("all metrics tests passed")
