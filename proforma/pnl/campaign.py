"""Shared campaign helpers — the promo/event-study math used by BOTH the API
(app/pnl_analysis/modelling/campaign.py orchestration) and the Streamlit UI
(proforma/ui/panels/_pinpoint_forecast.py). ONE copy; see docs/DIVERGENCES.md §1.

  campaign_conv_pct — retail->membership conversion rate by membership share.
  campaign_effect   — per-month (mem, ret, opex) multipliers over the promo window + tail.
  campaign_months_by_site / _campaigns_df — detect real promo OPEX spikes from the P&L panel.

PURE module: numpy / pandas only, NO streamlit. The P&L panel comes from proforma.pnl.data.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from proforma.pnl import data as D


# ── campaign signal — what the operators' P&L actually shows (event study on opex-data.csv) ──
# A "campaign" = a promotional OPEX spike. HONEST read: the apparent 12-month "lift" in the raw event
# study is mostly the site's own organic ramp. Once each site's trend is removed the clean incremental
# effect is a SHORT (~1–6 month) retail→MEMBERSHIP CONVERSION — biggest where there's retail headroom
# (low membership share), best ROI in dense markets. The promo OPEX is front-loaded (hot launch, short tail).
CAMP_OPEX_TAIL = [1.33, 1.17, 1.10, 1.05, 1.03, 1.02]    # opex multiplier during the campaign window (the spend)


def campaign_conv_pct(mem_share: float) -> float:
    """Membership-wash lift a campaign delivers, scaled by the site's membership share (= retail headroom).
    From the timing analysis: low-share sites convert the most retail customers into members."""
    if mem_share < 0.65:
        return 0.30          # lots of retail to convert (data: +36% lift; tempered for the new-site case)
    if mem_share < 0.78:
        return 0.14
    return 0.07              # already mostly members → little headroom


def campaign_effect(launch, mem_share, intensity=1.0, window=6, horizon=61):
    """Per-month multipliers (membership washes, retail washes, opex) for a SHORT campaign. The effect is a
    retail→membership conversion that ramps in over ~2 months, holds through `window` months, then fades
    (members partly stick, ~12-mo half-life). Membership washes lift by the share-scaled amount; retail
    gives up ~half as many washes. OPEX carries the promo spend over the window. Returns (mem, ret, opx)
    numpy arrays of length `horizon` (kept as numpy for pnl.py / internal callers that multiply them in)."""
    lift = campaign_conv_pct(mem_share) * intensity
    mem, ret, opx = np.ones(horizon), np.ones(horizon), np.ones(horizon)
    for t in range(horizon):
        k = t - launch
        if k < 0:
            continue
        if k < window:
            ramp = min(1.0, (k + 1) / 2.0)                          # conversion ramps in over ~2 months
            mem[t] = 1 + lift * ramp
            ret[t] = 1 - 0.5 * lift * ramp                          # members wash more → retail falls ~half as much
            opx[t] = 1 + (CAMP_OPEX_TAIL[k] - 1) * intensity if k < len(CAMP_OPEX_TAIL) else 1.0
        else:
            f = 0.5 ** ((k - window) / 12.0)                        # membership base partly sticks, slowly fades
            mem[t] = 1 + lift * f
            ret[t] = 1 - 0.5 * lift * f
    return mem, ret, opx


# ─────────────────────────── campaign-spike detection (opex-data.csv event study) ───────────────────────────
def campaign_months_by_site() -> Dict[str, List[str]]:
    """site_key -> list of campaign month ISO date strings (real promo OPEX spikes), from the campaign panel.
    Spike = true_opex (cogs+expenses) > median+3·MAD AND > 1.3× trailing-6mo median; interior months only.
    Mirrors app.campaign_months_by_site(); reads via D.load_campaign_panel() instead of the csv directly."""
    p = D.load_campaign_panel().copy()
    p["date"] = pd.to_datetime(p["report_date"]).dt.to_period("M").dt.to_timestamp()   # month start (matches main-ds)
    p["true_opex"] = p.cogs.fillna(0) + p.expenses.fillna(0)
    # one row per (site, month): keep the real financial row, dropping all-zero artifact duplicates
    p = p.sort_values("total_income").drop_duplicates(["site_key", "date"], keep="last")
    out: Dict[str, List[str]] = {}
    for sk, g in p.sort_values("date").groupby("site_key"):
        s = g.set_index("date")["true_opex"]
        if len(s) < 12:
            continue
        med = s.median(); mad = 1.4826 * (s - med).abs().median()
        troll = s.shift(1).rolling(6, min_periods=4).median()
        spike = (s > med + 3 * mad) & (s > 1.3 * troll)
        cutoff = s.index[-3]                                        # need a post-window → drop last 3 months
        dates = [d for d in s.index[spike.fillna(False)] if d <= cutoff]
        if dates:
            out[str(sk)] = [d.strftime("%Y-%m-%d") for d in dates]
    return out


def _campaigns_df() -> pd.DataFrame:
    """Detect OPEX spikes (true_opex > 1.2× trailing-6mo mean) and cluster consecutive spike months
    (gap ≤ 1) into campaigns. Returns a DataFrame (site_key / campaign_start / duration_months).
    Mirrors app._campaigns_df(); reads via D.load_campaign_panel()."""
    data = D.load_campaign_panel()
    sub = data.sort_values(["site_key", "report_date"]).copy()
    sub["report_date"] = pd.to_datetime(sub["report_date"])
    sub["true_opex"] = sub["cogs"] + sub["expenses"]
    sub["opex_baseline"] = (sub.groupby("site_key")["true_opex"]
                            .transform(lambda s: s.shift(1).rolling(6, min_periods=4).mean()))
    sub["opex_vs_baseline"] = sub["true_opex"] / sub["opex_baseline"]
    spikes = sub[sub["opex_vs_baseline"] > 1.2].copy()
    records: List[Dict[str, Any]] = []
    for site_key, grp in spikes.sort_values("report_date").groupby("site_key"):
        rows = grp.reset_index(drop=True); i = 0
        while i < len(rows):
            start_date = rows.loc[i, "report_date"]; months = [rows.loc[i, "report_date"]]; j = i + 1
            while j < len(rows):
                gap = ((rows.loc[j, "report_date"].year - rows.loc[j - 1, "report_date"].year) * 12 +
                       (rows.loc[j, "report_date"].month - rows.loc[j - 1, "report_date"].month))
                if gap <= 1:
                    months.append(rows.loc[j, "report_date"]); j += 1
                else:
                    break
            records.append({"site_key": site_key, "campaign_start": start_date, "duration_months": len(months)}); i = j
    return pd.DataFrame(records)
