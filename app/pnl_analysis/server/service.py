"""Logic the /v1/pnl_analysis/* route handlers used to inline: lat/lon resolution shared by every
pin-driven endpoint, local-market data assembly for the grounded Key-Insights pipeline, and the
Key-Insights response shaping. Split out of the former monolithic routes.py so router.py's handlers
stay thin delegators; the heavier forecasting/market math itself still lives in
app.pnl_analysis.modelling.* and is called directly from router.py."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from fastapi import HTTPException

from app.core import common as calib
from app.pnl_analysis.modelling import data as D
from app.pnl_analysis.modelling import market

logger = logging.getLogger(__name__)

WASH_METRICS = {"mem_share_wash", "mem_wash_count", "ret_wash_count"}


def resolve_lat_lon(latitude, longitude, address) -> Tuple[float, float]:
    """lat/lon if given, else geocode the address via TomTom. 400 if neither resolves."""
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude)
    if not address:
        raise HTTPException(status_code=400, detail="Provide either latitude/longitude or address.")
    try:
        return calib.resolve_lat_lon(address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def grounded_inputs(lat, lon, radius_km, min_months, demo):
    """Build (panel, meta, focal) for the grounded Key-Insights pipeline over the local market — the SAME subset
    the KPI panels draw, with ASP recomputed per the Streamlit definitions. Returns None if no rich-history sites.
    Used by /insights."""
    df, site = D.load_panel()
    site_rich = site[site.n_obs >= min_months]
    nb = market._neighbourhood(site_rich, lat, lon, radius_km)
    if nb.empty:
        return None
    focal = market._focal_key(nb)
    panel = df[df.site_key.isin(nb.site_key.tolist())].copy()
    panel["asp_ret"] = panel.ret_revenue / panel.ret_wash_count.replace(0, np.nan)     # ASP as the chart draws it
    panel["asp_mem"] = panel.mem_revenue / panel.mem_purchase_count.replace(0, np.nan)  # membership ASP via purchases
    meta = nb[["site_key", "op_start", "dist_km", "is_entrant", "left_censored"]].copy()
    if demo:
        anon = {k: f"Site {i + 1}" for i, k in enumerate(nb.sort_values("op_start").site_key)}
        meta["name"] = meta.site_key.map(anon)
    else:
        meta["name"] = meta.site_key.map(site.set_index("site_key").client_name.to_dict())
    return panel, meta, focal


def known_site_names(lat, lon, radius_km, min_months, demo):
    """The client's OWN car washes in the radius (their portfolio) — names fed to the competition read so the LLM
    can cross-reference them. [] in demo (don't leak identities)."""
    if demo:
        return []
    _, site = D.load_panel()
    pool = site[site.n_obs >= min_months] if min_months > 1 else site
    nb = market._neighbourhood(pool, lat, lon, radius_km)
    return [str(n) for n in nb.client_name.dropna().tolist()] if not nb.empty else []


def render_insights_summary(blocks) -> dict:
    """Post-process the grounded Key-Insights pipeline's per-group narrative blocks (Washes/Revenue/ASPs)
    into the single `{summary}` response: drop empty/neutral/error placeholders, keep only substantive
    blocks, falling back to everything if nothing looked substantive."""
    def _has_substance(v: str) -> bool:  # the model returns one holistic block — drop empty/neutral/error placeholders
        low = (v or "").lower()
        if any(s in low for s in ("did not return", "could not generate", "generation failed")):
            return False
        return ("\n- " in v) or (len(v.strip()) > 60)

    parts = [v.strip() for v in blocks.values() if v and _has_substance(v)]
    return {"summary": "\n\n".join(parts) or "\n\n".join(v.strip() for v in blocks.values() if v)}
