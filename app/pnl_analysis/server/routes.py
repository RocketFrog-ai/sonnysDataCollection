from __future__ import annotations

import logging
from typing import Tuple

from fastapi import APIRouter, HTTPException

from app.utils import common as calib
from app.pnl_analysis.modelling import market
from app.pnl_analysis.modelling import pnl as pnl_engine
from app.pnl_analysis.modelling import campaign as campaign_engine
from app.pnl_analysis.server.models import (
    ExploreMarketRequest,
    ExploreKpisRequest,
    PinpointForecastRequest,
    PnlForecastRequest,
    CampaignVerdictRequest,
    EatingMarketRequest,
    LocalCampaignsRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl_analysis")

_WASH_METRICS = {"mem_share_wash", "mem_wash_count", "ret_wash_count"}


def _resolve_lat_lon(latitude, longitude, address) -> Tuple[float, float]:
    """lat/lon if given, else geocode the address via TomTom. 400 if neither resolves."""
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude)
    if not address:
        raise HTTPException(status_code=400, detail="Provide either latitude/longitude or address.")
    try:
        return calib.resolve_lat_lon(address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────── lookups ───────────────────────────
@router.get("/brands")
def get_brands():
    """Operator/brand dropdown for the Forecast tab: client_id (model key) + client_name + site count +
    whether the cold-start model has a learned prior for it."""
    return market.list_brands()


@router.get("/operators")
def get_operators():
    """Operator/brand names for the Explore-markets 'highlight operator' dropdown (client_name, alphabetical)."""
    return market.list_operators()


# ─────────────────────────── Explore-markets (tab 1) ───────────────────────────
@router.post("/explore-market")
def explore_market(req: ExploreMarketRequest):
    """Tab 1 — the local-market MAP + header counts. Returns the in-market site markers (role-tagged
    focal/entrant/incumbent), geographic reference dots, and the highlighted operator's footprint — no
    time series (those come from /explore-market/kpis). Backs the map + the 4 header metric cards."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.explore_market(
        lat=lat, lon=lon, radius_km=req.radius_km, max_sites=req.max_sites,
        min_months=req.min_months, operator=req.operator, demo=req.demo,
    )


@router.post("/explore-market/kpis")
def explore_market_kpis(req: ExploreKpisRequest):
    """Tab 1 — the 6 grouped per-site KPI series (washes x3 / revenue x3 / ASP x2) for the whole local market
    (every in-radius site with >= `min_months` of history). Backs the 'Local-market KPIs over time' panels."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.explore_market_kpis(
        lat=lat, lon=lon, radius_km=req.radius_km, smoothing=req.smoothing,
        min_months=req.min_months, demo=req.demo,
    )


# ─────────────────────────── Forecast (tab 2) ───────────────────────────
@router.post("/pinpoint-forecast")
def pinpoint_forecast(req: PinpointForecastRequest):
    """Tab 2 — the NEW SITE's own predicted 5-year monthly trajectory (total/membership/retail with P10-P90
    bands) + the summary KPI cards. The whole-market growth plot is a separate call: POST /market-forecast."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.pinpoint_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        horizon_months=req.horizon_months,
    )


@router.post("/market-forecast")
def market_forecast(req: PinpointForecastRequest):
    """Tab 2 — the TOTAL LOCAL-MARKET wash count: actual history + 5-year forecast (with vs without the new
    site, a trend-CI band, and the entrant's own journey). Same inputs as /pinpoint-forecast."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.market_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        horizon_months=req.horizon_months,
    )


@router.post("/pnl-forecast")
def pnl_forecast(req: PnlForecastRequest):
    """Tab 2 — the 💰 P&L chart: monthly revenue vs operating expense vs net over the 5-year horizon, with an
    optional retail→membership conversion campaign overlay. Revenue = forecast washes × cluster ASP; opex = the
    learned new-site ramp × mature $/wash (scope-aware), escalated by the cost-growth input."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return pnl_engine.pnl_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        asp_override=req.asp_override, opex_growth_pct=req.opex_growth_pct,
        campaign_on=req.campaign_on, campaign_launch=req.campaign_launch,
        campaign_intensity=req.campaign_intensity, window=req.window, horizon_months=req.horizon_months,
    )


@router.post("/campaign/verdict")
def campaign_verdict(req: CampaignVerdictRequest):
    """Tab 2 — 🎯 the campaign recommendation + the 3 supporting metrics (neighbours' membership share, established
    incumbents, this site's predicted membership). Only recommends a promo where the membership market is proven."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return campaign_engine.campaign_verdict(
        lat=lat, lon=lon, radius_km=req.radius_km, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
    )


@router.post("/campaign/eating-the-market")
def eating_the_market(req: EatingMarketRequest):
    """Tab 2 — 📈 your site vs each incumbent, each forecast forward 5 years; with a campaign, the incumbents drift
    down as your promo steals their retail share (theft scales with market density, recovers as the promo fades)."""
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return campaign_engine.eating_the_market(
        lat=lat, lon=lon, radius_km=req.radius_km, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        campaign_on=req.campaign_on, campaign_launch=req.campaign_launch,
        campaign_intensity=req.campaign_intensity, window=req.window, max_incumbents=req.max_incumbents,
    )


@router.get("/campaign/snapshot")
def campaign_snapshot():
    """Tab 2 — what a campaign does to OPEX / Revenue / Profit / Membership purchases, from the real P&L event
    study (book_v4): median values by month-offset for 1-month / 2-month / 3+-month campaigns. Not pin-specific."""
    return campaign_engine.campaign_snapshot()


@router.post("/campaign/local-evidence")
def local_campaign_evidence(req: LocalCampaignsRequest):
    """Tab 2 — real campaigns in this local market: the nearest in-radius sites' monthly series for the chosen
    metric, with each site's detected promo-OPEX-spike months marked — the evidence behind the campaign model."""
    if req.metric not in _WASH_METRICS:
        raise HTTPException(status_code=400, detail=f"metric must be one of: {', '.join(sorted(_WASH_METRICS))}")
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return campaign_engine.local_campaign_evidence(
        lat=lat, lon=lon, radius_km=req.radius_km, metric=req.metric, max_sites=req.max_sites, demo=req.demo,
    )
