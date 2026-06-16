from __future__ import annotations

import logging
from typing import Tuple

from fastapi import APIRouter, HTTPException

from app.utils import common as calib
from app.pnl_analysis.modelling import market
from app.pnl_analysis.server.models import (
    ExploreMarketRequest,
    PinpointForecastRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl_analysis")


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


@router.post("/explore-market")
def explore_market(req: ExploreMarketRequest):
    """
    Tab 1 — interactive local-market explorer.
    Returns plot-ready per-site monthly series for every site within `radius_km` of the pin, role-tagged
    (entrant vs incumbent), plus entry markers and axis labels for the frontend line charts.
    """
    if req.metric not in market.METRIC_LABEL_BY_COL:
        raise HTTPException(
            status_code=400,
            detail=f"metric must be one of: {', '.join(market.METRIC_LABEL_BY_COL)}",
        )
    if req.x_axis not in ("date", "months_since_open"):
        raise HTTPException(status_code=400, detail="x_axis must be 'date' or 'months_since_open'.")
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.explore_market(
        lat=lat, lon=lon, metric_col=req.metric, radius_km=req.radius_km,
        smoothing=req.smoothing, max_sites=req.max_sites, x_axis=req.x_axis,
    )


@router.post("/pinpoint-forecast")
def pinpoint_forecast(req: PinpointForecastRequest):
    """
    Tab 2 — drop-a-pin 5-year forecast.
    Returns the new site's predicted monthly trajectory (total/membership/retail with P10–P90 bands) plus the
    total local-market wash count: actual history + 5-year forecast (with/without the new site, trend-CI band,
    and the entrant's own journey).
    """
    lat, lon = _resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.pinpoint_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        horizon_months=req.horizon_months,
    )
