from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExploreMarketRequest(BaseModel):
    """Tab 1 — interactive local-market explorer. Provide lat/lon OR an address."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="Pin latitude.")
    longitude: Optional[float] = Field(None, description="Pin longitude.")
    metric: str = Field(
        "tot_wash_count",
        description="Metric column: tot_wash_count | mem_wash_count | ret_wash_count | tot_revenue | mem_share_wash.",
    )
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Neighbour radius in km (the local market).")
    smoothing: int = Field(1, ge=1, le=12, description="Centered rolling-mean window in months (1 = none).")
    max_sites: int = Field(10, ge=1, le=50, description="Max sites shown; every entrant is always kept.")
    x_axis: str = Field("date", description="X axis: 'date' or 'months_since_open'.")


class PinpointForecastRequest(BaseModel):
    """Tab 2 — drop-a-pin 5-year forecast. Provide lat/lon OR an address."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="New-site latitude.")
    longitude: Optional[float] = Field(None, description="New-site longitude.")
    brand: Optional[str] = Field(None, description="Operator/brand client_id (see GET /pnl_analysis/brands). Strongest predictor of scale.")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo (0/None = use model).")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra yr3-5 membership drift (%/yr) on top of the market trend.")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra yr3-5 retail drift (%/yr) on top of the market trend.")
    horizon_months: int = Field(60, ge=12, le=60, description="Forecast horizon in months (<=60).")
