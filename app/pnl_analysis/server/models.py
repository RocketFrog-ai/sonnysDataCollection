from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class _PinRequest(BaseModel):
    """Base for every pin-driven request: provide latitude+longitude OR an address (geocoded server-side)."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="Pin latitude.")
    longitude: Optional[float] = Field(None, description="Pin longitude.")


# ─────────────────────────── Explore-markets (tab 1) ───────────────────────────
class ExploreMarketRequest(_PinRequest):
    """Tab 1 — the local-market MAP + header counts (markers, reference dots, operator footprint). No time
    series — the per-site KPI series are served by /explore-market/kpis. Provide lat/lon OR an address."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Neighbour radius in km (the local market).")
    max_sites: int = Field(10, ge=1, le=50, description="Max in-market site markers drawn; every entrant is always kept.")
    min_months: int = Field(36, ge=1, le=72, description="Keep rich-history sites (>= this many monthly records). 36 = the Explore view (≥3 yrs).")
    operator: Optional[str] = Field(None, description="Operator client_name (see GET /operators) to plot its whole footprint on the map.")
    demo: bool = Field(False, description="Anonymized client demo: hide names, drop exact site dots, shade cluster regions.")


class ExploreKpisRequest(_PinRequest):
    """Tab 1 — the 6 grouped per-site KPI panels (washes x3 / revenue x3 / ASP x2) for the local market."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Neighbour radius in km (the local market).")
    smoothing: int = Field(1, ge=1, le=12, description="Centered rolling-mean window in months (1 = none).")
    min_months: int = Field(36, ge=1, le=72, description="Keep only sites with >= this many monthly records (rich-history filter).")
    demo: bool = Field(False, description="Anonymized client demo: sites become 'Site N' by opening order.")


class InsightsRequest(_PinRequest):
    """Tab 1 — AI Key Insights: the 2-node pipeline (compute_metrics -> generate_insights) over the local
    market's KPI panels. Returns structured investor metrics + an LLM narrative per group (Washes/Revenue/ASPs)."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    min_months: int = Field(36, ge=1, le=72, description="Rich-history filter (>= this many monthly records).")
    last_n_months: int = Field(12, ge=3, le=36, description="Trailing window for the 'recent' metrics (months).")
    backend: Optional[str] = Field(None, description="LLM backend: 'azure' | 'local'. None = INSIGHTS_LLM_BACKEND env (default azure).")
    demo: bool = Field(False, description="Anonymized client demo: sites become 'Site N' by opening order (no real names to the LLM).")


# ─────────────────────────── Forecast (tab 2) ───────────────────────────
class PinpointForecastRequest(_PinRequest):
    """Tab 2 — drop-a-pin 5-year forecast (new site's trajectory + local market history/forecast)."""
    brand: Optional[str] = Field(None, description="Operator/brand client_id (see GET /pnl_analysis/brands). Strongest predictor of scale.")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo (0/None = use model).")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra yr3-5 membership drift (%/yr) on top of the market trend.")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra yr3-5 retail drift (%/yr) on top of the market trend.")
    horizon_months: int = Field(60, ge=12, le=60, description="Forecast horizon in months (<=60).")


class PnlForecastRequest(_PinRequest):
    """Tab 2 — the 💰 P&L chart: monthly revenue vs operating expense vs net, with an optional campaign overlay."""
    brand: Optional[str] = Field(None, description="Operator/brand client_id (see GET /pnl_analysis/brands).")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo (0/None = use model).")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra yr3-5 membership drift (%/yr).")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra yr3-5 retail drift (%/yr).")
    asp_override: Optional[float] = Field(None, ge=1.0, le=30.0, description="Blended $/wash. None = cluster average (the ≤20 km neighbours' last 12 months).")
    opex_growth_pct: float = Field(0.0, ge=-10.0, le=15.0, description="Opex cost escalation (%/yr) on top of the learned new-site ramp. Default flat.")
    campaign_on: bool = Field(False, description="Apply a retail→membership conversion campaign.")
    campaign_launch: int = Field(13, ge=1, le=48, description="Campaign launch — months after opening.")
    campaign_intensity: float = Field(1.0, ge=0.5, le=1.5, description="Campaign intensity (× a typical observed campaign).")
    window: int = Field(6, ge=1, le=12, description="Campaign impact window in months.")
    horizon_months: int = Field(60, ge=12, le=60, description="Forecast horizon in months (<=60).")


class ExpensePlanRequest(_PinRequest):
    """Tab 2 — user-driven EXPENSE PLAN: project monthly OPEX, CAPEX and combined expenses over the horizon.

    `asp`, `opex` and `capex` are all {year: value} maps keyed by year (1, 2, 3, …):
      • `asp`   — $/wash for that year (e.g. {1: 12.5, 2: 13}); empty/missing years → the cluster average.
      • `opex`  — OPEX as a % of sales for that year (e.g. {1: 60, 2: 50, 3: 45}), fitted onto the LEARNED new-site
                  opex pattern (hot early, easing to mature) so the monthly shape is realistic while each year hits
                  the given average; years past the last supplied one are extrapolated (escalated by `opex_growth_pct`).
      • `capex` — total CAPEX $ spent in that year (e.g. {1: 500000, 2: 100000}), spread across its months.
    Sales = the pin's forecast washes × ASP. Provide lat/lon OR an address."""
    brand: Optional[str] = Field(None, description="Operator/brand client_id (see GET /pnl_analysis/brands).")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo (0/None = use model).")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra yr3-5 membership drift (%/yr).")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra yr3-5 retail drift (%/yr).")
    asp: Dict[int, float] = Field(default_factory=dict, description="$/wash per year, e.g. {1: 12.5, 2: 13, 3: 13.5}. Empty = cluster average (≤20 km neighbours' last 12 months); missing years fall back to it.")
    opex: Dict[int, float] = Field(default_factory=dict, description="OPEX as % of sales per year, e.g. {1: 60, 2: 50, 3: 45}.")
    capex: Dict[int, float] = Field(default_factory=dict, description="CAPEX $ per year, e.g. {1: 500000, 2: 100000}.")
    opex_growth_pct: float = Field(0.0, ge=-10.0, le=15.0, description="OPEX %/yr escalation applied to years past the last one supplied in `opex`.")
    horizon_months: int = Field(60, ge=12, le=60, description="Forecast horizon in months (<=60).")


class CampaignVerdictRequest(_PinRequest):
    """Tab 2 — the 🎯 campaign recommendation + supporting metrics for this pin's local market."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    brand: Optional[str] = Field(None, description="Operator/brand client_id.")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo.")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra membership drift (%/yr).")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra retail drift (%/yr).")


class EatingMarketRequest(_PinRequest):
    """Tab 2 — 📈 eating-the-market: your site vs each incumbent forecast forward, with the campaign's share theft."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    brand: Optional[str] = Field(None, description="Operator/brand client_id.")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo.")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra membership drift (%/yr).")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra retail drift (%/yr).")
    campaign_on: bool = Field(False, description="Apply the campaign (drives the share theft).")
    campaign_launch: int = Field(13, ge=1, le=48, description="Campaign launch — months after opening.")
    campaign_intensity: float = Field(1.0, ge=0.5, le=1.5, description="Campaign intensity.")
    window: int = Field(6, ge=1, le=12, description="Campaign impact window in months.")
    max_incumbents: int = Field(6, ge=1, le=12, description="Max incumbents drawn (largest by recent volume).")


class LocalCampaignsRequest(_PinRequest):
    """Tab 2 — real campaigns in this local market: per-site series + detected promo months (the evidence)."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    metric: str = Field("mem_share_wash", description="Series: mem_share_wash | mem_wash_count | ret_wash_count.")
    max_sites: int = Field(8, ge=1, le=20, description="Max (nearest) in-market sites drawn.")
    demo: bool = Field(False, description="Anonymized client demo: sites become 'Site N' by opening order.")
