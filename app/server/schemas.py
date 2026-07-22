from __future__ import annotations

from typing import Dict, List, Literal, Optional

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
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


class ExploreKpisRequest(_PinRequest):
    """Tab 1 — the 9 grouped per-site KPI panels (washes x3 / revenue x3 / ASP x2 / membership purchases x1,
    always the last group) for the local market."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Neighbour radius in km (the local market).")
    smoothing: int = Field(1, ge=1, le=12, description="Centered rolling-mean window in months (1 = none).")
    min_months: int = Field(36, ge=1, le=72, description="Keep only sites with >= this many monthly records (rich-history filter).")
    demo: bool = Field(False, description="Anonymized client demo: sites become 'Site N' by opening order.")
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


class SiteFactorsRequest(_PinRequest):
    """Tab 1 — the council site-factors extract for a pin: demographics, income bands, vehicle counts,
    car-wash competitors, mass-merchant anchors and StreetLight traffic for the NEAREST covered site,
    matched at 3 -> 6 -> 9 miles. No coverage within 9 miles => found=false ("Don't have data coverage")."""


class InsightsRequest(_PinRequest):
    """Tab 1 — AI Key Insights: the 2-node pipeline (compute_metrics -> generate_insights) over the local
    market's KPI panels. Returns structured investor metrics + an LLM narrative per group (Washes/Revenue/ASPs)."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    min_months: int = Field(36, ge=1, le=72, description="Rich-history filter (>= this many monthly records).")
    last_n_months: int = Field(12, ge=3, le=36, description="Trailing window for the 'recent' metrics (months).")
    backend: Optional[str] = Field(None, description="LLM backend: 'azure' | 'local'. None = INSIGHTS_LLM_BACKEND env (default azure).")
    demo: bool = Field(False, description="Anonymized client demo: sites become 'Site N' by opening order (no real names to the LLM).")


class LocationSummaryRequest(_PinRequest):
    """Tab 1 — the location-only LLM market read: world-knowledge assessment from the pin ALONE (no operating
    data sent). Free-form markdown covering demographics, roads, competition, climate, verdict."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local trade-area radius in km (context for the model).")
    backend: Optional[str] = Field(None, description="LLM backend: 'azure' | 'local'. None = INSIGHTS_LLM_BACKEND env (default azure).")


class CompetitionScaleRequest(_PinRequest):
    """Tab 1 — the competitive landscape: an LLM estimate of the FULL competitive set in the trade area (total +
    express tunnels, a typed competitor table, saturation/intensity/pricing/headroom) vs the client's OWN portfolio
    of sites in the radius, yielding a 'coverage scale' multiple. Returns strict-JSON data + the scale multiples."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Trade-area radius in km.")
    min_months: int = Field(1, ge=1, le=72, description="Count the client's sites with >= this many months as the known portfolio (1 = every in-radius site).")
    backend: Optional[str] = Field(None, description="LLM backend: 'azure' | 'local'. None = env default.")
    demo: bool = Field(False, description="Anonymized demo: don't send the client's real site names to the LLM.")


class PollinatedSummaryRequest(_PinRequest):
    """Tab 1 — the fused ('pollinated') read: it CONSUMES the three insight responses the frontend already
    fetched — Key Insights (B, from /insights), Local Market Analysis (A, from /insights/location) and
    Competition Coverage (C, from /insights/competition) — and synthesises them into one consolidated summary
    with a Final Verdict. It does NOT recompute A/B/C, so it makes a single LLM call. Pass the responses through."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km (prompt context only).")
    key_insights: Optional[str] = Field(None, description="(B) The /insights response text (grounded data/plots). Optional but recommended.")
    location_analysis: Optional[str] = Field(None, description="(A) The /insights/location response text (Local Market Analysis).")
    competition: Optional[str] = Field(None, description="(C) The /insights/competition response — pass its `summary` markdown (or the whole JSON as a string).")
    backend: Optional[str] = Field(None, description="LLM backend: 'azure' | 'local'. None = env default.")


class IndependentResearchRequest(_PinRequest):
    """Tab 1 — the independent, EXTERNAL-KNOWLEDGE-ONLY market research: can a capable LLM size a new car-wash
    market for this pin from its own world knowledge (plus optional web search) with NO internal/operator data? Run
    separately per radius; each returns the requested business metrics, and any metric it can't responsibly size is
    returned as null-with-reason rather than fabricated."""
    radii_miles: List[float] = Field(default=[3.0, 6.0, 9.0], min_length=1, max_length=6,
                                     description="Trade-area radii (miles) to size independently, e.g. [3, 6, 9].")
    use_web_search: bool = Field(False, description="Ground the estimate on fresh web results (returns citable sources) when a provider is configured.")
    backend: Optional[str] = Field(None, description="LLM backend: 'azure' | 'local'. None = env default.")


# ─────────────────────────── Forecast (tab 2) ───────────────────────────
class PinpointForecastRequest(_PinRequest):
    """Tab 2 — drop-a-pin 5-year forecast (new site's trajectory + local market history/forecast)."""
    brand: Optional[str] = Field(None, description="Operator/brand client_id (see GET /pnl_analysis/brands). Strongest predictor of scale.")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo (0/None = use model).")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra yr3-5 membership drift (%/yr) on top of the market trend.")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra yr3-5 retail drift (%/yr) on top of the market trend.")
    horizon_months: int = Field(60, ge=12, le=60, description="Forecast horizon in months (<=60).")
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


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
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


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
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


class CampaignVerdictRequest(_PinRequest):
    """Tab 2 — the 🎯 campaign recommendation + supporting metrics for this pin's local market."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    brand: Optional[str] = Field(None, description="Operator/brand client_id.")
    plateau_override: Optional[float] = Field(None, ge=0, description="Override mature total washes/mo.")
    mem_growth_pct: float = Field(0.0, ge=-15.0, le=25.0, description="Extra membership drift (%/yr).")
    ret_growth_pct: float = Field(0.0, ge=-20.0, le=15.0, description="Extra retail drift (%/yr).")
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


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
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


class LocalCampaignsRequest(_PinRequest):
    """Tab 2 — real campaigns in this local market: per-site series + detected promo months (the evidence)."""
    radius_km: float = Field(20.0, ge=2.0, le=40.0, description="Local-market radius in km.")
    metric: str = Field("mem_share_wash", description="Series: mem_share_wash | mem_wash_count | ret_wash_count.")
    max_sites: int = Field(8, ge=1, le=20, description="Max (nearest) in-market sites drawn.")
    demo: bool = Field(False, description="Anonymized client demo: sites become 'Site N' by opening order.")
    express_only: bool = Field(False, description="Restrict the local market, the cluster gate and the level anchor to Express Tunnel sites with >=30 months of history. Mirrors the Streamlit \"Express-only sites\" toggle. Default False = every site, unchanged.")


# ─────────────────────────── Bosch prediction (proforma volume estimate) ───────────────────────────
class SiteFactorsInput(BaseModel):
    """The 10 site-selection factors from Rafal's proforma sheet — one categorical choice each,
    each option carrying a fixed 'site score' weight (app/pnl_analysis/modelling/bosch_forecast.py
    SITE_FACTOR_WEIGHTS has the full weight table). Option keys are ordered best -> worst site."""
    area_profile: Literal["shopping", "business", "residential", "industrial"] = Field(..., description="Area profile around the site.")
    nearest_competition: Literal["one_in_4mi", "multiple_in_4mi", "one_in_2mi", "multiple_in_2mi"] = Field(..., description="Nearest car-wash competitor(s).")
    weekly_hours: Literal["more_than_70", "hours_65_70", "hours_60_64", "less_than_60"] = Field(..., description="Weekly hours of operation.")
    type_of_site: Literal["corner_with_light", "corner_no_light", "inside_near_light", "inside_no_light"] = Field(..., description="Corner vs. inside lot, and light proximity.")
    site_accessibility: Literal["easy_in_out", "easy_in_out_divided_hwy", "easy_one_way", "difficult"] = Field(..., description="Ease of entering/exiting the site.")
    entrance_stack_up: Literal["more_than_20", "veh_20_15", "veh_14_10", "less_than_10"] = Field(..., description="Vehicle stack-up capacity at the entrance.")
    free_vacuum_slots: Literal["more_than_20", "veh_12_20", "less_than_12", "coin_or_none"] = Field(..., description="Free vacuum slot capacity.")
    pay_stations: Literal["three_or_more", "two", "one", "live_person"] = Field(..., description="Number of pay stations.")
    visibility: Literal["more_than_500ft", "ft_400_500", "ft_300_400", "less_than_300ft"] = Field(..., description="Site visibility, both directions.")
    traffic_speed: Literal["less_than_30mph", "mph_30_40", "mph_40_50", "more_than_50mph"] = Field(..., description="Posted traffic speed past the site.")


class BoschForecastRequest(BaseModel):
    """The Bosch prediction — Rafal's proforma Excel formula ported to an API (see
    experiments/bosch-prediction-api/agent.md): 10 site factors + 4 demographic components +
    a traffic-count input -> a Year 1-5 car-wash-volume estimate. Deterministic formula, NOT the
    coldstart ML model behind /pinpoint-forecast. Not pin-driven — every input is supplied
    directly (the front end may auto-fill some from /site-factors or the council dataset)."""
    site_factors: SiteFactorsInput = Field(..., description="The 10 site-selection factor choices.")
    avg_household_size: float = Field(..., gt=0, description="Average household size near the site (target 2.1).")
    pct_pop_25_65: float = Field(..., ge=0, le=1, description="Fraction of population aged 25-65 (0.55 = 55%; target 0.55). NOT a 0-100 percentage.")
    pct_hh_income_over_35k: float = Field(..., ge=0, le=1, description="Fraction of households with income > $35k (target 0.5). NOT a 0-100 percentage.")
    base_price_carwash: float = Field(..., gt=0, description="Base price of the car wash, $ (target/baseline $5).")
    base_traffic: float = Field(..., ge=0, description="Estimated 24-hour bidirectional traffic count at the site.")
    year3_growth_pct: float = Field(0.0, ge=-1.0, description="Year-3 traffic growth rate, additive (0.0 = flat, 0.02 = +2%). Applied independently to base_traffic, NOT compounded with year 4/5.")
    year4_growth_pct: float = Field(0.0, ge=-1.0, description="Year-4 traffic growth rate, additive. Applied independently to base_traffic, NOT compounded on year 3.")
    year5_growth_pct: float = Field(0.0, ge=-1.0, description="Year-5 traffic growth rate, additive. Applied independently to base_traffic, NOT compounded on year 4.")
    operating_days_per_year: float = Field(300.0, gt=0, le=366, description="Assumed operating days/year. 300 is the modal value across the source workbooks, but 280/310/330 all appear too -- a per-site assumption, not a fixed constant.")
