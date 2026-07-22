"""Route decorators ONLY for /v1/pnl_analysis/*: parse the request, delegate to the modelling engines
(app.pnl_analysis.modelling.*) or to service.py for shared helper logic, serialize the response. Split
out of the former monolithic routes.py; see schemas.py / service.py for the request models and the
extracted helper logic respectively.

Every pin-based forecast/market endpoint accepts `express_only` (default False). It restricts the
local market, the cluster gate and the model's level anchor to Express Tunnel sites with >=30 months
of history, mirroring the Streamlit "Express-only sites" toggle. The /insights/* endpoints do not
take it -- they annotate, they do not model."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.pnl_analysis.modelling import market
from app.pnl_analysis.modelling import site_factors as site_factors_engine
from app.pnl_analysis.modelling import pnl as pnl_engine
from app.pnl_analysis.modelling import campaign as campaign_engine
from app.pnl_analysis.modelling import bosch_forecast as bosch_engine
from app.pnl_analysis.insights.graph import market_insights as _insights_pipeline
from app.pnl_analysis.insights import location_poc as _loc
from app.pnl_analysis.insights import llm as _llm
from app.server import service
from app.server.cache import cached
from app.server.schemas import (
    ExploreMarketRequest,
    ExploreKpisRequest,
    SiteFactorsRequest,
    InsightsRequest,
    LocationSummaryRequest,
    CompetitionScaleRequest,
    PollinatedSummaryRequest,
    IndependentResearchRequest,
    PinpointForecastRequest,
    PnlForecastRequest,
    ExpensePlanRequest,
    CampaignVerdictRequest,
    EatingMarketRequest,
    LocalCampaignsRequest,
    BoschForecastRequest,
)

router = APIRouter(prefix="/pnl_analysis")


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
@cached("explore-market")
def explore_market(req: ExploreMarketRequest):
    """Tab 1 — the local-market MAP + header counts. Returns the in-market site markers (role-tagged
    focal/entrant/incumbent), geographic reference dots, and the highlighted operator's footprint — no
    time series (those come from /explore-market/kpis). Backs the map + the 4 header metric cards."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.explore_market(
        lat=lat, lon=lon, radius_km=req.radius_km, max_sites=req.max_sites,
        min_months=req.min_months, operator=req.operator, demo=req.demo,
        express_only=req.express_only,
    )


@router.post("/explore-market/kpis")
@cached("explore-market/kpis")
def explore_market_kpis(req: ExploreKpisRequest):
    """Tab 1 — the 9 grouped per-site KPI series (washes x3 / revenue x3 / ASP x2 / membership purchases x1,
    always the LAST group) for the whole local market (every in-radius site with >= `min_months` of history).
    Backs the 'Local-market KPIs over time' panels."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.explore_market_kpis(
        lat=lat, lon=lon, radius_km=req.radius_km, smoothing=req.smoothing,
        min_months=req.min_months, demo=req.demo,
        express_only=req.express_only,
    )


@router.post("/site-factors")
@cached("site-factors")
def site_factors(req: SiteFactorsRequest):
    """Tab 1 — the council site-factors extract for a pin (demographics / income bands / vehicles /
    competitors / mass-merchant anchors / StreetLight traffic), from the NEAREST covered site: matched
    within 3 miles, escalating to 6 then 9. Returns found=false + "Don't have data coverage" past 9 mi."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return site_factors_engine.site_factors(lat=lat, lon=lon)


@router.post("/insights")
@cached("insights")
def insights(req: InsightsRequest):
    """Tab 1 — AI Key Insights (grounded on the local market's KPI panels). Returns JSON `{summary}` — the full
    narrative as react-markdown-compatible markdown (plain `$`, `**bold**`, `- bullets`). 404 if the market is
    too thin."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    grounded = service.grounded_inputs(lat, lon, req.radius_km, req.min_months, req.demo)
    if grounded is None:
        raise HTTPException(status_code=404, detail=f"No rich-history sites within {req.radius_km} km of this pin.")
    panel, meta, focal = grounded
    # escape_dollars=False → plain `$` so the returned markdown renders cleanly in react-markdown (not KaTeX-escaped).
    blocks = _insights_pipeline(panel, meta, focal, backend=req.backend,
                                last_n_months=req.last_n_months, escape_dollars=False)["insights"]
    return service.render_insights_summary(blocks)


@router.post("/insights/location")
@cached("insights/location")
def insights_location(req: LocationSummaryRequest):
    """Tab 1 — location-only LLM market read (world-knowledge from the pin alone). Returns JSON `{summary}` — the
    full Local Market Analysis markdown (no verdict; the verdict lives only in the pollinated summary). 503 if no
    LLM backend answers."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    try:
        text = _loc.location_market_analysis(lat, lon, address=req.address, radius_km=req.radius_km,
                                             backend=req.backend)["text"]
        return {"summary": text}
    except _llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


@router.post("/insights/competition")
@cached("insights/competition")
def insights_competition(req: CompetitionScaleRequest):
    """Tab 1 — competitive landscape (LLM sizes the full competitive set vs the client's own portfolio). Returns
    JSON `{summary}` — react-markdown that embeds the competitors table + saturation/positioning. 503 if no LLM
    answers."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    known = service.known_site_names(lat, lon, req.radius_km, req.min_months, req.demo)
    try:
        result = _loc.competition_scale_analysis(lat, lon, known_sites=known, address=req.address,
                                                 radius_km=req.radius_km, backend=req.backend)
        return _loc.build_competition_response(result)
    except _llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


@router.post("/insights/pollinated")
@cached("insights/pollinated")
def insights_pollinated(req: PollinatedSummaryRequest):
    """Tab 1 — the fused ('pollinated') summary. CONSUMES the three insight responses the frontend already
    fetched — Key Insights (B), Local Market Analysis (A) and Competition Coverage (C) — passed in on the request,
    and synthesises them into one consolidated summary ending in a Final Verdict. One LLM call, no recomputation
    of A/B/C. Returns JSON: the consolidated `summary` plus, under `sources`, the exact three summary responses it
    consumed (so the frontend can render everything from this one payload). 503 if no LLM answers."""
    if not any((req.key_insights, req.location_analysis, req.competition)):
        raise HTTPException(status_code=400,
                            detail="Provide at least one of key_insights, location_analysis or competition.")
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    try:
        out = _loc.pollinate_analysis(req.location_analysis, req.key_insights, lat=lat, lon=lon,
                                      radius_km=req.radius_km, competition=req.competition, backend=req.backend)
        return {
            "summary": out["text"],
            "sources": {
                "key_insights": req.key_insights,
                "location_analysis": req.location_analysis,
                "competition": req.competition,
            },
        }
    except _llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


@router.post("/insights/independent-research")
@cached("insights/independent-research")
def insights_independent_research(req: IndependentResearchRequest):
    """Tab 1 — independent EXTERNAL-LLM market research: can the model size a NEW car-wash market for this pin from
    world knowledge ALONE (no internal/operator data; optional web search)? Sizes each radius separately (default
    3/6/9 mi) and estimates the requested business metrics, returning null-with-reason for anything it can't
    responsibly estimate rather than fabricating. Returns JSON `{radii, summary, sources}`. 503 if no LLM answers."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    try:
        result = _loc.independent_market_research(lat, lon, address=req.address, radii_miles=req.radii_miles,
                                                  backend=req.backend, use_web_search=req.use_web_search)
        return _loc.build_independent_research_response(result)
    except _llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


# ─────────────────────────── Forecast (tab 2) ───────────────────────────
@router.post("/pinpoint-forecast")
@cached("pinpoint-forecast")
def pinpoint_forecast(req: PinpointForecastRequest):
    """Tab 2 — the NEW SITE's own predicted 5-year monthly trajectory (total/membership/retail with P10-P90
    bands) + the summary KPI cards. The whole-market growth plot is a separate call: POST /market-forecast."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.pinpoint_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        horizon_months=req.horizon_months,
        express_only=req.express_only,
    )


@router.post("/market-forecast")
@cached("market-forecast")
def market_forecast(req: PinpointForecastRequest):
    """Tab 2 — the TOTAL LOCAL-MARKET wash count: actual history + 5-year forecast (with vs without the new
    site, a trend-CI band, and the entrant's own journey). Same inputs as /pinpoint-forecast."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return market.market_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        horizon_months=req.horizon_months,
        express_only=req.express_only,
    )


@router.post("/pnl-forecast")
@cached("pnl-forecast")
def pnl_forecast(req: PnlForecastRequest):
    """Tab 2 — the 💰 P&L chart: monthly revenue vs operating expense vs net over the 5-year horizon, with an
    optional retail→membership conversion campaign overlay. Revenue = forecast washes × cluster ASP; opex = the
    learned new-site ramp × mature $/wash (scope-aware), escalated by the cost-growth input."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return pnl_engine.pnl_forecast(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        asp_override=req.asp_override, opex_growth_pct=req.opex_growth_pct,
        campaign_on=req.campaign_on, campaign_launch=req.campaign_launch,
        campaign_intensity=req.campaign_intensity, window=req.window, horizon_months=req.horizon_months,
        express_only=req.express_only,
    )


@router.post("/expense-plan")
@cached("expense-plan")
def expense_plan(req: ExpensePlanRequest):
    """Tab 2 — user-driven EXPENSE PLAN: monthly OPEX, CAPEX and combined-expenses lines over the 5-year horizon.
    `opex` is {year: % of revenue} fitted onto the learned new-site opex pattern; `capex` is {year: $} spread over
    that year's months. Returns the three lines (+ revenue/net) per month, plus annual rollups and totals."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return pnl_engine.expense_plan(
        lat=lat, lon=lon, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        asp_by_year=req.asp, opex_pct_by_year=req.opex, capex_by_year=req.capex,
        opex_growth_pct=req.opex_growth_pct, horizon_months=req.horizon_months,
        express_only=req.express_only,
    )


@router.post("/campaign/verdict")
@cached("campaign/verdict")
def campaign_verdict(req: CampaignVerdictRequest):
    """Tab 2 — 🎯 the campaign recommendation + the 3 supporting metrics (neighbours' membership share, established
    incumbents, this site's predicted membership). Only recommends a promo where the membership market is proven."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return campaign_engine.campaign_verdict(
        lat=lat, lon=lon, radius_km=req.radius_km, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        express_only=req.express_only,
    )


@router.post("/campaign/eating-the-market")
@cached("campaign/eating-the-market")
def eating_the_market(req: EatingMarketRequest):
    """Tab 2 — 📈 your site vs each incumbent, each forecast forward 5 years; with a campaign, the incumbents drift
    down as your promo steals their retail share (theft scales with market density, recovers as the promo fades)."""
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return campaign_engine.eating_the_market(
        lat=lat, lon=lon, radius_km=req.radius_km, brand=req.brand, plateau_override=req.plateau_override,
        mem_growth_pct=req.mem_growth_pct, ret_growth_pct=req.ret_growth_pct,
        campaign_on=req.campaign_on, campaign_launch=req.campaign_launch,
        campaign_intensity=req.campaign_intensity, window=req.window, max_incumbents=req.max_incumbents,
        express_only=req.express_only,
    )


@router.get("/campaign/snapshot")
def campaign_snapshot():
    """Tab 2 — what a campaign does to OPEX / Revenue / Profit / Membership purchases, from the real P&L event
    study (book_v4): median values by month-offset for 1-month / 2-month / 3+-month campaigns. Not pin-specific."""
    return campaign_engine.campaign_snapshot()


@router.post("/campaign/local-evidence")
@cached("campaign/local-evidence")
def local_campaign_evidence(req: LocalCampaignsRequest):
    """Tab 2 — real campaigns in this local market: the nearest in-radius sites' monthly series for the chosen
    metric, with each site's detected promo-OPEX-spike months marked — the evidence behind the campaign model."""
    if req.metric not in service.WASH_METRICS:
        raise HTTPException(status_code=400, detail=f"metric must be one of: {', '.join(sorted(service.WASH_METRICS))}")
    lat, lon = service.resolve_lat_lon(req.latitude, req.longitude, req.address)
    return campaign_engine.local_campaign_evidence(
        lat=lat, lon=lon, radius_km=req.radius_km, metric=req.metric, max_sites=req.max_sites, demo=req.demo,
        express_only=req.express_only,
    )


# ─────────────────────────── Bosch prediction (proforma volume estimate) ───────────────────────────
@router.post("/bosch-forecast")
def bosch_forecast(req: BoschForecastRequest):
    """The 10 site factors + 4 demographics + traffic -> Year 1-5 car-wash-volume estimate (yearly
    and monthly), ported from Rafal's proforma Excel formula. Deterministic, not the coldstart ML
    model. Not pin-driven -- every input is supplied directly on the request, so this skips
    @cached (that decorator assumes a lat/lon pin; this is a pure, cheap computation anyway). See
    experiments/bosch-prediction-api/agent.md for the formula derivation."""
    return bosch_engine.bosch_forecast(
        site_factors=req.site_factors.model_dump(),
        avg_household_size=req.avg_household_size,
        pct_pop_25_65=req.pct_pop_25_65,
        pct_hh_income_over_35k=req.pct_hh_income_over_35k,
        base_price_carwash=req.base_price_carwash,
        base_traffic=req.base_traffic,
        year3_growth_pct=req.year3_growth_pct,
        year4_growth_pct=req.year4_growth_pct,
        year5_growth_pct=req.year5_growth_pct,
        operating_days_per_year=req.operating_days_per_year,
    )
