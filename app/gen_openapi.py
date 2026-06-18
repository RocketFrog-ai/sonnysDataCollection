"""Generate a full-scale OpenAPI YAML from the live FastAPI app, enriched with request/response examples."""
import yaml
from app.site_analysis.server.main import app

spec = app.openapi()
# Keep FastAPI's NATIVE OpenAPI version (3.1.0) — pydantic-v2 schemas use 3.1 constructs
# (`type: "null"` Optionals), so forcing 3.0.x would make the document fail strict validation.
spec["info"] = {
    "title": "Earnest Proforma backend",
    "version": "2.0",
    "description": (
        "Backend for the Earnest Proforma Streamlit app — three modes:\n"
        "- **Explore markets** (`/v1/pnl_analysis/explore-market*`, `/operators`)\n"
        "- **Forecast** (`/v1/pnl_analysis/pinpoint-forecast`, `/pnl-forecast`, `/campaign/*`, `/brands`)\n"
        "- **Site analysis** — synchronous `/v1/site-context` and the async `/v1/analyze-site` pipeline.\n\n"
        "Every pin endpoint accepts `latitude`+`longitude` **OR** `address` (geocoded server-side).\n"
        "Series are parallel arrays; missing months come through as `null`. Example values are real "
        "outputs for the Atlanta GA pin (33.7489, -84.3879); long series are trimmed for readability."
    ),
}
spec["servers"] = [{"url": "http://localhost:8002", "description": "Local dev (uvicorn)"}]
spec["tags"] = [
    {"name": "Lookups", "description": "Dropdown values (operators / brands)."},
    {"name": "Explore markets", "description": "Local-market line explorer + map + KPI panels."},
    {"name": "Forecast", "description": "Drop-a-pin 5-year wash & P&L forecast."},
    {"name": "Campaign", "description": "Promotion decision, share-theft and real-market evidence."},
    {"name": "Site analysis (sync)", "description": "One-call lat/lon location context."},
    {"name": "Site analysis (async)", "description": "Queued address pipeline — submit, poll, read by task_id."},
    {"name": "Utilities", "description": "Health, geocoded single-fetch helpers, cache."},
]


def tag_for(path):
    if path == "/" or "/health" in path or "/cache/" in path or "/traffic-lights" in path or "/nearby-stores" in path:
        return "Utilities"
    if "/campaign/" in path:
        return "Campaign"
    if "/explore-market" in path:
        return "Explore markets"
    if path.endswith("/brands") or path.endswith("/operators"):
        return "Lookups"
    if "/pinpoint-forecast" in path or "/market-forecast" in path or "/pnl-forecast" in path:
        return "Forecast"
    if path == "/v1/site-context":
        return "Site analysis (sync)"
    return "Site analysis (async)"


# trimmed series helper for example readability
def s(*vals, note="0..60"):
    return list(vals)  # short illustrative slice


# ── request examples (POST bodies) ──
REQ = {
    "/v1/pnl_analysis/explore-market": {
        "latitude": 28.929, "longitude": -82.038, "radius_km": 20, "max_sites": 10,
        "min_months": 36, "operator": None, "demo": False},
    "/v1/pnl_analysis/explore-market/kpis": {
        "latitude": 33.7489, "longitude": -84.3879, "radius_km": 20, "smoothing": 1, "min_months": 36, "demo": False},
    "/v1/pnl_analysis/pinpoint-forecast": {
        "latitude": 33.7489, "longitude": -84.3879, "brand": None, "plateau_override": None,
        "mem_growth_pct": 0, "ret_growth_pct": 0, "horizon_months": 60},
    "/v1/pnl_analysis/market-forecast": {
        "latitude": 33.7489, "longitude": -84.3879, "brand": None, "plateau_override": None,
        "mem_growth_pct": 0, "ret_growth_pct": 0, "horizon_months": 60},
    "/v1/pnl_analysis/pnl-forecast": {
        "latitude": 33.7489, "longitude": -84.3879, "brand": None, "plateau_override": None,
        "mem_growth_pct": 0, "ret_growth_pct": 0, "asp_override": None, "opex_growth_pct": 0,
        "campaign_on": False, "campaign_launch": 13, "campaign_intensity": 1.0, "window": 6, "horizon_months": 60},
    "/v1/pnl_analysis/campaign/verdict": {
        "latitude": 33.7489, "longitude": -84.3879, "radius_km": 20, "brand": None,
        "plateau_override": None, "mem_growth_pct": 0, "ret_growth_pct": 0},
    "/v1/pnl_analysis/campaign/eating-the-market": {
        "latitude": 33.7489, "longitude": -84.3879, "radius_km": 20, "brand": None, "plateau_override": None,
        "mem_growth_pct": 0, "ret_growth_pct": 0, "campaign_on": True, "campaign_launch": 13,
        "campaign_intensity": 1.0, "window": 6, "max_incumbents": 6},
    "/v1/pnl_analysis/campaign/local-evidence": {
        "latitude": 33.7489, "longitude": -84.3879, "radius_km": 20, "metric": "mem_share_wash",
        "max_sites": 8, "demo": False},
    "/v1/site-context": {"latitude": 33.7489, "longitude": -84.3879, "include_ai": False, "demo": False},
    "/v1/analyze-site": {"address": "1 Peachtree St NE, Atlanta, GA"},
    "/v1/traffic-lights": {"address": "1 Peachtree St NE, Atlanta, GA"},
    "/v1/nearby-stores": {"address": "1 Peachtree St NE, Atlanta, GA"},
}

# ── response examples (200) ──
RESP = {
    ("get", "/"): {"status": "running"},
    ("get", "/v1/health"): {"status": "healthy", "service": "site-analysis-pipeline"},
    ("get", "/v1/pnl_analysis/operators"): {
        "n_operators": 780, "operators": ["1003339carnutz", "Aloha Car Wash", "BlueWave"]},
    ("get", "/v1/pnl_analysis/brands"): {
        "n_brands": 780, "brands": [
            {"client_id": "bluewave_000567", "client_name": "BlueWave", "n_sites": 92, "model_known": True},
            {"client_id": "luvcarwash_000457", "client_name": "Luv Car Wash", "n_sites": 71, "model_known": True}]},
    ("post", "/v1/pnl_analysis/explore-market"): {
        "lat": 28.929, "lon": -82.038, "radius_km": 20, "min_months": 36,
        "n_sites_in_market": 3, "n_entrants": 1, "n_shown": 3,
        "focal_site_key": "bigdans_000378::9", "operator": None,
        "map": {"center": {"lat": 28.929, "lon": -82.038}, "radius_km": 20,
                "markers": [
                    {"site_key": "mrclean_000xx::1", "name": "Mr Clean Car Wash", "lat": 28.81, "lon": -81.88,
                     "dist_km": 15.6, "op_start": "2021-03", "role": "incumbent", "is_entrant": False, "color": "#5b8db8"},
                    {"site_key": "bigdans_000378::9", "name": "Big Dan's", "lat": 28.93, "lon": -81.92,
                     "dist_km": 11.1, "op_start": "2022-07", "role": "focal", "is_entrant": True, "color": "#e6194B"}],
                "reference_dots": [{"site_key": "caliberocala_000153::6", "name": "Caliber Car Wash",
                                    "lat": 29.18, "lon": -82.14, "dist_km": 28.4}],
                "cluster_regions": [], "operator_footprint": []}},
    ("post", "/v1/pnl_analysis/explore-market/kpis"): {
        "lat": 28.929, "lon": -82.038, "radius_km": 20, "smoothing": 1, "min_months": 36, "n_sites": 3,
        "focal_site_key": "bigdans_000378::9",
        "sites": [
            {"site_key": "mrclean_000xx::1", "name": "Mr Clean Car Wash", "is_focal": False, "is_entrant": False,
             "dist_km": 15.6, "op_start": "2021-03"},
            {"site_key": "bigdans_000378::9", "name": "Big Dan's", "is_focal": True, "is_entrant": True,
             "dist_km": 11.1, "op_start": "2022-07"}],
        "groups": [{"name": "Washes", "panels": [
            {"col": "tot_wash_count", "label": "Total washes", "unit": "count",
             "series": [{"site_key": "bigdans_000378::9", "name": "Big Dan's", "is_focal": True, "is_entrant": True,
                         "x": ["2022-07-01", "2022-08-01", "2022-09-01"], "y": [1523.0, 8201.0, 9544.0]}]}]}]},
    ("post", "/v1/pnl_analysis/pinpoint-forecast"): {
        "lat": 33.7489, "lon": -84.3879, "brand": None,
        "summary": {"plateau_med": 6465.17, "plateau_lo": 3005.35, "plateau_hi": 10231.81, "mem_share": 0.516,
                    "n_neighbours_20km": 10, "brand_known": False, "ramp_source": "region: South",
                    "region": "South", "state": "GA", "mem_growth": 0.0018, "ret_growth": -0.0656},
        "trajectory": {"months": [0, 1, 2], "total_med": [2931.0, 3574.9, 4755.4], "total_lo": [],
                       "total_hi": [], "mem_med": [], "ret_med": []}},
    ("post", "/v1/pnl_analysis/market-forecast"): {
        "lat": 33.7489, "lon": -84.3879, "brand": None, "open_date": "2026-07-01", "has_neighbours": True,
        "history": {"dates": ["2020-01-01", "2020-02-01"], "values": [1053.0, 1125.0]},
        "forecast": {"dates": ["2026-07-01", "2026-08-01"], "with_new_site": [], "without_new_site": [],
                     "band_lo": [], "band_hi": [], "new_entrant_journey": []},
        "net_change_year5": 6017.7},
    ("post", "/v1/pnl_analysis/pnl-forecast"): {
        "lat": 33.7489, "lon": -84.3879, "brand": None, "horizon_months": 60, "months": [0, 1, 2],
        "asp": {"mem": 12.51, "ret": 15.81, "blend": 14.10, "used": 14.10, "scope": "cluster <=20 km · 10 sites",
                "refs": {"ov_mem": 9.74, "ov_ret": 14.77, "cl_mem": 11.00, "cl_ret": 17.19, "scope": "state GA"}},
        "opex": {"mature_opex": 6538.99, "ramp_scope": "region South", "opw_scope": "state GA",
                 "ramp_hage": 30, "hist_yoy": -0.264},
        "scopes": {"opex": "state GA"},
        "series": {"revenue_base": [], "revenue": [43970.1, 52939.8, 69521.1], "opex_base": [], "opex": [],
                   "net": [], "net_base": []},
        "campaign": {"applied": False, "launch": 13, "intensity": 1.0, "window": 6, "conv_pct": 0.3,
                     "mem_share_settled": 0.572},
        "summary": {"plateau_med": 6465.17, "mem_share": 0.516, "state": "GA", "region": "South",
                    "total_revenue_5yr": 5128064.33, "total_opex_5yr": 339058.70, "net_5yr": 4789005.64,
                    "breakeven_month": 0}},
    ("post", "/v1/pnl_analysis/campaign/verdict"): {
        "ok": True, "verdict_level": "recommended",
        "verdict_text": "Recommended. 5 established incumbents within 20 km at 60% membership.",
        "neighbours_mem_share": 0.602, "n_established_incumbents": 5, "this_site_mem_share": 0.572,
        "conv_pct": 0.3, "radius_km": 20.0},
    ("post", "/v1/pnl_analysis/campaign/eating-the-market"): {
        "lat": 33.7489, "lon": -84.3879, "radius_km": 20, "months": [0, 1, 2],
        "your_site": {"base": [2931.0, 3574.9, 4755.4], "with_campaign": []},
        "incumbents": [{"site_key": "luvcarwash_000457::54", "name": "Luv Car Wash",
                        "expected": [13672.0, 13672.0, 14074.08], "with_campaign": []}],
        "n_incumbents": 5, "n_shown": 5, "steal_peak": 0.06,
        "campaign": {"applied": True, "launch": 13, "intensity": 1.0, "window": 6}},
    ("get", "/v1/pnl_analysis/campaign/snapshot"): {
        "buckets": [{"bucket": "1 month", "title": "1-Month Campaigns", "n_campaigns": 141, "camp_months": [0],
                     "mfs": [-3, -2, -1], "opex": [71047.51, 75836.78, 75743.92],
                     "revenue": [109153.96, 104778.89, 114198.58], "profit": [34181.60, 26645.00, 33678.88],
                     "mem_purchases": [2753.0, 2642.5, 2605.0]}]},
    ("post", "/v1/pnl_analysis/campaign/local-evidence"): {
        "lat": 33.7489, "lon": -84.3879, "radius_km": 20, "metric": "mem_share_wash",
        "sites": [{"site_key": "luvcarwash_000457::54", "name": "Luv Car Wash",
                   "x": ["2024-02-01", "2024-03-01", "2024-04-01"], "y": [0.192, 0.487, 0.548],
                   "campaign_months": []}]},
    ("post", "/v1/site-context"): {
        "lat": 33.7489, "lon": -84.3879, "address": None, "has_api_key": True,
        "metrics": {"competitors_4mi": 19, "nearest_competitor_mi": 0.56, "retail_anchors_3mi": 25,
                    "gas_stations_3mi": 20, "comfortable_days": 172},
        "markers": [{"category": "origin", "name": "Site", "lat": 33.7489, "lon": -84.3879,
                     "distance_miles": 0.0, "address": None},
                    {"category": "car_wash", "name": "Express Wash", "lat": 33.75, "lon": -84.39,
                     "distance_miles": 0.56, "rating": 4.3, "user_rating_count": 412,
                     "primary_type": "car_wash", "address": "…"}],
        "dimensions": {
            "competition": {"count": 19, "radius_miles": 4.0,
                            "nearest": {"name": "Express Wash", "distance_miles": 0.56, "rating": 4.3,
                                        "user_rating_count": 412, "primary_type": "car_wash", "address": "…"},
                            "competitors": [], "insight": {"insight": "19 competing car washes within 4 miles…",
                                                           "pro": "A proven car-wash market…",
                                                           "con": "Several competitors share the local demand.",
                                                           "conclusion": "Crowded — differentiation matters."}},
            "retail": {"anchors": [], "key_distances": {"costco": 5.1, "walmart": 1.2, "target": 2.4},
                       "grocery_1mi": 3, "food_0_5mi": 8,
                       "insight": {"insight": "25 retail anchors within 3 miles…", "pro": "…", "con": "…",
                                   "conclusion": "Good retail co-tenancy."}},
            "gas": {"count": 20, "radius_miles": 3.0, "stations": [],
                    "insight": {"insight": "20 gas stations within 3 miles…", "pro": "…", "con": "…",
                                "conclusion": "Strong fuel-driven traffic."}},
            "weather": {"metrics": [{"key": "rainy_days", "display": "Rainy days", "subtitle": "per year",
                                     "value": 113.0, "unit": "days"}],
                        "insight": {"insight": "Weather here: 113 rainy days/yr…", "pro": "…", "con": "…",
                                    "conclusion": "Weather is favorable for steady wash demand."}}}},
    ("post", "/v1/analyze-site"): {
        "task_id": "3f2c1a90-7b6e-4c2d-9a1e-1b2c3d4e5f60", "status": "PENDING",
        "message": "Site successfully submitted for analysis"},
    ("get", "/v1/task/{task_id}"): {
        "task_id": "3f2c1a90-7b6e-4c2d-9a1e-1b2c3d4e5f60", "status": "SUCCESS",
        "result": {"address": "1 Peachtree St NE, Atlanta, GA", "lat": 33.7489, "lon": -84.3879,
                   "fetched": {"climate": {"rainy_days": 113}, "gas_stations": [], "retail_anchors": {},
                               "competitors_data": {"count": 19, "competitors": []}}},
        "error": None, "created_at": None, "completed_at": None},
    ("get", "/v1/result/{task_id}"): {
        "task_id": "3f2c1a90-7b6e-4c2d-9a1e-1b2c3d4e5f60", "status": "success",
        "result": {"address": "1 Peachtree St NE, Atlanta, GA", "lat": 33.7489, "lon": -84.3879, "fetched": {}}},
    ("get", "/v1/weather/data-by-task/{task_id}"): {
        "task_id": "3f2c1a90-…", "address": "1 Peachtree St NE, Atlanta, GA", "complete": True,
        "metrics": [{"metric_key": "rainy_days", "display_name": "Rainy days", "subtitle": "per year",
                     "value": 113.0, "unit": "days"},
                    {"metric_key": "days_pleasant_temp", "display_name": "Comfortable days", "subtitle": "per year",
                     "value": 172.0, "unit": "days"}]},
    ("get", "/v1/weather/summary-by-task/{task_id}"): {
        "task_id": "3f2c1a90-…", "summary": {"insight": "…", "pro": "…", "con": "…", "conclusion": "…"}},
    ("get", "/v1/competition/data-by-task/{task_id}"): {
        "task_id": "3f2c1a90-…", "address": "1 Peachtree St NE, Atlanta, GA", "complete": True,
        "nearby_car_washes": {"count": 19, "list": [
            {"name": "Express Wash", "rating": 4.3, "user_rating_count": 412, "address": "…",
             "distance_miles": 0.56, "official_website": "https://…", "primary_carwash_type": "Express Tunnel"}]},
        "nearest": {"name": "Express Wash", "distance_miles": 0.56, "rating": 4.3, "user_rating_count": 412}},
    ("get", "/v1/competition/summary-by-task/{task_id}"): {"task_id": "3f2c1a90-…", "summary": {"insight": "…"}},
    ("get", "/v1/retail/data-by-task/{task_id}"): {
        "task_id": "3f2c1a90-…", "address": "1 Peachtree St NE, Atlanta, GA", "complete": True,
        "nearest_anchor": {"name": "Walmart Supercenter", "type": "Supercenter", "distance_miles": 1.2},
        "key_anchors": {"warehouse_club": {"name": "Costco", "type": "Warehouse Club", "distance_miles": 5.1},
                        "big_box": {"name": "Walmart Supercenter", "type": "Supercenter", "distance_miles": 1.2},
                        "grocery": {"name": "Kroger", "type": "Grocery Anchor", "distance_miles": 0.8},
                        "food_beverage": {"name": "Starbucks", "type": "Food & Beverage", "distance_miles": 0.3}},
        "retail_anchors": {"within_1_mile": {"count": 6, "list": [
            {"name": "Kroger", "type": "Grocery Anchor", "distance_miles": 0.8}]},
                           "within_3_miles": {"count": 19, "list": []}}},
    ("get", "/v1/retail/summary-by-task/{task_id}"): {"task_id": "3f2c1a90-…", "summary": {"insight": "…"}},
    ("get", "/v1/gas/data-by-task/{task_id}"): {
        "task_id": "3f2c1a90-…", "address": "1 Peachtree St NE, Atlanta, GA", "complete": True,
        "nearest": {"name": "QuikTrip", "distance_miles": 0.4, "high_traffic_brand": True},
        "gas_stations": {"within_1_mile": {"count": 5, "list": [
            {"name": "QuikTrip", "distance_miles": 0.4, "rating": 4.1, "user_rating_count": 230,
             "high_traffic_brand": True}]},
                         "within_3_miles": {"count": 15, "list": []}}},
    ("get", "/v1/gas/summary-by-task/{task_id}"): {"task_id": "3f2c1a90-…", "summary": {"insight": "…"}},
    ("get", "/v1/map/data-by-task/{task_id}"): {
        "task_id": "3f2c1a90-…", "address": "1 Peachtree St NE, Atlanta, GA", "lat": 33.7489, "lon": -84.3879,
        "complete": True,
        "counts": {"markers_total": 45, "gas_stations": 15, "competitors": 19, "retail_anchors": 10},
        "markers": [{"id": "origin", "name": "Input Site", "category": "origin", "latitude": 33.7489,
                     "longitude": -84.3879, "distance_miles": 0.0, "address": "1 Peachtree St NE, Atlanta, GA"},
                    {"id": "ChIJ…", "name": "Express Wash", "category": "car_wash", "latitude": 33.75,
                     "longitude": -84.39, "distance_miles": 0.56, "rating": 4.3, "user_rating_count": 412,
                     "address": "…"}]},
    ("post", "/v1/traffic-lights"): {
        "address": "1 Peachtree St NE, Atlanta, GA", "lat": 33.7489, "lon": -84.3879,
        "data": {"traffic_lights_count": 7, "radius_miles": 1.0}},
    ("post", "/v1/nearby-stores"): {
        "address": "1 Peachtree St NE, Atlanta, GA", "lat": 33.7489, "lon": -84.3879,
        "data": {"walmart": [{"name": "Walmart Supercenter", "distance_miles": 1.2}],
                 "target": [], "costco": [{"name": "Costco", "distance_miles": 5.1}]}},
    ("get", "/v1/cache/site-analysis/all"): {
        "page": 1, "page_size": 50, "total": 0, "items": []},
}


def ensure_json_content(obj):
    obj.setdefault("content", {}).setdefault("application/json", {})
    return obj["content"]["application/json"]


for path, methods in spec["paths"].items():
    for method, op in methods.items():
        m = method.lower()
        op["tags"] = [tag_for(path)]
        # request example
        if m in ("post", "put", "patch") and path in REQ and "requestBody" in op:
            media = ensure_json_content(op["requestBody"])
            media["example"] = REQ[path]
        # response example
        ex = RESP.get((m, path))
        if ex is not None:
            responses = op.setdefault("responses", {})
            r200 = responses.setdefault("200", {"description": "Successful Response"})
            media = ensure_json_content(r200)
            media["example"] = ex

out = "/Users/dhruvsood/sonnysDataCollection/app/openapi.yaml"
with open(out, "w") as f:
    f.write("# Auto-generated from the live FastAPI app, enriched with request/response examples.\n")
    f.write("# Regenerate from the repo root as a MODULE (so app/celery does not shadow the real celery):\n")
    f.write("#   python -m app.gen_openapi   (conda env sonnysDataCollection)\n")
    yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=120)

# report
n_paths = len(spec["paths"])
n_ops = sum(len(v) for v in spec["paths"].values())
n_req = sum(1 for p, ms in spec["paths"].items() for me, op in ms.items()
            if "requestBody" in op and op["requestBody"].get("content", {}).get("application/json", {}).get("example") is not None)
n_resp = sum(1 for p, ms in spec["paths"].items() for me, op in ms.items()
             if op.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {}).get("example") is not None)
print(f"wrote {out}")
print(f"paths={n_paths} operations={n_ops} request_examples={n_req} response_examples={n_resp}")
