"""
Site analysis page — "what surrounds a location".

Given the shared map pin (the same (lat, lon) used by Explore markets & Forecast) this pulls live location context around the
site using app/site_analysis:
  • weather / climate     (Open-Meteo)
  • competing car washes  (Google Places, ≤4 mi driving)
  • retail anchors        (Google Places, ≤3 mi: warehouse clubs, big box, grocery, food)
  • gas stations          (Google Places, ≤3 mi driving)

It draws everything on a map with markers and shows per-dimension insights — rule-based always,
plus an optional AI write-up from the internal LLM.

The heavy lifting lives in app/site_analysis/*; here we call its feature fetchers directly (the
celery-wrapped orchestrator fetch_all_features isn't importable in the streamlit venv) and render.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Make the repo-root `app.*` package importable when run from streamlits/.
# Streamlit runs this from the streamlits/ dir, which contains app.py — that local module would
# shadow the repo-root `app/` namespace package (a regular module beats a namespace portion in
# import resolution). So pre-register `app` as a namespace package pointing at <repo>/app, which
# makes `from app... import ...` resolve correctly regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_APP_DIR = _REPO_ROOT / "app"
if _APP_DIR.is_dir() and (
    "app" not in sys.modules or not hasattr(sys.modules.get("app"), "__path__")
):
    _app_mod = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("app", None, is_package=True)
    )
    _app_mod.__path__ = [str(_APP_DIR)]
    sys.modules["app"] = _app_mod

from app.utils import common as calib
from app.site_analysis.features.active.nearbyGasStations.get_nearby_gas_stations import get_nearby_gas_stations
from app.site_analysis.features.active.nearbyRetailers.get_nearby_retail_anchors import get_nearby_retail_anchors
from app.site_analysis.features.active.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
from app.site_analysis.features.active.weather.open_meteo import fetch_climate_for_site, get_default_weather_range
from app.site_analysis.server.config import (
    GAS_RADIUS_FAR_MILES,
    RETAIL_RADIUS_FAR_MILES,
    WEATHER_METRIC_CONFIG,
    WEATHER_METRIC_DISPLAY,
    get_weather_metric_value_from_climate,
    is_high_traffic_gas_brand,
)

COMPETITOR_RADIUS_MILES = 4.0
METERS_PER_MILE = 1609.34

# marker colour + legend label + legend swatch emoji per category
_CAT_STYLE = {
    "origin":      ("#111111", "site", "⬛"),
    "car_wash":    ("#e6194B", "car wash (competitor)", "🟥"),
    "gas_station": ("#2e8b57", "gas station", "🟩"),
    "retail":      ("#3b7dd8", "retail anchor", "🟦"),
}


# ───────────────────────── data layer (app/site_analysis) ─────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_features(lat: float, lon: float) -> dict:
    """Parallel external fetch around (lat, lon): {climate, gas_stations, retail_anchors, competitors_data}.
    Mirrors app/site_analysis fetch_all_features, calling the feature fetchers directly."""
    start_date, end_date = get_default_weather_range()
    api_key = calib.GOOGLE_MAPS_API_KEY or ""

    def _climate():
        out = fetch_climate_for_site(lat, lon, start_date=start_date, end_date=end_date)
        return out if out and not out.get("error") else {}

    def _gas():
        if not api_key:
            return []
        try:
            return get_nearby_gas_stations(api_key, lat, lon, radius_miles=GAS_RADIUS_FAR_MILES,
                                           max_results=20, fetch_place_details=False) or []
        except Exception:
            return []

    def _retail():
        if not api_key:
            return {}
        try:
            return get_nearby_retail_anchors(api_key, lat, lon, radius_miles=RETAIL_RADIUS_FAR_MILES) or {}
        except Exception:
            return {}

    def _competitors():
        if not api_key:
            return {}
        try:
            return get_nearby_competitors(api_key, lat, lon, radius_miles=COMPETITOR_RADIUS_MILES,
                                          fetch_place_details=False) or {}
        except Exception:
            return {}

    out = {"climate": {}, "gas_stations": [], "retail_anchors": {}, "competitors_data": {}}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_climate): "climate", ex.submit(_gas): "gas_stations",
                ex.submit(_retail): "retail_anchors", ex.submit(_competitors): "competitors_data"}
        for fut in as_completed(futs):
            try:
                out[futs[fut]] = fut.result()
            except Exception:
                pass
    return out


def _llm_reachable(timeout: float = 2.0) -> bool:
    """Fast socket pre-check on the internal LLM host:port — so toggling AI never hangs the UI
    on the client's 120s × 3 retry timeout when the endpoint is down (e.g. off-network)."""
    url = calib.LLM_REALTIME_URL or ""
    if not url:
        return False
    p = urlparse(url)
    host, port = p.hostname, (p.port or (443 if p.scheme == "https" else 80))
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@st.cache_data(show_spinner=False, ttl=3600)
def ai_summaries(fetched: dict) -> dict:
    """Internal-LLM write-ups per dimension. Returns {} when the LLM is unreachable (fast pre-check)."""
    if not _llm_reachable():
        return {}
    from app.site_analysis.modelling.ai import (
        summarize_weather, summarize_competition, summarize_retail, summarize_gas,
    )
    return {
        "weather": summarize_weather(fetched.get("climate") or {}),
        "competition": summarize_competition(fetched.get("competitors_data") or {}),
        "retail": summarize_retail(fetched.get("retail_anchors") or {}),
        "gas": summarize_gas(fetched.get("gas_stations") or []),
    }


def build_markers(lat, lon, fetched, address=None):
    """Origin + gas + competitor + retail markers (lat/lon already returned by the fetchers)."""
    markers = [{"category": "origin", "name": "Site", "lat": lat, "lon": lon,
                "distance_miles": 0.0, "address": address}]
    for s in (fetched.get("gas_stations") or []):
        if s.get("latitude") is not None and s.get("longitude") is not None:
            markers.append({"category": "gas_station", "name": s.get("name"),
                            "lat": s["latitude"], "lon": s["longitude"],
                            "distance_miles": s.get("distance_miles"), "rating": s.get("rating"),
                            "address": s.get("address"),
                            "high_traffic": is_high_traffic_gas_brand(s.get("name"))})
    for c in ((fetched.get("competitors_data") or {}).get("competitors") or []):
        if c.get("latitude") is not None and c.get("longitude") is not None:
            markers.append({"category": "car_wash", "name": c.get("name"),
                            "lat": c["latitude"], "lon": c["longitude"],
                            "distance_miles": c.get("distance_miles"), "rating": c.get("rating"),
                            "user_rating_count": c.get("user_rating_count"),
                            "primary_type": c.get("primary_type"), "address": c.get("address")})
    for a in ((fetched.get("retail_anchors") or {}).get("anchors") or []):
        if a.get("latitude") is not None and a.get("longitude") is not None:
            markers.append({"category": "retail", "name": a.get("name"), "type": a.get("type"),
                            "lat": a["latitude"], "lon": a["longitude"],
                            "distance_miles": a.get("distance_miles"), "address": a.get("address")})
    return markers


# ───────────────────────── rule-based insights (always available) ─────────────────────────
def rule_weather(climate: dict) -> dict:
    climate = climate or {}
    if not climate:
        return {}
    rainy = climate.get("rainy_days")
    snow = climate.get("total_snowfall_cm")
    pleasant = climate.get("days_pleasant_temp")
    freeze = climate.get("days_below_freezing")
    bits = []
    if rainy is not None:
        bits.append(f"{int(rainy)} rainy days/yr")
    if snow is not None and snow >= 1:
        bits.append(f"{snow:.0f} cm snow/yr")
    if pleasant is not None:
        bits.append(f"{int(pleasant)} comfortable days/yr")
    if freeze is not None:
        bits.append(f"{int(freeze)} freezing days/yr")
    insight = ("Weather here: " + ", ".join(bits) + ". Rain and snow dirty cars (more wash demand); "
               "comfortable days are prime washing days, while freezing days suppress it.") if bits else None
    pro = (f"~{int(rainy)} rainy days a year keep cars dirty and create wash demand."
           if rainy else "Mild, dry weather keeps cars washable year-round.")
    con = (f"~{int(freeze)} freezing days a year can shut washing down."
           if freeze else "Little weather-driven dirt to force frequent washes.")
    favorable = (pleasant or 0) >= 150 and (freeze or 0) <= 60
    return {"insight": insight, "pro": pro, "con": con,
            "conclusion": "Weather is " + ("favorable" if favorable else "mixed") + " for steady wash demand."}


def rule_competition(cdata: dict) -> dict:
    cdata = cdata or {}
    comps = cdata.get("competitors") or []
    count = cdata.get("count", len(comps))
    nearest = comps[0] if comps else {}
    nd = nearest.get("distance_miles")
    nr = nearest.get("rating")
    insight = f"{count} competing car wash{'es' if count != 1 else ''} within {COMPETITOR_RADIUS_MILES:.0f} miles"
    if nd is not None:
        insight += f"; nearest is {nd:.1f} mi away"
        if nr is not None:
            insight += f" ({nr:.1f}★)"
    insight += "."
    pro = ("Few nearby competitors — less rivalry for wash demand." if count <= 2
           else "A proven car-wash market — demand clearly exists here.")
    con = ("A close, well-rated competitor can pull demand away."
           if (nd is not None and nd <= 1.0) else "Several competitors share the local demand.")
    conclusion = ("Low competition — favorable." if count <= 2
                  else "Crowded — differentiation matters." if count >= 6 else "Moderate competition.")
    return {"insight": insight, "pro": pro, "con": con, "conclusion": conclusion}


def rule_retail(retail: dict) -> dict:
    retail = retail or {}
    anchors = retail.get("anchors") or []
    grocery = retail.get("grocery_count_1mile") or 0
    food = retail.get("food_count_0_5miles") or 0
    parts = []
    for label, key in [("warehouse club", "costco_dist"), ("Walmart", "walmart_dist"), ("Target", "target_dist")]:
        d = retail.get(key)
        if d is not None:
            parts.append(f"{label} {d:.1f} mi")
    has_any = bool(anchors or grocery or food)
    insight = f"{len(anchors)} retail anchors within {RETAIL_RADIUS_FAR_MILES:.0f} miles"
    if parts:
        insight += f" ({', '.join(parts)})"
    insight += f"; {grocery} grocery within 1 mi, {food} food spots within ½ mi. "
    insight += ("Busy retail feeds errand trips and impulse washes." if has_any
                else "Sparse retail means weaker passing/errand traffic.")
    pro = ("Strong retail draw nearby pulls steady passing traffic." if has_any
           else "Less retail competition for the parcel.")
    con = ("Retail traffic alone doesn't guarantee wash conversion." if has_any
           else "Limited nearby shopping traffic to feed impulse washes.")
    conclusion = ("Good retail co-tenancy." if (len(anchors) >= 3 or grocery >= 2) else "Modest retail support.")
    return {"insight": insight, "pro": pro, "con": con, "conclusion": conclusion}


def rule_gas(gas: list) -> dict:
    gas = gas or []
    within = sorted([s for s in gas if s.get("distance_miles") is not None
                     and s["distance_miles"] <= GAS_RADIUS_FAR_MILES], key=lambda s: s["distance_miles"])
    nearest = within[0] if within else {}
    nd = nearest.get("distance_miles")
    nm = nearest.get("name")
    ht = is_high_traffic_gas_brand(nm)
    insight = f"{len(within)} gas stations within {GAS_RADIUS_FAR_MILES:.0f} miles"
    if nm and nd is not None:
        insight += f"; nearest {nm} at {nd:.1f} mi"
    if ht:
        insight += " (high-traffic brand)"
    insight += ". Busy fuel stops drive impulse car washes."
    pro = ("A close, high-traffic fuel brand means lots of passing drivers." if ht
           else "Nearby fuel stops add passing traffic.")
    con = ("Few/distant gas stations mean weaker fuel-stop traffic." if len(within) <= 1
           else "Fuel traffic helps but isn't decisive on its own.")
    conclusion = ("Strong fuel-driven traffic." if (ht or len(within) >= 4) else "Some fuel-driven traffic.")
    return {"insight": insight, "pro": pro, "con": con, "conclusion": conclusion}


# ───────────────────────── rendering ─────────────────────────
def _insight_card(d: dict, source: str):
    d = d or {}
    if not any(d.get(k) for k in ("insight", "pro", "conclusion")):
        st.caption("No insight available for this dimension.")
        return
    if d.get("insight"):
        st.markdown(f"**Insight.** {d['insight']}")
    if d.get("pro"):
        st.success(f"**Helps** · {d['pro']}")
    if d.get("conclusion"):
        st.markdown(f"**Takeaway.** {d['conclusion']}")
    st.caption(source)


def _render_map(lat, lon, markers, demo=False):
    fmap = folium.Map(location=[lat, lon], zoom_start=13, tiles="cartodbpositron", control_scale=True)
    for r_mi, rcol in [(1, "#9aa6b2"), (3, "#c4ccd4")]:          # 1 mi / 3 mi reference rings
        folium.Circle([lat, lon], radius=r_mi * METERS_PER_MILE, color=rcol, weight=1,
                      fill=False, dash_array="4").add_to(fmap)
    # one toggleable FeatureGroup per category → the LayerControl is the legend with removable markers
    counts = {}
    for mk in markers:
        if mk["category"] != "origin":
            counts[mk["category"]] = counts.get(mk["category"], 0) + 1
    groups = {}
    for cat, (col, lbl, emoji) in _CAT_STYLE.items():
        if cat == "origin":
            continue
        groups[cat] = folium.FeatureGroup(name=f"{emoji} {lbl} ({counts.get(cat, 0)})", show=True)
    for mk in markers:
        col, lbl, emoji = _CAT_STYLE.get(mk["category"], ("#777777", mk["category"], "•"))
        if mk["category"] == "origin":
            folium.Marker([mk["lat"], mk["lon"]], icon=folium.Icon(color="black", icon="star"),
                          tooltip=("📍 site" if demo else f"📍 {mk.get('address') or 'site'}")).add_to(fmap)
            continue
        grp = groups.get(mk["category"])
        if grp is None:
            continue
        bits = [mk.get("name") or lbl]
        if mk.get("type"):
            bits.append(mk["type"])
        elif mk.get("primary_type"):
            bits.append(mk["primary_type"])
        if mk.get("distance_miles") is not None:
            bits.append(f"{mk['distance_miles']:.1f} mi")
        if mk.get("rating") is not None:
            bits.append(f"{mk['rating']:.1f}★")
        if mk.get("high_traffic"):
            bits.append("high-traffic")
        folium.CircleMarker([mk["lat"], mk["lon"]], radius=6, color=col, fill=True, fill_color=col,
                            fill_opacity=0.85, weight=1,
                            tooltip=" · ".join(str(b) for b in bits)).add_to(grp)
    for grp in groups.values():
        grp.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)           # legend: uncheck a row to remove those markers
    out = st_folium(fmap, height=520, use_container_width=True, returned_objects=["last_clicked"])
    return (out or {}).get("last_clicked")


def render(demo: bool = False):
    st.title("🔎 Site analysis — what surrounds a location")

    if "pin" not in st.session_state:
        st.info("No pin yet — drop one in 🗺️ Explore markets or 📍 Forecast, then come back here.")
        return

    lat, lon = st.session_state.pin
    lat, lon = round(lat, 5), round(lon, 5)
    label = f"{lat:.5f}, {lon:.5f}"

    if not calib.GOOGLE_MAPS_API_KEY:
        st.warning("`GOOGLE_MAPS_API_KEY` is not set — competitor / retail / gas data will be empty.")

    with st.spinner("Fetching weather, competitors, retail anchors & gas stations…"):
        fetched = fetch_features(lat, lon)

    climate = fetched.get("climate") or {}
    cdata = fetched.get("competitors_data") or {}
    retail = fetched.get("retail_anchors") or {}
    gas = fetched.get("gas_stations") or []
    comps = cdata.get("competitors") or []
    anchors = retail.get("anchors") or []
    gas_in = [s for s in gas if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_FAR_MILES]

    # ── headline metrics ──
    m = st.columns(5)
    m[0].metric("Competitors ≤4 mi", len(comps))
    nd = comps[0].get("distance_miles") if comps else None
    m[1].metric("Nearest competitor", f"{nd:.1f} mi" if nd is not None else "—")
    m[2].metric("Retail anchors ≤3 mi", len(anchors))
    m[3].metric("Gas stations ≤3 mi", len(gas_in))
    pleasant = climate.get("days_pleasant_temp")
    m[4].metric("Comfortable days/yr", int(pleasant) if pleasant is not None else "—")

    # ── map (click to move the shared pin) ──
    st.markdown("#### 🗺️ Map")
    markers = build_markers(lat, lon, fetched, None if demo else label)
    click = _render_map(lat, lon, markers, demo=demo)
    if click and (round(click["lat"], 5), round(click["lng"], 5)) != (lat, lon):
        st.session_state.pin = (click["lat"], click["lng"])             # map click moves the SHARED pin
        st.rerun()

    # ── insights (rule-based always; optional AI write-up) ──
    use_ai = st.toggle("✨ Rewrite insights with the internal LLM", value=False,
                       help="Calls the app/site_analysis AI summaries on the internal LLM server. Needs network "
                            "access to it (a 2s reachability check guards the UI); otherwise the rule-based "
                            "insights are shown.")
    ai = {}
    if use_ai:
        with st.spinner("Writing insights with the internal LLM…"):
            ai = ai_summaries(fetched)
        if not ai:
            st.info("Internal LLM unreachable right now — showing rule-based insights instead.")

    def _pick(dim, rule):
        d = (ai.get(dim) or {}) if use_ai else {}
        if any(d.get(k) for k in ("insight", "pro", "con", "conclusion")):
            return d, "AI summary (internal LLM, grounded only on the fetched data)."
        return rule, "Rule-based read of the fetched data."

    tabs = st.tabs(["🏁 Competition", "🛒 Retail anchors", "⛽ Gas stations", "🌦️ Weather"])

    with tabs[0]:
        st.markdown(f"**{len(comps)} competing car washes within {COMPETITOR_RADIUS_MILES:.0f} miles** (driving distance).")
        if comps:
            st.dataframe(pd.DataFrame([{
                "Name": c.get("name"), "Distance (mi)": c.get("distance_miles"),
                "Rating": c.get("rating"), "Reviews": c.get("user_rating_count"),
                "Type": c.get("primary_type"), "Address": c.get("address"),
            } for c in comps]), hide_index=True, use_container_width=True)
        else:
            st.caption("No nearby competing car washes found (or no API key).")
        _insight_card(*_pick("competition", rule_competition(cdata)))

    with tabs[1]:
        rc = st.columns(4)
        def _dist(key):
            v = retail.get(key)
            return f"{v:.1f} mi" if v is not None else "—"
        rc[0].metric("Nearest warehouse club", _dist("costco_dist"))
        rc[1].metric("Nearest Walmart", _dist("walmart_dist"))
        rc[2].metric("Grocery ≤1 mi", retail.get("grocery_count_1mile") or 0)
        rc[3].metric("Food ≤½ mi", retail.get("food_count_0_5miles") or 0)
        if anchors:
            st.dataframe(pd.DataFrame([{
                "Name": a.get("name"), "Type": a.get("type"),
                "Distance (mi)": a.get("distance_miles"), "Address": a.get("address"),
            } for a in anchors]), hide_index=True, use_container_width=True)
        else:
            st.caption("No retail anchors found nearby (or no API key).")
        _insight_card(*_pick("retail", rule_retail(retail)))

    with tabs[2]:
        st.markdown(f"**{len(gas_in)} gas stations within {GAS_RADIUS_FAR_MILES:.0f} miles** (driving distance).")
        if gas_in:
            st.dataframe(pd.DataFrame([{
                "Name": s.get("name"), "Distance (mi)": s.get("distance_miles"),
                "Rating": s.get("rating"), "Reviews": s.get("rating_count"),
                "High-traffic brand": "✓" if is_high_traffic_gas_brand(s.get("name")) else "",
                "Address": s.get("address"),
            } for s in sorted(gas_in, key=lambda s: s["distance_miles"])]),
                hide_index=True, use_container_width=True)
        else:
            st.caption("No gas stations found nearby (or no API key).")
        _insight_card(*_pick("gas", rule_gas(gas)))

    with tabs[3]:
        if climate:
            wc = st.columns(len(WEATHER_METRIC_CONFIG))
            for i, mk in enumerate(WEATHER_METRIC_CONFIG):
                val, _unit = get_weather_metric_value_from_climate(climate, mk)
                disp, sub = WEATHER_METRIC_DISPLAY.get(mk, (mk, ""))
                wc[i].metric(disp, f"{val:.0f}" if val is not None else "—", help=sub)
            st.caption("Annual climatology from Open-Meteo (prior full year).")
        else:
            st.caption("No climate data (Open-Meteo unreachable).")
        _insight_card(*_pick("weather", rule_weather(climate)))
