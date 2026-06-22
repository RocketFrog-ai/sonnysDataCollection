"""
Site analysis (visual) · beta — an immersive Google-Earth-style site explorer.

Given the shared map pin (the same (lat, lon) used by Explore markets & Forecast) — or a typed
address / lat-lon — this renders a Google Maps Platform dashboard:

  LEFT  (70%)  Photorealistic 3D tiles centred + tilted on the site, with a site marker, rotate /
               zoom / pan / tilt, a Street-View toggle, and Nearby-Places overlays (competitors,
               restaurants, gas, shopping, schools, hospitals, gyms, hotels, offices) you can filter.
  RIGHT (30%)  "Site intelligence" — trade-area feature values pulled from
               earnest-proforma-2.0/data/merged_all_sites.csv (matched to the nearest site by lat/lon):
               population & growth, income, vehicles, hourly traffic, commercial activity & competitor
               density.  The ONE thing read live from Google (not the CSV) is the *nearest competitor*
               list, surfaced in the map's Places layer.

Two API keys (complementary projects), kept separate on purpose:
  • GOOGLE_MAPS_JS_API_KEY  — loads the browser map + Photorealistic 3D tiles + Street View (Map Tiles API).
  • GOOGLE_MAPS_API_KEY     — server-side Places (New) nearby search + Geocoding (Places API New + Geocoding API).
Falls back to GOOGLE_MAPS_API_KEY for the map loader if the JS key isn't set.

No AI "corner-plot / true-false" labels — the map itself reveals frontage, intersections, parking and
surrounding activity.

Run:  cd earnest-proforma-2.0/streamlits && streamlit run app.py
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Make the repo-root `app.*` package importable when run from streamlits/ (see site_analysis_page.py).
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

from app.utils import common as calib  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "merged_all_sites.csv"
EARTH_KM = 6371.0088
FETCH_RADIUS_M = 5000          # fetch once at the widest ring; the UI filters to 500 m–5 km client-side

# Place categories → Places(New) primary types + marker colour. Shared by Python (fetch) & JS (render).
CATS = [
    {"id": "car_wash",   "label": "🏁 Competitors", "types": ["car_wash"], "color": "#e53935", "on": True},
    {"id": "restaurant", "label": "🍔 Restaurants", "types": ["restaurant"], "color": "#fb8c00", "on": False},
    {"id": "mall",       "label": "🛍️ Shopping",    "types": ["shopping_mall", "department_store"], "color": "#8e24aa", "on": False},
    {"id": "gas",        "label": "⛽ Gas",          "types": ["gas_station"], "color": "#00897b", "on": False},
    {"id": "school",     "label": "🏫 Schools",      "types": ["school", "primary_school", "secondary_school"], "color": "#3949ab", "on": False},
    {"id": "hospital",   "label": "🏥 Hospitals",    "types": ["hospital"], "color": "#d81b60", "on": False},
    {"id": "gym",        "label": "💪 Gyms",         "types": ["gym", "fitness_center"], "color": "#43a047", "on": False},
    {"id": "hotel",      "label": "🏨 Hotels",       "types": ["lodging", "hotel"], "color": "#6d4c41", "on": False},
    {"id": "office",     "label": "🏢 Offices",      "types": ["corporate_office"], "color": "#546e7a", "on": False},
]


def js_map_key() -> str:
    """Key for the browser map loader / 3D tiles (Map Tiles project). Falls back to the server key."""
    return os.getenv("GOOGLE_MAPS_JS_API_KEY", "") or (calib.GOOGLE_MAPS_API_KEY or "")


def server_key() -> str:
    """Key for server-side Places (New) + Geocoding."""
    return calib.GOOGLE_MAPS_API_KEY or ""


# ───────────────────────────── trade-area data (from the CSV) ─────────────────────────────
@st.cache_data(show_spinner="Loading trade-area features…")
def load_features() -> pd.DataFrame:
    df = pd.read_csv(CSV, low_memory=False)
    df = df[df[["lat", "lon"]].notna().all(axis=1)].reset_index(drop=True)
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_site(df: pd.DataFrame, lat: float, lon: float):
    """Closest CSV site to the pin → (row, distance_km). The CSV's trade-area features describe this site."""
    d = haversine_km(lat, lon, df.lat.values, df.lon.values)
    i = int(np.argmin(d))
    return df.iloc[i], float(d[i])


def _num(row, col):
    if col not in row or pd.isna(row[col]):
        return None
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return None


def _money(v):
    return f"${v:,.0f}" if v is not None else "—"


def _int(v):
    return f"{v:,.0f}" if v is not None else "—"


def _pct(v, *, frac=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if frac else f"{v:.1f}%"


def _one(v):
    return f"{v:.1f}" if v is not None else "—"


def build_intelligence(row, dist_km: float) -> list[dict]:
    """Format the CSV trade-area values into the right-panel sections (label / value / sub)."""
    g25 = _num(row, "Growth 2025-2020")
    g30 = _num(row, "Growth 2030-2025")
    traffic_keys = [
        ("Overnight", "Nearest StreetLight US Hourly-ttl_overnight"),
        ("Breakfast", "Nearest StreetLight US Hourly-ttl_breakfast"),
        ("Lunch", "Nearest StreetLight US Hourly-ttl_lunch"),
        ("Afternoon", "Nearest StreetLight US Hourly-ttl_afternoon"),
        ("Dinner", "Nearest StreetLight US Hourly-ttl_dinner"),
        ("Night", "Nearest StreetLight US Hourly-ttl_night"),
        ("Highway", "Nearest StreetLight US Hourly-Highway"),
    ]
    return [
        {
            "title": "Population & growth", "icon": "👥",
            "rows": [
                {"label": "2025 population (market)", "value": _int(_num(row, "2025 Estimate"))},
                {"label": "Growth ’20→’25", "value": _pct(g25, frac=True)},
                {"label": "Growth ’25→’30 (proj.)", "value": _pct(g30, frac=True)},
                {"label": "Average age", "value": _one(_num(row, "2025 Average Age"))},
                {"label": "Labor force", "value": _int(_num(row, "Labor Force"))},
                {"label": "Renter-occupied units", "value": _int(_num(row, "Renter-Occupied"))},
            ],
        },
        {
            "title": "Household income", "icon": "💵",
            "rows": [
                {"label": "Median HH income", "value": _money(_num(row, "Median Household Income"))},
                {"label": "Average HH income", "value": _money(_num(row, "Average Household Income"))},
                {"label": "HH with income $50k+", "value": _pct(_num(row, "2025 % HH with Income $50K+"))},
            ],
        },
        {
            "title": "Demand drivers — vehicles", "icon": "🚗",
            "rows": [
                {"label": "Total vehicles in market", "value": _int(_num(row, "Total Vehicles Available in the Market"))},
                {"label": "Avg vehicles / household", "value": _one(_num(row, "Average Number of Vehicles Available"))},
            ],
        },
        {
            "title": "Hourly traffic (StreetLight)", "icon": "🛣️",
            "rows": [{"label": k, "value": _int(_num(row, c))} for k, c in traffic_keys],
        },
        {
            "title": "Commercial activity", "icon": "🏬",
            "rows": [
                {"label": "Mass-merchant stores", "value": _int(_num(row, "Count of ChainXY VT - Mass Merchant"))},
                {"label": "Grocery stores", "value": _int(_num(row, "Count of ChainXY VT - Grocery"))},
                {"label": "Department stores", "value": _int(_num(row, "Count of ChainXY VT - Department Store"))},
            ],
        },
        {
            "title": "Competition (trade area)", "icon": "🏁",
            "rows": [
                {"label": "Car-wash competitors", "value": _int(_num(row, "Count of Car Wash Competitors"))},
                {"label": "Nearest competitor", "value": "live ↗", "live": True},
            ],
        },
    ]


# ───────────────────────────── live Places (New) — server-side, old key ─────────────────────────────
def _search_one(types, lat, lon, key):
    """One Places(New) searchNearby POST → list of {name, lat, lng, m, rating, n, type}."""
    import requests
    body = {
        "includedPrimaryTypes": types, "maxResultCount": 20, "rankPreference": "DISTANCE",
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": float(FETCH_RADIUS_M)}},
    }
    headers = {
        "Content-Type": "application/json", "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "places.displayName,places.location,places.rating,places.userRatingCount,places.primaryType",
    }
    r = requests.post("https://places.googleapis.com/v1/places:searchNearby",
                      json=body, headers=headers, timeout=10)
    out = []
    for p in (r.json().get("places") or []):
        loc = p.get("location") or {}
        plat, plon = loc.get("latitude"), loc.get("longitude")
        if plat is None or plon is None:
            continue
        m = float(haversine_km(lat, lon, np.array([plat]), np.array([plon]))[0] * 1000.0)
        out.append({
            "name": (p.get("displayName") or {}).get("text") or "—",
            "lat": plat, "lng": plon, "m": round(m),
            "rating": p.get("rating"), "n": p.get("userRatingCount"),
            "type": p.get("primaryType"),
        })
    out.sort(key=lambda x: x["m"])
    return out


@st.cache_data(show_spinner="Fetching nearby places…", ttl=60 * 60)
def fetch_places(lat: float, lon: float, key: str) -> dict:
    """All categories at FETCH_RADIUS_M, fetched in parallel and cached per (lat, lon). {} on no key."""
    if not key:
        return {}
    results: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=len(CATS)) as ex:
        futs = {ex.submit(_search_one, c["types"], lat, lon, key): c["id"] for c in CATS}
        for fut, cid in [(f, futs[f]) for f in futs]:
            try:
                results[cid] = fut.result()
            except Exception:
                results[cid] = []
    return results


@st.cache_data(show_spinner="Geocoding…")
def geocode(address: str, key: str):
    """Address → (lat, lon) via Google Geocoding API (server key). Cached. None on failure."""
    if not key:
        return None
    try:
        import requests
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                         params={"address": address, "key": key}, timeout=8)
        js = r.json()
        if js.get("status") == "OK" and js.get("results"):
            loc = js["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None
    return None


# ───────────────────────────── the immersive dashboard (Maps JS) ─────────────────────────────
def dashboard_html(map_key: str, lat: float, lon: float, label: str,
                   match_name: str, match_dist_km: float, sections: list[dict], places: dict) -> str:
    cfg = json.dumps({
        "key": map_key, "lat": lat, "lon": lon, "label": label,
        "matchName": match_name, "matchDist": round(match_dist_km, 2),
        "sections": sections, "places": places,
        "cats": [{"id": c["id"], "label": c["label"], "color": c["color"], "on": c["on"]} for c in CATS],
    })
    return r"""
<div id="root" style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <style>
    #root *{box-sizing:border-box}
    .dash{display:flex;gap:0;height:760px;border:1px solid #e3e6ea;border-radius:12px;overflow:hidden;background:#0b1726}
    .left{flex:7;position:relative;background:#0b1726;min-width:0}
    .right{flex:3;background:#f7f8fa;overflow-y:auto;padding:14px 14px 30px;border-left:1px solid #e3e6ea}
    #map3d,#map2d{position:absolute;inset:0;width:100%;height:100%}
    #map2d,#sv{display:none}
    #sv{position:absolute;inset:0}
    .ctrlbar{position:absolute;top:12px;left:12px;z-index:5;display:flex;gap:6px;flex-wrap:wrap}
    .btn{background:rgba(17,28,45,.86);color:#fff;border:1px solid rgba(255,255,255,.18);
         border-radius:8px;padding:7px 11px;font-size:12.5px;font-weight:600;cursor:pointer;backdrop-filter:blur(6px)}
    .btn:hover{background:rgba(34,52,82,.95)}
    .btn.on{background:#2f6df6;border-color:#2f6df6}
    .chips{position:absolute;bottom:12px;left:12px;right:12px;z-index:5;display:flex;gap:6px;flex-wrap:wrap}
    .chip{background:rgba(17,28,45,.86);color:#dfe7f5;border:1px solid rgba(255,255,255,.16);
          border-radius:20px;padding:5px 11px;font-size:12px;cursor:pointer;backdrop-filter:blur(6px);user-select:none}
    .chip.on{color:#fff;font-weight:600}
    .banner{position:absolute;top:12px;right:12px;z-index:6;background:rgba(180,60,30,.92);color:#fff;
            font-size:11.5px;padding:6px 10px;border-radius:8px;max-width:240px;display:none}
    .secthead{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#8a93a3;margin:16px 0 6px;font-weight:700}
    .h1{font-size:16px;font-weight:800;color:#10243f;margin:0 0 2px}
    .sub{font-size:12px;color:#6b7686;margin:0 0 4px}
    .card{background:#fff;border:1px solid #e7eaef;border-radius:10px;padding:10px 12px;margin-bottom:10px}
    .card .ct{font-size:13px;font-weight:700;color:#1f2d44;margin-bottom:7px}
    .kv{display:flex;justify-content:space-between;align-items:baseline;padding:3px 0;font-size:13px;border-top:1px dashed #eef1f5}
    .kv:first-of-type{border-top:0}
    .kv .k{color:#56606e}.kv .v{font-weight:700;color:#10243f}
    .kv .v.live{color:#2f6df6}
    .poi{font-size:12.5px;padding:6px 0;border-top:1px solid #eef1f5;display:flex;justify-content:space-between;gap:8px}
    .poi:first-child{border-top:0}
    .poi .nm{font-weight:600;color:#1f2d44}.poi .meta{color:#7b8494;white-space:nowrap}
    .star{color:#f5a623}
    .legend{font-size:11px;color:#8a93a3;margin-top:6px}
  </style>

  <div class="dash">
    <div class="left">
      <gmp-map-3d id="map3d" mode="hybrid"></gmp-map-3d>
      <div id="map2d"></div>
      <div id="sv"></div>
      <div class="ctrlbar">
        <div class="btn on" id="b3d">🌍 3D</div>
        <div class="btn" id="b2d">🗺️ Satellite</div>
        <div class="btn" id="bsv">🚶 Street View</div>
        <div class="btn" id="brot">↻ Orbit</div>
      </div>
      <div class="banner" id="banner"></div>
      <div id="fatal" style="display:none;position:absolute;inset:0;z-index:9;background:#0b1726;color:#e7eef9;
           padding:28px;font-size:13px;line-height:1.55;overflow:auto">
        <div style="font-size:15px;font-weight:800;margin-bottom:8px">🛑 Map could not load</div>
        <div id="fatalmsg"></div>
      </div>
      <div class="chips" id="chips"></div>
    </div>
    <div class="right" id="panel"></div>
  </div>
</div>

<script>
const CFG = __CFG__;
const SITE = {lat: CFG.lat, lng: CFG.lon};
const PLACES = CFG.places || {};
const CATS = CFG.cats;
const RADII = [500,1000,3000,5000];
let radius = 1000;

function distLabel(m){return m>=1000?(m/1000).toFixed(1)+' km':Math.round(m)+' m';}

// ── right-hand intelligence panel ────────────────────────────────────────────
function renderPanel(){
  const p=document.getElementById('panel');
  let h=`<div class="h1">Site intelligence</div>
         <div class="sub">${CFG.label}</div>
         <div class="sub">Trade area: <b>${CFG.matchName}</b> · ${CFG.matchDist} km from pin</div>`;
  h+=`<div class="secthead">Nearby (live · Google Places)</div>
      <div class="card"><div class="ct">Nearest competitors</div><div id="nearComp"></div>
      <div class="legend">Live from Google Places — the one layer not read from the CSV.</div></div>`;
  h+=`<div class="secthead">Trade-area features (merged_all_sites.csv)</div>`;
  for(const s of CFG.sections){
    h+=`<div class="card"><div class="ct">${s.icon} ${s.title}</div>`;
    for(const r of s.rows){
      h+=`<div class="kv"><span class="k">${r.label}</span><span class="v ${r.live?'live':''}">${r.value}</span></div>`;
    }
    h+=`</div>`;
  }
  p.innerHTML=h;
}

function renderChips(){
  const c=document.getElementById('chips'); let h='';
  for(const cat of CATS){
    const n=(PLACES[cat.id]||[]).filter(x=>x.m<=radius).length;
    h+=`<div class="chip ${cat.on?'on':''}" data-cat="${cat.id}" style="${cat.on?'background:'+cat.color:''}">${cat.label} ${n}</div>`;
  }
  h+=`<div class="chip" id="radchip" style="margin-left:auto">◎ ${radius>=1000?radius/1000+' km':radius+' m'}</div>`;
  c.innerHTML=h;
  c.querySelectorAll('.chip[data-cat]').forEach(el=>el.onclick=()=>{
    const cat=CATS.find(x=>x.id===el.dataset.cat); cat.on=!cat.on; renderChips(); updateMarkers();
  });
  document.getElementById('radchip').onclick=()=>{
    radius=RADII[(RADII.indexOf(radius)+1)%RADII.length]; renderChips(); updateMarkers();
  };
}

function renderNearComp(){
  const el=document.getElementById('nearComp'); if(!el) return;
  const list=(PLACES['car_wash']||[]).filter(x=>x.m<=radius).slice(0,6);
  if(!list.length){el.innerHTML='<span style="color:#7b8494">none within '+distLabel(radius)+'</span>';return;}
  el.innerHTML=list.map(c=>{
    const rt=c.rating?`<span class="star">★</span>${(+c.rating).toFixed(1)}`:'';
    return `<div class="poi"><span class="nm">${c.name}</span><span class="meta">${rt} · ${distLabel(c.m)}</span></div>`;
  }).join('');
}

// ── Maps bootstrap ───────────────────────────────────────────────────────────
let map2d, pano, advLib, the3d, orbiting=false, orbitTimer=null;
const built={};   // catId -> [{data, m, am(2D), m3(3D)}]

(g=>{var h,a,k,p="The Google Maps JavaScript API",c="google",l="importLibrary",q="__ib__",m=document,b=window;
b=b[c]||(b[c]={});var d=b.maps||(b.maps={}),r=new Set,e=new URLSearchParams,
u=()=>h||(h=new Promise(async(f,n)=>{await (a=m.createElement("script"));e.set("libraries",[...r]+"");
for(k in g)e.set(k.replace(/[A-Z]/g,t=>"_"+t[0].toLowerCase()),g[k]);e.set("callback",c+".maps."+q);
a.src=`https://maps.${c}apis.com/maps/api/js?`+e;d[q]=f;a.onerror=()=>h=n(Error(p+" could not load."));
a.nonce=m.querySelector("script[nonce]")?.nonce||"";m.head.append(a)}));d[l]?console.warn(p+" only loads once. Ignoring:",g):d[l]=(f,...n)=>r.add(f)&&u().then(()=>d[l](f,...n))})
({key:CFG.key, v:"alpha"});

function showFatal(html){
  const f=document.getElementById('fatal'); document.getElementById('fatalmsg').innerHTML=html;
  f.style.display='block';
}
window.gm_authFailure=function(){
  showFatal('Google rejected the Maps key (<code>gm_authFailure</code>).<br>Check that <b>Maps JavaScript API</b> '
    +'is enabled and billing is on for the JS-key project, and that any key restriction allows this domain.');
};
function hasWebGL2(){ try{return !!document.createElement('canvas').getContext('webgl2');}catch(e){return false;} }

async function init(){
  if(!hasWebGL2()){
    showFatal('Your browser isn’t providing <b>WebGL2</b>, which Google’s Photorealistic 3D Maps require.<br><br>'
      +'➊ Open <code>chrome://settings/system</code> → turn on <b>“Use graphics acceleration when available”</b>.<br>'
      +'➋ Fully <b>quit &amp; relaunch</b> Chrome (not just reload).<br>'
      +'➌ Check <code>chrome://gpu</code> — the <b>WebGL2</b> line should say <i>Hardware accelerated</i>.<br><br>'
      +'On a remote desktop / VM / locked-down corporate browser the GPU may be unavailable and 3D can’t run there.');
    return;
  }
  try{ await google.maps.importLibrary("maps3d"); }catch(e){}
  advLib = await google.maps.importLibrary("marker");

  the3d=document.getElementById('map3d');
  if(the3d && customElements.get('gmp-map-3d')){
    try{
      the3d.center={lat:SITE.lat,lng:SITE.lng,altitude:0};
      the3d.range=600; the3d.tilt=67; the3d.heading=0;
      const {Marker3DElement}=await google.maps.importLibrary("maps3d");
      the3d.append(new Marker3DElement({position:{lat:SITE.lat,lng:SITE.lng,altitude:5},
        altitudeMode:"RELATIVE_TO_GROUND",extruded:true,label:"SITE"}));
    }catch(e){ show3dBanner(); }
  } else { show3dBanner(); }

  const {Map}=await google.maps.importLibrary("maps");
  map2d=new Map(document.getElementById('map2d'),{
    center:SITE, zoom:18, tilt:47, heading:0, mapTypeId:'hybrid',
    mapId:'DEMO_MAP_ID', streetViewControl:false, mapTypeControl:false, fullscreenControl:false});
  new advLib.AdvancedMarkerElement({map:map2d,position:SITE,content:pin('#2f6df6','SITE')});

  const {StreetViewPanorama}=await google.maps.importLibrary("streetView");
  pano=new StreetViewPanorama(document.getElementById('sv'),{position:SITE,pov:{heading:30,pitch:5},zoom:1});

  wireButtons(); updateMarkers();
}

function pin(color,txt){
  const d=document.createElement('div');
  d.style.cssText=`background:${color};color:#fff;font:700 10px sans-serif;padding:3px 7px;border-radius:14px;
    border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.4);white-space:nowrap`;
  d.textContent=txt; return d;
}
function dot(color){
  const d=document.createElement('div');
  d.style.cssText=`width:13px;height:13px;border-radius:50%;background:${color};border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.5)`;
  return d;
}
function show3dBanner(){
  const b=document.getElementById('banner'); b.style.display='block';
  b.textContent='Photorealistic 3D unavailable on this key — showing tilted satellite. Enable “Map Tiles API”.';
  document.getElementById('map3d').style.display='none';
  document.getElementById('map2d').style.display='block';
  document.getElementById('b2d').classList.add('on'); document.getElementById('b3d').classList.remove('on');
}

function wireButtons(){
  const show=(id)=>['map3d','map2d','sv'].forEach(x=>document.getElementById(x).style.display=(x===id?'block':'none'));
  const b3=document.getElementById('b3d'), b2=document.getElementById('b2d'), bs=document.getElementById('bsv');
  const setOn=(el)=>{[b3,b2,bs].forEach(x=>x.classList.remove('on'));el.classList.add('on');};
  b3.onclick=()=>{ if(document.getElementById('banner').style.display==='block'){show('map2d');setOn(b2);return;} show('map3d');setOn(b3); };
  b2.onclick=()=>{show('map2d');setOn(b2);};
  bs.onclick=()=>{show('sv');setOn(bs);};
  document.getElementById('brot').onclick=()=>{
    orbiting=!orbiting; document.getElementById('brot').classList.toggle('on',orbiting);
    if(orbiting){orbitTimer=setInterval(()=>{
      if(the3d&&the3d.style.display!=='none'){the3d.heading=(the3d.heading+0.6)%360;}
      else if(map2d){map2d.setHeading((map2d.getHeading()+1)%360);}
    },60);} else clearInterval(orbitTimer);
  };
}

// ── overlay markers from server-fetched places; toggle by category + radius ──
async function updateMarkers(){
  let Marker3DElement=null;
  if(the3d&&customElements.get('gmp-map-3d')&&document.getElementById('banner').style.display!=='block'){
    try{({Marker3DElement}=await google.maps.importLibrary("maps3d"));}catch(e){}
  }
  for(const cat of CATS){
    const data=PLACES[cat.id]||[];
    if(!built[cat.id]) built[cat.id]=data.map(d=>({d, m:d.m, am:null, m3:null}));
    for(const it of built[cat.id]){
      const vis = cat.on && it.m<=radius;
      const loc={lat:it.d.lat,lng:it.d.lng};
      // 2D
      if(vis && !it.am){
        it.am=new advLib.AdvancedMarkerElement({map:map2d,position:loc,content:dot(cat.color),
          title:`${it.d.name} · ${distLabel(it.m)}`});
      } else if(it.am){ it.am.map = vis ? map2d : null; }
      // 3D
      if(vis && !it.m3 && Marker3DElement){
        it.m3=new Marker3DElement({position:{lat:loc.lat,lng:loc.lng,altitude:2},
          altitudeMode:"RELATIVE_TO_GROUND",label:it.d.name});
        the3d.append(it.m3);
      } else if(!vis && it.m3){ it.m3.remove(); it.m3=null; }
    }
  }
  renderNearComp();
}

renderPanel(); renderChips(); init();
</script>
""".replace("__CFG__", cfg)


# ───────────────────────────── page entry point ─────────────────────────────
def render(demo: bool = False):
    st.title("🛰️ Site analysis (visual) · beta")
    st.caption("Google-Earth-style 3D explorer + live Places overlays, with trade-area intelligence "
               "from `merged_all_sites.csv`. Pan / tilt / orbit the map to read frontage, intersections, "
               "parking and surrounding activity for yourself.")

    df = load_features()
    if "pin" not in st.session_state:
        r0 = df.iloc[0]
        st.session_state.pin = (float(r0.lat), float(r0.lon))

    # ── location input: address (geocoded) or lat/lon → moves the shared pin ──
    with st.expander("📍 Location — address or coordinates", expanded=False):
        c1, c2 = st.columns([3, 2])
        with c1:
            addr = st.text_input("Address", placeholder="e.g. 1600 Amphitheatre Pkwy, Mountain View, CA")
            if st.button("Geocode address", use_container_width=True) and addr.strip():
                ll = geocode(addr.strip(), server_key())
                if ll:
                    st.session_state.pin = ll
                    st.rerun()
                else:
                    st.warning("Couldn’t geocode that address (check the address / Geocoding API).")
        with c2:
            plat, plon = st.session_state.pin
            ilat = st.number_input("Latitude", value=float(plat), format="%.5f", key="vlat")
            ilon = st.number_input("Longitude", value=float(plon), format="%.5f", key="vlon")
            if st.button("Drop pin here", use_container_width=True):
                st.session_state.pin = (float(ilat), float(ilon))
                st.rerun()

    lat, lon = st.session_state.pin
    lat, lon = round(float(lat), 5), round(float(lon), 5)
    label = f"📍 {lat:.5f}, {lon:.5f}"

    if not js_map_key():
        st.error("No Google Maps key set — the map won't load. Set `GOOGLE_MAPS_JS_API_KEY` "
                 "(Map Tiles + Maps JavaScript API) and/or `GOOGLE_MAPS_API_KEY`.")
        return
    if not server_key():
        st.warning("`GOOGLE_MAPS_API_KEY` is not set — the Nearby-Places overlays & nearest-competitor "
                   "list (Places API New) will be empty. The 3D map still works.")

    row, dist_km = nearest_site(df, lat, lon)
    name = str(row.get("client_name") or "nearest site")
    sections = build_intelligence(row, dist_km)
    if dist_km > 15:
        st.info(f"Closest trade-area record is **{name}**, ~{dist_km:.0f} km away — its feature values may not "
                "describe this exact spot. The map & live Places still reflect the pin.")

    places = fetch_places(lat, lon, server_key()) if server_key() else {}

    components.html(
        dashboard_html(js_map_key(), lat, lon, label, name, dist_km, sections, places),
        height=790, scrolling=False,
    )
