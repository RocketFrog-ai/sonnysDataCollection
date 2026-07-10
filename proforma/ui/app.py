"""
Local Market Explorer — car-wash clusters, new-site entry & ramp-up.

Pick a pin on the map (or 🎲 Random). The app finds its neighbours within a radius
(the local market), draws each site's wash-count time series, and highlights the
NEW site(s) that entered the market in a distinct colour — so you can watch the new
site ramp up and see how the incumbents' series respond before/after the opening.

Run:   streamlit run proforma/ui/app.py   (from the repo root)

This is a thin entrypoint: sys.path/`app` namespace bootstrap, page config, and dispatch to the
three modes, each implemented in its own module under pages/ (kept out of Streamlit's automatic
multipage-nav discovery via the leading underscore — see pages/_shared.py for why):
  pages/_shared.py            constants + data/geo helpers used by 2+ modes
  pages/_pinpoint_forecast.py 📍 Pinpoint forecast (drop_pin_ui)
  pages/_explore_markets.py   🗺️ Explore markets (render)
  site_visual_page.py         🛰️ Sitewise (unchanged, already its own module)
"""
from __future__ import annotations
from pathlib import Path

# ── make the repo-root `app.*` package importable from streamlits/ (the local app.py would otherwise
#    shadow it). Mirrors site_analysis_page.py: register `app` as a namespace package at <repo>/app. ──
import importlib.machinery
import importlib.util
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]   # proforma/ui/x.py -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_APP_DIR = _REPO_ROOT / "app"
if _APP_DIR.is_dir() and ("app" not in sys.modules or not hasattr(sys.modules.get("app"), "__path__")):
    _app_mod = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("app", None, is_package=True))
    _app_mod.__path__ = [str(_APP_DIR)]
    sys.modules["app"] = _app_mod

import streamlit as st
import site_visual_page as svp
from proforma.ui.panels import _shared, _pinpoint_forecast, _explore_markets

# ───────────────────────────── UI ─────────────────────────────
st.set_page_config(page_title="Local Market Explorer", layout="wide")
# ── light global polish: card-style metrics, tidier expanders/headers (no logic, pure CSS) ──
st.markdown("""
<style>
/* theme-agnostic: semi-transparent grey tints work on BOTH light and dark backgrounds (no hardcoded white) */
[data-testid="stMetric"]{background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.22);
  border-radius:12px;padding:10px 16px;}
[data-testid="stMetricLabel"]{opacity:.8;}
[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:600;}
[data-testid="stMetricDelta"]{font-size:.82rem;}
div[data-testid="stExpander"] details{border:1px solid rgba(128,128,128,.22);border-radius:12px;}
h1{letter-spacing:-.02em;} h2,h3{letter-spacing:-.01em;padding-top:.15rem;}
hr{margin:.7rem 0;}
/* readable AI summaries — generous line-height, spaced headings & lists so a long LLM report scans cleanly */
[data-testid="stMarkdownContainer"] p{line-height:1.58;margin:.35rem 0;}
[data-testid="stMarkdownContainer"] li{line-height:1.5;margin:.14rem 0;}
[data-testid="stMarkdownContainer"] ul,[data-testid="stMarkdownContainer"] ol{margin:.25rem 0 .6rem;}
[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{margin:.85rem 0 .3rem;font-weight:650;}
[data-testid="stMarkdownContainer"] h4{font-size:1.02rem;}
/* bordered summary cards: rounded, soft tint, comfortable padding */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;}
[data-testid="stVerticalBlockBorderWrapper"]>div{padding:.2rem .25rem;}
</style>
""", unsafe_allow_html=True)
MODES = ["🛰️ Sitewise", "🗺️ Explore markets", "📍 Pinpoint forecast"]
with st.sidebar:
    st.header("Controls")
    demo = st.toggle("👔 Client demo (anonymized)", value=False,
                     help="Hide identities: sites become 'Site 1, 2, 3…' by opening order, operators become "
                          "'Operator N', and the map shows a shaded red region instead of exact dots.")
    express_only = st.toggle("🚿 Express-only sites", value=False,
                     help="Restrict markets, clusters, KPIs and the forecast's wash trajectory / level anchor to "
                          "Express Tunnel car washes (drops Flex, full-service, self-serve, unknown, etc.). The "
                          "P&L cost curves and the global campaign-evidence popover still use all sites.")
# load AFTER the express toggle so the whole app (clusters included) reruns on the filtered universe
df, site = _shared.load_data(express_only)
pins = _shared.interesting_pins(site)
with st.sidebar:
    # mode switcher — segmented control (acts like app tabs)
    app_mode = st.segmented_control("Mode", MODES, default=MODES[0], key="app_mode") or MODES[0]
    if app_mode.startswith("🗺️"):
        # distance + smoothing apply to Explore only (Forecast is fixed to the trained cluster radius)
        st.divider()
        radius = st.slider("Neighbour radius (km)", 2, 40, 20, 1, key="radius")
        smooth = st.select_slider("Smoothing (months)", [1, 2, 3, 4, 6], value=1, key="smooth")
if app_mode.startswith("📍"):
    _pinpoint_forecast.drop_pin_ui(df, site, _shared.get_model(), demo, express_only)
    st.stop()
if app_mode.startswith("🛰️"):
    if "pin" not in st.session_state:                        # seed the SAME default point Explore uses
        _k0 = _shared.pick_default_pin(site, df, tuple(pins))
        _s0 = site.loc[site.site_key == _k0].iloc[0]
        st.session_state.pin = (float(_s0.lat), float(_s0.lon))
    svp.render(demo)
    st.stop()
_explore_markets.render(df, site, pins, demo, express_only, radius, smooth)
