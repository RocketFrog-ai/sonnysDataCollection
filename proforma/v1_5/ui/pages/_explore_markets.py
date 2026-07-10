"""
🗺️ Explore markets — the local-market explorer: pick a pin, see its neighbours within a radius,
watch how new entrants ramp up, the cluster KPI panels, the AI Key-Insights views, the council
verdict, and the tunnel-length proxy chart.

Extracted from app.py during the UI page-split refactor (behavior-preserving code motion — the
logic below is unchanged, only relocated). See proforma/v1_5/ui/app.py for the entrypoint.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import folium
from streamlit_folium import st_folium

from proforma.v1_5.ui.pages._shared import (
    EXPRESS_MIN_MONTHS, ASP_MIN_WASH, ASP_FLOOR_MEM, ASP_FLOOR_RET,
    neighbourhood, add_cluster_regions, add_all_site_dots, pick_default_pin, anon_names,
    gran_picker, rs_dates, gran_date_tickformat,
)

def render(df, site, pins, demo, express_only, radius, smooth):
    try:                                                          # keep the dashboard alive if the package can't import
        from app.pnl_analysis.insights.graph import market_insights
        from app.pnl_analysis.insights.llm import insights_llm_ready
        _INSIGHTS_OK = True
    except Exception as _insights_imp_err:                        # pragma: no cover
        market_insights = None
        def insights_llm_ready(*_a, **_k):
            return False
        _INSIGHTS_OK = False

    # ── POC (kept deliberately separate from the grounded pipeline above): raw-GPT market analysis from the
    #    pin's location alone, no data fed in. Its own module/prompt; reuses only the shared Azure transport. ──
    try:
        from app.pnl_analysis.insights.location_poc import (location_market_analysis, pollinate_analysis,
                                                            competition_scale_analysis, build_competition_response,
                                                            independent_market_research, build_independent_research_response)
        _LOC_POC_OK = True
    except Exception:                                             # pragma: no cover
        location_market_analysis = None
        pollinate_analysis = None
        competition_scale_analysis = None
        build_competition_response = None
        independent_market_research = None
        build_independent_research_response = None
        _LOC_POC_OK = False

    # Optional web-search grounding for the location-only market read (returns citable links). Best-effort:
    # if the module/provider isn't available, market analysis silently runs un-grounded (its old behaviour).
    try:
        from app.pnl_analysis.insights.websearch import search_available as _web_search_available
    except Exception:                                             # pragma: no cover
        def _web_search_available() -> bool:
            return False


    st.title("PROFORMA DEMO")
    if express_only:
        st.caption(f"🚿 **Express-only sites** — every market, cluster, KPI and forecast below uses only the "
                   f"{len(site):,} Express Tunnel sites with ≥{EXPRESS_MIN_MONTHS} months of history.")

    with st.sidebar:
        max_sites = 10  # cap on sites shown; the pin and all new entrants are always kept
        hl_client = None
        if not demo:                                          # highlight one operator's whole footprint (hidden in demo)
            _clients = sorted(site.client_name.dropna().unique())
            _sel = st.selectbox("Highlight operator / brand", ["(none)"] + _clients, index=0)
            hl_client = None if _sel == "(none)" else _sel
        st.divider()
        if "pin" not in st.session_state:
            _k0 = pick_default_pin(site, df, tuple(pins))
            _s0 = site.loc[site.site_key == _k0].iloc[0]
            st.session_state.pin = (float(_s0.lat), float(_s0.lon))       # pin is a free (lat, lon) point
        # freedom: drop a pin anywhere by typing coordinates → shows whatever sites fall in the radius
        with st.expander("📍 Or type any location (lat, lon)"):
            _plat, _plon = st.session_state.pin
            ilat = st.number_input("Latitude", value=float(_plat), format="%.4f", key="ex_lat")
            ilon = st.number_input("Longitude", value=float(_plon), format="%.4f", key="ex_lon")
            if st.button("Drop pin here", width="stretch"):
                st.session_state.pin = (float(ilat), float(ilon))
                st.rerun()

    pin = st.session_state.pin                                            # (lat, lon) free point
    plat, plon = pin
    # Explore: rich-history sites only — ≥30 monthly records (the SAME floor as express) → no half-drawn lines, and
    # we don't dot thin/young sites you can't actually analyze
    MIN_MONTHS = EXPRESS_MIN_MONTHS
    site_rich = site[site.n_obs >= MIN_MONTHS]
    nb_full = neighbourhood(site_rich, plat, plon, radius)
    if nb_full.empty:
        st.warning(f"No sites with ≥{MIN_MONTHS} months of data within {radius} km of this pin — drop it elsewhere or widen the radius.")
        # still show the map so you can see where the data actually is and move the pin there
        st.subheader("Map")
        fmap = folium.Map(location=[plat, plon], zoom_start=9, tiles="cartodbpositron", prefer_canvas=True)
        folium.Circle([plat, plon], radius=radius * 1000, color="#999", weight=1, fill=True, fill_opacity=0.05).add_to(fmap)
        if demo:
            add_cluster_regions(fmap, site, plat, plon, max_km=50)
        else:
            add_all_site_dots(fmap, site_rich)                                # rich-history sites (≥30 mo) on the map, pan anywhere
            if hl_client:
                for _, s in site[site.has_coords & (site.client_name == hl_client)].iterrows():
                    folium.CircleMarker([s.lat, s.lon], radius=6, color="#d4a500", fill=True, fill_color="#ffd000",
                                        fill_opacity=0.95, weight=2, tooltip=f"{s.client_name} (operator)").add_to(fmap)
        folium.Marker([plat, plon], icon=folium.Icon(color="black", icon="star"), tooltip="📍 pin").add_to(fmap)
        mp = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
        lc = (mp or {}).get("last_clicked")
        if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(plat, 5), round(plon, 5)):
            st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
        st.caption((f"Dots = every express site" if express_only else "Dots = every site")
                   + f" with ≥{MIN_MONTHS} months of history — click anywhere on the map to move the pin there.")
        st.stop()
    # cap clutter: always keep every entrant, fill the rest with the nearest incumbents
    keep = nb_full[nb_full.is_entrant]
    n_inc = max(0, max_sites - len(keep))
    inc = nb_full[~nb_full.is_entrant].nsmallest(n_inc, "dist_km")
    nb = pd.concat([keep, inc]).drop_duplicates("site_key").sort_values("op_start").reset_index(drop=True)
    entrants = nb[nb.is_entrant]
    # focal new site = the newest entrant (fallback to the nearest site to the pin)
    focal_key = (entrants.sort_values("op_start").site_key.iloc[-1] if len(entrants)
                 else nb.sort_values("dist_km").site_key.iloc[0])

    _dom = set(nb_full.site_key)
    demo_label = anon_names(site, _dom) if demo else {}                    # site_key -> "Site N" by opening order
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pin", "your pin" if demo else f"{plat:.3f}, {plon:.3f}")
    c2.metric("Sites in market", len(nb_full), help=f"within {radius} km; showing {len(nb)}")
    c3.metric("New entrants", int(nb_full.is_entrant.sum()))
    c4.metric("Local market", f"≤{radius} km")

    # ── map, full width left-to-right ──
    st.subheader("Map")
    if hl_client and not demo:                                # an operator is highlighted → USA-level view of its footprint
        fmap = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron", prefer_canvas=True)
    else:
        fmap = folium.Map(location=[plat, plon], zoom_start=10, tiles="cartodbpositron", prefer_canvas=True)
    if demo:
        # confidential demo: no site dots / no exact pin — nearby cluster regions (≤200 km) shaded by colour,
        # with a soft red blur marking the chosen local market on top
        add_cluster_regions(fmap, site, plat, plon, max_km=200)
        for rad_km, fop in [(radius, 0.08), (radius * 0.55, 0.12), (radius * 0.28, 0.18)]:
            folium.Circle([plat, plon], radius=rad_km * 1000, color="#c0392b", weight=0,
                          fill=True, fill_color="#e6194B", fill_opacity=fop).add_to(fmap)
        mp = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
        lc = (mp or {}).get("last_clicked")           # click anywhere → drop the pin right there
        if lc:
            st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
    else:
        folium.Circle([plat, plon], radius=radius * 1000, color="#999", weight=1, fill=True,
                      fill_opacity=0.05).add_to(fmap)
        # rich-history sites (≥30 mo) as light dots — full footprint, all distances — so you can pan to any market
        # (thin/young sites are intentionally NOT dotted: they can't be analyzed and would just be clutter)
        add_all_site_dots(fmap, site_rich)
        for _, s in nb.iterrows():
            if s.site_key == focal_key:
                color, rad = "#e6194B", 9
            elif s.is_entrant:
                color, rad = "#f58231", 7
            else:
                color, rad = "#5b8db8", 6
            folium.CircleMarker(
                [s.lat, s.lon], radius=rad, color=color, fill=True, fill_color=color, fill_opacity=0.9, weight=2,
                tooltip=f"{s.client_name} · opened {s.op_start:%b %Y} · {s.dist_km:.1f} km"
                        + (" · NEW" if s.is_entrant else "")).add_to(fmap)
        if hl_client:                                            # mark every site of the chosen operator in a separate colour
            for _, s in site[site.has_coords & (site.client_name == hl_client)].iterrows():
                folium.CircleMarker([s.lat, s.lon], radius=6, color="#d4a500", fill=True, fill_color="#ffd000",
                                    fill_opacity=0.95, weight=2, tooltip=f"{s.client_name} (operator)").add_to(fmap)
        folium.Marker([plat, plon], icon=folium.Icon(color="black", icon="star"),
                      tooltip="📍 pin").add_to(fmap)
        mp = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
        lc = (mp or {}).get("last_clicked")                          # click ANYWHERE -> drop the pin there
        if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(plat, 5), round(plon, 5)):
            st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
        if hl_client:
            st.caption(f"🟡 yellow = every site operated by **{hl_client}**.")

    # ───────────────────────── cluster-wise KPI panels ─────────────────────────
    # The 6 KPIs (retail/membership wash & revenue + the two ASPs), summed across the whole
    # cluster each month — the same set shown in final_modelling/six_year_app.py, but cluster-level.
    st.divider()
    ckeys = nb_full.site_key.tolist()
    cdesc = f"this local market · {len(ckeys)} sites" if demo else f"local market ≤{radius} km · {len(ckeys)} sites"
    st.subheader(f"Local-market KPIs over time — {cdesc}")
    sub = df[df.site_key.isin(ckeys)].copy()
    # Null corrupted revenue (feed dropped to ~0 with washes intact) so the revenue/ASP charts and the AI
    # insights below aren't deflated by it. Same floor as the P&L block; washes & purchases are left intact.
    _subr = (sub.ret_wash_count >= ASP_MIN_WASH) & (sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan) < ASP_FLOOR_RET)
    _subm = (sub.mem_wash_count >= ASP_MIN_WASH) & (sub.mem_revenue / sub.mem_wash_count.replace(0, np.nan) < ASP_FLOOR_MEM)
    sub.loc[_subr.fillna(False), "ret_revenue"] = np.nan
    sub.loc[_subm.fillna(False), "mem_revenue"] = np.nan
    sub["tot_revenue"] = sub[["mem_revenue", "ret_revenue"]].sum(axis=1, min_count=1)
    sub["asp_ret"] = sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan)        # retail ASP = revenue ÷ retail washes
    sub["asp_mem"] = sub.mem_revenue / sub.mem_purchase_count.replace(0, np.nan)    # membership ASP = revenue ÷ membership PURCHASES
    # one figure per group (washes → revenue → ASP). Grounded Key Insights are shown ONCE as a card up top
    # (the Analysis → Key Insights view), not repeated under each chart.
    GROUPS = [
        ("Washes", [("tot_wash_count", "Total washes", "count"), ("ret_wash_count", "Retail washes", "count"), ("mem_wash_count", "Membership washes", "count")]),
        ("Revenue", [("tot_revenue", "Total revenue ($)", "$"), ("ret_revenue", "Retail revenue ($)", "$"), ("mem_revenue", "Membership revenue ($)", "$")]),
        ("ASPs", [("asp_ret", "ASP per wash — retail ($)", "$"), ("asp_mem", "ASP per membership — membership ($)", "$")]),
    ]
    name_of = demo_label if demo else site.set_index("site_key").client_name.to_dict()
    PALETTE = ["#2E86DE", "#16a085", "#8e44ad", "#e67e22", "#27ae60", "#2980b9", "#c0392b", "#d35400", "#7f8c8d",
               "#2c3e50", "#1abc9c", "#9b59b6", "#34495e", "#f39c12", "#3498db", "#e74c3c", "#95a5a6", "#0a84ff"]
    order = [k for k in ckeys if k != focal_key] + ([focal_key] if focal_key in ckeys else [])   # draw the focal site LAST so it sits on top
    gframes = {}                                                                   # even monthly grid per site (reused across groups)
    for k in order:
        g = sub[sub.site_key == k].set_index("date").sort_index()
        gframes[k] = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS")) if len(g) else g

    # ── AI Key Insights — 2-node pipeline (compute_metrics -> generate_insights), one read-out per group ──
    # Computed once per market on demand (the button) and stored by market signature, so flipping a group's
    # granularity radio re-renders instantly without another LLM call.
    insights_backend = os.getenv("INSIGHTS_LLM_BACKEND", "azure").strip().lower()
    imeta = nb_full[[c for c in ["site_key", "op_start", "dist_km", "is_entrant", "left_censored"] if c in nb_full.columns]].copy()
    if "left_censored" not in imeta.columns:                                       # safety: pull from `site` if dropped
        imeta = imeta.merge(site[["site_key", "left_censored"]], on="site_key", how="left")
    imeta["name"] = imeta.site_key.map(name_of)                                    # demo-safe ("Site N") names
    isig = (tuple(sorted(ckeys)), str(focal_key), int(radius), bool(demo), insights_backend)


    @st.cache_data(show_spinner=False, ttl=3600)
    def _market_insights_cached(_sig, _sub, _meta, focal_key, backend):
        """Cached per market signature `_sig` (the big frames are underscore-prefixed so they aren't hashed).
        escape_dollars=False → plain `$` (exactly what the /insights API returns), so the react-markdown preview matches."""
        return market_insights(_sub, _meta, focal_key, backend=backend, escape_dollars=False)["insights"]


    # ── Summaries are prepared AUTOMATICALLY on pin/market change, then VIEWED via the dropdown ──
    # No "Generate" click: when a market is chosen we eagerly compute and cache every LLM-backed summary we can
    # support for that market, so switching the dropdown only renders already-prepared output.
    insights_store = st.session_state.setdefault("insights_store", {})
    loc_poc_store = st.session_state.setdefault("loc_poc_store", {})
    pollinate_store = st.session_state.setdefault("pollinate_store", {})
    compete_store = st.session_state.setdefault("compete_store", {})
    independent_store = st.session_state.setdefault("independent_store", {})
    loc_sig = (round(plat, 5), round(plon, 5), int(radius))
    _INDEP_RADII = (3.0, 6.0, 9.0)                                # miles — sized independently by the external-LLM research


    @st.cache_data(show_spinner=False, ttl=3600)
    def _location_poc_cached(_sig, lat, lon, radius_km, backend, use_web):
        """Location-only LLM summary, cached per (rounded location, radius, backend, web-search on/off).
        `use_web` grounds the read on fresh web results and returns citable source links."""
        return location_market_analysis(lat, lon, radius_km=radius_km, backend=backend, use_web_search=use_web)


    @st.cache_data(show_spinner=False, ttl=3600)
    def _nearby_washes_cached(lat, lon, radius_miles=11):
        """Real nearby car washes (name + distance) from Google Places — the ground truth that anchors the
        competitive-saturation read. Cached per (rounded location, radius). [] if the key/fetch is unavailable."""
        try:
            from app.site_analysis.features.active.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
            from app.core import common as _calib
            key = _calib.GOOGLE_MAPS_API_KEY or ""
            if not key:
                return []
            data = get_nearby_competitors(key, lat, lon, radius_miles=radius_miles,
                                          fetch_place_details=False, max_results=20)
            return [{"name": c.get("name"), "distance_miles": c.get("distance_miles")}
                    for c in (data.get("competitors") or []) if c.get("name")]
        except Exception:
            return []


    @st.cache_data(show_spinner=False, ttl=3600)
    def _pollinate_cached(_sig, _qual_text, _quant, _comp, lat, lon, radius_km, backend):
        """Fusion of the summaries, cached per market signature `_sig` (text/dict args underscore-prefixed → not hashed).
        `_comp` is the competition-scale read (or None) folded in as the competitive-saturation dimension."""
        return pollinate_analysis(_qual_text, _quant, lat=lat, lon=lon, radius_km=radius_km, competition=_comp,
                                  backend=backend)


    @st.cache_data(show_spinner=False, ttl=3600)
    def _compete_cached(_sig, lat, lon, radius_km, known_sites, backend, _nearby):
        """LLM competitive-saturation estimate — client footprint vs total landscape, anchored to the real nearby
        washes `_nearby` (name + distance from Google Places). Cached per (location, radius, known set)."""
        return competition_scale_analysis(lat, lon, known_sites=list(known_sites), radius_km=radius_km, backend=backend,
                                          nearby_washes=list(_nearby or []))


    @st.cache_data(show_spinner=False, ttl=3600)
    def _independent_cached(_sig, lat, lon, radii_miles, backend, use_web):
        """Independent EXTERNAL-LLM market research — sizes each radius (3/6/9 mi) from world knowledge only, NO internal
        data. Cached per (location, radii, backend, web on/off). `use_web` grounds it on fresh web results."""
        return independent_market_research(lat, lon, radii_miles=list(radii_miles), backend=backend, use_web_search=use_web)


    MODE_KEY = "✨ Key Insights — grounded in this market's data"
    MODE_DIRECT = "🌍 Direct LLM summary — location only, no data"
    MODE_POLLINATE = "🔀 Pollinated summary — data × location × competition → verdict"
    MODE_COMPETE = "🏁 Competitive saturation — client footprint vs the trade area"
    MODE_INDEPENDENT = "🌐 Independent research — external LLM only, per 3/6/9-mi radius"
    _modes = []
    if _INSIGHTS_OK:
        _modes.append(MODE_KEY)
    if (not demo) and _LOC_POC_OK:                                # Direct/Pollinated/Competition/Independent reveal the city → not in demo
        _modes.append(MODE_DIRECT)
        if _INSIGHTS_OK:
            _modes.append(MODE_POLLINATE)
        _modes.append(MODE_COMPETE)
        if independent_market_research:
            _modes.append(MODE_INDEPENDENT)
    if not _modes:
        _modes = [MODE_KEY]

    # names of the car washes we actually have data for in this radius (real names — this mode reveals the city)
    _known_names = tuple(sorted(name_of.get(k, str(k)) for k in ckeys)) if not demo else tuple()

    # Eager precompute (cached → runs once per new market, then served instantly on every later rerun).
    _llm_ready = insights_llm_ready(insights_backend)
    _web_on = _web_search_available()                             # web-grounded market read available? (Tavily/DDG)

    # ── one central "Try again" — wipe every cached/stored insight so ALL views regenerate from scratch ──
    if _llm_ready and (_INSIGHTS_OK or _LOC_POC_OK):
        if st.button("🔄 Try again — regenerate all insights",
                     help="Clear the cached Key Insights / location read / competition / pollinated summaries and "
                          "rebuild them all fresh for this market (the model output varies each time)."):
            for _fn in (_market_insights_cached, _location_poc_cached, _pollinate_cached,
                        _compete_cached, _nearby_washes_cached, _independent_cached):
                try:
                    _fn.clear()                                   # drop the st.cache_data memoization
                except Exception:
                    pass
            for _store in (insights_store, loc_poc_store, pollinate_store, compete_store, independent_store):
                _store.clear()                                    # drop the per-session stored results
            st.rerun()

    if not _llm_ready:
        st.caption(f"⚠️ `{insights_backend}` LLM endpoint unavailable — summaries can't be prepared right now.")
    else:
        # ── Prepare A / B / C CONCURRENTLY. They are independent (each its own network/LLM call), so on a cold market
        #    we fan them out in a thread pool and wait for all three — cold-start ≈ max(A,B,C) instead of A+B+C. Cached
        #    results are served instantly on rerun (the `… not in …store` guards), so the pool only spins on a new pin. ──
        import threading
        from concurrent.futures import ThreadPoolExecutor
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        except Exception:                                          # pragma: no cover — st internals moved
            add_script_run_ctx = get_script_run_ctx = None

        _ckey = (loc_sig, _known_names)
        _ikey = (loc_sig, _INDEP_RADII, _web_on)                  # independent-research cache key (per location, radii, web)
        _loc_ok = (not demo) and _LOC_POC_OK

        def _prep_B():                                             # Key Insights — grounded on the market's data
            return _market_insights_cached(isig, sub, imeta, focal_key, insights_backend)

        def _prep_A():                                             # Local Market Analysis — location-only read
            return _location_poc_cached(loc_sig, plat, plon, int(radius), insights_backend, _web_on)

        def _prep_C():                                             # Competition Coverage — needs the Places ground truth first
            _nearby = _nearby_washes_cached(plat, plon, 11)
            return _compete_cached(_ckey, plat, plon, int(radius), _known_names, insights_backend, _nearby)

        def _prep_D():                                             # Independent research — external LLM only, per 3/6/9-mi radius
            return _independent_cached(_ikey, plat, plon, _INDEP_RADII, insights_backend, _web_on)

        _jobs = [(k, fn) for k, fn, go in (
            ("B", _prep_B, _INSIGHTS_OK and isig not in insights_store),
            ("A", _prep_A, _loc_ok and loc_sig not in loc_poc_store),
            ("C", _prep_C, _loc_ok and _ckey not in compete_store),
            ("D", _prep_D, _loc_ok and bool(independent_market_research) and _ikey not in independent_store),
        ) if go]

        if _jobs:
            _LABEL = {"A": "Location summary", "B": "Key Insights", "C": "Competition estimate",
                      "D": "Independent research"}
            _ctx = get_script_run_ctx() if get_script_run_ctx else None

            def _with_ctx(fn):                                     # attach the ScriptRunContext so st.cache_data works off-thread
                def _inner():
                    if add_script_run_ctx and _ctx is not None:
                        add_script_run_ctx(threading.current_thread(), _ctx)
                    return fn()
                return _inner

            with st.spinner("Preparing insights for this market (Key Insights · location · competition · independent, in parallel)…"):
                with ThreadPoolExecutor(max_workers=len(_jobs)) as _ex:
                    _futs = {k: _ex.submit(_with_ctx(fn)) for k, fn in _jobs}
                    for k, f in _futs.items():
                        try:
                            _res = f.result()
                            if k == "B":
                                insights_store[isig] = _res
                            elif k == "A":
                                loc_poc_store[loc_sig] = _res
                            elif k == "C":
                                compete_store[_ckey] = _res
                            else:
                                independent_store[_ikey] = _res
                        except Exception as e:
                            st.caption(f"_{_LABEL[k]} couldn't be prepared: {e}_")

        # ── Then pollinate — needs all three in hand (runs after the parallel prep resolves). ──
        _qual, _quant, _out_c = loc_poc_store.get(loc_sig), insights_store.get(isig), compete_store.get(_ckey)
        if _loc_ok and _qual and _quant and _out_c and isig not in pollinate_store:
            try:
                with st.spinner("Combining Key Insights × location × competition → Final Verdict…"):
                    pollinate_store[isig] = _pollinate_cached(
                        isig, _qual["text"], _quant, _out_c, plat, plon, int(radius), insights_backend
                    )
            except Exception as e:
                st.caption(f"_Pollinated summary couldn't be prepared: {e}_")

    gen_mode = st.selectbox("Analysis — pick a view (all summaries prepare automatically when you choose a pin)", _modes,
                            key="analysis_mode")
    group_insights = insights_store.get(isig, {})                 # the per-chart loop below renders this


    def _strip_leading_h1(md: str) -> str:
        """Drop a leading '# Title' line from the model's markdown. The prompts emit an H1 for react-markdown API
        consumers; the Streamlit views supply their own section header, so the H1 would render as a duplicate title."""
        s = (md or "").lstrip()
        if s.startswith("# "):
            return s.partition("\n")[2].lstrip("\n")
        return md


    import json as _json
    import streamlit.components.v1 as _components


    def _react_md(md: str, *, height: int | None = None) -> None:
        """Render markdown through the SAME renderer the frontend uses — react-markdown + remark-gfm (so GFM tables,
        bullets and bold all render exactly as they will in production). Fed the raw API markdown verbatim (plain `$`,
        no KaTeX), this is a faithful preview of what the frontend will show for each /insights* response."""
        md = md or ""
        if height is None:                                            # rough auto-height from content (tables are taller)
            height = int(min(1500, 140 + md.count("\n") * 26 + md.count("|") * 3))
        try:
            base = st.get_option("theme.base") or "light"
        except Exception:
            base = "light"
        dark = base == "dark"
        fg = "#e7e9ee" if dark else "#1f2329"
        border = "#3a3f4b" if dark else "#dfe2e8"
        zebra = "rgba(255,255,255,0.05)" if dark else "rgba(0,0,0,0.03)"
        payload = _json.dumps(md)
        html = """
    <!doctype html><html><head><meta charset="utf-8"><style>
      body{margin:0;padding:2px 4px;background:transparent;color:__FG__;
           font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.55;}
      h1{font-size:1.5em;} h2{font-size:1.25em;} h3{font-size:1.08em;} h1,h2,h3,h4{margin:.7em 0 .35em;font-weight:700;}
      table{border-collapse:collapse;width:100%;margin:.5em 0;font-size:.93em;} th,td{border:1px solid __BORDER__;padding:6px 10px;text-align:left;vertical-align:top;}
      thead th{background:__ZEBRA__;font-weight:700;} tbody tr:nth-child(even){background:__ZEBRA__;}
      ul,ol{margin:.3em 0;padding-left:1.25em;} li{margin:.12em 0;} p{margin:.4em 0;}
      code{background:__ZEBRA__;padding:1px 5px;border-radius:4px;font-size:.9em;} a{color:#4c8bf5;text-decoration:none;}
      hr{border:0;border-top:1px solid __BORDER__;margin:.8em 0;}
    </style></head><body><div id="root"></div>
    <script type="module">
      import React from 'https://esm.sh/react@18';
      import { createRoot } from 'https://esm.sh/react-dom@18/client';
      import ReactMarkdown from 'https://esm.sh/react-markdown@9?deps=react@18,react-dom@18';
      import remarkGfm from 'https://esm.sh/remark-gfm@4';
      const md = __PAYLOAD__;
      createRoot(document.getElementById('root')).render(
        React.createElement(ReactMarkdown, { remarkPlugins: [remarkGfm] }, md)
      );
    </script></body></html>
    """.replace("__FG__", fg).replace("__BORDER__", border).replace("__ZEBRA__", zebra).replace("__PAYLOAD__", payload)
        _components.html(html, height=height, scrolling=True)


    def _key_insights_summary(blocks: dict) -> str:
        """Reconstruct the `/insights` response `summary` from the {Washes,Revenue,ASPs} block dict — mirrors the
        route so the Streamlit view shows exactly what the API returns."""
        def _sub(v):
            low = (v or "").lower()
            if any(s in low for s in ("did not return", "could not generate", "generation failed")):
                return False
            return ("\n- " in v) or (len(v.strip()) > 60)
        parts = [v.strip() for v in blocks.values() if v and _sub(v)]
        return "\n\n".join(parts) or "\n\n".join(v.strip() for v in blocks.values() if v)


    # Each view renders the EXACT markdown its /insights* endpoint returns, through react-markdown + remark-gfm (the same
    # renderer the frontend uses) — a faithful preview — plus an expander with the exact JSON response.
    def _sources_md(srcs):
        if not srcs:
            return ""
        # Blank line (two newlines) MUST separate this from the summary — a pipe-less line right after a GFM
        # table is swallowed as a table row, so a single newline makes "Sources" render inside the table.
        lines = ["", "", "**Sources (live web search):**", ""]
        for i, s in enumerate(srcs, 1):
            lines.append(f"{i}. [{(s.get('title') or s.get('url') or 'source').strip()}]({s.get('url')})")
        return "\n".join(lines)


    if gen_mode == MODE_DIRECT:
        _out = loc_poc_store.get(loc_sig)
        if _out:
            _resp = {"summary": _out["text"]}                 # POST /insights/location → {summary}
            with st.container(border=True):
                st.markdown("#### 🌍 Local Market Analysis")
                _react_md(_strip_leading_h1(_out["text"]) + _sources_md(_out.get("sources")))
                with st.expander("🔎 API response (JSON) — POST /insights/location"):
                    st.json(_resp)
    elif gen_mode == MODE_POLLINATE:
        _qual, _quant = loc_poc_store.get(loc_sig), insights_store.get(isig)
        if _qual and _quant:
            _ckey = (loc_sig, _known_names)
            _out_c = compete_store.get(_ckey)
            _out = pollinate_store.get(isig)
            if _out:
                _comp_summary = build_competition_response(_out_c)["summary"] if (build_competition_response and _out_c) else None
                _resp = {                                     # POST /insights/pollinated → {summary, sources}
                    "summary": _out["text"],
                    "sources": {
                        "key_insights": _key_insights_summary(_quant),
                        "location_analysis": _qual["text"],
                        "competition": _comp_summary,
                    },
                }
                with st.container(border=True):
                    st.markdown("#### 🔀 Pollinated summary — final consolidated read + verdict")
                    _react_md(_strip_leading_h1(_out["text"]))
                    with st.expander("🔎 API response (JSON) — POST /insights/pollinated (summary + the 3 source responses)"):
                        st.json(_resp)
    elif gen_mode == MODE_COMPETE:
        _ckey = (loc_sig, _known_names)
        _out = compete_store.get(_ckey)
        if _out and build_competition_response:
            _resp = build_competition_response(_out)          # POST /insights/competition → {summary}
            with st.container(border=True):
                st.markdown("#### 🏁 Competition Coverage")
                _react_md(_resp["summary"])
                with st.expander("🔎 API response (JSON) — POST /insights/competition"):
                    st.json(_resp)
    elif gen_mode == MODE_KEY:
        if group_insights:
            _summary = _key_insights_summary(group_insights)  # POST /insights → {summary}
            with st.container(border=True):
                st.markdown("#### ✨ Key Insights — grounded in this market's data")
                _react_md(_summary)
                with st.expander("🔎 API response (JSON) — POST /insights"):
                    st.json({"summary": _summary})
    elif gen_mode == MODE_INDEPENDENT:
        _ikey = (loc_sig, _INDEP_RADII, _web_on)
        _out_d = independent_store.get(_ikey)
        if _out_d and build_independent_research_response:
            _resp = build_independent_research_response(_out_d)   # POST /insights/independent-research → {radii, summary, sources}
            with st.container(border=True):
                st.markdown("#### 🌐 Independent market research — external LLM knowledge only")
                st.caption("No internal/operator data used — each radius (3 / 6 / 9 mi) sized from public knowledge alone"
                           + (" (web-grounded)." if _resp.get("sources") else "; web search off.")
                           + " Anything it can't responsibly size is shown as **Not estimable** (never fabricated).")
                _react_md(_strip_leading_h1(_resp["summary"]) + _sources_md(_resp.get("sources")))
                with st.expander("🔎 API response (JSON) — POST /insights/independent-research (per-radius metrics)"):
                    st.json(_resp)

    for gi, (gname, panels) in enumerate(GROUPS):
        gk = gran_picker(f"gran_kpi_{gname}")
        gk_how = "mean" if gname == "ASPs" else "sum"                 # ASP is a per-wash rate → average; washes/$ → sum
        gfig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels], horizontal_spacing=0.06)
        for si, k in enumerate(order):
            g = gframes[k]
            is_focal = (k == focal_key)
            color = "#e6194B" if is_focal else PALETTE[si % len(PALETTE)]
            nm = (str(name_of.get(k, "?"))[:18]) + (" 🆕" if is_focal else "")
            for i, (c, lbl, unit) in enumerate(panels):
                ya = rs_dates(g[c], gk, gk_how)
                if gk == "M" and smooth and smooth > 1:
                    ya = ya.rolling(smooth, center=True, min_periods=1).mean()   # smoothing slider (monthly view only)
                vfmt = "$%{y:,.2f}" if unit == "$" else "%{y:,.0f}"
                gfig.add_trace(go.Scatter(x=ya.index, y=ya.values, mode="lines", name=nm, legendgroup=k, showlegend=(gi == 0 and i == 0),
                                          line=dict(color=color, width=3 if is_focal else 1.4), opacity=1.0 if is_focal else 0.7,
                                          hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}} · {vfmt}<extra></extra>"),
                               row=1, col=i + 1)
        gfig.update_layout(height=340, template="plotly_white", margin=dict(l=8, r=8, t=44, b=10),
                           hovermode="closest", legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        if gk == "Y":
            gfig.update_xaxes(dtick="M12", tickformat="%Y")
        else:
            gfig.update_xaxes(tickformat=gran_date_tickformat(gk))
        st.plotly_chart(gfig, width="stretch", key=f"kpi_{gname}")
        st.divider()

    # ───────────────────────── 🧭 Council verdict (signal-driven; LLM seats = explanation) ─────────────────────────
    # Isolated council/ package: a leakage-clean data signal (market structure + operator scale) makes the
    # build/pass call; the LLM seats explain but don't decide. Also surfaces the honest out-of-fold backtest
    # report + rebuild meeting notes. Wrapped so it can never take down the dashboard.
    try:
        import proforma.v1_6.streamlit_view as _council_view
        _council_pin = st.session_state.get("pin")
        if _council_pin:
            _council_view.render_council(_council_pin[0], _council_pin[1], radius_km=radius)
        _council_view.render_reports()                                # backtest report + meeting notes (always shown)
    except Exception as _council_err:                                 # keep the dashboard alive
        st.caption(f"🧭 Council view unavailable: {_council_err}")
    st.divider()

    # ───────────────────────── Tunnel-length proxy (estimated metres) ─────────────────────────
    # Proxy for tunnel LENGTH in metres, from peak monthly volume:
    #   peak-month total washes ÷ 25 operating days ÷ 10 hours/day ÷ 3.2 cars/hr per metre ≈ tunnel metres.
    # Toggle Operator-wise (an operator's sites here collapsed to one bar, median length) vs Site-wise.
    # Horizontal bars + 10-m range bands keep it readable. Demo-safe: `name_of` ("Site N") / "Operator N".
    st.subheader("Tunnel length proxy — estimated metres")
    tl_group = st.radio("Group by", ["Operator", "Site"], horizontal=True, key="tl_group",
                        help="Operator-wise collapses an operator's sites in this market into one bar "
                             "(median tunnel length); Site-wise shows every site.")
    st.caption("**peak-month washes ÷ 25 days ÷ 10 hours ÷ 3.2** ≈ tunnel length (m), in 10-m range bands "
               "(0–10 · 10–20 · 20–30 · 30–40 · 40m+). Further right / darker = longer; 🆕 = the new site.")
    _BANDS = [("0–10m", "#c6dbef"), ("10–20m", "#9ecae1"), ("20–30m", "#6baed6"), ("30–40m", "#3182bd"), ("40m+", "#08519c")]
    def _bandlabel(m):
        return _BANDS[min(int(m // 10), 4)][0]                                   # 0–9→band0 … ≥40→band4 (40m+)
    def _dedupe(labels):                                                         # unique, readable y-axis names
        seen, out = {}, []
        for l in labels:
            seen[l] = seen.get(l, 0) + 1
            out.append(l if seen[l] == 1 else f"{l} ({seen[l]})")
        return out

    _peak = sub.groupby("site_key")["tot_wash_count"].max()                      # peak month total washes per site
    _site_m = (_peak / 25 / 10 / 3.2).replace([np.inf, -np.inf], np.nan).dropna()   # ÷25d ÷10h ÷3.2 → tunnel metres
    _site_m = _site_m[_site_m.index.isin(ckeys)]
    if len(_site_m):
        base = pd.DataFrame({"site_key": _site_m.index, "metres": _site_m.values})
        base["peak"] = base.site_key.map(_peak)
        base["is_focal"] = base.site_key == focal_key
        if tl_group == "Operator":                                              # collapse each operator's sites → one bar
            base["oid"] = base.site_key.str.split("::").str[0]
            agg = (base.groupby("oid")
                   .agg(metres=("metres", "median"), peak=("peak", "max"), is_focal=("is_focal", "any"),
                        n=("site_key", "size"), k=("site_key", "first")).reset_index())
            if demo:                                                            # demo-safe anonymous operator labels
                agg = agg.sort_values("metres").reset_index(drop=True)
                agg["label"] = [f"Operator {i + 1}" for i in range(len(agg))]
            else:
                agg["label"] = agg["k"].map(lambda x: str(name_of.get(x, "?"))[:26])
            plot_df = agg[["label", "metres", "peak", "is_focal", "n"]].copy()
        else:
            base["label"] = base.site_key.map(lambda x: str(name_of.get(x, "?"))[:26])
            plot_df = base.assign(n=1)[["label", "metres", "peak", "is_focal", "n"]].copy()
        plot_df["label"] = _dedupe(plot_df["label"].tolist())                   # disambiguate same-name sites/operators
        plot_df.loc[plot_df.is_focal, "label"] = plot_df.loc[plot_df.is_focal, "label"] + " 🆕"
        plot_df = plot_df.sort_values("metres").reset_index(drop=True)          # ascending → longest at the TOP (h-bars)
        tlfig = go.Figure()
        for _bl, _bc in _BANDS:                                                 # one trace per band → discrete legend
            d = plot_df[plot_df.metres.map(_bandlabel) == _bl]
            if d.empty:
                continue
            tlfig.add_trace(go.Bar(
                orientation="h", y=d.label, x=d.metres, name=_bl, marker_color=_bc,
                marker_line_color=["#e6194B" if f else "rgba(0,0,0,0)" for f in d.is_focal],
                marker_line_width=[3 if f else 0 for f in d.is_focal],
                customdata=np.stack([d.peak.values, d.n.values], axis=-1),
                text=[f"{m:.0f} m" for m in d.metres], textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>~%{x:.0f} m tunnel (proxy) · " + _bl +
                              "<br>peak month %{customdata[0]:,.0f} washes · %{customdata[1]:.0f} site(s)<extra></extra>"))
        tlfig.update_layout(height=max(260, 42 * len(plot_df) + 120), template="plotly_white", barmode="overlay",
                            bargap=0.35, margin=dict(l=8, r=8, t=10, b=10),
                            legend=dict(orientation="h", y=-0.18, title="range band"),
                            xaxis_title="estimated tunnel length (m)", yaxis_title=None)
        tlfig.update_yaxes(categoryorder="array", categoryarray=plot_df.label.tolist())
        tlfig.update_xaxes(dtick=10, ticksuffix="m", rangemode="tozero")
        for _xv in (10, 20, 30, 40):                                            # range-band boundary gridlines
            tlfig.add_vline(x=_xv, line_dash="dot", line_color="#cccccc", line_width=1)
        st.plotly_chart(tlfig, width="stretch", key="tunnel_length")
        _foc = plot_df[plot_df.is_focal]
        if len(_foc):
            _fr = _foc.iloc[-1]
            _extra = f", median across {int(_fr.n)} sites" if tl_group == "Operator" and _fr.n > 1 else ""
            st.caption(f"🆕 The new site ≈ **{_fr.metres:.0f} m** ({_bandlabel(_fr.metres)} band{_extra}).")
    else:
        st.caption("_No wash data available for this market to estimate tunnel length._")
