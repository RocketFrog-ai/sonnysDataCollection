"""
Section — Operator clusters. Pick a multi-site operator; get their whole story.

The reader picks a company, not a place. Everything below is that one operator: every site it owns
on a map, the **3-mile trade-area circle** around each pin, how far apart it built them and in what
order, what each site has washed month on month and year on year, and what the sites already
trading did each time it opened another.

Why the 3-mile circle is the point. A site's demographics, traffic counts and competitor counts are
pulled for a trade area of about that radius — so where two of one operator's circles overlap, both
sites are being credited with the same households. §④ finds the trade area explains none of the
variance in wash volume; this section shows one mechanical reason that can happen inside a single
company's own estate.

The maths is in `cluster_data.py` and is Streamlit-free, so the notebook reports the same numbers.

Chart choices, since one operator can hold 79 sites across 5 states and nine is past any honest
categorical palette:

  • **opening order is the one colour encoding**, on a single blue ramp, light = opened first, used
    identically on the map and the per-site panels so a colour means the same thing twice;
  • **wash volume is a separate single-hue orange ramp**, the same one §⓪ uses for volume, so the
    two encodings are never confused for each other;
  • per-site history is **small multiples on a shared y-axis** when the selection is small enough to
    read, and a **site × month heatmap** when it is not — never 79 overlaid lines;
  • every chart has a table underneath carrying the same numbers, which is also the required relief
    for the pale end of the ramps on the light surface.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import cluster_data as cd
from ui import DARK, GRID, INK2, MUTED, S1, S2, S3, SURFACE, callout, html_table, style

# Sequential ramp for OPENING ORDER — pale = the operator's first site, deep = its newest. One hue,
# light→dark, same direction in both themes. The deep end stops at a saturated blue rather than a
# navy so the newest sites stay legible on the dark surface (§⓪'s orange ramp stops short of black
# for the same reason). Lightness is strictly monotonic; every step clears 3:1 on the dark surface,
# and on the light surface the two palest steps are given a mark outline instead — which is why
# every marker and bar below carries one.
ORDER_RAMP = ["#dbe9fc", "#aecbf3", "#7ba9e8", "#4a86dd", "#2f6fd0"]

# Sequential ramp for WASH VOLUME, shared with §⓪ so intensity reads the same across the app.
VOLUME_SCALE = [[0.0, "#fde3cf"], [0.25, "#f9bf90"], [0.5, "#f2914f"],
                [0.75, "#dd6524"], [1.0, "#a8410f"]]

KM_PER_MILE = cd.KM_PER_MILE

# Above this many sites, per-site panels stop being readable and the heatmap takes over.
PANEL_LIMIT = 12


# =================================================================================================
# Cached wrappers — the module memoises too, but Streamlit's cache also survives a widget change.
# =================================================================================================

@st.cache_data(show_spinner=False)
def _operators() -> pd.DataFrame:
    return cd.operator_index()


@st.cache_data(show_spinner=False)
def _sites(client_id: str, max_km: float, radius_mi: float) -> pd.DataFrame:
    return cd.operator_sites(client_id, max_km, radius_mi)


@st.cache_data(show_spinner=False)
def _headline(client_id: str, max_km: float, radius_mi: float) -> dict:
    return cd.operator_headline(client_id, max_km, radius_mi)


@st.cache_data(show_spinner=False)
def _months(client_id: str, keys: tuple) -> pd.DataFrame:
    return cd.operator_months(client_id, keys)


@st.cache_data(show_spinner=False)
def _years(client_id: str, keys: tuple) -> pd.DataFrame:
    return cd.operator_years(client_id, keys)


@st.cache_data(show_spinner=False)
def _effect(client_id: str, keys: tuple, max_km: float, radius_mi: float) -> pd.DataFrame:
    m = cd.operator_sites(client_id, max_km, radius_mi)
    return cd.effect_for(m[m.site_key.isin(keys)])


@st.cache_data(show_spinner=False)
def _effect_all(max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.opening_effect_all(max_km, min_sites)


@st.cache_data(show_spinner=False)
def _clusters(max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.build(max_km, min_sites)[0]


@st.cache_data(show_spinner=False)
def _coords() -> dict:
    return cd.coord_quality()


# =================================================================================================
# Helpers
# =================================================================================================

def _shade(rank: int, n: int) -> str:
    """The opening-order colour for site `rank` of `n`, interpolated along `ORDER_RAMP`."""
    if n <= 1:
        return ORDER_RAMP[-2]
    pos = (rank - 1) / (n - 1) * (len(ORDER_RAMP) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ORDER_RAMP) - 1)
    t = pos - lo
    a = [int(ORDER_RAMP[lo][i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(ORDER_RAMP[hi][i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def _is_dark(hex_colour: str) -> bool:
    """Is this ramp step dark enough that a label on it must be light?

    Relative luminance, the same quantity a contrast ratio is built from, rather than a guess at
    which steps "look dark". 0.35 puts the flip between the ramp's middle and fourth steps.
    """
    srgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.35


def _rings(m: pd.DataFrame, radius_mi: float, points: int = 64) -> tuple[list, list]:
    """Lat/lon for a circle of `radius_mi` around every site, as one `None`-separated path.

    Drawn as real geography rather than a fixed pixel radius, so the circles keep their true size
    when the reader zooms — a pixel radius would silently mean 3 miles at one zoom and 30 at
    another. Longitude is stretched by 1/cos(lat) because a degree of longitude is shorter than a
    degree of latitude everywhere except the equator; without that the "circles" are ellipses.
    """
    r_km = radius_mi * KM_PER_MILE
    dlat = r_km / 111.32
    lats: list[float | None] = []
    lons: list[float | None] = []
    ang = np.linspace(0, 2 * np.pi, points)
    for la, lo in zip(m.lat.values, m.lon.values):
        dlon = dlat / max(math.cos(math.radians(la)), 1e-6)
        lats.extend(list(la + dlat * np.sin(ang)) + [None])
        lons.extend(list(lo + dlon * np.cos(ang)) + [None])
    return lats, lons


def _view(m: pd.DataFrame, radius_mi: float) -> tuple[dict, float]:
    """Map centre and zoom that fit every selected site, plus its trade-area circle, with a margin.

    The floor on the span is the circle's own diameter — fitting a single site to its pin alone
    would zoom to street furniture and push its 3-mile ring off screen.
    """
    pad_deg = (radius_mi * KM_PER_MILE) / 111.32 * 2.4
    lat_span = max(float(m.lat.max() - m.lat.min()), pad_deg)
    lon_span = max(float(m.lon.max() - m.lon.min()), pad_deg)
    zoom = min(math.log2(360.0 / (lon_span * 1.7)), math.log2(180.0 / (lat_span * 1.7)))
    centre = dict(lat=float((m.lat.max() + m.lat.min()) / 2),
                  lon=float((m.lon.max() + m.lon.min()) / 2))
    return centre, float(np.clip(zoom, 3.0, 13.5))


# =================================================================================================
# Render
# =================================================================================================

def render() -> None:
    ops = _operators()

    st.markdown("<div class='kicker'>Evidence pack · ⑥</div>", unsafe_allow_html=True)
    st.title("Operator clusters")
    st.markdown("Pick a **multi-site operator** and get their whole story: every site on a map with "
                "its **3-mile trade area** drawn, how far apart they sit and in what order they "
                "opened, what each one washes month on month and year on year, and what the sites "
                "already trading did each time another one arrived.")

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        labels = dict(zip(ops.client_id, ops.label))
        pick = st.selectbox(f"Operator ({len(ops)} hold more than one site)",
                            list(ops.client_id), format_func=lambda c: labels.get(c, c))
    with c2:
        radius_mi = st.slider("Trade-area radius (miles)", 1.0, 6.0, float(cd.TRADE_AREA_MI), 0.5,
                              help="The circle drawn around each site. A site's demographics and "
                                   "traffic are pulled for about a 3-mile radius.")
    with c3:
        max_km = st.slider("Group sites into a market within (km)", 5, 60, 25, 1,
                           help="Complete linkage: every site in a market is within this distance "
                                "of every other one, not just of its nearest neighbour.")

    m_all = _sites(pick, float(max_km), float(radius_mi))
    if m_all.empty:
        st.warning("This operator has no site with a usable coordinate. See “Data & method”.")
        return
    h = _headline(pick, float(max_km), float(radius_mi))

    markets = (m_all.groupby(["market_id", "market"]).size().rename("sites")
                    .reset_index().sort_values("sites", ascending=False))
    opts = ["__all__"] + list(markets.market_id)
    mk_label = dict(zip(markets.market_id,
                        markets.market + " · " + markets.sites.astype(str) + " sites"))
    mk_label["__all__"] = f"All markets · {len(m_all)} sites"
    chosen = st.radio("Zoom to", opts, format_func=lambda k: mk_label[k], horizontal=True,
                      index=0, key=f"mk_{pick}")
    m = m_all if chosen == "__all__" else m_all[m_all.market_id == chosen].reset_index(drop=True)
    keys = tuple(sorted(m.site_key))
    # Shading is always relative to the OPERATOR's full build-out, never to the current selection —
    # otherwise zooming into a market repaints the survivors and site #40 becomes "the first one".
    n_all = len(m_all)

    # =============================================================================================
    st.divider()
    st.header(f"1 · {h['operator']} — the whole estate")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sites", f"{h['n_sites']:,}",
              f"{h['n_states']} state{'s' if h['n_states'] != 1 else ''} · "
              f"{h['n_markets']} markets", delta_color="off")
    k2.metric("Washes a year", f"{h['washes_per_year']:,.0f}",
              f"typical site {h['median_site']:,.0f}", delta_color="off")
    k3.metric("Built out over", f"{h['build_out_months']:,.0f} mo",
              f"{h['first_open']} → {h['last_open']}", delta_color="off")
    k4.metric(f"Sites sharing a {radius_mi:g}-mile catchment", f"{h['share_overlapping']:.0%}",
              f"{h['n_overlapping']} of {h['n_sites']}", delta_color="off")
    st.caption(f"`client_id` **{pick}** · {h['states']}"
               + (f" · **{h['unplaceable']} further site(s) are not shown** — their coordinate is "
                  "a placeholder, see “Data & method”." if h["unplaceable"] else ""))

    # =============================================================================================
    st.divider()
    st.header("2 · Where they are, and whose catchment is whose")
    st.caption(f"Every selected site, with a **{radius_mi:g}-mile circle** around it — roughly the "
               "trade area a site's demographics, traffic and competitor counts are pulled for. "
               "**Where two circles overlap, both sites are being credited with the same "
               "households.** The number inside a pin is the order this operator opened it, "
               f"1 to {n_all}; pins darken as they get newer.")

    mm = m.copy()
    mm["shade"] = [_shade(int(r), n_all) for r in mm.open_rank]
    mm["disc"] = np.sqrt(mm.washes_per_year.clip(lower=1)) / 24 + 18

    HOVER = ("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
             "Opened <b>%{customdata[2]}</b> — the operator's #%{customdata[5]}<br>"
             "<b>%{customdata[3]:,.0f}</b> washes a year "
             "<span style='opacity:.7'>(%{customdata[6]:.0f} months trading)</span><br>"
             "Nearest sibling %{customdata[4]:.1f} km — <b>%{customdata[7]:.0%}</b> of this "
             "catchment shared<extra></extra>")

    def _cd(f: pd.DataFrame) -> np.ndarray:
        return np.stack([f.site, f.address, f.opened, f.washes_per_year,
                         f.nearest_km.fillna(0), f.open_rank, f.months, f.overlap_nearest],
                        axis=-1)

    centre, zoom = _view(m, float(radius_mi))
    figm = go.Figure()
    rlat, rlon = _rings(m, float(radius_mi))
    figm.add_scattermap(lat=rlat, lon=rlon, mode="lines", fill="toself",
                        fillcolor="rgba(57,135,229,0.10)",
                        line=dict(color="rgba(57,135,229,0.55)", width=1),
                        hoverinfo="skip", showlegend=False)
    # A surface-coloured disc under each pin: sites in a tight market overlap, and without a ring
    # two touching pins read as one blob.
    figm.add_scattermap(lat=mm.lat, lon=mm.lon, mode="markers", hoverinfo="skip",
                        showlegend=False, marker=dict(size=mm.disc + 5, color=SURFACE))
    # The opening-order number goes INSIDE the pin, which needs dark ink on the pale early pins and
    # light ink on the deep late ones. `textfont.color` is not array-capable on maplibre traces, so
    # the split is two traces rather than a per-point colour.
    #
    # `textfont.family` MUST be named, and must be a font the basemap's glyph server actually
    # serves. maplibre renders labels from pre-baked glyph tiles, not from CSS; ask it for a family
    # the style does not carry and the whole text layer silently draws nothing — no error, no
    # warning, just bare pins. Leaving it unset is not safe either, because `ui.style()` sets
    # `layout.font.family` to a CSS stack for every other chart and that cascades in here. Tested
    # against carto's glyph server: "Open Sans Regular" renders, "Arial Unicode MS Regular" does not.
    for dark_pin in (False, True):
        f = mm[[_is_dark(c) == dark_pin for c in mm.shade]]
        if f.empty:
            continue
        figm.add_scattermap(
            lat=f.lat, lon=f.lon, mode="markers+text",
            text=[str(int(r)) for r in f.open_rank], textposition="middle center",
            textfont=dict(size=12, family="Open Sans Regular",
                          color="#ffffff" if dark_pin else "#0b0b0b"),
            marker=dict(size=f.disc, color=list(f.shade)),
            customdata=_cd(f), hovertemplate=HOVER, showlegend=False)
    figm.update_layout(map=dict(style="carto-darkmatter" if DARK else "carto-positron",
                                center=centre, zoom=zoom))
    st.plotly_chart(style(figm, height=560, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
    st.caption("Pin size is washes a year. Zoom and pan freely — the circles are drawn in real "
               "geography, so they stay the same width in miles at every zoom. Every site's "
               "address links to Google Maps in the table below.")

    if (m.n_overlapping > 0).any():
        callout("What this shows", f"""
          <b>{h['share_overlapping']:.0%} of {h['operator']}'s sites sit inside another one's
            catchment.</b> {h['n_overlapping']} of their {h['n_sites']} sites have at least one
            sibling within {2 * radius_mi:g} miles, which is where two {radius_mi:g}-mile circles
            begin to intersect.
          <b>Where they overlap, they overlap a lot.</b> The median overlapping pair shares
            <b>{h['median_overlap']:.0%}</b> of a site's trade area, and the worst shares
            <b>{h['max_overlap']:.0%}</b>. A site-selection model fed a {radius_mi:g}-mile trade
            area counts those households once for each site.
          <b>Why this matters for a proforma.</b> Two sites handed nearly the same catchment get
            handed nearly the same demographic inputs, so the sheet cannot tell them apart — yet
            §④ finds real wash volume varies far more within a market than between markets.
            Section 6 below measures what that costs in actual washes.
        """)
    else:
        st.info(f"No two of these sites are within {2 * radius_mi:g} miles of each other, so no "
                f"{radius_mi:g}-mile catchments overlap. Widen the radius, or pick “All markets”.")

    # =============================================================================================
    st.divider()
    st.header("3 · Every site, one row each")
    # Column order is priority order, because the table is wider than the page and scrolls: the
    # opening date, the volume and the catchment overlap have to be visible before the reader has
    # to drag anything. ZIP and market are in the CSV, not on screen.
    show = m[["open_rank", "site", "address", "state", "opened", "months", "washes_per_year",
              "overlap_nearest", "nearest_site", "nearest_km", "washes", "mem_share",
              "asp"]].copy()
    show["address"] = [f"<a href='{u}' target='_blank' rel='noopener'>{a}</a>"
                       for a, u in zip(show.address, m.maps_url)]
    show.columns = ["Opened #", "Site", "Address (opens Google Maps)", "State", "Opened",
                    "Months trading", "Washes a year", "Catchment shared", "Nearest sibling",
                    "Nearest (km)", "Washes to date", "Membership share", "Revenue per wash"]
    show.index = range(1, len(show) + 1)
    html_table(show, fmt={"Opened #": "{:,.0f}", "Months trading": "{:,.0f}",
                          "Washes to date": "{:,.0f}", "Washes a year": "{:,.0f}",
                          "Membership share": "{:.0%}", "Revenue per wash": "${:,.2f}",
                          "Nearest (km)": "{:,.2f}", "Catchment shared": "{:.0%}"})
    st.caption("“Washes a year” is the site's own rate — total washes ÷ months trading × 12 — so a "
               "site opened last year is comparable with one opened in 2020. “Catchment shared” is "
               f"the share of this site's {radius_mi:g}-mile circle that its **nearest** sibling "
               "also covers; it is pairwise, not the union of every sibling.")
    st.download_button("Download these sites (CSV)", m.to_csv(index=False),
                       f"operator_sites_{pick}.csv", "text/csv", key=f"dl_sites_{pick}")

    # =============================================================================================
    st.divider()
    st.header("4 · Month on month")
    mo = _months(pick, keys)
    if mo.empty:
        st.info("No monthly wash data for this selection.")
    else:
        tot = mo.groupby("date").washes.sum().reset_index()
        figt = go.Figure()
        figt.add_scatter(x=tot.date, y=tot.washes, mode="lines", line=dict(color=S1, width=2.5),
                         fill="tozeroy", fillcolor="rgba(57,135,229,0.13)",
                         hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> washes<extra></extra>")
        # Openings are grouped by month before they are drawn: an operator that opened five sites in
        # one month would otherwise stack five labels on one pixel column. Past a dozen opening
        # months the rules stop being a reading aid and become a picket fence, so they are dropped
        # and the caption says so rather than silently thinning them.
        opens = m.dropna(subset=["opened_ym"])
        groups = list(opens.groupby("opened_ym"))
        if len(groups) <= 14:
            for ym, grp in groups:
                yr, mth = divmod(int(ym), 12)
                when = pd.Timestamp(year=yr, month=mth + 1, day=1)
                if not (tot.date.min() <= when <= tot.date.max()):
                    continue
                ranks = sorted(int(r) for r in grp.open_rank)
                tag = (f"{ranks[0]}–{ranks[-1]}"
                       if len(ranks) > 2 and ranks[-1] - ranks[0] == len(ranks) - 1
                       else ",".join(str(r) for r in ranks))
                figt.add_vline(x=when, line=dict(color=MUTED, width=1, dash="dot"))
                figt.add_annotation(x=when, y=1.0, yref="paper", text=tag, showarrow=False,
                                    yanchor="bottom", font=dict(size=10, color=MUTED))
        st.plotly_chart(style(figt, height=330, yaxis_title="Washes a month", xaxis_title=None,
                              showlegend=False, margin=dict(l=70, r=20, t=30, b=40)),
                        width="stretch")
        st.caption("Total washes across the selected sites." + (
            " Each dotted rule is an opening, numbered as on the map."
            if len(groups) <= 14 else
            f" Opening rules are omitted — {len(groups)} separate opening months would fence the "
            "chart. Zoom to a single market to see them."))

        # One row per site scales to 79 sites; 79 overlaid lines do not.
        piv = (mo.merge(m[["site_key", "site", "open_rank"]], on="site_key")
                 .pivot_table(index=["open_rank", "site"], columns="date", values="washes",
                              aggfunc="sum").sort_index())
        st.markdown("**Each site, month by month.** One row per site, in the order they opened. "
                    "Colour is washes in that month — darker is busier; a blank cell is a month "
                    "the site was not trading.")
        figh = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns, y=[s for _, s in piv.index],
            colorscale=VOLUME_SCALE, hoverongaps=False,
            colorbar=dict(title="Washes<br>a month", thickness=12, tickformat=",.0f"),
            hovertemplate="<b>%{y}</b><br>%{x|%b %Y}<br><b>%{z:,.0f}</b> washes<extra></extra>"))
        st.plotly_chart(style(figh, height=max(260, 22 * len(piv) + 110), xaxis_title=None,
                              yaxis=dict(autorange="reversed"),
                              margin=dict(l=175, r=20, t=20, b=40)), width="stretch")

    # =============================================================================================
    st.divider()
    st.header("5 · Year on year")
    y = _years(pick, keys)
    if y.empty:
        st.info("No complete calendar year yet — every selected site opened after "
                f"{cd.last_complete_year()}.")
    else:
        through = y.attrs.get("through", cd.last_complete_year())
        y = y.merge(m[["site_key", "site", "open_rank", "opened_ym"]], on="site_key", how="left")
        years = sorted(int(v) for v in y.year.unique())
        st.caption(f"Years run to **{through}** — the panel only holds part of {through + 1}, and "
                   "half a year plotted against full ones reads as a collapse that did not happen. "
                   "A site's **own** opening year is genuinely partial and stays in, marked.")

        tot_y = y.groupby("year").agg(washes=("washes", "sum"),
                                      sites=("site_key", "nunique")).reset_index()
        figy = go.Figure(go.Bar(
            x=tot_y.year, y=tot_y.washes,
            marker=dict(color=S1, line=dict(color=SURFACE, width=1)), customdata=tot_y.sites,
            hovertemplate="<b>%{x}</b><br><b>%{y:,.0f}</b> washes<br>"
                          "<span style='opacity:.7'>%{customdata:.0f} sites trading</span>"
                          "<extra></extra>"))
        st.plotly_chart(style(figy, height=300, yaxis_title="Washes a year", bargap=0.35,
                              xaxis=dict(dtick=1), showlegend=False,
                              margin=dict(l=70, r=20, t=25, b=40)), width="stretch")
        st.caption("The whole selection. It grows partly because each site grows and partly "
                   "because there are more of them — the per-site view below separates the two.")

        order = m.sort_values("open_rank")
        if len(order) <= PANEL_LIMIT:
            # `shared_yaxes=True` only links panels WITHIN a row — across rows every panel would
            # still autoscale and a 39k site would draw a bar as tall as a 235k one. The common
            # range is set explicitly below, which is what makes the caption's promise true.
            ymax = float(y.washes.max()) * 1.08
            cols = min(4, len(order))
            rows = math.ceil(len(order) / cols)
            figp = make_subplots(rows=rows, cols=cols, shared_yaxes=True,
                                 subplot_titles=[f"{int(r.open_rank)} · {r.site}"
                                                 for _, r in order.iterrows()],
                                 vertical_spacing=0.30 / rows, horizontal_spacing=0.03)
            for i, (_, r) in enumerate(order.iterrows()):
                sub = y[y.site_key == r.site_key].set_index("year").reindex(years)
                base = _shade(int(r.open_rank), n_all)
                part = sub.part_year.fillna(False).astype(bool)
                figp.add_bar(
                    x=years, y=sub.washes,
                    marker=dict(color=[base if not p else "rgba(0,0,0,0)" for p in part],
                                line=dict(color=base, width=1.4)),
                    customdata=np.stack([sub.months.fillna(0), sub.mem.fillna(0),
                                         sub.ret.fillna(0)], axis=-1),
                    hovertemplate=f"<b>{r.site}</b> · %{{x}}<br><b>%{{y:,.0f}}</b> washes<br>"
                                  "%{customdata[1]:,.0f} member · %{customdata[2]:,.0f} drive-up"
                                  "<br><span style='opacity:.7'>%{customdata[0]:.0f} months traded"
                                  "</span><extra></extra>",
                    showlegend=False, row=i // cols + 1, col=i % cols + 1)
            for a in figp.layout.annotations:
                a.font = dict(size=11, color=INK2)
            figp.update_xaxes(dtick=1, tickangle=0, tickfont=dict(size=10, color=MUTED),
                              gridcolor="rgba(0,0,0,0)", linecolor=GRID)
            figp.update_yaxes(gridcolor=GRID, tickfont=dict(size=10, color=MUTED),
                              range=[0, ymax])
            st.plotly_chart(style(figp, height=205 * rows + 80, bargap=0.25,
                                  margin=dict(l=60, r=15, t=40, b=35)), width="stretch")
            st.caption("One panel per site, in the order the operator opened them, **all on the "
                       "same vertical scale**. An outlined bar is a part year.")
        else:
            st.caption(f"{len(order)} sites is past the point where one panel each can be read — "
                       "the table below carries every number, and the month-by-month heatmap above "
                       "shows the same thing at higher resolution. Zoom to a single market for the "
                       "per-site panels.")

        piv_y = (y.pivot_table(index="site_key", columns="year", values="washes", aggfunc="sum")
                  .reindex(order.site_key.values))
        piv_y.index = order.site.values
        piv_y.columns = [str(int(c)) for c in piv_y.columns]
        html_table(piv_y, fmt={c: "{:,.0f}" for c in piv_y.columns}, index_label="Site")
        st.caption("Washes per calendar year, one row per site, in opening order. Blank = the site "
                   "was not trading that year.")
        st.download_button("Download year-by-year (CSV)", piv_y.to_csv(),
                           f"operator_years_{pick}.csv", "text/csv", key=f"dl_years_{pick}")

    # =============================================================================================
    st.divider()
    st.header("6 · What happened when they opened the next one")
    eff = _effect(pick, keys, float(max_km), float(radius_mi))
    all_eff = _effect_all(float(max_km), 3)

    if eff.empty:
        st.info("No opening in this selection has a settled neighbour to compare against — the "
                "sites here either opened within a year of each other, or the panel does not "
                "cover a full six months either side. Try “All markets”.")
    else:
        st.caption("For each new site: what the neighbours **that were already settled** (a year "
                   "or more of trading behind them) washed in the six months after it opened, "
                   "against the six months before. The grey bar is the same comparison for this "
                   "operator's sites **outside** the selection over the very same months — "
                   "company-wide and seasonal moves show up there, so what is left over is what "
                   "is specific to this place.")
        e = eff.copy()
        lbl = [f"{int(r.open_rank)} · {r.site}" for _, r in e.iterrows()]
        figE = go.Figure()
        figE.add_bar(x=lbl, y=e.control_change * 100, name="operator's sites elsewhere",
                     marker=dict(color=GRID, line=dict(color=MUTED, width=1)),
                     hovertemplate="elsewhere: <b>%{y:+.1f}%</b><extra></extra>")
        figE.add_bar(x=lbl, y=e.incumbent_change * 100, name="neighbours here",
                     marker=dict(color=[S2 if v < 0 else S3 for v in e.incumbent_change],
                                 line=dict(color=SURFACE, width=1)),
                     customdata=np.stack([e.n_incumbents, e.nearest_km, e.opened,
                                          e.excess * 100], axis=-1),
                     hovertemplate="<b>%{x}</b> opened %{customdata[2]}<br>"
                                   "neighbours here: <b>%{y:+.1f}%</b><br>"
                                   "after taking off elsewhere: <b>%{customdata[3]:+.1f}pp</b><br>"
                                   "<span style='opacity:.7'>%{customdata[0]:.0f} settled "
                                   "neighbours · nearest %{customdata[1]:.1f} km</span>"
                                   "<extra></extra>")
        figE.add_hline(y=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(style(figE, height=380, barmode="group", bargap=0.3,
                              yaxis_title="Change in washes, 6 months after vs before (%)",
                              yaxis=dict(ticksuffix="%"),
                              margin=dict(l=70, r=25, t=78, b=95),
                              legend=dict(orientation="h", y=1.13, x=0)), width="stretch")

        et = e[["open_rank", "site", "opened", "n_incumbents", "nearest_km", "incumbent_change",
                "control_change", "excess"]].copy()
        et.columns = ["Opened #", "New site", "Opened", "Settled neighbours", "Nearest (km)",
                      "Neighbours here", "Operator elsewhere", "Difference"]
        et.index = range(1, len(et) + 1)
        html_table(et, fmt={"Opened #": "{:,.0f}", "Settled neighbours": "{:,.0f}",
                            "Nearest (km)": "{:,.2f}", "Neighbours here": "{:+.1%}",
                            "Operator elsewhere": "{:+.1%}", "Difference": "{:+.1%}"})

    if not all_eff.empty:
        x = all_eff.excess.dropna()
        med, neg = float(x.median()), float((x < 0).mean())
        raw, ctl = float(all_eff.incumbent_change.median()), float(all_eff.control_change.median())
        this = ""
        if not eff.empty and eff.excess.notna().any():
            this = (f"<b>This selection</b> comes in at <b>{eff.excess.median() * 100:+.1f}pp</b>, "
                    f"against an estate median of <b>{med * 100:+.1f}pp</b>.")
        callout("What this shows", f"""
          <b>Opening next to yourself costs you something, and it is small.</b> Across
            <b>{len(x)}</b> openings in <b>{all_eff.cluster_id.nunique()}</b> clustered markets
            estate-wide, the settled neighbours give up a median <b>{med * 100:+.1f}pp</b> of
            washes relative to the same operator's sites elsewhere, and <b>{neg:.0%}</b> of
            openings land on the losing side.
          <b>Most of the raw drop is not the neighbour's fault.</b> The neighbours' own
            before/after change is <b>{raw * 100:+.1f}%</b> and the operator's other sites moved
            <b>{ctl * 100:+.1f}%</b> over the identical months. Skip that control and you charge
            the whole seasonal and company-wide swing to the new site.
          {this}
          <b>The catchment overlap is far larger than the wash loss.</b> Sites routinely share
            30–90% of a {radius_mi:g}-mile circle, yet the measured hit is a few points. Either the
            real trade area is wider than {radius_mi:g} miles, or a second site brings enough new
            demand to cover most of what it takes — both readings argue against sizing a site on
            its circle alone.
          <b>How much to lean on this.</b> It is descriptive. Openings inside one market are only
            months apart, so their six-month windows overlap and the events are not independent.
            §⑤ Competition is where entry is estimated properly.
        """, S3)

    # =============================================================================================
    with st.expander("The estate-wide picture — every operator's clustered markets"):
        cl = _clusters(float(max_km), 3)
        if cl.empty:
            st.info("No operator has three or more sites that close together at this setting.")
        else:
            t = cl.head(20)[["place", "n_sites", "diameter_km", "nearest_km", "median_km",
                             "first_open", "last_open", "build_out_months",
                             "washes_per_year"]].copy()
            for c in ("diameter_km", "nearest_km", "median_km"):
                t[c] = t[c] / KM_PER_MILE
            t.columns = ["Operator · state", "Sites", "Widest apart (mi)", "Closest pair (mi)",
                         "Typical pair (mi)", "First opened", "Last opened",
                         "Built out over (months)", "Washes a year"]
            t.index = range(1, len(t) + 1)
            html_table(t, fmt={"Sites": "{:,.0f}", "Widest apart (mi)": "{:,.1f}",
                               "Closest pair (mi)": "{:,.1f}", "Typical pair (mi)": "{:,.1f}",
                               "Built out over (months)": "{:,.0f}", "Washes a year": "{:,.0f}"})
            st.caption(f"Biggest 20 of {len(cl)} clustered markets across all operators, at "
                       f"{max_km} km — three or more sites. This table is the only thing on the "
                       f"page that is not about {h['operator']}.")

    with st.expander("Data & method"):
        cq = _coords()
        st.markdown(f"""
**Input.** `conclusion/data/historical_data_5yrs_monthly.csv` — the monthly wash panel,
{cq['n_sites']:,} sites, 2020 to {cd.last_complete_year() + 1}. It is byte-identical to
`proforma/data/panel/main-data-v2-stitched.csv`; this section reads it through `conclusion/data/`
only because every other section's data module does. **The site key is `client_id` + `site_id`** —
`site_id` alone is a within-brand index and collides across operators.

**The operator dropdown** holds the {len(ops)} operators with two or more placeable sites. An
operator's **markets** are its sites grouped by complete-linkage on great-circle distance at the
slider's setting, so the slider caps a market's **diameter** — every site is within it of *every*
other, not merely of its nearest neighbour. Single linkage was tried and chains: the Rio Grande
Valley joins into one 120 km "market" through a string of 15 km hops. Distances are straight-line,
**not drive time**, so two sites either side of a river read closer than they are.

**The trade-area circle** is drawn at the slider's radius, default 3 miles, because that is roughly
the radius a site's demographics, traffic and competitor counts are pulled for. "Catchment shared"
is the exact circle-circle lens area over circle area for a site and its **nearest** sibling — a
pairwise number, not the union across every sibling, which would need a polygon union (`shapely` is
not in the `sonnys` env). It therefore **understates** total overlap wherever three or more circles
pile up.

**The basemap is CARTO/OpenStreetMap, not Google.** Google's tiles cannot be used as a raster tile
source outside their own Maps JavaScript API, so a Google basemap is not something a Plotly chart
can legally draw. Instead every address in the sitewise table links to **that exact coordinate in
Google Maps**, which needs no API key. The pins, circles and distances here are all computed from
the panel's own lat/lon.

**A coordinate defect, stated rather than hidden.** {cq['n_placeholder']} sites across
{cq['n_placeholder_points']} coordinate points carry a **placeholder** latitude/longitude — one
coordinate shared by several sites of the same operator whose street addresses are all different.
BlueWave stamps 21 Houston-area sites on a single point; Buckeye stamps 10 sites spread across six
Ohio towns on another. Their wash data is real; their location is not. All {cq['n_placeholder']}
are dropped — they carry {cq['share_of_washes']:.1%} of all washes — and each operator's header
says how many of **its** sites that cost. A further {cq['n_outside_box']} sites fall outside the
continental-US box. {cq['n_usable']:,} sites remain. Sites sharing a coordinate **and** a street
address are a different thing — a second tunnel at one address, or an operator handoff — and are
kept, at a true distance of 0; there are {cq['n_co_located']}.

**Calendar years stop at {cd.last_complete_year()}**, the last year the panel holds all twelve
months of. A site's own opening year is genuinely partial and stays in, flagged. The monthly view
has no such cut — a monthly axis has no part-year to hide.

**Section 6's guards.** An "incumbent" needs 12 months of trading before the new opening, because
§⓪ measures a new wash reaching ~98% of its eventual volume only by year 2 — a younger neighbour is
still climbing and would show growth that has nothing to do with anybody. Both the neighbour and
control sums are **balanced**: a site counts only if it traded in the before window *and* the after
window, and the control is held to the same settled test. Without those guards the control pool
reads **+13%** growth that is nothing but the operator's own new sites elsewhere ramping, and that
would be charged to the new neighbour as cannibalization.

**Known gaps.** Straight-line distance is not drive time, and a 3-mile circle is not a real
catchment — it has no roads, rivers or county lines in it. The panel is Sonny's customers only, so
an operator's "market" says nothing about who else washes cars in that town. `operational_start` is
a month stamp; where it is missing the first month that washed a car is used instead.

**Tables are rendered as raw HTML.** `st.dataframe` / `st.table` segfault the Streamlit server on
the second script run in this environment (pyarrow 25.0.0 + pandas 3.0.2 + streamlit 1.58.0); the
proper fix is an environment pin.
""")
