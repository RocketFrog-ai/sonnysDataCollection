"""
Section — Operator clusters. One operator, several sites, one town.

§⓪ shows the estate from 30,000 feet. This one drops to street level: pick a place where a single
operator has put three or more washes inside a 25 km circle, and look at what it built — how far
apart the sites sit, the order they opened in, what each has washed year by year, and what the
sites already there did when the next one arrived.

The maths is in `cluster_data.py` and is Streamlit-free, so the notebook reports the same numbers.

Chart choices, since a cluster can hold nine sites and nine is past any honest categorical palette:

  • **opening order is the one colour encoding**, on a single blue ramp, light = opened first. It is
    used identically on the map and on the per-site panels, so a colour means the same thing twice.
  • the per-site history is **small multiples on a shared y-axis**, not nine overlaid lines. Nine
    lines would need nine hues; shared axes also stop each panel from auto-scaling itself into
    looking like every site is the same size.
  • every chart here has a table underneath it carrying the same numbers — which is also the
    required relief for the pale end of the ramp on the light surface.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import cluster_data as cd
from ui import (BORDER, DARK, GOOD, GRID, INK, INK2, MUTED, S1, S2, S3, SURFACE, callout,
                html_table, style)

# Sequential ramp for OPENING ORDER — pale = the operator's first site here, deep = its newest.
# One hue, light→dark, and the same direction in both themes. The deep end stops at a saturated
# blue rather than a navy so the newest sites stay legible on the dark surface (§⓪'s orange ramp
# stops short of black for the same reason). Lightness is strictly monotonic; every step clears
# 3:1 on the dark surface, and on the light surface the two palest steps are given a mark outline
# instead, which is why every marker and bar below carries one.
ORDER_RAMP = ["#dbe9fc", "#aecbf3", "#7ba9e8", "#4a86dd", "#2f6fd0"]

KM_PER_MILE = cd.KM_PER_MILE


# =================================================================================================
# Cached wrappers — the module memoises too, but Streamlit's cache also survives a widget change.
# =================================================================================================

@st.cache_data(show_spinner=False)
def _clusters(max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.build(max_km, min_sites)[0]


@st.cache_data(show_spinner=False)
def _headline(max_km: float, min_sites: int) -> dict:
    return cd.headline(max_km, min_sites)


@st.cache_data(show_spinner=False)
def _sites(cluster_id: str, max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.cluster_sites(cluster_id, max_km, min_sites)


@st.cache_data(show_spinner=False)
def _distances(cluster_id: str, max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.distance_table(cluster_id, max_km, min_sites)


@st.cache_data(show_spinner=False)
def _years(cluster_id: str, max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.site_years(cluster_id, max_km, min_sites)


@st.cache_data(show_spinner=False)
def _months(cluster_id: str, max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.site_months(cluster_id, max_km, min_sites)


@st.cache_data(show_spinner=False)
def _effect(cluster_id: str, max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.opening_effect(cluster_id, max_km, min_sites)


@st.cache_data(show_spinner=False)
def _effect_all(max_km: float, min_sites: int) -> pd.DataFrame:
    return cd.opening_effect_all(max_km, min_sites)


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

    Relative luminance, the same quantity the contrast ratio is built from, rather than a guess at
    which ramp steps "look dark". 0.35 puts the flip between the ramp's middle and fourth steps.
    """
    srgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.35


def _view(m: pd.DataFrame) -> tuple[dict, float]:
    """Map centre and zoom that fit every site in the cluster with a margin.

    Derived from the cluster's own bounding box rather than fixed, because the whole point is that
    the map is zoomed to *this* place. The floor on the span keeps a cluster whose sites share one
    address from zooming to street furniture.
    """
    lat_span = max(float(m.lat.max() - m.lat.min()), 0.012)
    lon_span = max(float(m.lon.max() - m.lon.min()), 0.012)
    zoom = min(math.log2(360.0 / (lon_span * 1.7)), math.log2(180.0 / (lat_span * 1.7)))
    centre = dict(lat=float((m.lat.max() + m.lat.min()) / 2),
                  lon=float((m.lon.max() + m.lon.min()) / 2))
    return centre, float(np.clip(zoom, 3.0, 14.0))


def _km(v: float) -> str:
    return f"{v:,.1f} km ({v / KM_PER_MILE:,.1f} mi)"


# =================================================================================================
# Render
# =================================================================================================

def render() -> None:
    st.markdown("<div class='kicker'>Evidence pack · ⑥</div>", unsafe_allow_html=True)
    st.title("Operator clusters")
    st.markdown("§⓪ looks at the estate from above. This looks at it from the street: **one "
                "operator, several washes, one town.** Where an operator has built a group of "
                "sites close together, we can see how far apart it put them, the order it opened "
                "them in, what each one washes, and what the sites already there did when the "
                "next one turned up.")

    c1, c2 = st.columns([2, 1])
    with c1:
        max_km = st.slider("How close counts as “one place” (km between the two furthest sites)",
                           5, 50, 25, 1,
                           help="Complete linkage: every site in a cluster is within this "
                                "distance of every other one, not just of its nearest neighbour.")
    with c2:
        min_sites = st.slider("Fewest sites to call it a cluster", 3, 8, 3, 1)

    clusters = _clusters(float(max_km), int(min_sites))
    if clusters.empty:
        st.warning("No operator has that many sites that close together. Widen either slider.")
        return
    h = _headline(float(max_km), int(min_sites))
    cq = _coords()

    # =============================================================================================
    st.divider()
    st.header("1 · How much of the business is built this way")
    st.caption(f"Every group of **{min_sites}+ sites owned by one operator** whose furthest two "
               f"sites are within **{max_km} km ({max_km / KM_PER_MILE:.0f} miles)** of each "
               f"other. Distances are straight-line, not drive time.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters", f"{h['n_clusters']:,}", f"{h['n_operators']} operators",
              delta_color="off")
    m2.metric("Sites inside one", f"{h['n_sites']:,}",
              f"{h['share_of_sites']:.0%} of sites we can place", delta_color="off")
    m3.metric("Share of all washes", f"{h['share_of_washes']:.0%}",
              "done inside a cluster", delta_color="off")
    m4.metric("Typical gap to the nearest sibling", f"{h['median_gap']:.1f} km",
              f"{h['median_gap'] / KM_PER_MILE:.1f} miles", delta_color="off")

    top = clusters.head(15).copy()
    tbl = top[["place", "n_sites", "diameter_km", "nearest_km", "median_km", "first_open",
               "last_open", "build_out_months", "washes_per_year"]].copy()
    tbl["diameter_km"] = tbl.diameter_km / KM_PER_MILE
    tbl["nearest_km"] = tbl.nearest_km / KM_PER_MILE
    tbl["median_km"] = tbl.median_km / KM_PER_MILE
    tbl.columns = ["Operator · state", "Sites", "Widest apart (mi)", "Closest pair (mi)",
                   "Typical pair (mi)", "First opened", "Last opened", "Built out over (months)",
                   "Washes a year"]
    tbl.index = range(1, len(tbl) + 1)
    html_table(tbl, fmt={"Sites": "{:,.0f}", "Widest apart (mi)": "{:,.1f}",
                         "Closest pair (mi)": "{:,.1f}", "Typical pair (mi)": "{:,.1f}",
                         "Built out over (months)": "{:,.0f}", "Washes a year": "{:,.0f}"})
    st.caption(f"Biggest {len(tbl)} of {h['n_clusters']} clusters, by site count.")

    callout("What this shows", f"""
      <b>Clustering is normal, not exceptional.</b> <b>{h['n_sites']:,}</b> sites —
        <b>{h['share_of_sites']:.0%}</b> of every site we can place, and
        <b>{h['share_of_washes']:.0%}</b> of all the washing — sit inside a group of
        {min_sites} or more owned by the same operator within {max_km} km.
      <b>They are built close and built fast.</b> The typical site's nearest sibling is
        <b>{h['median_gap']:.1f} km ({h['median_gap'] / KM_PER_MILE:.1f} miles)</b> away, and the
        typical cluster went from its first opening to its last in
        <b>{h['median_build_out']:.0f} months</b>. This is a land-grab pattern — take a town, then
        fill it — not a site-by-site one.
      <b>The tightest of all is {h['tightest']}</b>, whose two closest sites are
        <b>{h['tightest_gap']:.2f} km</b> apart. At that range they are not two trade areas.
    """)

    # =============================================================================================
    st.divider()
    st.header("2 · Inside one place")

    # `.get(c, c)` rather than a lookup that raises: the widget holds a `cluster_id`, and the
    # option list is rebuilt every time a slider moves.
    labels = dict(zip(clusters.cluster_id, clusters.label))
    pick = st.selectbox("Pick a cluster", list(clusters.cluster_id),
                        format_func=lambda c: labels.get(c, c))
    row = clusters.set_index("cluster_id").loc[pick]
    m = _sites(pick, float(max_km), int(min_sites))
    n = len(m)

    st.markdown(f"**{row.operator}** · {row.state} · **{n} sites** · first opened "
                f"{row.first_open}, last {row.last_open} · anchor site *{row.anchor}*, "
                f"{row.anchor_address}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sites here", f"{n}")
    k2.metric("Widest apart", f"{row.diameter_km:,.1f} km",
              f"{row.diameter_km / KM_PER_MILE:,.1f} miles", delta_color="off")
    k3.metric("Closest pair", f"{row.nearest_km:,.1f} km",
              f"{row.nearest_km / KM_PER_MILE:,.1f} miles", delta_color="off")
    k4.metric("Built out over", f"{row.build_out_months:,.0f} mo",
              f"{row.washes_per_year:,.0f} washes a year", delta_color="off")

    # --- the map --------------------------------------------------------------------------------
    centre, zoom = _view(m)
    mm = m.copy()
    mm["shade"] = [_shade(int(r), n) for r in mm.open_rank]
    # Area on sqrt of volume so a 220k site does not swamp an 88k one; the floor keeps the
    # opening-order number inside the disc readable at every size.
    mm["disc"] = np.sqrt(mm.washes_per_year.clip(lower=1)) / 24 + 18

    HOVER = ("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
             "Opened <b>%{customdata[2]}</b> — the operator's #%{customdata[5]} here<br>"
             "<b>%{customdata[3]:,.0f}</b> washes a year "
             "<span style='opacity:.7'>(%{customdata[6]:.0f} months of trading)</span>"
             "<br>Nearest sibling %{customdata[4]:.1f} km<extra></extra>")

    def _cd(f: pd.DataFrame) -> np.ndarray:
        return np.stack([f.site, f.address, f.opened, f.washes_per_year, f.nearest_km,
                         f.open_rank, f.months], axis=-1)

    figm = go.Figure()
    # A surface-coloured disc under each marker: the sites in a tight cluster overlap, and without
    # a ring two touching discs read as one blob.
    figm.add_scattermap(lat=mm.lat, lon=mm.lon, mode="markers", hoverinfo="skip",
                        showlegend=False, marker=dict(size=mm.disc + 5, color=SURFACE))
    # The opening-order number is drawn INSIDE the disc, which needs dark ink on the pale early
    # discs and light ink on the deep late ones. `textfont.color` is not array-capable on maplibre
    # traces, so the split is two traces rather than a per-point colour.
    #
    # `textfont.family` MUST be named, and must be a font the basemap's glyph server actually
    # serves. maplibre renders map labels from pre-baked glyph tiles, not from CSS; ask it for a
    # family the style does not carry and the entire text layer silently draws nothing — no error,
    # no warning, just bare discs. Leaving it unset is not safe either, because `ui.style()` sets
    # `layout.font.family` to a CSS stack for every other chart in the app and that cascades in
    # here. Tested against carto's glyph server: "Open Sans Regular" renders, "Arial Unicode MS
    # Regular" does not.
    for dark_disc in (False, True):
        f = mm[[_is_dark(c) for c in mm.shade] if dark_disc
               else [not _is_dark(c) for c in mm.shade]]
        if f.empty:
            continue
        figm.add_scattermap(
            lat=f.lat, lon=f.lon, mode="markers+text",
            text=[str(int(r)) for r in f.open_rank], textposition="middle center",
            textfont=dict(size=12, family="Open Sans Regular",
                          color="#ffffff" if dark_disc else "#0b0b0b"),
            marker=dict(size=f.disc, color=list(f.shade)),
            customdata=_cd(f), hovertemplate=HOVER, showlegend=False)
    figm.update_layout(map=dict(style="carto-darkmatter" if DARK else "carto-positron",
                                center=centre, zoom=zoom))
    st.plotly_chart(style(figm, height=520, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
    st.caption("The number inside each site is the order the operator opened it here — **1 is the "
               "first**, and the discs darken as they get newer. Disc size is washes a year. "
               "Zoom and pan freely; the view is fitted to this cluster.")

    # --- distances ------------------------------------------------------------------------------
    st.subheader("How far apart are they?")
    d = _distances(pick, float(max_km), int(min_sites))
    near = m[["open_rank", "site", "address", "opened", "nearest_km"]].copy()
    # Positional, not by label: two sites in one cluster can legitimately carry the same short name
    # once the operator's brand is stripped off the front, and a label lookup would then return a
    # frame instead of a column and blow up.
    dm = d.to_numpy(copy=True)
    np.fill_diagonal(dm, np.inf)
    near["nearest_site"] = ([m.site.iloc[int(i)] for i in dm.argmin(axis=1)] if n > 1
                            else ["—"] * n)
    near["miles"] = near.nearest_km / KM_PER_MILE
    near = near[["open_rank", "site", "address", "opened", "nearest_site", "nearest_km", "miles"]]
    near.columns = ["Opened #", "Site", "Address", "Opened", "Nearest sibling", "km", "miles"]
    near.index = range(1, len(near) + 1)
    html_table(near, fmt={"Opened #": "{:,.0f}", "km": "{:,.2f}", "miles": "{:,.2f}"})

    with st.expander("Every pair, in km"):
        dd = d.round(2)
        html_table(dd, fmt={c: "{:,.2f}" for c in dd.columns}, index_label="km")
        st.download_button("Download this cluster's distances (CSV)", d.to_csv(),
                           f"cluster_distances_{pick.replace('::', '_')}.csv", "text/csv",
                           key=f"dl_dist_{pick}")

    # =============================================================================================
    st.divider()
    st.header("3 · Every site in this place")
    show = m[["open_rank", "site", "address", "state", "postal_code", "opened", "months",
              "washes", "washes_per_year", "mem_share", "asp", "nearest_km"]].copy()
    show.columns = ["Opened #", "Site", "Address", "State", "ZIP", "Opened", "Months trading",
                    "Washes to date", "Washes a year", "Membership share", "Revenue per wash",
                    "Nearest sibling (km)"]
    show.index = range(1, len(show) + 1)
    html_table(show, fmt={"Opened #": "{:,.0f}", "Months trading": "{:,.0f}",
                          "Washes to date": "{:,.0f}", "Washes a year": "{:,.0f}",
                          "Membership share": "{:.0%}", "Revenue per wash": "${:,.2f}",
                          "Nearest sibling (km)": "{:,.2f}"})
    st.caption("“Washes a year” is the site's own rate — total washes ÷ months trading × 12 — so a "
               "site opened last year is comparable with one opened in 2020.")
    st.download_button("Download these sites (CSV)", m.to_csv(index=False),
                       f"cluster_sites_{pick.replace('::', '_')}.csv", "text/csv",
                       key=f"dl_sites_{pick}")

    # =============================================================================================
    st.divider()
    st.header("4 · Year by year, site by site")
    y = _years(pick, float(max_km), int(min_sites))
    through = y.attrs.get("through", cd.last_complete_year())
    if y.empty:
        st.info(f"No complete calendar year yet — every site here opened after {through}.")
    else:
        st.caption(f"Washes per calendar year, one panel per site, **in the order the operator "
                   f"opened them**. All panels share the same vertical scale, so panel height is "
                   f"comparable across sites. Years run to **{through}** — the panel only holds "
                   f"part of {through + 1}, and half a year plotted against full ones reads as a "
                   f"collapse that did not happen. A paler, outlined bar is a **part year**: "
                   f"usually the site's opening year.")

        order = m.sort_values("open_rank")
        cols = min(4, len(order))
        rows = math.ceil(len(order) / cols)
        # `shared_yaxes=True` only links the panels WITHIN a row — across rows every panel would
        # still autoscale itself, and a 39k site would draw a bar exactly as tall as a 235k one.
        # The common range is therefore set explicitly below, which is what actually makes the
        # caption's promise true.
        ymax = float(y.washes.max()) * 1.08
        figy = make_subplots(rows=rows, cols=cols, shared_yaxes=True,
                             subplot_titles=[f"{int(r.open_rank)} · {r.site}"
                                             for _, r in order.iterrows()],
                             vertical_spacing=0.30 / rows, horizontal_spacing=0.03)
        years = sorted(y.year.unique())
        for i, (_, r) in enumerate(order.iterrows()):
            sub = y[y.site_key == r.site_key].set_index("year").reindex(years)
            base = _shade(int(r.open_rank), n)
            part = sub.part_year.fillna(False).astype(bool)
            figy.add_bar(
                x=years, y=sub.washes,
                marker=dict(color=[base if not p else "rgba(0,0,0,0)" for p in part],
                            line=dict(color=base, width=1.4)),
                customdata=np.stack([sub.months.fillna(0), sub.mem.fillna(0),
                                     sub.ret.fillna(0)], axis=-1),
                hovertemplate=f"<b>{r.site}</b> · %{{x}}<br><b>%{{y:,.0f}}</b> washes<br>"
                              "%{customdata[1]:,.0f} member · %{customdata[2]:,.0f} drive-up<br>"
                              "<span style='opacity:.7'>%{customdata[0]:.0f} months traded"
                              "</span><extra></extra>",
                showlegend=False, row=i // cols + 1, col=i % cols + 1)
        for a in figy.layout.annotations:
            a.font = dict(size=11, color=INK2)
        figy.update_xaxes(dtick=1, tickangle=0, tickfont=dict(size=10, color=MUTED),
                          gridcolor="rgba(0,0,0,0)", linecolor=GRID)
        figy.update_yaxes(gridcolor=GRID, tickfont=dict(size=10, color=MUTED),
                          range=[0, ymax])
        st.plotly_chart(style(figy, height=205 * rows + 80, bargap=0.25,
                              margin=dict(l=60, r=15, t=40, b=35)), width="stretch")

        # Pivot on `site_key`, then relabel — `site` is a display name and is not guaranteed
        # unique inside a cluster, which would make `reindex` raise.
        piv = y.pivot_table(index="site_key", columns="year", values="washes",
                            aggfunc="sum").reindex(order.site_key.values)
        piv.index = order.site.values
        piv.columns = [str(int(c)) for c in piv.columns]
        html_table(piv, fmt={c: "{:,.0f}" for c in piv.columns}, index_label="Site")
        st.caption("The same numbers as the panels above. Blank = the site was not trading yet.")

    # --- the cluster as one thing -----------------------------------------------------------
    st.subheader("The place as a whole, month by month")
    mo = _months(pick, float(max_km), int(min_sites))
    tot = mo.groupby("date").washes.sum().reset_index()
    figt = go.Figure()
    figt.add_scatter(x=tot.date, y=tot.washes, mode="lines", name="every site here",
                     line=dict(color=S1, width=2.5), fill="tozeroy",
                     fillcolor="rgba(57,135,229,0.13)",
                     hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> washes<extra></extra>")
    # Openings are grouped by month before they are drawn. An operator that opened five sites in
    # one month would otherwise stack five labels on one pixel column and render an unreadable
    # smudge — "1–5" on a single rule says the same thing and can be read.
    opens = m.dropna(subset=["opened_ym"]).sort_values("open_rank")
    for ym, grp in opens.groupby("opened_ym"):
        yr, mth = divmod(int(ym), 12)
        when = pd.Timestamp(year=yr, month=mth + 1, day=1)
        if not (tot.date.min() <= when <= tot.date.max()):
            continue
        ranks = sorted(int(r) for r in grp.open_rank)
        tag = (f"{ranks[0]}–{ranks[-1]}" if len(ranks) > 2 and ranks[-1] - ranks[0] == len(ranks) - 1
               else ",".join(str(r) for r in ranks))
        figt.add_vline(x=when, line=dict(color=MUTED, width=1, dash="dot"))
        figt.add_annotation(x=when, y=1.0, yref="paper", text=tag, showarrow=False,
                            yanchor="bottom", font=dict(size=10, color=MUTED))
    st.plotly_chart(style(figt, height=330, yaxis_title="Washes a month",
                          xaxis_title=None, showlegend=False,
                          margin=dict(l=60, r=20, t=25, b=40)), width="stretch")
    st.caption("Total washes across every site in this cluster. Each dotted line is an opening, "
               "numbered in the same order as the map.")

    # =============================================================================================
    st.divider()
    st.header("5 · What happened to the sites already there")
    eff = _effect(pick, float(max_km), int(min_sites))
    all_eff = _effect_all(float(max_km), int(min_sites))

    if eff.empty:
        st.info("No opening here has a settled neighbour to compare against — every site in this "
                "cluster opened within a year of the one before it. Pick a cluster that was "
                "built out over a longer period.")
    else:
        st.caption("For each new site: what the neighbours **that were already settled** "
                   "(a year or more of trading behind them) washed in the six months after it "
                   "opened, against the six months before. The grey bar is the same comparison "
                   "for this operator's sites **outside** this cluster over the very same months "
                   "— company-wide or seasonal moves show up there, so what is left over is what "
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
                                   "after taking off elsewhere: <b>%{customdata[3]:+.1f}pp</b>"
                                   "<br><span style='opacity:.7'>%{customdata[0]:.0f} settled "
                                   "neighbours · nearest %{customdata[1]:.1f} km</span>"
                                   "<extra></extra>")
        figE.add_hline(y=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(style(figE, height=360, barmode="group", bargap=0.3,
                              yaxis_title="Change in washes, 6 months after vs before (%)",
                              yaxis=dict(ticksuffix="%"),
                              margin=dict(l=70, r=25, t=78, b=45),
                              legend=dict(orientation="h", y=1.13, x=0)), width="stretch")

        et = e[["open_rank", "site", "opened", "n_incumbents", "nearest_km", "incumbent_change",
                "control_change", "excess"]].copy()
        et.columns = ["Opened #", "New site", "Opened", "Settled neighbours",
                      "Nearest (km)", "Neighbours here", "Operator elsewhere", "Difference"]
        et.index = range(1, len(et) + 1)
        html_table(et, fmt={"Opened #": "{:,.0f}", "Settled neighbours": "{:,.0f}",
                            "Nearest (km)": "{:,.2f}", "Neighbours here": "{:+.1%}",
                            "Operator elsewhere": "{:+.1%}", "Difference": "{:+.1%}"})

    if not all_eff.empty:
        x = all_eff.excess.dropna()
        med = float(x.median())
        neg = float((x < 0).mean())
        raw = float(all_eff.incumbent_change.median())
        ctl = float(all_eff.control_change.median())
        cluster_line = ""
        if not eff.empty and eff.excess.notna().any():
            cluster_line = (f"<b>This cluster</b> comes in at "
                            f"<b>{eff.excess.median() * 100:+.1f}pp</b> against an estate median "
                            f"of <b>{med * 100:+.1f}pp</b>.")
        callout("What this shows", f"""
          <b>Opening next to yourself costs you something, and it is small.</b> Across
            <b>{len(x)}</b> openings in <b>{all_eff.cluster_id.nunique()}</b> clusters, the settled
            neighbours give up a median <b>{med * 100:+.1f}pp</b> of washes relative to the same
            operator's sites elsewhere, and <b>{neg:.0%}</b> of openings land on the losing side.
          <b>Most of the raw drop is not the neighbour's fault.</b> The neighbours' own before/after
            change is <b>{raw * 100:+.1f}%</b> and the operator's other sites moved
            <b>{ctl * 100:+.1f}%</b> over the identical months. Skip that control and you charge
            the whole seasonal and company-wide swing to the new site.
          {cluster_line}
          <b>How much to lean on this.</b> It is descriptive. Openings inside one cluster are only
            months apart, so their six-month windows overlap and the events are not independent —
            this is a magnitude, not an identified effect. §⑤ Competition is where entry is
            estimated properly.
        """, S3)

    # =============================================================================================
    with st.expander("Data & method"):
        st.markdown(f"""
**Input.** `conclusion/data/historical_data_5yrs_monthly.csv` — the monthly wash panel,
{cq['n_sites']:,} sites, 2020 to {cd.last_complete_year() + 1}. It is byte-identical to
`proforma/data/panel/main-data-v2-stitched.csv`; this section reads it through `conclusion/data/`
only because every other section's data module does.

**What a cluster is.** Sites sharing a `client_id` that all sit within **{max_km} km** of each
other, by complete-linkage agglomerative clustering on great-circle distance. Complete linkage
means `{max_km} km` caps the cluster's **diameter** — every site is within it of *every* other,
not merely of its nearest neighbour. Single linkage was tried and chains: the Rio Grande Valley
joins into one 120 km "place" through a string of 15 km hops, which is not somewhere you can put
on one screen. Distances are straight-line, **not drive time**, so two sites either side of a
river read closer than they are.

**The site key is `client_id` + `site_id`.** `site_id` on its own is a within-brand index and
collides across operators.

**A coordinate defect, stated rather than hidden.** {cq['n_placeholder']} sites across
{cq['n_placeholder_points']} coordinate points carry a **placeholder** latitude/longitude — one
coordinate shared by several sites of the same operator whose street addresses are all different.
BlueWave stamps 21 Houston-area sites on a single point; Buckeye stamps 10 sites spread across six
Ohio towns on another. Their wash data is real; their location is not. All
{cq['n_placeholder']} are **dropped before clustering** — they carry
{cq['share_of_washes']:.1%} of all washes, and a section about how far apart sites are cannot
include sites whose distance is a geocoding artefact. A further {cq['n_outside_box']} sites fall
outside the continental-US box. {cq['n_usable']:,} sites remain.

Sites sharing a coordinate **and** a street address are a different thing — a second tunnel at one
address, or an operator handoff — and are kept, at a true distance of 0. There are
{cq['n_co_located']} of them.

**Calendar years stop at {cd.last_complete_year()}**, the last year the panel holds all twelve
months of. A site's own opening year is genuinely partial and stays in, flagged as an outlined
bar — that part-year is what the site actually did.

**Section 5's guards.** An "incumbent" needs 12 months of trading before the new opening, because
§⓪ measures a new wash reaching ~98% of its eventual volume only by year 2 — a younger neighbour
is still climbing and would show growth that has nothing to do with anybody. Both the neighbour
and the control sums are **balanced**: a site counts only if it traded in the before window *and*
the after window. The control is held to the same settled test. Without those guards the control
pool reads **+13%** growth that is nothing but the operator's own new sites elsewhere ramping,
and that would be charged to the new neighbour as cannibalization.

**Known gaps.** Straight-line distance is not drive time. The panel is Sonny's customers only, so
a "cluster" is one operator's sites and says nothing about who else washes cars in that town.
`operational_start` is a month stamp; where it is missing the first month that washed a car is
used instead.

**Tables are rendered as raw HTML.** `st.dataframe` / `st.table` segfault the Streamlit server on
the second script run in this environment (pyarrow 25.0.0 + pandas 3.0.2 + streamlit 1.58.0); the
proper fix is an environment pin.
""")
