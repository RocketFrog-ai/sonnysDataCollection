"""
Section — Operator clusters. One operator, one locality, and everybody else in it.

The reader picks a company that **actually clustered** — three or more sites inside one location,
each with whole calendar years behind it — and then picks **one of its localities**. Everything
below is that operator in that town: their sites on a map with the **3-mile trade-area circle**
around each pin, how far apart they built them and in what order, what each has washed month on
month and year on year, and finally **who else washes cars there** — every other operator in the
panel within a few miles, their own circles, and **every single site's wash-count trajectory on one
axis**, this operator's beside its rivals'.

There is deliberately **no whole-estate roll-up**. A national view of an operator answers "how big
are they", which is not the question; this section exists to answer "what did they do to one town",
and those two want different pictures.

Why the 3-mile circle is the point. A site's population, income, traffic and competitor counts are
pulled for a trade area of about that radius — so where two circles overlap, both sites are being
credited with the same households, and the hover box puts those vendor figures next to the wash
counts they are supposed to explain. §④ finds the trade area explains none of the variance in wash
volume; this section shows one mechanical reason that can happen both inside a single company's own
estate and across the operators sharing a town.

**The panel is Sonny's customers only.** The rivals this section can draw are the ones we also sell
to, never all of them, so a locality that looks thin may not be. That limitation is one-directional
and is stated on the page rather than buried: where a rival *is* in the panel, we have its real
monthly wash counts over the same months in the same few square miles, which is not something the
outside world can see.

Distances are shown in **miles** throughout, including the market-grouping slider; the maths
underneath is in kilometres because the haversine is.

The maths is in `cluster_data.py` and is Streamlit-free, so the notebook reports the same numbers.

Chart choices, since one operator can hold 79 sites across 5 states and nine is past any honest
categorical palette:

  • **opening order is the one colour encoding**, on a single blue ramp, light = opened first, used
    identically on the map and the per-site panels so a colour means the same thing twice;
  • **wash volume is a separate single-hue orange ramp**, the same one §⓪ uses for volume, so the
    two encodings are never confused for each other;
  • per-site history is **small multiples on a shared y-axis** when the selection is small enough to
    read, and a **site × month heatmap** when it is not — never 79 overlaid lines;
  • the trajectory chart is the one place lines ARE drawn per site, because comparing this
    operator's sites against its rivals' is the whole point of it. Ours take the blue ramp, rivals
    take one warm hue per company with a dash per site of that company, and companies past the
    fourth fold into a single grey line;
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
from ui import DARK, GRID, INK2, MUTED, S1, S2, SURFACE, callout, html_table, style

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

# Hues for RIVAL COMPANIES on the trajectory chart — one per company, against the blue ORDER_RAMP
# that this operator's own sites use. Validated with the dataviz skill's checker on BOTH surfaces
# alongside the ramp's mid-blue: lightness band, chroma floor, CVD separation, normal-vision floor
# and 3:1 contrast all pass at five slots. A sixth hue fails the lightness band, which is why the
# rival tail folds into one grey line instead of being handed a generated colour.
RIVAL_HUES = (["#d95926", "#199e70", "#a879e6", "#c08a1e"] if DARK
              else ["#c74e1f", "#158a61", "#9a63d8", "#a97a1a"])
RIVAL_LINES = len(RIVAL_HUES)


# =================================================================================================
# Cached wrappers — the module memoises too, but Streamlit's cache also survives a widget change.
# =================================================================================================

@st.cache_data(show_spinner=False)
def _operators(max_km: float, min_market: int, min_years: int) -> pd.DataFrame:
    return cd.operator_index(max_km, min_market, min_years)


@st.cache_data(show_spinner=False)
def _sites(client_id: str, max_km: float, radius_mi: float, min_market: int,
           min_years: int) -> pd.DataFrame:
    return cd.operator_sites(client_id, max_km, radius_mi, min_market, min_years)


@st.cache_data(show_spinner=False)
def _months(client_id: str, keys: tuple) -> pd.DataFrame:
    return cd.operator_months(client_id, keys)


@st.cache_data(show_spinner=False)
def _years(client_id: str, keys: tuple) -> pd.DataFrame:
    return cd.operator_years(client_id, keys)


@st.cache_data(show_spinner=False)
def _locality(client_id: str, market_id: str, max_km: float, radius_mi: float, min_market: int,
              min_years: int, within_mi: float) -> dict:
    return cd.locality_headline(client_id, market_id, max_km, radius_mi, min_market, min_years,
                                within_mi)


@st.cache_data(show_spinner=False)
def _rivals(client_id: str, market_id: str, max_km: float, radius_mi: float, min_market: int,
            min_years: int, within_mi: float) -> pd.DataFrame:
    return cd.region_competitors(client_id, market_id, max_km, radius_mi, min_market, min_years,
                                 within_mi)


@st.cache_data(show_spinner=False)
def _rival_months(client_id: str, market_id: str, max_km: float, radius_mi: float, min_market: int,
                  min_years: int, within_mi: float) -> pd.DataFrame:
    return cd.competitor_months(client_id, market_id, max_km=max_km, radius_mi=radius_mi,
                                min_market_sites=min_market, min_full_years=min_years,
                                within_mi=within_mi)


@st.cache_data(show_spinner=False)
def _kml(client_id: str, market_id: str, max_km: float, radius_mi: float, min_market: int,
         min_years: int, within_mi: float) -> str:
    return cd.kml(client_id, market_id, max_km, radius_mi, min_market, min_years, within_mi)


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


# The three trajectories, drawn identically so they can be read against each other:
#   (subheader, column, y-axis title, hover format, how to pool the folded rival tail)
# Volume pools by SUM; a price and a share pool by wash-weighted MEAN, because averaging an average
# over sites of different sizes would let a tiny site swing the line as hard as a busy one.
TRAJECTORIES = [
    ("Wash-count trajectory — every site in this locality",
     "washes", "Washes a month", "{:,.0f} washes", "sum"),
    ("Price trajectory — revenue per wash",
     "asp", "Revenue per wash ($)", "${:,.2f} a wash", "weighted"),
    ("Membership trajectory — share of washes sold on a plan",
     "mem_share", "Membership share of washes", "{:.0%} on a plan", "weighted"),
]


def _pool(frame: pd.DataFrame, col: str, how: str) -> pd.DataFrame:
    """Collapse several sites into one series — summed for volume, wash-weighted for rates."""
    if how == "sum":
        return frame.groupby("date")[col].sum().reset_index()
    f = frame.dropna(subset=[col]).copy()
    f["_w"] = f[col] * f.washes
    g = f.groupby("date").agg(_w=("_w", "sum"), washes=("washes", "sum")).reset_index()
    g[col] = g._w / g.washes.replace(0, np.nan)
    return g[["date", col]]


def _trajectory(mo_sites: pd.DataFrame, rm: pd.DataFrame, m: pd.DataFrame, h: dict, n_all: int,
                named: list, tail: list, col: str, fmt: str, pooled: str) -> go.Figure:
    """One line per site for a single measure, this operator's against its rivals'.

    Two colour families, so a line's side is legible before any label is read:

      • this operator's sites take the blue opening-order ramp — the SAME shade as the site's pin
        on the map, with the legend leading on the same number, so a line matches a pin without a
        lookup;
      • rival sites take the validated warm hues, one hue per rival COMPANY, and where a company
        has more than one site here they share the hue and differ by dash. A company is a real
        grouping; giving its two sites unrelated colours would invent a distinction the town does
        not have. Companies past `RIVAL_LINES` fold into one grey line.

    Gaps are never bridged (`connectgaps=False`). A missing month is a month the site did not trade
    or — for revenue per wash — one whose revenue is known to be corrupt; joining across it would
    draw a straight line through data that does not exist.
    """
    hov = "<br>%{x|%b %Y}<br><b>" + fmt.replace("{:,.0f}", "%{y:,.0f}").replace(
        "{:,.2f}", "%{y:,.2f}").replace("{:.0%}", "%{y:.0%}") + "</b><extra></extra>"
    fig = go.Figure()

    for _, r in m.sort_values("open_rank").iterrows():
        sub = mo_sites[mo_sites.site_key == r.site_key].sort_values("date")
        if sub.empty or sub[col].notna().sum() == 0:
            continue
        fig.add_scatter(
            x=sub.date, y=sub[col], mode="lines", connectgaps=False,
            name=f"{int(r.open_rank)} · {r.site}",
            line=dict(color=_shade(int(r.open_rank), n_all), width=2),
            legendgroup="ours", legendgrouptitle_text=h["operator"],
            hovertemplate=f"<b>{r.site}</b> · {h['operator']} #{int(r.open_rank)}" + hov)

    DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]
    for i, op_name in enumerate(named):
        sites = rm[rm.operator == op_name]
        for j, (_, sub) in enumerate(sites.groupby("site_key")):
            sub = sub.sort_values("date")
            if sub[col].notna().sum() == 0:
                continue
            label = sub.site.iloc[0]
            fig.add_scatter(
                x=sub.date, y=sub[col], mode="lines", connectgaps=False,
                name=f"{op_name} · {label}" if sites.site_key.nunique() > 1 else op_name,
                line=dict(color=RIVAL_HUES[i], width=1.8, dash=DASHES[j % len(DASHES)]),
                legendgroup="rivals", legendgrouptitle_text="Other operators",
                hovertemplate=f"<b>{label}</b> · {op_name}" + hov)
    if tail:
        sub = _pool(rm[rm.operator.isin(tail)], col, pooled)
        if sub[col].notna().any():
            fig.add_scatter(
                x=sub.date, y=sub[col], mode="lines", connectgaps=False,
                name=f"{len(tail)} smaller operators, combined",
                line=dict(color=MUTED, width=1.6), legendgroup="rivals",
                legendgrouptitle_text="Other operators",
                hovertemplate="<b>" + ", ".join(tail[:4]) + ("…" if len(tail) > 4 else "") + hov)
    return fig


def _mark_openings(fig: go.Figure, m: pd.DataFrame, mo_ours: pd.DataFrame,
                   rm: pd.DataFrame) -> None:
    """A dotted rule at each of this operator's openings, numbered as on the map.

    Openings are grouped by month first: five sites opened in one month would otherwise stack five
    labels on one pixel column. Past fourteen opening months the rules stop being a reading aid and
    become a picket fence, so they are dropped entirely rather than thinned into something
    misleading.
    """
    opens = m.dropna(subset=["opened_ym"])
    groups = list(opens.groupby("opened_ym"))
    if len(groups) > 14:
        return
    lo = min(mo_ours.date.min(), rm.date.min())
    hi = max(mo_ours.date.max(), rm.date.max())
    for ym, grp in groups:
        yr, mth = divmod(int(ym), 12)
        when = pd.Timestamp(year=yr, month=mth + 1, day=1)
        if not (lo <= when <= hi):
            continue
        ranks = sorted(int(v) for v in grp.open_rank)
        tag = (f"{ranks[0]}–{ranks[-1]}"
               if len(ranks) > 2 and ranks[-1] - ranks[0] == len(ranks) - 1
               else ",".join(str(v) for v in ranks))
        fig.add_vline(x=when, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=when, y=1.0, yref="paper", text=tag, showarrow=False,
                           yanchor="bottom", font=dict(size=10, color=MUTED))


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
    st.markdown("<div class='kicker'>Evidence pack · ⑥</div>", unsafe_allow_html=True)
    st.title("Operator clusters")
    g1, g2, g3 = st.columns(3)
    with g1:
        market_mi = st.slider("Sites count as one market within (miles)", 2, 40, 15, 1,
                              help="Complete linkage: every site in a market is within this "
                                   "distance of every other one, not just of its nearest "
                                   "neighbour.")
    with g2:
        min_market = st.slider("Fewest sites to count as a location", 3, 8, 3, 1,
                               help="Markets smaller than this are dropped entirely, and so are "
                                    "operators left with none.")
    with g3:
        min_years = st.slider("Fewest complete years a site must have", 0, 4, 1, 1,
                              help="A site with only a stub of a year has no year-on-year to "
                                   "compare. Filtered before the sites are grouped into markets.")
    max_km = float(market_mi) * KM_PER_MILE

    ops = _operators(max_km, int(min_market), int(min_years))
    if ops.empty:
        st.warning("No operator has that many sites that close together with that much trading "
                   "history. Loosen one of the three sliders above.")
        return

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        labels = dict(zip(ops.client_id, ops.label))
        pick = st.selectbox(f"Operator — {len(ops)} qualify", list(ops.client_id),
                            format_func=lambda c: labels.get(c, c))
    with c2:
        radius_mi = st.slider("Trade-area radius (miles)", 1.0, 6.0, float(cd.TRADE_AREA_MI), 0.5,
                              help="The circle drawn around each site. A site's demographics, "
                                   "traffic and competitor counts are pulled for roughly a 3-mile "
                                   "radius.")
    with c3:
        within_mi = st.slider("Show other operators within (miles)", 1.0, 15.0, 5.0, 0.5,
                              help="Rival sites this close to any of the operator's sites in the "
                                   "chosen locality. Distance is to the nearest of their sites, "
                                   "not to the middle of the market.")

    m_all = _sites(pick, max_km, float(radius_mi), int(min_market), int(min_years))
    if m_all.empty:
        st.warning("Nothing left for this operator at these settings. Loosen a slider.")
        return

    # ONE locality at a time, deliberately. A national roll-up of an operator answers "how big are
    # they"; this section is here to answer "what did they do to one town", and those want different
    # pictures. The whole-estate view is gone rather than hidden behind a toggle.
    markets = (m_all.groupby(["market_id", "market"]).size().rename("sites")
                    .reset_index().sort_values("sites", ascending=False))
    mk_label = dict(zip(markets.market_id,
                        markets.market + " · " + markets.sites.astype(str) + " sites"))
    chosen = st.radio(f"Locality — {len(markets)} where this operator has "
                      f"{min_market} or more sites",
                      list(markets.market_id), format_func=lambda k: mk_label[k],
                      horizontal=True, index=0, key=f"mk_{pick}")
    m = m_all[m_all.market_id == chosen].reset_index(drop=True)
    keys = tuple(sorted(m.site_key))
    # Shading is relative to the OPERATOR's full build-out, not to this locality — a site numbered
    # 23 here is their 23rd overall, which is the fact worth carrying between localities.
    n_all = len(m_all)

    h = _locality(pick, chosen, max_km, float(radius_mi), int(min_market), int(min_years),
                  float(within_mi))
    rivals = _rivals(pick, chosen, max_km, float(radius_mi), int(min_market), int(min_years),
                     float(within_mi))

    # =============================================================================================
    st.divider()
    st.header(f"1 · {h['operator']} in {h['market']}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Their sites here", f"{h['n_sites']}",
              f"closest pair {h['nearest_gap_mi']:.1f} mi", delta_color="off")
    k2.metric("Washes a year", f"{h['washes_per_year']:,.0f}",
              f"typical site {h['median_site']:,.0f}", delta_color="off")
    k3.metric("Catchment they share with each other",
              f"{h['shared_population'] / h['population']:.0%}"
              if pd.notna(h["population"]) and h["population"] else "—",
              f"{h['shared_population']:,.0f} of {h['population']:,.0f} people"
              if pd.notna(h["population"]) else "no trade-area data", delta_color="off")
    k4.metric(f"Other operators within {within_mi:g} mi", f"{h['n_rivals']}",
              f"{h['n_rival_operators']} companies" if h["n_rivals"]
              else "none in the panel", delta_color="off")

    # =============================================================================================
    st.divider()
    st.header("2 · Where they are, and whose catchment is whose")

    mm = m.copy()
    mm["shade"] = [_shade(int(r), n_all) for r in mm.open_rank]
    mm["disc"] = np.sqrt(mm.washes_per_year.clip(lower=1)) / 24 + 18

    # The hover box is the section's densest surface, so it is ordered the way the question gets
    # asked: which site, when did it open, what does it wash, who lives around it, who else is
    # after them, and whose catchment is it really. Trade-area figures come from
    # `historical_data_sitewise.csv`; a site the vendor pull missed shows an em dash rather than a
    # zero, because zero people is a failed lookup, not a place.
    HOVER = (
        "<b>%{customdata[0]}</b> — the operator's #%{customdata[5]}<br>"
        "<span style='opacity:.75'>%{customdata[1]}</span><br>"
        "<br><b>Opened %{customdata[2]}</b> · %{customdata[6]:.0f} months trading · "
        "%{customdata[11]:.0f} full years<br>"
        "<b>%{customdata[3]:,.0f}</b> washes a year · %{customdata[12]:,.0f} to date<br>"
        "%{customdata[13]:.0%} membership · $%{customdata[14]:,.2f} a wash<br>"
        "<br><b>Trade area</b><br>"
        "Population <b>%{customdata[8]}</b> · median income %{customdata[9]}<br>"
        "Traffic %{customdata[15]} a day · %{customdata[16]} car-wash competitors<br>"
        "<br><b>Nearest sibling</b> %{customdata[4]:.1f} mi (%{customdata[10]})<br>"
        "<b>%{customdata[7]:.0%}</b> of this catchment is shared with it"
        "<extra></extra>")

    def _num(s: pd.Series, fmt: str = "{:,.0f}") -> list:
        """Vendor figures that are missing print as an em dash, never as 0."""
        return [fmt.format(v) if pd.notna(v) else "—" for v in s]

    def _cd(f: pd.DataFrame) -> np.ndarray:
        return np.stack([
            f.site, f.address, f.opened, f.washes_per_year,
            f.nearest_km.fillna(0) / KM_PER_MILE, f.open_rank, f.months, f.overlap_nearest,
            _num(f.population), _num(f.median_income, "${:,.0f}"), f.nearest_site, f.full_years,
            f.washes, f.mem_share.fillna(0), f.asp.fillna(0), _num(f.traffic),
            _num(f.competitors),
        ], axis=-1)

    # The view has to hold the rivals too, or half the story sits off screen.
    centre, zoom = _view(pd.concat([m[["lat", "lon"]], rivals[["lat", "lon"]]])
                         if len(rivals) else m, float(radius_mi))
    figm = go.Figure()
    if len(rivals):
        # Rivals get the SAME circle, in the opposing hue. Blue against orange is the safest pair
        # under every form of colour blindness, and the two are the only categories on this map, so
        # identity never rests on a shade of one ramp.
        qlat, qlon = _rings(rivals, float(radius_mi))
        figm.add_scattermap(lat=qlat, lon=qlon, mode="lines", fill="toself",
                            fillcolor="rgba(217,89,38,0.10)",
                            line=dict(color="rgba(217,89,38,0.55)", width=1, ),
                            hoverinfo="skip", name=f"other operators ({len(rivals)})",
                            legendgroup="rival", showlegend=True)
    rlat, rlon = _rings(m, float(radius_mi))
    figm.add_scattermap(lat=rlat, lon=rlon, mode="lines", fill="toself",
                        fillcolor="rgba(57,135,229,0.10)",
                        line=dict(color="rgba(57,135,229,0.55)", width=1),
                        hoverinfo="skip", name=f"{h['operator']} ({len(m)})",
                        legendgroup="ours", showlegend=True)
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
    if len(rivals):
        q = rivals.copy()
        q["disc"] = np.sqrt(q.washes_per_year.clip(lower=1)) / 26 + 12
        figm.add_scattermap(
            lat=q.lat, lon=q.lon, mode="markers",
            marker=dict(size=q.disc, color=S2, opacity=0.95), showlegend=False,
            customdata=np.stack([q.operator, q.site, q.address, q.opened, q.washes_per_year,
                                 q.km_to_operator / KM_PER_MILE, q.nearest_ours, q.overlap_ours,
                                 q.full_years], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                          "<span style='opacity:.75'>%{customdata[2]}</span><br>"
                          "<br>Opened <b>%{customdata[3]}</b> · %{customdata[8]:.0f} full years<br>"
                          "<b>%{customdata[4]:,.0f}</b> washes a year<br>"
                          "<br><b>%{customdata[5]:.1f} mi</b> from %{customdata[6]}<br>"
                          "<b>%{customdata[7]:.0%}</b> catchment overlap<extra></extra>")
    figm.update_layout(map=dict(style="carto-darkmatter" if DARK else "carto-positron",
                                center=centre, zoom=zoom))
    st.plotly_chart(style(figm, height=580, margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(orientation="h", y=0.99, x=0.01,
                                      bgcolor="rgba(0,0,0,0.35)" if DARK
                                      else "rgba(255,255,255,0.75)")), width="stretch")

    # The same locality as a Google Earth file. Earth gives what a flat basemap cannot — real
    # imagery, the actual forecourts and queue lanes, and a layer tree the reader can switch on and
    # off: this operator's pins, their circles, each rival company as its own sub-folder, and the
    # rivals' circles. Every pin carries the figures the hover box shows here, because in Earth
    # there is no hover box.
    d1, d2 = st.columns([1, 3])
    with d1:
        st.download_button(
            "⬇︎ Open this locality in Google Earth (.kml)",
            _kml(pick, chosen, max_km, float(radius_mi), int(min_market), int(min_years),
                 float(within_mi)),
            f"{pick}_{chosen.split('::')[-1]}_{radius_mi:g}mi.kml",
            "application/vnd.google-earth.kml+xml", key=f"kml_{pick}_{chosen}")
    with d2:
        st.markdown("Drag the file onto **earth.google.com/web**, or open it in Google Earth Pro. "
                    "The sidebar tree lets you toggle this operator's pins, their "
                    f"{radius_mi:g}-mile circles, and each rival company separately.")

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
    show = m[["open_rank", "site", "address", "state", "opened", "full_years", "washes_per_year",
              "population", "median_income", "traffic", "competitors", "nearest_competitor_mi",
              "overlap_nearest", "shared_population", "nearest_site", "nearest_km", "washes",
              "mem_share", "asp"]].copy()
    show["nearest_km"] = show.nearest_km / KM_PER_MILE
    show["address"] = [f"<a href='{u}' target='_blank' rel='noopener'>{a}</a>"
                       for a, u in zip(show.address, m.maps_url)]
    show.columns = ["Opened #", "Site", "Address (opens Google Maps)", "State", "Opened",
                    "Full years", "Washes a year", "Trade-area population", "Median income",
                    "Traffic a day", "Competitors", "Nearest competitor (mi)", "Catchment shared",
                    "People double-counted", "Nearest sibling", "Nearest sibling (mi)",
                    "Washes to date", "Membership share", "Revenue per wash"]
    show.index = range(1, len(show) + 1)
    html_table(show, fmt={"Opened #": "{:,.0f}", "Full years": "{:,.0f}",
                          "Washes to date": "{:,.0f}", "Washes a year": "{:,.0f}",
                          "Trade-area population": "{:,.0f}", "Median income": "${:,.0f}",
                          "Traffic a day": "{:,.0f}", "Competitors": "{:,.0f}",
                          "Nearest competitor (mi)": "{:,.2f}",
                          "People double-counted": "{:,.0f}",
                          "Membership share": "{:.0%}", "Revenue per wash": "${:,.2f}",
                          "Nearest sibling (mi)": "{:,.2f}", "Catchment shared": "{:.0%}"})
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
        y = y.merge(m[["site_key", "site", "open_rank", "opened_ym"]], on="site_key", how="left")
        years = sorted(int(v) for v in y.year.unique())

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
        # Past PANEL_LIMIT sites there are no per-site panels — one panel each stops being
        # readable, and the table below plus the month-by-month heatmap already carry every number.

        piv_y = (y.pivot_table(index="site_key", columns="year", values="washes", aggfunc="sum")
                  .reindex(order.site_key.values))
        piv_y.index = order.site.values
        piv_y.columns = [str(int(c)) for c in piv_y.columns]
        html_table(piv_y, fmt={c: "{:,.0f}" for c in piv_y.columns}, index_label="Site")
        st.download_button("Download year-by-year (CSV)", piv_y.to_csv(),
                           f"operator_years_{pick}.csv", "text/csv", key=f"dl_years_{pick}")
    # =============================================================================================
    st.divider()
    st.header(f"6 · Who else washes cars in {h['market']}")

    if rivals.empty:
        st.info(f"No other operator in the panel has a site within {within_mi:g} miles of these "
                f"{h['n_sites']}. That does **not** mean the locality is empty — the panel only "
                "holds Sonny's customers, so a rival we do not sell to is invisible here. Widen "
                "the “other operators within” slider to look further out.")
    else:

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rival sites", f"{h['n_rivals']}",
                  f"{h['n_rival_operators']} companies", delta_color="off")
        r2.metric("Their washes a year", f"{h['rival_washes_per_year']:,.0f}",
                  f"{h['rival_share']:.0%} of the washing we can see", delta_color="off")
        r3.metric("Closest rival", f"{h['nearest_rival_mi']:.1f} mi",
                  "to their nearest site", delta_color="off")
        r4.metric("Rivals inside a catchment",
                  f"{int((rivals.overlap_ours > 0).sum())}",
                  f"of {len(rivals)} share a {radius_mi:g}-mile circle", delta_color="off")

        rt = rivals[["operator", "site", "address", "state", "opened", "full_years",
                     "washes_per_year", "km_to_operator", "nearest_ours", "overlap_ours",
                     "population"]].copy()
        rt["km_to_operator"] = rt.km_to_operator / KM_PER_MILE
        rt["address"] = [f"<a href='https://www.google.com/maps/search/?api=1&query="
                         f"{la:.6f},{lo:.6f}' target='_blank' rel='noopener'>{a}</a>"
                         for a, la, lo in zip(rt.address, rivals.lat, rivals.lon)]
        rt.columns = ["Operator", "Site", "Address (opens Google Maps)", "State", "Opened",
                      "Full years", "Washes a year", "Miles away", "Nearest of theirs",
                      "Catchment overlap", "Trade-area population"]
        rt.index = range(1, len(rt) + 1)
        html_table(rt, fmt={"Full years": "{:,.0f}", "Washes a year": "{:,.0f}",
                            "Miles away": "{:,.2f}", "Catchment overlap": "{:.0%}",
                            "Trade-area population": "{:,.0f}"})

        # --- trajectories ------------------------------------------------------------------------
        rm = _rival_months(pick, chosen, max_km, float(radius_mi), int(min_market),
                           int(min_years), float(within_mi))
        mo_ours = _months(pick, keys)
        if not rm.empty and not mo_ours.empty:
            mo_sites = mo_ours.merge(m[["site_key", "site", "open_rank"]], on="site_key")
            by_op = rm.groupby("operator").washes.sum().sort_values(ascending=False)
            named, tail = list(by_op.index[:RIVAL_LINES]), list(by_op.index[RIVAL_LINES:])
            n_lines = len(m) + int(rm.site_key.nunique()) + (1 if tail else 0)

            for title, col, y_title, fmt, pooled in TRAJECTORIES:
                st.subheader(title)
                # An operator whose revenue never reaches the panel gets an empty price chart, and
                # an empty chart with no explanation reads as a bug. Say whose data is missing and
                # keep the chart, because the rivals in the same town usually do have prices.
                if col == "asp":
                    bad = int(mo_sites.corrupt_asp.sum())
                    if mo_sites.asp.notna().sum() == 0:
                        st.info(f"**{h['operator']} reports no usable revenue in this panel** — "
                                f"{bad} of {len(mo_sites)} of their site-months carry washes with "
                                "the revenue recorded as zero, so revenue per wash cannot be "
                                "computed for them. Their rivals' prices are still drawn below. "
                                "See “Data & method”.")
                    elif bad:
                        st.info(f"**{bad} of {len(mo_sites)}** of {h['operator']}'s site-months "
                                "record washes against zero revenue and are left as gaps rather "
                                "than plotted at zero.")
                figc = _trajectory(mo_sites, rm, m, h, n_all, named, tail, col, fmt, pooled)
                _mark_openings(figc, m, mo_ours, rm)
                # One line per site means the legend can run past twenty entries, which no
                # horizontal strip holds. It becomes a grouped column on the right, and the plot
                # keeps the full width minus that column.
                st.plotly_chart(style(figc, height=max(430, 190 + 17 * min(n_lines, 26)),
                                      yaxis_title=y_title, xaxis_title=None,
                                      margin=dict(l=75, r=10, t=40, b=40),
                                      legend=dict(orientation="v", y=1, x=1.01, xanchor="left",
                                                  font=dict(size=10),
                                                  groupclick="toggleitem")),
                                width="stretch")


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
can legally draw. Two things close that gap without a key: every address in the sitewise tables
links to **that exact coordinate in Google Maps**, and the **`.kml` download** opens the whole
locality in **Google Earth** — pins, {cd.TRADE_AREA_MI:g}-mile circles as real polygons, and a
layer tree with this operator, their circles, and one sub-folder per rival company, each togglable.
The circles in that file are generated by the same geometry as the ones on screen and measure
2.996–2.997 miles from pin to ring. The pins, circles and distances here are all computed from the
panel's own lat/lon.

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

**Who counts as a rival.** Another operator with a site within the slider's distance of **any** of
this operator's sites in the chosen locality — distance to the nearest of their sites, not to the
market's centroid, so a rival across the road from the newest site counts even if the middle of the
market is six miles away. No full-year filter is applied to rivals: that filter exists so a
*subject* site can carry a year-on-year comparison, and a rival that opened last month is still
competition.

**The panel is Sonny's customers only**, which bounds section 6 in one direction. A rival we do not
sell to is invisible, so a locality can look emptier than it is, and the measured effect is the
effect on the *visible* competition — biased toward zero if the unseen rivals absorbed the hit
instead. What is not bounded is the other direction: where a rival is in the panel, these are its
real monthly wash counts over the same months in the same few square miles.

**Section 6's guards.** A rival counts only once it has 12 months of trading behind it, because §⓪
measures a new wash reaching ~98% of its eventual volume only by year 2 — a younger rival is still
climbing and would show growth that has nothing to do with the newcomer. Both the rival and control
sums are **balanced**: a site counts only if it traded in the before window *and* the after window.
The control is **those same rival companies' own sites outside this locality**, held to the same
settled test, so a regional or seasonal move is subtracted rather than charged to the new arrival.
The identical guards, on the operator's own siblings, move a naive control from +13% to −0.3%.

Openings inside one locality are only months apart, so their six-month windows overlap and the
events are not independent. This is descriptive; **§⑤ Competition is where entry is estimated
properly.**

**Revenue per wash, and where it goes missing.** A site-month is set aside when it records
**{cd.ASP_MIN_WASH}+ washes against a price below ${cd.ASP_FLOOR_MEM:.0f} a membership wash or
${cd.ASP_FLOOR_RET:.0f} a retail wash** — the rule and the thresholds come from
`proforma/pnl/opex.py::_drop_corrupt_asp_rows`, restated in `cluster_data.py` because
`conclusion/` is standalone. It is not a rounding problem: **8.2% of all site-months, 315 sites
across 95 operators** carry a normal wash count with revenue recorded as **zero**, and Luvcarwash —
the largest operator here — does it in 99% of its months, membership revenue absent while thousands
of membership washes are logged. Left in, those months drag the 5th percentile of $/wash from
$10.30 to $5.95 and draw the affected sites diving toward zero, which is not what happened to their
prices. They are plotted as **gaps, never as zeros**, and lines never bridge a gap.

Membership share is unaffected — it is a ratio of two wash counts and needs no revenue at all.

**Known gaps.** Straight-line distance is not drive time, and a 3-mile circle is not a real
catchment — it has no roads, rivers or county lines in it. The panel is Sonny's customers only, so
an operator's "market" says nothing about who else washes cars in that town. `operational_start` is
a month stamp; where it is missing the first month that washed a car is used instead.

**Tables are rendered as raw HTML.** `st.dataframe` / `st.table` segfault the Streamlit server on
the second script run in this environment (pyarrow 25.0.0 + pandas 3.0.2 + streamlit 1.58.0); the
proper fix is an environment pin.
""")
