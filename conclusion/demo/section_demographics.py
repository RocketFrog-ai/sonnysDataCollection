"""
Section — Demographics. Does the market a site sits in explain how much it washes?

Vertical scroll, same as the other sections. The order is the order of the argument:

  1  the estate on a map, with any feature laid over it — the visual version of the question
  2  every measure's quintile curve on ONE axis — the exhibit that needs no statistics. It is a
     single static plot on purpose: any one measure shown alone through a dropdown looks like it
     might be doing something, and the finding is that they are all flat together.
  3  the full correlation grid — measures × wash types, then measures × region. The regional cut
     is the honest robustness check: a flat national average can hide a real signal somewhere.
  4  every feature ranked, three ways (raw, within state, within operator)
  5  give a model all 31 features at once and score it on markets it has never seen
  6  so what *does* explain volume
  7  state by state
  8  the whole table
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import demographics_data as dd
from ui import (CRITICAL, GOOD, GRID, INK2, MUTED, S1, S2, S3, SERIOUS, SURFACE, WARNING,
                callout, html_table, style)


@st.cache_data(show_spinner=False)
def _cohort() -> pd.DataFrame:
    return dd.cohort()


@st.cache_data(show_spinner=False)
def _corr(target: str) -> pd.DataFrame:
    return dd.correlations(target)


@st.cache_data(show_spinner="Scoring the model on unseen markets…")
def _oos() -> pd.DataFrame:
    return dd.oos_scores()


@st.cache_data(show_spinner="Working out what does explain volume…")
def _decomp() -> pd.DataFrame:
    return dd.variance_decomposition()


@st.cache_data(show_spinner=False)
def _neighbours() -> pd.DataFrame:
    return dd.neighbour_curve()


@st.cache_data(show_spinner=False)
def _states() -> pd.DataFrame:
    return dd.state_table()


@st.cache_data(show_spinner=False)
def _state_link() -> pd.DataFrame:
    return dd.state_feature_link()


@st.cache_data(show_spinner=False)
def _curves() -> pd.DataFrame:
    return dd.quintile_curves()


@st.cache_data(show_spinner=False)
def _target_grid() -> pd.DataFrame:
    return dd.target_grid()


@st.cache_data(show_spinner=False)
def _region_grid() -> pd.DataFrame:
    return dd.region_grid("Retail washes")


@st.cache_data(show_spinner="Scoring each region on its own held-out states…")
def _region_verdict() -> pd.DataFrame:
    return dd.region_verdict("Retail washes")


@st.cache_data(show_spinner="Working out what noise alone gives at each sample size…")
def _noise_floor() -> pd.DataFrame:
    return dd.region_noise_floor("Retail washes")


def _k(v: float) -> str:
    return f"{v/1000:,.0f}k"


# A correlation is signed and zero is the meaningful midpoint, so this is a diverging ramp: two
# hues meeting at a neutral grey, never a rainbow. Symmetric range so +0.20 and −0.20 read equally
# strong, and capped at ±0.40 rather than the data max — an auto-scaled ramp would paint a grid
# whose largest cell is 0.16 in full saturation and make nothing look like something.
DIVERGING = [[0.0, "#b04a1e"], [0.25, "#e0a487"], [0.5, "#9c9c96"],
             [0.75, "#7fa9e0"], [1.0, "#1a5fb4"]]
HEAT_LIMIT = 0.40


def _heatmap(grid: pd.DataFrame, cols: list[str], title: str, height: int = 720,
             col_note: dict[str, str] | None = None) -> None:
    """One measures × columns correlation grid, rows already ordered by the caller."""
    z = grid[cols].to_numpy()
    xlab = [c if not col_note else f"{c}<br><span style='font-size:10px;opacity:.65'>"
            f"{col_note[c]}</span>" for c in cols]
    fig = go.Figure(go.Heatmap(
        z=z, x=xlab, y=list(grid.index), colorscale=DIVERGING,
        zmin=-HEAT_LIMIT, zmax=HEAT_LIMIT, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="rank<br>correlation", font=dict(size=10)),
                      thickness=11, len=.5, tickvals=[-.4, -.2, 0, .2, .4],
                      ticktext=["−0.40", "−0.20", "0", "+0.20", "+0.40"],
                      tickfont=dict(size=9)),
        customdata=np.stack([np.tile(np.array(grid.family)[:, None], (1, len(cols)))], axis=-1),
        hovertemplate="<b>%{y}</b><br>%{x}: <b>%{z:+.3f}</b><br>"
                      "<span style='opacity:.7'>%{customdata[0]}</span><extra></extra>"))
    st.plotly_chart(style(fig, height=height, title=dict(text=title, font=dict(size=13)),
                          xaxis=dict(side="top", showgrid=False, tickfont=dict(size=11)),
                          yaxis=dict(autorange="reversed", showgrid=False,
                                     tickfont=dict(size=10)),
                          margin=dict(l=250, r=25, t=90, b=20)), width="stretch")


def render() -> None:
    h = dd.headline()

    st.markdown("<div class='kicker'>Evidence pack · section ④</div>", unsafe_allow_html=True)
    st.title("Demographics")
    st.markdown(
        f"Every site-selection proforma in this industry rests on one premise: **score the "
        f"neighbourhood — people, income, cars, traffic, competitors — and the score tells you what "
        f"the wash will do.** This section tests that premise against **{h['sites']:,} sites** that "
        f"traded every single month of 2025, in **{h['states']} states**, using "
        f"**{h['features']} market measures** per site.")

    st.caption(f"Twelve complete months means every site's yearly total is a real sum, never scaled "
               f"up from a part-year — so no site looks small just because it opened in August. "
               f"Median site: **{h['median_washes']:,.0f} washes** in 2025; the busiest tenth do "
               f"**{h['spread']:.1f}×** what the quietest tenth do. That gap is what we are trying "
               f"to explain.")

    # =============================================================================================
    st.divider()
    st.header("1 · The estate on a map")
    st.markdown("Two maps of the same 1,249 sites. The left one is coloured by how much each site "
                "actually washes. The right one is coloured by whatever market measure you pick. "
                "**If the market drove volume, the two maps would look like the same map.**")

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        feat = st.selectbox("Market measure to lay over the map",
                            list(dd.FEATURES), index=list(dd.FEATURES).index(
                                "Population in the trade area"), key="dem_mapfeat")
    with mc2:
        view = st.radio("View", ["Every site", "By state"], horizontal=True, key="dem_mapview")

    if view == "Every site":
        left, right = dd.map_frame("Total washes"), dd.map_frame(feat)
        cols = st.columns(2)
        for c, frame, title, cs in ((cols[0], left, "Washes in 2025", "Blues"),
                                    (cols[1], right, feat, "Oranges")):
            f = go.Figure(go.Scattergeo(
                lat=frame.lat, lon=frame.lon, mode="markers",
                marker=dict(size=6, color=frame.pct, colorscale=cs, cmin=0, cmax=100,
                            line=dict(width=.5, color=SURFACE),
                            colorbar=dict(title=dict(text="percentile", font=dict(size=10)),
                                          thickness=9, len=.55, tickfont=dict(size=9))),
                customdata=np.stack([frame.site, frame.state, frame.value, frame.pct,
                                     frame.total_washes], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                              f"{title}: %{{customdata[2]:,.0f}}<br>"
                              "= %{customdata[3]:.0f}th percentile<br>"
                              "<span style='opacity:.7'>2025 washes: %{customdata[4]:,.0f}"
                              "</span><extra></extra>"))
            f.update_geos(scope="usa", bgcolor=SURFACE, landcolor=SURFACE, lakecolor=SURFACE,
                          subunitcolor=GRID, countrycolor=GRID, showlakes=False)
            with c:
                st.markdown(f"<div class='kicker'>{title}</div>", unsafe_allow_html=True)
                st.plotly_chart(style(f, height=380, margin=dict(l=0, r=0, t=10, b=0)),
                                width="stretch")
    else:
        stt = _states()
        ranked = stt[stt.enough_sites]
        fcol = {"Population in the trade area": "median_pop",
                "Median household income": "median_income", "Traffic, all day": "median_traffic",
                "Car washes nearby": "median_competitors"}.get(feat)
        cols = st.columns(2)
        panes = [("Median washes per site", ranked.median_washes, "Blues", ",.0f")]
        if fcol:
            panes.append((f"Median {feat.lower()}", ranked[fcol], "Oranges", ",.0f"))
        else:
            grp = _cohort().groupby("state")[dd.FEATURES[feat][0]].median().reindex(ranked.state)
            panes.append((f"Median {feat.lower()}", grp.values, "Oranges", ",.2f"))
        for c, (title, vals, cs, fmt) in zip(cols, panes):
            f = go.Figure(go.Choropleth(
                locations=ranked.state, locationmode="USA-states", z=pd.Series(vals).rank(pct=True) * 100,
                colorscale=cs, zmin=0, zmax=100, marker_line_color=SURFACE, marker_line_width=1,
                colorbar=dict(title=dict(text="percentile", font=dict(size=10)), thickness=9,
                              len=.55, tickfont=dict(size=9)),
                customdata=np.stack([ranked.sites, np.asarray(vals)], axis=-1),
                hovertemplate="<b>%{location}</b><br>" + title + ": %{customdata[1]:" + fmt + "}<br>"
                              "<span style='opacity:.7'>%{customdata[0]:.0f} sites"
                              "</span><extra></extra>"))
            f.update_geos(scope="usa", bgcolor=SURFACE, lakecolor=SURFACE, showlakes=False)
            with c:
                st.markdown(f"<div class='kicker'>{title}</div>", unsafe_allow_html=True)
                st.plotly_chart(style(f, height=380, margin=dict(l=0, r=0, t=10, b=0)),
                                width="stretch")
        st.caption(f"Only the {int(ranked.sites.count())} states with "
                   f"{dd.MIN_STATE_SITES}+ sites are shaded — below that a state median is one or "
                   "two sites and moves on noise.")

    link = _state_link()
    row = link[link.feature == feat].iloc[0]
    sl = link[link.family != "The site itself"]
    strongest = sl.iloc[0]
    callout("Reading the map", f"""
        Across the {int(row.states)} states big enough to rank, a state's typical
        <b>{feat.lower()}</b> and its typical wash volume line up at a rank correlation of
        <b>{row.rho:+.2f}</b> (p = {row.p:.2f}).
            A correlation of ±1.00 would mean the two maps are the same map; 0.00 means knowing
            one tells you nothing about the other.
        The strongest market measure of all {len(sl)} across states is
            <b>{strongest.feature.lower()}</b> at {strongest.rho:+.2f}, and even that does not
            clear the bar for significance (p = {strongest.p:.2f}).
        The one thing that <i>does</i> separate busy states from quiet ones is not a market
            measure at all: it is <b>membership share</b>
            ({link[link.family == 'The site itself'].rho.iloc[0]:+.2f},
            p = {link[link.family == 'The site itself'].p.iloc[0]:.3f}). New York, New Jersey and
            Arizona sit at the bottom of the volume table and also at the bottom of the
            membership table.
    """, accent=S1)

    # =============================================================================================
    st.divider()
    st.header("2 · All 31 measures, one picture")
    st.markdown("Take a measure — say population. Sort all 1,263 sites on it, cut them into five "
                "equal groups from the lowest fifth to the highest, and see what each group washed. "
                "**A real driver makes that line climb.** Here is that line for every one of the 31 "
                "measures at once, each starting from its own lowest fifth so they can share an "
                "axis.")

    qc = _curves()
    mk = qc[qc.is_market]
    order = ["Lowest fifth", "2nd", "Middle", "4th", "Highest fifth"]

    fq = go.Figure()
    fq.add_hline(y=1, line=dict(color=MUTED, width=1.5, dash="dot"))
    for name, g in mk.groupby("measure", sort=False):
        g = g.set_index("bucket").reindex(order).reset_index()
        fq.add_scatter(x=g.bucket, y=g.ratio, mode="lines", name=name, legendgroup="market",
                       showlegend=False, line=dict(color=S2, width=1.4), opacity=.42,
                       hovertemplate=f"<b>{name}</b><br>%{{x}}: <b>%{{y:.2f}}×</b> the lowest "
                                     f"fifth<extra></extra>")
    # One dummy trace so the 31 faint lines get a single legend entry instead of 31.
    fq.add_scatter(x=[None], y=[None], mode="lines", name="Each market measure (31)",
                   line=dict(color=S2, width=1.8), opacity=.6)

    mem = qc[qc.measure == "Membership customers"].set_index("bucket").reindex(order).reset_index()
    fq.add_scatter(x=mem.bucket, y=mem.ratio, mode="lines+markers",
                   name="How many members the site has (not a market measure)",
                   line=dict(color=S1, width=3.5),
                   marker=dict(size=11, line=dict(width=2, color=SURFACE)),
                   customdata=np.stack([mem["median"]], axis=-1),
                   hovertemplate="<b>Members</b><br>%{x}: <b>%{y:.2f}×</b> the lowest fifth"
                                 "<br>%{customdata[0]:,.0f} washes<extra></extra>")
    fq.add_annotation(x="Highest fifth", y=qc.attrs["members"], xanchor="right", yshift=-24,
                      text=f"<b>{qc.attrs['members']:.1f}×</b>", showarrow=False,
                      font=dict(size=15, color=S1))
    fq.add_annotation(x="Highest fifth", y=qc.attrs["widest_spread"], xanchor="right", yshift=16,
                      text="every market measure lands here", showarrow=False,
                      font=dict(size=12, color=INK2))
    st.plotly_chart(style(fq, height=470,
                          yaxis_title="Washes vs the lowest fifth on that measure",
                          yaxis=dict(tickvals=[1, 2, 3, 4, 5],
                                     ticktext=["same", "2×", "3×", "4×", "5×"],
                                     range=[0, qc.attrs["members"] * 1.12]),
                          xaxis=dict(type="category", title="", showgrid=False),
                          margin=dict(l=80, r=30, t=86, b=45),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    callout("One picture, one finding", f"""
        <b>All {qc.attrs['n_market']} market measures are the flat bundle at the bottom.</b> Sort
            the sites on any of them and the busiest fifth washes about the same as the quietest:
            the median measure moves volume by <b>{qc.attrs['median_spread']:.2f}×</b>, and
            {qc.attrs['within_10pct']} of {qc.attrs['n_market']} move it by less than 10% in either
            direction. The widest one in the whole set is
            <b>{qc.attrs['widest_market'].lower()}</b> at just
            <b>{qc.attrs['widest_spread']:.2f}×</b>.
        <b>The single blue line is what a real driver looks like.</b> Sort the same 1,263 sites on
            how many members they have and the top fifth washes <b>{qc.attrs['members']:.1f}×</b>
            the bottom fifth, climbing at every step. Nothing about the neighbourhood does that.
        <b>Put it against the gap we are trying to explain.</b> Our busiest sites do
            <b>{h['spread']:.1f}×</b> what our quietest do. A measure that moves the median by
            {qc.attrs['median_spread']:.2f}× cannot account for a {h['spread']:.1f}× gap — not
            individually, and (section 5) not collectively either.
        <b>Why lines and not a dropdown.</b> Any one of these measures shown on its own looks like
            it might be doing something. The finding is that they are <i>all</i> flat, and that
            only shows up when you put them on one axis.
    """, accent=CRITICAL)

    # =============================================================================================
    st.divider()
    st.header("3 · The whole correlation grid")
    st.markdown("Every measure against every wash type, in one grid. Blue is a positive "
                "relationship, orange negative, grey nothing. **A grid with real structure in it "
                "would have strong colour somewhere.**")

    tg = _target_grid()
    _heatmap(tg, list(dd.TARGETS), "Rank correlation with wash volume, all 1,263 sites",
             height=740)

    callout("Two things worth seeing here", f"""
        <b>The grid is grey.</b> The strongest of all {tg.attrs['cells']} cells is
            <b>{tg.attrs['max_abs']:.2f}</b>, and <b>{tg.attrs['under_10']:.0%}</b> of them sit
            under 0.10. Nothing in the neighbourhood is strongly attached to wash volume.
        <b>But the columns are not the same.</b> Compare the middle and right columns: the typical
            measure correlates <b>{tg.attrs['retail_median']:.2f}</b> with retail washes and only
            <b>{tg.attrs['member_median']:.2f}</b> with membership washes — nearly three times
            weaker. That direction makes sense: a retail wash is a stranger driving past who
            decides to pull in, so the neighbourhood gets a say. A membership wash is a
            subscription that was already sold, and the neighbourhood does not.
        <b>Income is the clearest case.</b> Households in the $150–250k bands correlate
            <b>+0.15</b> with retail washes and <b>−0.02</b> with membership washes — the sign
            actually flips. Only {tg.attrs['agree']:.0%} of the 31 measures even agree on direction
            between the two.
        <b>So what.</b> If the market matters anywhere, it matters to the drive-up half of the
            business — which is the half that is shrinking (see ⓪ General). It has almost nothing
            to say about the subscription half, which is where the volume now is.
    """, accent=S1)

    st.markdown("#### Does the same grid hold in every region?")
    st.markdown("The flat result above is a national average, and an average can hide a real "
                "signal in one part of the country. Same 31 measures, same **retail washes** — the "
                "column where the market had most to say — split by census region.")

    rg = _region_grid()
    nf = _noise_floor().set_index("region")
    # The column header carries the noise floor, because the raw grid flatters the small regions:
    # at n=95 the typical |rho| from pure chance is 0.07, against 0.02 at n=823. Without it a
    # reader compares four columns that were never on the same footing.
    _heatmap(rg, dd.REGION_ORDER,
             "Rank correlation with retail washes, by region", height=740,
             col_note={r: f"n={rg.attrs['sites'][r]} · chance alone gives "
                          f"±{nf.loc[r, 'noise_floor']:.2f}" for r in dd.REGION_ORDER})

    st.markdown("##### Correcting for the fact that the regions are wildly unequal")
    st.markdown("823 sites against 95. A correlation drifts further from zero the fewer sites you "
                "have, so the small regions get a head start. Three ways of taking it away.")

    fn = go.Figure()
    nfr = _noise_floor()
    fn.add_bar(x=nfr.region, y=nfr.observed, name="What the grid shows",
               marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
               customdata=np.stack([nfr.n], axis=-1),
               hovertemplate="<b>%{x}</b><br>typical |correlation| = <b>%{y:.3f}</b>"
                             "<br><span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                             "<extra></extra>")
    fn.add_bar(x=nfr.region, y=nfr.noise_floor, name="What pure chance gives at that n",
               marker=dict(color=MUTED, line=dict(width=2, color=SURFACE)),
               hovertemplate="<b>%{x}</b><br>noise floor = <b>%{y:.3f}</b><extra></extra>")
    fn.add_bar(x=nfr.region, y=nfr.balanced,
               name=f"Every region cut to {nfr.attrs['balanced_n']} sites",
               marker=dict(color=S3, line=dict(width=2, color=SURFACE)),
               error_y=dict(type="data", symmetric=False,
                            array=(nfr.balanced_hi - nfr.balanced).fillna(0),
                            arrayminus=(nfr.balanced - nfr.balanced_lo).fillna(0),
                            color=MUTED, thickness=1.5, width=5),
               hovertemplate="<b>%{x}</b><br>at equal sample size = <b>%{y:.3f}</b>"
                             "<extra></extra>")
    st.plotly_chart(style(fn, height=380, barmode="group",
                          yaxis_title="Typical |rank correlation| across the 31 measures",
                          xaxis=dict(type="category", title=""), margin=dict(t=86),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    nt = nfr[["region", "n", "observed", "noise_floor", "excess", "balanced", "p_perm"]].copy()
    nt.columns = ["Region", "Sites", "Typical |rho| observed", "Noise floor at that n",
                  "Excess over noise", f"Balanced to {nfr.attrs['balanced_n']} sites",
                  "Permutation p"]
    html_table(nt.set_index("Region"), index_label="Region",
               fmt={"Typical |rho| observed": "{:.3f}", "Noise floor at that n": "{:.3f}",
                    "Excess over noise": "{:.3f}",
                    f"Balanced to {nfr.attrs['balanced_n']} sites": "{:.3f}",
                    "Permutation p": "{:.3f}"})

    so_n = nfr[nfr.region == "South"].iloc[0]
    ne_n = nfr[nfr.region == "Northeast"].iloc[0]
    we_n = nfr[nfr.region == "West"].iloc[0]
    callout("The sample-size objection, tested", f"""
        <b>The objection is right that the floor moves.</b> Permuting the wash counts inside each
            region — which destroys any real relationship but keeps the 31 measures as tangled as
            they really are — gives a typical |rho| of <b>{ne_n.noise_floor:.2f}</b> at
            n&nbsp;=&nbsp;{int(ne_n.n)} and only <b>{so_n.noise_floor:.2f}</b> at
            n&nbsp;=&nbsp;{int(so_n.n)}. Small regions really do start
            <b>{ne_n.noise_floor / so_n.noise_floor:.0f}×</b> further from zero for free.
        <b>It does not explain the gap.</b> Subtract each region's own floor and the excess is
            <b>{so_n.excess:.2f}</b> in the South against <b>{we_n.excess:.2f}</b> in the West and
            <b>{ne_n.excess:.2f}</b> in the Northeast — still a 4–5× difference.
        <b>Nor does equalising the samples.</b> Cut every region to
            {nfr.attrs['balanced_n']} sites and re-measure {nfr.attrs['draws']} times: the South
            comes out at <b>{so_n.balanced:.2f}</b> [{so_n.balanced_lo:.2f}–{so_n.balanced_hi:.2f}]
            against the Northeast's <b>{ne_n.balanced:.2f}</b>. Same power, same ordering. (The
            Northeast has no band because it <i>is</i> the reference size.)
        <b>Note the South's green bar sits above its blue one.</b> That is not an error — it is the
            whole effect in one place. Shrinking a region from {int(so_n.n)} sites to
            {nfr.attrs['balanced_n']} <i>raises</i> its measured correlation, from
            {so_n.observed:.2f} to {so_n.balanced:.2f}, purely because small samples wander further
            from zero. It is the reason the four columns of the grid above could never be compared
            as they stood.
        <b>All four regions beat their own noise, including the South.</b> Permutation
            p&nbsp;=&nbsp;{so_n.p_perm:.3f} / {we_n.p_perm:.3f} / {ne_n.p_perm:.3f}. This is an
            omnibus test on the whole grid at once, so it is not vulnerable to the "31 measures are
            really 7–9 things" problem that counting individually-significant measures has.
            There is something everywhere; it is just far smaller in the South.
    """, accent=S3)

    rv = _region_verdict()
    vt = rv[["region", "sites", "states", "strongest", "rho", "n_sig", "median_abs",
             "median_within_state", "components", "oos_r2"]].copy()
    vt.columns = ["Region", "Sites", "States", "Strongest measure", "Its rho",
                  "Measures clearing FDR", "Typical |rho|", "Typical |rho| within state",
                  "Independent factors", "Held-out R² (total washes)"]
    html_table(vt.set_index("Region"), index_label="Region",
               fmt={"Its rho": "{:+.2f}", "Typical |rho|": "{:.2f}",
                    "Typical |rho| within state": "{:.2f}",
                    "Held-out R² (total washes)": "{:+.3f}"})

    ne = rv[rv.region == "Northeast"].iloc[0]
    we = rv[rv.region == "West"].iloc[0]
    so = rv[rv.region == "South"].iloc[0]
    callout("This one qualifies the headline — read it", f"""
        <b>Outside the South, the market does say something about retail washes.</b> In the
            <b>Northeast</b> the strongest measure reaches <b>{ne.rho:+.2f}</b> with
            {int(ne.n_sig)} of 31 clearing significance; in the <b>West</b>,
            <b>{we.rho:+.2f}</b> with {int(we.n_sig)}. Both survive holding the state fixed
            (typical |rho| {ne.median_within_state:.2f} and {we.median_within_state:.2f}), so it is
            not simply "one good state". That is a real, if weak, relationship and the section
            would be wrong to bury it.
        <b>The South — our biggest and best-measured region — has none of it.</b>
            {int(so.sites)} sites across {int(so.states)} states, and the strongest measure is
            <b>{so.rho:+.2f}</b> with a typical |rho| of {so.median_abs:.2f}. The regions where
            something shows up are the three where we hold the fewest sites.
        <b>It is one factor, not fifteen.</b> Read the "measures clearing FDR" column with care:
            the 31 measures are only about <b>{int(ne.components)}–{int(we.components)} independent
            things</b> (they share ~40% of their variance in a single component). What the Northeast
            and West are really showing is one thing — <i>how big the local market is</i> — counted
            many times over. The permutation test above is the honest version of that column: it
            asks whether the whole grid beats chance, once.
        <b>And it leans on one state each.</b> Drop {ne.biggest_state} from the Northeast and the
            strongest correlation falls to {ne.without_state_rho:+.2f}
            (p&nbsp;=&nbsp;{ne.without_state_p:.2f}, n&nbsp;=&nbsp;{int(ne.without_state_n)}); drop
            {we.biggest_state} from the West and it falls to {we.without_state_rho:+.2f}
            (p&nbsp;=&nbsp;{we.without_state_p:.2f}).
        <b>None of it forecasts.</b> The last column is the test that matters: a model trained
            inside a single region, scored on states it has not seen, is <b>negative in every
            region</b> ({so.oos_r2:+.2f} South, {we.oos_r2:+.2f} West,
            {rv[rv.region == 'Midwest'].oos_r2.iloc[0]:+.2f} Midwest; the Northeast has too few
            states to split honestly). Correlation in a sample of 95 and the ability to rank two
            candidate sites are different things, and only the second one buys anything.
    """, accent=WARNING)

    # =============================================================================================
    st.divider()
    st.header("4 · Every market measure, ranked")
    st.markdown("All 31 measures against one wash type, scored three ways. The third column is the "
                "one that matters for site selection.")

    tsel = st.radio("Wash type", list(dd.TARGETS), horizontal=True, key="dem_corrtarget")
    cr = _corr(tsel)
    fam = st.multiselect("Show families", dd.FAMILIES, default=dd.FAMILIES, key="dem_fam")
    crf = cr[cr.family.isin(fam)] if fam else cr

    fc = go.Figure()
    top = crf.head(18).iloc[::-1]
    labels = [f"{n}" for n in top.feature]
    fc.add_bar(y=labels, x=top.rho, orientation="h", name="All sites",
               marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
               hovertemplate="<b>%{y}</b><br>all sites: %{x:+.3f}<extra></extra>")
    fc.add_bar(y=labels, x=top.rho_within_operator, orientation="h",
               name="Within one operator's own sites",
               marker=dict(color=S2, line=dict(width=2, color=SURFACE)),
               hovertemplate="<b>%{y}</b><br>within operator: %{x:+.3f}<extra></extra>")
    fc.add_vline(x=0, line=dict(color=MUTED, width=1))
    st.plotly_chart(style(fc, height=560, barmode="group",
                          xaxis_title="Rank correlation with " + tsel.lower()
                                      + "   (0 = no relationship, ±1 = perfect)",
                          # Explicit ticks: this plotly build ignores `tickformat` on these axes
                          # and falls back to raw float repr ("0.14999999999999997").
                          xaxis=dict(range=[-.25, .25],
                                     tickvals=[-.2, -.1, 0, .1, .2],
                                     ticktext=["−0.20", "−0.10", "0", "+0.10", "+0.20"]),
                          margin=dict(l=230, r=25, t=86, b=55),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    show = crf.head(14)[["feature", "family", "rho", "rho_within_state", "rho_within_operator",
                         "q", "verdict"]].copy()
    show.columns = ["Market measure", "Family", "All sites", "Within state", "Within operator",
                    "q-value", "Verdict"]
    html_table(show.set_index("Market measure"), index_label="Market measure",
               fmt={"All sites": "{:+.3f}", "Within state": "{:+.3f}",
                    "Within operator": "{:+.3f}", "q-value": "{:.3f}"})

    best = cr.iloc[0]
    nsig = int((cr.q > .05).sum())
    nsmall = int((cr.abs_rho < .10).sum())
    callout("Reading this table", f"""
        The strongest of all 31 measures against {tsel.lower()} is <b>{best.feature.lower()}</b>,
            at <b>{best.rho:+.3f}</b>. On a scale where 1.00 is a perfect relationship that is
            close to nothing: the measure accounts for roughly <b>{best.rho**2*100:.1f}%</b> of
            the differences between sites.
        <b>{nsmall} of 31</b> measures come in under ±0.10, detectable on 1,263 sites only
            because 1,263 is a large number, and useless for choosing between two real locations.
        The third column is the decisive one. It asks: <i>within one operator's own portfolio,
            does this measure pick out which of their sites does better?</i> The answer is
            essentially no for every measure. The values collapse toward zero and several flip
            sign. A measure that flips sign once you control for who runs the site was never
            measuring the market; it was measuring which kind of operator builds where.
        The q-value is corrected for the fact that 31 measures were tested at once.
            {nsig} of them do not clear it at all.
    """, accent=S1)

    # =============================================================================================
    st.divider()
    st.header("5 · Give a model everything at once")
    st.markdown("The measures are weak one at a time — but a scoring model uses all of them "
                "together. So: hand a model **all 31 measures** and ask it to forecast a site it "
                "has never seen, in a **state it has never seen**. Two different model types, so "
                "the answer cannot be blamed on the choice of model.")

    oos = _oos()
    fo = go.Figure()
    for i, (mdl, colr) in enumerate([("Gradient-boosted trees", S1), ("Ridge regression", S3)]):
        sub = oos[oos.model == mdl]
        fo.add_bar(x=sub.target, y=sub.r2, name=mdl,
                   marker=dict(color=colr, line=dict(width=2, color=SURFACE)),
                   hovertemplate="<b>%{x}</b><br>" + mdl + "<br>R² = %{y:+.3f}<extra></extra>")
    fo.add_hline(y=0, line=dict(color=MUTED, width=2))
    fo.add_annotation(x=.02, y=0, xref="paper", yshift=12, text="0 = no better than just quoting "
                      "the estate-wide median", showarrow=False,
                      font=dict(size=11, color=MUTED), xanchor="left")
    st.plotly_chart(style(fo, height=380, barmode="group",
                          yaxis_title="Accuracy on unseen markets (R²)",
                          yaxis=dict(range=[-.16, .16],
                                     tickvals=[-.15, -.10, -.05, 0, .05, .10, .15],
                                     ticktext=["−0.15", "−0.10", "−0.05", "0",
                                               "+0.05", "+0.10", "+0.15"]),
                          xaxis=dict(type="category"), margin=dict(t=86),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    ot = oos.copy()
    ot["R² on unseen markets"] = ot.r2
    ot["Typical error"] = ot.mdape
    html_table(ot[["target", "model", "R² on unseen markets", "Typical error"]]
               .rename(columns={"target": "Forecasting", "model": "Model"})
               .set_index("Forecasting"), index_label="Forecasting",
               fmt={"R² on unseen markets": "{:+.3f}", "Typical error": "{:.0f}%"})

    worst = oos.r2.min()
    callout("What this says", f"""
        Every bar is at or below zero. A model given all 31 market measures forecasts a new site's
            volume <b>no better than simply quoting the median of the whole estate</b>, and the
            flexible model does slightly worse than that ({worst:+.3f}), which is what happens
            when a model finds patterns in the training markets that do not exist in new ones.
        In plain terms: the typical forecast is out by
            <b>{oos.query("target=='Total washes'").mdape.min():.0f}%</b> on total washes, and
            that error is what you get with <i>or without</i> the demographics. The trade-area
            score is not adding information.
        Retail washes are the market's best case (a retail wash really is a stranger driving in)
            and even there the model lands at
            {oos.query("target=='Retail washes'").r2.max():+.3f}.
        This is not a claim that markets are irrelevant to a car wash. It is a claim that
            <i>these measures, at this resolution, cannot rank two candidate sites</i>. Every site
            here was chosen by someone who already believed in these measures, which compresses
            the range, but that is exactly the range a new site is picked from.
    """, accent=CRITICAL)

    # =============================================================================================
    st.divider()
    st.header("6 · So what does explain volume?")
    st.markdown("Same question, same honest scoring — each candidate explanation is asked to "
                "predict a site it has not seen.")

    dec = _decomp()
    fd = go.Figure()
    # Four steps, not three: a small *positive* explanation and a *negative* one are different
    # claims, and colouring both red reads as "these two are the same kind of nothing".
    colours = [GOOD if v > .2 else WARNING if v > .03 else SERIOUS if v > 0 else CRITICAL
               for v in dec.r2]
    fd.add_bar(y=dec.explanation[::-1], x=dec.r2[::-1], orientation="h",
               marker=dict(color=colours[::-1], line=dict(width=2, color=SURFACE)),
               customdata=np.stack([dec.detail[::-1], dec.sites[::-1]], axis=-1),
               hovertemplate="<b>%{y}</b><br>explains %{x:.1%} of the differences between sites"
                             "<br>%{customdata[0]}<br><span style='opacity:.7'>%{customdata[1]:.0f}"
                             " sites</span><extra></extra>")
    fd.add_vline(x=0, line=dict(color=MUTED, width=1))
    st.plotly_chart(style(fd, height=330, showlegend=False,
                          xaxis_title="Share of the differences between sites explained",
                          xaxis=dict(tickformat=".0%"), margin=dict(l=250, r=25, t=30, b=55)),
                    width="stretch")

    ops = dec[dec.explanation == "Who operates it"].iloc[0]
    nb = _neighbours()
    bestnb = nb.loc[nb.r2.idxmax()]
    callout("The ranking", f"""
        <b>Who operates the site explains {ops.r2:.0%}</b> of the differences, by a wide margin
            the biggest single factor, and it is measured fairly: each site is predicted from that
            operator's <i>other</i> sites, never from itself, across the {ops.detail}.
        Geography carries a little: the median of a site's ten nearest neighbours explains
            <b>{bestnb.r2:.1%}</b> (those neighbours sit a median of {bestnb.median_km:.0f} km
            away). Small, but real and positive, which the demographics are not. The ground knows
            something the census file does not.
        All 31 demographic measures together explain <b>nothing</b>. They rank below knowing which
            of four regions the site is in.
        The practical read: a forecast for a new site is better anchored on <b>how comparable
            nearby sites actually trade</b> and on <b>who will run it</b> than on a trade-area
            score.
        Caveat on operator: it partly captures things an operator brings that are not skill:
            brand, pricing, tunnel spec, how long the sites have been open. It is not
            {ops.r2:.0%} of pure management quality.
    """, accent=GOOD)

    with st.expander("Does distance matter — how close do the neighbours have to be?"):
        nbt = nb.copy()
        nbt.columns = ["Neighbours used", "Explains", "Rank agreement", "Typical distance (km)",
                       "Sites"]
        html_table(nbt.set_index("Neighbours used"), index_label="Nearest neighbours used",
                   fmt={"Explains": "{:+.3f}", "Rank agreement": "{:+.3f}",
                        "Typical distance (km)": "{:.0f}"})
        st.caption("One neighbour is too noisy to be useful on its own (a single next-door site can "
                   "be anything). Five to ten neighbours is the sweet spot. Beyond twenty the "
                   "'neighbours' are over 100 km away and the signal fades.")

    # =============================================================================================
    st.divider()
    st.header("7 · State by state")

    stt = _states()
    ranked = stt[stt.enough_sites].copy()
    fs = go.Figure()
    fs.add_bar(x=ranked.state, y=ranked.median_washes, name="Median site",
               marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
               customdata=np.stack([ranked.sites, ranked.median_mem_share, ranked.median_pop],
                                   axis=-1),
               hovertemplate="<b>%{x}</b><br>median site: <b>%{y:,.0f} washes</b><br>"
                             "membership share: %{customdata[1]:.0%}<br>"
                             "median trade-area population: %{customdata[2]:,.0f}"
                             "<br><span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                             "<extra></extra>")
    st.plotly_chart(style(fs, height=400, showlegend=False, xaxis=dict(type="category"),
                          yaxis_title="Washes in 2025, median site"), width="stretch")

    stab = ranked[["state", "sites", "median_washes", "median_mem_share", "median_pop",
                   "median_income", "median_competitors"]].copy()
    stab.columns = ["State", "Sites", "Median washes", "Membership share", "Median population",
                    "Median income", "Car washes nearby"]
    html_table(stab.set_index("State"), index_label="State",
               fmt={"Membership share": "{:.0%}", "Median income": "${:,.0f}",
                    "Car washes nearby": "{:.0f}"})

    lo, hi = ranked.iloc[-1], ranked.iloc[0]
    callout("Reading the states", f"""
        The typical <b>{hi.state}</b> site washed <b>{hi.median_washes:,.0f}</b> in 2025; the
            typical <b>{lo.state}</b> site washed <b>{lo.median_washes:,.0f}</b>, a
            <b>{hi.median_washes/lo.median_washes:.1f}×</b> gap between the best and worst of the
            {len(ranked)} states we hold enough sites in to rank.
        That gap does <b>not</b> follow population: across these {len(ranked)} states, median
            trade-area population and median volume correlate at
            <b>{_state_link().query("feature == 'Population in the trade area'").rho.iloc[0]:+.2f}</b>,
            which is no relationship at all. {hi.state} sites sit in trade areas of
            {hi.median_pop:,.0f} people; {lo.state} sites sit in trade areas of
            {lo.median_pop:,.0f}.
        It follows membership. {hi.state} runs at {hi.median_mem_share:.0%} membership,
            {lo.state} at {lo.median_mem_share:.0%}. Across all {len(ranked)} states that link is
            <b>{_state_link().query("family == 'The site itself'").rho.iloc[0]:+.2f}</b>.
        Caveat: states are also operator clusters. A state with three big operators in it inherits
            their performance, so part of this ranking is section 5's finding wearing a different
            hat.
    """, accent=S1)

    # =============================================================================================
    st.divider()
    st.header("8 · All the data")
    st.markdown("Every site in the cohort, sortable, with the market measures behind it.")

    d = _cohort()
    f1, f2, f3 = st.columns(3)
    with f1:
        pick_state = st.multiselect("State", sorted(d.state.dropna().unique()), key="dem_state")
    with f2:
        sort_by = st.selectbox("Sort by", ["Total washes", "Membership washes", "Retail washes",
                                           "Membership share"] + list(dd.FEATURES),
                               key="dem_sort")
    with f3:
        n_show = st.slider("Rows", 10, 200, 40, step=10, key="dem_rows")

    view_df = d[d.state.isin(pick_state)] if pick_state else d
    scol = {"Total washes": "total_washes", "Membership washes": "mem_washes",
            "Retail washes": "ret_washes", "Membership share": "mem_share"}.get(
        sort_by) or dd.FEATURES[sort_by][0]
    view_df = view_df.sort_values(scol, ascending=False).head(n_show)

    tbl = pd.DataFrame({
        "Site": view_df.site, "State": view_df.state,
        "Total washes": view_df.total_washes, "Membership": view_df.mem_washes,
        "Retail": view_df.ret_washes, "Mem. share": view_df.mem_share,
        "Population": view_df["2025 Estimate"],
        "Median income": view_df["Median Household Income"],
        "Traffic/day": view_df.traffic_total,
        "Washes nearby": view_df["Count of Car Wash Competitors"],
    }).set_index("Site")
    html_table(tbl, index_label="Site",
               fmt={"Mem. share": "{:.0%}", "Median income": "${:,.0f}"})

    with st.expander("How this was built"):
        gaps = dd.data_gaps()
        st.markdown(f"""
**Cohort.** The monthly wash panel joined to the site-attribute file on `client_id_1 + site_id`
(the number-first client id), then cut to sites with **all twelve months of 2025 present and
trading** — {h['sites']:,} sites, {h['operators']} operators, {h['states']} states. A twelve-month
gate means every annual total is a real sum; nothing is scaled up from a part-year.

**Measures.** {h['features']} market measures from the site-attribute file, in six families:
population, income, vehicles, competition, retail anchors, traffic.

**Scoring.** Rank correlations (Spearman) rather than straight-line correlations, so one enormous
market cannot drag a result. Every p-value is corrected for testing 31 measures at once
(Benjamini-Hochberg). Model accuracy is measured on **held-out states** — folds are split by state,
so the model is always scored on markets it has never seen.

**Cells read as missing.** A trade area with exactly zero people, or zero vehicles passing all day,
is a failed geocode rather than a real desert. Those cells are treated as missing so they cannot
anchor the bottom of a ranking:
""")
        gt = gaps.copy()
        gt.columns = ["Field", "Sites affected", "Share of cohort"]
        html_table(gt.set_index("Field"), index_label="Field", fmt={"Share of cohort": "{:.1%}"})
        st.caption("This section shares no data with sections ① or ②; each stands on its own file.")
