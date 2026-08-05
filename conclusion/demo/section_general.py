"""
Section — General. The estate-wide facts the other sections lean on.

Two things live here because they are facts about car washes, not about tunnels or proformas:
how a new site ramps up, and where the business actually is.

The ramp maths still lives in `tunnel_data.ramp_*` (it reads the same monthly panel and section ①
uses it to carry young sites forward); the map comes from `general_data.py`. Both are
Streamlit-free, so the notebook imports the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import general_data as gd
import tunnel_data as td
from ui import DARK, GRID, INK, INK2, MUTED, S1, S2, S3, SURFACE, callout, html_table, style


@st.cache_data(show_spinner=False)
def _ramp_curve() -> pd.DataFrame:
    return td.ramp_curve()


@st.cache_data(show_spinner=False)
def _ramp_validation() -> pd.DataFrame:
    return td.ramp_validation()


@st.cache_data(show_spinner=False)
def _sites() -> pd.DataFrame:
    return gd.site_map()


@st.cache_data(show_spinner=False)
def _states() -> pd.DataFrame:
    return gd.state_totals()


@st.cache_data(show_spinner=False)
def _mix() -> pd.DataFrame:
    return gd.mix_by_year()


@st.cache_data(show_spinner=False)
def _basis() -> dict:
    return gd.same_site_basis()


@st.cache_data(show_spinner=False)
def _mix_states() -> pd.DataFrame:
    return gd.mix_by_state_year()


def render() -> None:
    mh = gd.map_headline()

    st.markdown("<div class='kicker'>Evidence pack · background</div>", unsafe_allow_html=True)
    st.title("General")
    st.markdown("The two facts about the estate that everything else rests on: **how a new wash "
                "fills up**, and **where the business actually is**.")

    # =============================================================================================
    st.divider()
    st.header("1 · How a car wash ramps up")

    rc = _ramp_curve()
    anchor = rc.attrs["anchor_year"]
    st.caption(f"Median washes by operating year, on the **{rc.attrs['sites']} sites** with at "
               f"least {anchor} complete operating years. Each site's real opening date lets "
               "operating years be cut exactly — a year counts only if all 12 months are there, so "
               "nothing is scaled or estimated. Shaded band is the middle half of sites.")

    figr = go.Figure()
    figr.add_scatter(x=list(rc.operating_year) + list(rc.operating_year[::-1]),
                     y=list(rc.p75) + list(rc.p25[::-1]), fill="toself",
                     fillcolor="rgba(57,135,229,0.15)", line=dict(width=0),
                     hoverinfo="skip", name="middle half of sites")
    figr.add_scatter(x=rc.operating_year, y=rc["median"], mode="lines+markers", name="median site",
                     line=dict(color=S1, width=3),
                     marker=dict(size=11, line=dict(width=2, color=SURFACE)),
                     customdata=np.stack([rc.sites, rc.share_of_final], axis=-1),
                     hovertemplate="Operating year %{x}<br><b>%{y:,.0f} washes</b><br>"
                                   f"= %{{customdata[1]:.0%}} of its year-{anchor} volume<br>"
                                   "<span style='opacity:.7'>%{customdata[0]:.0f} sites"
                                   "</span><extra></extra>")
    st.plotly_chart(style(figr, height=420, xaxis_title="Operating year",
                          yaxis_title="Washes per year", xaxis=dict(dtick=1),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")

    sh = rc.set_index("operating_year").share_of_final
    callout("What this shows", f"""
      <b>How fast does a new wash fill up?</b> In its first year it does <b>{sh.get(1):.0%}</b> of
        the business it will eventually do. By its second year it is at <b>{sh.get(2):.0%}</b>,
        essentially all of it.
      <b>And then it stops.</b> The later years sit within a few points of year 2. The growth is one
        jump in the first twelve months, not a five-year climb.
      <b>Why that matters.</b> You do not have to wait five years to know what a site is worth. Its
        second full year is already telling you.
    """)

    rv = _ramp_validation()
    st.markdown(f"**How reliable is that?** Every one of these {rc.attrs['sites']} sites, with its "
                f"year-{anchor} volume predicted from each earlier year and checked against what "
                "actually happened.")
    vt = rv[["from_operating_year", "sites", "mdape_naive", "mdape_ramp", "bias"]].copy()
    vt.columns = ["From operating year", "Sites", "Error if you assume no growth",
                  "Error using the ramp", "Bias"]
    vt.index = range(1, len(vt) + 1)
    html_table(vt, fmt={"Error if you assume no growth": "{:,.1f}%",
                        "Error using the ramp": "{:,.1f}%", "Bias": "{:,.2f}×"})
    v1 = rv[rv.from_operating_year == 1].iloc[0]
    vlast = rv.iloc[-1]
    callout("What this shows", f"""
      <b>Can we trust a call made this early?</b> We tested it on sites where we already know the
        answer: hide the later years, predict, then check. From <b>one year</b> of trading the
        long-run number lands within <b>{v1.mdape_ramp:.0f}%</b>.
      <b>By year {vlast.from_operating_year:.0f} it is within
        {vlast.mdape_ramp:.0f}%</b>. Most of what there is to learn about a site is known by
        the end of its second year.
      <b>Why that matters.</b> A site can be given a defensible long-run number after its first full
        year, instead of carrying a five-year projection nobody has tested.
    """, S3)

    # =============================================================================================
    st.divider()
    st.header("2 · Where the washing actually happens")
    st.caption(f"Every one of the **{mh['n_sites']:,} sites** we hold wash data for. Everything "
               "here is **per site, per year** — never a state total. We are not the only operator "
               "in any of these states, so a state total mostly measures how many sites we happen "
               "to own there, not how strong the market is.")

    view = st.radio("View", ["Every site", "By state"], horizontal=True, index=0)
    s = _sites()
    # A single-hue ORANGE ramp rather than the default blue: blue is already the section's series
    # colour, and a warm ramp reads as intensity on a map without anyone having to check the key.
    # One hue only, no rainbow.
    #
    # Direction is the conventional one and is the SAME in both themes — pale = fewer washes,
    # deep = more. The deep end stops at a saturated burnt orange rather than going near-black, so
    # the busiest sites stay legible on the dark surface instead of sinking into it.
    SCALE = [[0.0, "#fde3cf"], [0.25, "#f9bf90"], [0.5, "#f2914f"],
             [0.75, "#dd6524"], [1.0, "#a8410f"]]

    if view == "Every site":
        # size on sqrt so a 300k site does not swamp a 30k one; colour on a single-hue ramp
        figm = go.Figure(go.Scattergeo(
            lat=s.lat, lon=s.lon, mode="markers",
            marker=dict(size=np.sqrt(s.washes_per_year) / 42 + 3,
                        color=s.washes_per_year, colorscale=SCALE,
                        cmin=0, cmax=float(s.washes_per_year.quantile(0.95)),
                        line=dict(width=0.4, color=SURFACE), opacity=0.85,
                        colorbar=dict(title="Washes<br>per year", thickness=12,
                                      tickformat=",.0f")),
            customdata=np.stack([s.operator, s.state, s.washes_per_year, s.years], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                          "<b>%{customdata[2]:,.0f}</b> washes a year<br>"
                          "<span style='opacity:.7'>%{customdata[3]:.1f} years of data</span>"
                          "<extra></extra>"))
    else:
        stt = _states()
        stt = stt[stt.enough_sites]          # a 4-site state's median is noise, not a signal
        figm = go.Figure(go.Choropleth(
            locations=stt.state, locationmode="USA-states", z=stt.median_site,
            colorscale=SCALE, marker_line_color=SURFACE, marker_line_width=0.6,
            colorbar=dict(title="Typical site<br>washes/yr", thickness=12, tickformat=",.0f"),
            customdata=np.stack([stt.sites, stt.p25_site, stt.p75_site], axis=-1),
            hovertemplate="<b>%{location}</b><br>Typical site: <b>%{z:,.0f}</b> washes a year<br>"
                          "Middle half run %{customdata[1]:,.0f}–%{customdata[2]:,.0f}<br>"
                          "<span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                          "<extra></extra>"))

    figm.update_geos(scope="usa", bgcolor=SURFACE,
                     landcolor="#242422" if DARK else "#f0efec",
                     lakecolor=SURFACE, subunitcolor=GRID, countrycolor=GRID)
    st.plotly_chart(style(figm, height=560, margin=dict(l=0, r=0, t=10, b=0)), width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sites on the map", f"{mh['n_sites']:,}")
    m2.metric("Typical site", f"{mh['median_site']:,.0f}", "washes a year", delta_color="off")
    m3.metric("Busiest state, per site", mh["busiest_state"],
              f"{mh['busiest_state_site']:,.0f} a year", delta_color="off")
    m4.metric("Most sites", mh["most_sites_state"],
              f"{mh['most_sites_n']} sites · {mh['most_sites_median']:,.0f} each",
              delta_color="off")

    stt = _states()
    stt = stt[stt.enough_sites].head(12).copy()
    tbl = stt[["state", "sites", "median_site", "p25_site", "p75_site"]]
    tbl.columns = ["State", "Sites", "Typical site", "Quieter quarter", "Busier quarter"]
    tbl.index = range(1, len(tbl) + 1)
    html_table(tbl, fmt={"Sites": "{:,.0f}", "Typical site": "{:,.0f}",
                         "Quieter quarter": "{:,.0f}", "Busier quarter": "{:,.0f}"})
    st.caption(f"Ranked by what a typical site does, not by state total. States with fewer than "
               f"5 sites are left out — {mh['n_ranked']} of {mh['n_states']} qualify.")

    callout("What this shows", f"""
      <b>Where we have the most sites is not where sites are busiest.</b>
        <b>{mh['most_sites_state']}</b> has by far the most ({mh['most_sites_n']}), but a typical
        site there does <b>{mh['most_sites_median']:,.0f}</b> washes a year, about the same as
        everywhere else. <b>{mh['busiest_state']}</b> tops the per-site ranking on just
        {mh['busiest_state_n']} sites.
      <b>A wash is a wash almost anywhere.</b> Across the {mh['n_ranked']} states with enough sites
        to judge, the typical site runs from <b>{mh['quietest_state_site']:,.0f}</b> to
        <b>{mh['busiest_state_site']:,.0f}</b> a year, a spread of only about
        <b>{mh['spread']:.1f}×</b> between the best state and the worst.
      <b>The real variation is between sites, not between states.</b> Across the whole estate the
        middle half of sites run <b>{mh['p25_site']:,.0f}</b> to <b>{mh['p75_site']:,.0f}</b> a
        year, a wider gap inside a single state than between states. Which state you build in
        matters far less than which site you pick.
    """, S3)

    with st.expander("Every site, as a table"):
        show = s[["operator", "state", "washes_per_year", "washes_per_month", "years"]].copy()
        show.columns = ["Operator", "State", "Washes a year", "Washes a month", "Years of data"]
        show.index = range(1, len(show) + 1)
        html_table(show.head(50), fmt={"Washes a year": "{:,.0f}", "Washes a month": "{:,.0f}",
                                       "Years of data": "{:,.1f}"})
        st.caption(f"Busiest 50 of {len(s):,} sites.")
        st.download_button("Download every site (CSV)", s.to_csv(index=False),
                           "sites_by_wash_volume.csv", "text/csv")

    # =============================================================================================
    st.divider()
    st.header("3 · Membership or drive-up — which half of the business is this?")
    st.caption("Every wash in the panel is either sold to a member on a plan or to somebody who "
               "drove up and paid once. The split is not a detail: it decides how a site behaves "
               "when a competitor opens (§⑤), how much a campaign can move (§③), and what a "
               "forecast is actually forecasting.")

    mx = _mix()
    mxh = gd.mix_headline()

    fig = go.Figure()
    fig.add_scatter(x=mx.year, y=mx.share, mode="lines+markers", name="Every site trading that year",
                    line=dict(color=S1, width=3),
                    marker=dict(size=10, line=dict(width=2, color=SURFACE)),
                    customdata=np.stack([mx.sites_all, mx.washes], axis=-1),
                    hovertemplate="<b>%{y:.0%}</b> of washes sold on a plan<br>"
                                  "<span style='opacity:.7'>%{customdata[0]:,.0f} sites · "
                                  "%{customdata[1]:,.0f} washes</span><extra></extra>")
    fig.add_scatter(x=mx.year, y=mx.share_same_sites, mode="lines+markers",
                    name=f"The same {mxh['n_same_sites']} sites throughout",
                    line=dict(color=S3, width=3, dash="dash"),
                    marker=dict(size=10, symbol="diamond", line=dict(width=2, color=SURFACE)),
                    hovertemplate="<b>%{y:.0%}</b> on a plan, same sites<extra></extra>")
    fig.add_hline(y=0.5, line=dict(color=GRID, width=1))
    for col, colour, lbl in ((mx.share, S1, "all sites"),
                             (mx.share_same_sites, S3, "same sites")):
        fig.add_annotation(x=float(mx.year.iloc[-1]), y=float(col.iloc[-1]), text=lbl,
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=colour, size=11))
    st.plotly_chart(style(fig, height=400, xaxis_title="Calendar year",
                          yaxis_title="Share of washes sold to members",
                          yaxis=dict(tickformat=".0%", range=[0, 0.75]),
                          xaxis=dict(dtick=1, range=[mx.year.min() - 0.2, mx.year.max() + 0.9]),
                          margin=dict(r=90),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")
    bs = _basis()
    st.caption(
        f"**What the {bs['n']} are.** A site counts in a calendar year if it traded at least "
        f"**{bs['min_months']} months** of it. The dashed line keeps only the sites that clear that "
        f"bar in **every one of the {bs['n_years']} years** ({bs['years'][0]}–{bs['years'][-1]}) — "
        f"{bs['n']} of {bs['eligible']:,}, or {bs['share_of_panel']:.0%} of the panel. "
        f"{int(mx.year.max())} is a half year (the panel ends in June); a *share* is still readable "
        "on six months, a total would not be.")
    st.caption(
        f"**They are the oldest sites, and survivors.** Every one of the {bs['n']} has "
        f"{bs['cohort_year']} as its first qualifying year — {bs['n']} of the "
        f"{bs['opened_that_year']} sites trading back then are still here "
        f"({bs['survival']:.0%}). So the dashed line is what happened to the "
        f"**{bs['cohort_year']}-and-earlier estate**; it says nothing about sites opened since, "
        "which is the point of having the pooled line beside it.")

    with st.expander(f"Does the answer depend on the {bs['n']} being the sites it is?"):
        sens = bs["sensitivity"].copy()
        sens.columns = ["Window", "Years", "Sites in the balanced set",
                        "Membership share, first year", "…last year", "Move"]
        html_table(sens.set_index("Window"), index_label="Window",
                   fmt={"Membership share, first year": "{:.1%}", "…last year": "{:.1%}",
                        "Move": "{:+.1f} pp"})
        st.caption(
            f"Shortening the window more than quadruples the sample — {bs['n']} sites becomes "
            f"{int(sens['Sites in the balanced set'].iloc[-1]):,} — and the same-site move stays "
            f"between **+{sens['Move'].min():.0f}** and **+{sens['Move'].max():.0f} points**, "
            f"always far below the pooled line's **+{bs['pooled_move_pp']:.0f}**. The count is very "
            "sensitive to the window; the conclusion is not. Note the shortest windows start at an "
            "already-high membership share — later cohorts open membership-led, which is the "
            "compositional effect the two lines exist to separate.")

    comp = (mxh["last_share"] - mxh["first_share"]) - (mxh["last_same"] - mxh["first_same"])
    callout("What this shows", f"""
      <b>The business flipped.</b> In {mxh['first_year']}, <b>{mxh['first_share']:.0%}</b> of washes
        were sold on a plan; by {mxh['last_year']} it is <b>{mxh['last_share']:.0%}</b>. Drive-up
        went from the majority of the business to the minority of it.
      <b>But half of that move is not conversion. It is arithmetic.</b> Hold the sites fixed and
        the same {mxh['n_same_sites']} washes go from <b>{mxh['first_same']:.0%}</b> to only
        <b>{mxh['last_same']:.0%}</b>. The remaining <b>{comp*100:.0f} points</b> is the panel
        filling up with newer sites that were membership-led from the day they opened.
      <b>Which is the more useful number depends on the question.</b> For "what does our estate look
        like today", the pooled line. For "what happens to a site we already own", the same-site
        line. An existing wash converts its book at roughly
        <b>{(mxh['last_same'] - mxh['first_same']) / (mxh['last_year'] - mxh['first_year']) * 100:.1f}
        points a year</b>, not the {(mxh['last_share'] - mxh['first_share']) * 100 / (mxh['last_year'] - mxh['first_year']):.1f} the pooled line implies.
    """, S3)

    st.markdown("#### And by state")
    st.caption("Membership share of washes, state by state, year by year. A state-year needs at "
               "least four sites to be drawn — below that the 'state' is one operator's pricing "
               "policy, and operators differ far more than states do.")

    sy = _mix_states()
    piv = sy.pivot(index="state", columns="year", values="share")
    n_piv = sy.pivot(index="state", columns="year", values="sites")
    last_col = piv.columns.max()
    piv = piv.sort_values(last_col, ascending=True)          # ascending: heat map draws bottom-up
    n_piv = n_piv.reindex(piv.index)

    # Single hue, light → dark, one direction in both themes: pale = drive-up, deep = membership.
    # Not a rainbow and not a diverging pair — this is a magnitude, and 50% is not a meaningful
    # midpoint for it.
    RAMP = [[0.0, "#e3edfa"], [0.25, "#a8c8f0"], [0.5, "#5f97e0"], [0.75, "#2a6fc4"],
            [1.0, "#14427a"]]
    figh = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[str(c) for c in piv.columns], y=piv.index.tolist(),
        colorscale=RAMP, zmin=0, zmax=1, xgap=2, ygap=2,
        customdata=n_piv.to_numpy(),
        colorbar=dict(title="On a plan", thickness=12, tickformat=".0%"),
        hovertemplate="<b>%{y}</b> · %{x}<br><b>%{z:.0%}</b> of washes on a plan<br>"
                      "<span style='opacity:.7'>%{customdata:.0f} sites</span><extra></extra>"))
    st.plotly_chart(style(figh, height=max(320, 19 * len(piv) + 90),
                          xaxis_title="", yaxis_title="",
                          xaxis=dict(showgrid=False, side="top"), yaxis=dict(showgrid=False),
                          margin=dict(l=52, r=10, t=44, b=20)), width="stretch")

    movers = (sy.sort_values("year").groupby("state")
                .agg(first=("share", "first"), last=("share", "last"),
                     sites=("sites", "last"), years=("year", "nunique")).reset_index())
    movers = movers[movers.years >= 3]
    movers["change"] = movers.last - movers.first
    top = pd.concat([movers.nlargest(5, "change"), movers.nsmallest(5, "change")])
    top = top[["state", "sites", "first", "last", "change"]]
    top.columns = ["State", "Sites", "First year", "Latest", "Change"]
    top.index = range(1, len(top) + 1)
    html_table(top, fmt={"Sites": "{:,.0f}", "First year": "{:.0%}", "Latest": "{:.0%}",
                         "Change": "{:+.0%}"})

    callout("What this shows", f"""
      <b>Every state moved the same way.</b> The five biggest risers and the five smallest are in
        the table above and there is no state that meaningfully went backwards. The largest fall
        across {mxh['n_states']} states is <b>{mxh['faller_change']*100:+.0f} points</b>
        ({mxh['faller']}), against <b>{mxh['riser_change']*100:+.0f}</b> at the top
        ({mxh['riser']}).
      <b>The spread between states is wide and it is not geography.</b> Latest year,
        <b>{mxh['top_state']}</b> sits at <b>{mxh['top_share']:.0%}</b> and
        <b>{mxh['bottom_state']}</b> at <b>{mxh['bottom_share']:.0%}</b>. Neighbouring states sit
        many points apart, which a regional customer preference would not do. It tracks which
        operators happen to hold sites there (§④: the operator explains 39% of volume, the trade
        area none).
      <b>Read the pale rows as an opportunity, carefully.</b> A low-membership state is either a
        book that has not been converted yet or an operator who has chosen not to. §③ is the
        cautionary half of that: campaigns move the number by about +7%, not by the 20 points this
        chart makes look routine.
    """, S3)
