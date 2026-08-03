"""
Section — Demographics. Does the market a site sits in explain how much it washes?

Vertical scroll, same as the other sections. The order is the order of the argument:

  1  the estate on a map, with any feature laid over it — the visual version of the question
  2  pick a feature, see volume by fifths — the exhibit that needs no statistics
  3  every feature ranked, three ways (raw, within state, within operator)
  4  give a model all 31 features at once and score it on markets it has never seen
  5  so what *does* explain volume
  6  the whole table
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import demographics_data as dd
from ui import (CRITICAL, GOOD, GRID, MUTED, S1, S2, S3, SERIOUS, SURFACE, WARNING,
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


def _k(v: float) -> str:
    return f"{v/1000:,.0f}k"


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
            <b>{strongest.feature.lower()}</b> at {strongest.rho:+.2f} — and even that does not
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
    st.header("2 · Sort the sites by any market measure")
    st.markdown("Pick a measure. Every site is sorted on it and split into five equal groups — the "
                "lowest fifth through the highest fifth. The bars are what those groups actually "
                "washed in 2025. **A real driver makes the bars climb.**")

    c1, c2 = st.columns([2, 1])
    with c1:
        qf = st.selectbox("Market measure", list(dd.FEATURES), key="dem_qfeat")
    with c2:
        qt = st.selectbox("Wash type", list(dd.TARGETS), key="dem_qtarget")

    q = dd.quintiles(qf, qt)
    qown = dd.own_fact_quintiles("Membership customers", qt)

    fq = go.Figure()
    fq.add_bar(x=q.bucket, y=q.median_washes, name=qf,
               marker=dict(color=S2, line=dict(width=2, color=SURFACE)),
               customdata=np.stack([q.sites, q.feature_lo, q.feature_hi, q.p25, q.p75], axis=-1),
               hovertemplate="<b>%{x}</b> on " + qf.lower() + "<br>"
                             "median: <b>%{y:,.0f} washes</b><br>"
                             "middle half of the group: %{customdata[3]:,.0f} – %{customdata[4]:,.0f}"
                             "<br>range of this group: %{customdata[1]:,.0f} – %{customdata[2]:,.0f}"
                             "<br><span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                             "<extra></extra>")
    # The contrast series, on the identical axis: sort the same sites on the one input that is not
    # a market measure at all — how many members the site has. Nobody has to rescale in their head.
    fq.add_bar(x=qown.bucket, y=qown.median_washes, name="For contrast: membership customers",
               marker=dict(color=S1, line=dict(width=2, color=SURFACE)), opacity=.55,
               hovertemplate="<b>%{x}</b> on membership customers<br>"
                             "median: <b>%{y:,.0f} washes</b><extra></extra>")
    st.plotly_chart(style(fq, height=400, barmode="group",
                          yaxis_title=f"{qt} in 2025 (median site)",
                          xaxis=dict(type="category", title=""), margin=dict(t=86),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    sp, mono = q.attrs["spread"], q.attrs["monotonic"]
    tone = GOOD if abs(np.log(sp)) > .35 else (WARNING if abs(np.log(sp)) > .18 else CRITICAL)
    verdict = ("a real lever" if abs(np.log(sp)) > .35 else
               "a nudge at best" if abs(np.log(sp)) > .18 else "flat — not a lever")
    callout(f"Highest fifth vs lowest fifth: {sp:.2f}× — {verdict}", f"""
        Sites in the highest fifth on <b>{qf.lower()}</b> washed a median of
            <b>{q.median_washes.iloc[-1]:,.0f}</b> in 2025; sites in the lowest fifth washed
            <b>{q.median_washes.iloc[0]:,.0f}</b>. That is a ratio of <b>{sp:.2f}×</b>, and the
            five bars {'climb in order' if mono else 'do <b>not</b> climb in order'} —
            {'consistent with a real effect' if mono else 'so even the small gap that is there is not a clean trend'}.
        For scale, the gap between our busiest and quietest sites is <b>{h['spread']:.1f}×</b>.
            A measure that moves the median by {sp:.2f}× cannot account for it.
        The blue bars are {int(qown.sites.sum()):,} sites sorted on the one input that is
            not a market measure at all — how many members the site has. Top fifth vs bottom
            fifth: <b>{qown.attrs['spread']:.1f}×</b>, climbing at
            {'every single step' if qown.attrs['monotonic'] else 'almost every step'}. That is the
            shape a real driver makes.
    """, accent=tone)

    # =============================================================================================
    st.divider()
    st.header("3 · Every market measure, ranked")
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
        <b>{nsmall} of 31</b> measures come in under ±0.10 — detectable on 1,263 sites only
            because 1,263 is a large number, and useless for choosing between two real locations.
        The third column is the decisive one. It asks: <i>within one operator's own portfolio,
            does this measure pick out which of their sites does better?</i> The answer is
            essentially no for every measure — the values collapse toward zero and several flip
            sign. A measure that flips sign once you control for who runs the site was never
            measuring the market; it was measuring which kind of operator builds where.
        The q-value is corrected for the fact that 31 measures were tested at once.
            {nsig} of them do not clear it at all.
    """, accent=S1)

    # =============================================================================================
    st.divider()
    st.header("4 · Give a model everything at once")
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
            volume <b>no better than simply quoting the median of the whole estate</b> — and the
            flexible model does slightly worse than that ({worst:+.3f}), which is what happens
            when a model finds patterns in the training markets that do not exist in new ones.
        In plain terms: the typical forecast is out by
            <b>{oos.query("target=='Total washes'").mdape.min():.0f}%</b> on total washes, and
            that error is what you get with <i>or without</i> the demographics. The trade-area
            score is not adding information.
        Retail washes are the market's best case — a retail wash really is a stranger driving in —
            and even there the model lands at
            {oos.query("target=='Retail washes'").r2.max():+.3f}.
        This is not a claim that markets are irrelevant to a car wash. It is a claim that
            <i>these measures, at this resolution, cannot rank two candidate sites</i>. Every site
            here was chosen by someone who already believed in these measures, which compresses
            the range — but that is exactly the range a new site is picked from.
    """, accent=CRITICAL)

    # =============================================================================================
    st.divider()
    st.header("5 · So what does explain volume?")
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
        <b>Who operates the site explains {ops.r2:.0%}</b> of the differences — by a wide margin
            the biggest single factor, and it is measured fairly: each site is predicted from that
            operator's <i>other</i> sites, never from itself, across the {ops.detail}.
        Geography carries a little: the median of a site's ten nearest neighbours explains
            <b>{bestnb.r2:.1%}</b> (those neighbours sit a median of {bestnb.median_km:.0f} km
            away). Small, but real and positive, which the demographics are not — the ground knows
            something the census file does not.
        All 31 demographic measures together explain <b>nothing</b>. They rank below knowing which
            of four regions the site is in.
        The practical read: a forecast for a new site is better anchored on <b>how comparable
            nearby sites actually trade</b> and on <b>who will run it</b> than on a trade-area
            score.
        Caveat on operator: it partly captures things an operator brings that are not skill —
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
    st.header("6 · State by state")

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
            typical <b>{lo.state}</b> site washed <b>{lo.median_washes:,.0f}</b> — a
            <b>{hi.median_washes/lo.median_washes:.1f}×</b> gap between the best and worst of the
            {len(ranked)} states we hold enough sites in to rank.
        That gap does <b>not</b> follow population: across these {len(ranked)} states, median
            trade-area population and median volume correlate at
            <b>{_state_link().query("feature == 'Population in the trade area'").rho.iloc[0]:+.2f}</b>
            — no relationship at all. {hi.state} sites sit in trade areas of
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
    st.header("7 · All the data")
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
