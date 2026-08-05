"""
Section ③ — promotional campaigns: do they work, and who pays for the lift?

A demo surface: charts, the findings that come out of them, and the controls a reviewer can drive.
All the maths lives in `campaign_data.py` (Streamlit-free, shared with `book_v4_revolt.ipynb`).

The section is deliberately shaped as an argument that turns on itself:
  1-3  what the raw event studies say — promotions look powerful;
  4    what happens when the same numbers are measured against a counterfactual — most of the
       effect is the site's opening ramp, and one placebo test fails outright;
  5    the lifecycle curve that explains why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import campaign_data as cd
from ui import (BORDER, CRITICAL, DARK, GOOD, GRID, INK, INK2, MUTED, S1, S2, S3, SERIOUS,
                SURFACE, WARNING, callout, html_table, style)

# The house palette carries three categorical hues. This section needs a fourth for "the gap"
# (treated − control), which is a distinct entity from either side of it. Both steps were run
# through the palette validator against their own surface: all six checks pass in both modes.
S4 = "#a97ae0" if DARK else "#8e58c8"

FOCAL, NEIGHBOUR, SHARE = S1, S2, S4
TREATED, CONTROL, GAP = S1, S2, S4

THRESHOLDS = [1.1, 1.15, 1.2, 1.3, 1.5]
CUTOFFS = [0, 6, 12, 15, 18, 24]

# The seven metrics the focal event study tracks, in the order the notebook plots them.
FOCAL_METRICS = [
    ("total_income_norm", "Revenue", S1),
    ("true_opex_norm", "OPEX", S2),
    ("ASP_mem_norm", "ASP — membership", S3),
    ("ASP_ret_norm", "ASP — retail", S4),
    ("total_washes_norm", "Wash count — total", S1),
    ("mem_wash_count_norm", "Wash count — membership", S3),
    ("ret_wash_count_norm", "Wash count — retail", S2),
]

# Shown by default in "1 · What a promotion looks like from the inside"; the rest are opt-in via
# the multiselect.
DEFAULT_FOCAL_METRICS = ["Revenue", "OPEX", "Wash count — membership", "Wash count — retail"]

SPILLOVER_PANELS = [
    ("ret_wash_count_norm", "Retail wash count"),
    ("total_income_norm", "Revenue"),
    ("mem_wash_count_norm", "Membership wash count"),
]

RAMP_PANELS = [
    ("opex", "OPEX ($ per month)", "$,.0f", False),
    ("mem_wash_count", "Membership washes per month", ",.0f", False),
    ("market_share", "Share of its own 20 km market (%)", ".0f", True),
]


# ==================================================================================================
# Pipeline — one cached build, so every panel on the page is the same universe
# ==================================================================================================
@st.cache_resource(show_spinner="Detecting campaigns and building the counterfactual…")
def _pipeline(threshold: float) -> dict:
    data = cd.load()
    site_pnl = cd.site_frame(data)
    _, spikes = cd.detect_opex_spikes(data, threshold=threshold)
    dist = cd.build_distance_matrix(site_pnl)

    events = cd.build_spike_event_study(data, spikes)
    nbr = cd.build_neighbor_event_study(data, spikes, dist)
    mshare = cd.build_market_share_panel(data, spikes, dist)
    camps = cd.cluster_campaigns(spikes)

    cf = cd.Counterfactual(data, spikes)
    seasonal = cd.seasonal_index(cf)
    aged = cf.age_of(camps)
    sweep = cd.age_sweep(cf, camps, thresholds=tuple(CUTOFFS))
    passing = sweep.loc[sweep["placebo_passes"], "min_age"]
    cutoff = int(passing.min()) if len(passing) else CUTOFFS[-1]

    panels = {c: cf.build(aged[aged["age_months"] >= c]) for c in CUTOFFS}
    metric_panels = {m: cf.build(camps, metric=m) for m in
                     ["total_washes", "ret_wash_count", "mem_wash_count", "mem_purchase_count"]}
    alt_events, alt_map = cd.expenses_only_events(data)

    return {
        "data": data, "site_pnl": site_pnl, "spikes": spikes, "dist": dist,
        "events": events, "nbr": nbr, "mshare": mshare, "camps": camps, "aged": aged,
        "stats": cd.compute_spillover_stats(events, nbr, mshare),
        "state_mix": cd.focal_state_mix(nbr, site_pnl),
        "spend": cd.campaign_spend_table(camps),
        "snapshot": cd.campaign_snapshot(data, camps),
        "ramp": cd.site_ramp(data, dist),
        "cf": cf, "seasonal": seasonal, "sweep": sweep, "cutoff": cutoff,
        "panels": panels, "metric_panels": metric_panels,
        "deseason": cd.deseasonalised_naive(cf, camps, seasonal),
        "start_months": cd.campaign_start_months(camps),
        "near_far": cd.near_vs_far(cf, camps),
        "roi": cd.roi_repricing(cf, panels[0], camps),
        "cogs_corr": cd.cogs_revenue_correlation(data),
        "placebo": cf.report(cf.build(camps, shift=-9, hi=3),
                             [("pre-trend −3..−1", range(-3, 0)),
                              ("FAKE effect +1..+3", range(1, 4))]),
        "robust": {
            "Region only (site age ignored)": cf.report(cf.build(camps, match_age=False)),
            "No matching at all": cf.report(cf.build(camps, match_region=False, match_age=False)),
            "Trigger = fixed expenses, COGS excluded":
                cf.report(cf.build(alt_events, spike_map=alt_map)),
        },
        "n_alt_events": len(alt_events),
    }


# ==================================================================================================
# Chart helpers
# ==================================================================================================
def _rgba(hex_col: str, alpha: float) -> str:
    h = hex_col.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _band(fig, x, lo, hi, color, row=None, col=None, alpha=0.13, name=None):
    """The IQR (or CI) ribbon: same hue as its line, no outline, never in the legend."""
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]), y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor=_rgba(color, alpha), line=dict(width=0),
        hoverinfo="skip", showlegend=False, name=name or ""), **kw)


def _line(fig, x, y, color, name, row=None, col=None, show_legend=True, dash=None,
          hovertemplate=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=name, legendgroup=name, showlegend=show_legend,
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=8, color=color, line=dict(width=2, color=SURFACE)),
        hovertemplate=hovertemplate), **kw)


def _sub(fig, rows: int, height: int, x_title: str, **kw):
    """House styling for a subplot grid — `style()` only reaches axis 1."""
    style(fig, height=height, showlegend=kw.pop("showlegend", True), **kw)
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_xaxes(title_text=x_title, row=rows, col=1)
    for ann in fig.layout.annotations:                       # subplot titles
        ann.font.update(color=INK, size=12.5)
        ann.update(x=0, xanchor="left")

    # A subplot title is an annotation pinned to the top of its row's domain, and the loop above
    # also pins it to x=0 — which is exactly where a horizontal legend starts. Any caller passing
    # a legend at y≈1.05 therefore lands it on top of the row-1 title. Give the legend its own band
    # above the titles here rather than trusting each call site to guess a y that clears them.
    leg = fig.layout.legend
    if fig.layout.showlegend and leg.orientation == "h" and (leg.y or 0) > 1:
        leg.update(y=1.115, yanchor="bottom")
        fig.layout.margin.t = max(fig.layout.margin.t or 60, 108)
    return fig


def _event_marks(fig, rows: int, ref_y: float | None = 1.0, event_label="spike"):
    """The two reference marks every event study carries: the baseline, and month zero."""
    for r in range(1, rows + 1):
        if ref_y is not None:
            fig.add_hline(y=ref_y, line=dict(color=MUTED, dash="dash", width=1), row=r, col=1)
        fig.add_vline(x=0, line=dict(color=CRITICAL, dash="dot", width=1.5), row=r, col=1)


def _pct_cell(v, digits=1, suffix="%"):
    return "—" if pd.isna(v) else f"{v:+,.{digits}f}{suffix}"


# ==================================================================================================
def render() -> None:
    st.markdown("<div class='kicker'>Marketing spend · evidence pack</div>", unsafe_allow_html=True)
    st.title("Do promotional campaigns work?")

    with st.sidebar:
        threshold = st.select_slider(
            "Campaign trigger — OPEX above trailing 6-month mean",
            THRESHOLDS, value=1.2, format_func=lambda v: f"{v:.2f}×",
            help="There is no campaign table in this business. A campaign is inferred from a jump "
                 "in a site's own operating spend. A lower bar finds more, weaker events.")

    P = _pipeline(threshold)
    st_, camps, cf = P["stats"], P["camps"], P["cf"]
    n_sites = P["data"].site_key.nunique()
    n_spike_sites = P["spikes"].site_key.nunique()

    st.markdown(
        f"**{len(P['spikes'])} months of unusual spend** across **{n_spike_sites} of {n_sites} "
        f"sites**, merged into **{len(camps)} campaigns**. Every number below is measured from "
        "them — and section 4 asks whether any of it survives a counterfactual.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Campaigns detected", f"{len(camps)}")
    m2.metric("Median campaign spend", f"${camps.total_incremental_opex.median():,.0f}")
    m3.metric("Revenue lift, naive", _pct_cell(P["deseason"]["raw"]),
              help="Months +1..+3 against the site's own pre-campaign average — the number every "
                   "section before 4 reports.")
    _row = P["sweep"][P["sweep"].min_age == P["cutoff"]].iloc[0]
    m4.metric("Revenue lift, counterfactual", _pct_cell(_row["effect"]),
              delta=f"{_row['effect'] - P['deseason']['raw']:+.0f} pp", delta_color="inverse",
              help=f"Against matched sites that ran no campaign, restricted to sites "
                   f"{P['cutoff']}+ months old — the only version whose placebo test passes.")

    # =============================================================================================
    st.divider()
    st.header("1 · What a promotion looks like from the inside")
    st.caption("Every tracked metric around the spike month, each divided by that site's own "
               "average over the six months before it — so 1.0 means 'normal for this site' and "
               "sites of different sizes can be averaged together. Line = median across events, "
               "band = the middle half of them.")

    picked = st.multiselect("Metrics", [m[1] for m in FOCAL_METRICS],
                            default=DEFAULT_FOCAL_METRICS)
    metrics = [m for m in FOCAL_METRICS if m[1] in picked]
    n_events = P["events"].groupby(["site_key", "spike_date"]).ngroups

    if metrics:
        fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True,
                            subplot_titles=[m[1] for m in metrics], vertical_spacing=0.035)
        for i, (col, label, color) in enumerate(metrics, start=1):
            agg = cd.event_curve(P["events"], col)
            _band(fig, agg["months_from_spike"], agg["q25"], agg["q75"], color, row=i, col=1)
            _line(fig, agg["months_from_spike"], agg["median"], color, label, row=i, col=1,
                  show_legend=False,
                  hovertemplate=f"<b>{label}</b><br>month %{{x}}<br>"
                                "<b>%{y:.2f}×</b> the site's own baseline<extra></extra>")
            fig.update_yaxes(title_text="× baseline", tickformat=".2f", row=i, col=1)
        _event_marks(fig, len(metrics))
        _sub(fig, len(metrics), 215 * len(metrics) + 60, "Months from the campaign month",
             showlegend=False, hovermode="x unified",
             margin=dict(l=70, r=30, t=45, b=55))
        st.plotly_chart(fig, width="stretch")

    e = P["events"]
    _win = e[e.months_from_spike.between(1, 3)].groupby(["site_key", "spike_date"]).median(
        numeric_only=True)
    callout("What this shows", f"""
      <b>Reading.</b> On {n_events} events the picture is the one the business tells itself:
        spend jumps, revenue follows, membership ASP falls at the spike month
        (<b>{(_win['ASP_mem_norm'].median() - 1) * 100:+.0f}%</b> over months +1..+3, the discount),
        member washes rise <b>{st_['median_focal_mem_wash_pct_change']:+.0f}%</b> and retail washes
        fall <b>{st_['median_focal_ret_wash_pct_change']:+.0f}%</b>. Revenue over months +1..+3 sits
        <b>{st_['median_focal_revenue_pct_change']:+.0f}%</b> above baseline.
      <b>So-what.</b> Read as a chain it says: discount → retail customers convert to members →
        member volume sticks. That chain is the case for the spend.
      <b>Caveat.</b> Every one of these numbers is measured against the site's <i>own past</i>.
        A site that was going to grow anyway produces this exact picture with no campaign at all.
        Section 4 builds the missing comparison.
    """, WARNING)

    # =============================================================================================
    st.divider()
    st.header("2 · Where do the extra washes come from?")
    st.caption(f"The same events, but now tracking every other site within "
               f"{cd.RADIUS_KM:.0f} km. Each neighbour is normalised to its own pre-spike baseline, "
               "so a big neighbour beside a small promoter cannot fake a gain. The last panel is "
               "the one that cannot rise for both sides: the focal site's share of all washes done "
               "in its own market.")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[p[1] for p in SPILLOVER_PANELS]
                        + ["Focal site's share of its own 20 km market"])
    for i, (col, label) in enumerate(SPILLOVER_PANELS, start=1):
        for df, color, name in [(P["events"], FOCAL, "The promoting site"),
                                (P["nbr"], NEIGHBOUR, "Its neighbours")]:
            agg = cd.event_curve(df, col)
            _band(fig, agg["months_from_spike"], agg["q25"], agg["q75"], color, row=i, col=1,
                  alpha=0.10)
            _line(fig, agg["months_from_spike"], agg["median"], color, name, row=i, col=1,
                  show_legend=(i == 1),
                  hovertemplate=f"<b>{name}</b><br>{label}, month %{{x}}<br>"
                                "<b>%{y:.2f}×</b> its own baseline<extra></extra>")
        fig.update_yaxes(title_text="× baseline", tickformat=".2f", row=i, col=1)

    ms = cd.event_curve(P["mshare"], "focal_wash_share")
    for c in ["median", "q25", "q75"]:
        ms[c] = ms[c] * 100
    _band(fig, ms["months_from_spike"], ms["q25"], ms["q75"], SHARE, row=4, col=1)
    _line(fig, ms["months_from_spike"], ms["median"], SHARE, "Market share of the promoting site",
          row=4, col=1, hovertemplate="month %{x}<br>share <b>%{y:.1f}%</b><extra></extra>")
    fig.add_hline(y=st_["median_focal_wash_share_pre"],
                  line=dict(color=SHARE, dash="dash", width=1), row=4, col=1)
    fig.update_yaxes(title_text="Share of market washes", ticksuffix="%", row=4, col=1)
    _event_marks(fig, 3)
    fig.add_vline(x=0, line=dict(color=CRITICAL, dash="dot", width=1.5), row=4, col=1)
    _sub(fig, 4, 1180, "Months from the campaign month", hovermode="x unified",
         legend=dict(orientation="h", y=1.045, x=0), margin=dict(l=70, r=30, t=70, b=55))
    st.plotly_chart(fig, width="stretch")

    comp = pd.DataFrame({
        "The promoting site": [st_["median_focal_ret_wash_pct_change"],
                               st_["median_focal_mem_wash_pct_change"],
                               st_["median_focal_total_wash_pct_change"],
                               st_["median_focal_revenue_pct_change"]],
        "Its neighbours": [st_["median_nbr_ret_wash_pct_change"],
                           st_["median_nbr_mem_wash_pct_change"],
                           st_["median_nbr_total_wash_pct_change"],
                           st_["median_nbr_revenue_pct_change"]],
    }, index=["Retail washes", "Membership washes", "Total washes", "Revenue"])
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Median change over months +1 to +3, against each site's own baseline**")
        html_table(comp, fmt={c: "{:+,.1f}%" for c in comp.columns}, index_label="")
        st.caption(f"{st_['n_neighbor_event_pairs']} neighbour-event pairs · "
                   f"{st_['n_focal_with_neighbors']} promoting sites that have a neighbour · "
                   f"{st_['pct_nbr_sites_revenue_decline']:.0f}% of neighbours see revenue fall.")
    with right:
        st.markdown("**Where this evidence comes from**")
        mix = P["state_mix"].head(5).set_index("focal_state")[["focal_sites", "share_of_sites"]]
        mix.columns = ["Promoting sites", "Share"]
        html_table(mix, fmt={"Promoting sites": "{:.0f}", "Share": "{:.0f}%"}, index_label="State")
        st.caption("The spillover result is not an estate-wide average.")

    top = P["state_mix"].iloc[0]
    callout("What this shows", f"""
      <b>Reading.</b> Neighbours lose <b>{st_['median_nbr_ret_wash_pct_change']:+.1f}%</b> of their
        retail washes over the three months after a nearby promotion, while the promoting site's
        share of its own market moves from <b>{st_['median_focal_wash_share_pre']:.1f}%</b> to
        <b>{st_['median_focal_wash_share_post']:.1f}%</b>. Neighbour <i>revenue</i> holds up
        (<b>{st_['median_nbr_revenue_pct_change']:+.1f}%</b>) because membership income is
        contracted and does not move month to month.
      <b>So-what.</b> On the face of it a promotion redistributes rather than grows: about
        {st_['pct_nbr_sites_revenue_decline']:.0f}% of neighbours are worse off in revenue terms,
        and the volume the promoter gains looks like volume somebody else lost.
      <b>Caveat.</b> <b>{top['focal_sites']:.0f} of the {P['state_mix'].focal_sites.sum():.0f}
        promoting sites with a neighbour are in {top['focal_state']}</b>: one dense cluster
        carries the result. And the median per-event share gain is
        <b>{st_['share_gain_pp']:+.1f} pp</b>, i.e. about zero: the shift is concentrated in a few
        markets, not general. Section 4.4 tests the stealing claim against sites 100 km away that
        cannot possibly be cannibalised.
    """, SERIOUS)

    # =============================================================================================
    st.divider()
    st.header("3 · What a campaign costs, and what comes back")
    st.caption("Consecutive spend months merge into one campaign. Spend is the OPEX **above** the "
               "site's own baseline — the incremental cost of the promotion, not its cost base.")

    sp = P["spend"].set_index("Campaign length")
    html_table(sp, fmt={"Campaigns": "{:.0f}", "Share": "{:.0f}%", "Mean spend": "${:,.0f}",
                        "p25": "${:,.0f}", "Median spend": "${:,.0f}", "p75": "${:,.0f}"},
               index_label="Campaign length")

    bucket = st.radio("Campaign length", ["1 month", "2 months"], horizontal=True, index=0)
    sub = P["snapshot"][P["snapshot"].bucket == bucket]
    n_camps = int((camps.duration_bucket == bucket).sum())
    camp_months = {"1 month": [0], "2 months": [0, 1]}[bucket]

    if len(sub):
        agg = sub.groupby("mfs")[["opex", "revenue", "profit", "mem_purchases"]].median().reset_index()
        # Dollars and headcounts do not share a y-axis. Two stacked panels on one x instead.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                            row_heights=[0.66, 0.34],
                            subplot_titles=["Money: median across campaigns",
                                            "New membership sign-ups: median across campaigns"])
        for col, name, color in [("opex", "OPEX", S2), ("revenue", "Revenue", S1),
                                 ("profit", "Profit (revenue − OPEX)", S3)]:
            fig.add_bar(x=agg["mfs"], y=agg[col], name=name, marker=dict(color=color),
                        hovertemplate=f"{name}, month %{{x}}<br><b>$%{{y:,.0f}}</b><extra></extra>",
                        row=1, col=1)
        fig.add_bar(x=agg["mfs"], y=agg["mem_purchases"], name="Membership sign-ups",
                    marker=dict(color=S4),
                    hovertemplate="Sign-ups, month %{x}<br><b>%{y:,.0f}</b><extra></extra>",
                    row=2, col=1)
        for r in (1, 2):
            fig.add_vrect(x0=min(camp_months) - 0.5, x1=max(camp_months) + 0.5,
                          fillcolor=_rgba(WARNING, 0.13), line_width=0, layer="below",
                          row=r, col=1)
        fig.update_yaxes(title_text="Median $ per month", tickprefix="$", tickformat=",.0f",
                         row=1, col=1)
        fig.update_yaxes(title_text="Sign-ups", tickformat=",.0f", row=2, col=1)
        _sub(fig, 2, 620, "Months from campaign start", barmode="group", bargap=0.22,
             hovermode="x unified", legend=dict(orientation="h", y=1.055, x=0),
             margin=dict(l=85, r=30, t=75, b=55))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{n_camps} campaigns · shaded months are the campaign itself · "
                   "medians, so one large operator cannot set the level.")

        pre = agg[agg.mfs < 0][["profit", "mem_purchases"]].median()
        post = agg[agg.mfs.between(max(camp_months) + 1, 6)][["profit", "mem_purchases"]].median()
        callout("What this shows", f"""
          <b>Reading.</b> During a {bucket.lower()} campaign OPEX steps up and profit is squeezed;
            once it ends, median profit runs
            <b>{(post['profit'] / pre['profit'] - 1) * 100:+.0f}%</b> against the pre-campaign
            months and sign-ups <b>{(post['mem_purchases'] / pre['mem_purchases'] - 1) * 100:+.0f}%</b>.
          <b>So-what.</b> The money panel is where the case usually gets made: a median
            <b>${camps[camps.duration_bucket == bucket].total_incremental_opex.median():,.0f}</b>
            of extra spend, against a revenue line that is visibly higher afterwards.
          <b>Caveat.</b> "Afterwards" is doing the work. Nothing here holds the site's own growth
            constant, and three-quarters of these campaigns are run by sites less than two years
            old. Sign-ups and dollars are plotted separately on purpose. They are not the same
            unit and must not share an axis.
        """, WARNING)

    # =============================================================================================
    st.divider()
    st.header("4 · Would it have happened anyway?")
    st.markdown(
        "Everything above measures a campaign as **actual minus that site's own pre-campaign "
        "average**. That is not a counterfactual. The fix is to build one: for every campaign, "
        "find sites that ran **no campaign** in the same window, observe them over the **same "
        "calendar months**, and subtract. Controls are matched on census region (same weather, "
        "same seasonal calendar) and on site age (± 6 months), because a site still ramping up "
        "after opening is the other thing that makes revenue rise on its own.")

    st.subheader("4.1 · Is it just seasonality?")
    st.caption("Measured directly, on campaign-free site-months only: log revenue minus each "
               "site's own average for that calendar year, which strips out site size and "
               "year-on-year growth and leaves the calendar.")

    c1, c2 = st.columns([1.35, 1])
    with c1:
        seas = P["seasonal"]
        fig = go.Figure()
        fig.add_bar(x=seas["label"], y=seas["pct_vs_site_year_avg"], marker=dict(color=S1),
                    customdata=seas["count"],
                    hovertemplate="%{x}: <b>%{y:+.1f}%</b> vs the site's own annual average<br>"
                                  "<span style='opacity:.7'>%{customdata} campaign-free "
                                  "site-months</span><extra></extra>")
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(style(fig, height=330, yaxis_title="% vs the site's own year",
                              yaxis=dict(ticksuffix="%"), showlegend=False,
                              margin=dict(l=65, r=25, t=20, b=45)), width="stretch")
    with c2:
        ds = P["deseason"]
        starts = P["start_months"]
        fig = go.Figure()
        fig.add_bar(x=starts["label"], y=starts["campaigns"], marker=dict(color=S3),
                    hovertemplate="%{x}: <b>%{y}</b> campaigns started<extra></extra>")
        st.plotly_chart(style(fig, height=330, yaxis_title="Campaigns started",
                              showlegend=False, margin=dict(l=55, r=25, t=20, b=45)),
                        width="stretch")

    direction = ("slightly <i>hiding</i> a bit of the lift, not creating it"
                 if ds["seasonality_pp"] < 0 else "adding a sliver to the lift")
    callout("What this shows", f"""
      <b>The worry.</b> Car washes are busier in some months than others. If campaigns happen to get
        launched just before the busy months, the "lift" is really just the calendar, and we would
        be paying for something the weather was going to deliver anyway.
      <b>Some months really are better (left).</b> The best month runs
        <b>{ds['swing_hi']:+.0f}%</b> against a site's own yearly average and the worst runs
        <b>{ds['swing_lo']:+.0f}%</b> — about <b>{ds['peak_to_trough']:.0f} points</b> from top to
        bottom. Real, but not enormous.
      <b>Campaigns are not timed to them (right).</b> Launches are scattered across all twelve
        months rather than piling up ahead of the good ones. Nobody is gaming the calendar.
      <b>And the maths agrees.</b> Strip the seasonal pattern out of revenue, redo the exact same
        calculation, and the lift goes from <b>{ds['raw']:+.1f}%</b> to
        <b>{ds['deseasonalised']:+.1f}%</b> — a shift of <b>{abs(ds['seasonality_pp']):.1f} points</b>,
        which is nothing. If anything the seasons were {direction}.
      <b>What this means.</b> Seasonality is <b>not</b> the explanation. That is one objection dead —
        but the lift still looks too big to be real, so something else is inflating it. The next
        chart goes looking for it.
    """, GOOD)

    st.subheader("4.2 · Against sites that ran no campaign")
    st.caption("Blue is what the earlier sections measure. Orange is the estimate of what would "
               "have happened anyway. Purple is the difference — the campaign effect. Read the "
               "shaded pre-campaign months first: if purple is already above zero there, the two "
               "groups were diverging **before** the campaign.")

    cut = st.select_slider(
        "Only campaigns at sites this old (months of trading before the campaign)",
        CUTOFFS, value=P["cutoff"], format_func=lambda v: "any age" if v == 0 else f"{v}+ months")
    panel = P["panels"][cut]
    path = cf.path(panel, min_events=cd.MIN_EVENTS)
    dropped = path.attrs.get("dropped", pd.DataFrame())
    n_camp_cut = int((P["aged"].age_months >= cut).sum())

    fig = go.Figure()
    fig.add_vrect(x0=cf.LO - 0.5, x1=-0.5, fillcolor=_rgba(MUTED, 0.10), line_width=0, layer="below")
    _band(fig, path["k"], path["lo"], path["hi"], GAP, alpha=0.16)
    for col, name, color in [("treated", "Campaign sites", TREATED),
                             ("control", "Matched sites that ran no campaign", CONTROL),
                             ("did", "The gap — the campaign effect", GAP)]:
        _line(fig, path["k"], path[col], color, name,
              hovertemplate=f"<b>{name}</b><br>month %{{x}}: <b>%{{y:+.1f}}%</b><extra></extra>")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=0, line=dict(color=INK2, width=1, dash="dot"))
    fig.add_annotation(x=cf.LO + 1, y=0, yshift=20, text="baseline months", showarrow=False,
                       font=dict(size=11, color=MUTED))
    fig.add_annotation(x=0, y=1, yref="paper", yshift=14, text="campaign starts", showarrow=False,
                       font=dict(size=11, color=INK2))
    st.plotly_chart(style(fig, height=520, xaxis_title="Months from campaign start",
                          yaxis_title="% vs its own baseline months",
                          xaxis=dict(dtick=2), yaxis=dict(ticksuffix="%"),
                          legend=dict(orientation="h", y=-0.22, x=0),
                          margin=dict(l=70, r=30, t=35, b=95)), width="stretch")

    cap = (f"{n_camp_cut} campaigns qualify. ")
    if len(dropped):
        cap += (f"Months +{int(dropped.k.min())} onward rest on {int(dropped.n.min())}–"
                f"{int(dropped.n.max())} events and are not plotted — that is too thin to read.")
    st.caption(cap)

    rep = cf.report(panel).set_index("Window")
    rep_show = rep[["Naive", "Counterfactual", "CI low", "CI high", "Events"]]
    html_table(rep_show, fmt={"Naive": "{:+,.1f}%", "Counterfactual": "{:+,.1f}%",
                             "CI low": "{:+,.1f}%", "CI high": "{:+,.1f}%", "Events": "{:.0f}"},
               index_label="Window")

    det = cf.detrended(panel)
    pre_row = rep.iloc[0]
    passes = not bool(pre_row["Significant"])
    verdict = ("includes zero. So that gap is within noise and the comparison holds up: the "
               "placebo <b>passes</b>" if passes else
               "does <b>not</b> include zero. The two groups were already diverging before "
               "anything happened, so the comparison is broken: the placebo <b>FAILS</b>")
    cost = ("Every campaign is in here, including the ones at brand-new sites — which is exactly "
            "why the pre-campaign gap is wide open. Push the slider right and it closes" if not cut
            else f"This holds only because every campaign at a site younger than {cut} months was "
                 "thrown away. Move the slider back to <i>any age</i> and the pre-campaign gap "
                 "opens up")
    callout("What this shows", f"""
      <b>The check first.</b> Before the campaign ran, the two groups should look the same — any gap
        there is a flaw in the comparison, not an effect. They differ by
        <b>{pre_row['Counterfactual']:+.1f}%</b>, and the range around that is
        [{pre_row['CI low']:+.1f}, {pre_row['CI high']:+.1f}], which {verdict}.
      <b>What the campaign appears to do.</b> Over the three months after launch, campaign sites run
        <b>{rep.iloc[1]['Counterfactual']:+.1f}%</b> ahead of sites that ran nothing
        [{rep.iloc[1]['CI low']:+.1f}, {rep.iloc[1]['CI high']:+.1f}]. That rests on just
        {int(rep.iloc[1]['Events'])} campaigns, which is why the range is wide.
      <b>The number to quote.</b> Take that {det['post']:+.1f}% and subtract the {det['pre']:+.1f}%
        head start the campaign sites already had before launch, campaign by campaign, and what is
        left is <b>{det['detrended']:+.1f}%</b> [{det['lo']:+.1f}, {det['hi']:+.1f}]. That is the
        part you can actually credit to the campaign.
      <b>Why the slider matters.</b> {cost}. New sites grow on their own, campaigns get run at new
        sites, and that growth gets credited to the campaign. The headline lift in the earlier
        sections is mostly that.
    """, GOOD if passes else CRITICAL)

    real = cd.real_example_panel(P["data"])
    with st.expander(f"Real example — {len(real)} sites, revenue, month by month"):
        st.caption(
            f"The chart above is a median across many matched campaigns. Here are {len(real)} real "
            "sites, one panel each: raw monthly revenue, the year the campaign ran against the "
            f"closest clean comparison year at the same site. All {len(real)} mature (12+ months "
            "old) campaigns with a usable clean comparison year on revenue — not a selection of "
            "the best-looking ones.")

        n_cols = 2 if len(real) <= 4 else 3  # a small set reads better as a square grid
        n_rows = -(-len(real) // n_cols)
        figr = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[r["site_key"] for r in real],
            vertical_spacing=0.16, horizontal_spacing=0.08)
        for i, r in enumerate(real):
            row, col = i // n_cols + 1, i % n_cols + 1
            months = r["raw_wide"].index.tolist()
            _line(figr, months, r["raw_normal_avg"], CONTROL, "normal year(s)",
                 row=row, col=col, show_legend=(i == 0),
                 hovertemplate="month %{x}: <b>$%{y:,.0f}</b><extra>normal</extra>")
            _line(figr, months, r["raw_wide"][r["campaign_year"]], TREATED, "campaign year",
                 row=row, col=col, show_legend=(i == 0),
                 hovertemplate="month %{x}: <b>$%{y:,.0f}</b><extra>campaign</extra>")
            figr.add_vrect(x0=r["campaign_month"] - 0.5, x1=r["campaign_month"] + 0.5,
                          fillcolor=_rgba(WARNING, 0.20), line_width=0, layer="below",
                          row=row, col=col)
            figr.update_xaxes(tickvals=months, tickangle=45, row=row, col=col)
        # `_sub()` isn't used here -- it force-left-aligns every subplot title to x=0, which only
        # works for the single-column grids it was built for. With 3 columns that collapses every
        # row's three titles onto the figure's left edge, on top of each other. Plotly's own
        # per-subplot title centering is correct as-is; only the color/size needs to match house
        # style. Order matters: style() first (base layout, reaches axis 1 only), then
        # update_xaxes/update_yaxes with no row/col to extend it to every subplot -- same order
        # `_sub()` itself uses.
        style(figr, height=720, showlegend=True,
             legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
             margin=dict(l=75, r=25, t=60, b=60))
        figr.update_xaxes(showgrid=False, showline=True, linecolor=GRID, tickfont=dict(color=MUTED))
        figr.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
        figr.update_yaxes(tickprefix="$", tickformat=",.0f")
        figr.update_yaxes(title_text="Revenue ($)", col=1)
        for ann in figr.layout.annotations:
            ann.font.update(color=INK, size=12)
        st.plotly_chart(figr, width="stretch")

        n_sig_pos = sum(1 for r in real if r["p_value"] < 0.05 and r["mean_lift_pct"] > 0)
        callout("What this shows", f"""
          <b>Every site shows a real spike in its own campaign month</b> — visible in every panel,
            regardless of what happens afterward.
          <b>{n_sig_pos} of {len(real)} sites shows a significant, positive, SUSTAINED lift</b> in
            the months that follow, tested against that same site's own normal year. The rest are
            flat or negative once the campaign month itself passes.
          <b>Why this matters.</b> A single site is not enough data to see a
            ~{P['deseason']['deseasonalised']:.0f}% effect — that is smaller than the ordinary
            year-to-year noise visible between the blue and orange lines above, at sites that ran
            no campaign at all in one of those years. This is exactly why the aggregate,
            matched-control chart above exists: the effect is real, but only visible in aggregate.
        """, WARNING if n_sig_pos == 0 else GOOD)

    # =============================================================================================
    st.divider()
    with st.expander("Data & method"):
        st.markdown(f"""
**One input file.** `proforma/data/opex/opex-data.csv` — the monthly P&L panel, {len(P['data']):,}
site-months across {n_sites} sites. Nothing is joined in, so this section shares no data with ① or
②. The site key is `client_id + site_id`; `site_id` alone is a within-brand index and collides.

**The campaign is inferred, not recorded.** There is no campaign table in this business. A campaign
month is one where `cogs + expenses` exceeds that site's own trailing 6-month mean by
{threshold:.2f}× (the baseline is shifted one month so the spike cannot inflate the bar it must
clear). Consecutive such months, with a gap of at most one, merge into a single campaign. **This is
the weakest link in the section** — section 4.3 re-runs the estimate on fixed expenses with COGS
excluded to show how much it matters.

**Normalisation.** Every event-study panel divides a site's metric by that site's own mean over the
six months before the event, so 1.0 is "normal for this site" and sites of different sizes can be
pooled. Aggregation is always the **median of per-event changes**, never the change of the medians —
each event weighs the same regardless of operator size.

**The local market** is every site within {cd.RADIUS_KM:.0f} km, straight-line. Each neighbour is
normalised to its own baseline, not the promoter's.

**The counterfactual** is a stacked difference-in-differences. Baseline months are −6..−4,
deliberately early, which leaves −3..−1 free as a placebo window a clean design must read zero in.
Controls must have run no campaign anywhere in the ±window and are matched on census region and on
site age (± 6 months since opening). An event needs {cd.Counterfactual.MIN_CTRL} usable controls to
count. Confidence intervals bootstrap over **events**, the unit of independence — resampling
site-months would treat 18 rows from one campaign as 18 independent observations. Bootstrap
intervals move in the last decimal between runs; the conclusions do not turn on that digit.

**Known limits.**
- The spillover result in section 2 is geographically concentrated —
  {top['focal_sites']:.0f} of {P['state_mix'].focal_sites.sum():.0f} promoting sites with a
  neighbour sit in {top['focal_state']}.
- Within-site correlation between COGS and revenue is {P['cogs_corr']:.2f}, so the trigger partly
  fires on volume rather than on marketing.
- Only {int(P['sweep'][P['sweep'].min_age == P['cutoff']]['campaigns'].iloc[0])} campaigns clear the
  {P['cutoff']}-month age bar, and the panel is only ~43 months long, so demanding a long history
  before a campaign forces the campaign into the last stretch of the data.

**What would settle it.** Actual campaign records (dates, spend, offer) instead of inferred spikes;
more campaigns at mature sites; and above all **staggered rollouts** — one operator running a
campaign at some sites but not others in the same market and month. That is a within-operator
control and removes the selection problem entirely.

**Environment note.** `st.dataframe` / `st.table` segfault the Streamlit server on the second script
run in this env (pyarrow 25.0.0 + pandas 3.0.2 + streamlit 1.58.0), so every table on this page is
hand-rendered HTML. The proper fix is an environment pin, deliberately not done here.
        """)
