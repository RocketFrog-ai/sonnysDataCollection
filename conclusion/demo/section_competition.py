"""
Section — Competition. What an opening does to the site itself and to its neighbours.

Two-body (one entrant, one incumbent), three-body (one entrant, two incumbents), and the n-body
saturation cut, plus a case explorer that puts **every** event in front of the reviewer with its
location, its distance and both opening dates.

The maths is in `competition_data.py` and is Streamlit-free, so the notebook reports the same
numbers. Nothing here computes anything except formatting.

Charts follow the pack's system (`ui.py`): three validated hues carrying identity — blue for the
incumbent, green for the second incumbent, orange for the entrant — each also dash-coded and
direct-labelled, so nothing is colour-alone. Signed effects use the same warm/cool pair against a
zero line rather than a status colour, because a neighbour losing washes is an outcome, not an
alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import competition_data as cd
from ui import (BORDER, DARK, GRID, INK, INK2, MUTED, S1, S2, S3, SURFACE, callout, html_table,
                style)

# The entrant is always orange, the incumbent always blue, the second incumbent always green —
# fixed by role, never by rank in a filtered list.
C_INC, C_INC2, C_ENT = S1, S3, S2
LOSS, GAIN = S2, S1

SPAN = 12          # months either side of the opening drawn in event time
_MONEY = {"revenue"}


# =================================================================================================
# Cached wrappers — the only Streamlit in the data path
# =================================================================================================

@st.cache_data(show_spinner="Rebuilding every opening…")
def _pairs(radius: float, window: int, min_age: int) -> pd.DataFrame:
    return cd.pair_events(radius, window, min_incumbent_age=min_age)


@st.cache_data(show_spinner=False)
def _triples(radius: float, window: int, min_age: int) -> pd.DataFrame:
    return cd.triple_events(_pairs(radius, window, min_age))


@st.cache_data(show_spinner=False)
def _saturation(radius: float, window: int, min_age: int, metric: str) -> pd.DataFrame:
    return cd.saturation(radius, window, min_incumbent_age=min_age, metric=metric)


@st.cache_data(show_spinner=False)
def _series(site_key: str, metric: str) -> pd.DataFrame:
    return cd.site_series(site_key, metric)


def _fmt(v: float, metric: str, digits: int = 0) -> str:
    if not np.isfinite(v):
        return "—"
    return (f"\\${v:,.0f}" if metric in _MONEY else f"{v:,.{digits}f}")


def _pp(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v:+.1f}%"


# =================================================================================================
# render
# =================================================================================================

def render() -> None:
    st.markdown("<div class='kicker'>Evidence pack · conclusion ⑤</div>", unsafe_allow_html=True)
    st.title("Competition")
    st.markdown("Somebody opens a car wash near one of ours. **What happens to the neighbour, and "
                "what happens to the new site?** Every opening in the panel, measured against what "
                "untouched sites did over the same months.")

    with st.sidebar:
        st.markdown("### ④ Competition")
        metric_label = st.selectbox("Measure", [v[0] for v in cd.METRICS.values()], index=0)
        metric = [k for k, v in cd.METRICS.items() if v[0] == metric_label][0]
        radius = st.slider("How near counts as competition (miles)", 3.0, 15.0, 10.0, 1.0)
        window = st.slider("Months either side of the opening", 3, 12, 6, 1)
        min_age = st.slider("Neighbour must be at least this old (months)", 0, 36, 12, 3,
                            help="A young neighbour is still climbing its own opening ramp, which "
                                 "shows up as growth and hides the entrant's effect. This is the "
                                 "confound that §③ of this pack was built around.")
        st.caption("Every number on this page moves with these four.")

    pairs = _pairs(radius, window, min_age)
    if pairs.empty:
        st.warning("No openings survive these settings — loosen the radius or the age floor.")
        return
    h = cd.pair_headline(pairs, metric)
    noun = cd.METRICS[metric][1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Openings measured", f"{h['n_entrants']:,}",
              f"{h['n_pairs']:,} neighbour pairs", delta_color="off")
    m2.metric("Typical neighbour, raw", _pp(h["raw"]), "before vs after", delta_color="off")
    m3.metric("…against untouched sites", _pp(h["did"]),
              f"they did {_pp(h['ctrl'])}", delta_color="off")
    m4.metric("Neighbours that lose", f"{h['share_down_did']:.0%}",
              "vs the counterfactual", delta_color="off")

    t1, t2, t3, t4, t5 = st.tabs(["Two-body", "Three-body", "The new site itself",
                                  "Every case", "Data & method"])

    with t1:
        _two_body(pairs, metric, window, h, noun)
    with t2:
        _three_body(pairs, metric, radius, window, min_age, noun)
    with t3:
        _entrant(pairs, metric, window, noun)
    with t4:
        _cases(pairs, metric, window)
    with t5:
        _method(pairs, radius, window, min_age)


# =================================================================================================
# ① two-body
# =================================================================================================

def _two_body(pairs: pd.DataFrame, metric: str, window: int, h: dict, noun: str) -> None:
    st.subheader("One opening, one neighbour")
    st.caption(f"Every (entrant, neighbour) pair inside the radius — **{h['n_pairs']:,} pairs** "
               f"from **{h['n_entrants']:,} openings** onto **{h['n_incumbents']:,} distinct "
               "neighbours**. A site hit by two openings appears twice, once per opening.")

    # --- event study --------------------------------------------------------------------------
    # The two comparison bands follow the radius rather than being hard-coded, so pulling the
    # radius slider down to 5 miles leaves both of them populated instead of emptying "far".
    radius = float(pairs.attrs.get("radius_mi", 10.0))
    near_cut = min(3.0, radius / 2)
    far_cut = max(near_cut + 0.5, radius / 2)
    close = pairs[pairs.distance_mi < near_cut].copy()
    far = pairs[pairs.distance_mi >= far_cut].copy()
    prof_close = cd.event_profile(close, metric, SPAN, window, group=None)
    prof_far = cd.event_profile(far, metric, SPAN, window, group=None)
    ent = cd.entrant_profile(close, metric, SPAN, window)

    fig = go.Figure()
    if not prof_close.empty:
        fig.add_scatter(x=list(prof_close.offset) + list(prof_close.offset[::-1]),
                        y=list(prof_close.p75) + list(prof_close.p25[::-1]), fill="toself",
                        fillcolor="rgba(57,135,229,0.13)", line=dict(width=0), hoverinfo="skip",
                        showlegend=False)
    for frame, colour, dash, name in (
            (prof_close, C_INC, "solid", f"Neighbour within {near_cut:.0f} mi (n={len(close):,})"),
            (prof_far, C_INC2, "dot",
             f"Neighbour {far_cut:.0f}–{radius:.0f} mi (n={len(far):,})")):
        if frame.empty:
            continue
        fig.add_scatter(x=frame.offset, y=frame["median"], mode="lines+markers", name=name,
                        line=dict(color=colour, width=2.5, dash=dash),
                        marker=dict(size=7, line=dict(width=1.5, color=SURFACE)),
                        customdata=frame.n,
                        hovertemplate="Month %{x} · <b>%{y:.0f}</b> vs its own past"
                                      "<br><span style='opacity:.7'>%{customdata} pairs</span>"
                                      "<extra></extra>")
    fig.add_scatter(x=ent.offset, y=ent["median"], mode="lines+markers",
                    name="The new site itself", line=dict(color=C_ENT, width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond", line=dict(width=1.5, color=SURFACE)),
                    hovertemplate="Month %{x} · <b>%{y:.0f}</b> of the neighbour's old volume"
                                  "<extra></extra>")
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_hline(y=100, line=dict(color=GRID, width=1))
    for frame, colour, label in ((prof_close, C_INC, f"within {near_cut:.0f} mi"),
                                 (prof_far, C_INC2, f"{far_cut:.0f}–{radius:.0f} mi"),
                                 (ent, C_ENT, "new site")):
        if frame.empty:
            continue
        fig.add_annotation(x=SPAN, y=float(frame["median"].iloc[-1]), text=label, showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=colour, size=11))
    st.plotly_chart(style(fig, height=430,
                          xaxis_title="Months from the opening", yaxis_title="Index (own pre = 100)",
                          xaxis=dict(dtick=3, range=[-SPAN, SPAN + 4]),
                          margin=dict(r=90),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")

    if prof_close.empty:
        return
    drop = 100 - float(prof_close[prof_close.offset.between(1, window)]["median"].median())
    pre_flat = float(prof_close[prof_close.offset < 0]["median"].median())
    ent_share = float(ent[ent.offset.between(1, window)]["median"].median())
    far_drop = (100 - float(prof_far[prof_far.offset.between(1, window)]["median"].median())
                if not prof_far.empty else float("nan"))
    callout("What this shows", f"""
      <b>The line is flat until the day they open.</b> For the year before the opening the
        neighbour sits at <b>{pre_flat:.0f}</b> against its own past. Nothing is drifting, which
        is what makes the step at month 0 readable as the opening rather than as a trend that was
        already running.
      <b>Then a close neighbour steps down about {drop:.0f}%</b> and stays there. It is a level
        change, not a dip: there is no recovery inside the following year.
      <b>A neighbour {far_cut:.0f}–{radius:.0f} miles away gives up {far_drop:.0f}%</b>. The two
        lines separate only after the opening, never before.
      <b>The new site is not simply taking that volume.</b> Within its first {window} months it is
        already doing <b>{ent_share:.0f}%</b> of what the neighbour used to do, against the
        neighbour's <b>{drop:.0f}%</b> loss. Most of what it washes is traffic that was not going
        to the neighbour.
    """)

    # --- distance decay ------------------------------------------------------------------------
    st.divider()
    st.markdown("#### How far away does it stop mattering?")
    st.caption("Each bar is the neighbour's change **minus** what untouched sites of the same age, "
               "in the same region, did over the same months. The line behind it is the middle "
               "half of pairs in that band — the spread, not the error.")

    bd = cd.by_distance(pairs, metric)
    fig2 = go.Figure()
    fig2.add_bar(x=bd.distance_band.astype(str), y=bd.did,
                 marker=dict(color=[LOSS if v < 0 else GAIN for v in bd.did],
                             line=dict(width=2, color=SURFACE)),
                 customdata=np.stack([bd.pairs, bd.raw, bd.control, bd.share_down], axis=-1),
                 hovertemplate="<b>%{x}</b><br>Effect <b>%{y:+.1f}%</b><br>"
                               "raw %{customdata[1]:+.1f}% · untouched %{customdata[2]:+.1f}%<br>"
                               "%{customdata[3]:.0%} of pairs down<br>"
                               "<span style='opacity:.7'>%{customdata[0]} pairs</span>"
                               "<extra></extra>", showlegend=False)
    for _, r in bd.iterrows():
        fig2.add_scatter(x=[str(r.distance_band)] * 2, y=[r.p25, r.p75], mode="lines",
                         line=dict(color=MUTED, width=1.5), hoverinfo="skip", showlegend=False)
        fig2.add_annotation(x=str(r.distance_band), y=r.did, text=f"{r.did:+.1f}%",
                            showarrow=False, yshift=-16 if r.did < 0 else 14,
                            font=dict(color=INK, size=12))
    fig2.add_hline(y=0, line=dict(color=AXIS_LINE(), width=1.5))
    st.plotly_chart(style(fig2, height=380, xaxis_title="Distance from the new site",
                          yaxis_title=f"{cd.METRICS[metric][0]} vs untouched sites (%)",
                          bargap=0.45), width="stretch")

    show = bd.copy()
    show["distance_band"] = show.distance_band.astype(str)
    show.columns = ["Distance", "Pairs", "Raw change", "Untouched sites", "Difference",
                    "p25", "p75", "Share down"]
    show = show[["Distance", "Pairs", "Raw change", "Untouched sites", "Difference", "Share down"]]
    show.index = range(1, len(show) + 1)
    html_table(show, fmt={"Pairs": "{:,.0f}", "Raw change": "{:+.1f}%",
                          "Untouched sites": "{:+.1f}%", "Difference": "{:+.1f}%",
                          "Share down": "{:.0%}"})

    b0, b_far = bd.iloc[0], bd.iloc[-1]
    callout("What this shows", f"""
      <b>The damage is a close-range effect.</b> Inside a mile a neighbour gives up
        <b>{b0.did:+.1f}%</b> of its {noun}; by {b_far.distance_band} it is
        <b>{b_far.did:+.1f}%</b>. The gradient does almost all of its work in the first three
        miles.
      <b>Raw before/after would have told you the wrong thing.</b> Untouched sites *grew*
        <b>{h['ctrl']:+.1f}%</b> over the same months, so the raw <b>{h['raw']:+.1f}%</b>
        understates the hit. The neighbour did not just fall, it failed to grow with everyone
        else.
      <b>It is not universal.</b> Even inside a mile, <b>{1 - b0.share_down:.0%}</b> of neighbours
        still come out ahead. "A wash opened near us" is not on its own a forecast.
    """, S3)

    # --- what kind of wash is lost ---------------------------------------------------------------
    st.divider()
    st.markdown("#### Which customers actually leave?")
    st.caption(f"The same close neighbours (within {near_cut:.0f} miles), split by how the wash was "
               "paid for. Independent of the measure selected in the sidebar — this comparison is "
               "the point of it.")

    split = pd.DataFrame([
        {"Kind": cd.METRICS[k][0], "key": k,
         **{c: v for c, v in cd.pair_headline(close, k).items()
            if c in ("raw", "ctrl", "did", "share_down_did")}}
        for k in ("retail", "membership", "total")])
    fig4 = go.Figure()
    fig4.add_bar(x=split.Kind, y=split.did,
                 marker=dict(color=[LOSS if v < 0 else GAIN for v in split.did],
                             line=dict(width=2, color=SURFACE)),
                 customdata=np.stack([split.raw, split.ctrl, split.share_down_did], axis=-1),
                 hovertemplate="<b>%{x}</b><br>%{y:+.1f}% vs untouched<br>"
                               "raw %{customdata[0]:+.1f}% · untouched %{customdata[1]:+.1f}%<br>"
                               "%{customdata[2]:.0%} of neighbours down<extra></extra>",
                 showlegend=False)
    for _, r in split.iterrows():
        fig4.add_annotation(x=r.Kind, y=r.did, text=f"{r.did:+.1f}%", showarrow=False,
                            yshift=-16 if r.did < 0 else 14, font=dict(color=INK, size=12))
    fig4.add_hline(y=0, line=dict(color=AXIS_LINE(), width=1.5))
    st.plotly_chart(style(fig4, height=330, bargap=0.55,
                          yaxis_title="vs untouched sites (%)"), width="stretch")

    ret = split[split.key == "retail"].iloc[0]
    mem = split[split.key == "membership"].iloc[0]
    callout("What this shows — the comfortable version of this does not survive", f"""
      <b>Raw, it looks like the members stay.</b> Retail washes at a close neighbour fall
        <b>{ret.raw:+.1f}%</b> while membership washes move <b>{mem.raw:+.1f}%</b>, which reads as
        "the drive-ups leave and the subscribers are locked in".
      <b>They were supposed to grow.</b> Over the same months, untouched sites put on
        <b>{mem.ctrl:+.1f}%</b> of membership washes against <b>{ret.ctrl:+.1f}%</b> of retail.
        Membership is a growing category everywhere; standing still in it is not standing still.
      <b>Against that, the membership book is hit almost as hard</b>: <b>{mem.did:+.1f}%</b> versus
        retail's <b>{ret.did:+.1f}%</b>. The moat is mostly an artefact of comparing a growing line
        to its own past.
      <b>What it changes.</b> A defence plan that protects retail and assumes the subscriber base
        holds is reading the raw column. The membership loss shows up as <em>growth that did not
        happen</em>. No month ever looks bad, and a year later the base is materially smaller than
        the untouched comparison.
    """, S2)

    # --- regimes -------------------------------------------------------------------------------
    st.divider()
    st.markdown("#### Is the market bigger afterwards, or just split?")
    rm = cd.regime_mix(pairs)
    rm = rm[rm.regime != "unknown"]
    ORDER_COLOUR = {"Pure cannibalisation": LOSS, "Cannibalisation + growth": "#b98b3a",
                    "Market expansion": GAIN, "Flat / mixed": MUTED}
    fig3 = go.Figure()
    for _, r in rm.iterrows():
        fig3.add_bar(y=["mix"], x=[r.share], orientation="h", name=f"{r.regime} ({r.share:.0%})",
                     marker=dict(color=ORDER_COLOUR.get(r.regime, MUTED),
                                 line=dict(width=2, color=SURFACE)),
                     hovertemplate=f"<b>{r.regime}</b><br>%{{x:.1%}} of pairs "
                                   f"({r.pairs:,})<extra></extra>")
    st.plotly_chart(style(fig3, height=190, barmode="stack", showlegend=True,
                          xaxis=dict(tickformat=".0%", title="Share of pairs"),
                          yaxis=dict(showticklabels=False),
                          legend=dict(orientation="h", y=-0.35, x=0),
                          margin=dict(l=10, r=10, t=20, b=70)), width="stretch")

    exp = float(rm[rm.regime == "Market expansion"].share.sum())
    pure = float(rm[rm.regime == "Pure cannibalisation"].share.sum())
    both = float(rm[rm.regime == "Cannibalisation + growth"].share.sum())
    callout("What this shows", f"""
      <b>Two sites together nearly always wash more cars than one did.</b> Adding the entrant's
        volume to the neighbour's, the pair is up a median <b>{h['combined']:+.0f}%</b> on what the
        neighbour alone was doing.
      <b>Pure cannibalisation, where the neighbour loses and the pair does not grow, is
        {pure:.0%} of cases.</b> The common outcome is
        <b>{exp:.0%} market expansion</b> and <b>{both:.0%} both at once</b>: the neighbour gives
        something up and the market still ends bigger.
      <b>Which is the honest version of "it grows the market".</b> It does, but the growth lands
        in the new site's till, and inside three miles some of it is billed to the neighbour.
    """, S3)


def AXIS_LINE() -> str:
    """Zero line — the neutral midpoint of the loss/gain pair, so it must not be a hue."""
    return "#4a4a46" if DARK else "#9a9992"


# =================================================================================================
# ② three-body
# =================================================================================================

def _three_body(pairs: pd.DataFrame, metric: str, radius: float, window: int, min_age: int,
                noun: str) -> None:
    st.subheader("One opening, two neighbours")
    tr = _triples(radius, window, min_age)
    if tr.empty:
        st.info("No opening in this cut lands on two qualifying neighbours.")
        return
    th = cd.triple_headline(tr, metric)

    st.caption(f"**{th['n']:,} openings** that landed on two qualifying neighbours at once. Both "
               "are measured in the same event, so the calendar, the region and the entrant are "
               "held fixed and the only thing that differs is **which one was closer**.")

    # --- paired slope --------------------------------------------------------------------------
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for name, key, subset in (("All triples", "", tr),
                                  (f"Nearest under {th['close_mi']:.0f} mi", "close_",
                                   tr[tr.near_distance_mi < th["close_mi"]])):
            near = float(subset[f"near_did_{metric}"].median())
            far = float(subset[f"far_did_{metric}"].median())
            dash = "solid" if not key else "dash"
            fig.add_scatter(x=["Nearer neighbour", "Further neighbour"], y=[near, far],
                            mode="lines+markers+text", name=f"{name} (n={len(subset):,})",
                            line=dict(color=C_INC if not key else C_INC2, width=2.5, dash=dash),
                            marker=dict(size=12, line=dict(width=2, color=SURFACE)),
                            text=[f"{near:+.1f}%", f"{far:+.1f}%"], textposition="middle right",
                            textfont=dict(color=INK, size=12),
                            hovertemplate="%{x}: <b>%{y:+.1f}%</b><extra></extra>")
        fig.add_hline(y=0, line=dict(color=AXIS_LINE(), width=1.5))
        st.plotly_chart(style(fig, height=380, yaxis_title="vs untouched sites (%)",
                              xaxis=dict(range=[-0.4, 1.6]),
                              legend=dict(orientation="h", y=1.02, x=0)), width="stretch")
    with right:
        gap = (tr[f"near_did_{metric}"] - tr[f"far_did_{metric}"]).dropna()
        figh = go.Figure(go.Histogram(x=gap.clip(-60, 60), nbinsx=34,
                                      marker=dict(color=C_INC, line=dict(width=1, color=SURFACE)),
                                      hovertemplate="gap %{x:.0f} pp · %{y} events<extra></extra>"))
        figh.add_vline(x=0, line=dict(color=AXIS_LINE(), width=1.5))
        figh.add_vline(x=float(gap.median()), line=dict(color=C_ENT, width=2, dash="dash"))
        figh.add_annotation(x=float(gap.median()), y=1, yref="paper", yanchor="bottom",
                            text=f"median {gap.median():+.1f} pp", showarrow=False,
                            font=dict(color=C_ENT, size=11))
        st.plotly_chart(style(figh, height=380, xaxis_title="Nearer minus further (pp)",
                              yaxis_title="Events", margin=dict(t=40)), width="stretch")

    tbl = pd.DataFrame({
        "Cut": ["All triples", f"Nearest under {th['close_mi']:.0f} mi"],
        "Events": [th["n"], th["n_close"]],
        "Nearer": [th["near"], th["close_near"]],
        "Further": [th["far"], th["close_far"]],
        "Gap": [th["gap"], th["close_gap"]],
        "Nearer worse": [th["share_near_worse"], th["close_share_near_worse"]],
        "Median distances": [f"{th['near_mi']:.1f} / {th['far_mi']:.1f} mi",
                             f"< {th['close_mi']:.0f} / {th['close_far_mi']:.1f} mi"]})
    tbl.index = range(1, len(tbl) + 1)
    html_table(tbl, fmt={"Events": "{:,.0f}", "Nearer": "{:+.1f}%", "Further": "{:+.1f}%",
                         "Gap": "{:+.1f} pp", "Nearer worse": "{:.0%}"})

    callout("What this shows — and it argues against the chart above it", f"""
      <b>Inside a single market, being the closer site does not make it worse.</b> The nearer
        neighbour comes out <b>{th['gap']:+.1f} pp</b> from the further one, and it is the worse
        of the two in <b>{th['share_near_worse']:.0%}</b> of events, a coin flip. Restricting to
        the {th['n_close']} events where the nearest is under {th['close_mi']:.0f} miles does not
        change it: <b>{th['close_gap']:+.1f} pp</b>.
      <b>Both of them get hit.</b> In those close events the further neighbour, a median
        {th['close_far_mi']:.1f} miles away, is down <b>{th['close_far']:+.1f}%</b>, almost
        exactly what the nearer one loses.
      <b>So the two-body distance gradient is partly about markets, not about metres.</b> Openings
        that land within a mile of somebody tend to land in dense, contested markets; those
        markets lose ground as a whole. Pooling across markets, that reads as "distance matters".
        Within one market, it does not.
      <b>What it means for a pin.</b> Do not ask "how far is my nearest competitor" and stop
        there. The exposure is the market you are entering, and every site in it is exposed,
        including the ones a comfortable five miles away.
      <b>Together the three sites still grow.</b> The two neighbours plus the entrant wash
        <b>{th['market']:+.0f}%</b> more than the two neighbours did alone.
    """, S2)

    # --- saturation ------------------------------------------------------------------------------
    st.divider()
    st.markdown("#### How many openings can one market take?")
    st.caption("The same data cut the other way: one row per neighbour per burst of openings, so a "
               "site hit three times in a quarter is counted once, against three.")
    sat = _saturation(radius, window, min_age, metric)
    if not sat.empty:
        figs = go.Figure()
        figs.add_bar(x=sat.label, y=sat.did,
                     marker=dict(color=[LOSS if v < 0 else GAIN for v in sat.did],
                                 line=dict(width=2, color=SURFACE)),
                     customdata=np.stack([sat.incumbents, sat.nearest_mi], axis=-1),
                     hovertemplate="<b>%{x} openings</b><br>%{y:+.1f}% vs untouched<br>"
                                   "nearest a median %{customdata[1]:.1f} mi<br>"
                                   "<span style='opacity:.7'>%{customdata[0]} neighbours</span>"
                                   "<extra></extra>", showlegend=False)
        for _, r in sat.iterrows():
            figs.add_scatter(x=[r.label] * 2, y=[r.p25, r.p75], mode="lines",
                             line=dict(color=MUTED, width=1.5), hoverinfo="skip", showlegend=False)
            figs.add_annotation(x=r.label, y=r.did, text=f"{r.did:+.1f}%", showarrow=False,
                                yshift=-16 if r.did < 0 else 14, font=dict(color=INK, size=12))
        figs.add_hline(y=0, line=dict(color=AXIS_LINE(), width=1.5))
        st.plotly_chart(style(figs, height=340, bargap=0.5,
                              xaxis_title=f"Openings within {radius:.0f} miles in the same window",
                              yaxis_title="vs untouched sites (%)"), width="stretch")
        rng = sat.did.max() - sat.did.min()
        callout("What this shows", f"""
          <b>There is no dose–response here.</b> A neighbour hit by one opening and a neighbour hit
            by {sat.label.iloc[-1]} come out within <b>{rng:.1f} pp</b> of each other, in no
            consistent order.
          <b>Read that as a ceiling on what this dataset can settle</b>, not as "saturation is
            free". Only {int(sat.incumbents.iloc[-1])} neighbours are in the busiest bucket, and a
            market that attracts four openings at once was probably growing to begin with. The
            thing that would separate those is a control for market growth we do not have.
        """, S2)


# =================================================================================================
# ③ the entrant
# =================================================================================================

def _entrant(pairs: pd.DataFrame, metric: str, window: int, noun: str) -> None:
    st.subheader("And the site you are about to build?")
    st.caption("Every entrant in the cut, grouped by the competition it walked into. **Opening** "
               f"is its first {window} months; **mature** is months 12–24, which §⓪ shows is "
               "roughly its settled level — so it is blank for anything opened in the last two "
               "years.")

    en = cd.entrant_outcomes(pairs, metric, window)
    by_dist = cd.entrant_by_crowding(en, "nearest_band")
    by_crowd = cd.entrant_by_crowding(en, "crowding")

    left, right = st.columns(2)
    for col, frame, key, title in ((left, by_dist, "nearest_band", "By distance to the nearest"),
                                   (right, by_crowd, "crowding",
                                    "By how many sit within 3 miles")):
        with col:
            st.markdown(f"**{title}**")
            fig = go.Figure()
            fig.add_bar(x=frame[key].astype(str), y=frame.opening, name="Opening 6 months",
                        marker=dict(color=C_ENT, line=dict(width=2, color=SURFACE)),
                        customdata=frame.entrants,
                        hovertemplate="<b>%{x}</b><br>%{y:,.0f} " + noun +
                                      "<br><span style='opacity:.7'>%{customdata} sites</span>"
                                      "<extra></extra>")
            fig.add_bar(x=frame[key].astype(str), y=frame.mature, name="Months 12–24",
                        marker=dict(color=C_INC, line=dict(width=2, color=SURFACE)),
                        customdata=frame.n_mature,
                        hovertemplate="<b>%{x}</b><br>%{y:,.0f} " + noun +
                                      "<br><span style='opacity:.7'>%{customdata} sites</span>"
                                      "<extra></extra>")
            st.plotly_chart(style(fig, height=340, barmode="group", bargap=0.35, bargroupgap=0.08,
                                  yaxis_title=f"{cd.METRICS[metric][0]} a month",
                                  legend=dict(orientation="h", y=1.02, x=0)),
                            width="stretch", key=f"ent_{key}")

    show = by_crowd.copy()
    show["crowding"] = show.crowding.astype(str)
    show.columns = ["Sites within 3 mi", "Entrants", "Opening 6 months", "Months 12–24",
                    "With 12–24 data"]
    show.index = range(1, len(show) + 1)
    html_table(show, fmt={"Entrants": "{:,.0f}", "Opening 6 months": "{:,.0f}",
                          "Months 12–24": "{:,.0f}", "With 12–24 data": "{:,.0f}"})

    # Medians over the entrant rows themselves, never an average of the two crowded buckets'
    # medians — the "3+" bucket has four sites with a mature figure and would carry equal weight.
    alone, crowded = en[en.neighbours_3mi == 0], en[en.neighbours_3mi >= 2]
    if len(alone) and len(crowded):
        a_open, c_open = float(alone.opening.median()), float(crowded.opening.median())
        a_mat, c_mat = float(alone.mature.median()), float(crowded.mature.median())
        n_crowd, n_crowd_mat = int(len(crowded)), int(crowded.mature.notna().sum())
        callout("What this shows", f"""
          <b>The competition costs the new site more than it costs the neighbour.</b> An entrant
            with nobody inside three miles opens at <b>{_fmt(a_open, metric)}</b> {noun} a month;
            one with two or more opens at <b>{_fmt(c_open, metric)}</b>, about
            <b>{(1 - c_open / a_open):.0%} less</b>. The neighbour, on the previous tab, gives up
            single digits.
          <b>And it does not catch up.</b> By months 12–24 the gap is still there:
            <b>{_fmt(a_mat, metric)}</b> against <b>{_fmt(c_mat, metric)}</b>. Whatever the
            crowded market costs, it costs at the settled level, not just during the ramp.
          <b>Hold this loosely.</b> Only <b>{n_crowd}</b> entrants have two or more neighbours that
            close and only <b>{n_crowd_mat}</b> of those have matured enough to score, and an
            operator who builds into a busy market may be picking a different kind of site to begin
            with. The direction is consistent; the size is on thin numbers.
          <b>The asymmetry is the decision-relevant part.</b> The question a pin should ask is not
            "how much will I hurt the neighbour", which is a few percent. It is "how much smaller is
            this site than the same building somewhere emptier".
        """, S3)

    st.markdown("**Every entrant, one dot**")
    st.caption("Mature volume against the distance to its nearest existing neighbour. Openings in "
               "the last two years have no mature figure and are not drawn.")
    e = en.dropna(subset=["mature"])
    figs = go.Figure(go.Scatter(
        x=e.nearest_mi, y=e.mature, mode="markers",
        marker=dict(size=8, color=C_ENT, opacity=0.55, line=dict(width=1, color=SURFACE)),
        customdata=np.stack([e.operator, e.state, e.neighbours_3mi,
                             e.event_month.dt.strftime("%b %Y")], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{y:,.0f} " + noun +
                      " a month<br>nearest %{x:.1f} mi · %{customdata[2]:.0f} within 3 mi<br>"
                      "<span style='opacity:.7'>opened %{customdata[3]}</span><extra></extra>"))
    med = e.groupby(pd.cut(e.nearest_mi, cd.DIST_BINS, labels=cd.DIST_LABELS,
                           right=False), observed=True).agg(
        mid=("nearest_mi", "median"), val=("mature", "median")).dropna()
    figs.add_scatter(x=med.mid, y=med.val, mode="lines+markers", name="band median",
                     line=dict(color=C_INC, width=2.5),
                     marker=dict(size=10, line=dict(width=2, color=SURFACE)),
                     hovertemplate="median <b>%{y:,.0f}</b><extra></extra>")
    st.plotly_chart(style(figs, height=400, xaxis_title="Miles to the nearest existing site",
                          yaxis_title=f"{cd.METRICS[metric][0]} a month, months 12–24",
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")


# =================================================================================================
# ④ every case
# =================================================================================================

def _cases(pairs: pd.DataFrame, metric: str, window: int) -> None:
    st.subheader("Every case, one at a time")
    st.caption("Nothing here is averaged. Pick any opening in the panel and see the two sites' "
               "actual monthly wash counts, where they sit relative to each other, how far apart "
               "they are and when each of them opened.")
    st.caption("**A single case is not an attribution.** Order by *biggest neighbour loss* and the "
               "first sites you meet are ones that were already winding down — a gradual slide to "
               "near zero that no filter can separate from competition without also deleting the "
               "finding. The evidence is the median across every pair; these are for seeing what "
               "the median is made of.")

    f1, f2, f3, f4 = st.columns([1.1, 1, 1, 1.3])
    states = ["All"] + sorted(pairs.state.dropna().unique().tolist())
    state = f1.selectbox("State", states)
    bands = ["All"] + [b for b in cd.DIST_LABELS if b in set(pairs.distance_band.astype(str))]
    band = f2.selectbox("Distance", bands)
    regimes = ["All"] + [r for r in cd.regime_mix(pairs).regime if r != "unknown"]
    regime = f3.selectbox("Outcome", regimes)
    # "Closest" leads deliberately. Ordering by the biggest loss puts the most extreme pair on
    # screen first, and the extreme tail is where sites that were quietly winding down anyway
    # live — see the caption below.
    sort_by = f4.selectbox("Order by", ["Closest", "Biggest neighbour loss",
                                        "Biggest neighbour gain", "Most recent"])

    sel = pairs.copy()
    if state != "All":
        sel = sel[sel.state == state]
    if band != "All":
        sel = sel[sel.distance_band.astype(str) == band]
    if regime != "All":
        sel = sel[sel.regime == regime]
    sel = sel.dropna(subset=[f"pct_{metric}"])
    if sel.empty:
        st.info("Nothing matches those filters.")
        return
    sel = sel.sort_values(
        {"Biggest neighbour loss": f"did_{metric}", "Biggest neighbour gain": f"did_{metric}",
         "Closest": "distance_mi", "Most recent": "event_month"}[sort_by],
        ascending=sort_by not in ("Biggest neighbour gain", "Most recent"))

    st.caption(f"**{len(sel):,}** of {len(pairs):,} pairs match.")
    pick = st.selectbox("Case", sel.label.tolist(), index=0, key="case_pick")
    row = sel[sel.label == pick].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Apart", f"{row.distance_mi:.1f} mi")
    k2.metric("Neighbour, raw", _pp(row[f"pct_{metric}"]),
              f"{_fmt(row[f'pre_{metric}'], metric)} → {_fmt(row[f'post_{metric}'], metric)}",
              delta_color="off")
    k3.metric("vs untouched sites", _pp(row[f"did_{metric}"]),
              f"they did {_pp(row[f'ctrl_{metric}'])}", delta_color="off")
    k4.metric("New site, first months", _fmt(row[f"entrant_post_{metric}"], metric),
              f"pair {_pp(row[f'combined_pct_{metric}'])}", delta_color="off")

    left, right = st.columns([1, 1.35])
    with left:
        st.plotly_chart(_geometry_fig(row), width="stretch", key="case_geo")
    with right:
        st.plotly_chart(_case_series_fig(row, metric, window), width="stretch", key="case_ts")

    facts = pd.DataFrame({
        "": ["Neighbour", "New site"],
        "Operator": [row.incumbent_operator, row.entrant_operator],
        "Address": [f"{row.incumbent_address}, {row.state}", f"{row.entrant_address}, {row.state}"],
        "Opened": [pd.Timestamp(row.incumbent_opened).strftime("%b %Y"),
                   pd.Timestamp(row.entrant_opened).strftime("%b %Y")],
        "Coordinates": [f"{row.incumbent_lat:.4f}, {row.incumbent_lon:.4f}",
                        f"{row.entrant_lat:.4f}, {row.entrant_lon:.4f}"]})
    facts.index = range(1, 3)
    html_table(facts)
    st.caption(f"Neighbour was **{int(row.incumbent_age_mo)} months old** when the new site "
               f"opened in **{pd.Timestamp(row.event_month).strftime('%B %Y')}**. Windows: "
               f"{int(row.n_pre)} months before, {int(row.n_post)} after. Outcome classed "
               f"**{row.regime.lower()}**. Counterfactual matched at the "
               f"**{row.ctrl_level or '—'}** level"
               + (f" on {int(row.n_control)} untouched sites." if np.isfinite(row.n_control)
                  else "."))

    with st.expander(f"The matching cases as a table ({len(sel):,})"):
        t = sel.head(200)[["incumbent_operator", "entrant_operator", "state", "distance_mi",
                           "event_month", f"pct_{metric}", f"did_{metric}", "regime"]].copy()
        t["event_month"] = t.event_month.dt.strftime("%b %Y")
        t.columns = ["Neighbour", "New site", "State", "Miles", "Opened", "Raw", "vs untouched",
                     "Outcome"]
        t.index = range(1, len(t) + 1)
        html_table(t, fmt={"Miles": "{:.1f}", "Raw": "{:+.1f}%", "vs untouched": "{:+.1f}%"})
        st.caption(f"First 200 of {len(sel):,} — the download has all of them.")
        st.download_button("Download every matching case (CSV)", sel.to_csv(index=False),
                           "competition_cases.csv", "text/csv")


def _geometry_fig(row: pd.Series) -> go.Figure:
    """Where the two sites sit relative to each other, in miles, with distance rings.

    Not a basemap. At two miles apart a country-scale basemap is a blank field with two dots on it,
    and there is no tile layer in this app to zoom into — so the geometry is drawn directly: the
    new site at the origin, the neighbour placed at its true bearing and distance, north up.
    """
    lat0, lon0 = float(row.entrant_lat), float(row.entrant_lon)
    dy = (float(row.incumbent_lat) - lat0) * 69.0
    dx = (float(row.incumbent_lon) - lon0) * 69.0 * np.cos(np.radians(lat0))
    span = max(abs(dx), abs(dy), 1.0) * 1.45

    fig = go.Figure()
    for r in (1, 3, 5, 10):
        if r > span * 1.2:
            continue
        th = np.linspace(0, 2 * np.pi, 120)
        fig.add_scatter(x=r * np.cos(th), y=r * np.sin(th), mode="lines",
                        line=dict(color=GRID, width=1, dash="dot"), hoverinfo="skip",
                        showlegend=False)
        fig.add_annotation(x=0, y=r, text=f"{r} mi", showarrow=False, yshift=8,
                           font=dict(color=MUTED, size=10))
    fig.add_scatter(x=[0, dx], y=[0, dy], mode="lines",
                    line=dict(color=MUTED, width=1.5, dash="dash"), hoverinfo="skip",
                    showlegend=False)
    fig.add_scatter(x=[dx], y=[dy], mode="markers+text", name="Neighbour",
                    marker=dict(size=17, color=C_INC, line=dict(width=2, color=SURFACE)),
                    text=["neighbour"], textposition="top center", textfont=dict(color=INK2),
                    hovertemplate=f"<b>{row.incumbent_operator}</b><br>"
                                  f"{row.distance_mi:.1f} mi away<br>opened "
                                  f"{pd.Timestamp(row.incumbent_opened).strftime('%b %Y')}"
                                  "<extra></extra>")
    fig.add_scatter(x=[0], y=[0], mode="markers+text", name="New site",
                    marker=dict(size=17, symbol="diamond", color=C_ENT,
                                line=dict(width=2, color=SURFACE)),
                    text=["new site"], textposition="bottom center", textfont=dict(color=INK2),
                    hovertemplate=f"<b>{row.entrant_operator}</b><br>opened "
                                  f"{pd.Timestamp(row.entrant_opened).strftime('%b %Y')}"
                                  "<extra></extra>")
    return style(fig, height=380, showlegend=False,
                 title=dict(text=f"{row.distance_mi:.1f} miles apart · north is up"),
                 xaxis=dict(range=[-span, span], title="miles east", scaleanchor="y",
                            zeroline=True, zerolinecolor=GRID),
                 yaxis=dict(range=[-span, span], title="miles north", zeroline=True,
                            zerolinecolor=GRID),
                 margin=dict(l=50, r=20, t=45, b=45))


def _case_series_fig(row: pd.Series, metric: str, window: int) -> go.Figure:
    """Both sites' actual monthly series, with the opening month and the two windows marked."""
    inc = _series(row.incumbent, metric)
    ent = _series(row.entrant, metric)
    ev = pd.Timestamp(row.event_month)

    fig = go.Figure()
    fig.add_vrect(x0=ev - pd.DateOffset(months=window), x1=ev,
                  fillcolor="rgba(137,135,129,0.12)", line_width=0, layer="below")
    fig.add_vrect(x0=ev, x1=ev + pd.DateOffset(months=window),
                  fillcolor="rgba(217,89,38,0.10)", line_width=0, layer="below")
    fig.add_scatter(x=inc.month, y=inc.value, mode="lines", name="Neighbour",
                    line=dict(color=C_INC, width=2.5),
                    hovertemplate="%{x|%b %Y} · <b>%{y:,.0f}</b><extra>neighbour</extra>")
    fig.add_scatter(x=ent.month, y=ent.value, mode="lines", name="New site",
                    line=dict(color=C_ENT, width=2.5, dash="dash"),
                    hovertemplate="%{x|%b %Y} · <b>%{y:,.0f}</b><extra>new site</extra>")
    fig.add_scatter(x=[ev - pd.DateOffset(months=window), ev], y=[row[f"pre_{metric}"]] * 2,
                    mode="lines", line=dict(color=C_INC, width=1.5, dash="dot"),
                    hovertemplate="before: <b>%{y:,.0f}</b><extra></extra>", showlegend=False)
    fig.add_scatter(x=[ev, ev + pd.DateOffset(months=window)], y=[row[f"post_{metric}"]] * 2,
                    mode="lines", line=dict(color=C_INC, width=1.5, dash="dot"),
                    hovertemplate="after: <b>%{y:,.0f}</b><extra></extra>", showlegend=False)
    fig.add_vline(x=ev.timestamp() * 1000, line=dict(color=MUTED, width=1.5))
    fig.add_annotation(x=ev, y=1, yref="paper", yanchor="bottom", text="new site opens",
                       showarrow=False, font=dict(color=MUTED, size=11))
    return style(fig, height=380, yaxis_title=cd.METRICS[metric][0],
                 legend=dict(orientation="h", y=1.02, x=0), margin=dict(t=55))


# =================================================================================================
# ⑤ method
# =================================================================================================

def _method(pairs: pd.DataFrame, radius: float, window: int, min_age: int) -> None:
    st.subheader("Data & method")
    st.markdown(f"""
**One input file** — `conclusion/data/historical_data_5yrs_monthly.csv`, the monthly wash panel:
2,103 sites, Jan 2020 → Jun 2026, each with its `operational_start`, coordinates and state. Nothing
is joined in from anywhere else, exactly as sections ① ② ③ each read their own single file.

**An event** is an opening. Every already-trading site within **{radius:.0f} miles** of it is a
neighbour exposed to that event, and each (entrant, neighbour) pair is one row — not just the
nearest one, which is what the archive's `interaction_outputs_nochem_v2` run measured on 85 pairs.
The window is **{window} months** either side of the opening month; the entrant contributes only a
post window, having no past.

**The counterfactual.** The same before/after change is computed for every *untouched* site — one
with no opening inside {radius:.0f} miles anywhere in the window — and the neighbour is scored
against the median of untouched sites in its own census region, age bracket and calendar months.
Where that cell holds fewer than 10 sites the match loosens to age-and-month nationwide, and the
row records which level it got. **{(pairs.ctrl_level == 'region_age').mean():.0%}** of pairs are
matched at the tightest level.
    """)
    callout("Four things that would make these numbers wrong, and what is done about each",
            f"""
      <b>The neighbour's own ramp.</b> A young site is climbing regardless, which cancels the
        entrant's damage. Neighbours must be <b>{min_age} months old</b> at the opening (the
        sidebar moves it), and controls are matched inside the same age bracket. This is the exact
        confound that overturned §③.
      <b>Seasonality and the market.</b> Both are absorbed by the counterfactual: the control sites
        are living through the same calendar months in the same region.
      <b>Sites that were never really trading, and sites that stopped.</b> A neighbour must have
        been washing <b>{int(pairs.attrs.get('min_pre_volume', 500))} cars a month</b> before the
        opening, and must not go dark afterwards. <b>Two or more months under 5%</b> of its own
        past means it closed or stopped reporting, which several verified offenders did (the series
        hits a literal zero and stays). That test is on the raw months, not on the window average,
        so a neighbour that genuinely halved is kept.
      <b>Openings that never opened.</b> The entrant must itself reach
        <b>{int(pairs.attrs.get('min_entrant_volume', 250))} washes a month</b>; without that gate
        a placeholder row doing one wash a month gets paired with an incumbent that happened to
        close, and reads as a −79% effect caused by nothing.
        <b>{pairs.attrs.get('n_skipped_entrants', 0)}</b> openings are excluded on it. All the
        exclusions together move the headline by about 0.2 pp. The finding does not live in the
        tail.
      <b>Operator handoffs.</b> A wash that changes hands reappears under a new
        <code>client_id</code>, and the old key dies the same month, a 100% "collapse" caused by an
        entrant thirty feet away. Pairs closer than <b>0.2 miles</b> are dropped.
      <b>Left-censored dates.</b> <code>operational_start</code> equals the site's first month in
        the panel for every site, so the 348 sites stamped 2020-01 are "open by then", not "opened
        then". They are used as neighbours and never as entrants; entrants start from 2020-07, the
        first month with a full pre-window inside the panel.
    """, S2)
    tr = _triples(radius, window, min_age)
    n_close = int((tr.near_distance_mi < 2.0).sum()) if not tr.empty else 0
    callout("What this section still cannot tell you", f"""
      <b>Nobody randomised anything.</b> Operators choose where to open, and they choose markets
        they expect to grow. If they pick well, the untouched control understates what the
        neighbour would have done, and the effect here is if anything too small.
      <b>The three-body null is not a proof of no proximity effect.</b> It says that inside a
        single market, being under two miles away rather than four does not measurably change the
        outcome, on {n_close} close events, which is enough to rule out a large gap and not enough
        to rule out a small one.
      <b>The entrant-side numbers rest on thin cells.</b> Only a few dozen entrants open with two
        or more neighbours inside three miles and have matured enough to score.
      <b>Nothing here is a forecast.</b> It is what happened to sites we hold data for. The
        forecasting model in <code>proforma/models/coldstart.py</code> is where a pin gets a
        number; this section says what the cannibalisation term in it should look like.
    """, S1)
    st.caption("Reference working outputs: `archive/hypothesis-testing/interaction_outputs_"
               "nochem_v2/` (the original two/three/four-body run, 85 / 42 / 25 events, "
               "single-nearest matching, no counterfactual).")
