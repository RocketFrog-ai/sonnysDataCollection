"""
Section ② — proforma backtest: when the proforma projected a wash count, how close did it land?

A demo surface: charts, the findings that come out of them, and an explorer the reader can drive.
All the maths lives in `proforma_data.py` (Streamlit-free, shared with the notebook).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import proforma_data as pf
from ui import INK, INK2, MUTED, S1, S2, S3, SURFACE, callout, html_table, style

# One fixed hue per forecaster, assigned in order and never cycled.
FCOLOR = {"proforma": S2, "coldstart": S3, "model5": S1}
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return pf.load(drop_collapsed=True)


@st.cache_data(show_spinner=False)
def _by_year(full_years: bool) -> pd.DataFrame:
    return pf.by_year(_load(), full_years_only=full_years)


@st.cache_data(show_spinner=False)
def _factors() -> pd.DataFrame:
    return pf.factor_table(_load())


@st.cache_data(show_spinner=False)
def _impact() -> pd.DataFrame:
    return pf.factor_impact(_load())


def render() -> None:
    d = _load()
    h = pf.headline(d)
    sc = pf.scorecard(d)
    h2h = pf.head_to_head(d)

    st.markdown("<div class='kicker'>Capital allocation · evidence pack</div>",
                unsafe_allow_html=True)
    st.title("Old proforma (Excel) Backtest")
    st.metric("Sites backtested", f"{h['n_sites']}")

    # =============================================================================================
    st.divider()
    st.header("1 · Every projection against what actually happened")
    st.caption("One dot per site. The dashed line is a perfect projection — anything above it was "
               "over-projected.")

    a, p = d.actual_mature_wash, d.proforma_y5
    over = p > a
    lim = [0, float(max(a.max(), p.max())) * 1.05]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", name="perfect projection",
                    line=dict(color=INK2, width=2, dash="dash"), hoverinfo="skip")
    for mask, name, col in [(over, "Over-projected", BAD), (~over, "Under-projected", S1)]:
        g = d[mask]
        ratio = (g.proforma_y5 / g.actual_mature_wash)
        fig.add_scatter(x=g.actual_mature_wash, y=g.proforma_y5, mode="markers", name=name,
                        marker=dict(size=11, color=col, line=dict(width=1.6, color=SURFACE)),
                        customdata=np.stack([g.client_name.fillna("—"), g.state.fillna("—"),
                                             ratio, g.open_year], axis=-1),
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} · opened "
                                      "%{customdata[3]:.0f}<br>"
                                      "Proforma said: <b>%{y:,.0f}</b> washes/mo<br>"
                                      "Actually delivered: <b>%{x:,.0f}</b> washes/mo<br>"
                                      "→ <b>%{customdata[2]:.1f}× the real volume</b><extra></extra>")
    st.plotly_chart(style(fig, height=520, xaxis_title="Washes the site actually does (per month)",
                          yaxis_title="Washes the proforma promised (per month)",
                          xaxis=dict(range=lim, constrain="domain"),
                          yaxis=dict(range=lim, scaleanchor="x", constrain="domain"),
                          margin=dict(l=60, r=25, t=72, b=50),
                          legend=dict(orientation="h", y=1.06, x=0)), width="stretch")

    callout("What this shows", f"""
      <b>Reading.</b> <b>{h['proforma_over_share']*100:.0f}% of the dots sit above the line</b> —
        those sites washed fewer cars than promised. The typical proforma promised
        <b>{h['proforma_bias']:.2f}× what the site actually delivers</b>, and one site in ten was
        promised <b>{h['proforma_p90']:.1f}× or more</b>.
      <b>So-what.</b> This is a consistent lean in one direction, not random scatter. Roughly three
        sites in four were signed off on a number that never arrived, and the worst cases were out
        by several times over — not by a few percent.
    """, BAD)

    # =============================================================================================
    st.divider()
    st.header("2 · Everything about one site")
    st.caption("Pick a site: what it actually washed, what it was built with, and the tunnel it got.")

    fcol, scol = st.columns([1, 2])
    with fcol:
        mode = st.radio("Order sites by", ["Most over-projected", "Most under-projected",
                                           "Largest sites", "Name"], index=0)
    v = d.assign(ratio=d.proforma_y5 / d.actual_mature_wash)
    key, asc = {"Most over-projected": ("ratio", False),
                "Most under-projected": ("ratio", True),
                "Largest sites": ("actual_mature_wash", False),
                "Name": ("client_name", True)}[mode]
    v = v.sort_values(key, ascending=asc)
    with scol:
        labels = {f"{r.client_name} — {r.state}  ({r.ratio:.1f}× projected)": r.site_key
                  for r in v.itertuples()}
        picked = st.selectbox("Site", list(labels), index=0)
    site_key = labels[picked]
    row = v[v.site_key == site_key].iloc[0]

    unit = st.radio("Show washes as", ["Per month", "Per day", "Per open hour"],
                    horizontal=True, index=0)
    suffix = {"Per month": "", "Per day": "_daily", "Per open hour": "_hourly"}[unit]
    fmt = ",.0f" if unit == "Per month" else ",.1f"
    t_ = pf.site_volume_views(d, site_key)
    hours = t_.attrs.get("open_hours_per_day", float("nan"))

    figt = go.Figure()
    figt.add_bar(x=t_.year, y=t_["actual" + suffix], name="Actually washed",
                 marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
                 customdata=t_.observed_months,
                 hovertemplate=f"Operating year %{{x}}<br>Actually washed: <b>%{{y:{fmt}}}</b>"
                               f" {unit.lower().replace('per ', 'per ')}<br>"
                               "<span style='opacity:.7'>%{customdata:.0f} months of data"
                               "</span><extra></extra>")
    for col, ckey, label in [("proforma", "proforma", "What the proforma promised"),
                             ("model5", "model5", "Model 5")]:
        figt.add_scatter(x=t_.year, y=t_[col + suffix], mode="lines+markers", name=label,
                         line=dict(color=FCOLOR[ckey], width=3),
                         marker=dict(size=10, line=dict(width=2, color=SURFACE)),
                         hovertemplate=f"Operating year %{{x}}<br>{label}: "
                                       f"<b>%{{y:{fmt}}}</b><extra></extra>")
    st.plotly_chart(style(figt, height=400, xaxis_title="Operating year",
                          yaxis_title=f"Washes {unit.lower()}", xaxis=dict(dtick=1),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")
    if unit != "Per month" and np.isfinite(hours):
        st.caption(f"Converted at 365 days a year and **{hours:.0f} open hours a day** — this "
                   "site's own trading hours from its proforma.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Actually washes", f"{row.actual_mature_wash:,.0f}", "per month at maturity",
              delta_color="off")
    k2.metric("Proforma promised", f"{row.proforma_y5:,.0f}", f"{row.ratio:.1f}× what it does",
              delta_color="off")
    k3.metric("Model 5", f"{row.model5_mature:,.0f}",
              f"{row.model5_mature/row.actual_mature_wash:.1f}× what it does", delta_color="off")
    k4.metric("Opened", f"{row.open_year:.0f}", row.proforma_type, delta_color="off")

    tun = pf.site_tunnel(d, site_key)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**The tunnel: recommended vs built**")
        if np.isfinite(tun.get("actual_m", float("nan"))):
            figtl = go.Figure(go.Bar(
                x=[tun["recommended_m"], tun["actual_m"]],
                y=["Formula recommends", "Actually built"], orientation="h",
                marker=dict(color=[S2, S1], line=dict(width=2, color=SURFACE)),
                text=[f"{tun['recommended_m']:.0f} m", f"{tun['actual_m']:.0f} m"],
                textposition="outside", textfont=dict(color=INK, size=13),
                hovertemplate="%{y}: <b>%{x:.0f} m</b><extra></extra>"))
            st.plotly_chart(style(figtl, height=250, xaxis_title="Tunnel length (m)",
                                  xaxis=dict(range=[0, max(tun["recommended_m"],
                                                           tun["actual_m"]) * 1.3]),
                                  yaxis=dict(showgrid=False)), width="stretch")
            gap = tun["gap_m"]
            word = "longer than" if gap > 0 else "shorter than"
            st.markdown(f"Built **{abs(gap):.0f} m {word}** the formula asked for. The formula's "
                        f"{tun['recommended_m']:.0f} m comes straight from the proforma's year-5 "
                        f"peak of **{tun['recommended_ft']:.0f} cars an hour** — one foot per car.")
        else:
            st.info("No measured tunnel length on file for this site.")
    with right:
        st.markdown("**What it was built with**")
        prof = pf.site_profile(d, site_key)
        prof.index = range(1, len(prof) + 1)
        html_table(prof, fmt={"Score": "{:+.3f}"})

    gap_w = row.proforma_y5 - row.actual_mature_wash
    st.markdown(
        f"**{row.client_name}** ({row.state}) opened in {row.open_year:.0f}. The proforma promised "
        f"**{row.proforma_y5:,.0f}** washes a month; it does **{row.actual_mature_wash:,.0f}** — "
        f"a gap of **{abs(gap_w):,.0f} washes a month**.")

    # =============================================================================================
    st.divider()
    st.header("3 · Does the projection get better as the site matures?")
    st.caption("How far each forecast sat from the washes the site actually did, year by year.")

    by = _by_year(True)
    figy = go.Figure()
    for key_, label, _, _, _ in pf.FORECASTERS:
        g = by[by.key == key_]
        figy.add_scatter(x=g.year, y=g.mdape, mode="lines+markers", name=label,
                         line=dict(color=FCOLOR[key_], width=3),
                         marker=dict(size=10, line=dict(width=2, color=SURFACE)),
                         customdata=np.stack([g.n, g.bias], axis=-1),
                         hovertemplate=f"<b>{label}</b> — operating year %{{x}}<br>"
                                       "Typical miss: <b>%{y:.0f}%</b><br>"
                                       "Runs at %{customdata[1]:.2f}× actual<br>"
                                       "<span style='opacity:.7'>%{customdata[0]} sites"
                                       "</span><extra></extra>")
    st.plotly_chart(style(figy, height=430, xaxis_title="Operating year",
                          yaxis_title="Typical miss (%)", xaxis=dict(dtick=1),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")

    p_y1 = float(by[(by.key == "proforma") & (by.year == 1)].mdape.iloc[0])
    p_y5 = float(by[(by.key == "proforma") & (by.year == 5)].mdape.iloc[0])
    m_y5 = float(by[(by.key == "model5") & (by.year == 5)].mdape.iloc[0])
    callout("What this shows", f"""
      <b>Reading.</b> The proforma does not get better with time — it is out by {p_y1:.0f}% in
        year 1 and <b>{p_y5:.0f}% by year 5</b>. Model 5 goes the other way, closing to
        {m_y5:.0f}%.
      <b>So-what.</b> The proforma is at its <b>least accurate in years 4 and 5</b> — the years the
        loan is being repaid on, and the whole reason the projection is made.
    """, BAD)

    # =============================================================================================
    st.divider()
    st.header("4 · The three forecasters side by side")
    st.caption("The same sites judged against the same washes. Model 5 has never seen the site "
               "it is predicting.")

    left, right = st.columns(2)
    with left:
        figb = go.Figure(go.Bar(
            x=sc.forecaster, y=sc.mdape,
            marker=dict(color=[FCOLOR[k] for k in sc.key], line=dict(width=2, color=SURFACE)),
            text=[f"{x:.0f}%" for x in sc.mdape], textposition="outside",
            textfont=dict(color=INK, size=14),
            customdata=np.stack([sc.bias, sc.within_25], axis=-1),
            hovertemplate="<b>%{x}</b><br>Typical miss: <b>%{y:.0f}%</b><br>"
                          "Runs at %{customdata[0]:.2f}× actual<br>"
                          "Lands within ±25% on %{customdata[1]:.0%} of sites<extra></extra>"))
        st.plotly_chart(style(figb, height=400, yaxis_title="Typical miss (%)",
                              yaxis=dict(range=[0, sc.mdape.max()*1.25]),
                              xaxis=dict(showgrid=False)), width="stretch")
    with right:
        figc = go.Figure()
        # bottom-right: the dot labels sit above their markers, so a top annotation collides
        figc.add_vline(x=1.0, line=dict(color=INK2, width=2, dash="dash"),
                       annotation_text="perfect", annotation_position="bottom right",
                       annotation_font=dict(color=INK2, size=11))
        figc.add_scatter(x=sc.bias, y=sc.forecaster, mode="markers+text",
                         marker=dict(size=20, color=[FCOLOR[k] for k in sc.key],
                                     line=dict(width=2, color=SURFACE)),
                         text=[f"{b:.2f}×" for b in sc.bias], textposition="top center",
                         textfont=dict(color=INK, size=12), showlegend=False,
                         hovertemplate="%{y}<br>median predicted/actual %{x:.2f}<extra></extra>")
        st.plotly_chart(style(figc, height=400, xaxis_title="Promised ÷ delivered",
                              yaxis=dict(showgrid=False)), width="stretch")

    tbl = sc[["forecaster", "n", "mdape", "bias", "over_share", "within_25", "p90"]].copy()
    tbl.columns = ["Forecaster", "Sites", "Median error", "Bias", "Over-projected",
                   "Within ±25%", "p90 ratio"]
    tbl.index = range(1, len(tbl) + 1)
    html_table(tbl, fmt={"Median error": "{:,.1f}%", "Bias": "{:,.2f}×", "Over-projected": "{:.0%}",
                         "Within ±25%": "{:.0%}", "p90 ratio": "{:,.2f}×"})

    callout("What this shows", f"""
      <b>Reading.</b> A typical proforma is out by <b>{h['proforma_mdape']:.0f}%</b> and leans
        high ({h['proforma_bias']:.2f}× actual). Model 5 is out by <b>{h['model5_mdape']:.0f}%</b>
        and does not lean either way ({h['model5_bias']:.2f}×). It gets closer on
        <b>{h2h['wins']} of {h2h['n']} sites</b>, and it has never seen the site it is predicting.
      <b>So-what.</b> The location-only model sits in between, which tells us where the value is:
        <b>about half the improvement comes from the location alone</b>, before anything about the
        build is taken into account.
    """, GOOD)

    # =============================================================================================
    st.divider()
    st.header("5 · Which site features actually bring more washes?")
    st.caption("Pick a feature to see what sites at each level really wash. This is raw data — no "
               "model, just the sites grouped by what they were built with.")

    imp = _impact()
    fkeys = [r.key for r in imp.itertuples()]
    flabels = {r.key: r.factor for r in imp.itertuples()}
    chosen = st.selectbox("Feature", fkeys, format_func=lambda k: flabels[k], index=0)

    lv = pf.factor_levels(d, chosen)
    lv = lv[lv.kept]
    sig = pf.factor_significance(d, chosen)
    if len(lv):
        figl = go.Figure(go.Bar(
            x=[str(e).title() for e in lv.example], y=lv.median_washes,
            marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
            text=[f"{v:,.0f}<br><span style='font-size:11px;opacity:.65'>{n:.0f} sites</span>"
                  for v, n in zip(lv.median_washes, lv.sites)],
            textposition="outside", textfont=dict(color=INK, size=13),
            customdata=np.stack([lv.sites, lv.median_pay, lv.median_vac], axis=-1),
            hovertemplate="Built as: <b>%{x}</b><br>Typical site washes <b>%{y:,.0f}</b> a month"
                          "<br><span style='opacity:.7'>%{customdata[0]:.0f} sites — these also "
                          "average %{customdata[1]:.2f} on pay stations</span><extra></extra>"))
        st.plotly_chart(style(figl, height=400,
                              xaxis_title=flabels[chosen],
                              yaxis_title="Washes a month (typical site)",
                              yaxis=dict(range=[0, lv.median_washes.max()*1.32]),
                              # type="category" is load-bearing: labels like "1" and "2" are read
                              # as numbers otherwise, which turns this into a numeric axis and
                              # silently drops the non-numeric levels ("Live Person", "3 or More")
                              xaxis=dict(showgrid=False, type="category")), width="stretch")

        best, worst = lv.iloc[-1], lv.iloc[0]
        mult = best.median_washes / worst.median_washes if worst.median_washes else float("nan")
        if sig.get("real"):
            st.success(
                f"**This one is real.** Sites built with **{str(best.example).title()}** wash a "
                f"typical **{best.median_washes:,.0f}** a month against "
                f"**{worst.median_washes:,.0f}** for **{str(worst.example).title()}** — about "
                f"**{mult:.1f}×**. The pattern holds across all {len(lv)} levels and is unlikely to "
                f"be chance (1 in {max(int(1/max(sig['p_rho'], 1e-6)), 1):,} odds).", icon="✅")
        else:
            small = int(lv.sites.min())
            st.warning(
                f"**Careful — this gap is not solid.** The {mult:.1f}× spread looks big, but the "
                f"smallest group here is only **{small} sites**, and the difference between levels "
                f"is well within what chance produces at this sample size "
                f"(1 in {max(int(1/max(sig['p_levels'], 1e-6)), 1)} odds — we would want at least "
                f"1 in 20).", icon="⚠️")

        if lv.median_pay.nunique() > 1:
            st.markdown("**Why the raw gap can mislead — what else changes across these levels**")
            conf = lv[["example", "sites", "median_washes", "median_pay", "median_vac"]].copy()
            conf.columns = [flabels[chosen], "Sites", "Washes a month",
                            "…and their pay-station score", "…and their vacuum score"]
            conf.index = range(1, len(conf) + 1)
            html_table(conf, fmt={"Sites": "{:.0f}", "Washes a month": "{:,.0f}",
                                  "…and their pay-station score": "{:.2f}",
                                  "…and their vacuum score": "{:.2f}"})
            st.caption(
                "Read the last two columns before believing the first. The groups that wash more "
                "usually also have more pay stations and more vacuum slots — so part of any gap "
                "here belongs to those, not to the feature you picked.")

    st.markdown("#### But these features come as a package")
    st.markdown(
        "Bigger sites tend to have more of everything, so you cannot simply add these up. The chart "
        "below separates the two questions: **how much a feature looks worth on its own**, and "
        "**how much it is still worth once every other feature is accounted for**.")

    figi = go.Figure()
    figi.add_bar(y=imp.factor, x=imp.raw_effect, orientation="h", name="on its own",
                 marker=dict(color=MUTED, line=dict(width=1.5, color=SURFACE)),
                 hovertemplate="<b>%{y}</b><br>One step better = <b>%{x:+.0%}</b> more washes"
                               "<br><span style='opacity:.7'>looking at this feature alone</span>"
                               "<extra></extra>")
    figi.add_bar(y=imp.factor, x=imp.controlled, orientation="h",
                 name="once everything else is accounted for",
                 marker=dict(color=S1, line=dict(width=1.5, color=SURFACE)),
                 hovertemplate="<b>%{y}</b><br>One step better = <b>%{x:+.0%}</b> more washes"
                               "<br><span style='opacity:.7'>with all other features held equal"
                               "</span><extra></extra>")
    figi.add_vline(x=0, line=dict(color=INK2, width=1.5))
    st.plotly_chart(style(figi, height=500, barmode="group",
                          xaxis_title="Extra washes from building one level better",
                          xaxis=dict(tickformat="+.0%"),
                          yaxis=dict(autorange="reversed", showgrid=False),
                          # the legend needs its own band above the plot, otherwise it lands on
                          # top of the first pair of bars
                          margin=dict(l=60, r=25, t=86, b=50),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    top = imp.iloc[0]
    corr = pf.factor_correlations(d)
    pay_vac = corr.iloc[0, 1]
    callout("What this shows", f"""
      <b>Reading.</b> <b>{top.factor}</b> is the strongest single feature: built one level better,
        a site washes about <b>{top.raw_effect:.0%} more</b> on its own, and still
        <b>{top.controlled:.0%} more</b> once every other feature is held equal. Only pay stations
        and vacuum slots survive both tests — a real trend across their levels <i>and</i> a real
        effect after everything else is accounted for.
      <b>So-what.</b> Where the grey bar is long and the blue one is short — area profile, entrance
        stack-up, site accessibility — the feature was never doing the work. Those sites simply also
        had more pay stations and more vacuum slots. Pay stations and vacuums themselves move
        together at {pay_vac:.2f}, so their gains do not stack either.
      <b>So-what.</b> All nine features together explain about <b>{imp.attrs['full_r2']:.0%}</b> of
        why one site washes more than another. The other {1-imp.attrs['full_r2']:.0%} is market and
        operator — nothing on the build sheet reaches it.
    """, S3)

    # =============================================================================================
    st.divider()
    st.header("6 · The tunnel length the formula recommends")
    st.caption("The proforma sizes the tunnel straight off its own year-5 peak projection — one "
               "foot per car per hour. So how close is that to the tunnel that actually got built?")

    tl = pf.tunnel_lengths(d)
    ts = pf.tunnel_length_stats(d)
    lim = [0, float(max(tl.actual_m.max(), tl.formula_m.max())) * 1.05]

    figl = go.Figure()
    figl.add_scatter(x=lim, y=lim, mode="lines", name="formula matches the build",
                     line=dict(color=INK2, width=2, dash="dash"), hoverinfo="skip")
    for mask, name, col in [(tl.gap_m > 0, "Built longer than the formula", BAD),
                            (tl.gap_m <= 0, "Built shorter than the formula", S1)]:
        g = tl[mask]
        figl.add_scatter(x=g.actual_m, y=g.formula_m, mode="markers", name=name,
                         marker=dict(size=11, color=col, line=dict(width=1.6, color=SURFACE)),
                         customdata=np.stack([g.client_name.fillna("—"), g.state.fillna("—"),
                                              g.actual_m, g.formula_m, g.gap_m], axis=-1),
                         hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                                       "Built: <b>%{customdata[2]:.0f} m</b><br>"
                                       "Formula asked for: %{customdata[3]:.0f} m<br>"
                                       "→ built %{customdata[4]:+.0f} m longer<extra></extra>")
    st.plotly_chart(style(figl, height=470, xaxis_title="Tunnel actually built (m)",
                          yaxis_title="Length the formula calls for (m)",
                          xaxis=dict(range=lim, constrain="domain"),
                          yaxis=dict(range=lim, scaleanchor="x", constrain="domain"),
                          margin=dict(l=60, r=25, t=72, b=50),
                          legend=dict(orientation="h", y=1.06, x=0)), width="stretch")

    t1, t2, t3 = st.columns(3)
    t1.metric("Median tunnel built", f"{ts['median_actual']:.0f} m",
              f"formula asked for {ts['median_formula']:.0f} m", delta_color="off")
    t2.metric("Built longer than the formula", f"{ts['built_longer_share']*100:.0f}%",
              f"of {ts['n']} sites", delta_color="off")
    t3.metric("Formula vs built, correlation", f"{ts['r']:.2f}",
              f"typical miss {ts['mae']:.0f} m", delta_color="off")

    sig = pf.length_signal_check(d)
    row_vol = sig[(sig.measure == "Formula length") &
                  (sig.tracks == "Proforma year-5 projection")].iloc[0]
    row_len = sig[(sig.measure == "Formula length") &
                  (sig.tracks == "Actual built length")].iloc[0]
    figsig = go.Figure(go.Bar(
        x=[row_vol.rho, row_len.rho],
        y=["…the proforma's own volume projection", "…the tunnel that actually got built"],
        orientation="h",
        marker=dict(color=[BAD, MUTED], line=dict(width=2, color=SURFACE)),
        # 3 dp on purpose: rendering 0.997 as "1.00" reads as a suspicious perfect correlation
        text=[f"{row_vol.rho:.3f}", f"{row_len.rho:.3f}"], textposition="outside",
        textfont=dict(color=INK, size=14),
        hovertemplate="Formula length tracks %{y}<br>correlation <b>%{x:.2f}</b><extra></extra>"))
    st.plotly_chart(style(figsig, height=280, xaxis_title="How closely the formula length tracks…",
                          xaxis=dict(range=[0, 1.15]), yaxis=dict(showgrid=False)),
                    width="stretch")

    st.markdown("#### The same picture as section ①, but with the *promised* peak")
    st.caption("Each dot is a site: the peak the proforma projected against the tunnel that got "
               "built. The dashed line is the sizing rule — a dot on it means the promised peak "
               "exactly fills the tunnel.")

    tc = pf.tunnel_cohorts(d)
    lim2 = [0, float(max(tc.actual_ft.max(), tc.year5_max_hourly.max())) * 1.06]
    COH = {"Year 3": S3, "Year 4+": S1}
    figc = go.Figure()
    figc.add_scatter(x=lim2, y=lim2, mode="lines", name="promised peak fills the tunnel",
                     line=dict(color=BAD, width=1.8, dash="dash"), hoverinfo="skip")
    for coh, colr in COH.items():
        g = tc[tc.cohort == coh]
        if g.empty:
            continue
        figc.add_scatter(x=g.actual_ft, y=g.year5_max_hourly, mode="markers",
                         name=f"{coh} ({len(g)} sites)",
                         marker=dict(size=10, color=colr, line=dict(width=1.4, color=SURFACE)),
                         customdata=np.stack([g.client_name.fillna("—"), g.share_used], axis=-1),
                         hovertemplate="<b>%{customdata[0]}</b><br>"
                                       "Tunnel built: %{x:.0f} ft<br>"
                                       "Peak the proforma promised: <b>%{y:.0f} cars</b><br>"
                                       "→ the promise fills %{customdata[1]:.0%} of the tunnel"
                                       "<extra></extra>")
    st.plotly_chart(style(figc, height=470, xaxis_title="Tunnel actually built (ft)",
                          yaxis_title="Peak the proforma projected (cars)",
                          xaxis=dict(range=lim2, constrain="domain"),
                          yaxis=dict(range=lim2, scaleanchor="x", constrain="domain"),
                          margin=dict(l=60, r=25, t=86, b=50),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    sh_all = tc.share_used.median()
    callout("What this shows", f"""
      <b>Reading.</b> On paper the tunnels are right-sized — the promised peak fills a median
        <b>{sh_all:.0%}</b> of the tunnel that got built, so the dots sit close to the line.
      <b>So-what.</b> Put this beside section ①, where the <i>measured</i> peak fills far less of
        the same tunnels. The tunnel was sized correctly <b>for a level of demand that never
        arrived</b> — the sizing rule was applied faithfully to a number that was too high.
    """, BAD)

    callout("What this shows", f"""
      <b>Reading.</b> The recommended length is really <b>the wash-volume promise wearing a
        different label</b>. It moves almost perfectly in step with the proforma's own volume number
        (<b>{row_vol.rho:.3f}</b> out of 1.00) but barely relates to the tunnel that actually got
        built (<b>{row_len.rho:.2f}</b>), missing it by <b>{ts['mae']:.0f} m</b> on tunnels of
        20–61 m.
      <b>So-what.</b> Two problems stack up. Because the recommendation is just the volume promise
        rescaled, it carries the same lean — an over-promised site gets an over-long
        recommendation. And the build then ignores it anyway:
        <b>{ts['built_longer_share']*100:.0f}% of sites end up longer</b> than recommended, a
        typical {ts['median_actual']:.0f} m built against {ts['median_formula']:.0f} m advised.
    """, BAD)

    # =============================================================================================
    st.divider()
    st.header("7 · All the data")
    st.caption("Filter, sort and download. Everything the charts above are drawn from.")

    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        sts = sorted(d.state.dropna().unique())
        pick_states = st.multiselect("State", sts, default=[], key="pf_states")
    with f2:
        pick_types = st.multiselect("Proforma type", sorted(d.proforma_type.dropna().unique()),
                                    default=[], key="pf_types")
    with f3:
        sort_by = st.selectbox("Sort by", ["Most over-projected", "Most under-projected",
                                           "Actual washes", "Operator"], key="pf_sort")

    view = d.copy()
    view["ratio"] = view.proforma_y5 / view.actual_mature_wash
    if pick_states:
        view = view[view.state.isin(pick_states)]
    if pick_types:
        view = view[view.proforma_type.isin(pick_types)]
    key, asc = {"Most over-projected": ("ratio", False), "Most under-projected": ("ratio", True),
                "Actual washes": ("actual_mature_wash", False), "Operator": ("client_name", True)}[sort_by]
    view = view.sort_values(key, ascending=asc)
    st.caption(f"Showing **{len(view)}** of {len(d)} sites.")

    cols_ = ["client_name", "state", "open_year", "proforma_type", "actual_mature_wash",
             "proforma_y5", "ratio", "coldstart_v15_y5", "model5_mature"]
    tblv = view[cols_].rename(columns={
        "client_name": "Operator", "state": "State", "open_year": "Opened",
        "proforma_type": "Type", "actual_mature_wash": "Actual/mo", "proforma_y5": "Proforma/mo",
        "ratio": "Proforma ÷ actual", "coldstart_v15_y5": "Cold-start/mo",
        "model5_mature": "Model 5/mo"})
    tblv.index = range(1, len(tblv) + 1)
    html_table(tblv, fmt={"Opened": "{:.0f}", "Actual/mo": "{:,.0f}", "Proforma/mo": "{:,.0f}",
                          "Proforma ÷ actual": "{:,.2f}×", "Cold-start/mo": "{:,.0f}",
                          "Model 5/mo": "{:,.0f}"}, index_label="#")
    st.download_button("Download this table (CSV)", view[cols_].to_csv(index=False),
                       "proforma_backtest.csv", "text/csv")

    with st.expander("Year-by-year detail for every site"):
        rows = []
        for r in view.itertuples():
            tr = pf.site_trajectory(d, r.site_key)
            for t_ in tr.itertuples():
                rows.append(dict(Site=r.client_name, State=r.state, Year=t_.year,
                                 Actual=t_.actual, Proforma=t_.proforma, Model5=t_.model5))
        detail = pd.DataFrame(rows)
        detail.index = range(1, len(detail) + 1)
        html_table(detail.head(50), fmt={"Year": "{:.0f}", "Actual": "{:,.0f}",
                                         "Proforma": "{:,.0f}", "Model5": "{:,.0f}"})
        st.caption(f"First 50 of {len(detail)} site-years.")
        st.download_button("Download the full year-by-year detail (CSV)",
                           detail.to_csv(index=False), "proforma_by_year.csv", "text/csv")
