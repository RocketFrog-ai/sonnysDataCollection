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


@st.cache_data(show_spinner=False)
def _selection() -> pd.DataFrame:
    return pf.selection_path(_load())


@st.cache_data(show_spinner=False)
def _traffic() -> pd.DataFrame:
    return pf.traffic_bands(_load())


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

    # Maturity is the underwriting question ("was the investment case right"); the year views are
    # the trajectory question ("was the shape right"), each year judged against the projection made
    # FOR that year rather than against a five-year-out promise.
    views = ["At maturity"] + [f"Year {y}" for y in range(1, 6)]
    view = st.radio("Compare on", views, index=0, horizontal=True, key="pf_year_view")
    year = None if view == "At maturity" else int(view.split()[-1])
    pp = pf.projection_pairs(d, year)
    st_ = pp.attrs

    lim = [0, float(max(pp.actual.max(), pp.projected.max())) * 1.05]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", name="perfect projection",
                    line=dict(color=INK2, width=2, dash="dash"), hoverinfo="skip")
    for mask, name, col in [(pp.over, "Over-projected", BAD), (~pp.over, "Under-projected", S1)]:
        g = pp[mask]
        fig.add_scatter(x=g.actual, y=g.projected, mode="markers", name=name,
                        marker=dict(size=11, color=col, line=dict(width=1.6, color=SURFACE)),
                        customdata=np.stack([g.client_name.fillna("—"), g.state.fillna("—"),
                                             g.ratio, g.open_year], axis=-1),
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} · opened "
                                      "%{customdata[3]:.0f}<br>"
                                      "Proforma said: <b>%{y:,.0f}</b> washes/mo<br>"
                                      "Actually delivered: <b>%{x:,.0f}</b> washes/mo<br>"
                                      "→ <b>%{customdata[2]:.1f}× the real volume</b><extra></extra>")
    st.plotly_chart(style(fig, height=520,
                          xaxis_title=f"Washes the site actually does — {st_['label'].lower()}"
                                      " (per month)",
                          yaxis_title="Washes the proforma promised (per month)",
                          xaxis=dict(range=lim, constrain="domain"),
                          yaxis=dict(range=lim, scaleanchor="x", constrain="domain"),
                          margin=dict(l=60, r=25, t=72, b=50),
                          legend=dict(orientation="h", y=1.06, x=0)), width="stretch")
    st.caption(f"**{st_['label']}** — {st_['sub']}. **{st_['n']} sites**"
               + (f"; {st_['n_dropped']} left out for not having all 12 months observed in year "
                  f"{year}." if year else "."))

    # Every year on one row, so the reviewer can see whether the miss is a level or a drift without
    # clicking through the five views.
    rows = []
    for y in [None] + list(range(1, 6)):
        s = pf.projection_pairs(d, y).attrs
        rows.append({"Compared on": s["label"], "Sites": s["n"], "Typical miss": s["mdape"],
                     "Runs at": s["bias"], "Over-projected": s["over_share"],
                     "Worst tenth": s["p90"], "Within ±25%": s["within_25"]})
    yt = pd.DataFrame(rows)
    yt.index = range(1, len(yt) + 1)
    html_table(yt, fmt={"Sites": "{:,.0f}", "Typical miss": "{:.0f}%", "Runs at": "{:.2f}×",
                        "Over-projected": "{:.0%}", "Worst tenth": "{:.1f}×",
                        "Within ±25%": "{:.0%}"})

    y1, y5 = pf.projection_pairs(d, 1).attrs, pf.projection_pairs(d, 5).attrs
    callout("What this shows", f"""
      <b>How to read this.</b> Each dot is one car wash. The dashed line is where the promise came
        true. Above the line = the site washed <i>fewer</i> cars than the proforma said it would.
      <b>On the view you have selected — {st_['label'].lower()} — {st_['over_share']*100:.0f}% of
        sites are above the line.</b> The typical proforma promised
        <b>{st_['bias']:.2f}×</b> what the site really does, and one site in ten was promised
        <b>{st_['p90']:.1f}× or more</b>.
      <b>It is not a slow start being misread.</b> Aligning each year against its own projection,
        the proforma is over by <b>{y1['bias']:.2f}×</b> in year 1 and by <b>{y5['bias']:.2f}×</b>
        by year 5 — it <i>widens</i> as the site fills up, on {y5['n']} sites old enough to check.
        The ramp it projected is not the ramp sites actually run: a real wash is at ~98% of its
        eventual volume by year 2 (§⓪), so a projection that keeps climbing to year 5 keeps walking
        away from the site.
      <b>Why that matters.</b> If the misses were random, half would sit above the line and half
        below. They do not — they nearly all lean the same way. That means roughly three sites in
        four were approved on a number that never arrived, and the worst were out by several times
        over, not by a few percent.
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

    ucol, hcol = st.columns([1.4, 1])
    with ucol:
        unit = st.radio("Show washes as", ["Per month", "Per day", "Per open hour"],
                        horizontal=True, index=0)
    default_hours = pf.site_volume_views(d, site_key).attrs.get("proforma_hours", float("nan"))
    default_hours = float(default_hours) if np.isfinite(default_hours) else 12.0
    with hcol:
        # Keyed on the site, so switching site resets the slider to that site's own proforma
        # figure instead of carrying the last site's assumption across.
        open_hours = st.slider("Open hours a day", 6.0, 24.0, value=round(default_hours, 1),
                               step=0.5, key=f"hours_{site_key}",
                               help="The proforma's own assumption for this site, and adjustable. "
                                    "Drives the per-hour view AND the per-hour metric below, which "
                                    "is shown whichever view is selected. "
                                    "The tunnel sizing rule is cars *per hour*, so this number "
                                    "moves the recommended length directly.")
    suffix = {"Per month": "", "Per day": "_daily", "Per open hour": "_hourly"}[unit]
    fmt = ",.0f" if unit == "Per month" else ",.1f"
    t_ = pf.site_volume_views(d, site_key, open_hours)
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
        src = (f"**{hours:.1f} open hours a day** — moved off the proforma's "
               f"{t_.attrs['proforma_hours']:.1f}" if t_.attrs.get("hours_overridden")
               else f"**{hours:.1f} open hours a day** — this site's figure from its proforma")
        st.caption(f"Converted at 365 days a year and {src}. Worth knowing that the proformas "
                   "assume **12 hours for 62 of the 63 sites** that carry the field, so this is a "
                   "house assumption rather than a measurement of the site — which is why the "
                   "slider is here.")

    # Per hour at maturity, on the hours currently set — the figure the tunnel is sized against,
    # and the one the proforma's own capacity rule consumes.
    act_hourly = row.actual_mature_wash * 12 / 365 / hours if hours and np.isfinite(hours) else np.nan
    pro_hourly = row.proforma_y5 * 12 / 365 / hours if hours and np.isfinite(hours) else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Actually washes", f"{row.actual_mature_wash:,.0f}", "per month at maturity",
              delta_color="off")
    k2.metric("Actually washes / hour", f"{act_hourly:,.1f}" if np.isfinite(act_hourly) else "—",
              f"average open hour · {hours:.1f} h/day" if np.isfinite(hours) else "no hours on file",
              delta_color="off")
    k3.metric("Proforma promised", f"{row.proforma_y5:,.0f}", f"{row.ratio:.1f}× what it does",
              delta_color="off")
    k4.metric("Model 5", f"{row.model5_mature:,.0f}",
              f"{row.model5_mature/row.actual_mature_wash:.1f}× what it does", delta_color="off")
    k5.metric("Opened", f"{row.open_year:.0f}", row.proforma_type, delta_color="off")
    if np.isfinite(act_hourly):
        st.caption(f"**{act_hourly:,.1f} cars an average open hour** against the proforma's "
                   f"**{pro_hourly:,.1f}**. That is the *average* hour — the peak hour is what "
                   "sizes a tunnel, and §① shows the busiest hour ever recorded at a site runs "
                   "well above its average one while still never filling the tunnel.")

    tun = pf.site_tunnel(d, site_key)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**The tunnel: recommended vs built**")
        if np.isfinite(tun.get("actual_ft", float("nan"))):
            g1, g2, g3 = st.columns(3)
            g1.metric("Proforma year-5 peak", f"{tun['recommended_ft']:.0f}", "cars per hour",
                      delta_color="off")
            g2.metric("→ Tunnel recommended", f"{tun['recommended_ft']:.0f} ft",
                      f"{tun['recommended_m']:.0f} m", delta_color="off")
            gap_ft = tun["actual_ft"] - tun["recommended_ft"]
            word = "longer" if gap_ft > 0 else "shorter"
            g3.metric("Actually built", f"{tun['actual_ft']:.0f} ft",
                      f"{abs(gap_ft):.0f} ft {word}", delta_color="off")
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
      <b>Does the projection get more accurate as the site settles down?</b> No. The proforma is
        out by <b>{p_y1:.0f}%</b> in year 1 and <b>{p_y5:.0f}% by year 5</b> — it gets slightly
        worse, not better.
      <b>Model 5 goes the other way</b>, tightening to <b>{m_y5:.0f}%</b> by year 5 as the site's
        own trading history builds up.
      <b>Why that matters.</b> Years 4 and 5 are the years the loan is being repaid on — the whole
        reason the projection is made. That is exactly where the proforma is least reliable.
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

    ex = d.assign(r=d.proforma_y5 / d.actual_mature_wash)
    ex = ex[ex.client_name.str.contains("Glacier Express", na=False)]
    over_pf = float((d.proforma_y5 / d.actual_mature_wash > 1).mean())
    over_m5 = float((d.model5_mature / d.actual_mature_wash > 1).mean())
    sc_i = sc.set_index("key")
    closed = ((sc_i.loc["proforma", "mdape"] - sc_i.loc["coldstart", "mdape"]) /
              (sc_i.loc["proforma", "mdape"] - sc_i.loc["model5", "mdape"]))
    callout("What this shows", f"""
      <b>How far off is a typical projection?</b> The proforma misses by about
        <b>{h['proforma_mdape']:.0f}%</b>. Model 5 misses by about <b>{h['model5_mdape']:.0f}%</b> —
        roughly half the error, on the same sites.
      <b>And the misses go a different way.</b> The proforma is too high on
        <b>{over_pf:.0%}</b> of sites, so its errors pile up on one side and the whole plan reads
        rich. Model 5 is too high on <b>{over_m5:.0%}</b> and too low on the rest, so its misses
        cancel out across a portfolio instead of adding up.
      <b>An example.</b> Glacier Express King Ave (MT) washes
        <b>{ex.actual_mature_wash.iloc[0]:,.0f}</b> a month. The proforma promised
        <b>{ex.proforma_y5.iloc[0]:,.0f}</b> — 70% too many. Model 5 said
        <b>{ex.model5_mature.iloc[0]:,.0f}</b>, within 13%. It had never seen this site: it was
        predicted by a model built without it, so this is a fair test rather than a fitted one.
      <b>Where does the improvement come from?</b> The middle bar knows only <i>where</i> the site
        is — nothing about pay stations, vacuums or the building. It already recovers
        <b>{closed:.0%}</b> of the gap. Location is most of the answer; the build sheet adds the
        rest.
    """, GOOD)

    # =============================================================================================
    st.divider()
    st.header("5 · Which site features actually bring more washes?")
    st.caption("Pick a feature to see what sites at each level really wash. This is raw data — no "
               "model, just the sites grouped by what they were built with.")

    imp = _impact()
    # traffic count is in `imp` but not in this picker: it is a measured count, not a scored level,
    # so it has no "levels" to group by. It gets its own chart below.
    fkeys = [r.key for r in imp.itertuples() if r.key != pf.TRAFFIC_KEY]
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

    # ---- traffic count ---------------------------------------------------------------------------
    st.markdown("#### Does a busier road bring more washes?")
    st.markdown("Traffic count is the one proforma input that is a **measured number** rather than a "
                "scored level — vehicles a day past the door, on every one of these proformas. It is "
                "also the input people most expect to drive volume, so it is worth its own chart. "
                "The 68 sites are split into four equal groups, quietest road to busiest.")

    tb = _traffic()
    figt = go.Figure()
    figt.add_bar(x=tb.label, y=tb.median_washes, name="washes a month",
                 marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
                 text=[f"{v:,.0f}" for v in tb.median_washes], textposition="outside",
                 textfont=dict(color=INK, size=13),
                 customdata=np.stack([tb.sites, tb.median_traffic, tb.capture], axis=-1),
                 hovertemplate="Road carries <b>%{x}</b> vehicles a day<br>"
                               "Typical site washes <b>%{y:,.0f}</b> a month<br>"
                               "Catches <b>%{customdata[2]:.2%}</b> of the cars that pass<br>"
                               "<span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                               "<extra></extra>")
    figt.add_scatter(x=tb.label, y=tb.capture, name="share of passing cars caught", yaxis="y2",
                     mode="lines+markers", line=dict(color=S2, width=3),
                     marker=dict(size=11, line=dict(width=2, color=SURFACE)),
                     hovertemplate="<b>%{y:.2%}</b> of passing cars caught<extra></extra>")
    st.plotly_chart(style(
        figt, height=430, xaxis_title="Vehicles a day past the site",
        yaxis_title="Washes a month (typical site)",
        yaxis=dict(range=[0, tb.median_washes.max() * 1.28]),
        yaxis2=dict(overlaying="y", side="right", tickformat=".1%", showgrid=False,
                    title=dict(text="Share of passing cars caught"),
                    tickfont=dict(color=MUTED), range=[0, tb.capture.max() * 1.35]),
        xaxis=dict(showgrid=False, type="category"),
        margin=dict(l=60, r=70, t=86, b=50),
        legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    callout("What this shows", f"""
      <b>Yes, but far less than you would think.</b> Going from the quietest quarter of roads to the
        busiest, traffic rises <b>{tb.attrs['traffic_ratio']:.1f}×</b>
        ({tb.median_traffic.iloc[0]:,.0f} to {tb.median_traffic.iloc[-1]:,.0f} cars a day) — but the
        typical site only washes <b>{tb.attrs['wash_ratio']:.1f}×</b> more
        ({tb.median_washes.iloc[0]:,.0f} to {tb.median_washes.iloc[-1]:,.0f} a month).
      <b>The catch rate falls as the road gets busier.</b> A site on a quiet road washes
        <b>{tb.capture.iloc[0]:.2%}</b> of the cars going past. On the busiest roads that drops to
        <b>{tb.capture.iloc[-1]:.2%}</b> — <b>{tb.attrs['capture_ratio']:.1f}× worse</b>. Roughly:
        triple the traffic and you get about half again as many washes, not triple.
      <b>What it means in money.</b> A site is not a net thrown across a road. Paying a premium for
        a 45,000-car road over a 25,000-car one buys about <b>5%</b> more washes on this evidence —
        so it is worth it only if the premium is small.
      <b>Worth knowing.</b> The direction is right and consistent across all four groups, but the
        link is loose: on its own, traffic count would put a site's forecast out by more than simply
        assuming it does what the average site does.
    """, S3)

    # ---- which features carry real information ---------------------------------------------------
    st.markdown("#### Which of these ten actually predict anything?")
    st.markdown(
        "Every feature above looks like it does *something* when you draw a line through the sites "
        "you already have. The real test is different: **hide one site, fit the line on the other "
        f"{imp.attrs['n'] - 1}, then try to predict the site you hid** — and repeat for all "
        f"{imp.attrs['n']}. Grey is how good a feature looks; blue is how good it actually is.")

    figi = go.Figure()
    figi.add_bar(y=imp.factor, x=imp.r2_fitted, orientation="h",
                 name="how good it looks on the sites we already have",
                 marker=dict(color=MUTED, line=dict(width=1.5, color=SURFACE)),
                 hovertemplate="<b>%{y}</b><br>Appears to explain <b>%{x:.0%}</b> of the gap "
                               "between sites<br><span style='opacity:.7'>measured on the same "
                               "sites the line was drawn through</span><extra></extra>")
    figi.add_bar(y=imp.factor, x=imp.loo_real, orientation="h",
                 name="how good it is on a site it has never seen",
                 marker=dict(color=[S1 if v > 0 else BAD for v in imp.loo_real],
                             line=dict(width=1.5, color=SURFACE)),
                 customdata=np.stack([imp.raw_effect, imp.step_label], axis=-1),
                 hovertemplate="<b>%{y}</b><br>Really explains <b>%{x:.0%}</b> of the gap between "
                               "sites<br><span style='opacity:.7'>%{customdata[1]} = "
                               "%{customdata[0]:+.0%} washes</span><extra></extra>")
    figi.add_vline(x=0, line=dict(color=INK2, width=1.5))
    st.plotly_chart(style(figi, height=520, barmode="group",
                          xaxis_title="Share of the gap between sites the feature explains",
                          xaxis=dict(tickformat=".0%"),
                          yaxis=dict(autorange="reversed", showgrid=False),
                          # the legend needs its own band above the plot, otherwise it lands on
                          # top of the first pair of bars
                          margin=dict(l=60, r=25, t=86, b=50),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    win = imp[imp.loo_real > 0]
    lose = imp[imp.loo_real <= 0]
    # the failing list is generated, never typed: it moves if the data does
    lose_names = ", ".join(lose.factor.str.lower().iloc[:-1]) + " and " + lose.factor.str.lower(
    ).iloc[-1]
    lose_scored = int((~lose.key.eq(pf.TRAFFIC_KEY)).sum())
    callout("What this shows", f"""
      <b>Read the blue bar, not the grey one.</b> Grey is the number a spreadsheet gives you: draw a
        line through the sites you already have and see how well it fits. It always looks like
        something. Blue is the same feature asked to call a site it has never seen.
      <b>Only {imp.attrs['n_survive']} of the {len(imp)} survive:</b>
        <b>{', '.join(win.factor.str.lower())}</b>. {win.iloc[0].factor} is the best of them — a
        site one level better washes about <b>{win.iloc[0].raw_effect:.0%} more</b>, and that holds
        up on sites the line never saw.
      <b>The other {len(lose)} score below zero</b>, which is worse than it sounds. Below zero means
        that if you had used that feature to forecast a new site, you would have been further out
        than if you had simply said "it will do what the average site does". {lose_names.capitalize()}
        all land there — they look like they matter only because busier sites happen to score well
        on everything at once.
      <b>Why that matters for a build decision.</b> Of the nine boxes on the scoring sheet,
        {lose_scored} are carrying no forecasting weight — and neither is the traffic count. Arguing
        over them
        in an investment committee is arguing over noise. The three that move the number are all
        about the site's own capacity — how many cars it can take at once, and what kind of site it
        is.
    """, S3)

    # ---- how many features are worth keeping -----------------------------------------------------
    sp = _selection()
    st.markdown("#### So how many features should the model use?")
    st.markdown(
        "Start with nothing, then add features one at a time — always whichever one helps most next. "
        "Grey is how good the set looks on the sites it was built from; blue is how well it calls a "
        "site it has never seen.")

    figs = go.Figure()
    figs.add_scatter(x=sp.step, y=sp.r2_fitted, mode="lines+markers",
                     name="how good it looks", line=dict(color=MUTED, width=2.5, dash="dot"),
                     marker=dict(size=8, line=dict(width=1.5, color=SURFACE)),
                     customdata=sp.added,
                     hovertemplate="%{x} features<br>Looks like <b>%{y:.0%}</b> explained"
                                   "<extra></extra>")
    figs.add_scatter(x=sp.step, y=sp.loo_real, mode="lines+markers",
                     name="how good it really is", line=dict(color=S1, width=3),
                     marker=dict(size=11, line=dict(width=2, color=SURFACE)),
                     customdata=sp.added,
                     hovertemplate="%{x} features — just added <b>%{customdata}</b><br>"
                                   "Really explains <b>%{y:.0%}</b><extra></extra>")
    figs.add_hline(y=0, line=dict(color=INK2, width=1.5))
    figs.add_vline(x=sp.attrs["best_step"], line=dict(color=S2, width=1.5, dash="dash"))
    figs.add_annotation(x=sp.attrs["best_step"], y=sp.attrs["best_loo"], yshift=26,
                        text=f"<b>best at {sp.attrs['best_step']} features</b>", showarrow=False,
                        font=dict(color=S2, size=12))
    st.plotly_chart(style(
        figs, height=430, xaxis_title="Number of features in the model",
        yaxis_title="Share of the gap between sites explained",
        xaxis=dict(dtick=1, tickmode="array", tickvals=list(sp.step),
                   ticktext=[f"{i}<br><span style='font-size:10px'>+{a}</span>"
                             for i, a in zip(sp.step, sp.added)]),
        yaxis=dict(tickformat=".0%"),
        margin=dict(l=60, r=25, t=86, b=90),
        legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    callout("What this shows", f"""
      <b>More features do not mean a better forecast.</b> The grey line only ever climbs — by the
        time all ten are in, it claims to explain <b>{sp.attrs['end_fitted']:.0%}</b> of the gap
        between sites. The blue line — the honest one — peaks at
        <b>{sp.attrs['best_step']} features</b> and then falls all the way to
        <b>{sp.attrs['worst_loo']:.0%}</b>. Using all ten is worse than using none.
      <b>What is going on.</b> With {sp.attrs['n']} sites and ten scores, the extra features are not
        learning anything about car washes; they are memorising these particular {sp.attrs['n']}
        sites. It fits beautifully and predicts nothing.
      <b>This is why the model carries only a handful.</b> Model 5 uses three of these —
        <b>{', '.join(sp.attrs['model5_labels']).lower()}</b> — alongside the location. Those three
        alone reach <b>{sp.attrs['model5_loo']:.0%}</b>, which is
        <b>{sp.attrs['model5_share']:.0%}</b> of the best any combination of all ten can manage —
        so the other six scored boxes, plus the traffic count, buy essentially nothing.
      <b>The size of the prize.</b> Even at its best, the whole build sheet explains about
        <b>{sp.attrs['best_loo']:.0%}</b> of why one site is busier than another. The other
        <b>{1 - sp.attrs['best_loo']:.0%}</b> is the market and the operator — which is exactly where
        the location model earns its keep.
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

    callout("What this shows", f"""
      <b>Where the recommended length comes from.</b> It is the proforma's own wash promise,
        converted into feet. Nothing else goes into it.
      <b>You can see that in the two bars.</b> The recommendation moves almost perfectly in step
        with the wash promise (<b>{row_vol.rho:.3f}</b> out of a possible 1.00) but hardly relates
        at all to the tunnel that actually got built (<b>{row_len.rho:.2f}</b>) — it is typically
        <b>{ts['mae']:.0f} m</b> away from it, on tunnels that are only 20–61 m long.
      <b>So two things go wrong at once.</b> Because the recommendation is just the wash promise in
        different units, an over-promised site automatically gets an over-long recommendation. And
        then the build ignores it anyway — <b>{ts['built_longer_share']*100:.0f}% of sites end up
        longer</b> than recommended, typically {ts['median_actual']:.0f} m built where
        {ts['median_formula']:.0f} m was advised.
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
