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
from ui import (CRITICAL, DARK, INK, INK2, MUTED, S1, S2, S3, SURFACE, WARNING, callout,
                html_table, style)

# One fixed hue per forecaster, assigned in order and never cycled.
FCOLOR = {"proforma": S2, "coldstart": S3, "model5": S1}
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"

# Traffic speed bands, one fixed hue + marker shape per band, assigned in order and never cycled.
# Deliberately NOT S1/S2/S3 -- those already carry a different meaning on the traffic chart itself
# (S1 = "Each site" baseline, S2/S3 = the two reference lines) and as the forecaster identity
# (FCOLOR) everywhere else in this file; reusing them here would make the same hue mean two
# different things on one screen. Color and shape are both varied so identity never rests on hue
# alone.
SPEED_COLOR = {
    "Under 30 mph": "#c98500" if DARK else "#eda100",   # yellow
    "30–40 mph":    "#d55181" if DARK else "#e87ba4",   # magenta
    "40–50 mph":    "#9085e9" if DARK else "#4a3aa7",   # violet
    "Over 50 mph":  "#008300",                          # green (mode-invariant)
}
SPEED_SYMBOL = {
    "Under 30 mph": "circle", "30–40 mph": "square",
    "40–50 mph": "diamond", "Over 50 mph": "triangle-up",
}


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
def _traffic_frame() -> pd.DataFrame:
    return pf.traffic_frame(_load())


@st.cache_data(show_spinner=False)
def _traffic_elast() -> pd.DataFrame:
    return pf.traffic_elasticity(_load())


@st.cache_data(show_spinner=False)
def _traffic_var() -> pd.DataFrame:
    return pf.traffic_variance(_load())


@st.cache_data(show_spinner=False)
def _traffic_robust() -> pd.DataFrame:
    return pf.traffic_elasticity_robustness(_load())


@st.cache_data(show_spinner=False)
def _traffic_capture() -> pd.DataFrame:
    return pf.traffic_capture(_load())


@st.cache_data(show_spinner=False)
def _traffic_over() -> pd.DataFrame:
    return pf.traffic_overshoot(_load())


@st.cache_data(show_spinner=False)
def _traffic_speed() -> pd.DataFrame:
    return pf.traffic_speed(_load())


@st.cache_data(show_spinner=False)
def _traffic_speed_test() -> dict:
    return pf.traffic_speed_test(_load())


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
    st.header("1 · Are we underestimating the business?")
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
      <b>On the view you have selected, {st_['label'].lower()}: {st_['over_share']*100:.0f}% of
        sites are above the line.</b> The typical proforma promised
        <b>{st_['bias']:.2f}×</b> what the site really does, and one site in ten was promised
        <b>{st_['p90']:.1f}× or more</b>.
      <b>It is not a slow start being misread.</b> Aligning each year against its own projection,
        the proforma is over by <b>{y1['bias']:.2f}×</b> in year 1 and by <b>{y5['bias']:.2f}×</b>
        by year 5. It <i>widens</i> as the site fills up, on {y5['n']} sites old enough to check.
        The ramp it projected is not the ramp sites actually run: a real wash is at ~98% of its
        eventual volume by year 2 (§⓪), so a projection that keeps climbing to year 5 keeps walking
        away from the site.
      <b>Why that matters.</b> If the misses were random, half would sit above the line and half
        below. They do not. They nearly all lean the same way. That means roughly three sites in
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
    # A line, like the two projections it is being compared against: bars and
    # lines on one axis read as two different kinds of quantity.
    #
    # Ink, not S1: S1 is Model 5's fixed identity in FCOLOR above, and as a bar
    # the actual series could share that hue without confusion. As a third line
    # it cannot — so what actually happened is drawn in the foreground ink, and
    # the two forecasts keep their own hues.
    figt.add_scatter(x=t_.year, y=t_["actual" + suffix], mode="lines+markers",
                     name="Actually washed",
                     line=dict(color=INK, width=3.5),
                     marker=dict(size=10, line=dict(width=2, color=SURFACE)),
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

    k1, k3, k4, k5 = st.columns(4)
    k1.metric("Actually washes", f"{row.actual_mature_wash:,.0f}", "per month at maturity",
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
        out by <b>{p_y1:.0f}%</b> in year 1 and <b>{p_y5:.0f}% by year 5</b>. It gets slightly
        worse, not better.
      <b>Model 5 goes the other way</b>, tightening to <b>{m_y5:.0f}%</b> by year 5 as the site's
        own trading history builds up.
      <b>Why that matters.</b> Years 4 and 5 are the years the loan is being repaid on, the whole
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
        <b>{h['proforma_mdape']:.0f}%</b>. Model 5 misses by about <b>{h['model5_mdape']:.0f}%</b>:
        roughly half the error, on the same sites.
      <b>And the misses go a different way.</b> The proforma is too high on
        <b>{over_pf:.0%}</b> of sites, so its errors pile up on one side and the whole plan reads
        rich. Model 5 is too high on <b>{over_m5:.0%}</b> and too low on the rest, so its misses
        cancel out across a portfolio instead of adding up.
      <b>An example.</b> Glacier Express King Ave (MT) washes
        <b>{ex.actual_mature_wash.iloc[0]:,.0f}</b> a month. The proforma promised
        <b>{ex.proforma_y5.iloc[0]:,.0f}</b>, 70% too many. Model 5 said
        <b>{ex.model5_mature.iloc[0]:,.0f}</b>, within 13%. It had never seen this site: it was
        predicted by a model built without it, so this is a fair test rather than a fitted one.
      <b>Where does the improvement come from?</b> The middle bar knows only <i>where</i> the site
        is: nothing about pay stations, vacuums or the building. It already recovers
        <b>{closed:.0%}</b> of the gap. Location is most of the answer; the build sheet adds the
        rest.
    """, GOOD)

    # =============================================================================================
    st.divider()
    st.header("5 · Which parts of the scoring sheet actually matter?")
    st.markdown("Before a site is built, the proforma scores it on **ten boxes**: the traffic "
                "outside, how visible it is, how many pay stations, what the competition looks "
                "like, and so on. Those scores are what the projection is built from. So the fair "
                "question is: **when a site scored well on a box, did it actually wash more cars?**")

    imp = _impact()
    sp = _selection()
    win = imp[imp.loo_real > 0]
    lose = imp[imp.loo_real <= 0]

    # ---- the explorer: raw evidence, one box at a time -------------------------------------------
    st.markdown("#### Look at any box yourself")
    st.caption("Pick a box and see what the sites at each level really washed. No model, no maths: "
               "just the real sites, sorted into groups by what they were built with.")

    # traffic count is in `imp` but not in this picker: it is a measured count, not a scored level,
    # so it has no "levels" to group by. It gets its own chart below.
    fkeys = [r.key for r in imp.itertuples() if r.key != pf.TRAFFIC_KEY]
    flabels = {r.key: r.factor for r in imp.itertuples()}
    chosen = st.selectbox("Box", fkeys, format_func=lambda k: flabels[k], index=0)

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
                          "<br><span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                          "<extra></extra>"))
        st.plotly_chart(style(figl, height=400,
                              xaxis_title=flabels[chosen],
                              yaxis_title="Washes a month, typical site",
                              yaxis=dict(range=[0, lv.median_washes.max()*1.32]),
                              # type="category" is load-bearing: labels like "1" and "2" are read
                              # as numbers otherwise, which turns this into a numeric axis and
                              # silently drops the non-numeric levels ("Live Person", "3 or More")
                              xaxis=dict(showgrid=False, type="category")), width="stretch")

        best, worst = lv.iloc[-1], lv.iloc[0]
        mult = best.median_washes / worst.median_washes if worst.median_washes else float("nan")
        # The odds are the p-value turned upside down: 1/p. Stated as "if this box made no
        # difference, luck would fake a gap this consistent about 1 sample in N", which is what a
        # p-value actually says. It is NOT the probability the finding is true.
        if sig.get("real"):
            odds = max(int(1 / max(sig["p_rho"], 1e-6)), 1)
            st.success(
                f"**This gap is a real one.** Sites built with **{str(best.example).title()}** "
                f"wash **{best.median_washes:,.0f}** a month; sites with "
                f"**{str(worst.example).title()}** wash **{worst.median_washes:,.0f}**. That is "
                f"**{mult:.1f}× as many**, and the groups line up in the right order all the way "
                f"up. A gap this tidy would happen by luck alone in about 1 case in {odds:,}.",
                icon=None)
        else:
            small = int(lv.sites.min())
            odds = max(int(1 / max(sig["p_levels"], 1e-6)), 1)
            # Several failing boxes run BACKWARDS (the best-scoring group washes fewer cars).
            # "It looks like 0.5x, but..." would be nonsense, so those get their own sentence.
            if mult < 1.05:
                lead = (f"**This box does not work at all.** The sites scoring *best* on it wash "
                        f"**{best.median_washes:,.0f}** a month; the ones scoring *worst* wash "
                        f"**{worst.median_washes:,.0f}**. Scoring better here buys nothing, and on "
                        f"these sites it runs slightly the wrong way.")
            else:
                lead = (f"**Do not trust this gap.** It looks like {mult:.1f}×, but the smallest "
                        f"group here is just **{small} sites**.")
            tail = ("A pattern like this comes up by luck most of the time, so there is nothing "
                    "here to explain." if odds <= 3 else
                    f"With groups this small, luck on its own throws up a gap this big about 1 "
                    f"time in {odds}, far too often to believe it. We would want 1 in 20 or rarer "
                    f"before calling it real.")
            st.warning(f"{lead} {tail}", icon=None)

        if lv.median_pay.nunique() > 1:
            st.markdown("**One thing to check before you believe it**")
            conf = lv[["example", "sites", "median_washes", "median_pay", "median_vac"]].copy()
            conf.columns = [flabels[chosen], "Sites", "Washes a month",
                            "Their pay-station score", "Their vacuum score"]
            conf.index = range(1, len(conf) + 1)
            html_table(conf, fmt={"Sites": "{:.0f}", "Washes a month": "{:,.0f}",
                                  "Their pay-station score": "{:.2f}",
                                  "Their vacuum score": "{:.2f}"})
            st.caption("The sites that wash more usually score higher on **everything** at once, "
                       "so they tend to have more pay stations and more vacuums too. Some of the "
                       "gap above belongs to those, not to the box you picked. Sorting that out is "
                       "what the next chart does.")

    # ---- the scoreboard --------------------------------------------------------------------------
    st.markdown("#### The test each box has to pass")
    st.markdown(
        f"Every box looks useful if you go looking for a pattern in sites you already know about. "
        f"So each one gets a harder test: **cover up one site, work out what the box predicts for "
        f"it from the other {imp.attrs['n'] - 1}, then uncover it and see how close you got.** "
        f"Repeat for all {imp.attrs['n']} sites. A box passes only if it beats the laziest possible "
        f"forecast, which is just saying *\"it will wash what an average site washes\"*. "
        f"**{imp.attrs['n_survive']} of the {len(imp)} boxes pass.**")

    w = win.sort_values("raw_effect", ascending=True)
    figw = go.Figure(go.Bar(
        y=w.factor, x=w.raw_effect, orientation="h",
        marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
        text=[f"+{v:.0%} washes" for v in w.raw_effect], textposition="outside",
        textfont=dict(color=INK, size=13),
        hovertemplate="<b>%{y}</b><br>One level better is worth <b>+%{x:.0%}</b> washes"
                      "<extra></extra>"))
    st.plotly_chart(style(figw, height=250,
                          title=dict(text="The boxes that passed, and what one level is worth"),
                          xaxis_title="Extra washes when the site scores one level better",
                          xaxis=dict(tickformat=".0%", range=[0, float(w.raw_effect.max()) * 1.35]),
                          yaxis=dict(showgrid=False),
                          margin=dict(l=140, r=30, t=52, b=48)), width="stretch")

    board = imp[["factor", "loo_real"]].copy()
    board["Verdict"] = np.where(board.loo_real > 0, "Passed", "No better than guessing")
    board["What it is worth"] = [
        f"+{r.raw_effect:.0%} washes per level" if r.loo_real > 0 else "nothing you can rely on"
        for r in imp.itertuples()]
    board = board[["factor", "Verdict", "What it is worth"]]
    board.columns = ["Box on the scoring sheet", "Verdict", "What it is worth"]
    board.index = range(1, len(board) + 1)
    html_table(board)

    callout("What this shows", f"""
      <b>Three boxes out of ten do the work.</b> <b>{', '.join(win.factor.str.lower())}</b>. A site
        that scores one level better on {win.iloc[0].factor.lower()} washes about
        <b>{win.iloc[0].raw_effect:.0%} more cars</b>, and that still holds on sites the test had
        never seen.
      <b>The other {len(lose)} are not just weak, they are misleading.</b> Used on their own to
        forecast a new site, each of them would leave you <b>further out</b> than if you had not
        looked at the sheet at all and simply quoted the average site.
      <b>Why they looked useful.</b> Good sites tend to score well on everything at once. So when a
        box appears to matter, it is usually borrowing the credit from the pay stations and vacuums
        sitting next to it.
      <b>The three survivors have one thing in common.</b> They are all about <b>how many cars the
        site can physically handle</b>. Not the neighbourhood, not the visibility, not the
        competition: the capacity you build.
      <b>What that means in a meeting.</b> Six of the nine scored boxes, plus the traffic count, are
        carrying no weight. Arguing over them is arguing over nothing.
    """, S3)

    # ---- traffic count ---------------------------------------------------------------------------
    # Four charts, in the order the argument runs: what the sheet IS → what the road actually pays →
    # what that does to the forecast → and the speed box, which is the other half of the question.
    tf, tel, tvar, tcap = _traffic_frame(), _traffic_elast(), _traffic_var(), _traffic_capture()
    tov, tsp, tspt = _traffic_over(), _traffic_speed(), _traffic_speed_test()
    tres = _traffic_robust()
    act, pfe = tel.attrs["actual"], tel.attrs["proforma"]

    st.markdown("#### Traffic: the number the sheet leans on hardest")
    st.markdown(
        "Traffic is the only box that is a **real measured number** rather than a judgement: "
        "vehicles a day past the door. It is also the number the old proforma is effectively "
        "built out of. The chart below plots every site's road against the wash volume — what "
        "the sheet promised, and what the site delivered.")

    pf_line = tcap[tcap.which == "What the proforma assumed"].iloc[0]
    xs = np.linspace(tf[pf.TRAFFIC_KEY].min() * .92, tf[pf.TRAFFIC_KEY].max() * 1.05, 60)

    figt = go.Figure()
    figt.add_scatter(x=xs, y=xs * pf_line["median"], mode="lines",
                     name=f"The sheet's rule: {pf_line['median']:.2%} of passing cars",
                     line=dict(color=S2, width=2.5),
                     hovertemplate="%{x:,.0f} cars a day → the sheet expects "
                                   "<b>%{y:,.0f} washes a day</b><extra></extra>")
    figt.add_scatter(x=xs, y=np.exp(act["a"]) * xs ** act["b"], mode="lines",
                     name=f"What the road really pays (slope {act['b']:.2f})",
                     line=dict(color=S3, width=2.5, dash="dash"),
                     hovertemplate="%{x:,.0f} cars a day → really "
                                   "<b>%{y:,.0f} washes a day</b><extra></extra>")
    # One trace per speed band -- Plotly only builds a categorical legend from separate traces,
    # and the fixed SPEED_COLOR/SPEED_SYMBOL order keeps a given band the same hue+shape wherever
    # it's plotted, rather than depending on which bands happen to be present.
    for band in pf.SPEED_BANDS:
        g = tf[tf.speed_band == band]
        if g.empty:
            continue
        figt.add_scatter(x=g[pf.TRAFFIC_KEY], y=g.daily_washes, mode="markers",
                         name=f"{band} ({len(g)} sites)",
                         marker=dict(size=9, color=SPEED_COLOR[band], symbol=SPEED_SYMBOL[band],
                                     line=dict(width=1.5, color=SURFACE)),
                         customdata=np.stack([g.site, g.capture, g.pf_capture,
                                              g.actual_mature_wash], axis=-1),
                         hovertemplate="<b>%{customdata[0]}</b><br>%{x:,.0f} cars a day<br>"
                                       "<b>%{y:,.0f} washes a day</b> "
                                       "(%{customdata[3]:,.0f} a month)<br>"
                                       "caught %{customdata[1]:.2%} of the traffic — "
                                       "the sheet assumed %{customdata[2]:.2%}<br>"
                                       f"<span style='opacity:.7'>{band}</span><extra></extra>")
    st.plotly_chart(style(figt, height=490,
                          xaxis_title="Vehicles a day past the site",
                          yaxis_title="Washes a day, at maturity",
                          xaxis=dict(tickformat=","), margin=dict(t=130),
                          # 6 legend entries (2 reference lines + 4 speed bands) wrap onto two
                          # rows at typical chart widths -- the taller top margin above makes room.
                          legend=dict(orientation="h", y=1.1, x=0)), width="stretch")

    callout("What the orange line is, and why the dots are not on it", f"""
      <b>The old proforma is, in effect, one multiplication.</b> Across all
        {int(pfe['n'])} sites its own year-5 projection is <b>{pf_line['median']:.2%} of the traffic
        count</b>, give or take — the road alone explains
        <b>{tvar.iloc[0].traffic_only:.0%}</b> of the sheet's numbers, and everything else on it
        (the nine scored boxes and the whole demographic block) explains
        <b>{tvar.iloc[0].scores_only:.1%}</b>. The orange line <i>is</i> the sheet.
      <b>Reality is a cloud, not a line.</b> The same traffic count explains only
        <b>{tvar.iloc[1].traffic_only:.1%}</b> of what the sites actually did. The rank correlation
        between traffic and mature volume is <b>{act['rho']:+.2f}</b> and does not reach
        significance (p&nbsp;=&nbsp;{act['p']:.2f}) on {int(act['n'])} sites.
      <b>The slope is the whole argument.</b> The sheet's line rises at
        <b>{pfe['b']:.2f}</b> — near-perfect proportionality, which is what assuming a fixed capture
        rate means. The real one rises at <b>{act['b']:.2f}</b>
        [95% CI {act['lo']:.2f} to {act['hi']:.2f}]. A slope of 1 is rejected
        (p&nbsp;=&nbsp;{act['p_vs_one']:.4f}); a slope of <i>zero</i> is not
        (p&nbsp;=&nbsp;{act['p_vs_zero']:.2f}).
      <b>In money terms.</b> Doubling the traffic buys about
        <b>{act['double']:+.0%}</b> more washes — and the honest interval runs from
        {act['double_lo']:+.0%} to {act['double_hi']:+.0%}. The sheet pays for
        <b>{pfe['double']:+.0%}</b>. Pay a premium for a busier road accordingly.
      <b>0.25 is the cautious end of the range, not a cherry-pick.</b> Five other ways of measuring
        the same slope land between <b>{tres.attrs['lo']:.2f}</b> and
        <b>{tres.attrs['hi']:.2f}</b> — doubling traffic buys {tres.attrs['double_lo']:+.0%} to
        {tres.attrs['double_hi']:+.0%} — with the most outlier-sensitive of them reaching
        {tres.attrs['widest']:.2f}. Every one is far below the sheet's {pfe['b']:.2f}, which is all
        the conclusion rests on. The workings are in the expander below.
    """, S3)

    with st.expander("Every way we measured that slope"):
        rt = tres[["method", "b", "double", "note"]].copy()
        rt.columns = ["How it was measured", "Slope", "Doubling traffic buys", "What it is"]
        html_table(rt.set_index("How it was measured"), index_label="How it was measured",
                   fmt={"Slope": "{:.3f}", "Doubling traffic buys": "{:+.0%}"})
        st.caption("The drawn line uses least squares on logs because it is the standard elasticity "
                   "estimator and because the confidence interval and the tests against 0 and 1 are "
                   "computed from it. It also happens to give the lowest number in the set, so it "
                   "is stated here rather than left for a reader to find. What none of these "
                   "methods can fix is the fit itself: R² is "
                   f"{act['r2']:.3f} — the trend is close to flat however you draw it.")

    st.markdown("##### How many of those cars actually convert?")
    st.markdown("Same question, stated as the sheet states it: of the cars going past, what share "
                "stop in? The proforma has to assume a number. Here is that assumption against "
                "what each site achieved.")

    # Sites laid out left-to-right by what they achieved, with the sheet's assumed band shaded
    # behind them. A scatter of assumed-vs-achieved was tried first and wasted two thirds of its
    # width: the assumption occupies a narrow stripe (0.7–1.5%) while outcomes run to 3.6%, so
    # equal axes — which the y=x line needs — left the data crushed into one corner.
    ranked = tf.sort_values("capture").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    a_lo, a_hi = tcap.iloc[0].p10, tcap.iloc[0].p90
    a_med = tcap.iloc[0]["median"]

    figc = go.Figure()
    figc.add_scatter(x=list(ranked["rank"]) + list(ranked["rank"][::-1]),
                     y=[a_hi] * len(ranked) + [a_lo] * len(ranked), fill="toself",
                     fillcolor="rgba(235,104,52,0.16)", line=dict(width=0), hoverinfo="skip",
                     name=f"What the sheet assumed ({a_lo:.2%}–{a_hi:.2%})")
    figc.add_hline(y=a_med, line=dict(color=S2, width=2, dash="dash"))
    figc.add_scatter(x=ranked["rank"], y=ranked.capture, mode="markers", name="Each site achieved",
                     marker=dict(size=9, line=dict(width=1.5, color=SURFACE),
                                 color=np.where(ranked.capture >= ranked.pf_capture, S3, S1)),
                     customdata=np.stack([ranked.site, ranked[pf.TRAFFIC_KEY], ranked.daily_washes,
                                          ranked.pf_capture, ranked.capture_vs_assumed], axis=-1),
                     hovertemplate="<b>%{customdata[0]}</b><br>"
                                   "%{customdata[1]:,.0f} cars a day → "
                                   "%{customdata[2]:,.0f} washes a day<br>"
                                   "caught <b>%{y:.2%}</b> · the sheet assumed "
                                   "<b>%{customdata[3]:.2%}</b><br>"
                                   "= %{customdata[4]:.0%} of the assumption<extra></extra>")
    figc.add_annotation(x=len(ranked), y=a_med, text=f"the sheet's typical assumption {a_med:.2%}",
                        showarrow=False, xanchor="right", yshift=13,
                        font=dict(size=11, color=S2))
    st.plotly_chart(style(figc, height=420,
                          xaxis_title=f"The {len(ranked)} sites, worst conversion to best",
                          yaxis_title="Share of passing cars that stopped",
                          yaxis=dict(tickformat=".1%"),
                          xaxis=dict(showgrid=False, showticklabels=False),
                          margin=dict(l=70, r=25, t=86, b=50),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    ct = tcap[["which", "median", "p10", "p90", "spread"]].copy()
    ct.columns = ["", "Typical", "Low (p10)", "High (p90)", "Spread p90÷p10"]
    html_table(ct.set_index(""), index_label="Conversion rate",
               fmt={"Typical": "{:.2%}", "Low (p10)": "{:.2%}", "High (p90)": "{:.2%}",
                    "Spread p90÷p10": "{:.1f}×"})

    callout("What this shows", f"""
      <b>The assumption is almost a constant; the outcome is anything but.</b> The sheet's
        conversion rate runs {tcap.iloc[0].p10:.2%} to {tcap.iloc[0].p90:.2%} across all
        {tcap.attrs['n']} sites — a spread of {tcap.iloc[0].spread:.1f}×. What the sites achieved
        runs {tcap.iloc[1].p10:.2%} to {tcap.iloc[1].p90:.2%} — a spread of
        <b>{tcap.iloc[1].spread:.1f}×</b>. No fixed rate can cover that.
      <b>And it is set too high.</b> Typical assumed conversion is
        <b>{tcap.iloc[0]['median']:.2%}</b>; typical achieved is
        <b>{tcap.iloc[1]['median']:.2%}</b>. Only <b>{tcap.attrs['beat']} of
        {tcap.attrs['n']}</b> sites beat their own assumption; the median site landed at
        <b>{tcap.attrs['median_ratio']:.0%}</b> of it.
      <b>Note what this is not.</b> Grouping sites into traffic bands and plotting the conversion
        rate is circular — conversion is washes ÷ traffic, so it has to fall as traffic rises
        whether or not anything real is happening. Every dot here is instead scored against
        <i>its own site's</i> assumption, two numbers set independently of each other, so it could
        have come out on either side of the band. Most of it came out below.
    """, S3)

    st.markdown("##### So where does the forecast break?")

    figo = go.Figure()
    figo.add_bar(x=tov.label, y=tov.median_projected, name="Proforma projected",
                 marker=dict(color=S2, line=dict(width=2, color=SURFACE)),
                 hovertemplate="Road carries %{x} a day<br>projected <b>%{y:,.0f}</b> "
                               "washes a month<extra></extra>")
    figo.add_bar(x=tov.label, y=tov.median_actual, name="Actually washed",
                 marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
                 customdata=np.stack([tov.sites, tov.overshoot], axis=-1),
                 hovertemplate="Road carries %{x} a day<br>actually washed <b>%{y:,.0f}</b> "
                               "a month<br>the sheet projected %{customdata[1]:.2f}× this<br>"
                               "<span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                               "<extra></extra>")
    st.plotly_chart(style(figo, height=400, barmode="group",
                          title=dict(text="Projected vs actual, by how busy the road is"),
                          xaxis_title="Vehicles a day past the site",
                          yaxis_title="Washes a month, typical site",
                          xaxis=dict(type="category", showgrid=False, tickfont=dict(size=11)),
                          margin=dict(t=86), legend=dict(orientation="h", y=1.07, x=0)),
                    width="stretch")

    ot = tov[["label", "sites", "median_traffic", "median_projected", "median_actual",
              "overshoot"]].copy()
    ot.columns = ["Road (vehicles a day)", "Sites", "Typical traffic", "Projected washes/mo",
                  "Actual washes/mo", "Projected ÷ actual"]
    html_table(ot.set_index("Road (vehicles a day)"), index_label="Road (vehicles a day)",
               fmt={"Projected ÷ actual": "{:.2f}×"})

    # The sheet's own wash-count formula, verbatim. It is the mechanism the
    # chart above is showing: the projection ends in × Traffic_Count, so it
    # scales with traffic by construction while the actual washes do not.
    with st.expander("👁 The formula the proforma projects with"):
        st.code("(Cumulative_site_score * (1 + cumulative_demographic_score) % 85 "
                "* (1 * (1 + Year 3 Increase)) * 300) * Traffic_Count", language="text")
        st.caption("Straight from the sheet. Everything before the last term is a site score, so "
                   "the projection is proportional to **Traffic_Count** — which is exactly why the "
                   "orange bars above climb with the road while the blue ones do not.")

    callout("The mechanism behind the 58% error", f"""
      <b>On a quiet road the sheet is roughly right; on a busy one it projects double.</b>
        Over-projection climbs {tov.overshoot.iloc[0]:.2f}× → {tov.overshoot.iloc[1]:.2f}× →
        {tov.overshoot.iloc[2]:.2f}× → <b>{tov.overshoot.iloc[3]:.2f}×</b> across the four traffic
        quartiles. That trend is real: rank correlation <b>{tov.attrs['rho']:+.2f}</b>,
        p&nbsp;=&nbsp;{tov.attrs['p']:.4f} on {tov.attrs['n']} sites.
      <b>This is the tautology-free version of the conversion chart.</b> Both bars are free to move
        independently. The projected bars rise <b>{tov.attrs['projected_ratio']:.1f}×</b> across the
        quartiles — tracking the {tov.attrs['traffic_ratio']:.1f}× rise in traffic almost exactly.
        The actual bars rise only <b>{tov.attrs['wash_ratio']:.1f}×</b>.
      <b>So the headline error has an address.</b> Section ②'s 58% median miss is not spread evenly:
        it is concentrated in sites on busy roads, and it is caused by one assumption — that a wash
        catches a fixed share of whatever drives past.
      <b>The fix is not a haircut.</b> Cutting every projection by a third would fix the busy-road
        sites and start under-projecting the quiet ones, which are already at
        {tov.overshoot.iloc[0]:.2f}×. The slope has to change, not the level.
    """, CRITICAL)

    st.markdown("##### And the other half of the question: how fast is that road?")
    st.markdown("The sheet scores speed too, and rewards a slow road — "
                f"**{tsp.iloc[0].score:.2f} points** under 30 mph down to "
                f"**{tsp.iloc[-1].score:.2f}** over 50. The thinking is that you cannot turn into a "
                "car wash at 55 mph. Here is whether the sites bear that out.")

    figs = go.Figure()
    # Site count in the tick label, not just the table: two of these bands are 4 and 2 sites, and a
    # bar chart makes them look as solid as the 35-site one.
    sp_lab = [f"{b}<br><span style='font-size:10px;opacity:.65'>{int(n)} sites</span>"
              for b, n in zip(tsp.speed_band, tsp.sites)]
    figs.add_bar(x=sp_lab, y=tsp.median_washes, name="Typical washes a month",
                 marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
                 error_y=dict(type="data", symmetric=False,
                              array=tsp.p75 - tsp.median_washes,
                              arrayminus=tsp.median_washes - tsp.p25,
                              color=MUTED, thickness=1.5, width=6),
                 customdata=np.stack([tsp.sites, tsp.median_traffic, tsp.score], axis=-1),
                 hovertemplate="<b>%{x}</b><br>typical site: <b>%{y:,.0f}</b> washes a month<br>"
                               "typical road: %{customdata[1]:,.0f} cars a day<br>"
                               "the sheet awards %{customdata[2]:.2f} points<br>"
                               "<span style='opacity:.7'>%{customdata[0]:.0f} sites</span>"
                               "<extra></extra>")
    st.plotly_chart(style(figs, height=380, showlegend=False,
                          xaxis_title="Speed of the road the site sits on",
                          yaxis_title="Washes a month, typical site",
                          xaxis=dict(type="category", showgrid=False),
                          margin=dict(t=40)), width="stretch")

    st_ = tsp[["speed_band", "sites", "score", "median_traffic", "median_washes",
               "capture"]].copy()
    st_.columns = ["Speed of road", "Sites", "Points the sheet awards", "Typical traffic",
                   "Typical washes/mo", "Share of cars caught"]
    html_table(st_.set_index("Speed of road"), index_label="Speed of road",
               fmt={"Points the sheet awards": "{:.2f}", "Share of cars caught": "{:.2%}"})

    callout("Traffic speed: no evidence either way, on this sample", f"""
      <b>The bars do not separate.</b> Across the {tspt['bands']} speed bands the difference in
        volume is not significant (Kruskal-Wallis p&nbsp;=&nbsp;{tspt['kruskal_p']:.2f}), and the
        points the sheet awards for speed have a rank correlation of just
        <b>{tspt['score_rho']:+.2f}</b> with actual volume (p&nbsp;=&nbsp;{tspt['score_p']:.2f}).
      <b>The two end bands are anecdotes, not findings.</b> Under 30 mph is
        {int(tsp.iloc[0].sites)} sites and over 50 mph is {int(tsp.iloc[-1].sites)}. Both look weak,
        and neither can carry a conclusion. The real comparison is 30–40 against 40–50
        ({int(tsp[tsp.speed_band == '30–40 mph'].sites.iloc[0])} against
        {int(tsp[tsp.speed_band == '40–50 mph'].sites.iloc[0])} sites), and those two are
        {tsp[tsp.speed_band == '30–40 mph'].median_washes.iloc[0]:,.0f} against
        {tsp[tsp.speed_band == '40–50 mph'].median_washes.iloc[0]:,.0f} washes a month — the same
        number.
      <b>Speed and traffic are tangled, so the raw comparison cannot be trusted either way.</b>
        The 40–50 mph roads here carry
        {tsp[tsp.speed_band == '40–50 mph'].median_traffic.iloc[0]:,.0f} cars a day against
        {tsp[tsp.speed_band == '30–40 mph'].median_traffic.iloc[0]:,.0f} on the 30–40 roads — faster
        roads are busier roads. Holding traffic constant, each extra 10 mph is worth
        <b>{tspt['per_10mph']:+.0%}</b> on volume, which sounds like a lot until you see the
        p-value: <b>{tspt['mph_p']:.2f}</b>. Directionally it agrees with the sheet — slower is
        better — but on {tspt['n']} sites it is indistinguishable from noise.
      <b>What to do with the box.</b> Keep it if it encodes something real about site access, but do
        not let it move a number. Nothing here supports weighting it.
    """, WARNING)

    # ---- the size of the prize -------------------------------------------------------------------
    st.markdown("#### And how much does the whole sheet decide?")
    best_loo = sp.attrs["best_loo"]
    figp = go.Figure()
    figp.add_bar(y=["what decides it"], x=[best_loo], orientation="h",
                 name=f"The build sheet ({best_loo:.0%})",
                 marker=dict(color=S1, line=dict(width=2, color=SURFACE)),
                 hovertemplate=f"The ten boxes explain <b>{best_loo:.0%}</b><extra></extra>")
    figp.add_bar(y=["what decides it"], x=[1 - best_loo], orientation="h",
                 name=f"The operator, the market, everything else ({1 - best_loo:.0%})",
                 marker=dict(color=MUTED, line=dict(width=2, color=SURFACE)),
                 hovertemplate=f"Everything else: <b>{1 - best_loo:.0%}</b><extra></extra>")
    st.plotly_chart(style(figp, height=190, barmode="stack",
                          xaxis=dict(tickformat=".0%", range=[0, 1]),
                          xaxis_title="Share of the difference between one site and another",
                          yaxis=dict(showticklabels=False, showgrid=False),
                          legend=dict(orientation="h", y=-0.5, x=0),
                          margin=dict(l=10, r=20, t=20, b=70)), width="stretch")

    callout("What this shows", f"""
      <b>Everything on the build sheet, at its very best, accounts for about
        {best_loo:.0%}</b> of why one car wash is busier than another.
      <b>The other {1 - best_loo:.0%} is who runs it and where it is.</b> That is not a gap in the
        data, it is the answer: the operator and the market decide most of it, and no scoring sheet
        about the building can reach them.
      <b>Adding more boxes makes it worse, not better.</b> Using the best <b>four</b> boxes is as
        good as it gets. Feed in all ten and the forecast ends up <b>worse than using none of
        them</b>, because with only {sp.attrs['n']} sites the extra boxes just memorise these
        particular sites instead of learning anything about car washes.
      <b>Which is why the model only carries three.</b> Model 5 uses
        <b>{', '.join(sp.attrs['model5_labels']).lower()}</b> and then leans on the location. Those
        three alone get <b>{sp.attrs['model5_share']:.0%}</b> of everything the full sheet could
        ever offer.
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
        # Formula on x, build on y: the formula is the input being tested and
        # the build is the outcome, so the "built longer" points now sit above
        # the parity line rather than below it.
        figl.add_scatter(x=g.formula_m, y=g.actual_m, mode="markers", name=name,
                         marker=dict(size=11, color=col, line=dict(width=1.6, color=SURFACE)),
                         customdata=np.stack([g.client_name.fillna("—"), g.state.fillna("—"),
                                              g.actual_m, g.formula_m, g.gap_m], axis=-1),
                         hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                                       "Built: <b>%{customdata[2]:.0f} m</b><br>"
                                       "Formula asked for: %{customdata[3]:.0f} m<br>"
                                       "→ built %{customdata[4]:+.0f} m longer<extra></extra>")
    st.plotly_chart(style(figl, height=470, xaxis_title="Length the formula calls for (m)",
                          yaxis_title="Tunnel actually built (m)",
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
    callout("What this shows", f"""
      <b>Where the recommended length comes from.</b> It is the proforma's own wash promise,
        converted into feet. Nothing else goes into it.
      <b>The numbers say the same thing.</b> The recommendation moves almost perfectly in step with
        the wash promise (<b>{row_vol.rho:.3f}</b> out of a possible 1.00) but hardly relates at all
        to the tunnel that actually got built (<b>{row_len.rho:.2f}</b>). It is typically
        <b>{ts['mae']:.0f} m</b> away from it, on tunnels that are only 20–61 m long.
      <b>So two things go wrong at once.</b> Because the recommendation is just the wash promise in
        different units, an over-promised site automatically gets an over-long recommendation. And
        then the build ignores it anyway: <b>{ts['built_longer_share']*100:.0f}% of sites end up
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
