"""
Section ① — tunnel length, wash trajectories and year-5 volume.

A demo surface: charts, the findings that come out of them, and an explorer the reader can drive.
All the maths lives in `tunnel_data.py` (Streamlit-free, shared with the notebook).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import tunnel_data as td
from ui import DARK, INK, INK2, S1, S2, S3, SURFACE, callout, html_table, style

STATUS = {"Overbuilt": "#d03b3b", "Right-sized": "#0ca30c", "At capacity": "#fab219"}
ICON = {"Overbuilt": "▼", "Right-sized": "●", "At capacity": "▲"}
ORDER = ["Overbuilt", "Right-sized", "At capacity"]
MAT_COLOR = {"5+ years observed": S1, "3–4 years observed": S3}


@st.cache_data(show_spinner=False)
def _build(basis: str) -> pd.DataFrame:
    return td.build(basis=basis)


@st.cache_data(show_spinner=False)
def _panel() -> pd.DataFrame:
    return td.operating_panel()


@st.cache_data(show_spinner=False)
def _cohort() -> pd.DataFrame:
    return td.cohort_curve()


@st.cache_data(show_spinner=False)
def _validation() -> pd.DataFrame:
    return td.validation()


@st.cache_data(show_spinner=False)
def _by_basis() -> pd.DataFrame:
    return td.utilisation_by_basis()


@st.cache_data(show_spinner=False)
def _traj(site_key: str) -> pd.DataFrame:
    return td.site_trajectory(site_key)


@st.cache_data(show_spinner=False)
def _cohort_peaks() -> pd.DataFrame:
    return td.cohort_peaks()


@st.cache_data(show_spinner=False)
def _cohort_util() -> pd.DataFrame:
    return td.cohort_utilisation()


def render() -> None:
    d = _build(td.DEFAULT_BASIS)
    h = td.headline(d)

    st.title("Tunnel length analysis")
    st.markdown(f"**{h['n_sites']} sites analysed** — every site with a measured tunnel, measured "
                f"hourly throughput and at least three years of trading. "
                f"{h['n_observed5']} of them have reached year 5.")

    # =============================================================================================
    st.divider()
    st.header("1 · How a car wash ramps up")
    st.caption(f"Median annualised wash rate by operating year, on the "
               f"{int(_cohort().sites.iloc[0])} sites observed in every one of their first five "
               "years. Shaded band is the middle half of those sites.")

    cc = _cohort()
    figr = go.Figure()
    figr.add_scatter(x=list(cc.operating_year) + list(cc.operating_year[::-1]),
                     y=list(cc.p75) + list(cc.p25[::-1]), fill="toself",
                     fillcolor="rgba(57,135,229,0.15)", line=dict(width=0),
                     hoverinfo="skip", name="middle half of sites")
    figr.add_scatter(x=cc.operating_year, y=cc["median"], mode="lines+markers", name="median site",
                     line=dict(color=S1, width=3),
                     marker=dict(size=10, line=dict(width=2, color=SURFACE)),
                     customdata=np.stack([cc.sites, cc.share_of_year5], axis=-1),
                     hovertemplate="Operating year %{x}<br><b>%{y:,.0f} washes/yr</b><br>"
                                   "= %{customdata[1]:.0%} of its year-5 volume<br>"
                                   "<span style='opacity:.7'>%{customdata[0]:.0f} sites"
                                   "</span><extra></extra>")
    st.plotly_chart(style(figr, height=420, xaxis_title="Operating year",
                          yaxis_title="Washes per year", xaxis=dict(dtick=1),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")

    v = _validation()
    y1 = v[v.from_operating_year == 1].iloc[0]
    y3 = v[v.from_operating_year == 3].iloc[0]
    sh = cc.set_index("operating_year").share_of_year5
    callout("What this shows", f"""
      <b>Reading.</b> A new site washes <b>{sh.get(1, float("nan")):.0%} of its eventual volume in
        its first year</b>, gets to <b>{sh.get(2, float("nan")):.0%} in year 2</b>, and then stops
        growing.
      <b>So-what.</b> Sites do not climb for five years — they arrive almost fully formed in year 2.
        That means <b>a site can be judged on real trading after two years</b> instead of waiting
        out the projection.
    """)

    st.markdown("**How reliable is the year-5 call?** Every site that reached year 5, predicted "
                "from each earlier year and checked against what actually happened.")
    vt = v.copy()
    vt.columns = ["From operating year", "Sites", "Error if you assume no growth",
                  "Error using the ramp", "Bias"]
    vt.index = range(1, len(vt) + 1)
    html_table(vt, fmt={"Error if you assume no growth": "{:,.1f}%",
                        "Error using the ramp": "{:,.1f}%", "Bias": "{:,.2f}×"})
    callout("What this shows", f"""
      <b>Reading.</b> Knowing that growth curve, one year of real trading is enough to call year 5
        to within <b>{y1.mdape_ramp:.0f}%</b>. Taking the first year at face value instead leaves
        you <b>{y1.mdape_naive:.0f}%</b> out. After three years the answer is within
        <b>{y3.mdape_ramp:.0f}%</b>.
      <b>So-what.</b> We can put a reliable long-run number on a site <b>after its first year</b> —
        roughly twice as accurate as reading that first year straight — so young sites do not have
        to sit outside the plan.
    """, S3)

    # =============================================================================================
    st.divider()
    st.header("2 · Explore any site")
    st.caption("Pick a site to see its actual trajectory. Sites with five or more years of trading "
               "are listed first.")

    fcol, scol = st.columns([1, 2])
    with fcol:
        cohorts = st.multiselect("Show which sites", td.MATURITY_ORDER,
                                 default=td.MATURITY_ORDER)
    pool = d[d.maturity.isin(cohorts)] if cohorts else d
    pool = pool.sort_values(["maturity", "year5_washes"], ascending=[True, False])
    with scol:
        if pool.empty:
            st.warning("No sites in that selection.")
            return
        labels = {f"{r.site} — {r.where}  ({r.years_observed:.0f} yrs)": r.site_key
                  for r in pool.itertuples()}
        picked = st.selectbox("Site", list(labels), index=0)
    row = d[d.site_key == labels[picked]].iloc[0]

    t = _traj(row.site_key)
    figt = go.Figure()
    for kind, colour, dash in [("Observed", S1, "solid"), ("Projected", S2, "dot")]:
        seg = t[t.kind == kind]
        if seg.empty:
            continue
        if kind == "Projected":  # join the dotted line to the last observed point
            last_obs = t[t.kind == "Observed"].tail(1)
            seg = pd.concat([last_obs, seg])
        figt.add_scatter(x=seg.opyear, y=seg.rate, mode="lines+markers", name=kind,
                         line=dict(color=colour, width=3, dash=dash),
                         marker=dict(size=11, line=dict(width=2, color=SURFACE)),
                         hovertemplate="Operating year %{x}<br><b>%{y:,.0f}</b> washes/yr"
                                       f"<br><span style='opacity:.7'>{kind.lower()}</span>"
                                       "<extra></extra>")
    figt.add_hline(y=row.year5_washes, line=dict(color=INK2, width=1.5, dash="dash"),
                   annotation_text=f"year 5: {row.year5_washes:,.0f}",
                   annotation_position="bottom right", annotation_font=dict(color=INK2, size=11))
    st.plotly_chart(style(figt, height=400, xaxis_title="Operating year",
                          yaxis_title="Washes per year", xaxis=dict(dtick=1),
                          legend=dict(orientation="h", y=1.02, x=0)), width="stretch")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Year-5 washes", f"{row.year5_washes:,.0f}", row.year5_source, delta_color="off")
    k2.metric("Tunnel", f"{row.tunnel_ft:.0f} ft", f"{row.tunnel_m:.0f} m", delta_color="off")
    k3.metric("Its busiest day ever", f"{row.peak_cars_per_hour:.0f} cars",
              f"rated for {row.capacity_cars_per_hour:.0f}", delta_color="off")
    k4.metric("Capacity used", f"{row.utilisation*100:.0f}%",
              f"{ICON[row.verdict]} {row.verdict}", delta_color="off")

    st.markdown(f"**{row.site}** opened {row.opened:%b %Y} in {row['where']}. It has "
                f"{row.years_observed:.0f} years of trading, grew "
                f"**{row.growth_y1_to_y5:.1f}×** from its first year to year 5, and its highest "
                f"recorded day used **{row.utilisation*100:.0f}%** of the tunnel it was built with.")

    # =============================================================================================
    st.divider()
    st.header("3 · Does a longer tunnel deliver more washes?")
    st.caption("How many washes a site does once mature, against the tunnel it was built with. "
               "Colour shows how much real trading history is behind each point.")

    fit = td.length_vs_volume(d)
    figs = go.Figure()
    for m in td.MATURITY_ORDER:
        g = d[(d.maturity == m) & d.year5_washes.notna()]
        if g.empty:
            continue
        figs.add_scatter(x=g.tunnel_ft, y=g.year5_washes, mode="markers", name=m,
                         marker=dict(size=11, color=MAT_COLOR[m],
                                     line=dict(width=1.6, color=SURFACE)),
                         customdata=np.stack([g.site, g["where"], g.utilisation, g.tunnel_m,
                                              g.year5_source], axis=-1),
                         hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                                       "Tunnel: <b>%{x:.0f} ft</b> (%{customdata[3]:.0f} m)<br>"
                                       "Year-5 washes: <b>%{y:,.0f}</b>/yr "
                                       "<span style='opacity:.7'>(%{customdata[4]})</span><br>"
                                       "Uses %{customdata[2]:.0%} of its tunnel at peak"
                                       "<extra></extra>")
    xs = np.linspace(d.tunnel_ft.min(), d.tunnel_ft.max(), 50)
    figs.add_scatter(x=xs, y=fit["slope"]*xs + fit["intercept"], mode="lines",
                     name=f"+{fit['slope']:,.0f} washes per foot",
                     line=dict(color=INK2, width=2, dash="dash"), hoverinfo="skip")
    st.plotly_chart(style(figs, height=470, xaxis_title="Built tunnel length (ft)",
                          yaxis_title="Washes per year once mature",
                          legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10))),
                    width="stretch")

    tier = (d.groupby("tier", observed=True)
              .agg(**{"Sites": ("site_key", "size"),
                      "Median length (ft)": ("tunnel_ft", "median"),
                      "Median year-5 washes": ("year5_washes", "median"),
                      "Best hour (cars)": ("peak_cars_per_hour", "median"),
                      "Capacity used": ("utilisation", "median")}))
    tier.index = [str(i) for i in tier.index]
    html_table(tier, fmt={"Median length (ft)": "{:,.0f}", "Median year-5 washes": "{:,.0f}",
                          "Best hour (cars)": "{:,.0f}", "Capacity used": "{:.0%}"},
               index_label="Length band")

    callout("What this shows", f"""
      <b>Reading.</b> Longer tunnels do wash more — about <b>{fit['slope']:,.0f} extra washes a
        year per foot</b> — but tunnel length accounts for only <b>{fit['r2']*100:.0f}%</b> of why
        one site is busier than another. Two sites with the same tunnel routinely differ several
        times over.
      <b>So-what.</b> The table underneath shows why. As tunnels get longer they attract more cars
        <i>and</i> gain more capacity at much the same rate, so <b>the share of the tunnel actually
        used hardly moves</b>. Extra length buys spare capacity, not extra business.
    """, S2)

    # =============================================================================================
    st.divider()
    st.header("4 · How much of the tunnel do we use — and does it close with age?")
    st.caption("Peak demand against tunnel length. The dashed line is the sizing rule: a site "
               "sitting on it is using its tunnel fully. Sites are grouped by how long they have "
               "been trading, and each site's peak scaled to how busy that year was. This is the "
               "one chart here that uses **all 77 sites in the file** rather than the 39-site "
               "analysis set — the young sites are the comparison.")

    cp = _cohort_peaks()
    show = st.multiselect("Peak levels to plot", td.PEAK_ORDER,
                          default=["Median daily peak", "p90 daily peak", "Highest daily peak"],
                          key="peak_levels")
    # One hue, stepped by peak level. The ramp runs the opposite way per theme so the most
    # important series (the highest peak) is always the most prominent against the surface —
    # a dark navy step is nearly invisible on the dark chart background.
    shade = ({"Median daily peak": "#1c5cab", "p75 daily peak": "#2a78d6",
              "p90 daily peak": "#5598e7", "Highest daily peak": "#b7d3f6"} if DARK else
             {"Median daily peak": "#9ec5f4", "p75 daily peak": "#5598e7",
              "p90 daily peak": "#2a78d6", "Highest daily peak": "#184f95"})

    hi = float(max(cp.tunnel_ft.max(),
                   cp[list(td.PEAK_BASIS.values())].max().max())) * 1.08
    present = [c for c in td.COHORTS if (cp.cohort == c).any()]
    cols = st.columns(len(present))
    for col, coh in zip(cols, present):
        g = cp[cp.cohort == coh]
        with col:
            fig = go.Figure()
            fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", showlegend=False,
                            line=dict(color=STATUS["Overbuilt"], width=1.5, dash="dash"),
                            hoverinfo="skip")
            for lvl in show:
                c = td.PEAK_BASIS[lvl]
                fig.add_scatter(x=g.tunnel_ft, y=g[c], mode="markers", name=lvl,
                                marker=dict(size=8, color=shade[lvl],
                                            line=dict(width=1, color=SURFACE)),
                                customdata=np.stack([g.site, g[c] / g.tunnel_ft], axis=-1),
                                hovertemplate="<b>%{customdata[0]}</b><br>"
                                              + lvl + ": %{y:.0f} cars<br>"
                                              "Tunnel %{x:.0f} ft<br>"
                                              "→ uses %{customdata[1]:.0%} of it<extra></extra>",
                                showlegend=(coh == present[0]))
            st.plotly_chart(style(fig, height=370,
                                  title=dict(text=f"{coh} (n={g.site_key.nunique()})",
                                             font=dict(size=13)),
                                  xaxis_title="Tunnel (ft)", yaxis_title="Peak cars",
                                  xaxis=dict(range=[0, hi]), yaxis=dict(range=[0, hi]),
                                  margin=dict(l=48, r=12, t=104, b=45),
                                  legend=dict(orientation="h", y=1.30, x=0, font=dict(size=9))),
                            width="stretch")

    piv = (_cohort_util().pivot(index="peak_level", columns="cohort", values="median_share")
           .reindex([l for l in td.PEAK_ORDER]))
    piv = piv[[c for c in present if c in piv.columns]]
    html_table(piv, fmt={c: "{:.0%}" for c in piv.columns}, index_label="Peak level")

    first, last = present[0], present[-1]
    p90_a, p90_b = piv.loc["p90 daily peak", first], piv.loc["p90 daily peak", last]
    hi_b = piv.loc["Highest daily peak", last]
    med_b = piv.loc["Median daily peak", last]
    callout("What this shows", f"""
      <b>Reading.</b> Every point sits below the line, in every cohort. On a busy (p90) day a
        <b>{first}</b> site uses <b>{p90_a:.0%}</b> of its tunnel and a <b>{last}</b> site uses
        <b>{p90_b:.0%}</b>. On an ordinary day the {last} group uses <b>{med_b:.0%}</b>, and even on
        its single highest recorded day it reaches <b>{hi_b:.0%}</b>.
      <b>So-what.</b> Age closes some of the gap and then stops — the curve flattens after year 3,
        exactly where §1 showed volume stops growing. The spare capacity is not a start-up problem
        that trades its way out; it is still there in the oldest sites.
    """, STATUS["Overbuilt"])

    # =============================================================================================
    st.divider()
    st.header("5 · Where the spare tunnel is")
    st.caption("For the sites furthest from their rating: the length their own highest recorded day calls "
               "for, against the length that was built.")

    over = d[d.verdict == "Overbuilt"].nlargest(15, "excess_ft")
    labels = [f"{i}. {str(r.site)[:26]}" for i, r in enumerate(over.itertuples(), 1)]
    fige = go.Figure()
    fige.add_bar(y=labels, x=over.required_ft, orientation="h", name="length its busiest day needs",
                 marker=dict(color=S1, line=dict(width=1.5, color=SURFACE)),
                 hovertemplate="%{y}<br>Its highest day needs <b>%{x:.0f} ft</b><extra></extra>")
    fige.add_bar(y=labels, x=over.excess_ft, orientation="h", name="spare length",
                 marker=dict(color=STATUS["Overbuilt"], line=dict(width=1.5, color=SURFACE)),
                 customdata=np.stack([over.excess_share, over.tunnel_ft], axis=-1),
                 hovertemplate="%{y}<br>Built %{customdata[1]:.0f} ft → <b>%{x:.0f} ft spare</b>"
                               "<br>= %{customdata[0]:.0%} of the tunnel<extra></extra>")
    st.plotly_chart(style(fige, height=560, barmode="stack", xaxis_title="Tunnel length (ft)",
                          yaxis=dict(autorange="reversed", showgrid=False),
                          # a horizontal bar chart puts its first bar hard against the top of the
                          # plot, so the legend needs its own band rather than sitting just above it
                          margin=dict(l=60, r=25, t=86, b=50),
                          legend=dict(orientation="h", y=1.07, x=0)), width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Sites under half their rating", f"{h['n_overbuilt']}",
              f"of {h['n_sites']}", delta_color="off")
    m2.metric("Median spare length", f"{h['median_excess_ft']:.0f} ft",
              f"{h['median_excess_ft']/td.FT_PER_M:.0f} m", delta_color="off")
    m3.metric("Spare share of the tunnel", f"{h['median_excess_share']*100:.0f}%")

    callout("What this shows", f"""
      <b>Reading.</b> <b>{h['n_overbuilt']} of {h['n_sites']}</b> sites never get past half their
        tunnel — not on an average day, but on their single highest recorded day. A typical one of
        them carries <b>{h['median_excess_ft']:.0f} ft
        ({h['median_excess_ft']/td.FT_PER_M:.0f} m) it has never needed —
        {h['median_excess_share']*100:.0f}% of the tunnel</b>.
      <b>So-what.</b> These are the builds to look at before the next site is signed off the same way.
        The cars that turned up would have fitted through a much shorter tunnel.
    """, STATUS["Overbuilt"])

    # =============================================================================================
    st.divider()
    st.header("6 · All the data")
    st.caption("Filter, sort and download. Everything the charts above are drawn from.")

    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        mats = st.multiselect("Data maturity", td.MATURITY_ORDER, default=td.MATURITY_ORDER,
                              key="tbl_mat")
    with f2:
        verds = st.multiselect("Capacity verdict", ORDER, default=ORDER, key="tbl_ver")
    with f3:
        sort_by = st.selectbox("Sort by", ["Year-5 washes", "Capacity used", "Spare length",
                                           "Tunnel length", "Site name"])
    key, asc = {"Year-5 washes": ("year5_washes", False), "Capacity used": ("utilisation", True),
                "Spare length": ("excess_ft", False), "Tunnel length": ("tunnel_ft", False),
                "Site name": ("site", True)}[sort_by]
    view = d[d.maturity.isin(mats) & d.verdict.isin(verds)].sort_values(key, ascending=asc)
    st.caption(f"Showing **{len(view)}** of {len(d)} sites.")

    cols = ["site", "where", "years_observed", "maturity", "year5_washes", "year5_source",
            "growth_y1_to_y5", "tunnel_ft", "peak_cars_per_hour", "capacity_cars_per_hour",
            "utilisation", "excess_ft", "verdict"]
    tbl = view[cols].rename(columns={
        "site": "Site", "where": "Where", "years_observed": "Yrs", "maturity": "History",
        "year5_washes": "Year-5 washes", "year5_source": "Basis", "growth_y1_to_y5": "Y1→Y5",
        "tunnel_ft": "Length (ft)", "peak_cars_per_hour": "Best hour",
        "capacity_cars_per_hour": "Rated", "utilisation": "Capacity used",
        "excess_ft": "Spare (ft)", "verdict": "Verdict"})
    tbl.index = range(1, len(tbl) + 1)
    html_table(tbl, fmt={"Yrs": "{:.0f}", "Year-5 washes": "{:,.0f}", "Y1→Y5": "{:,.1f}×",
                         "Length (ft)": "{:,.0f}", "Best hour": "{:,.0f}", "Rated": "{:,.0f}",
                         "Spare (ft)": "{:,.0f}"},
               bars={"Capacity used": 1.0}, index_label="#")
    st.download_button("Download this table (CSV)", view[cols].to_csv(index=False),
                       "tunnel_analysis.csv", "text/csv")

    with st.expander("Year-by-year wash history for every site"):
        p = _panel()
        wide = (p[p.usable].pivot_table(index=["site"], columns="opyear", values="rate")
                .round(0))
        wide.columns = [f"Year {c}" for c in wide.columns]
        wide = wide.reindex(d.set_index("site").index).dropna(how="all")
        html_table(wide.head(40), fmt={c: "{:,.0f}" for c in wide.columns}, index_label="Site")
        st.caption(f"Annualised washes per operating year — part-years scaled to a full year. "
                   f"First 40 of {len(wide)} sites.")
        st.download_button("Download the full year-by-year history (CSV)",
                           wide.to_csv(), "wash_history_by_operating_year.csv", "text/csv")
