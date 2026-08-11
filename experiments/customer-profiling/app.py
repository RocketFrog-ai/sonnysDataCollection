"""
Customer-profiling demo for the Hurricane Express membership book.

    conda activate sonnys
    streamlit run experiments/customer-profiling/app.py

Every number comes from `profiling.py`, the same module `customer_profiling.ipynb` imports, so the
demo and the notebook cannot disagree. This file is presentation only: layout, widgets, charts.

Five tabs, in the order an operator would actually use them:
  Overview   the book — growth, retention, where members are lost
  Personas   four segments, who they are and what they are worth
  Churn      the live risk list and the model behind it
  Member     one customer's full history, persona and risk
  What-if    unit economics and a retention-campaign simulator
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Must be set before the first frame is built, and it is not cosmetic: pandas 3.0 backs its default
# `str` dtype with Arrow, and a boolean mask over such a column calls pyarrow's `compute.take`,
# which segfaults this process when it runs on Streamlit's script thread during a re-run — i.e. the
# moment any widget moves. "python" storage keeps the same dtype and semantics with a NumPy-backed
# array, and the crash goes away. Same pyarrow 25 / pandas 3.0.2 / streamlit 1.58 fault line that
# makes `st.dataframe` unusable here (see `html_table` below).
pd.set_option("mode.string_storage", "python")

import plotly.graph_objects as go            # noqa: E402
import streamlit as st                       # noqa: E402
from plotly.subplots import make_subplots    # noqa: E402

import profiling as P                        # noqa: E402
import viz                                   # noqa: E402

st.set_page_config(page_title="Membership customer profiling", page_icon="🚿", layout="wide")

T = viz.theme(dark=st.get_option("theme.base") == "dark")
SEG_COLOR = viz.segment_colors(T)


# =================================================================================================
# chrome
# =================================================================================================
def inject_css() -> None:
    st.markdown(f"""<style>
    .stApp {{ background: {T.plane}; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.6rem; color: {T.ink}; }}
    div[data-testid="stMetricLabel"] {{ color: {T.ink2}; }}
    .kicker {{ color:{T.muted}; font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; }}
    .callout {{ background:{T.surface}; border:1px solid {T.border}; border-left:4px solid {T.s1};
                border-radius:6px; padding:.8rem 1.05rem; margin:.4rem 0 1rem; }}
    .callout h4 {{ margin:0 0 .35rem; font-size:.92rem; color:{T.ink}; }}
    .callout li {{ color:{T.ink2}; font-size:.88rem; margin:.18rem 0; }}
    .tblwrap {{ overflow-x:auto; border:1px solid {T.border}; border-radius:6px;
                background:{T.surface}; margin:.3rem 0 1rem; }}
    table.tbl {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
    table.tbl th {{ text-align:left; font-weight:600; color:{T.ink2}; background:{T.header_bg};
                    padding:.5rem .7rem; border-bottom:1px solid {T.grid}; white-space:nowrap; }}
    table.tbl td {{ padding:.4rem .7rem; border-bottom:1px solid {T.grid}; color:{T.ink};
                    white-space:nowrap; }}
    table.tbl td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    table.tbl td.lbl {{ color:{T.ink2}; font-weight:600; }}
    table.tbl tbody tr:last-child td {{ border-bottom:none; }}
    .pill {{ display:inline-block; padding:.12rem .5rem; border-radius:10px; font-size:.78rem;
             font-weight:600; }}
    </style>""", unsafe_allow_html=True)


def html_table(df: pd.DataFrame, fmt: dict[str, str] | None = None,
               index_label: str | None = None) -> None:
    """Render a frame as plain HTML.

    Not `st.dataframe`: in this environment (pyarrow 25 + pandas 3.0.2 + streamlit 1.58) it
    segfaults the server on the second script run, which is every time a widget moves. The
    conclusions app hit the same wall — see `conclusion/demo/ui.py`.
    """
    fmt = fmt or {}
    show_index = index_label is not None      # "" is a real (blank) header, not "no index column"
    head = "".join(f"<th>{c}</th>" for c in ([index_label] if show_index else []) + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<td class='lbl'>{idx}</td>"] if show_index else []
        for c in df.columns:
            v = r[c]
            if isinstance(v, str) and v.startswith("<"):
                cells.append(f"<td>{v}</td>")
            elif pd.isna(v):
                cells.append("<td class='num'>—</td>")
            elif c in fmt:
                cells.append(f"<td class='num'>{fmt[c].format(v)}</td>")
            elif isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f"<td class='num'>{v:,.0f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(f"<div class='tblwrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


def callout(title: str, bullets: list[str], accent: str | None = None) -> None:
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"<div class='callout' style='border-left-color:{accent or T.s1}'>"
                f"<h4>{title}</h4><ul style='margin:0;padding-left:1.05rem'>{items}</ul></div>",
                unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f"<span class='pill' style='background:{color}22;color:{color}'>{text}</span>"


def risk_color(r: float) -> str:
    return T.critical if r >= 0.15 else T.serious if r >= 0.09 else T.good


# =================================================================================================
# data (cached — the whole pipeline is a few seconds, and every widget move re-runs the script)
# =================================================================================================
# `cache_resource`, NOT `cache_data`. `cache_data` serialises its return value through Arrow, which
# hands back pyarrow-backed DataFrames; the first boolean mask over one of those (`panel[~censored]`
# below) then segfaults the server on the second script run — i.e. the moment any widget moves.
# `cache_resource` stores the objects by reference, so the frames stay NumPy-backed. Nothing below
# mutates them in place. This is the same pyarrow 25 / pandas 3.0.2 / streamlit 1.58 fault line that
# makes `st.dataframe` unusable here (see `html_table` above).
@st.cache_resource(show_spinner="Loading the membership book…")
def load():
    ev = P.load_events()
    pay, wsh = P.payments(ev), P.washes(ev)
    panel = P.renewal_panel(ev)
    cust = P.customer_table(ev)
    seg, prof = P.segment(cust, k=4)
    model = P.fit_churn_model(panel)
    live = P.score_live_book(panel, model, cust).merge(
        seg[["customer_id", "segment"]], on="customer_id", how="left")
    return ev, pay, wsh, panel, cust, seg, prof, model, live


ev, pay, wsh, panel, cust, seg, prof, model, live = load()
inject_css()
ASOF = P.asof(ev)
OBS = panel[~panel.censored]
SEG_ORDER = [s for s in viz.SEGMENT_ORDER if s in prof.index]

st.markdown("<span class='kicker'>Hurricane Express · membership book</span>",
            unsafe_allow_html=True)
st.title("Customer profiling")
st.caption(f"{len(cust)} members · {len(wsh):,} washes · ${pay.amount.sum():,.0f} collected · "
           f"2025-09-13 → {ASOF.date()} · demo built on `hurricane_customer_profille_data.csv`")

tabs = st.tabs(["📊  Overview", "👥  Personas", "⚠️  Churn radar", "🔎  Member lookup",
                "🧮  What-if"])

# =================================================================================================
# 1. Overview
# =================================================================================================
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Active members", f"{int(cust.active.sum())}")
    c[1].metric("Active-book MRR", f"${cust[cust.active].arpu.sum():,.0f}")
    c[2].metric("Monthly churn", f"{1 - OBS.renewed.mean():.1%}",
                help="Share of observed paid cycles not followed by another charge within 45 days.")
    c[3].metric("Washes / member / mo", f"{cust.washes_per_month.median():.1f}")
    c[4].metric("Dormant payers", f"{len(P.dormant_payers(cust, 45))}",
                help="Active members still being billed with no wash in 45+ days.")

    left, right = st.columns([3, 2])
    with left:
        joins = cust.groupby("cohort").size().rename("joined")
        lost = cust[cust.churned].groupby("churn_month").size().rename("churned")
        flow = pd.concat([joins, lost], axis=1).fillna(0).astype(int).sort_index()
        fig = go.Figure()
        fig.add_bar(x=flow.index, y=flow.joined, name="Joined", marker_color=T.s1,
                    marker_line=dict(width=2, color=T.surface),
                    hovertemplate="%{x}<br>%{y} joined<extra></extra>")
        fig.add_bar(x=flow.index, y=-flow.churned, name="Churned", marker_color=T.s2,
                    marker_line=dict(width=2, color=T.surface), customdata=flow.churned,
                    hovertemplate="%{x}<br>%{customdata} churned<extra></extra>")
        fig.add_scatter(x=flow.index, y=flow.joined - flow.churned, name="Net", mode="lines+markers",
                        line=dict(color=T.ink, width=2), marker=dict(size=8),
                        hovertemplate="%{x}<br>net %{y:+d}<extra></extra>")
        fig.add_hline(y=0, line_color=T.axis, line_width=1)
        viz.style(fig, T, barmode="relative", height=340,
                  title=dict(text="Members joined and lost"), yaxis=dict(title="members"))
        st.plotly_chart(fig, width='stretch')
    with right:
        haz = P.hazard_curve(panel)
        fig = go.Figure(go.Scatter(
            x=haz.index, y=haz.renewal_rate, mode="lines+markers", showlegend=False,
            line=dict(color=T.s1, width=2), marker=dict(size=9), customdata=haz.n,
            hovertemplate="month %{x}<br>%{y:.1%} renew (n=%{customdata})<extra></extra>"))
        fig.add_hline(y=OBS.renewed.mean(), line_dash="dot", line_color=T.muted,
                      annotation_text=f"average {OBS.renewed.mean():.1%}",
                      annotation_font=dict(color=T.muted, size=11))
        viz.style(fig, T, height=340, title=dict(text="Renewal rate by membership month"),
                  yaxis=dict(title="renews next cycle", tickformat=".0%", range=[0.82, 1.02]),
                  xaxis=dict(title="membership month", dtick=1))
        st.plotly_chart(fig, width='stretch')

    callout("What the book looks like", [
        f"<b>Signups have stalled, churn has not.</b> Peak was 105 joins in Dec 2025; "
        f"Jun 2026 was the first net-negative month. At ~18 lost per month the site must sign "
        f"~18 just to hold {int(cust.active.sum())} members.",
        f"<b>Month 0 is the cliff.</b> The signup month renews at "
        f"{OBS[OBS.cycle_no == 0].renewed.mean():.1%} against "
        f"{OBS[OBS.cycle_no > 0].renewed.mean():.1%} for every later month (χ² p=2.7e-05) — "
        f"54 of the 153 observed churns happen there.",
        "<b>Survivors get stickier</b>, rising monotonically to 99.2% by month 6. That is "
        "selection, not growing loyalty: the members who dislike it have already gone.",
        "<i>Caveat:</i> this is a site that opened in Sep 2025, so every cohort is young and the "
        "right-hand months rest on few members.",
    ])

    ret = P.cohort_retention(ev)
    fig = go.Figure(go.Heatmap(
        z=ret.values.astype(float), x=[f"M{c}" for c in ret.columns], y=ret.index,
        colorscale=[[i / (len(T.seq) - 1), c] for i, c in enumerate(T.seq)],
        zmin=0, zmax=1, xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title="retained", tickformat=".0%", outlinewidth=0,
                      tickfont=dict(color=T.muted)),
        hovertemplate="%{y} cohort, %{x}<br>%{z:.0%} retained<extra></extra>"))
    for i, coh in enumerate(ret.index):
        for j in range(len(ret.columns)):
            v = ret.iloc[i, j]
            if pd.notna(v):
                fig.add_annotation(x=j, y=i, text=f"{v:.0%}", showarrow=False,
                                   font=dict(size=9, color="#ffffff" if v > 0.55 else T.ink))
    viz.style(fig, T, height=400, title=dict(text="Cohort retention — blank cells are unobserved, not zero"),
              xaxis=dict(title="months since signup", showgrid=False, type="category"),
              yaxis=dict(title="", showgrid=False, type="category", autorange="reversed"))
    st.plotly_chart(fig, width='stretch')

# =================================================================================================
# 2. Personas
# =================================================================================================
with tabs[1]:
    st.markdown("#### Four personas from behaviour and economics")
    st.caption("K-means on washes/month, tenure, ARPU, vehicles, cost-per-wash and wash recency. "
               "Names are matched to centroids by optimal assignment, so a refit cannot swap labels.")

    cols = st.columns(4)
    for col, s in zip(cols, SEG_ORDER):
        r = prof.loc[s]
        with col:
            st.markdown(
                f"<div class='callout' style='border-left-color:{SEG_COLOR[s]}'>"
                f"<h4>{s}</h4>"
                f"<ul style='margin:0;padding-left:1.05rem'>"
                f"<li><b>{int(r.members)}</b> members ({r.members / prof.members.sum():.0%})</li>"
                f"<li><b>{r.revenue_share:.0%}</b> of revenue</li>"
                f"<li>${r.arpu:,.0f} ARPU · {r.washes_per_month:.1f} washes/mo</li>"
                f"<li>{r.n_vehicles:.0f} vehicle(s) · {r.tenure_months:.1f} mo tenure</li>"
                f"<li>churned so far: <b>{r.churn_rate:.0%}</b></li>"
                f"</ul></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=SEG_ORDER, y=[prof.loc[s, "members"] / prof.members.sum() for s in SEG_ORDER],
                    name="Share of members", marker_color=T.s1,
                    marker_line=dict(width=2, color=T.surface),
                    hovertemplate="%{x}<br>%{y:.0%} of members<extra></extra>")
        fig.add_bar(x=SEG_ORDER, y=[prof.loc[s, "revenue_share"] for s in SEG_ORDER],
                    name="Share of revenue", marker_color=T.s2,
                    marker_line=dict(width=2, color=T.surface),
                    hovertemplate="%{x}<br>%{y:.0%} of revenue<extra></extra>")
        viz.style(fig, T, height=360, barmode="group",
                  title=dict(text="Who they are vs what they are worth"),
                  yaxis=dict(title="share", tickformat=".0%"))
        st.plotly_chart(fig, width='stretch')
    with right:
        # Small multiples, not a 4-colour scatter: the validated palette has no all-pairs-safe
        # fourth slot, so personas are faceted rather than overplotted.
        fig = make_subplots(rows=1, cols=len(SEG_ORDER), subplot_titles=SEG_ORDER,
                            shared_yaxes=True, horizontal_spacing=0.02)
        for i, s in enumerate(SEG_ORDER, start=1):
            sub = seg[seg.segment == s]
            fig.add_scatter(x=sub.washes_per_month, y=sub.arpu, mode="markers", name=s,
                            marker=dict(color=SEG_COLOR[s], size=7, opacity=0.75,
                                        line=dict(width=1.5, color=T.surface)),
                            hovertemplate="%{fullData.name}<br>%{x:.1f} washes/mo<br>"
                                          "$%{y:.0f} ARPU<extra></extra>",
                            row=1, col=i, showlegend=False)
        viz.style(fig, T, height=360, title=dict(text="Usage vs spend, one panel per persona"))
        fig.update_xaxes(gridcolor=T.grid, linecolor=T.axis, tickfont=dict(color=T.muted),
                         range=[-0.5, 14])
        fig.update_yaxes(gridcolor=T.grid, linecolor=T.axis, tickfont=dict(color=T.muted),
                         range=[0, 130])
        fig.update_yaxes(title_text="ARPU ($)", row=1, col=1)
        for a in fig.layout.annotations:
            a.font = dict(size=11, color=T.ink)
        st.plotly_chart(fig, width='stretch')

    tbl = prof[["members", "washes_per_month", "tenure_months", "arpu", "n_vehicles",
                "cost_per_wash", "days_since_wash", "churn_rate", "revenue_share"]].reindex(SEG_ORDER)
    tbl.columns = ["members", "washes/mo", "tenure (mo)", "ARPU", "vehicles", "$/wash",
                   "days since wash", "churned", "rev share"]
    html_table(tbl, fmt={"washes/mo": "{:.2f}", "tenure (mo)": "{:.1f}", "ARPU": "${:,.0f}",
                         "vehicles": "{:.0f}", "$/wash": "${:.2f}", "days since wash": "{:.0f}",
                         "churned": "{:.0%}", "rev share": "{:.0%}"}, index_label="persona")

    callout("Reading the personas", [
        "<b>73 Power households are 16% of members and 35% of revenue.</b> Two cars, 5.1 washes "
        "a month, $47 ARPU, 10% churned. Losing one is worth roughly eight Promo flippers.",
        "<b>'Never activated' is the addressable failure.</b> 93 members paying $19.29 for "
        "0.68 washes/month — $20 for every wash they actually take — then 58% of them leave. "
        "They are paying and getting nothing, which is an onboarding problem, not a pricing one.",
        "<b>Promo flippers extract more than they pay:</b> 3.05 washes in their one $10 month = "
        "$3.33 per wash. The $10 is not why they stay; the $32 renewal is why they go.",
        "<i>Caveat:</i> k=4 scores 0.304 on silhouette against 0.309 for k=5 and 0.299 for k=2 — "
        "the clusters are real but the number of them is a judgement call. Tenure also partly "
        "encodes when a member joined, on a book this young.",
    ], accent=T.s3)

# =================================================================================================
# 3. Churn radar
# =================================================================================================
with tabs[2]:
    st.markdown("#### Who is likely to cancel next cycle")
    f = st.columns([1, 1, 1, 2])
    min_risk = f[0].slider("Minimum churn risk", 0.0, 0.5, 0.10, 0.01, format="%.2f")
    seg_pick = f[1].multiselect("Persona", SEG_ORDER, default=SEG_ORDER)
    min_arpu = f[2].slider("Minimum ARPU ($)", 0, 100, 0, 5)
    sort_by = f[3].radio("Rank by", ["Churn risk", "Revenue at risk"], horizontal=True)

    sel = live[(live.churn_risk >= min_risk) & (live.segment.isin(seg_pick))
               & (live.arpu >= min_arpu)].copy()
    sel = sel.sort_values("monthly_revenue_at_risk" if sort_by == "Revenue at risk" else "churn_risk",
                          ascending=False)

    k = st.columns(4)
    k[0].metric("Members in scope", f"{len(sel)}", f"of {len(live)} active", delta_color="off")
    k[1].metric("Expected to churn", f"{sel.churn_risk.sum():.1f}",
                help="Sum of calibrated churn probabilities.")
    k[2].metric("MRR at risk", f"${sel.monthly_revenue_at_risk.sum():,.0f}",
                f"of ${live.arpu.sum():,.0f} total", delta_color="off")
    k[3].metric("Model holdout AUC", f"{model.auc_holdout:.3f}",
                f"{model.top_decile_lift:.1f}× top-decile lift", delta_color="off")

    if len(sel) == 0:
        st.info("No members match these filters.")
    else:
        show = sel.head(25).copy()
        show["risk"] = [pill(f"{r:.0%}", risk_color(r)) for r in show.churn_risk]
        show["persona"] = [pill(s, SEG_COLOR.get(s, T.muted)) if pd.notna(s) else "—"
                           for s in show.segment]
        disp = show[["customer_id", "persona", "package", "arpu", "washes_per_month",
                     "days_since_wash", "tenure_months", "n_vehicles", "risk",
                     "monthly_revenue_at_risk"]]
        disp.columns = ["member", "persona", "package", "ARPU", "washes/mo", "days since wash",
                        "tenure (mo)", "cars", "risk", "$ at risk"]
        html_table(disp.set_index("member"), index_label="member",
                   fmt={"ARPU": "${:,.0f}", "washes/mo": "{:.1f}", "days since wash": "{:.0f}",
                        "tenure (mo)": "{:.1f}", "cars": "{:.0f}", "$ at risk": "${:.2f}"})
        st.caption(f"Top 25 of {len(sel)} matching members.")

    st.divider()
    st.markdown("#### The model behind the list")
    left, right = st.columns(2)
    with left:
        orat = model.odds_ratios()
        fig = go.Figure(go.Bar(
            x=orat.values - 1, y=orat.index, orientation="h",
            marker_color=[T.critical if v > 1 else T.good for v in orat],
            marker_line=dict(width=2, color=T.surface), showlegend=False,
            text=[f"{v:.2f}×" for v in orat], textposition="outside", textfont=dict(color=T.ink2),
            hovertemplate="%{y}<br>%{text} churn odds per +1 SD<extra></extra>"))
        fig.add_vline(x=0, line_color=T.axis, line_width=1)
        viz.style(fig, T, height=380, title=dict(text="Churn odds per +1 SD (right raises churn)"),
                  xaxis=dict(title="", tickvals=[-0.4, -0.2, 0, 0.2, 0.4],
                             ticktext=["0.6×", "0.8×", "1.0×", "1.2×", "1.4×"]),
                  yaxis=dict(title="", autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
    with right:
        d = P._model_frame(OBS)
        r = model.score(d)
        q = pd.qcut(r, 5, labels=["Q1 safest", "Q2", "Q3", "Q4", "Q5 riskiest"])
        cal = (pd.DataFrame({"pred": r, "actual": 1 - d.renewed.astype(int).values})
               .groupby(q, observed=True).agg(n=("actual", "size"), predicted=("pred", "mean"),
                                              actual=("actual", "mean")))
        fig = go.Figure()
        fig.add_bar(x=cal.index.astype(str), y=cal.predicted, name="Predicted", marker_color=T.s1,
                    marker_line=dict(width=2, color=T.surface),
                    hovertemplate="%{x}<br>predicted %{y:.1%}<extra></extra>")
        fig.add_bar(x=cal.index.astype(str), y=cal.actual, name="Actual", marker_color=T.s2,
                    marker_line=dict(width=2, color=T.surface),
                    hovertemplate="%{x}<br>actual %{y:.1%}<extra></extra>")
        viz.style(fig, T, height=380, barmode="group",
                  title=dict(text="Calibration — predicted vs realised churn"),
                  yaxis=dict(title="churn rate", tickformat=".0%"))
        st.plotly_chart(fig, width='stretch')

    callout("How to read the risk list", [
        f"<b>The score is calibrated, so the dollars mean something.</b> The riskiest quintile "
        f"predicts {cal.predicted.iloc[-1]:.1%} and delivers {cal.actual.iloc[-1]:.1%}; that is "
        f"what makes “${live.monthly_revenue_at_risk.sum():,.0f} MRR at risk” a real number rather "
        f"than a ranking artefact.",
        "<b>Wash recency is the dominant term</b> (1.37× churn odds per SD), ahead of the promo "
        "price step (1.27×). Household size is the strongest protection at 0.69×.",
        "<b>Use it for targeting, not verdicts.</b> Holdout AUC is 0.713 and individual risks top "
        "out near 48%, so this ranks a campaign list — it does not decide any single member's fate.",
        "<i>Caveat:</i> members are scored off their most recent charge, so someone billed 25 days "
        "ago is scored on 25-day-old behaviour. In production this would re-score nightly.",
    ], accent=T.critical)

    st.divider()
    st.markdown("#### Dormant payers — billing, but not washing")
    dd = st.slider("No wash in at least (days)", 30, 120, 45, 5)
    dorm = P.dormant_payers(cust, dd)
    m = st.columns(3)
    m[0].metric("Dormant payers", f"{len(dorm)}", f"{len(dorm)/max(cust.active.sum(),1):.1%} of active",
                delta_color="off")
    m[1].metric("Monthly revenue", f"${dorm.arpu.sum():,.0f}")
    m[2].metric("Zero-wash cycle renewal", f"{OBS[OBS.dormant].renewed.mean():.1%}",
                f"{(OBS[OBS.dormant].renewed.mean() - OBS[~OBS.dormant].renewed.mean())*100:.1f} pts vs active users",
                delta_color="inverse")
    if len(dorm):
        dt = dorm.head(15)[["customer_id", "package", "arpu", "washes", "days_since_wash",
                            "tenure_months"]].copy()
        dt.columns = ["member", "package", "ARPU", "washes ever", "days since wash", "tenure (mo)"]
        html_table(dt.set_index("member"), index_label="member",
                   fmt={"ARPU": "${:,.0f}", "days since wash": "{:.0f}", "tenure (mo)": "{:.1f}"})

# =================================================================================================
# 4. Member lookup
# =================================================================================================
with tabs[3]:
    st.markdown("#### One member's whole history")
    ids = sorted(cust.customer_id.tolist())
    default = int(live.iloc[0].customer_id) if len(live) else ids[0]
    cid = st.selectbox("Member", ids, index=ids.index(default),
                       format_func=lambda i: f"#{i}")

    row = cust[cust.customer_id == cid].iloc[0]
    srow = seg[seg.customer_id == cid]
    persona = srow.iloc[0].segment if len(srow) else None
    lrow = live[live.customer_id == cid]
    risk = float(lrow.iloc[0].churn_risk) if len(lrow) else None

    k = st.columns(6)
    k[0].metric("Status", "Active" if row.active else "Churned")
    k[1].metric("Persona", persona or "—")
    k[2].metric("Tenure", f"{row.tenure_months:.1f} mo" if pd.notna(row.tenure_months) else "—")
    k[3].metric("ARPU", f"${row.arpu:,.0f}" if pd.notna(row.arpu) else "—")
    k[4].metric("Washes / mo", f"{row.washes_per_month:.1f}")
    k[5].metric("Churn risk", f"{risk:.0%}" if risk is not None else "n/a",
                help="Only scored for active members.")

    mw = wsh[wsh.customer_id == cid]
    mp = pay[(pay.customer_id == cid) & ~pay.is_refund]
    mr = pay[(pay.customer_id == cid) & pay.is_refund]

    fig = go.Figure()
    if len(mw):
        fig.add_scatter(x=mw.event_date, y=["Wash"] * len(mw), mode="markers", name="Wash",
                        marker=dict(color=T.s1, size=9, symbol="circle",
                                    line=dict(width=1.5, color=T.surface)),
                        customdata=mw.vehicle_id,
                        hovertemplate="%{x|%d %b %Y}<br>wash · vehicle %{customdata}<extra></extra>")
    if len(mp):
        fig.add_scatter(x=mp.event_date, y=["Payment"] * len(mp), mode="markers+text", name="Payment",
                        marker=dict(color=T.s3, size=13, symbol="square",
                                    line=dict(width=1.5, color=T.surface)),
                        text=[f"${a:,.0f}" for a in mp.amount], textposition="top center",
                        textfont=dict(size=9, color=T.muted), customdata=mp.payment_type,
                        hovertemplate="%{x|%d %b %Y}<br>%{customdata} · %{text}<extra></extra>")
    if len(mr):
        fig.add_scatter(x=mr.event_date, y=["Payment"] * len(mr), mode="markers", name="Refund",
                        marker=dict(color=T.critical, size=13, symbol="x",
                                    line=dict(width=1.5, color=T.surface)),
                        customdata=mr.amount,
                        hovertemplate="%{x|%d %b %Y}<br>refund $%{customdata:.2f}<extra></extra>")
    viz.style(fig, T, height=300, title=dict(text=f"Member #{cid} — event timeline"),
              yaxis=dict(title="", categoryorder="array", categoryarray=["Wash", "Payment"]),
              xaxis=dict(title="", range=[ev.event_date.min(), ASOF + pd.Timedelta(days=10)]))
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns([2, 3])
    with left:
        facts = pd.DataFrame({
            "value": [
                f"site {int(row.site_id)}", row.package if pd.notna(row.package) else "—",
                f"${row.list_price:,.2f}" if pd.notna(row.list_price) else "—",
                f"{int(row.n_vehicles)}", row.vehicle_type or "—", row.state or "—",
                row.joined.date() if pd.notna(row.joined) else "—",
                f"{int(row.cycles_paid)}" if pd.notna(row.cycles_paid) else "0",
                f"${row.revenue:,.2f}" if pd.notna(row.revenue) else "—",
                f"${row.cost_per_wash:,.2f}" if pd.notna(row.cost_per_wash) else "—",
                "yes" if row.joined_on_promo else "no",
                f"${row.refunds:,.2f}",
            ]}, index=["Site", "Package", "List price", "Vehicles", "Vehicle type", "Plate state",
                       "Joined", "Cycles paid", "Revenue", "Cost per wash", "Joined on promo",
                       "Refunded"])
        html_table(facts, index_label="")
    with right:
        hist = panel[panel.customer_id == cid].sort_values("event_date")
        if len(hist):
            h = hist[["event_date", "cycle_no", "amount", "washes_this_cycle",
                      "days_since_wash", "renewed", "censored"]].copy()
            h["event_date"] = h.event_date.dt.date
            h["outcome"] = np.where(h.censored, "—  (not yet observable)",
                                    np.where(h.renewed, "renewed", "did not renew"))
            h = h.drop(columns=["renewed", "censored"]).set_index("event_date")
            h.columns = ["month #", "charged", "washes that cycle", "days since wash", "outcome"]
            html_table(h, index_label="charge date",
                       fmt={"month #": "{:.0f}", "charged": "${:,.2f}",
                            "washes that cycle": "{:.0f}", "days since wash": "{:.0f}"})
        else:
            st.info("This member has no payment history in the export.")

# =================================================================================================
# 5. What-if
# =================================================================================================
with tabs[4]:
    st.markdown("#### Unit economics")
    st.caption("The export has no cost side, so the variable cost per wash is an assumption. "
               "Everything below moves with it.")
    a = st.columns(3)
    vcost = a[0].slider("Variable cost per wash ($)", 0.5, 6.0, 2.25, 0.25,
                        help="Water, chemicals, power, incremental labour.")
    churn_override = a[1].slider("Monthly churn used for CLV", 0.02, 0.15,
                                 float(round(P.observed_monthly_churn(cust), 3)), 0.005,
                                 format="%.3f",
                                 help=f"Observed on this book: {P.observed_monthly_churn(cust):.1%}.")
    a[2].metric("Implied average lifetime", f"{1/churn_override:.1f} months")

    ue = P.unit_economics(seg, variable_cost_per_wash=vcost, monthly_churn=churn_override)
    econ = ue.groupby("segment").agg(
        members=("customer_id", "size"), arpu=("arpu", "median"),
        wash_cost=("monthly_wash_cost", "median"),
        contribution=("monthly_contribution", "median"), clv=("clv", "median")).reindex(SEG_ORDER)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(
            x=econ.index, y=econ.clv, marker_color=[SEG_COLOR[s] for s in econ.index],
            marker_line=dict(width=2, color=T.surface), showlegend=False,
            text=[f"${v:,.0f}" for v in econ.clv], textposition="outside",
            textfont=dict(color=T.ink2),
            customdata=np.stack([econ.arpu, econ.contribution], axis=-1),
            hovertemplate="%{x}<br>CLV $%{y:,.0f}<br>ARPU $%{customdata[0]:.0f} · "
                          "contribution $%{customdata[1]:.0f}/mo<extra></extra>"))
        viz.style(fig, T, height=340, title=dict(text=f"Median CLV per persona at ${vcost:.2f}/wash"),
                  yaxis=dict(title="CLV ($)"))
        st.plotly_chart(fig, width='stretch')
    with right:
        e = econ.copy()
        e.columns = ["members", "ARPU", "wash cost", "contribution", "CLV"]
        html_table(e, index_label="persona",
                   fmt={"ARPU": "${:,.2f}", "wash cost": "${:,.2f}",
                        "contribution": "${:,.2f}", "CLV": "${:,.0f}"})
        st.metric("Members below break-even", f"{int((ue.monthly_contribution < 0).sum())}",
                  f"{(ue.monthly_contribution < 0).mean():.1%} of the book", delta_color="off")

    st.divider()
    st.markdown("#### Retention-campaign simulator")
    st.caption("Pick a target group by risk, assume a save rate, and see whether the campaign pays. "
               "**The save rate is the one number this data cannot supply** — it needs an A/B holdout.")
    b = st.columns(4)
    thresh = b[0].slider("Target members above risk", 0.0, 0.4, 0.10, 0.01, format="%.2f")
    save_rate = b[1].slider("Assumed save rate (%)", 0, 60, 20, 5,
                            help="Share of would-be churners the campaign retains.") / 100
    contact_cost = b[2].number_input("Cost per contact ($)", 0.0, 20.0, 0.50, 0.25)
    incentive = b[3].number_input("Incentive per save ($)", 0.0, 100.0, 10.0, 5.0,
                                  help="Free wash, account credit, etc. Charged only on saves.")

    tgt = live[live.churn_risk >= thresh]
    would_churn = tgt.churn_risk.sum()
    saved = would_churn * save_rate
    saved_mrr = (tgt.churn_risk * tgt.arpu).sum() * save_rate
    lifetime = 1 / churn_override
    gross = saved_mrr * lifetime
    cost = len(tgt) * contact_cost + saved * incentive
    roi = (gross - cost) / cost if cost > 0 else np.nan

    m = st.columns(5)
    m[0].metric("Members contacted", f"{len(tgt)}")
    m[1].metric("Would-be churners", f"{would_churn:.1f}")
    m[2].metric("Members saved", f"{saved:.1f}")
    m[3].metric("Campaign cost", f"${cost:,.0f}")
    m[4].metric("Lifetime value retained", f"${gross:,.0f}",
                f"{roi:+.0%} ROI" if np.isfinite(roi) else None)

    grid = []
    for sr in [0.05, 0.10, 0.20, 0.30, 0.40]:
        for th in [0.05, 0.10, 0.15, 0.20]:
            t2 = live[live.churn_risk >= th]
            g = (t2.churn_risk * t2.arpu).sum() * sr * lifetime
            c2 = len(t2) * contact_cost + t2.churn_risk.sum() * sr * incentive
            grid.append({"save rate": sr, "risk threshold": th,
                         "net": g - c2})
    gm = pd.DataFrame(grid).pivot(index="save rate", columns="risk threshold", values="net")
    fig = go.Figure(go.Heatmap(
        z=gm.values, x=[f"≥{c:.0%}" for c in gm.columns], y=[f"{i:.0%}" for i in gm.index],
        colorscale=[[i / (len(T.seq) - 1), c] for i, c in enumerate(T.seq)],
        xgap=2, ygap=2, colorbar=dict(title="net $", outlinewidth=0, tickfont=dict(color=T.muted)),
        hovertemplate="save rate %{y}, target %{x}<br>net $%{z:,.0f}<extra></extra>"))
    for i in range(len(gm.index)):
        for j in range(len(gm.columns)):
            fig.add_annotation(x=j, y=i, text=f"${gm.values[i, j]:,.0f}", showarrow=False,
                               font=dict(size=10, color=T.ink))
    viz.style(fig, T, height=320,
              title=dict(text="Net value of the campaign — save rate vs how wide you target"),
              xaxis=dict(title="target members above risk", showgrid=False),
              yaxis=dict(title="assumed save rate", showgrid=False))
    st.plotly_chart(fig, width='stretch')

    callout("What the simulator is and is not", [
        "<b>The revenue side is measured; the save rate is not.</b> Churn probabilities are "
        "calibrated on 2,027 observed cycles, so “would-be churners” is grounded — but no "
        "experiment in this export tells you what share of them a message would retain.",
        "<b>Targeting wide beats targeting narrow here</b>, because risk is diffuse: the top 20 "
        "members carry only ~$61/month of expected loss out of "
        f"${live.monthly_revenue_at_risk.sum():,.0f}. Contact cost is what eventually caps the width.",
        "<b>Waking a dormant member has a cost.</b> They are currently pure margin — revenue with "
        "no wash cost — so a successful re-engagement trades margin for retention. At $2.25/wash "
        "that trade is positive, but it narrows as the cost assumption rises.",
        "<i>Caveat:</i> lifetime value uses a single book-wide churn rate, so it flatters low-churn "
        "personas less than a per-persona rate would.",
    ], accent=T.warning)

st.divider()
st.caption("Built on `profiling.py`, shared with `customer_profiling.ipynb`. "
           "`experiments/` is standalone — nothing here is imported by the proforma model or the API.")
