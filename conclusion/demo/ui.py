"""
Shared UI chrome for the conclusions app — palette, Plotly layout, CSS, table + callout helpers.

Every section (`section_*.py`) imports from here so the whole app reads as one system. Kept
Streamlit-aware on purpose; the *analysis* modules (`tunnel_data`, `proforma_data`) stay
Streamlit-free so the notebook can import them.

`init_page()` must be called once, first, by app.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- chart system: the validated palette ---------------------------------------------------------
# The repo pins `base = "dark"` in .streamlit/config.toml for the main proforma app, and this app is
# launched from the same working directory — so it inherits whatever that says. Rather than fight it
# with a competing config, both modes are selected here from the same ramps: the dark column is the
# same hues stepped for the dark surface, not an automatic flip of the light one.
DARK = st.get_option("theme.base") == "dark"

if DARK:
    INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#898781"
    AXIS = "#383835"
    GRID, SURFACE, PLANE = "#2c2c2a", "#1a1a19", "#0d0d0d"
    S1, S2, S3 = "#3987e5", "#d95926", "#199e70"
    HEADER_BG, BORDER = "#242422", "rgba(255,255,255,0.10)"
else:
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    AXIS = "#c3c2b7"
    GRID, SURFACE, PLANE = "#e1e0d9", "#fcfcfb", "#f9f9f7"
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
    HEADER_BG, BORDER = "#f0efec", "rgba(11,11,11,0.10)"

# Status steps are mode-invariant by design — all four clear 3:1 on both surfaces, and are always
# paired with a glyph and a label so hue never carries meaning alone. Sections map their own
# vocabulary onto these four roles.
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK2, size=12),
    # text="" matters: with only a font set, Plotly leaves the title undefined and Streamlit
    # renders the literal string "undefined" above the chart.
    title=dict(text="", font=dict(color=INK, size=15)),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
    margin=dict(l=60, r=25, t=55, b=50), hovermode="closest",
)


def style(fig: go.Figure, **kw) -> go.Figure:
    """Apply the house layout, letting the caller override any slot.

    Deep-merges one level so a caller passing `xaxis=dict(tickformat=".0%")` keeps the house
    gridcolour instead of replacing the whole axis dict (and doesn't collide on the keyword).
    """
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in LAYOUT.items()}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    fig.update_layout(**merged)
    return fig


def inject_css() -> None:
    """One CSS block for the whole app; call once from app.py."""
    st.markdown(f"""<style>
    .stApp {{ background: {PLANE}; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.75rem; color: {INK}; }}
    div[data-testid="stMetricLabel"] {{ color: {INK2}; }}
    .kicker {{ color:{MUTED}; font-size:0.82rem; letter-spacing:.06em; text-transform:uppercase; }}
    .callout {{ background:{SURFACE}; border:1px solid {BORDER};
               border-left:4px solid {S1}; border-radius:6px; padding:.85rem 1.1rem; margin:.4rem 0 1rem; }}
    .callout h4 {{ margin:0 0 .35rem; font-size:.95rem; color:{INK}; }}
    .callout p, .callout li {{ color:{INK2}; font-size:.9rem; margin:.2rem 0; }}
    .tblwrap {{ overflow-x:auto; border:1px solid {BORDER}; border-radius:6px;
                background:{SURFACE}; margin:.3rem 0 1rem; }}
    table.tbl {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
    table.tbl th {{ text-align:left; font-weight:600; color:{INK2}; background:{HEADER_BG};
                    padding:.5rem .7rem; border-bottom:1px solid {GRID}; white-space:nowrap; }}
    table.tbl td {{ padding:.42rem .7rem; border-bottom:1px solid {GRID}; color:{INK};
                    white-space:nowrap; }}
    table.tbl td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    table.tbl td.lbl {{ color:{INK2}; font-weight:600; }}
  table.tbl td.wrap {{ white-space:normal; line-height:1.45; min-width:20rem; }}
    table.tbl tbody tr:last-child td {{ border-bottom:none; }}
    .barwrap {{ display:inline-block; width:70px; height:7px; background:{GRID};
                border-radius:4px; overflow:hidden; vertical-align:middle; margin-right:.5rem; }}
    .bar {{ display:block; height:100%; background:{S1}; border-radius:4px; }}
    .barval {{ font-variant-numeric:tabular-nums; }}
    </style>""", unsafe_allow_html=True)


def html_table(df: pd.DataFrame, fmt: dict[str, str] | None = None,
               bars: dict[str, float] | None = None, index_label: str | None = None) -> None:
    """Render a table as plain HTML.

    `st.dataframe` / `st.table` **segfault the Streamlit server on the second script run** in this
    environment (pyarrow 25.0.0 + pandas 3.0.2 + streamlit 1.58.0) — a three-row toy frame is enough
    to reproduce, and the process dies the moment any widget is touched. Since the whole point of
    this app is that the reviewer moves the sliders, every table is rendered as HTML instead. See
    the "Data & method" tab for the note and the proper fix.
    """
    fmt, bars = fmt or {}, bars or {}
    head = "".join(f"<th>{c}</th>" for c in ([index_label] if index_label else []) + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<td class='lbl'>{idx}</td>"] if index_label else []
        for c in df.columns:
            v = r[c]
            if c in bars and pd.notna(v):
                pct = min(max(float(v) / bars[c], 0), 1) * 100
                cells.append(
                    f"<td class='num'><span class='barwrap'>"
                    f"<span class='bar' style='width:{pct:.0f}%'></span></span>"
                    f"<span class='barval'>{float(v)*100:.0f}%</span></td>")
            elif pd.isna(v):
                cells.append("<td class='num'>—</td>")
            elif c in fmt:
                cells.append(f"<td class='num'>{fmt[c].format(v)}</td>")
            elif isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f"<td class='num'>{v:,.0f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f"<div class='tblwrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


def callout(title: str, body_md: str, accent: str = S1) -> None:
    """Insight box: one bullet per point, wrapped freely in the source.

    Bullets are split on INDENTATION, not on a list of known opening phrases. A line sitting at the
    block's outermost indent starts a new bullet; anything indented further is a continuation of it.
    That means a bullet can open with any wording — an earlier version keyed off a fixed set of
    lead-ins ("Reading", "So-what") and silently merged every bullet that opened differently.
    """
    raw = [ln for ln in body_md.strip("\n").split("\n") if ln.strip()]
    if not raw:
        return
    base = min(len(ln) - len(ln.lstrip()) for ln in raw)
    bullets: list[str] = []
    for ln in raw:
        indent = len(ln) - len(ln.lstrip())
        if indent <= base or not bullets:
            bullets.append(ln.strip())
        else:
            bullets[-1] += " " + ln.strip()
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"<div class='callout' style='border-left-color:{accent}'>"
                f"<h4>{title}</h4><ul style='margin:0;padding-left:1.1rem'>{items}</ul></div>",
                unsafe_allow_html=True)

