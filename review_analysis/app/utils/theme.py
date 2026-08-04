"""Shared visual shell for the review-analytics app.

This module is the whole reason both pages look like the product dashboard
rather than like default Streamlit: it hides Streamlit's own header/sidebar
chrome, paints the fixed top bar + left icon rail, and provides the card,
mini-chart and table primitives the pages compose.

Everything here is presentation-only — no data access, no math. Charts that
live *inside* a KPI card are hand-built HTML/SVG rather than Plotly on
purpose: a card holds one markdown blob, and nine embedded Plotly figures on
a landing page cost far more than nine <div>s for marks this small.

Streamlit-specific note: per-element CSS is attached via container keys.
`st.container(key="foo")` renders a div carrying the class `st-key-foo`, so
the stylesheet targets `[class*="st-key-qtile_"]` etc. That is the supported
hook (Streamlit >= 1.48) and is what makes the transparent full-card click
target possible.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import streamlit as st

# ---------------------------------------------------------------------------
# Palette — sampled from the product dashboard being matched
# ---------------------------------------------------------------------------
APP_BG = "#eaeef7"
CARD_BG = "#ffffff"
CARD_BORDER = "#e8ebf2"
INK = "#2b3648"          # headings / big numbers
INK_SOFT = "#48536a"     # body text
MUTED = "#8d97ab"        # labels, captions
LINE = "#e6e9f1"

UP = "#12b76a"           # green delta
DOWN = "#f04438"         # red delta
BLUE = "#4a90d9"         # primary bar series
ORANGE = "#f2a54a"       # secondary bar series
ACCENT = "#2f6fed"       # links, active nav, tab underline

# Mini-chart segment colors, same family as the reference tiles
MINT = "#bdf0dc"
CREAM = "#fbfbe4"
PURPLE = "#9b8fe0"
YELLOW = "#f7f59a"
CORAL = "#f2825f"
SLATE = "#3a4250"

# Sentiment colors used consistently everywhere (tiles, charts, review cards)
POS_COLOR = "#12b76a"
NEU_COLOR = "#c7cede"
NEG_COLOR = "#f2825f"

BRAND = "Quivio Metrix"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
def _css() -> str:
    return f"""
<style>
  /* Inter matches the product UI; falls back to the system stack offline. */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* --- kill default Streamlit chrome so our own shell can take over --- */
  header[data-testid="stHeader"] {{ display: none; }}
  section[data-testid="stSidebar"] {{ display: none; }}
  div[data-testid="stSidebarCollapsedControl"] {{ display: none; }}
  #MainMenu, footer {{ visibility: hidden; }}

  .stApp {{ background: {APP_BG}; }}
  /* Fill the window. Streamlit caps the block container and centres it, which
     on a wide monitor parks the whole grid in the middle with dead space on
     both sides; `none` lets the tiles span the viewport like the reference. */
  .stMainBlockContainer, div[data-testid="stMainBlockContainer"] {{
      padding: 96px 44px 48px 122px; max-width: none; width: 100%;
  }}
  html, body, .stApp {{
      font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
      color: {INK_SOFT};
  }}

  /* --- fixed top bar --- */
  .q-topbar {{
      position: fixed; top: 0; left: 0; right: 0; height: 74px; z-index: 999;
      background: #ffffff; display: flex; align-items: center;
      padding: 0 28px; border-bottom: 1px solid {LINE};
  }}
  .q-logo {{ display: flex; align-items: center; gap: 9px; }}
  .q-logo-mark {{
      width: 34px; height: 34px; border-radius: 50%;
      background: {ACCENT}; color: #fff; font-weight: 700; font-size: 20px;
      display: flex; align-items: center; justify-content: center;
  }}
  .q-logo-text {{ font-size: 25px; font-weight: 700; color: {INK}; letter-spacing: -.02em; }}
  .q-logo-text span {{ color: {ACCENT}; }}
  .q-topbar-right {{ margin-left: auto; display: flex; align-items: center; gap: 20px; }}
  .q-avatar {{
      width: 34px; height: 34px; border-radius: 50%; background: {ACCENT};
      color: #fff; font-size: 13px; font-weight: 600;
      display: flex; align-items: center; justify-content: center;
  }}

  /* --- fixed left icon rail --- */
  .q-rail {{
      position: fixed; top: 74px; bottom: 0; left: 0; width: 94px; z-index: 998;
      background: #ffffff; display: flex; flex-direction: column;
      align-items: center; padding-top: 22px; gap: 12px; border-right: 1px solid {LINE};
  }}
  .q-rail-item {{
      width: 46px; height: 46px; border-radius: 11px; display: flex;
      align-items: center; justify-content: center; color: #7c8698;
      border: 1.5px solid transparent;
  }}
  .q-rail-item.active {{ color: {ACCENT}; border-color: {ACCENT}; }}
  .q-rail-spacer {{ flex: 1; }}

  /* --- page heading + tab strip --- */
  .q-title {{ font-size: 33px; font-weight: 700; color: {INK}; margin: 0 0 4px; letter-spacing: -.02em; }}
  .q-subtitle {{ font-size: 16px; color: {MUTED}; margin: 0 0 18px; }}
  .q-tab {{ font-size: 15px; color: {INK_SOFT}; padding: 0 2px 11px; }}
  .q-tab.add {{ color: {ACCENT}; white-space: nowrap; }}
  div[class*="st-key-qtabbar"] {{ border-bottom: 1px solid {LINE}; margin-bottom: 22px; }}
  div[class*="st-key-qtab_"] button {{
      background: transparent; border: none; border-radius: 0; min-height: 0;
      color: {INK_SOFT}; padding: 0 2px 10px; justify-content: flex-start; width: 100%;
  }}
  div[class*="st-key-qtab_"] button p {{ font-size: 15px; margin: 0; }}
  div[class*="st-key-qtab_"] button:hover {{ color: {ACCENT}; background: transparent; }}
  div[class*="st-key-qtab_on"] button {{
      color: {INK}; border-bottom: 2.5px solid {ACCENT};
  }}
  div[class*="st-key-qtab_on"] button p {{ font-weight: 600; }}

  /* --- KPI cards --- */
  .qcard {{
      background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 14px;
      padding: 20px 22px 18px; height: 236px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
      display: flex; flex-direction: column;
  }}
  .qcard-head {{ display: flex; align-items: flex-start; }}
  .qcard-title {{
      font-size: 19px; font-weight: 600; color: {INK}; line-height: 1.2;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 88%;
  }}
  .qcard-kebab {{ margin-left: auto; color: #9aa3b5; font-size: 17px; line-height: 1; letter-spacing: 1px; }}
  .qcard-delta {{ font-size: 14px; color: {MUTED}; margin-top: 7px; }}
  .qcard-delta b {{ font-weight: 600; }}
  .qcard-body {{ display: flex; align-items: flex-end; flex: 1; gap: 10px; margin-top: 6px; }}
  .qcard-left {{ display: flex; flex-direction: column; justify-content: flex-end; height: 100%; flex: 1; min-width: 0; }}
  .qcard-value {{ font-size: 40px; font-weight: 700; color: {INK}; line-height: 1.05; letter-spacing: -.02em; }}
  .qcard-subs {{ display: flex; gap: 26px; margin-top: 14px; }}
  .qcard-sub-val {{ font-size: 16px; color: {INK}; font-weight: 500; line-height: 1.25; }}
  .qcard-sub-lbl {{ font-size: 13px; color: {MUTED}; line-height: 1.3;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .qcard-right {{ display: flex; align-items: flex-end; justify-content: flex-end; }}

  /* mini charts inside cards */
  .mini-stack {{ width: 96px; border-radius: 2px; overflow: hidden; display: flex; flex-direction: column; }}
  .mini-stack div {{ width: 100%; }}
  .mini-bars {{ display: flex; align-items: flex-end; gap: 5px; height: 120px; }}
  .mini-bars i {{ width: 11px; border-radius: 2px; display: block; }}
  .mini-dots {{ display: flex; flex-direction: column; gap: 26px; padding-right: 6px; }}
  .mini-dots i {{ width: 9px; height: 9px; border-radius: 50%; display: block; }}

  /* Transparent hit-area that makes a whole card clickable. Every wrapper
     Streamlit puts between the container and the <button> (stElementContainer,
     the tooltip target, stButton) is sized to its content, so all of them have
     to be stretched -- otherwise the click target is a ~56x40 patch in the
     card's top-left corner and the tile only "works" if you happen to hit it. */
  div[class*="st-key-qtile_"] {{ position: relative; }}
  div[class*="st-key-qhit_"] {{ position: absolute; inset: 0; z-index: 3; }}
  div[class*="st-key-qhit_"] > div,
  div[class*="st-key-qhit_"] .stElementContainer,
  div[class*="st-key-qhit_"] .stTooltipHoverTarget,
  div[class*="st-key-qhit_"] .stButton {{
      width: 100% !important; height: 100% !important;
  }}
  /* The button is positioned against the inset:0 container directly, because a
     percentage height only resolves if every ancestor has a definite one --
     the wrappers do not, which left the target 657x40 instead of 657x236. */
  div[class*="st-key-qhit_"] button {{
      position: absolute; inset: 0; width: 100% !important; height: 100% !important;
      opacity: 0; border: none; background: transparent; cursor: pointer;
  }}
  div[class*="st-key-qtile_"]:hover .qcard {{
      box-shadow: 0 6px 18px rgba(16,24,40,.12); border-color: #cdd7ee;
      transform: translateY(-1px);
  }}
  .qcard {{ transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; }}
  div[class*="st-key-qtile_"]:hover .qcard-kebab {{ color: {ACCENT}; }}
  div[class*="st-key-qtile_"]:hover .qcard-title::after {{
      content: " →"; color: {ACCENT}; font-weight: 600;
  }}

  /* --- panel card (wraps charts / tables) --- */
  .q-panel-head {{ display: flex; align-items: center; margin-bottom: 2px; }}
  .q-panel-title {{ font-size: 24px; font-weight: 700; color: {INK}; }}
  div[class*="st-key-qpanel_"] {{
      background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 14px;
      padding: 22px 26px 18px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
  }}

  /* --- drill-down table --- */
  .q-th {{ font-size: 14px; color: {MUTED}; font-weight: 500; padding: 6px 0; }}
  /* Sort headers. Zero horizontal padding on both the button and its inner
     <p> so the label's edge lands exactly on the cell's edge below it —
     any padding here shows up as a column that looks misaligned. */
  /* !important because Streamlit's own emotion rule sizes buttons to their
     content: without this the header button is ~83px wide inside a 168px
     column, so right-aligning the label inside it changes nothing. */
  div[class*="st-key-qsort_"] > div, div[class*="st-key-qsortr_"] > div,
  div[class*="st-key-qsort_"] .stElementContainer, div[class*="st-key-qsortr_"] .stElementContainer,
  div[class*="st-key-qsort_"] .stButton, div[class*="st-key-qsortr_"] .stButton {{
      width: 100% !important;
  }}
  div[class*="st-key-qsort_"] button, div[class*="st-key-qsortr_"] button {{
      background: transparent; border: none; color: {MUTED}; font-size: 14px;
      font-weight: 500; padding: 4px 0; width: 100% !important; min-height: 0;
  }}
  div[class*="st-key-qsort_"] button:hover, div[class*="st-key-qsortr_"] button:hover {{
      color: {ACCENT}; background: transparent;
  }}
  div[class*="st-key-qsort_"] button p, div[class*="st-key-qsortr_"] button p {{
      font-size: 14px; margin: 0; padding: 0; width: 100%;
  }}
  /* The label lives inside a flex wrapper *within* the button that centres it,
     so aligning the button alone leaves the text floating mid-column. Both the
     button and that inner div have to agree. */
  /* name column: label left, matching .q-cell.name */
  div[class*="st-key-qsort_"] button {{ justify-content: flex-start; text-align: left; }}
  div[class*="st-key-qsort_"] button > div {{ justify-content: flex-start !important; width: 100%; }}
  div[class*="st-key-qsort_"] button p {{ text-align: left; }}
  /* numeric columns: label right, matching .q-cell's right alignment */
  div[class*="st-key-qsortr_"] button {{ justify-content: flex-end; text-align: right; }}
  div[class*="st-key-qsortr_"] button > div {{ justify-content: flex-end !important; width: 100%; }}
  div[class*="st-key-qsortr_"] button p {{ text-align: right; }}
  div[class*="st-key-qrow_"] {{ border-top: 1px solid {LINE}; padding: 2px 0; }}
  div[class*="st-key-qrow_"]:hover {{ background: #f7f9fd; }}
  /* the expanded parent row, highlighted like the reference drill-down */
  div[class*="st-key-qrowopen_"] {{
      border-top: 1px solid {LINE}; padding: 2px 0; background: #eef3fc;
  }}
  div[class*="st-key-qrowopen_"] .q-cell {{ color: {ACCENT}; font-weight: 600; }}
  /* chevron: same vertical padding as .q-cell so it sits on the row's baseline */
  div[class*="st-key-qopen_"] button {{
      background: transparent; border: none; color: {ACCENT}; font-size: 15px;
      padding: 12px 0; min-height: 0; text-align: left; justify-content: flex-start;
  }}
  div[class*="st-key-qopen_"] button:hover {{ background: transparent; color: {INK}; }}
  .q-cell {{ font-size: 15px; color: {INK_SOFT}; padding: 12px 0; text-align: right; }}
  .q-cell.name {{ text-align: left; color: {ACCENT}; font-weight: 500; }}
  .q-cell.strong {{ color: {INK}; font-weight: 600; }}

  /* --- review cards inside an expanded site --- */
  .q-review {{
      background: #f8fafd; border: 1px solid {LINE}; border-left: 3px solid {NEU_COLOR};
      border-radius: 10px; padding: 13px 16px; margin-bottom: 10px;
  }}
  .q-review.pos {{ border-left-color: {POS_COLOR}; }}
  .q-review.neg {{ border-left-color: {NEG_COLOR}; }}
  .q-review-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }}
  .q-review-name {{ font-weight: 600; color: {INK}; font-size: 15px; }}
  .q-review-stars {{ color: #f0b429; font-size: 14px; letter-spacing: 1px; }}
  .q-review-date {{ color: {MUTED}; font-size: 13px; }}
  .q-chip {{
      font-size: 11.5px; font-weight: 600; padding: 2px 9px; border-radius: 999px;
      color: #ffffff; letter-spacing: .01em;
  }}
  .q-review-text {{ font-size: 14.5px; color: {INK_SOFT}; line-height: 1.5; white-space: pre-wrap; }}
  .q-review-text.empty {{ color: {MUTED}; font-style: italic; }}
  .q-review-owner {{
      margin-top: 9px; padding: 9px 12px; background: #eef3fc; border-radius: 8px;
      font-size: 13.5px; color: {INK_SOFT};
  }}
  .q-review-owner b {{ color: {INK}; }}

  /* --- generic bits --- */
  .q-note {{ font-size: 13px; color: {MUTED}; }}
  .q-pill-label {{ font-size: 13px; color: {MUTED}; margin-bottom: 4px; }}
  div[data-testid="stPopover"] button, div[class*="st-key-qfilter"] button {{
      background: #ffffff; border: 1px solid #d9dfec; border-radius: 9px;
      color: {INK}; font-size: 14.5px; padding: 9px 16px;
  }}
  .stSelectbox div[data-baseweb="select"] > div {{
      background: #ffffff; border-color: #d9dfec; border-radius: 9px;
  }}
</style>
"""


# Left-rail glyphs (inline SVG, stroke-only, matching the reference rail)
_RAIL_ICONS = [
    ("M4 4h6v6H4zM4 14h6v6H4zM14 4h6v16h-6z", False),                       # dashboards
    ("M12 3v18M8 7h6a3 3 0 010 6H9a3 3 0 000 6h7", False),                  # revenue
    ("M3 5h18v14H3zM7 10h4M7 14h8", True),                                  # memberships (active)
    ("M5 20V10M12 20V4M19 20v-7", False),                                   # analytics
    ("M8 3h9v13H8zM5 7v14h9", False),                                       # reports
    ("M4 8h14l-3-3M20 16H6l3 3", False),                                    # transfers
    ("M12 3a9 9 0 100 18 9 9 0 000-18zM12 8a4 4 0 100 8 4 4 0 000-8z", False),  # targets
    ("M13 2L4 14h7l-1 8 9-12h-7z", False),                                  # automations
]


def _rail_html() -> str:
    items = []
    for path, active in _RAIL_ICONS:
        cls = "q-rail-item active" if active else "q-rail-item"
        items.append(
            f'<div class="{cls}"><svg width="23" height="23" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
            f'stroke-linejoin="round"><path d="{path}"/></svg></div>'
        )
    return f'<div class="q-rail">{"".join(items)}<div class="q-rail-spacer"></div></div>'


def _topbar_html(initials: str = "SS") -> str:
    brand_head, brand_tail = BRAND.split(" ", 1)
    return f"""
<div class="q-topbar">
  <div class="q-logo">
    <div class="q-logo-mark">{brand_head[0]}</div>
    <div class="q-logo-text"><span>{brand_head}</span>{brand_tail}</div>
  </div>
  <div class="q-topbar-right">
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#6b7488" stroke-width="1.6"
         stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 01-8 8H7l-4 3V12a8 8 0 018-8h2a8 8 0 018 8z"/></svg>
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#6b7488" stroke-width="1.6"
         stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 10-12 0c0 7-3 8-3 8h18s-3-1-3-8M13.7 21a2 2 0 01-3.4 0"/></svg>
    <div class="q-avatar">{initials}</div>
  </div>
</div>
"""


def setup_page(page_title: str, page_icon: str = "🚗") -> None:
    """Configure the Streamlit page and paint the fixed shell (bar + rail)."""
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    st.markdown(_css(), unsafe_allow_html=True)
    st.markdown(_topbar_html() + _rail_html(), unsafe_allow_html=True)


def page_heading(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="q-title">{title}</div><div class="q-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def tab_strip(tabs: Sequence[tuple[str, bool, str | None]],
              add_label: str = "+  Add Site Group") -> None:
    """Navigation tab strip. `tabs` is [(label, is_active, page_path), ...].

    Real buttons rather than styled text, because the Streamlit page nav in
    the sidebar is hidden by this shell — without these the drill-down pages
    would be unreachable except by tile click.
    """
    with st.container(key="qtabbar"):
        widths = [max(1.0, len(label) * 0.075) for label, _, _ in tabs]
        cols = st.columns(widths + [max(1.6, len(add_label) * 0.075), 6.0])
        for col, (label, active, page) in zip(cols, tabs):
            with col:
                slug = "".join(c for c in label if c.isalnum())
                with st.container(key=f"qtab_{'on' if active else 'off'}_{slug}"):
                    if st.button(label, key=f"tabbtn_{slug}") and page and not active:
                        st.switch_page(page)
        with cols[len(tabs)]:
            st.markdown(f'<div class="q-tab add">{add_label}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Mini charts (pure HTML — see module docstring for why not Plotly)
# ---------------------------------------------------------------------------
# Minimum on-screen height for a non-zero mini-chart segment.
MIN_SEGMENT_PX = 3


def mini_stack(segments: Iterable[tuple[float, str]], height: int = 130) -> str:
    """Vertical stacked bar. `segments` is [(magnitude, color), ...].

    Non-zero segments get a 3px floor so a category that exists but is tiny
    (8 negative reviews against 223 positive) still shows as a sliver rather
    than vanishing into a solid block. The floor is taken out of the
    proportional space, so the bar stays the same total height.
    """
    segs = [(float(s), c) for s, c in segments if s and float(s) > 0]
    if not segs:
        return f'<div class="mini-stack" style="height:{height}px;background:#eef1f6;"></div>'
    total = sum(s for s, _ in segs)
    floor_px = MIN_SEGMENT_PX * len(segs)
    scalable = max(0.0, height - floor_px)
    parts = "".join(
        f'<div style="height:{MIN_SEGMENT_PX + s / total * scalable:.1f}px;background:{c};"></div>'
        for s, c in segs
    )
    return f'<div class="mini-stack" style="height:{height}px;">{parts}</div>'


def mini_bars(values: Sequence[float], color: str = BLUE, height: int = 120,
              colors: Sequence[str] | None = None, baseline: float | None = None) -> str:
    """Small bar series, scaled to the largest value (same 3px floor).

    `baseline` zooms the scale: star ratings all sit between 4.4 and 4.9, so
    drawn from zero they are five identical full-height bars saying nothing.
    Passing baseline=4.0 makes the month-to-month movement legible. Only use
    it where the reader can see the scale is truncated (a rating tile), never
    for counts.
    """
    vals = [float(v) for v in values]
    if not vals:
        return f'<div class="mini-bars" style="height:{height}px;"></div>'
    base = float(baseline) if baseline is not None else 0.0
    top = max(vals)
    span = (top - base) or 1.0
    bars = "".join(
        f'<i style="height:{max(float(MIN_SEGMENT_PX), (v - base) / span * height):.1f}px;'
        f'background:{(colors[i] if colors else color)};"></i>'
        for i, v in enumerate(vals)
    )
    return f'<div class="mini-bars" style="height:{height}px;">{bars}</div>'


def mini_dots(colors: Sequence[str]) -> str:
    dots = "".join(f'<i style="background:{c};"></i>' for c in colors)
    return f'<div class="mini-dots">{dots}</div>'


# ---------------------------------------------------------------------------
# KPI card
# ---------------------------------------------------------------------------
def delta_html(pct: float | None, comparison: str = "from last month",
               higher_is_better: bool = True) -> str:
    """The '↑ 2.41% from last month' line, colored by whether it is good news."""
    if pct is None:
        return f'<div class="qcard-delta">--&nbsp; {comparison}</div>'
    good = (pct >= 0) if higher_is_better else (pct <= 0)
    color = UP if good else DOWN
    arrow = "↑" if pct >= 0 else "↓"
    return (f'<div class="qcard-delta"><b style="color:{color}">{arrow} {abs(pct):.2f}%</b>'
            f'&nbsp; {comparison}</div>')


def kpi_card_html(title: str, value: str, delta: str = "", chart: str = "",
                  subs: Sequence[tuple[str, str]] = ()) -> str:
    """One dashboard tile. `subs` is [(value, label), ...] (max 2 render well)."""
    sub_html = "".join(
        f'<div><div class="qcard-sub-val">{v}</div><div class="qcard-sub-lbl">{l}</div></div>'
        for v, l in subs
    )
    return f"""
<div class="qcard">
  <div class="qcard-head"><div class="qcard-title">{title}</div><div class="qcard-kebab">⋮</div></div>
  {delta}
  <div class="qcard-body">
    <div class="qcard-left">
      <div class="qcard-value">{value}</div>
      <div class="qcard-subs">{sub_html}</div>
    </div>
    <div class="qcard-right">{chart}</div>
  </div>
</div>
"""


def clickable_card(key: str, html: str, on_click: Callable[[], None] | None = None,
                   help_text: str = "Open drill-down") -> None:
    """Render a KPI card with an invisible full-bleed button on top of it."""
    with st.container(key=f"qtile_{key}"):
        st.markdown(html, unsafe_allow_html=True)
        if on_click is not None:
            with st.container(key=f"qhit_{key}"):
                if st.button("Open", key=f"qbtn_{key}", help=help_text):
                    on_click()


def panel_start(key: str):
    """Context manager for a white panel that wraps charts/tables."""
    return st.container(key=f"qpanel_{key}")


def stars(rating: float, size: int = 14) -> str:
    filled = int(round(rating)) if rating == rating else 0
    filled = max(0, min(5, filled))
    return (f'<span class="q-review-stars" style="font-size:{size}px">'
            f'{"★" * filled}{"☆" * (5 - filled)}</span>')


def chip(label: str, color: str) -> str:
    return f'<span class="q-chip" style="background:{color}">{label}</span>'


def fmt_compact(n: float) -> str:
    """1194 -> '1.19k' — the reference tiles' number format."""
    if n is None or n != n:
        return "--"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.2f}k"
    return f"{n:,.0f}"


def pct_change(current: float | None, prior: float | None) -> float | None:
    """Percent change, with the divide-by-zero / missing cases returning None."""
    if current is None or prior is None or prior in (0, None) or prior != prior or current != current:
        return None
    return (current - prior) / abs(prior) * 100
