"""Shared drill-down widgets: table rows and the expanded review list.

Both the insights page (period -> site -> reviews) and the site breakdown
page (site -> reviews) render the same row and the same review list, so the
markup lives here once. `scope_key` is whatever uniquely identifies the open
branch -- a site name, or "2026-01|Big Dan's Rome" -- and namespaces the
widget keys and the "show more" counter for that branch.
"""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from app.utils import theme as T
from app.utils.data_loader import (
    DATE_COL,
    OWNER_RESPONSE_TEXT_COL,
    RATING_COL,
    REVIEWER_NAME_COL,
    TEXT_COL,
)
from app.utils.metrics import LABEL_COL, REVIEW_SORTS, SCORE_COL, sort_reviews

PAGE_SIZE = 15

# Which reviews to show. Sentiment labels plus two non-sentiment lenses the
# dashboard tiles need to be able to hand over (a 1-2 star tile and a
# "still awaiting a reply" tile can't be expressed as a VADER label).
REVIEW_FILTERS = {
    "All": None,
    "Positive": ("label", "positive"),
    "Neutral": ("label", "neutral"),
    "Negative": ("label", "negative"),
    "Rating-only (no text)": ("label", "no_text"),
    "1–2 star": ("rating_low", None),
    "Awaiting owner reply": ("no_response", None),
}

# Session keys holding the lens a tile handed over. They seed every branch's
# controls, so expanding a *different* site keeps the same lens instead of
# silently reverting to "All" -- which made a click on "Negative Reviews" show
# positive reviews as soon as you opened another location.
DEFAULT_FILTER_KEY = "rv_default_filter"
DEFAULT_SORT_KEY = "rv_default_sort"
SEARCH_KEY = "rv_search"


def apply_filter(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Restrict a review frame to one lens from REVIEW_FILTERS."""
    rule = REVIEW_FILTERS.get(choice)
    if rule is None:
        return df
    kind, value = rule
    if kind == "label":
        return df[df[LABEL_COL] == value]
    if kind == "rating_low":
        return df[pd.to_numeric(df[RATING_COL], errors="coerce") <= 2]
    if kind == "no_response":
        return df[df[OWNER_RESPONSE_TEXT_COL].isna()]
    return df


def clear_branch_state() -> None:
    """Reset per-branch state (the "show more" counters) for a new lens.

    A reader arriving with a fresh lens should start at the first page of
    results, not wherever an earlier visit had paged to.
    """
    # The control widgets go too: Streamlit keeps a widget's value across page
    # switches once its key exists, and that stale value would beat the `index=`
    # seeding and override the lens the tile just handed over.
    for key in [k for k in st.session_state
                if k.startswith(("filt_", "n_shown_"))
                or k in {"rv_sort_widget", "rv_filter_widget", "rv_search_widget"}
                or (k.startswith("sort_") and not k.startswith("sortbtn_"))]:
        del st.session_state[key]


_CHIPS = {
    "positive": ("Positive", T.POS_COLOR),
    "negative": ("Negative", T.NEG_COLOR),
    "neutral": ("Neutral", "#98a2b8"),
    "no_text": ("No text", "#b6bece"),
}


def safe_key(text: str) -> str:
    """Widget-key-safe slug (Streamlit keys must not collide across rows)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")


def review_controls(is_open: bool) -> None:
    """The one and only sort / search / filter row, above the table.

    Rendered once per page instead of once per expanded row: repeating it
    under every location restated a filter the reader had already chosen (from
    a tile, or here) every time they opened another site. The widgets write
    straight to the same session keys a dashboard tile hands its lens over in,
    so a tile click pre-selects them and changing them here applies to every
    branch opened afterwards.

    Only drawn when something is actually expanded — controls for a review
    list that isn't on screen are noise.

    The lens lives in plain session keys, and the widgets are keyed separately
    and seeded from them with `index=`. This is not ceremony: Streamlit drops
    the state of any widget it did not render, so when the reader collapsed
    every row (no controls on screen) the lens a tile had handed over was
    discarded, and re-expanding a row rebuilt the widgets at their first
    option — "All" / "most positive" — which is how a negative-reviews tile
    ended up showing five-star reviews.
    """
    if not is_open:
        return

    sort_value = st.session_state.get(DEFAULT_SORT_KEY, list(REVIEW_SORTS)[0])
    filter_value = st.session_state.get(DEFAULT_FILTER_KEY, "All")
    sort_index = list(REVIEW_SORTS).index(sort_value) if sort_value in REVIEW_SORTS else 0
    filter_index = list(REVIEW_FILTERS).index(filter_value) if filter_value in REVIEW_FILTERS else 0

    c1, c2, c3 = st.columns([2.6, 2.6, 2.2])
    with c1:
        chosen_sort = st.selectbox("Sort reviews by", options=list(REVIEW_SORTS),
                                   index=sort_index, key="rv_sort_widget")
    with c2:
        typed = st.text_input("Search text", value=st.session_state.get(SEARCH_KEY, ""),
                              key="rv_search_widget", placeholder="e.g. wait, staff, vacuum")
    with c3:
        chosen_filter = st.selectbox("Show", options=list(REVIEW_FILTERS),
                                     index=filter_index, key="rv_filter_widget")

    # The canonical values, which survive a run that drew no controls at all.
    st.session_state[DEFAULT_SORT_KEY] = chosen_sort
    st.session_state[DEFAULT_FILTER_KEY] = chosen_filter
    st.session_state[SEARCH_KEY] = typed


def cell(value, kind: str = "num", indent: int = 0) -> str:
    """One table cell. `kind`: num | name | strong; `indent` nests child rows."""
    style = f' style="padding-left:{indent}px;"' if indent else ""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f'<div class="q-cell"{style}>--</div>'
    return f'<div class="q-cell {kind}"{style}>{value}</div>'


def sort_header(columns, sort_col_key: str, sort_asc_key: str, chevron_width: float = 0.5):
    """Header row of clickable sort buttons.

    The first column is the row's name (left-aligned); every other column is
    numeric and right-aligned, and its header has to be right-aligned too or
    the label floats left of the figures it labels. The alignment is carried
    by the container-key prefix (`qsort_` vs `qsortr_`), which theme.py
    styles.
    """
    sort_col = st.session_state[sort_col_key]
    ascending = st.session_state[sort_asc_key]
    boxes = st.columns([chevron_width] + [w for _, _, w in columns])
    for i, (box, (key, label, _w)) in enumerate(zip(boxes[1:], columns)):
        prefix = "qsort_" if i == 0 else "qsortr_"
        with box:
            with st.container(key=f"{prefix}{safe_key(key)}"):
                arrow = (" ↑" if ascending else " ↓") if sort_col == key else " ⇅"
                if st.button(label + arrow, key=f"sortbtn_{safe_key(key)}"):
                    if sort_col == key:
                        st.session_state[sort_asc_key] = not ascending
                    else:
                        st.session_state[sort_col_key] = key
                        # Text sorts read naturally A->Z, numbers biggest-first.
                        st.session_state[sort_asc_key] = key in {"site", "label", "period"}
                    st.rerun()


def table_row(row_key: str, values, widths, is_open: bool, on_toggle,
              indent: int = 0, highlight: bool = False) -> None:
    """One table row with a chevron toggle. `values` is [(text, kind), ...]."""
    container_key = ("qrowopen_" if highlight else "qrow_") + safe_key(row_key)
    with st.container(key=container_key):
        boxes = st.columns([0.5] + list(widths))
        with boxes[0]:
            with st.container(key=f"qopen_{safe_key(row_key)}"):
                if st.button("⌄" if is_open else "›", key=f"open_{safe_key(row_key)}"):
                    on_toggle()
                    st.rerun()
        for i, (box, (val, kind)) in enumerate(zip(boxes[1:], values)):
            with box:
                st.markdown(cell(val, kind, indent if i == 0 else 0), unsafe_allow_html=True)


def render_reviews(scope_key: str, reviews: pd.DataFrame, caption: str = "") -> None:
    """The review cards for one expanded branch.

    Takes its sort/search/filter from the page-level controls (see
    `review_controls`) rather than rendering its own copy of them.
    """
    key = safe_key(scope_key)

    sort_choice = st.session_state.get(DEFAULT_SORT_KEY, list(REVIEW_SORTS)[0])
    which = st.session_state.get(DEFAULT_FILTER_KEY, "All")
    keyword = st.session_state.get(SEARCH_KEY, "") or ""

    view = reviews
    if keyword.strip():
        view = view[view[TEXT_COL].fillna("").str.contains(keyword.strip(), case=False, na=False)]
    view = apply_filter(view, which)
    view = sort_reviews(view, sort_choice)

    shown = st.session_state.get(f"n_shown_{key}", PAGE_SIZE)

    cards = []
    for _, r in view.head(shown).iterrows():
        label = r[LABEL_COL]
        chip_label, chip_color = _CHIPS.get(label, _CHIPS["neutral"])
        score = r[SCORE_COL]
        score_txt = f"score {score:+.2f}" if pd.notna(score) else "no text to score"
        rating = int(r[RATING_COL]) if pd.notna(r[RATING_COL]) else 0
        date = r[DATE_COL].strftime("%b %d, %Y") if pd.notna(r[DATE_COL]) else "Unknown date"
        name = html.escape(str(r.get(REVIEWER_NAME_COL) or "Anonymous"))
        text = r.get(TEXT_COL)
        body = (f'<div class="q-review-text">{html.escape(str(text))}</div>'
                if pd.notna(text) and str(text).strip()
                else '<div class="q-review-text empty">Rating-only review — no text submitted.</div>')
        owner = r.get(OWNER_RESPONSE_TEXT_COL)
        owner_html = (
            f'<div class="q-review-owner"><b>Owner replied:</b> {html.escape(str(owner))}</div>'
            if pd.notna(owner) and str(owner).strip() else ""
        )
        css = "pos" if label == "positive" else ("neg" if label == "negative" else "")
        cards.append(
            f'<div class="q-review {css}"><div class="q-review-head">'
            f'<span class="q-review-name">{name}</span>{T.stars(rating)}'
            f'<span class="q-review-date">{date}</span>{T.chip(chip_label, chip_color)}'
            f'<span class="q-review-date">{score_txt}</span></div>{body}{owner_html}</div>'
        )

    if cards:
        st.markdown("".join(cards), unsafe_allow_html=True)
    else:
        st.caption("No reviews match this filter.")

    if len(view) > shown:
        if st.button(f"Show {min(PAGE_SIZE, len(view) - shown)} more", key=f"more_{key}"):
            st.session_state[f"n_shown_{key}"] = shown + PAGE_SIZE
            st.rerun()
