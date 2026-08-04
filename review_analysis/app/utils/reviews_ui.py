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

SENTIMENT_FILTERS = {
    "All": None,
    "Positive": "positive",
    "Neutral": "neutral",
    "Negative": "negative",
    "Rating-only (no text)": "no_text",
}

_CHIPS = {
    "positive": ("Positive", T.POS_COLOR),
    "negative": ("Negative", T.NEG_COLOR),
    "neutral": ("Neutral", "#98a2b8"),
    "no_text": ("No text", "#b6bece"),
}


def safe_key(text: str) -> str:
    """Widget-key-safe slug (Streamlit keys must not collide across rows)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")


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
    """Sort/search/filter controls plus the review cards for one branch."""
    key = safe_key(scope_key)
    if caption:
        st.markdown(f'<div class="q-note" style="margin:6px 0 10px;">{caption}</div>',
                    unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2.4, 2.4, 2.2])
    with c1:
        sort_choice = st.selectbox("Sort reviews by", options=list(REVIEW_SORTS),
                                   index=0, key=f"sort_{key}")
    with c2:
        keyword = st.text_input("Search text", key=f"kw_{key}",
                                placeholder="e.g. wait, staff, vacuum")
    with c3:
        which = st.selectbox("Sentiment", options=list(SENTIMENT_FILTERS), key=f"filt_{key}")

    view = reviews
    if keyword.strip():
        view = view[view[TEXT_COL].fillna("").str.contains(keyword.strip(), case=False, na=False)]
    wanted = SENTIMENT_FILTERS[which]
    if wanted:
        view = view[view[LABEL_COL] == wanted]
    view = sort_reviews(view, sort_choice)

    shown = st.session_state.get(f"n_shown_{key}", PAGE_SIZE)
    st.markdown(
        f'<div class="q-note" style="margin:2px 0 10px;">{len(view):,} of {len(reviews):,} '
        f'reviews · sorted by {sort_choice.lower()}</div>', unsafe_allow_html=True)

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
