"""
Streamlit view for the live council verdict — the only council module that imports streamlit.

`render_council(lat, lon)` builds a present-day snapshot of the local market (from the 1.6 panel),
runs the four seats, adjudicates deterministically, and renders the verdict + the per-seat table.
Cached in session_state by pin+radius so it only spends LLM calls on an explicit button click, not on
every Streamlit rerun.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from proforma.v1_6 import config
from proforma.v1_6.snapshot import build_live_snapshot
from proforma.v1_6.council import council_decision, council_notes

_VERDICT_COLOR = {"Build": "green", "Pass": "red", "Conditional": "orange", "Insufficient": "gray"}


def render_council(lat: float, lon: float, *, radius_km: float = 20.0, backend: Optional[str] = None) -> None:
    st.subheader("🧭 Council verdict — build here?")
    st.caption("A leakage-clean **data signal** (local-market structure + operator scale) makes the call; the "
               "LLM seats explain it but don't decide. Honest out-of-fold edge: **+10pt build precision, AUC "
               "0.57**. Reads the 1.6 market panel.")

    key = f"{round(lat, 4)}::{round(lon, 4)}::{int(radius_km)}"
    store = st.session_state.setdefault("council_store", {})
    if st.button("Run council", key="council_run_btn"):
        with st.spinner("Convening the council (4 seats)…"):
            try:
                snap = build_live_snapshot(lat, lon, radius_km=radius_km)
                store[key] = {"dec": council_decision(snap, backend=backend, use_web_search=False,
                                                      radius_km=radius_km), "n": snap.n_neighbours}
            except Exception as exc:                                   # keep the dashboard alive
                store[key] = {"error": str(exc)}

    rec = store.get(key)
    if not rec:
        st.info("Click **Run council** for an adjudicated build/pass verdict at this pin.")
        return
    if rec.get("error"):
        st.error(f"Council unavailable: {rec['error']}")
        return

    dec = rec["dec"]
    v = dec["verdict"]
    lbl = config.VERDICT_LABELS.get(v, v)
    col = _VERDICT_COLOR.get(v, "gray")
    prob = dec.get("prob")
    prob_txt = f"  ·  P(good build) {prob:.0%}" if prob is not None else ""
    st.markdown(f"### :{col}[{lbl}]  ·  confidence {dec['confidence']:.0%}{prob_txt}  ·  "
                f"{dec.get('n_matured_nbrs', 0)} matured neighbours")

    # Council notes — plain-language analysis summary (deterministic; internal-vs-external conflict shown inline)
    st.markdown("#### 📝 Council notes")
    st.markdown(council_notes(dec))

    pr = dec.get("projected_range")
    if pr and dec.get("anchor"):
        st.caption(f"Projected mature washes/mo — seats span **{pr['low']:,.0f}–{pr['high']:,.0f}** "
                   f"(median {pr['median']:,.0f}). Internal anchor **{dec['anchor']:,.0f}** vs healthy "
                   f"floor {dec['floor']:,.0f}. Competition assessed within "
                   f"**{dec.get('competition_radius_mi', config.COMPETITION_RADIUS_MI):g} mi**.")

    with st.expander("Per-seat detail"):
        rows = []
        for s in dec["seats"]:
            p = s["projected_mature_washes"]
            rows.append({"Seat": s["seat"], "Access": s["access"],
                         "Lean": config.VERDICT_LABELS.get(s["lean"], s["lean"] or "—"),
                         "Proj washes/mo": f"{p:,.0f}" if p else "—", "Conf": f"{s['confidence']:.0%}",
                         "Why": (s["reasons"][:1] or [""])[0]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("The **📊 data signal** (market structure + operator scale) decides; the internal read + the "
                   "three LLM seats are **context/explanation only** — they can flag disagreement but can't flip "
                   "the verdict. For how it holds up historically, see the backtest notes below.")


def render_reports() -> None:
    """Show the honest backtest report + the rebuild meeting notes (from proforma/v1_6/ files) in expanders."""
    from pathlib import Path
    base = Path(__file__).resolve().parent
    st.subheader("🗒️ Council backtest & meeting notes")
    report, notes = base / "outputs" / "retro_council_report.md", base / "COUNCIL_MEETING_NOTES.md"
    with st.expander("Backtest report — honest, out-of-fold (does it beat 'always build'?)"):
        st.markdown(report.read_text() if report.exists()
                    else "_No backtest yet — run `python -m proforma.v1_6.harness`._")
    with st.expander("Council meeting notes — the multi-agent rebuild session"):
        st.markdown(notes.read_text() if notes.exists() else "_No meeting notes found._")
