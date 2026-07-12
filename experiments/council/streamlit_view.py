"""
Streamlit view — the ONLY council module that imports streamlit.

`render_council(lat, lon, radius_km=..., backend=None)` convenes the committee at a pin (cached in
session_state, run only on a button click) and renders: the animated **Council Chamber** (an iframe from
`chamber.build_chamber_html`), then the detail panels — verdict + signal-divergence, the in-depth report,
the workspace evidence board, the full discussion transcript, and the per-seat belief evolution.
`render_reports()` shows the honest signal backtest + a saved committee transcript.

Both entry points keep the exact signatures the dangling hook in `proforma/ui/panels/_explore_markets.py`
already calls, so wiring is automatic (zero production edits). Everything is wrapped so it can never take
down the host dashboard. `backend` is accepted-and-ignored (the council is Azure-only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from experiments.council import chamber
from experiments.council import config as C

_BASE = Path(__file__).resolve().parent
_CHAMBER_H = 660


def render_council(lat: float, lon: float, *, radius_km: float = 20.0, backend: Optional[str] = None) -> None:
    st.subheader("🧭 AI Site-Selection Committee")
    st.caption("Five domain experts fetch real data, publish it to a shared board, then **challenge / "
               "question / revise / vote** across rounds until they converge on a build/no-build call. The "
               "leakage-clean data signal sits on the table as evidence; the committee decides. Azure-only.")

    key = f"{round(lat, 4)}::{round(lon, 4)}::{int(radius_km)}"
    store = st.session_state.setdefault("committee_store", {})
    c1, c2 = st.columns([1, 2])
    with c1:
        run = st.button("▶ Convene committee", key="committee_run", type="primary")
    with c2:
        light = st.checkbox("Light mode (data only, no LLM discussion)", value=False,
                            help="Skip the LLM debate — the data-weighted verdict only. Fast + free.")

    if run:
        with st.spinner("The committee is meeting — 5 experts investigating, then debating…"):
            try:
                from experiments.council.committee import run_committee_pin
                store[key] = {"res": run_committee_pin(lat, lon, radius_km=radius_km, light=light)}
            except Exception as exc:                       # never take down the host dashboard
                store[key] = {"error": str(exc)}

    rec = store.get(key)
    if not rec:
        st.info("Drop a pin and click **Convene committee** to watch the seats deliberate.")
        return
    if rec.get("error"):
        st.error(f"Committee unavailable: {rec['error']}")
        return

    res = rec["res"]
    dec = res.decision
    try:
        cd = res.chamber_data()
    except Exception as exc:
        st.error(f"Chamber render failed: {exc}")
        cd = None

    # ── the animated chamber (the centerpiece) ──
    if cd is not None:
        try:
            components.html(chamber.build_chamber_html(cd, height=_CHAMBER_H), height=_CHAMBER_H + 12, scrolling=False)
        except Exception as exc:
            st.warning(f"Chamber animation unavailable ({exc}); see the details below.")

    # ── verdict + signal divergence banner ──
    lbl = C.VERDICT_LABELS.get(dec.verdict, dec.verdict)
    col = {"Build": "green", "Pass": "red", "Conditional": "orange"}.get(dec.verdict, "gray")
    prob = f"  ·  P(good build) {dec.prob:.0%}" if dec.prob is not None else ""
    st.markdown(f"### :{col}[{lbl}]  ·  {dec.confidence:.0%} confidence{prob}")
    st.caption(f"Basis: {dec.basis}")
    if dec.condition:
        st.warning(f"**Condition:** {dec.condition}")
    if dec.note:
        st.warning(dec.note)

    # ── final report ──
    with st.expander("📄 In-depth committee report", expanded=True):
        st.markdown(res.report or "_(no report — light mode)_")

    # ── workspace evidence board ──
    with st.expander("🗂️ Workspace board — what each seat put on the table"):
        cols = st.columns(len(C.EXPERT_ORDER))
        for col_w, ekey in zip(cols, C.EXPERT_ORDER):
            meta = C.EXPERT_META.get(ekey, {})
            with col_w:
                st.markdown(f"**{meta.get('emoji','')} {meta.get('name', ekey)}**")
                evs = res.workspace.evidence_of(ekey)
                if not evs:
                    st.caption("—")
                for e in evs:
                    v = e.value
                    if isinstance(v, float):
                        v = f"{v:,.0f}"
                    st.caption(f"{e.badge()} {e.label}: **{v}**{(' ' + e.unit) if e.unit else ''}")

    # ── discussion transcript ──
    with st.expander("💬 Full discussion transcript"):
        if not res.log.messages:
            st.caption("_(light mode — no discussion)_")
        last = -1
        for m in res.log.messages:
            if m.round != last:
                st.markdown(f"**── Round {m.round} ──**")
                last = m.round
            color = C.MSG_COLORS.get(m.mtype.value, "#64748b")
            snd = C.EXPERT_META.get(m.sender, {}).get("name", m.sender)
            tgt = f" → {C.EXPERT_META.get(m.to, {}).get('name', m.to)}" if m.to else ""
            st.markdown(f"<span style='color:{color};font-weight:600'>[{m.mtype.value}]</span> "
                        f"**{snd}**{tgt}: {m.text}", unsafe_allow_html=True)

    # ── belief evolution ──
    with st.expander("📈 Belief evolution (confidence per round)"):
        rows = []
        for ekey, b in res.workspace.beliefs.items():
            for h in b.history:
                rows.append({"round": h["round"], "seat": C.EXPERT_META.get(ekey, {}).get("name", ekey),
                             "confidence": h["confidence"]})
        if rows:
            df = pd.DataFrame(rows).pivot_table(index="round", columns="seat", values="confidence")
            st.line_chart(df)
        else:
            st.caption("_(no belief history)_")


def render_reports() -> None:
    """The honest backtest report + a saved committee transcript, in expanders."""
    st.subheader("🗒️ Council backtest & method")
    rep = _BASE / "outputs" / "retro_council_report.md"
    with st.expander("Signal backtest — honest, out-of-fold (does the anchor beat 'always build'?)"):
        st.markdown(rep.read_text() if rep.exists()
                    else "_Run `python -m experiments.council.harness` to generate this report._")
    samples = _BASE / "outputs" / "committee_samples"
    txts = sorted(samples.glob("*.txt")) if samples.exists() else []
    if txts:
        with st.expander("A sample committee transcript (retrospective, frozen at the site's opening)"):
            st.text(txts[0].read_text()[:8000])


# standalone page: `streamlit run experiments/council/streamlit_view.py`
def _standalone() -> None:
    st.set_page_config(page_title="AI Site-Selection Committee", page_icon="🧭", layout="wide")
    st.title("🧭 AI Site-Selection Committee")
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude", value=33.7490, format="%.4f")
    lon = c2.number_input("Longitude", value=-84.3880, format="%.4f")
    radius = c3.number_input("Radius (km)", value=20.0, min_value=5.0, max_value=50.0)
    render_council(float(lat), float(lon), radius_km=float(radius))
    st.divider()
    render_reports()


if __name__ == "__main__":
    _standalone()
