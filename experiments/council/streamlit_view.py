"""
Streamlit view — the ONLY council module that imports streamlit.

`render_council(lat, lon, radius_km=..., backend=None)` convenes the committee at a pin and renders: the
animated **Council Chamber** (an iframe from `chamber.build_chamber_html`), then the detail panels — verdict
+ signal-divergence, the in-depth report, the workspace evidence board, the discussion transcript, and the
per-seat belief evolution. `render_reports()` shows the honest signal backtest + a saved transcript.

IMPORTANT: the committee runs in a **subprocess** (`python -m experiments.council.committee …`), not in
Streamlit's ScriptRunner thread — the committee's pandas/numpy work segfaults in that thread on bleeding-edge
numpy/pyarrow stacks, but is rock-solid in a fresh process. The view only handles the JSON result + widgets.

Both entry points keep the exact signatures the dangling hook in `proforma/ui/panels/_explore_markets.py`
already calls (zero production edits). Everything is wrapped so it can never take down the host dashboard.
`backend` is accepted-and-ignored (Azure-only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# `streamlit run experiments/council/streamlit_view.py` (standalone) puts THIS file's dir on sys.path,
# not the repo root, so `import experiments.council.*` fails. Put the repo root on the path first.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from experiments.council import chamber
from experiments.council import config as C

_BASE = Path(__file__).resolve().parent
_CHAMBER_H = 820          # tall enough for the boardroom + speech bubbles + caption (use ⛶ for true fullscreen)


def _run_committee_subprocess(lat: float, lon: float, radius_km: float, light: bool) -> Dict[str, Any]:
    """Run the committee in a FRESH process (main thread) and parse its JSON — avoids the pandas/numpy
    segfault that hits Streamlit's ScriptRunner thread on some stacks."""
    proc = subprocess.run(
        [sys.executable, "-m", "experiments.council.committee", str(lat), str(lon), str(radius_km),
         "1" if light else "0"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=420, env=dict(os.environ))
    out = proc.stdout or ""
    marker = "__COUNCIL_JSON__"
    i = out.rfind(marker)
    if i < 0:
        err = (proc.stderr or "")[-1200:] or out[-800:] or "(no output)"
        raise RuntimeError(f"committee subprocess exited {proc.returncode}: {err}")
    return json.loads(out[i + len(marker):])


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt_value(x)}" for k, x in v.items() if x is not None)
    if isinstance(v, list):
        return f"{len(v)} item(s)"
    return str(v)


def render_council(lat: float, lon: float, *, radius_km: float = 20.0, backend: Optional[str] = None) -> None:
    st.subheader("🧭 AI Site-Selection Committee")
    st.caption("Five domain experts fetch real data, publish it to a shared board, then **challenge / "
               "question / revise / vote** across rounds until they converge on a build/no-build call. A "
               "data signal sits on the table as a quiet cross-check; the committee decides. Azure-only.")

    key = f"{round(lat, 4)}::{round(lon, 4)}::{int(radius_km)}"
    store = st.session_state.setdefault("committee_store", {})
    c1, c2 = st.columns([1, 2])
    with c1:
        run = st.button("▶ Convene committee", key="committee_run", type="primary")
    with c2:
        light = st.checkbox("Light mode (data only, no LLM discussion)", value=False,
                            help="Skip the LLM debate — the data-weighted verdict only. Fast + free.")

    if run:
        with st.spinner("The committee is meeting — 5 experts investigating, then debating… (~30–90s)"):
            try:
                store[key] = {"data": _run_committee_subprocess(lat, lon, radius_km, light)}
            except Exception as exc:                       # never take down the host dashboard
                store[key] = {"error": str(exc)}

    rec = store.get(key)
    if not rec:
        st.info("Drop a pin and click **Convene committee** to watch the seats deliberate.")
        return
    if rec.get("error"):
        st.error(f"Committee unavailable: {rec['error']}")
        return

    data = rec["data"]

    # ── the animated chamber (the centerpiece) ──
    try:
        components.html(chamber.build_chamber_html(data.get("chamber", {}), height=_CHAMBER_H),
                        height=_CHAMBER_H + 12, scrolling=True)
        st.caption("Tip: click **⛶** (bottom-right of the chamber) for true fullscreen — the speech "
                   "bubbles have the most room there. Each message also shows in the caption bar under the table.")
    except Exception as exc:
        st.warning(f"Chamber animation unavailable ({exc}); see the details below.")

    # ── verdict + signal-divergence banner ──
    verdict = data.get("verdict", "—")
    lbl = C.VERDICT_LABELS.get(verdict, verdict)
    col = {"Build": "green", "Pass": "red", "Conditional": "orange"}.get(verdict, "gray")
    prob = data.get("prob")
    prob_txt = f"  ·  signal P(good build) {prob:.0%}" if isinstance(prob, (int, float)) else ""
    conf = data.get("confidence") or 0.0
    st.markdown(f"### :{col}[{lbl}]  ·  {conf:.0%} committee confidence{prob_txt}")
    st.caption(f"Basis: {data.get('basis', '')}")
    st.caption("**Committee confidence** = how firmly the seats landed on this verdict (from the data-weighted "
               "vote — near 50% means a genuinely split committee). **Signal · P(good build)** = the data-signal "
               "model's *independent* probability this is a good build, shown as a quiet cross-check (it does not "
               "vote). When it disagrees with the committee, that's the 🔍 note below.")

    with st.expander("ℹ️ What do “P(good build)” and “leakage-clean” actually mean?"):
        st.markdown(
            "**P(good build)** — the *data signal's* probability that this site matures into a "
            "**good build**: a wash that reaches an **above-median mature volume with a healthy ramp**. "
            "_Example:_ **P = 24%** ≈ a 1-in-4 chance it becomes a strong performer — so the hard data leans "
            "*against* building, even if the LLM seats sound optimistic.\n\n"
            "**Leakage-clean** — the signal is validated with **no peeking at the future**. To grade it "
            "honestly we *froze the clock* at each past site's opening month, showed the model **only what "
            "existed before then** (the neighbouring washes' prior history, the operator's scale at the time), "
            "and scored its call against what the site *actually* did afterward. _Example:_ the old trained "
            "forecast looked brilliant at **0.74 in-sample** but **collapsed to ~0** under this honest test — "
            "it had *memorized* existing sites, not forecast new ones. This signal held at **AUC 0.572** "
            "(> the 0.53 noise ceiling), so its **+10-pt edge is real, not a mirage** — which is exactly why "
            "its disagreement with the committee is worth flagging.")
    if data.get("condition"):
        st.warning(f"**Condition:** {data['condition']}")
    if data.get("note"):
        st.caption(data["note"])          # quiet cross-check, not a loud warning

    # ── how the vote is weighted (one seat does NOT equal one vote) ──
    with st.expander("⚖️ How the vote is weighted — do all experts have equal power?"):
        st.caption("No — data-grounded seats out-weigh world-knowledge ones (the guardrail against a room of "
                   "bullish LLMs drifting to 'always build'). The data signal does **not** vote; it's a quiet "
                   "cross-check. Only seats that took a lean vote; Capacity sizes the tunnel and abstains on go/no-go.")
        for e in data.get("chamber", {}).get("experts", []):
            role = "🌐 world-knowledge (down-weighted)" if e.get("is_world") else "🔒 data-grounded"
            lean = e.get("lean") or "abstains"
            st.caption(f"{e.get('emoji','')} **{e.get('name')}** — vote weight **{e.get('weight')}** · {role} · leans *{lean}*")
        st.caption("🎯 **Data signal** — a P(good-build) **cross-check; does NOT vote** (validated out-of-fold, "
                   "the one component with a *measured* edge — shown for reference, not to drive the verdict).")

    # ── final report ──
    with st.expander("📄 In-depth committee report", expanded=True):
        st.markdown(data.get("report") or "_(no report — light mode)_")

    # ── workspace evidence board ──
    with st.expander("🗂️ Workspace board — what each seat put on the table"):
        by_expert: Dict[str, List[dict]] = {}
        for e in data.get("evidence", []):
            by_expert.setdefault(e.get("expert"), []).append(e)
        cols = st.columns(len(C.EXPERT_ORDER))
        for col_w, ekey in zip(cols, C.EXPERT_ORDER):
            meta = C.EXPERT_META.get(ekey, {})
            with col_w:
                st.markdown(f"**{meta.get('emoji','')} {meta.get('name', ekey)}**")
                evs = by_expert.get(ekey, [])
                if not evs:
                    st.caption("—")
                for e in evs:
                    unit = f" {e['unit']}" if e.get("unit") else ""
                    st.caption(f"{e.get('badge','')} {e.get('label','')}: **{_fmt_value(e.get('value'))}**{unit}")

    # ── discussion transcript (from the chamber messages) ──
    with st.expander("💬 Full discussion transcript"):
        msgs = data.get("chamber", {}).get("messages", [])
        if not msgs:
            st.caption("_(light mode — no discussion)_")
        last = -1
        for m in msgs:
            if m.get("round") != last:
                last = m.get("round")
                st.markdown(f"**── Round {last} ──**")
            color = m.get("type_color") or C.MSG_COLORS.get(m.get("type"), "#64748b")
            tgt = f" → {m.get('to_name')}" if m.get("to_name") else ""
            st.markdown(f"<span style='color:{color};font-weight:600'>[{m.get('type')}]</span> "
                        f"**{m.get('sender_name')}**{tgt}: {m.get('text','')}", unsafe_allow_html=True)

    # ── belief evolution (text; no in-thread pandas) ──
    with st.expander("📈 Belief evolution (lean · confidence per round)"):
        hist = data.get("chamber", {}).get("belief_history", {})
        if not hist:
            st.caption("_(no belief history)_")
        for ekey in C.EXPERT_ORDER:
            seq = hist.get(ekey) or []
            if not seq:
                continue
            name = C.EXPERT_META.get(ekey, {}).get("name", ekey)
            trail = " → ".join(f"{h.get('lean') or '—'}/{int((h.get('confidence') or 0)*100)}%" for h in seq)
            moved = len({h.get("lean") for h in seq}) > 1
            st.markdown(f"- **{name}**: {trail}" + ("  🔁 *moved*" if moved else ""))


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
