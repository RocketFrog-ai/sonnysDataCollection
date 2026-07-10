"""
The council — assemble the existing insight modes into a panel of seats, extract a structured
verdict from each, then adjudicate with a DETERMINISTIC rulebook (not a free-form LLM fusion).

Seats (the four insight panels that already ship, called UNCHANGED):
  • internal    — `compute_metrics` (grounded, pure-Python read of the pre-T neighbour panel — the
                  metrics half of market_insights, no LLM narration) + a leakage-free local
                  mature-neighbour ANCHOR as the projected mature-wash level. WEIGHT = max.
  • independent — `independent_market_research` (external LLM sizing; emits its own wash_volume).
  • competition — `competition_scale_analysis` (saturation / headroom).
  • location    — `location_market_analysis` (qualitative world-knowledge read; lean via a small extract).

`adjudicate` encodes Dhruv's anti-hallucination rule: when the internal (data) seat contradicts the
external majority, the data wins but the disagreement is surfaced explicitly rather than averaged away.
Nothing under app/ is modified; this module only imports and orchestrates.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from council import config
from council import data_1_6 as D
from council import features as F
from council import decider as DEC
from council.snapshot import Snapshot
from app.pnl_analysis.insights.metrics import compute_metrics
from app.pnl_analysis.insights import location_poc as LP
from app.pnl_analysis.insights import llm as llm_client

logger = logging.getLogger(__name__)

SEAT_INTERNAL, SEAT_INDEPENDENT, SEAT_COMPETITION, SEAT_LOCATION = (
    "internal", "independent", "competition", "location")
SEATS = (SEAT_INTERNAL, SEAT_INDEPENDENT, SEAT_COMPETITION, SEAT_LOCATION)
_YES = {"Build", "Conditional"}
_CONF = {"high": 0.8, "medium": 0.5, "low": 0.3}


# ─────────────────────────── numeric helpers ───────────────────────────
def _num_mid(v: Any) -> Optional[float]:
    """Midpoint of a numeric estimate — a number, a {'low','high'} dict, or a string range like
    '8,000–12,000' / '~10k'. None when nothing numeric is present."""
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    if isinstance(v, dict):
        nums = [float(x) for x in (v.get("low"), v.get("high")) if isinstance(x, (int, float))]
        return sum(nums) / len(nums) if nums else None
    if isinstance(v, str):
        nums = [float(x.replace(",", "")) for x in re.findall(r"[\d,]+(?:\.\d+)?", v)]
        nums = [n for n in nums if n > 0]
        return sum(nums) / len(nums) if nums else None
    return None


def _conf(x: Any, default: float = 0.4) -> float:
    return _CONF.get(str(x).strip().lower(), default)


def _lean_from_level(projected: Optional[float], floor: float, trend: Optional[float] = None) -> Optional[str]:
    """Turn a projected mature level into a build lean, relative to the healthy-site floor. A weak market
    trend can veto a borderline Build. None (abstain) when there's no projection."""
    if projected is None or floor <= 0:
        return None
    ratio = projected / floor
    if ratio >= 1.0 and (trend is None or trend > -0.05):
        return "Build"
    if ratio < 0.5 or (trend is not None and trend < -0.15):
        return "Pass"
    return "Conditional"


# ─────────────────────────── internal mature-neighbour anchor (leakage-free) ───────────────────────────
def internal_anchor(snap: Snapshot) -> Optional[float]:
    """Projected mature-wash level for the focal build = median of the pre-T neighbours' OWN mature levels
    (their months 18–30), computed strictly from `date < T` rows. Falls back to the neighbours' recent
    pre-T level when none has matured yet. Pure Python, no trained model, no future leakage."""
    p = snap.panel_asof
    if p.empty:
        return None
    mat = p[(p.rel >= D.MAT_LO) & (p.rel <= D.MAT_HI)].groupby("site_key")["tot_wash_count"].mean().dropna()
    if len(mat):
        return float(mat.median())
    recent = (p.sort_values("date").groupby("site_key").tail(6)
              .groupby("site_key")["tot_wash_count"].mean().dropna())
    return float(recent.median()) if len(recent) else None


# ─────────────────────────── run the seats (existing fns, unchanged) ───────────────────────────
def run_seats(snap: Snapshot, *, backend: Optional[str] = None, use_web_search: bool = False,
              radius_km: float = config.RADIUS_KM,
              competition_radius_mi: float = config.COMPETITION_RADIUS_MI,
              independent_radius_mi: float = config.INDEPENDENT_RADIUS_MI) -> Dict[str, Any]:
    """Call the four insight functions concurrently. Per-seat trade-area radii (see config): internal +
    location use `radius_km`; competition uses a tight `competition_radius_mi`; independent sizes at
    `independent_radius_mi`. A seat that raises is captured as {'error': ...} so the council degrades."""
    comp_km = competition_radius_mi * config.MI_TO_KM
    comp_nearby = [w for w in snap.nearby_washes
                   if (w.get("distance_miles") if w.get("distance_miles") is not None else 1e9) <= competition_radius_mi]

    def _internal():
        # compute_metrics only (pure Python, no LLM) — the council verdict uses the metrics + anchor, not
        # the narrative that market_insights would additionally spend an LLM call on. Leakage-free.
        if snap.panel_asof.empty or snap.n_neighbours == 0:   # no pre-T neighbours → seat abstains cleanly
            return {"error": "no pre-T neighbour data"}
        return {"metrics": compute_metrics(snap.panel_asof, snap.sites_meta_asof, snap.focal_key)}

    def _location():
        return LP.location_market_analysis(snap.lat, snap.lon, radius_km=radius_km,
                                           backend=backend, use_web_search=use_web_search)

    def _competition():
        # tight trade area + only the ground-truth washes inside it, so the LLM doesn't over-count rivals
        return LP.competition_scale_analysis(snap.lat, snap.lon, known_sites=[], radius_km=comp_km,
                                             backend=backend, nearby_washes=comp_nearby)

    def _independent():
        # single radius: the site-level wash_volume is ~radius-independent, so one call, not three.
        return LP.independent_market_research(snap.lat, snap.lon, radii_miles=(independent_radius_mi,),
                                              backend=backend, use_web_search=use_web_search)

    jobs = {SEAT_INTERNAL: _internal, SEAT_LOCATION: _location,
            SEAT_COMPETITION: _competition, SEAT_INDEPENDENT: _independent}
    out: Dict[str, Any] = {}

    def _safe(name, fn):
        try:
            return name, fn()
        except Exception as exc:                       # a down backend must not kill the whole council
            logger.warning("Seat %s failed: %s", name, exc)
            return name, {"error": str(exc)}

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for name, res in ex.map(lambda kv: _safe(*kv), jobs.items()):
            out[name] = res
    return out


# ─────────────────────────── per-seat structured verdict ───────────────────────────
def _v(seat, lean, projected, confidence, reasons, access) -> Dict[str, Any]:
    return {"seat": seat, "lean": lean, "projected_mature_washes": projected,
            "confidence": float(confidence), "reasons": reasons, "access": access}


def _verdict_internal(raw: Any, anchor: Optional[float], floor: float) -> Dict[str, Any]:
    if not isinstance(raw, dict) or "metrics" not in raw:
        return _v(SEAT_INTERNAL, None, anchor, 0.2, ["internal seat unavailable"], "internal")
    m = raw["metrics"]
    trend = (((m.get("washes") or {}).get("total") or {}).get("trend_annual"))
    n_sites = ((m.get("site_selection") or {}).get("sites_in_market")) or 0
    lean = _lean_from_level(anchor, floor, trend)
    conf = 0.35 + min(0.4, 0.08 * n_sites) + (0.1 if trend is not None else 0)   # more neighbours → firmer
    reasons = []
    if anchor is not None:
        reasons.append(f"local mature-neighbour anchor ≈ {anchor:,.0f} washes/mo vs floor {floor:,.0f}")
    if trend is not None:
        reasons.append(f"pre-T market wash trend {trend*100:+.0f}%/yr over {n_sites} neighbours")
    return _v(SEAT_INTERNAL, lean, anchor, min(conf, 0.9), reasons or ["thin local data"], "internal")


def _verdict_independent(raw: Any, floor: float) -> Dict[str, Any]:
    radii = (raw or {}).get("radii") if isinstance(raw, dict) else None
    metrics = (radii[0].get("metrics") if radii else None) or {}
    wv = metrics.get("wash_volume") if isinstance(metrics.get("wash_volume"), dict) else {}
    projected = _num_mid(wv.get("estimate"))
    lean = _lean_from_level(projected, floor)
    conf = _conf(wv.get("confidence"))
    reason = (wv.get("basis") or "external LLM market sizing").strip()
    return _v(SEAT_INDEPENDENT, lean, projected, conf, [reason], "external")


def _verdict_competition(raw: Any, floor: float) -> Dict[str, Any]:
    data = (raw or {}).get("data") if isinstance(raw, dict) else None
    data = data or {}
    sat = str(data.get("saturation") or "").strip().lower()
    # more express supply headroom → Build lean; saturated → Pass
    lean = "Build" if sat.startswith("low") else ("Pass" if sat.startswith("high") else
                                                  ("Conditional" if sat else None))
    conf = _conf(data.get("confidence"))
    reasons = []
    if data.get("saturation"):
        reasons.append(f"express saturation: {data.get('saturation')}")
    if data.get("headroom"):
        reasons.append(str(data.get("headroom"))[:160])
    return _v(SEAT_COMPETITION, lean, None, conf, reasons or ["no competition read"], "external")


_LOC_EXTRACT_SYS = (
    "You classify a car-wash market analysis into a build decision. Read the analyst's location read "
    "and return STRICT JSON only: {\"lean\": \"Build|Pass|Conditional\", \"confidence\": 0.0-1.0, "
    "\"reason\": \"one short clause\"}. Build = demand/location clearly support a new express tunnel; "
    "Pass = they clearly do not; Conditional = mixed. Judge ONLY from the text given."
)


def _verdict_location(raw: Any, *, backend: Optional[str], extract: bool) -> Dict[str, Any]:
    text = (raw or {}).get("text") if isinstance(raw, dict) else None
    if not text:
        return _v(SEAT_LOCATION, None, None, 0.2, ["location seat unavailable"], "external")
    if not extract:                                    # context-only mode (no extra LLM call)
        return _v(SEAT_LOCATION, None, None, 0.3, ["qualitative context (no vote)"], "external")
    try:
        msgs = [{"role": "system", "content": _LOC_EXTRACT_SYS},
                {"role": "user", "content": text[:6000]}]
        out, _ = llm_client.complete_cascade(msgs, backend=backend, max_tokens=200,
                                             temperature=0.0, json_mode=True)
        j = LP._parse_json_lax(out)
        lean = str(j.get("lean") or "").title()
        lean = lean if lean in {"Build", "Pass", "Conditional"} else None
        conf = float(j.get("confidence") or 0.4)
        return _v(SEAT_LOCATION, lean, None, min(max(conf, 0.0), 1.0),
                  [str(j.get("reason") or "world-knowledge read")], "external")
    except Exception as exc:
        logger.warning("Location lean extraction failed: %s", exc)
        return _v(SEAT_LOCATION, None, None, 0.3, ["location lean unavailable"], "external")


def extract_verdicts(raw: Dict[str, Any], *, anchor: Optional[float], floor: float,
                     backend: Optional[str], extract_location: bool = True) -> List[Dict[str, Any]]:
    return [
        _verdict_internal(raw.get(SEAT_INTERNAL), anchor, floor),
        _verdict_independent(raw.get(SEAT_INDEPENDENT), floor),
        _verdict_competition(raw.get(SEAT_COMPETITION), floor),
        _verdict_location(raw.get(SEAT_LOCATION), backend=backend, extract=extract_location),
    ]


# ─────────────────────────── deterministic adjudication (the Skill) ───────────────────────────
def _score(lean: Optional[str]) -> Optional[float]:
    return {"Build": 1.0, "Conditional": 0.5, "Pass": 0.0}.get(lean)


def adjudicate(verdicts: List[Dict[str, Any]], *, problem_type: str = "go_no_go",
               w_internal: Optional[float] = None) -> Dict[str, Any]:
    """Deterministic go/no-go rulebook. Internal seat weighted highest; agreement → confident plain call;
    internal-vs-external conflict → the data (internal) wins but the split is surfaced, never averaged away."""
    if problem_type != "go_no_go":
        raise NotImplementedError(f"adjudication rule-set '{problem_type}' not implemented (v1 = go_no_go)")

    voting = [v for v in verdicts if v.get("lean") is not None]
    internal = next((v for v in verdicts if v.get("access") == "internal"), None)
    int_voting = internal if internal and internal.get("lean") is not None else None
    externals = [v for v in voting if v.get("access") == "external"]

    # projected level range across every seat that gave a number (internal first)
    projs = [v["projected_mature_washes"] for v in verdicts if v.get("projected_mature_washes") is not None]
    projected_range = ({"low": float(min(projs)), "median": float(np.median(projs)), "high": float(max(projs))}
                       if projs else None)

    if not voting:
        return {"verdict": "Insufficient", "confidence": 0.2, "weighted_yes": None,
                "projected_range": projected_range, "conflict": False,
                "conflict_note": "No seat produced a lean.", "condition_that_flips_it": None,
                "abstained": True, "w_internal": None, "seats": verdicts}

    # weighted yes-score (Build=1, Conditional=.5, Pass=0); internal carries a dominant weight
    ext_w = {v["seat"]: max(v["confidence"], 0.1) for v in externals}
    total_ext = sum(ext_w.values()) or 1.0
    if w_internal is None:
        w_internal = total_ext                          # internal ≈ all externals combined
    num = sum(ext_w[v["seat"]] * _score(v["lean"]) for v in externals)
    den = total_ext
    ext_yes_score = num / den if den else None          # 0..1 external consensus
    if int_voting is not None:
        num += w_internal * _score(int_voting["lean"])
        den += w_internal
    weighted_yes = num / den                            # 0..1 overall

    leans = [v["lean"] for v in voting]
    all_agree = len(set(leans)) == 1

    conflict, conflict_note, condition = False, None, None
    if all_agree:
        verdict = leans[0]
        confidence = min(0.9, 0.6 + 0.08 * len(voting))
    else:
        int_yes = (_score(int_voting["lean"]) >= 0.5) if int_voting is not None else None
        ext_yes = (ext_yes_score >= 0.5) if ext_yes_score is not None else None
        conflict = int_voting is not None and ext_yes is not None and (int_yes != ext_yes)
        if conflict:
            # the data (internal) wins the direction; commit only if internal is confident, else Conditional
            verdict = int_voting["lean"] if int_voting["confidence"] >= 0.6 else "Conditional"
            confidence = 0.5
            n_ext = len(externals)
            side = "BUILD" if int_yes else "PASS"
            oside = "BUILD" if ext_yes else "PASS"
            conflict_note = (f"Internal (data-grounded) leans {int_voting['lean']} [{side}]; "
                             f"{n_ext} external world-knowledge seats lean {oside}. "
                             f"The data takes precedence — surfaced, not averaged.")
            condition = ("revisit if the local neighbour data moves toward the external view "
                         "(demand/level turns up or down)")
        else:
            verdict = "Build" if weighted_yes >= 0.6 else ("Pass" if weighted_yes <= 0.4 else "Conditional")
            confidence = 0.5 + 0.4 * abs(weighted_yes - 0.5) * 2
            if len(set(leans)) > 1:
                conflict_note = "Seats split but internal aligns with the external majority."

    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 2),
        "weighted_yes": round(float(weighted_yes), 3),
        "projected_range": projected_range,
        "conflict": conflict,
        "conflict_note": conflict_note,
        "condition_that_flips_it": condition,
        "abstained": False,
        "w_internal": round(float(w_internal), 2),
        "seats": verdicts,
    }


# ─────────────────────────── the SIGNAL seat (data-driven, DECIDING) ───────────────────────────
def _finite(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None and np.isfinite(float(x)) else None
    except (TypeError, ValueError):
        return None


def signal_verdict(snap: Snapshot) -> Dict[str, Any]:
    """The DECIDING seat: a leakage-clean market-structure + operator classifier → P(good build) → lean.
    This is the only component with a real out-of-fold edge (AUC ~0.57, +10pt build precision); the LLM
    seats are demoted to context because they carry no discriminating signal."""
    feat = F.site_features(snap.focal_key, snap.lat, snap.lon, snap.as_of)
    prob = DEC.score_features(feat)
    lean, conf = DEC.decide(prob)
    reasons: List[str] = []
    mn, dem = _finite(feat.get("nbr_mat_min_strict")), _finite(feat.get("local_recent_wash_mean"))
    if mn is not None:
        reasons.append(f"weakest matured rival ~{mn:,.0f}/mo ({'headroom' if mn < 6000 else 'already big → tight'})")
    if dem is not None:
        reasons.append(f"local demand ~{dem:,.0f} washes/mo/site")
    reasons.append(f"operator runs ~{int(feat.get('op_n_sites_preT') or 1)} site(s) by then")
    if not feat.get("n_matured_pre_nbrs"):
        reasons.append("no matured neighbours yet → weak signal")
    return {"seat": "signal", "access": "signal", "lean": lean, "prob": prob, "confidence": conf,
            "reasons": reasons, "projected_mature_washes": None, "features": feat,
            "n_matured_nbrs": int(feat.get("n_matured_pre_nbrs") or 0)}


def _signal_context_note(sig: Dict[str, Any], ctx: List[Dict[str, Any]]) -> Optional[str]:
    """When the data signal and the bullish LLM context disagree, say so — but the DATA decides."""
    ctx_leans = [v.get("lean") for v in ctx if v.get("lean") and v.get("access") == "context"]
    if not ctx_leans or sig.get("lean") is None:
        return None
    sig_yes = sig["lean"] in _YES
    ctx_yes = sum(1 for l in ctx_leans if l in _YES)
    if not sig_yes and ctx_yes > len(ctx_leans) / 2:
        return ("The world-knowledge seats are bullish, but the leakage-clean market-structure signal says "
                f"{config.VERDICT_LABELS.get(sig['lean'], sig['lean'])} (weak headroom / tight market) — the data decides, not the LLM optimism.")
    if sig_yes and ctx_yes <= len(ctx_leans) / 2:
        return ("The world-knowledge seats are cautious, but the market data supports it — the data decides.")
    return None


# ─────────────────────────── convenience wrapper (signal-driven; LLMs = context) ───────────────────────────
def council_decision(snap: Snapshot, *, backend: Optional[str] = None, use_web_search: bool = False,
                     radius_km: float = config.RADIUS_KM,
                     competition_radius_mi: float = config.COMPETITION_RADIUS_MI,
                     independent_radius_mi: float = config.INDEPENDENT_RADIUS_MI,
                     extract_location: bool = True, run_context: bool = True,
                     w_internal: Optional[float] = None) -> Dict[str, Any]:
    """Signal-first: the data decider makes the call; the internal descriptive read + the three LLM seats are
    CONTEXT (explanation + a disagreement flag) and can no longer flip the verdict."""
    floor = D.mature_floor()
    anchor = internal_anchor(snap)
    sig = signal_verdict(snap)

    context: List[Dict[str, Any]] = []
    if run_context:
        raw = run_seats(snap, backend=backend, use_web_search=use_web_search, radius_km=radius_km,
                        competition_radius_mi=competition_radius_mi, independent_radius_mi=independent_radius_mi)
        context = extract_verdicts(raw, anchor=anchor, floor=floor, backend=backend,
                                   extract_location=extract_location)
        for v in context:
            v["access"] = "context"                              # demote — they explain, they don't decide

    projs = [p for p in ([anchor] + [v.get("projected_mature_washes") for v in context]) if _finite(p)]
    projected_range = ({"low": float(min(projs)), "median": float(np.median(projs)), "high": float(max(projs))}
                       if projs else None)

    if sig["lean"] is not None:
        verdict, confidence = sig["lean"], sig["confidence"]
        conflict_note = _signal_context_note(sig, context)
        conflict = bool(conflict_note)
    else:                                                        # signal abstains (no matured neighbours) → context fallback
        adj = adjudicate(context or [sig], w_internal=w_internal if w_internal is not None else config.W_INTERNAL)
        verdict, confidence, conflict, conflict_note = (adj["verdict"], adj["confidence"],
                                                        adj.get("conflict", False), adj.get("conflict_note"))

    return {"verdict": verdict, "confidence": confidence, "prob": sig["prob"], "projected_range": projected_range,
            "conflict": conflict, "conflict_note": conflict_note, "condition_that_flips_it": None,
            "anchor": anchor, "floor": floor, "n_neighbours": snap.n_neighbours,
            "n_matured_nbrs": sig["n_matured_nbrs"], "focal_key": snap.focal_key,
            "as_of": str(pd.Timestamp(snap.as_of).date()), "competition_radius_mi": competition_radius_mi,
            "seats": [sig] + context}


# ─────────────────────────── human-readable council notes (deterministic, no LLM) ───────────────────────────
_SEAT_LABEL = {"signal": "📊 Data signal (decides)", SEAT_INTERNAL: "Local data read",
               SEAT_INDEPENDENT: "External market sizing", SEAT_COMPETITION: "Competition",
               SEAT_LOCATION: "Location read"}


def council_notes(dec: Dict[str, Any]) -> str:
    """A short markdown summary — leads with the DECIDING data signal, then the LLM context (explanation
    only). Composed from the structured decision, no extra LLM call."""
    lbl = config.VERDICT_LABELS.get(dec["verdict"], dec["verdict"])
    seats = {s["seat"]: s for s in dec.get("seats", [])}
    prob = dec.get("prob")
    head = f"**Verdict: {lbl}** — {dec['confidence']:.0%} confidence"
    if prob is not None:
        head += f", P(good build) ≈ {prob:.0%}"
    head += f", {dec.get('n_matured_nbrs', 0)} matured neighbours to learn from."
    lines = [head, ""]

    sig = seats.get("signal", {})
    if sig:
        why = "; ".join((sig.get("reasons") or [])[:3])
        lean = config.VERDICT_LABELS.get(sig.get("lean"), sig.get("lean") or "no signal")
        lines.append(f"- **{_SEAT_LABEL['signal']}:** {lean} — {why}.")

    ctx = [n for n in (SEAT_INTERNAL, SEAT_INDEPENDENT, SEAT_COMPETITION, SEAT_LOCATION) if n in seats]
    if ctx:
        lines += ["", "_Context (explanation only — does not decide):_"]
        for name in ctx:
            s = seats[name]
            why = ((s.get("reasons") or [""])[0] or "").rstrip(". ").strip()
            lean = config.VERDICT_LABELS.get(s.get("lean"), s.get("lean") or "—")
            proj = s.get("projected_mature_washes")
            proj_txt = f" (~{proj:,.0f}/mo)" if proj else ""
            lines.append(f"  - {_SEAT_LABEL[name]}: {lean}{proj_txt}" + (f" — {why}" if why else ""))

    if dec.get("conflict_note"):
        lines += ["", f"⚔️ {dec['conflict_note']}"]
    return "\n".join(lines)
