"""
The signal exhibit + the deterministic decision rule (the "Skill" — computed, not free-associated).

`compute_anchor(snap) -> Anchor` — the leakage-clean signal decider's independent Build/Pass call (the only
component with measured out-of-fold edge). It goes on the board as evidence; it is NOT a veto.

`decide_final(ws, mode) -> Decision`:
  • MODE="deliberative" (default) → the **committee's consensus vote** decides, tallied with data-grounded
    experts weighted above world-knowledge ones (`_weight`); the signal contributes its own weighted vote;
    the signal's independent call is recorded and its **divergence** from the committee is surfaced.
  • MODE="anchored" → the signal governs the binary whenever informative; the committee only refines the
    numbers, nudges confidence (≤±CONF_NUDGE_MAX), and may push Build→Conditional on a real open challenge.

Salvages `signal_verdict`/`internal_anchor`/`_signal_context_note` + `adjudicate`'s weighted math from the
old council.py. Self-contained: imports only council modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.council import config as C
from experiments.council import data_1_6 as D
from experiments.council import decider as DEC
from experiments.council import features as F
from experiments.council.protocol import Evidence, MsgType

_YES = {"Build", "Conditional"}
_SCORE = {"Build": 1.0, "Conditional": 0.5, "Pass": 0.0}


# ─────────────────────────── the signal exhibit ───────────────────────────
@dataclass
class Anchor:
    lean: Optional[str]                 # Build | Pass | Conditional | None
    prob: Optional[float]               # P(good build)
    confidence: float
    n_matured: int
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)

    @property
    def abstains(self) -> bool:
        return self.lean is None or self.n_matured < 2

    def as_evidence(self) -> Evidence:
        return Evidence(
            eid="signal.decider", expert="historical",
            label="data signal — P(good build) from market structure + operator scale (a quiet cross-check, validated out-of-fold)",
            value={"lean": self.lean, "prob": self.prob, "n_matured": self.n_matured},
            kind="flag", source="decider.score_features", confidence=self.confidence, leakage_safe=True)

    def to_dict(self) -> Dict[str, Any]:
        return {"lean": self.lean, "prob": self.prob, "confidence": round(float(self.confidence), 3),
                "n_matured": self.n_matured, "reasons": self.reasons, "abstains": self.abstains}


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def compute_anchor(snap) -> Anchor:
    """The signal decider's independent call for the focal build (leakage-clean features → P(good build))."""
    feat = F.site_features(snap.focal_key, snap.lat, snap.lon, snap.as_of)
    prob = DEC.score_features(feat)
    lean, conf = DEC.decide(prob)
    n_mat = int(feat.get("n_matured_pre_nbrs") or 0)
    reasons: List[str] = []
    mn, dem = _finite(feat.get("nbr_mat_min_strict")), _finite(feat.get("local_recent_wash_mean"))
    if mn is not None:
        reasons.append(f"weakest matured rival ~{mn:,.0f}/mo ({'headroom' if mn < 6000 else 'already big → tight'})")
    if dem is not None:
        reasons.append(f"local demand ~{dem:,.0f} washes/mo/site")
    reasons.append(f"operator runs ~{int(feat.get('op_n_sites_preT') or 1)} site(s)")
    if not n_mat:
        reasons.append("no matured neighbours yet → weak signal")
    return Anchor(lean=lean, prob=prob, confidence=conf, n_matured=n_mat, reasons=reasons, features=feat)


# ─────────────────────────── the decision ───────────────────────────
@dataclass
class Decision:
    verdict: str                        # Build | Pass | Conditional | Insufficient
    confidence: float
    basis: str                          # how the verdict was reached
    prob: Optional[float] = None        # P(good build) from the signal
    numbers: Dict[str, Any] = field(default_factory=dict)   # revenue / memberships / washcount / tunnel / capex / net / breakeven
    condition: Optional[str] = None     # the condition that would flip a Conditional
    note: Optional[str] = None          # signal-vs-committee divergence, surfaced not averaged
    signal_lean: Optional[str] = None
    diverges_from_signal: bool = False
    yes_frac: Optional[float] = None    # committee weighted yes-score (0..1)
    n_votes: int = 0
    mode: str = "deliberative"

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "confidence": round(float(self.confidence), 3), "basis": self.basis,
                "prob": self.prob, "numbers": self.numbers, "condition": self.condition, "note": self.note,
                "signal_lean": self.signal_lean, "diverges_from_signal": self.diverges_from_signal,
                "yes_frac": self.yes_frac, "n_votes": self.n_votes, "mode": self.mode}


def _weight(expert: str) -> float:
    return C.WORLD_EXPERT_WEIGHT if expert in C.WORLD_EXPERTS else C.DATA_EXPERT_WEIGHT


def _tally(ws) -> tuple:
    """Weighted consensus over experts' current leans. Data-grounded experts outweigh world-knowledge ones;
    the signal exhibit casts its own weighted vote (unless it abstains). Returns (yes_frac 0..1 | None, n_votes)."""
    num = den = 0.0
    n_votes = 0
    for b in ws.beliefs.values():
        if b.lean is None:
            continue
        w = _weight(b.expert) * max(float(b.confidence), 0.1)
        num += w * _SCORE.get(b.lean, 0.5)
        den += w
        n_votes += 1
    a = getattr(ws, "anchor", None)
    if a is not None and not a.abstains:
        w = C.SIGNAL_WEIGHT * max(float(a.confidence), 0.1)
        num += w * _SCORE.get(a.lean, 0.5)
        den += w
    return ((num / den) if den else None), n_votes


_NUM_EIDS = {
    "revenue_5yr": "fin.revenue_5yr", "net_5yr": "fin.net_5yr", "breakeven_month": "fin.breakeven",
    "membership_share": "fin.membership", "capex": "fin.capex", "tunnel_ft": "cap.tunnel_ft",
    "mature_washes": "hist.projected_mature", "peak_month_washes": "cap.peak_month_washes",
}


def _collect_numbers(ws) -> Dict[str, Any]:
    """Gather the committee's headline projected numbers off the board (each expert owns its own)."""
    out: Dict[str, Any] = {}
    for name, eid in _NUM_EIDS.items():
        ev = ws.evidence.get(eid)
        out[name] = (ev.value if ev is not None else None)
    return out


def _open_challenge_exists(ws) -> bool:
    return bool([m for m in getattr(ws, "open_challenges", []) if m.answered_by is None])


def _signal_context_note(a: Optional[Anchor], verdict: str, yes: Optional[float]) -> Optional[str]:
    """When the committee and the leakage-clean signal disagree, say so plainly — the data is the one thing
    with measured edge, so a divergence is worth a human's eye."""
    if a is None or a.lean is None or verdict in (None, "Insufficient"):
        return None
    sig_yes, com_yes = a.lean in _YES, verdict in _YES
    if sig_yes == com_yes:
        return None
    lbl = C.VERDICT_LABELS.get(a.lean, a.lean)
    if not com_yes and sig_yes:
        return (f"🔍 Cross-check: the committee leans **{C.VERDICT_LABELS.get(verdict, verdict)}**, while the data "
                f"signal (validated out-of-fold) reads **{lbl}** — worth a second look.")
    return (f"🔍 Cross-check: the committee leans **{C.VERDICT_LABELS.get(verdict, verdict)}**, while the data "
            f"signal reads **{lbl}** (weak headroom / tight market) — worth a second look.")


def decide_final(ws, *, mode: Optional[str] = None) -> Decision:
    """Turn the settled board into a verdict. Deliberative = committee decides (data-weighted); anchored =
    the signal governs the binary and the committee only refines. Deterministic, no LLM."""
    mode = (mode or C.MODE).strip().lower()
    a = getattr(ws, "anchor", None)
    numbers = _collect_numbers(ws)
    yes, n_votes = _tally(ws)
    signal_lean = a.lean if a else None
    condition: Optional[str] = None

    if mode == "anchored" and a is not None and not a.abstains:
        verdict, confidence, basis = a.lean, float(a.confidence), "signal decider (leakage-clean)"
        # committee may nudge confidence toward its consensus, bounded
        if yes is not None:
            nudge = max(-C.CONF_NUDGE_MAX, min(C.CONF_NUDGE_MAX, (yes - 0.5) * 0.3))
            confidence = float(np.clip(confidence + nudge, 0.20, 0.95))
        # Build → Conditional only on a real unanswered challenge AND a not-high-confidence signal
        if verdict == "Build" and _open_challenge_exists(ws) and a.confidence < C.SIGNAL_HIGH_CONF:
            verdict, condition = "Conditional", "revisit if the open challenge on the board is resolved against the build"
    else:
        # deliberative: the committee's consensus decides
        if yes is None:
            verdict, confidence, basis = "Insufficient", 0.2, "no expert produced a lean"
        else:
            verdict = "Build" if yes >= 0.6 else ("Pass" if yes <= 0.4 else "Conditional")
            confidence = round(float(0.45 + 0.4 * abs(yes - 0.5) * 2), 2)
            basis = "committee consensus (data-weighted vote)"
            if verdict == "Conditional" and _open_challenge_exists(ws):
                condition = "resolve the open challenge(s) on the board"

    diverges = bool(signal_lean and verdict not in (None, "Insufficient")
                    and (signal_lean in _YES) != (verdict in _YES))
    note = _signal_context_note(a, verdict, yes)
    return Decision(verdict=verdict, confidence=round(float(confidence), 2), basis=basis,
                    prob=(a.prob if a else None), numbers=numbers, condition=condition, note=note,
                    signal_lean=signal_lean, diverges_from_signal=diverges, yes_frac=yes,
                    n_votes=n_votes, mode=mode)
