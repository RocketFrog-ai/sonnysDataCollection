"""
The deterministic decision rule — the committee decides; computed, not free-associated.

`decide_final(ws) -> Decision` — the **committee's weighted-MAJORITY lean** is the verdict (data-grounded
seats outweigh the world-knowledge one; a Build must be earned: a real majority AND no challenge left
standing, else it degrades to Conditional). No model, no signal — the deliberation IS the decision.

`compute_anchor(snap) -> Anchor` remains for the OFFLINE backtest tooling only (`harness.py` grades the
leakage-clean decider out-of-fold there); the live committee no longer computes or displays it — per Dhruv,
the product is the council alone. Self-contained: imports only council modules.
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
    numbers: Dict[str, Any] = field(default_factory=dict)   # revenue / memberships / washcount / tunnel / capex / net / breakeven
    condition: Optional[str] = None     # the condition that would flip a Conditional
    yes_frac: Optional[float] = None    # committee weighted yes-score (0..1)
    n_votes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "confidence": round(float(self.confidence), 3), "basis": self.basis,
                "numbers": self.numbers, "condition": self.condition,
                "yes_frac": self.yes_frac, "n_votes": self.n_votes}


def _weight(expert: str) -> float:
    return C.WORLD_EXPERT_WEIGHT if expert in C.WORLD_EXPERTS else C.DATA_EXPERT_WEIGHT


def _tally(ws) -> tuple:
    """Weighted mean of experts' current leans (a 0..1 optimism scalar, used for confidence + the signal
    cross-check — NOT the verdict). Data-grounded experts outweigh world-knowledge ones. The demoted data
    signal does not vote. Returns (yes_frac 0..1 | None, n_votes)."""
    num = den = 0.0
    n_votes = 0
    for b in ws.beliefs.values():
        if b.lean is None:
            continue
        w = _weight(b.expert) * max(float(b.confidence), 0.1)
        num += w * _SCORE.get(b.lean, 0.5)
        den += w
        n_votes += 1
    return ((num / den) if den else None), n_votes


def _lean_masses(ws) -> Dict[str, float]:
    """Weighted vote-mass behind each lean. The verdict is the committee's weighted-MAJORITY lean — so one
    seat (e.g. Finance's mechanical economics lean) cannot override a room that deliberated the other way, the
    way a weighted *mean* + threshold silently could. Data-grounded seats outweigh world-knowledge ones."""
    mass: Dict[str, float] = {"Build": 0.0, "Conditional": 0.0, "Pass": 0.0}
    for b in ws.beliefs.values():
        if b.lean is None or b.lean not in mass:
            continue
        mass[b.lean] += _weight(b.expert) * max(float(b.confidence), 0.1)
    return mass


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


def decide_final(ws, *, mode: Optional[str] = None) -> Decision:
    """Turn the settled board into the committee's verdict: the weighted-MAJORITY lean decides (so one seat
    can't override the room). Deterministic, no LLM, no signal. `mode` is accepted-and-ignored (compat)."""
    numbers = _collect_numbers(ws)
    yes, n_votes = _tally(ws)
    condition: Optional[str] = None

    masses = _lean_masses(ws)
    total = sum(masses.values())
    if total <= 0:
        verdict, confidence, basis = "Insufficient", 0.2, "no expert produced a lean"
    elif n_votes < 2:
        # one lone (usually hedged) lean is not a committee decision — say "not enough signal" honestly
        verdict, confidence = "Insufficient", 0.3
        basis = "fewer than two seats could form a data-based lean (no local comparables / no local market data)"
    else:
        verdict = max(masses, key=masses.get)          # the lean the committee actually landed on
        share = masses[verdict] / total                # how dominant that majority is → confidence
        basis = "committee consensus (weighted-majority lean)"
        # a "Build" must be EARNED: a mere plurality, or challenges the room never resolved, → Conditional
        if verdict == "Build" and _open_challenge_exists(ws):
            verdict, condition = "Conditional", "resolve the open challenge(s) the committee left standing"
        elif verdict == "Build" and share < 0.5:
            verdict, condition = "Conditional", "the committee is split — no clear majority for an unconditional build"
        elif verdict == "Conditional" and _open_challenge_exists(ws):
            condition = "resolve the open challenge(s) on the board"
        confidence = round(float(0.45 + 0.4 * max(0.0, share - 0.5) * 2), 2)

    return Decision(verdict=verdict, confidence=round(float(confidence), 2), basis=basis,
                    numbers=numbers, condition=condition, yes_frac=yes, n_votes=n_votes)
