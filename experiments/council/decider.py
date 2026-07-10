"""
The SIGNAL decider — a small, calibrated, leakage-clean classifier that actually drives the build/pass
decision (the LLM seats are demoted to explanation only, because they carry no discriminating signal).

Model: median-impute → standardize → logistic regression over the market-structure + operator features in
`features.FEATURES`. Honest evaluation is OUT-OF-FOLD with GroupKFold grouped by OPERATOR (an operator's
sites never straddle train/test), so the reported edge can't be operator-identity leakage.

We cache only the FEATURE MATRIX (a portable CSV) and fit the tiny model IN-PROCESS on load — never pickle
the sklearn estimator — so it works identically under the venv (sklearn 1.8) and the conda env (1.6.1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.council import data_1_6 as D
from experiments.council import features as F

# Was D.ROOT / "council" / "outputs" -- i.e. repo-root-relative. The outputs are owned by this
# version, so anchor them on this module instead of borrowing another module's root constant.
MATRIX_CSV = Path(__file__).resolve().parent / "outputs" / "decider_matrix.csv"
_CACHE: Dict[str, Any] = {}


def _pipe():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def get_matrix(*, rebuild: bool = False) -> pd.DataFrame:
    """Feature+label matrix for all ~420 focal sites, cached to a portable CSV (offline, no LLM)."""
    if "m" in _CACHE and not rebuild:
        return _CACHE["m"]
    if MATRIX_CSV.exists() and not rebuild:
        m = pd.read_csv(MATRIX_CSV)
    else:
        m = F.build_matrix()
        MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(MATRIX_CSV, index=False)
    _CACHE["m"] = m
    return m


# ─────────────────────────── honest evaluation ───────────────────────────
def oof_probs(m: pd.DataFrame, *, kind: str = "group") -> np.ndarray:
    """Out-of-fold P(good build). kind='group' → GroupKFold by operator (honest); 'strat' → StratifiedKFold."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    X, y = m[F.FEATURES].values, m["y"].values
    if kind == "group":
        cv, groups = GroupKFold(n_splits=5), m["group"].values
        return cross_val_predict(_pipe(), X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return cross_val_predict(_pipe(), X, y, cv=cv, method="predict_proba")[:, 1]


def evaluate(*, rebuild: bool = False) -> Dict[str, Any]:
    """Rigorous offline eval: OOF AUC (operator-grouped + stratified), top-K build precision vs base rate,
    and a label-permutation baseline (the real ceiling for 'is this signal or noise')."""
    from sklearn.metrics import roc_auc_score
    m = get_matrix(rebuild=rebuild)
    y = m["y"].values
    base = float(y.mean())
    pg, ps = oof_probs(m, kind="group"), oof_probs(m, kind="strat")
    auc_g, auc_s = float(roc_auc_score(y, pg)), float(roc_auc_score(y, ps))

    # build-decision value: precision (good-build rate) among the top-scored sites
    topk = {}
    order = np.argsort(-pg)
    for pct in (20, 30, 40):
        k = max(1, int(len(y) * pct / 100))
        topk[pct] = float(y[order[:k]].mean())

    # permutation baseline — shuffle labels, same pipeline; p95 of shuffled AUC is the noise ceiling
    rng = np.random.RandomState(0)
    perm = []
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    for _ in range(20):
        ysh = rng.permutation(y)
        p = cross_val_predict(_pipe(), m[F.FEATURES].values, ysh,
                              cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
        perm.append(roc_auc_score(ysh, p))
    return {
        "n": int(len(y)), "base_rate": base, "auc_group": auc_g, "auc_strat": auc_s,
        "topk_precision": topk, "perm_auc_mean": float(np.mean(perm)), "perm_auc_p95": float(np.percentile(perm, 95)),
        "auc_matured_subset": _auc_on_subset(m, roc_auc_score),
    }


def _auc_on_subset(m: pd.DataFrame, roc_auc_score) -> Optional[float]:
    """AUC on the sites the signal is actually informative for — those with ≥2 already-matured neighbours."""
    sub = m[m["n_matured_pre_nbrs"] >= 2]
    if len(sub) < 40 or sub["y"].nunique() < 2:
        return None
    return float(roc_auc_score(sub["y"].values, oof_probs(sub, kind="strat")))


# ─────────────────────────── fit + decide (for live use) ───────────────────────────
def load_decider(*, rebuild: bool = False) -> Dict[str, Any]:
    """Fit the pipeline on ALL sites (in-process; never pickled) + derive decision thresholds so that the
    top ~35% by score → Build and the bottom ~35% → Pass. Cached per process."""
    if "decider" in _CACHE and not rebuild:
        return _CACHE["decider"]
    m = get_matrix(rebuild=rebuild)
    pipe = _pipe().fit(m[F.FEATURES].values, m["y"].values)
    probs = pipe.predict_proba(m[F.FEATURES].values)[:, 1]
    dec = {"pipe": pipe, "p_build": float(np.quantile(probs, 0.65)),
           "p_pass": float(np.quantile(probs, 0.35)), "base_rate": float(m["y"].mean())}
    _CACHE["decider"] = dec
    return dec


def score_features(feat: Dict[str, Any]) -> Optional[float]:
    """P(good build) for one feature dict (live). None if the model can't be built."""
    try:
        dec = load_decider()
        x = np.array([[feat.get(k, np.nan) for k in F.FEATURES]], dtype=float)
        return float(dec["pipe"].predict_proba(x)[:, 1][0])
    except Exception:
        return None


def decide(prob: Optional[float]) -> Tuple[Optional[str], float]:
    """Map P(good build) → (lean, confidence). Build above the top-35% cut, Pass below the bottom-35% cut."""
    if prob is None or not np.isfinite(prob):
        return None, 0.2
    dec = load_decider()
    lean = "Build" if prob >= dec["p_build"] else ("Pass" if prob <= dec["p_pass"] else "Conditional")
    # confidence = how far from the base rate, scaled
    conf = min(0.9, 0.4 + abs(prob - dec["base_rate"]) * 2.5)
    return lean, round(conf, 2)
