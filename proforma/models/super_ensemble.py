"""SUPER ensemble — the calibrated level layer on top of coldstart Model 3.

Two prediction paths (both return a full 60-month trajectory + bands):

  with user inputs  — pay stations, free-vacuum slots, lot type, daily traffic count:
                      ridge level_A (LOSO-estimated mature MdAPE ~29.6%, within±20% ~38.6%)
  pin only          — level_B calibration of the coldstart plateau
                      (panel-CV mature MdAPE ~31.8%)

Both paths then scale Model 3's ramp/bands to the calibrated level and apply the
per-operating-year debias constants. Artifact: proforma/artifacts/ensemble_super (model 5)/
(fitted by experiments/old-proforma-analysis/ensemble/fit_super_artifact.py — see the
README there for the model card). Loads lazily; importing this module has no side effects.

Usage:
    from proforma.models import super_ensemble as se
    traj, info = se.predict_site_super(lat, lon, open_year=2026,
                                       pay_stations="2", vacuum_slots="12 - 20",
                                       lot_type="corner lot with light", traffic_count=25000)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "ensemble_super (model 5)" / "super_ensemble_v1.joblib"
_CACHE: dict = {}

# The 3 ridge site-inputs and the score-sheet factor they correspond to. The score sheet
# (site_score_factors.FACTORS) is the single source of truth for the frontend `value` AND `label`
# spellings; the ridge maps in the artifact key on their own human-readable labels. We accept any of
# the three (ridge key / frontend value / frontend label) by matching on the shared option SCORE —
# every option's score is unique within its factor and identical across the two configs.
_RIDGE_TO_FACTOR = {"pay_map": "numberOfPayStations",
                    "vac_map": "numberOfFreeVacuumSlots",
                    "lot_map": "typeOfSite"}


def _choice_key(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _accepted_spellings(mapping: Dict[str, float], factor_name: Optional[str]) -> Dict[str, float]:
    """Every normalized spelling -> ridge score for one input: the artifact's own map keys plus, when
    the score-sheet factor is known, that factor's frontend `value` and `label` (paired to the ridge
    key by their identical score). Built per call off the (small) maps — no import-time side effects."""
    out = {_choice_key(k): v for k, v in mapping.items()}
    try:
        from proforma.models.site_score_factors import _BY_NAME
        f = _BY_NAME.get(factor_name) if factor_name else None
        if f:
            score_to_val = {round(v, 6): v for v in mapping.values()}   # ridge score -> itself
            for o in f["options"]:
                v = score_to_val.get(round(o["score"], 6))
                if v is not None:
                    out.setdefault(_choice_key(o["value"]), v)
                    out.setdefault(_choice_key(o["label"]), v)
    except Exception:                                                   # config missing -> ridge keys only
        pass
    return out


def _resolve_choice(mapping: Dict[str, float], choice: str, factor_name: Optional[str] = None) -> float:
    accepted = _accepted_spellings(mapping, factor_name)
    key = _choice_key(choice)
    if key in accepted:
        return accepted[key]
    labels = sorted({k for k in mapping})
    raise ValueError(f"unrecognized site input {choice!r}; accepted values: {labels}")


def load() -> dict:
    if "art" not in _CACHE:
        _CACHE["art"] = joblib.load(MODEL_PATH)
    return _CACHE["art"]


def _ridge_predict(layer: dict, x: np.ndarray) -> float:
    z = (x - layer["mu"]) / layer["sd"]
    return float(z @ layer["coef"] + layer["intercept"] + layer["debias"])


def level_with_inputs(plateau: float, pay_stations: str, vacuum_slots: str,
                      lot_type: str, traffic_count: float, art: Optional[dict] = None) -> float:
    """level_A: user supplies the 4 runtime inputs. Choice strings are case-insensitive
    and must match the proforma option labels (see the maps in the artifact)."""
    a = (art or load())["level_A"]
    pay = _resolve_choice(a["pay_map"], pay_stations, "numberOfPayStations")
    vac = _resolve_choice(a["vac_map"], vacuum_slots, "numberOfFreeVacuumSlots")
    lot = _resolve_choice(a["lot_map"], lot_type, "typeOfSite")
    x = np.array([np.log(plateau), pay, vac, lot, np.log(max(traffic_count, 1.0))])
    return float(np.exp(_ridge_predict(a, x)))


def level_pin_only(plateau: float, model_plateau: float, anchor_level: float,
                   n_local_mature: int, region: str, open_year: int,
                   art: Optional[dict] = None) -> float:
    """level_B: calibration of the coldstart plateau from predict_site's own info dict."""
    b = (art or load())["level_B"]
    has_anchor = float(np.isfinite(anchor_level) and anchor_level > 0)
    lanchor = np.log(max(anchor_level, 1.0)) - np.log(plateau) if has_anchor else 0.0
    x = [np.log(plateau), np.log(max(model_plateau, 1.0)), lanchor, has_anchor,
         np.log1p(max(n_local_mature, 0)), open_year - 2022.0]
    x += [1.0 if region == r else 0.0 for r in b["regions"]]
    return float(np.exp(_ridge_predict(b, np.array(x))))


def apply_super(traj, info: dict, open_year: int,
                pay_stations: Optional[str] = None, vacuum_slots: Optional[str] = None,
                lot_type: Optional[str] = None, traffic_count: Optional[float] = None,
                extra_factors: Optional[dict] = None):
    """Rescale an existing coldstart (traj, info) to the SUPER calibrated level and apply
    the per-operating-year debias. Used by predict_site_super, the Streamlit drop-pin
    panel and the API's compute_trajectory.

    `extra_factors` ({factor: option} from the client's proforma score sheet) applies the bounded
    relative-bucketing level multiplier on TOP of the calibrated level — see
    proforma/models/site_score_factors.py. The three sheet factors that are already ridge inputs
    are skipped there, never double counted. info gains `factor_adjustment` when supplied."""
    art = load()
    plateau = info["plateau_med"]
    have_inputs = all(v is not None for v in (pay_stations, vacuum_slots, lot_type, traffic_count))
    if have_inputs:
        level = level_with_inputs(plateau, pay_stations, vacuum_slots, lot_type, traffic_count, art)
        src = "user_inputs"
    else:
        level = level_pin_only(plateau, info["model_plateau"], info["anchor_level"],
                               info["n_local_mature"], info["region"], open_year, art)
        src = "pin_only"
    fadj = None
    fmult = 1.0
    if extra_factors:
        from proforma.models.site_score_factors import factor_adjustment
        fadj = factor_adjustment(extra_factors)
        fmult = float(fadj["multiplier"])
        level *= fmult
    scale = level / max(plateau, 1e-9)
    out = traj.copy()
    cy = art["year_debias"]
    op_year = (out["month"] // 12 + 1).clip(upper=5)
    adj = op_year.map(lambda y: float(np.exp(cy.get(int(y), cy.get(str(int(y)), 0.0))))) * scale
    for c in [c for c in out.columns if c != "month"]:
        out[c] = out[c] * adj
    info = dict(info, level_source=src, super_level=level, super_scale=scale,
                super_version=art["version"],
                plateau_med=level, plateau_lo=info["plateau_lo"] * scale,
                plateau_hi=info["plateau_hi"] * scale)
    if fadj is not None:
        info["factor_adjustment"] = fadj
    return out, info


def predict_site_super(lat: float, lon: float, open_year: int,
                       pay_stations: Optional[str] = None, vacuum_slots: Optional[str] = None,
                       lot_type: Optional[str] = None, traffic_count: Optional[float] = None,
                       extra_factors: Optional[dict] = None,
                       coldstart_art: Optional[dict] = None, **predict_site_kwargs):
    """Full trajectory: coldstart Model 3 -> calibrated level -> per-year debias
    (-> bounded score-sheet factor multiplier when `extra_factors` is supplied).

    Returns (traj_df, info) like coldstart.predict_site; info gains
    level_source ('user_inputs'|'pin_only'), super_level, rescaled plateau fields,
    and `factor_adjustment` when extra_factors were given.
    """
    from proforma.models import coldstart as cm
    traj, info = cm.predict_site(lat, lon, art=coldstart_art,
                                 model_kind=predict_site_kwargs.pop("model_kind", "et"),
                                 **predict_site_kwargs)
    return apply_super(traj, info, open_year, pay_stations, vacuum_slots, lot_type, traffic_count,
                       extra_factors=extra_factors)

