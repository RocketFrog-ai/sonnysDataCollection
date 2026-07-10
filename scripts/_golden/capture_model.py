"""Capture golden outputs for the cold-start model. Deterministic: no fit, no RNG.

    python scripts/_golden/capture_model.py <out_dir>

Writes <out_dir>/model.json. Floats are written via Python's json (repr round-trip,
so a reload is bit-exact); NaN/Infinity are emitted as literals and read back with
the same json module, which accepts them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import PINS, load_coldstart  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def enc(obj):
    """Losslessly encode DataFrames / numpy scalars into plain JSON structures."""
    if isinstance(obj, pd.DataFrame):
        return {
            "__df__": True,
            "columns": [str(c) for c in obj.columns],
            "index": [enc(i) for i in obj.index.tolist()],
            "records": [[enc(v) for v in row] for row in obj.itertuples(index=False, name=None)],
        }
    if isinstance(obj, pd.Series):
        return {"__series__": True, "index": [enc(i) for i in obj.index.tolist()], "values": [enc(v) for v in obj.tolist()]}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [enc(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): enc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [enc(v) for v in obj]
    return obj


def main(out_dir: str) -> None:
    cm, origin = load_coldstart()

    # The module resolves both of these relative to its own __file__. Moving the module
    # without fixing the depth leaves CSV/MODEL_PATH pointing at nothing -- and `load()`
    # would then raise, or worse, a future `fit()` would silently write to a new location.
    assert Path(cm.CSV).is_file(), f"coldstart CSV does not resolve: {cm.CSV}"
    assert Path(cm.MODEL_PATH).is_file(), f"coldstart artifact does not resolve: {cm.MODEL_PATH}"

    art = cm.load()

    # The pickled Model-3 ExtraTreesRegressor carries n_jobs=-1. sklearn's forest predict
    # sums each tree's contribution into a shared buffer from a joblib thread pool, so the
    # summation ORDER varies run to run and the last bits of the float64 wander. Measured
    # spread: 1.9e-15 relative, ~9x machine epsilon, confined to the model_kind="et" cases.
    # (OMP_NUM_THREADS does not fix this -- it governs BLAS/OpenMP, not joblib's pool.)
    #
    # Pin it to 1 for capture ONLY, so the golden file is byte-stable and an empty diff is
    # a real signal rather than a coin flip. This does not touch the artifact on disk and
    # does not change production, which keeps n_jobs=-1. The 1e-9 tolerance in diff.py sits
    # far above this noise either way; the pin just removes churn.
    et = art.get("models", {}).get("et")
    if et is not None and getattr(et, "n_jobs", None) != 1:
        et.n_jobs = 1

    # a real brand id, chosen deterministically so the brand-known branch is exercised
    brand = sorted(art["brand_mean"].keys())[0]

    out = {
        "_origin": origin,
        "_constants": {
            "ANCHOR_CALIB_W": cm.ANCHOR_CALIB_W,
            "H_DEFAULT": cm.H_DEFAULT,
            "MODEL_MIN_MONTHS": cm.MODEL_MIN_MONTHS,
            "FEAT": list(cm.FEAT),
            "CAT": list(cm.CAT),
        },
        # Recorded for the human reader. NOT relied on for the diff: `_` keys are
        # informational there, so the hard check is the assert below.
        "_paths_exist": {
            "CSV": Path(cm.CSV).is_file(),
            "MODEL_PATH": Path(cm.MODEL_PATH).is_file(),
        },
        "_artifact_keys": sorted(art.keys()),
        "_brand_used": brand,
        "cases": {},
    }

    variants = [
        ("default", dict()),
        ("brand_known", dict(brand=brand)),
        ("plateau_override", dict(plateau_override=50000.0)),
        ("model_kind_et", dict(model_kind="et")),
        ("no_local_anchor", dict(local_anchor=False)),
        ("drift", dict(annual_mem_growth=0.03, annual_ret_change=-0.02)),
    ]

    for pin_name, lat, lon in PINS:
        for vname, kw in variants:
            key = f"{pin_name}::{vname}"
            out["cases"][key] = enc(cm.predict_site(lat, lon, art=art, **kw))
        out["cases"][f"{pin_name}::neighbours"] = enc(cm.predict_neighbours(lat, lon, art=art))
        out["cases"][f"{pin_name}::cannib_params"] = enc(cm.cannib_params(art, lat, lon))

    dest = Path(out_dir) / "model.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"[capture_model] {len(out['cases'])} cases -> {dest}  (origin: {origin})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/_refactor/baseline")
