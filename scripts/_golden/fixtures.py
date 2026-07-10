"""Fixed inputs for the refactor golden-output capture. NEVER edit during the refactor.

The three pins are real sites drawn from the canonical panel, chosen to span the
`predict_site` local-anchor regimes (that branch is the one most likely to break
when the module moves and its HERE-relative data paths are rewritten):

  dense    43 neighbours within 20 km -> anchor fires, CoV guard active
  medium    3 neighbours              -> exactly at anchor_min_n, boundary case
  isolated  0 neighbours              -> anchor suppressed, brand_loo stays NaN

Also resolves the two things the refactor relocates, tolerating either layout so
one harness can capture the baseline AND verify the result:
  coldstart model  earnest-proforma-2.0/streamlits/coldstart_model.py -> proforma/models/coldstart.py
  ASGI app         serve_pnl:app -> app/pnl_only.py:app -> app/main.py:app (entrypoints collapsed)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (name, lat, lon) — real panel sites; see module docstring for why these three.
PINS = [
    ("dense_houston_tx", 29.798555, -95.719688),
    ("medium_lexington_ky", 38.051504, -84.715945),
    ("isolated_santa_maria_ca", 34.913719, -120.435347),
]


def load_coldstart():
    """Return (module, origin_label). Prefers the post-refactor location.

    Falls back to the legacy module ONLY when the new package genuinely does not exist on
    disk. A broken new module must raise, not silently fall through -- otherwise the smoke
    test would keep passing against the old code while the new code is broken.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    if (REPO / "proforma" / "models" / "coldstart.py").is_file():
        from proforma.models import coldstart as cm  # noqa

        return cm, "proforma.models.coldstart"

    legacy = REPO / "earnest-proforma-2.0" / "streamlits"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    import coldstart_model as cm  # noqa

    return cm, "coldstart_model (legacy streamlits/)"


def load_pnl_app():
    """Return (fastapi_app, origin_label). Falls back on ABSENCE, never on breakage."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    # Older layouts are reached via importlib, not a static `import`, so the repo-wide import
    # resolver does not flag modules that only exist at an older tag.
    import importlib

    for path, mod in (("app/main.py", "app.main"),
                      ("app/pnl_only.py", "app.pnl_only"),      # before the entrypoints were collapsed
                      ("serve_pnl.py", "serve_pnl")):           # tag: pre-refactor
        if (REPO / path).is_file():
            return importlib.import_module(mod).app, f"{mod}:app"

    raise ImportError("no FastAPI entrypoint found (looked for app/main.py, app/pnl_only.py, serve_pnl.py)")
