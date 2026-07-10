"""Fixed inputs for the refactor golden-output capture. NEVER edit during the refactor.

The three pins are real sites drawn from the canonical panel, chosen to span the
`predict_site` local-anchor regimes (that branch is the one most likely to break
when the module moves and its HERE-relative data paths are rewritten):

  dense    43 neighbours within 20 km -> anchor fires, CoV guard active
  medium    3 neighbours              -> exactly at anchor_min_n, boundary case
  isolated  0 neighbours              -> anchor suppressed, brand_loo stays NaN

Also resolves the two things the refactor relocates, tolerating either layout so
one harness can capture the baseline AND verify the result:
  coldstart model  earnest-proforma-2.0/streamlits/coldstart_model.py -> proforma/v1_5/models/coldstart.py
  pnl-only ASGI    serve_pnl:app                                      -> app/pnl_only.py:app
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
    """Return (module, origin_label). Prefers the post-refactor location."""
    try:
        from proforma.v1_5.models import coldstart as cm  # noqa

        return cm, "proforma.v1_5.models.coldstart"
    except Exception:
        pass
    legacy = REPO / "earnest-proforma-2.0" / "streamlits"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    import coldstart_model as cm  # noqa

    return cm, "coldstart_model (legacy streamlits/)"


def load_pnl_app():
    """Return (fastapi_app, origin_label). Prefers the post-refactor entrypoint."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from app.pnl_only import app  # noqa

        return app, "app.pnl_only:app"
    except Exception:
        pass
    from serve_pnl import app  # noqa

    return app, "serve_pnl:app (legacy)"
