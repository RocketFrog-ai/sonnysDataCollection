"""Back-compat shim. The pnl-only entrypoint now lives at app/pnl_only.py.

    uvicorn app.pnl_only:app --host 127.0.0.1 --port 8010   # preferred
    uvicorn serve_pnl:app    --host 127.0.0.1 --port 8010   # still works, via this module

Kept because `serve_pnl:app` may be baked into deploy scripts, supervisor units, or someone's
shell history. It re-exports the same FastAPI instance -- not a copy -- so both names serve the
identical app object. Delete once nothing outside this repo names it.
"""
from app.pnl_only import app  # noqa: F401
