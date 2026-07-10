"""The FastAPI backend. One app, one entrypoint.

    python -m app.main                              # host/port from FAST_API_HOST / FAST_API_PORT
    uvicorn app.main:app --host 127.0.0.1 --port 8010

Serves the Explore-markets + Forecast API under /v1/pnl_analysis/... — the same math the Streamlit
app runs in-process (see docs/DIVERGENCES.md §1: it is a second implementation, not shared code).

History, so nobody goes looking: there were two entrypoints (app/main.py + app/pnl_only.py) because
`main` also mounted a `site_analysis` router with a heavier dependency footprint. Celery, that router
and its whole subsystem were removed in 2026-07 — nothing rendered them — so the two entrypoints
became the same app and were collapsed into this one.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import common as calib
from app.server.router import router as pnl_analysis_router

FAST_API_HOST = calib.FAST_API_HOST
FAST_API_PORT = calib.FAST_API_PORT

app = FastAPI(title="Earnest Proforma backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pnl_analysis_router, prefix="/v1")


@app.get("/")
async def root():
    """Root endpoint with basic service information"""
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host=FAST_API_HOST, port=int(FAST_API_PORT))
