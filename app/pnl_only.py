"""Lightweight API entrypoint — mounts ONLY the pnl_analysis router (Explore-markets + Forecast).

Mirrors app/site_analysis/server/main.py's mounting/CORS, but skips the site_analysis router so the
server doesn't require its heavy deps (openai, the live feature fetchers). Same /v1/pnl_analysis/...
paths as main.py.

    uvicorn app.pnl_only:app --host 127.0.0.1 --port 8010
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.pnl_analysis.server.router import router as pnl_analysis_router

app = FastAPI(title="Earnest Proforma backend (pnl_analysis)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pnl_analysis_router, prefix="/v1")


@app.get("/")
async def root():
    return {"status": "running"}
