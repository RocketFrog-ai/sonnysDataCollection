"""Capture golden outputs for the deterministic /v1/pnl_analysis/* endpoints.

    python scripts/_golden/capture_api.py <out_dir>

Runs the ASGI app in-process via httpx's ASGITransport -- no port, no uvicorn, no network.
Requires the conda `sonnys` env.

DELIBERATELY EXCLUDED: every /insights/* route. Those call an LLM, are non-deterministic
by construction, and are documented as annotating rather than altering modelled numbers.
Golden-testing them would produce a flaky baseline that fails for the wrong reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import PINS, load_pnl_app  # noqa: E402

P = "/v1/pnl_analysis"
LAT, LON = PINS[0][1], PINS[0][2]  # dense Houston pin: exercises neighbours + cannibalization
LAT2, LON2 = PINS[2][1], PINS[2][2]  # isolated pin: empty-market edge cases

# (case_name, method, path, payload_or_None)
CASES = [
    ("brands", "GET", f"{P}/brands", None),
    ("operators", "GET", f"{P}/operators", None),
    ("campaign_snapshot", "GET", f"{P}/campaign/snapshot", None),
    # express_only is pinned False here: the Explore-markets schemas default it to True (2026-07,
    # express-first exploration), but these goldens freeze the all-sites numbers captured before the flip.
    ("explore_market", "POST", f"{P}/explore-market",
     dict(latitude=LAT, longitude=LON, radius_km=20.0, max_sites=10, min_months=36, demo=False,
          express_only=False)),
    ("explore_market_isolated", "POST", f"{P}/explore-market",
     dict(latitude=LAT2, longitude=LON2, radius_km=20.0, max_sites=10, min_months=36, demo=False,
          express_only=False)),
    ("explore_market_kpis", "POST", f"{P}/explore-market/kpis",
     dict(latitude=LAT, longitude=LON, radius_km=20.0, smoothing=3, min_months=36, demo=False,
          express_only=False)),
    ("site_factors", "POST", f"{P}/site-factors",
     dict(latitude=LAT, longitude=LON)),
    ("site_factors_no_coverage", "POST", f"{P}/site-factors",
     dict(latitude=44.5, longitude=-110.4)),  # Yellowstone: nothing within 9 miles
    ("pinpoint_forecast", "POST", f"{P}/pinpoint-forecast",
     dict(latitude=LAT, longitude=LON, horizon_months=60)),
    ("pinpoint_forecast_override", "POST", f"{P}/pinpoint-forecast",
     dict(latitude=LAT, longitude=LON, plateau_override=50000.0, mem_growth_pct=3.0,
          ret_growth_pct=-2.0, horizon_months=60)),
    ("market_forecast", "POST", f"{P}/market-forecast",
     dict(latitude=LAT, longitude=LON, horizon_months=60)),
    ("pnl_forecast", "POST", f"{P}/pnl-forecast",
     dict(latitude=LAT, longitude=LON, horizon_months=60, campaign_on=False)),
    ("pnl_forecast_campaign", "POST", f"{P}/pnl-forecast",
     dict(latitude=LAT, longitude=LON, horizon_months=60, campaign_on=True,
          campaign_launch=13, campaign_intensity=1.0, window=6, asp_override=12.5)),
    ("expense_plan", "POST", f"{P}/expense-plan",
     dict(latitude=LAT, longitude=LON, horizon_months=60,
          asp={"1": 12.5, "2": 13.0}, opex={"1": 60.0, "2": 50.0, "3": 45.0},
          capex={"1": 500000.0, "2": 100000.0}, opex_growth_pct=2.0)),
    ("campaign_verdict", "POST", f"{P}/campaign/verdict",
     dict(latitude=LAT, longitude=LON, radius_km=20.0)),
    ("campaign_eating_market", "POST", f"{P}/campaign/eating-the-market",
     dict(latitude=LAT, longitude=LON, radius_km=20.0, campaign_on=True,
          campaign_launch=13, campaign_intensity=1.0, window=6, max_incumbents=6)),
    ("campaign_local_evidence", "POST", f"{P}/campaign/local-evidence",
     dict(latitude=LAT, longitude=LON, radius_km=20.0, metric="mem_share_wash", max_sites=8, demo=False)),
]


def main(out_dir: str) -> None:
    # Drive the ASGI app in-process. httpx>=0.28 removed starlette TestClient's `app=`
    # shortcut, so go through ASGITransport directly -- version-proof and no event loop
    # juggling beyond the one anyio call.
    import httpx

    app, origin = load_pnl_app()
    out = {"_origin": origin, "cases": {}}

    async def run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://golden", timeout=600.0) as client:
            for name, method, path, payload in CASES:
                r = await (client.get(path) if method == "GET" else client.post(path, json=payload))
                try:
                    body = r.json()
                except Exception:
                    body = {"__unparseable__": r.text[:500]}
                out["cases"][name] = {"status": r.status_code, "body": body}
                print(f"  [{r.status_code}] {method:4s} {path}  ({name})")

    import anyio

    anyio.run(run)

    dest = Path(out_dir) / "api.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    bad = [k for k, v in out["cases"].items() if v["status"] != 200]
    print(f"[capture_api] {len(out['cases'])} cases -> {dest}  (origin: {origin})")
    if bad:
        print(f"[capture_api] WARNING non-200: {bad}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/_golden/baseline")
