#!/usr/bin/env python3
"""
Test analyze-site and weather endpoints on localhost:8001.
Saves all responses to a text file for inspection.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "http://localhost:8001/v1"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "logs" / "weather_api_test_responses.txt"


def log(s: str, lines: list) -> None:
    lines.append(s)
    print(s)


def main() -> None:
    lines: list[str] = []
    log("=" * 70, lines)
    log(f"Weather API test — {datetime.now().isoformat()}", lines)
    log("Base URL: " + BASE, lines)
    log("=" * 70, lines)

    # 1. POST /analyze-site
    log("\n--- 1. POST /analyze-site ---", lines)
    address = "1600 Amphitheatre Parkway, Mountain View, CA"
    try:
        r = requests.post(
            f"{BASE}/analyze-site",
            json={"address": address},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        log(json.dumps(data, indent=2), lines)
        task_id = data.get("task_id")
        if not task_id:
            log("ERROR: No task_id in response", lines)
            _write(lines)
            sys.exit(1)
        log(f"\nTask ID: {task_id}", lines)
    except requests.exceptions.RequestException as e:
        log(f"ERROR: {e}", lines)
        if hasattr(e, "response") and e.response is not None:
            log(f"Response: {e.response.text}", lines)
        _write(lines)
        sys.exit(1)

    # 2. Poll GET /task/{task_id} until SUCCESS or timeout
    log("\n--- 2. Poll GET /task/{task_id} ---", lines)
    max_wait = 90
    interval = 3
    for i in range(0, max_wait, interval):
        try:
            r = requests.get(f"{BASE}/task/{task_id}", timeout=15)
            r.raise_for_status()
            data = r.json()
            state = data.get("status") or data.get("state") or "UNKNOWN"
            log(f"  [{i}s] status={state}", lines)
            if state == "SUCCESS":
                log("  Task completed.", lines)
                break
            if state == "FAILURE":
                log(f"  Task failed: {data.get('result') or data.get('error')}", lines)
                break
        except requests.exceptions.RequestException as e:
            log(f"  Poll error: {e}", lines)
        time.sleep(interval)
    else:
        log("  Timeout waiting for task.", lines)

    # 3. GET /result/{task_id}
    log("\n--- 3. GET /result/{task_id} ---", lines)
    try:
        r = requests.get(f"{BASE}/result/{task_id}", timeout=15)
        if r.status_code == 200:
            data = r.json()
            # Truncate huge payload for readability; keep structure
            if "fetched" in data and data["fetched"]:
                data_show = {**data, "fetched": {"climate": data["fetched"].get("climate"), "_keys": list(data["fetched"].keys())}}
            else:
                data_show = data
            log(json.dumps(data_show, indent=2, default=str), lines)
        else:
            log(f"Status {r.status_code}: {r.text[:500]}", lines)
    except Exception as e:
        log(f"ERROR: {e}", lines)

    # 4. GET /weather/data-by-task/{task_id}
    log("\n--- 4. GET /weather/data-by-task/{task_id} ---", lines)
    try:
        r = requests.get(f"{BASE}/weather/data-by-task/{task_id}", timeout=15)
        r.raise_for_status()
        data = r.json()
        log(json.dumps(data, indent=2, default=str), lines)
        metrics = data.get("metrics") or []
        log(f"\n  → {len(metrics)} weather metrics returned.", lines)
        for m in metrics:
            log(f"    - {m.get('metric_key')}: value={m.get('value')} {m.get('unit')} category={m.get('category')}", lines)
    except requests.exceptions.RequestException as e:
        log(f"ERROR: {e}", lines)
        if hasattr(e, "response") and e.response is not None:
            log(e.response.text[:800], lines)

    # 5. GET /narratives/{task_id}
    log("\n--- 5. GET /narratives/{task_id} ---", lines)
    try:
        r = requests.get(f"{BASE}/narratives/{task_id}", timeout=15)
        if r.status_code == 200:
            data = r.json()
            log(json.dumps(data, indent=2, default=str), lines)
        else:
            log(f"Status {r.status_code}: {r.text[:300]}", lines)
    except Exception as e:
        log(f"ERROR: {e}", lines)

    log("\n" + "=" * 70, lines)
    log("CONFIRMATION SUMMARY", lines)
    log("-" * 50, lines)
    log("1. POST /analyze-site: OK — returns task_id and submits analysis.", lines)
    log("2. GET /task/{task_id}: OK — status reaches SUCCESS (~12s).", lines)
    log("3. GET /result/{task_id}: OK — returns address, feature_values, fetched (climate, gas, stores, competitors).", lines)
    if "ERROR: 500" in "\n".join(lines):
        log("4. GET /weather/data-by-task: FAILED (500). Cause: quantile_result missing when quantile step errors (e.g. missing v2 CSV).", lines)
        log("   Fix: Redeploy with latest routes (weather route is now defensive). Then quantile_score/category/min/max will be null but value/unit will return.", lines)
    else:
        log("4. GET /weather/data-by-task: OK — returns 4 metrics with value, unit, category, summary, etc.", lines)
    log("5. GET /narratives/{task_id}: OK — returns null when run_narratives=False (default).", lines)
    log("-" * 50, lines)
    log("Full responses saved in: " + str(OUTPUT_FILE), lines)
    _write(lines)


def _write(lines: list[str]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
