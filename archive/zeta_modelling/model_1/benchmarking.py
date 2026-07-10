from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def load_model1_benchmark_metrics(
    data_1_dir: Path | None = None,
    phase3_report_path: Path | None = None,
) -> pd.DataFrame:
    """
    Aggregate offline benchmark metrics from `data_1` JSON artifacts (Phase 2 variants + Phase 3 backtest).
    Paths default to `zeta_modelling/data_1/` under the repo root.
    """
    root = _repo_root_from_here()
    d1 = data_1_dir or (root / "zeta_modelling" / "data_1")
    report = phase3_report_path or (d1 / "phase3_advanced_report.json")

    rows: list[dict[str, object]] = []

    p2 = d1 / "phase2_metrics_2024_2025.json"
    if p2.exists():
        payload = json.loads(p2.read_text())
        for m in payload.get("metrics", []):
            rows.append(
                {
                    "stage": "Phase 2 baseline",
                    "model": m["model_name"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "notes": "Time split: train before 2025-01, test from 2025-01.",
                }
            )

    p2u = d1 / "phase2_no_lag_upgrade_metrics.json"
    if p2u.exists():
        payload = json.loads(p2u.read_text())
        for key, label in (
            ("baseline_no_lag", "No-lag baseline (core features)"),
            ("upgraded_no_lag", "No-lag upgraded (cluster pseudo-lags + trend)"),
            ("warm_model_with_true_lags", "Warm model (true site lags — benchmark only, leakage)"),
        ):
            block = payload.get(key)
            if isinstance(block, dict) and "mae" in block:
                rows.append(
                    {
                        "stage": "Phase 2 no-lag upgrade",
                        "model": label,
                        "mae": block["mae"],
                        "rmse": block["rmse"],
                        "notes": "Same split as phase2_metrics; tuned grid on no-lag path.",
                    }
                )

    p2d = d1 / "phase2_deployable_multiplier_metrics.json"
    if p2d.exists():
        payload = json.loads(p2d.read_text())
        full = payload.get("metrics_full_test", {})
        cold = payload.get("metrics_cold_start_early_months", {})
        if full:
            rows.append(
                {
                    "stage": "Phase 2 deployable",
                    "model": "Multiplier + inferred site_type (full test)",
                    "mae": full.get("mae"),
                    "rmse": full.get("rmse"),
                    "notes": payload.get("model_type", ""),
                }
            )
        if cold and cold.get("rows", 0):
            rows.append(
                {
                    "stage": "Phase 2 deployable",
                    "model": "Cold-start sim (held-out sites, age ≤ 6 mo)",
                    "mae": cold.get("mae_early_months"),
                    "rmse": cold.get("rmse_early_months"),
                    "notes": f"n_rows={cold.get('rows')}",
                }
            )

    p2s = d1 / "phase2_site_profile_multiplier_metrics.json"
    if p2s.exists():
        payload = json.loads(p2s.read_text())
        mapping = [
            ("baseline_no_lag_absolute_target", "Site profile: baseline absolute target"),
            ("upgraded_no_lag_multiplier_target", "Site profile: upgraded multiplier target"),
            ("warm_lag_absolute_target", "Site profile: warm absolute + lags"),
        ]
        for key, label in mapping:
            block = payload.get(key)
            if isinstance(block, dict) and "mae" in block:
                rows.append(
                    {
                        "stage": "Phase 2 site profile",
                        "model": label,
                        "mae": block["mae"],
                        "rmse": block["rmse"],
                        "notes": "Train-only site behavior stats for typing.",
                    }
                )

    if report.exists():
        payload = json.loads(report.read_text())
        bt = payload.get("backtest", {})
        if bt and "mae" in bt:
            rows.append(
                {
                    "stage": "Phase 3 (production forecaster)",
                    "model": "Quantile multiplier + site-type blend (P50 backtest)",
                    "mae": bt["mae"],
                    "rmse": bt["rmse"],
                    "notes": f"P10–P90 coverage={bt.get('p10_p90_coverage', 'n/a')} on sampled test rows.",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mae"] = pd.to_numeric(out["mae"], errors="coerce")
    out["rmse"] = pd.to_numeric(out["rmse"], errors="coerce")
    return out


def default_volume_sample_path(data_1_dir: Path | None = None) -> Path:
    root = _repo_root_from_here()
    d1 = data_1_dir or (root / "zeta_modelling" / "data_1")
    return d1 / "model1_benchmark_volume_sample.csv"
