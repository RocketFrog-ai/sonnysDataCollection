"""Generate TEST_SPLIT_HIT_MISS_REPORT.md from test-split predictions only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib

import zeta_modelling.model_2.cohort_forecast_pipeline as m2_pipe
from zeta_modelling.model_2.cohort_forecast_pipeline import (
    _model2_artifact_path,
    _predict_less_volume_mixed,
    _predict_lgbm_volume,
    add_less_features,
    add_more_features,
    add_site_type_feature,
    assign_dbscan_clusters,
    infer_site_type_proba,
    prepare_less,
    prepare_more,
)


def _load_model2_artifacts(model_save_dir: Path) -> dict:
    path = _model2_artifact_path(model_save_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    import sys

    main_backup = sys.modules.get("__main__")
    sys.modules["__main__"] = m2_pipe
    try:
        return joblib.load(path)
    finally:
        if main_backup is not None:
            sys.modules["__main__"] = main_backup


BUFFER = 25_000
OUT = Path(__file__).resolve().parent / "TEST_SPLIT_HIT_MISS_REPORT.md"


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):,.0f}"


def hit_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    err = np.abs(actual - predicted)
    hits = int((err <= BUFFER).sum())
    n = len(actual)
    return {
        "hits": hits,
        "n": n,
        "accuracy_pct": 100 * hits / n if n else 0.0,
        "mae": float(mean_absolute_error(actual, predicted)) if n else 0.0,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))) if n else 0.0,
        "median_abs_err": float(np.median(err)) if n else 0.0,
    }


def site_table(df: pd.DataFrame) -> str:
    out = df.copy()
    out["site_id"] = out["site_id"].astype(str)
    out["actual"] = out["actual"].map(fmt_num)
    out["predicted"] = out["predicted"].map(fmt_num)
    out["abs_error"] = out["abs_error"].map(fmt_num)
    cols = ["site_id", "n_test_months", "actual", "predicted", "abs_error", "result"]
    if "cohort" in out.columns:
        cols = ["site_id", "cohort", "n_test_months", "actual", "predicted", "abs_error", "result"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in out.iterrows()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    panel = pd.read_csv(_REPO / "zeta_modelling/data_1/phase1_final_monthly_2024_2025.csv")
    panel["date"] = pd.to_datetime(panel["date"])
    panel_train = panel[panel["date"] < pd.Timestamp("2025-01-01")]
    panel_test = panel[panel["date"] >= pd.Timestamp("2025-01-01")]

    bench = pd.read_csv(_REPO / "zeta_modelling/data_1/model1_benchmark_volume_sample.csv")
    bench["date"] = pd.to_datetime(bench["date"])
    n_train_in_bench = int((bench["date"] < pd.Timestamp("2025-01-01")).sum())
    if n_train_in_bench:
        raise ValueError(f"benchmark CSV contains {n_train_in_bench} train-period rows")
    if len(bench) != len(panel_test):
        raise ValueError(f"benchmark rows {len(bench)} != panel test rows {len(panel_test)}")

    m1 = (
        bench.groupby("site_id", as_index=False)
        .agg(
            actual=("y_actual", "sum"),
            predicted=("pred_phase3_p50", "sum"),
            n_test_months=("date", "count"),
        )
        .sort_values("site_id")
    )
    m1["abs_error"] = (m1["predicted"] - m1["actual"]).abs()
    m1["hit"] = m1["abs_error"] <= BUFFER
    m1["result"] = np.where(m1["hit"], "Hit", "Miss")

    m1_all = hit_metrics(m1["actual"].to_numpy(), m1["predicted"].to_numpy())
    m1_full12 = m1[m1["n_test_months"] == 12]
    m1_partial = m1[m1["n_test_months"] < 12]
    m1_f12 = hit_metrics(m1_full12["actual"].to_numpy(), m1_full12["predicted"].to_numpy())
    m1_part = hit_metrics(m1_partial["actual"].to_numpy(), m1_partial["predicted"].to_numpy())
    m1_monthly = hit_metrics(bench["y_actual"].to_numpy(), bench["pred_phase3_p50"].to_numpy())

    data_dir = _REPO / "zeta_modelling/data_2"
    less_raw = prepare_less(data_dir / "less_than-2yrs.csv")
    more_raw = prepare_more(data_dir / "more_than-2yrs.csv")
    less, _ = assign_dbscan_clusters(less_raw, eps_km=12.0)
    more, _ = assign_dbscan_clusters(more_raw, eps_km=12.0)
    less_f = add_site_type_feature(add_less_features(less))
    more_f = add_site_type_feature(add_more_features(more))

    less_train = less_f[less_f["date"] < pd.Timestamp("2025-01-01")]
    less_test = less_f[less_f["date"] >= pd.Timestamp("2025-01-01")].copy()
    more_train = more_f[more_f["date"] < pd.Timestamp("2025-01-01")]
    more_test = more_f[more_f["date"] >= pd.Timestamp("2025-01-01")].copy()

    saved = _load_model2_artifacts(data_dir / "model_saves_2")
    _, pred_site_less = infer_site_type_proba(less_test, saved["stm"])
    less_test_typed = less_test.copy()
    less_test_typed["site_type"] = pred_site_less.astype(str)
    less_pred = _predict_less_volume_mixed(saved["less_pair"], less_test_typed, None, None)
    more_pred = _predict_lgbm_volume(saved["more_bundle"], more_test)

    less_test = less_test.assign(predicted=less_pred, actual=less_test["monthly_volume"].to_numpy(dtype=float))
    more_test = more_test.assign(predicted=more_pred, actual=more_test["monthly_volume"].to_numpy(dtype=float))

    m2_less = (
        less_test.groupby("site_id", as_index=False)
        .agg(actual=("actual", "sum"), predicted=("predicted", "sum"), n_test_months=("date", "count"))
        .assign(cohort="<2y")
    )
    m2_more = (
        more_test.groupby("site_id", as_index=False)
        .agg(actual=("actual", "sum"), predicted=("predicted", "sum"), n_test_months=("date", "count"))
        .assign(cohort=">2y")
    )
    m2 = pd.concat([m2_less, m2_more], ignore_index=True).sort_values(["cohort", "site_id"])
    m2["abs_error"] = (m2["predicted"] - m2["actual"]).abs()
    m2["hit"] = m2["abs_error"] <= BUFFER
    m2["result"] = np.where(m2["hit"], "Hit", "Miss")

    m2_all = hit_metrics(m2["actual"].to_numpy(), m2["predicted"].to_numpy())
    m2_full12 = m2[m2["n_test_months"] == 12]
    m2_partial = m2[m2["n_test_months"] < 12]
    m2_f12 = hit_metrics(m2_full12["actual"].to_numpy(), m2_full12["predicted"].to_numpy())
    m2_part = hit_metrics(m2_partial["actual"].to_numpy(), m2_partial["predicted"].to_numpy())
    m2_monthly_df = pd.concat([less_test[["actual", "predicted"]], more_test[["actual", "predicted"]]])
    m2_monthly = hit_metrics(m2_monthly_df["actual"].to_numpy(), m2_monthly_df["predicted"].to_numpy())

    m2_less_sub = m2[m2["cohort"] == "<2y"]
    m2_more_sub = m2[m2["cohort"] == ">2y"]
    m2_less_met = hit_metrics(m2_less_sub["actual"].to_numpy(), m2_less_sub["predicted"].to_numpy())
    m2_more_met = hit_metrics(m2_more_sub["actual"].to_numpy(), m2_more_sub["predicted"].to_numpy())

    m2_metrics_json = json.loads((data_dir / "model_2_outputs/model2_metrics.json").read_text())

    m1_total_sites = panel["site_id"].nunique()
    m1_overlap = len(set(panel_train["site_id"]) & set(panel_test["site_id"]))

    lines: list[str] = []
    lines.append("# Zeta modelling — test split hit/miss report\n\n")

    lines.append("## Important: why test sites ≈ total sites\n\n")
    lines.append(
        "This report scores **only the test period** (`date >= 2025-01-01`). "
        "**No 2024 (train) months** are included in actual, predicted, or annual sums.\n\n"
    )
    lines.append(
        "The split is **by calendar time**, not by holding out random sites:\n\n"
    )
    lines.append("| | What it means |\n")
    lines.append("|---|---|\n")
    lines.append(
        "| **Train** | All sites’ months in **2024** (Model 1: 10,462 rows) |\n"
    )
    lines.append(
        "| **Test** | The **same sites’** months in **2025** (Model 1: 8,403 rows) |\n"
    )
    lines.append(
        f"| **Model 1** | Panel has **{m1_total_sites} sites total** → "
        f"**{panel_test['site_id'].nunique()} sites in test** = **100% of sites**, "
        f"because every site has both 2024 train and 2025 test months |\n"
    )
    lines.append(
        f"| **Site overlap** | {m1_overlap} sites appear in **both** train and test "
        f"(expected for a temporal split) |\n"
    )
    lines.append(
        "| **NOT included** | Train-period volumes (2024) are **never** summed into “actual” or “predicted” here |\n\n"
    )
    lines.append(
        "**Verified:** `model1_benchmark_volume_sample.csv` has **0** rows before `2025-01-01` "
        f"and **{len(bench):,}** rows = exactly the panel’s test slice.\n\n"
    )
    lines.append(
        "If you expected fewer test sites (e.g. 20% site holdout), that is a **different** evaluation design. "
        "Zeta models use **2024 → train, 2025 → test** on the same site universe.\n\n"
    )

    lines.append("---\n\n")
    lines.append("## Scoring rules\n\n")
    lines.append("- **Hit:** `|actual − predicted| ≤ 25,000`\n")
    lines.append("- **Monthly table:** each row is one **test month** (2025 only)\n")
    lines.append(
        "- **Annual site table:** sum of **test months only** per site (`n_test_months` shows how many "
        "2025 months that site has; 376 Model 1 sites have &lt; 12 months)\n\n"
    )

    lines.append("---\n\n")
    lines.append("## Train / test split (both models)\n\n")
    lines.append("| | Train period | Test period |\n")
    lines.append("|---|---|---|\n")
    lines.append("| **Cutoff** | `date < 2025-01-01` | `date >= 2025-01-01` |\n")
    lines.append(
        f"| **Model 1** | {len(panel_train):,} rows, {panel_train['site_id'].nunique()} sites (2024) | "
        f"{len(panel_test):,} rows, {panel_test['site_id'].nunique()} sites (2025) |\n"
    )
    lines.append(
        f"| **Model 2 `<2y`** | {len(less_train):,} rows, {less_train['site_id'].nunique()} sites | "
        f"{len(less_test):,} rows, {less_test['site_id'].nunique()} sites |\n"
    )
    lines.append(
        f"| **Model 2 `>2y`** | {len(more_train):,} rows, {more_train['site_id'].nunique()} sites | "
        f"{len(more_test):,} rows, {more_test['site_id'].nunique()} sites |\n"
    )
    lines.append(
        f"| **Model 2 combined** | — | {len(m2_monthly_df):,} monthly test rows → **{m2_all['n']} sites** (annual) |\n\n"
    )

    lines.append("---\n\n")
    lines.append("## Model 1 — Phase 3 P50 (test period only)\n\n")
    lines.append("Prediction column: `pred_phase3_p50`. Source: `model1_benchmark_volume_sample.csv`.\n\n")
    lines.append("### Final metrics\n\n")
    lines.append("| Cohort | Sites | Hits | ±25k accuracy | MAE (annual sum) |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(
        f"| All sites with any 2025 month | {m1_all['n']} | {m1_all['hits']} | **{m1_all['accuracy_pct']:.2f}%** | {m1_all['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Full 2025 calendar (12 test months) | {m1_f12['n']} | {m1_f12['hits']} | **{m1_f12['accuracy_pct']:.2f}%** | {m1_f12['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Partial 2025 (&lt; 12 test months) | {m1_part['n']} | {m1_part['hits']} | **{m1_part['accuracy_pct']:.2f}%** | {m1_part['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Monthly (8,403 test rows, not annualized) | — | {m1_monthly['hits']:,} / {m1_monthly['n']:,} | "
        f"{m1_monthly['accuracy_pct']:.2f}% | {m1_monthly['mae']:,.0f} |\n\n"
    )

    lines.append("### All sites — test-period volume sum (Model 1)\n\n")
    lines.append(
        "_`actual` / `predicted` = sum over **2025 test months only** (see `n_test_months`)._\n\n"
    )
    lines.append(site_table(m1))
    lines.append("\n\n")

    lines.append("---\n\n")
    lines.append("## Model 2 — cohort LGBM (test period only)\n\n")
    lines.append("Predictions from saved `model2_artifacts.joblib` on rows with `date >= 2025-01-01`.\n\n")
    lines.append("### Final metrics\n\n")
    lines.append("| Cohort | Sites | Hits | ±25k accuracy | MAE (annual sum) |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(
        f"| All test sites | {m2_all['n']} | {m2_all['hits']} | **{m2_all['accuracy_pct']:.2f}%** | {m2_all['mae']:,.0f} |\n"
    )
    lines.append(
        f"| `<2y` | {m2_less_met['n']} | {m2_less_met['hits']} | **{m2_less_met['accuracy_pct']:.2f}%** | {m2_less_met['mae']:,.0f} |\n"
    )
    lines.append(
        f"| `>2y` | {m2_more_met['n']} | {m2_more_met['hits']} | **{m2_more_met['accuracy_pct']:.2f}%** | {m2_more_met['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Full 12 test months | {m2_f12['n']} | {m2_f12['hits']} | **{m2_f12['accuracy_pct']:.2f}%** | {m2_f12['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Partial (&lt; 12 test months) | {m2_part['n']} | {m2_part['hits']} | **{m2_part['accuracy_pct']:.2f}%** | {m2_part['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Monthly ({m2_monthly['n']:,} test rows) | — | {m2_monthly['hits']:,} | "
        f"{m2_monthly['accuracy_pct']:.2f}% | {m2_monthly['mae']:,.0f} |\n"
    )
    lines.append(
        f"| Official monthly MAE (metrics.json) | `<2y` {m2_metrics_json['less_than_2y']['mae']:,.0f} | "
        f"`>2y` {m2_metrics_json['more_than_2y']['mae']:,.0f} | — | — |\n\n"
    )

    lines.append("### All sites — test-period volume sum (Model 2)\n\n")
    lines.append(site_table(m2))
    lines.append("\n\n")

    lines.append("---\n\n")
    lines.append("## Side-by-side (test period, all sites with any 2025 month)\n\n")
    lines.append("| | Model 1 | Model 2 |\n")
    lines.append("|---|---|---|\n")
    lines.append(f"| Test monthly rows scored | {len(bench):,} | {m2_monthly['n']:,} |\n")
    lines.append(f"| Sites in annual table | {m1_all['n']} | {m2_all['n']} |\n")
    lines.append(
        f"| Total sites in modelling universe | {m1_total_sites} | "
        f"{less_raw['site_id'].nunique() + more_raw['site_id'].nunique()} (cohorts separate) |\n"
    )
    lines.append(
        f"| ±25k hit rate (annual, all test sites) | **{m1_all['accuracy_pct']:.2f}%** ({m1_all['hits']}/{m1_all['n']}) | "
        f"**{m2_all['accuracy_pct']:.2f}%** ({m2_all['hits']}/{m2_all['n']}) |\n"
    )
    lines.append(f"| Annual MAE | {m1_all['mae']:,.0f} | {m2_all['mae']:,.0f} |\n")
    lines.append(f"| Monthly MAE | {m1_monthly['mae']:,.0f} | {m2_monthly['mae']:,.0f} |\n")

    OUT.write_text("".join(lines))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"Model 1: {m1_all['hits']}/{m1_all['n']} = {m1_all['accuracy_pct']:.2f}% (test 2025 only)")
    print(f"Model 2: {m2_all['hits']}/{m2_all['n']} = {m2_all['accuracy_pct']:.2f}% (test 2025 only)")


if __name__ == "__main__":
    main()
