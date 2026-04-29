from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zeta_modelling.model_1.phase3_advanced_forecast import (
    add_confidence_label,
    apply_global_uncertainty_calibration,
    final_report,
    load_artifacts,
)


ARTIFACTS_PATH = _REPO_ROOT / "zeta_modelling" / "model_1" / "phase3_artifacts.joblib"
DEFAULT_REPORT_PATH = _REPO_ROOT / "zeta_modelling" / "data_1" / "phase3_advanced_report.json"


@st.cache_resource
def get_artifacts(path: str):
    return load_artifacts(Path(path))


def get_current_coverage() -> float:
    if DEFAULT_REPORT_PATH.exists():
        payload = json.loads(DEFAULT_REPORT_PATH.read_text())
        return float(payload.get("backtest", {}).get("p10_p90_coverage", 0.454))
    return 0.454


def scenario_adjust(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()
    if scenario == "Conservative":
        out["volume"] = out["low"]
    elif scenario == "Aggressive":
        out["volume"] = out["high"]
    out["cumulative_volume"] = out["volume"].cumsum()
    return out


def recommendation_text(confidence: str, break_even: str, risk: str, margin: float) -> str:
    rec = []
    if "Low" in confidence:
        rec.append("High uncertainty; validate with local demand checks before commit.")
    if "Not reached" in break_even:
        rec.append(f"Proceed only if margin_per_wash can improve above {margin:.1f}.")
    if "High" in risk:
        rec.append("Use phased rollout due to high variability cluster behavior.")
    if not rec:
        rec.append("Scenario is investable with current assumptions.")
    return "\n".join([f"- {r}" for r in rec])


st.set_page_config(page_title="Car Wash Forecast Engine", layout="wide")
st.title("Car Wash Site Decision Engine")
st.caption("Forecasting tool for 3y/5y volume, uncertainty, and business viability.")

if not ARTIFACTS_PATH.exists():
    st.error(
        "Artifacts not found. Run this once before using app:\n"
        "`python zeta_modelling/model_1/build_phase3_artifacts.py`"
    )
    st.stop()

with st.sidebar:
    st.header("Inputs")
    st.subheader("Location")
    lat = st.number_input("Latitude", value=32.646615, format="%.6f")
    lon = st.number_input("Longitude", value=-96.533319, format="%.6f")

    st.subheader("Business")
    margin_per_wash = st.number_input("margin_per_wash", value=4.0, min_value=0.0, step=0.1)
    fixed_monthly_cost = st.number_input("fixed_monthly_cost", value=50000.0, min_value=0.0, step=1000.0)
    ramp_up_cost = st.number_input("ramp_up_cost", value=150000.0, min_value=0.0, step=5000.0)

    with st.expander("Advanced", expanded=False):
        scenario = st.selectbox("Scenario", ["Expected", "Conservative", "Aggressive"], index=0)
        horizon_choice = st.selectbox("Time horizon", ["3y", "5y"], index=1)
        target_coverage = st.number_input("Target coverage", value=0.80, min_value=0.1, max_value=0.99, step=0.01)

    run = st.button("Run Forecast", type="primary", use_container_width=True)

if run:
    artifacts = get_artifacts(str(ARTIFACTS_PATH))
    months = 36 if horizon_choice == "3y" else 60

    forecast, summary = final_report(
        lat=lat,
        lon=lon,
        artifacts=artifacts,
        months=months,
        start_date="2026-01-01",
        margin_per_wash=margin_per_wash,
        fixed_monthly_cost=fixed_monthly_cost,
        ramp_up_cost=ramp_up_cost,
    )
    current_coverage = get_current_coverage()
    forecast, scale = apply_global_uncertainty_calibration(
        forecast=forecast,
        current_coverage=current_coverage,
        target_coverage=target_coverage,
    )
    forecast, overall_confidence = add_confidence_label(forecast)
    forecast = scenario_adjust(forecast, scenario)

    risk = "High variability" if float(forecast["interval_width_ratio"].mean()) > 1.0 else "Moderate variability"
    break_even_text = (
        f"Month {summary['break_even_month']}"
        if summary["break_even_month"] is not None
        else f"Not reached (at margin={margin_per_wash:g})"
    )

    expected_volume = float(forecast["volume"].sum())
    range_low = float(forecast["low"].sum())
    range_high = float(forecast["high"].sum())
    total_profit = float(forecast["monthly_profit"].sum())

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Volume", f"{expected_volume:,.0f}")
    c2.metric("Range", f"{range_low:,.0f} - {range_high:,.0f}")
    c3.metric("Break-even", break_even_text)
    c4, c5, c6 = st.columns(3)
    c4.metric("Total Profit", f"{total_profit:,.0f}")
    c5.metric("Confidence", overall_confidence)
    c6.metric("Risk", risk)

    st.subheader("Forecast (P50 with P10-P90 band)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast["date"], forecast["volume"], label="P50", linewidth=2)
    ax.fill_between(forecast["date"], forecast["low"], forecast["high"], alpha=0.2, label="P10-P90")
    ax.set_ylabel("Monthly Volume")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Cluster Trend Context")
    primary_cluster = str(pd.DataFrame(summary["top3_clusters"]).iloc[0]["cluster_id"])
    cluster_curve = artifacts.cluster_age_avg[
        artifacts.cluster_age_avg["cluster_id"].astype(str) == primary_cluster
    ].copy()
    compare = forecast[["age_in_months", "volume"]].copy()
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 4))
    if not cluster_curve.empty:
        cluster_curve = cluster_curve.sort_values("age_in_months")
        ax_cluster.plot(
            cluster_curve["age_in_months"],
            cluster_curve["cluster_age_avg"],
            label=f"Cluster {primary_cluster} avg trend",
            linewidth=2,
            alpha=0.8,
        )
    ax_cluster.plot(
        compare["age_in_months"],
        compare["volume"],
        label="New site expected (P50/scenario-adjusted)",
        linewidth=2,
        linestyle="--",
    )
    ax_cluster.set_xlabel("Age in months")
    ax_cluster.set_ylabel("Monthly Volume")
    ax_cluster.legend()
    st.pyplot(fig_cluster, clear_figure=True)

    st.subheader("Cumulative Profit")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(forecast["date"], forecast["cumulative_profit"], linewidth=2)
    if summary["break_even_month"] is not None:
        be_row = forecast[forecast["age_in_months"] == summary["break_even_month"]].head(1)
        if len(be_row):
            ax2.scatter(be_row["date"], be_row["cumulative_profit"], color="green", s=80, label="Break-even")
            ax2.legend()
    ax2.set_ylabel("Cumulative Profit")
    st.pyplot(fig2, clear_figure=True)

    with st.expander("Monthly Table", expanded=False):
        show_cols = ["date", "age_in_months", "volume", "low", "high", "monthly_profit", "cumulative_profit", "confidence_label"]
        st.dataframe(forecast[show_cols], use_container_width=True)

    st.subheader("Cluster Info")
    st.dataframe(pd.DataFrame(summary["top3_clusters"]), use_container_width=True)

    st.subheader("Risk + Recommendation")
    st.info(recommendation_text(overall_confidence, break_even_text, risk, margin_per_wash))

    st.subheader("Downloads")
    csv_bytes = forecast.to_csv(index=False).encode("utf-8")
    summary_payload = {
        "expected_volume": expected_volume,
        "range_low": range_low,
        "range_high": range_high,
        "break_even": break_even_text,
        "total_profit": total_profit,
        "confidence": overall_confidence,
        "risk": risk,
        "scenario": scenario,
        "horizon_months": months,
        "calibration_scale": scale,
        "current_coverage": current_coverage,
    }
    st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast_output.csv", mime="text/csv")
    st.download_button(
        "Download summary JSON",
        data=json.dumps(summary_payload, indent=2).encode("utf-8"),
        file_name="forecast_summary.json",
        mime="application/json",
    )

    st.subheader("Map")
    st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))
