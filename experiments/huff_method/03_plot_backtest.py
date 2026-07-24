"""
Step 3: predicted-vs-actual scatter for the Huff-lite backtest (02_huff_backtest.py's output).
Styling follows /Users/lakshyatomar/Desktop/OX/plotting_guidelines.md (this workspace's Plotly
convention); colors reuse that file's categorical palette rather than inventing new ones.

Run from the repo root: python experiments/huff-model-backtest/03_plot_backtest.py
Output: experiments/huff-model-backtest/backtest_scatter.png
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "backtest_results.csv"
OUT_PNG = HERE / "backtest_scatter.png"

TIER_COLOR = {"grounded": "#00838F", "thin": "#EF6C00", "isolated": "#AD1457"}


def main():
    df = pd.read_csv(RESULTS_CSV)

    fig = go.Figure()
    lo, hi = 20, 30000
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(color="#999999", width=1.5, dash="dot"),
                              name="Perfect prediction (y=x)", hoverinfo="skip"))

    for tier in ["grounded", "thin", "isolated"]:
        sub = df[df.grounding == tier]
        fig.add_trace(go.Scatter(
            x=sub["actual_mature_wash_mo"], y=sub["huff_pred_monthly"],
            mode="markers", name=f"{tier} (n={len(sub)})",
            marker=dict(color=TIER_COLOR[tier], size=9, line=dict(color="white", width=1)),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:,.0f}/mo<br>Huff-lite pred: %{y:,.0f}/mo<extra></extra>",
            text=sub["site_key"],
        ))

    fig.update_layout(
        title=dict(
            text="<b>Huff-lite backtest — predicted vs. actual mature monthly wash volume</b>"
                 "<br><sup>n70-final-considered resolved sites · log-log axes</sup>",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="closest",
        xaxis=dict(title=dict(text="Actual mature washes/mo", font=dict(size=11)), type="log", gridcolor="#eeeeee"),
        yaxis=dict(title=dict(text="Huff-lite predicted washes/mo", font=dict(size=11)), type="log", gridcolor="#eeeeee"),
        height=650, margin=dict(l=80, r=50, t=110, b=150),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
    )
    fig.write_image(str(OUT_PNG), scale=2)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
