"""
Demo bar graph for POST /pnl_analysis/bosch-forecast -- reproduces what Amit's front end is meant
to render per the 2026-07-22 meeting ("you have to make an API ... and will show it as a bar
graph"). Calls the REAL engine (app.pnl_analysis.modelling.bosch_forecast), not fabricated numbers;
the site factors/demographics/traffic below are the same real values validated against the
"South Charleston, WV" source workbook in agent.md, with a +2%/yr traffic growth assumption for
years 3-5 to make the demo shape more representative than the flat-growth validation case.

Styling follows /Users/lakshyatomar/Desktop/OX/plotting_guidelines.md (Plotly, this workspace's
convention): white background, hovermode="x unified", <b>/<sup> stacked title, "Wash Count" =
#00838F. Run from the repo root: python experiments/bosch-prediction-api/demo_plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go

from app.pnl_analysis.modelling import bosch_forecast as bosch_engine

WASH_COUNT_COLOR = "#00838F"  # plotting_guidelines.md color convention

request_payload = {
    "site_factors": {
        "area_profile": "business", "nearest_competition": "multiple_in_4_miles",
        "weekly_hours_category": "more_than_70", "type_of_site": "corner_lot_with_light",
        "site_accessibility": "easy_in_easy_out", "entrance_stack_up_area": "20_to_15_vehicles",
        "number_of_free_vacuum_slots": "12_to_20", "number_of_pay_stations": "2",
        "visibility": "more_than_500_ft", "traffic_speed": "40_to_50_mph",
    },
    "avg_household_size": 2.2, "pct_pop_25_65": 0.492, "pct_hh_income_over_35k": 0.621,
    "base_price_carwash": 5, "base_traffic": 19504,
    "year3_growth_pct": 0.02, "year4_growth_pct": 0.02, "year5_growth_pct": 0.02,
}

result = bosch_engine.bosch_forecast(
    site_factors=request_payload["site_factors"],
    avg_household_size=request_payload["avg_household_size"],
    pct_pop_25_65=request_payload["pct_pop_25_65"],
    pct_hh_income_over_35k=request_payload["pct_hh_income_over_35k"],
    base_price_carwash=request_payload["base_price_carwash"],
    base_traffic=request_payload["base_traffic"],
    year3_growth_pct=request_payload["year3_growth_pct"],
    year4_growth_pct=request_payload["year4_growth_pct"],
    year5_growth_pct=request_payload["year5_growth_pct"],
)

years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
yearly_values = [result["yearly"][f"year{i}"] for i in range(1, 6)]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=years, y=yearly_values,
    name="Wash Count",
    marker=dict(color=WASH_COUNT_COLOR),
    text=[f"{v:,.0f}" for v in yearly_values],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Est. wash count: %{y:,.0f}<extra></extra>",
))

fig.update_layout(
    title=dict(
        text="<b>Bosch Prediction — Estimated Yearly Car-Wash Volume</b>"
             "<br><sup>Site score 1.20 · demographic score +1.8% · 19,504 base traffic, +2%/yr growth Y3-Y5</sup>",
        x=0.5, xanchor="center", font=dict(size=13),
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    bargap=0.22,
    height=550,
    margin=dict(l=80, r=50, t=110, b=80),
    xaxis=dict(title=dict(text="Forecast Year", font=dict(size=11))),
    yaxis=dict(title=dict(text="Estimated Washes / Year", font=dict(size=11)), gridcolor="#eeeeee"),
    showlegend=False,  # single series -- the title already names it (dataviz convention)
)

out_dir = Path(__file__).resolve().parent
fig.write_image(str(out_dir / "demo_plot.png"), scale=2)
fig.write_html(str(out_dir / "demo_plot.html"), include_plotlyjs="cdn")
print("Wrote demo_plot.png and demo_plot.html to", out_dir)
print("Yearly values:", dict(zip(years, yearly_values)))
