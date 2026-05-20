"""Render close_site_clusters_combined sites on an interactive USA map."""
import os

import pandas as pd
import plotly.express as px

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "data", "close_site_clusters_combined.csv")
OUT = os.path.join(HERE, "close_site_clusters_usa_map.html")

df = pd.read_csv(CSV)

# One marker per physical site.
sites = (
    df.sort_values("calendar_day")
    .drop_duplicates("client_id_location_id")
    .copy()
)
sites["cluster_id"] = sites["cluster_id"].astype(str)
sites["localisation"] = sites["localisation"].astype(str)
sites["total_washes"] = (
    df.groupby("client_id_location_id")["wash_count_total"].sum()
    .reindex(sites["client_id_location_id"]).values
)

# Stable ordering so each localisation always maps to the same color.
loc_order = sorted(sites["localisation"].unique(), key=lambda x: int(float(x)))

fig = px.scatter_geo(
    sites,
    lat="latitude",
    lon="longitude",
    color="localisation",
    category_orders={"localisation": loc_order},
    color_discrete_sequence=px.colors.qualitative.Alphabet,
    scope="usa",
    hover_name="client_id",
    hover_data={
        "address": True,
        "city": True,
        "state": True,
        "client_type": True,
        "source": True,
        "cluster_id": True,
        "localisation": True,
        "total_washes": ":,.0f",
        "latitude": False,
        "longitude": False,
    },
    title=f"Close-site clusters by localisation — {len(sites)} sites, {sites['localisation'].nunique()} localisations",
)
fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
fig.update_layout(legend_title_text="localisation", height=800)
fig.write_html(OUT, include_plotlyjs="cdn")
print(OUT)
