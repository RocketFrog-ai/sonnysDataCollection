#!/usr/bin/env python3
"""
Zip-level clustering + visualization:
- Build feature vectors per zip
- KMeans and DBSCAN clustering
- t-SNE projection view
- Honeycomb-style hexbin density with cluster centroids
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


EDA_DIR = Path(__file__).resolve().parent
CSV_PATH = EDA_DIR / "master_daily_with_site_metadata.csv"
ZIP_LOOKUP = EDA_DIR / "zip_lat_lon_lookup.csv"
PLOTS_DIR = EDA_DIR / "plots"


def normalize_zip(v: object) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(5)[:5]


def build_zip_feature_table() -> pd.DataFrame:
    usecols = [
        "zip",
        "region",
        "state",
        "city",
        "wash_count_total",
        "wash_count_retail",
        "wash_count_membership",
        "wash_count_voucher",
        "weather_total_precipitation_mm",
        "weather_total_snowfall_cm",
        "weather_total_sunshine_hours",
        "weather_avg_daily_max_windspeed_ms",
    ]
    df = pd.read_csv(CSV_PATH, usecols=usecols, low_memory=False)
    for c in [
        "wash_count_total",
        "wash_count_retail",
        "wash_count_membership",
        "wash_count_voucher",
        "weather_total_precipitation_mm",
        "weather_total_snowfall_cm",
        "weather_total_sunshine_hours",
        "weather_avg_daily_max_windspeed_ms",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["zip5"] = df["zip"].map(normalize_zip)
    df = df.dropna(subset=["zip5"]).copy()

    agg = (
        df.groupby("zip5", as_index=False)
        .agg(
            total_washes=("wash_count_total", "sum"),
            avg_washes=("wash_count_total", "mean"),
            n_site_days=("wash_count_total", "size"),
            retail_total=("wash_count_retail", "sum"),
            membership_total=("wash_count_membership", "sum"),
            voucher_total=("wash_count_voucher", "sum"),
            precip_mean=("weather_total_precipitation_mm", "mean"),
            snowfall_mean=("weather_total_snowfall_cm", "mean"),
            sunshine_mean=("weather_total_sunshine_hours", "mean"),
            wind_mean=("weather_avg_daily_max_windspeed_ms", "mean"),
            region=("region", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
            state=("state", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
            city=("city", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
        )
        .sort_values("zip5")
    )
    ch_tot = agg[["retail_total", "membership_total", "voucher_total"]].sum(axis=1).replace(0, np.nan)
    agg["retail_share"] = agg["retail_total"] / ch_tot
    agg["membership_share"] = agg["membership_total"] / ch_tot
    agg["voucher_share"] = agg["voucher_total"] / ch_tot

    geo = pd.read_csv(ZIP_LOOKUP, dtype={"zip5": str})
    geo["zip5"] = geo["zip5"].map(normalize_zip)
    geo = geo[["zip5", "latitude", "longitude", "has_lat_lon"]]

    out = agg.merge(geo, on="zip5", how="left")
    out = out[out["has_lat_lon"] == True].copy()  # noqa: E712
    return out


def run_clustering(z: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [
        "total_washes",
        "avg_washes",
        "n_site_days",
        "retail_share",
        "membership_share",
        "voucher_share",
        "precip_mean",
        "snowfall_mean",
        "sunshine_mean",
        "wind_mean",
        "latitude",
        "longitude",
    ]
    m = z.dropna(subset=feat_cols).copy()
    X = m[feat_cols].to_numpy(dtype=float)
    finite_mask = np.isfinite(X).all(axis=1)
    m = m.loc[finite_mask].copy()
    X = X[finite_mask]
    Xs = StandardScaler().fit_transform(X)

    k = 6
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    m["kmeans_cluster"] = km.fit_predict(Xs)

    # Conservative eps; labels -1 as noise.
    db = DBSCAN(eps=1.25, min_samples=8)
    m["dbscan_cluster"] = db.fit_predict(Xs)

    perplexity = 35 if len(m) > 120 else max(10, min(30, len(m) // 5))
    ts = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
    )
    emb = ts.fit_transform(Xs)
    m["tsne_x"] = emb[:, 0]
    m["tsne_y"] = emb[:, 1]

    m.to_csv(EDA_DIR / "zip_feature_vectors_clustered.csv", index=False)
    return m


def plot_honeycomb_with_centroids(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    hb = ax.hexbin(
        m["longitude"],
        m["latitude"],
        C=m["total_washes"],
        reduce_C_function=np.sum,
        gridsize=38,
        cmap="viridis",
        mincnt=1,
        alpha=0.9,
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Sum of total_washes inside hex cell")

    # Centroid markers from KMeans cluster groups (in geo coordinates).
    for c, g in m.groupby("kmeans_cluster"):
        w = g["total_washes"].values
        lon_c = np.average(g["longitude"], weights=w)
        lat_c = np.average(g["latitude"], weights=w)
        ax.scatter(lon_c, lat_c, marker="X", s=220, edgecolor="black", linewidth=1, color="#ff4d4f")
        ax.text(lon_c, lat_c, f" C{c}", fontsize=8, color="white", weight="bold")

    ax.set_title("Honeycomb-style zip density map + weighted cluster centroids")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "09a_honeycomb_zip_density_with_centroids.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_views(m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc1 = axes[0].scatter(
        m["tsne_x"], m["tsne_y"], c=m["kmeans_cluster"], cmap="tab10", s=35, alpha=0.82
    )
    axes[0].set_title("t-SNE projection colored by KMeans cluster")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    plt.colorbar(sc1, ax=axes[0], label="KMeans cluster")

    db_labels = m["dbscan_cluster"].copy()
    # Shift for color continuity while preserving -1 as noise.
    color_vals = db_labels.where(db_labels >= 0, -1)
    sc2 = axes[1].scatter(m["tsne_x"], m["tsne_y"], c=color_vals, cmap="tab20", s=35, alpha=0.82)
    axes[1].set_title("t-SNE projection colored by DBSCAN cluster")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    plt.colorbar(sc2, ax=axes[1], label="DBSCAN cluster (-1=noise)")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "09b_tsne_kmeans_dbscan.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_map(m: pd.DataFrame) -> None:
    fig = px.scatter_map(
        m,
        lat="latitude",
        lon="longitude",
        color=m["kmeans_cluster"].astype(str),
        size="total_washes",
        hover_name="zip5",
        hover_data={
            "city": True,
            "state": True,
            "region": True,
            "total_washes": ":,.0f",
            "avg_washes": ":.1f",
            "n_site_days": ":,",
            "kmeans_cluster": True,
            "dbscan_cluster": True,
            "latitude": False,
            "longitude": False,
        },
        map_style="open-street-map",
        zoom=3.1,
        center={"lat": 37.8, "lon": -96.0},
        opacity=0.6,
        size_max=22,
        title="Zip feature clusters (KMeans) on US map",
    )
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=8), legend_title_text="KMeans")
    fig.write_html(PLOTS_DIR / "09c_zip_feature_clusters_map.html")


def build_city_feature_table(m: pd.DataFrame) -> pd.DataFrame:
    wcols = [
        "total_washes",
        "avg_washes",
        "n_site_days",
        "retail_share",
        "membership_share",
        "voucher_share",
        "precip_mean",
        "snowfall_mean",
        "sunshine_mean",
        "wind_mean",
        "latitude",
        "longitude",
    ]

    def wmean(x: pd.Series, w: pd.Series) -> float:
        ww = w.fillna(0).to_numpy(dtype=float)
        xx = pd.to_numeric(x, errors="coerce").fillna(0).to_numpy(dtype=float)
        if np.sum(ww) <= 0:
            return float(np.mean(xx))
        return float(np.average(xx, weights=ww))

    out = (
        m.groupby(["state", "city"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "region": g["region"].dropna().astype(str).mode().iloc[0] if g["region"].dropna().size else None,
                    "n_zips": int(g["zip5"].nunique()),
                    **{c: wmean(g[c], g["total_washes"]) for c in wcols},
                }
            )
        )
        .reset_index(drop=True)
    )
    out["city_state"] = out["city"].astype(str) + ", " + out["state"].astype(str)
    return out


def run_city_clustering(city_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    feat_cols = [
        "total_washes",
        "avg_washes",
        "n_site_days",
        "retail_share",
        "membership_share",
        "voucher_share",
        "precip_mean",
        "snowfall_mean",
        "sunshine_mean",
        "wind_mean",
        "latitude",
        "longitude",
    ]
    c = city_df.dropna(subset=feat_cols).copy()
    X = c[feat_cols].to_numpy(dtype=float)
    finite_mask = np.isfinite(X).all(axis=1)
    c = c.loc[finite_mask].copy()
    X = X[finite_mask]
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=min(k, max(2, len(c) // 8)), random_state=42, n_init=20)
    c["city_kmeans_cluster"] = km.fit_predict(Xs)
    return c


def plot_multi_layer_cluster_map(m: pd.DataFrame) -> None:
    """
    One interactive map with layer switch:
    - Zip feature clusters (KMeans)
    - City-level clusters (KMeans on city aggregated features)
    - Zip DBSCAN clusters / anomalies
    """
    city_df = run_city_clustering(build_city_feature_table(m))

    fig = go.Figure()

    # Layer 1: zip kmeans
    fig.add_trace(
        go.Scattermap(
            lat=m["latitude"],
            lon=m["longitude"],
            mode="markers",
            name="Zip clusters (KMeans)",
            visible=True,
            text=m["zip5"],
            customdata=np.stack(
                [
                    m["city"].astype(str),
                    m["state"].astype(str),
                    m["region"].astype(str),
                    m["total_washes"].astype(float),
                    m["avg_washes"].astype(float),
                    m["kmeans_cluster"].astype(int),
                ],
                axis=1,
            ),
            hovertemplate=(
                "ZIP %{text}<br>%{customdata[0]}, %{customdata[1]} | %{customdata[2]}<br>"
                "Total washes: %{customdata[3]:,.0f}<br>"
                "Avg washes: %{customdata[4]:.1f}<br>"
                "KMeans cluster: %{customdata[5]}<extra></extra>"
            ),
            marker=dict(
                size=np.clip(np.sqrt(m["total_washes"]) / 16, 5, 24),
                color=m["kmeans_cluster"],
                colorscale="Turbo",
                cmin=0,
                cmax=max(1, int(m["kmeans_cluster"].max())),
                showscale=True,
                colorbar=dict(title="Zip KMeans", x=1.01),
                opacity=0.62,
            ),
        )
    )

    # Layer 2: city kmeans
    fig.add_trace(
        go.Scattermap(
            lat=city_df["latitude"],
            lon=city_df["longitude"],
            mode="markers",
            name="City clusters (KMeans)",
            visible=False,
            text=city_df["city_state"],
            customdata=np.stack(
                [
                    city_df["region"].astype(str),
                    city_df["n_zips"].astype(int),
                    city_df["total_washes"].astype(float),
                    city_df["avg_washes"].astype(float),
                    city_df["city_kmeans_cluster"].astype(int),
                ],
                axis=1,
            ),
            hovertemplate=(
                "%{text}<br>Region: %{customdata[0]}<br>"
                "Zips in city node: %{customdata[1]}<br>"
                "Weighted total washes: %{customdata[2]:,.0f}<br>"
                "Weighted avg washes: %{customdata[3]:.1f}<br>"
                "City KMeans cluster: %{customdata[4]}<extra></extra>"
            ),
            marker=dict(
                size=np.clip(np.sqrt(city_df["total_washes"]) / 8, 10, 34),
                color=city_df["city_kmeans_cluster"],
                colorscale="Plasma",
                cmin=0,
                cmax=max(1, int(city_df["city_kmeans_cluster"].max())),
                showscale=True,
                colorbar=dict(title="City KMeans", x=1.01),
                opacity=0.78,
            ),
        )
    )

    # Layer 3: zip dbscan
    fig.add_trace(
        go.Scattermap(
            lat=m["latitude"],
            lon=m["longitude"],
            mode="markers",
            name="Zip DBSCAN (-1 noise)",
            visible=False,
            text=m["zip5"],
            customdata=np.stack(
                [
                    m["city"].astype(str),
                    m["state"].astype(str),
                    m["dbscan_cluster"].astype(int),
                    m["total_washes"].astype(float),
                ],
                axis=1,
            ),
            hovertemplate=(
                "ZIP %{text}<br>%{customdata[0]}, %{customdata[1]}<br>"
                "DBSCAN cluster: %{customdata[2]}<br>"
                "Total washes: %{customdata[3]:,.0f}<extra></extra>"
            ),
            marker=dict(
                size=np.clip(np.sqrt(m["total_washes"]) / 16, 5, 24),
                color=m["dbscan_cluster"],
                colorscale="RdYlBu",
                cmin=min(-1, int(m["dbscan_cluster"].min())),
                cmax=max(1, int(m["dbscan_cluster"].max())),
                showscale=True,
                colorbar=dict(title="DBSCAN", x=1.01),
                opacity=0.68,
            ),
        )
    )

    fig.update_layout(
        title="Cluster map (single file): switch layer for ZIP/CITY/DBSCAN views",
        map=dict(
            style="open-street-map",
            center=dict(lat=37.8, lon=-96.0),
            zoom=3.2,
        ),
        margin=dict(l=8, r=8, t=48, b=8),
        updatemenus=[
            dict(
                type="dropdown",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                bgcolor="white",
                bordercolor="#999",
                buttons=[
                    dict(
                        label="Zip feature clusters (KMeans)",
                        method="update",
                        args=[{"visible": [True, False, False]}],
                    ),
                    dict(
                        label="City-level clusters (KMeans)",
                        method="update",
                        args=[{"visible": [False, True, False]}],
                    ),
                    dict(
                        label="Zip feature clusters (DBSCAN)",
                        method="update",
                        args=[{"visible": [False, False, True]}],
                    ),
                ],
            )
        ],
    )

    fig.write_html(PLOTS_DIR / "09d_cluster_map_layers.html")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    z = build_zip_feature_table()
    m = run_clustering(z)
    plot_honeycomb_with_centroids(m)
    plot_tsne_views(m)
    plot_cluster_map(m)
    plot_multi_layer_cluster_map(m)
    print(f"Wrote: {EDA_DIR / 'zip_feature_vectors_clustered.csv'}")
    print(f"Wrote: {PLOTS_DIR / '09a_honeycomb_zip_density_with_centroids.png'}")
    print(f"Wrote: {PLOTS_DIR / '09b_tsne_kmeans_dbscan.png'}")
    print(f"Wrote: {PLOTS_DIR / '09c_zip_feature_clusters_map.html'}")
    print(f"Wrote: {PLOTS_DIR / '09d_cluster_map_layers.html'}")
    print(f"Rows clustered: {len(m):,}")


if __name__ == "__main__":
    main()
