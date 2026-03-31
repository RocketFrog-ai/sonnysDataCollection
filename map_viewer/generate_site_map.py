"""
Generate a 2D map view for an input site with nearby markers.

Outputs:
- output/site_places.json         (origin + nearby places with coordinates)
- output/site_map.html            (interactive 2D tile map with markers)
- output/site_map_static.png      (Google Static Maps road-map image)
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv


METERS_PER_MILE = 1609.34
EARTH_RADIUS_MILES = 3958.8
PLACES_SEARCH_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
DEFAULT_ADDRESS = "1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
DEFAULT_RADIUS_MILES = 2.0

# Keep the field mask narrow to reduce payload and cost.
PLACES_FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.types",
        "places.primaryType",
        "places.rating",
        "places.userRatingCount",
        "places.googleMapsUri",
    ]
)


def copy_marker_assets_to_output(output_dir: Path) -> None:
    """Copy bundled marker PNGs next to site_map.html so file:// loads work."""
    script_dir = Path(__file__).resolve().parent
    src = script_dir / "assets"
    if not src.is_dir():
        return
    dst = output_dir / "assets"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("marker_car_wash.png", "marker_walmart.png", "marker_costco.png"):
        f = src / name
        if f.is_file():
            shutil.copy2(f, dst / name)


def load_env() -> None:
    """Load project .env first, then allow current cwd override."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in miles."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c


def geocode_with_tomtom(address: str, geocode_url: str, tomtom_api_key: str) -> dict[str, Any]:
    encoded_address = quote(address)
    url = f"{geocode_url.rstrip('/')}/{encoded_address}.json"
    response = requests.get(url, params={"key": tomtom_api_key}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"No geocode result found for address: {address}")
    first = results[0]
    pos = first.get("position") or {}
    if "lat" not in pos or "lon" not in pos:
        raise RuntimeError(f"Geocode response missing coordinates for address: {address}")
    return {
        "address": address,
        "formatted_address": (first.get("address") or {}).get("freeformAddress", address),
        "latitude": float(pos["lat"]),
        "longitude": float(pos["lon"]),
    }


def places_search_nearby(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float,
    included_types: list[str],
    max_results: int = 20,
) -> list[dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }
    payload = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_miles * METERS_PER_MILE,
            }
        },
        "includedTypes": included_types,
        "rankPreference": "DISTANCE",
        "maxResultCount": min(max(1, max_results), 20),
    }
    response = requests.post(
        PLACES_SEARCH_NEARBY_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=30,
    )
    response.raise_for_status()
    return (response.json() or {}).get("places") or []


def places_search_text(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float,
    text_query: str,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }
    payload = {
        "textQuery": text_query,
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_miles * METERS_PER_MILE,
            }
        },
        "rankPreference": "RELEVANCE",
        "maxResultCount": min(max(1, max_results), 20),
    }
    response = requests.post(
        PLACES_SEARCH_TEXT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=30,
    )
    response.raise_for_status()
    return (response.json() or {}).get("places") or []


def normalize_place(
    place: dict[str, Any],
    category: str,
    origin_lat: float,
    origin_lon: float,
) -> dict[str, Any] | None:
    location = place.get("location") or {}
    lat = location.get("latitude")
    lon = location.get("longitude")
    if lat is None or lon is None:
        return None
    name_info = place.get("displayName") or {}
    distance_miles = haversine_miles(origin_lat, origin_lon, float(lat), float(lon))
    return {
        "id": place.get("id"),
        "name": name_info.get("text") if isinstance(name_info, dict) else None,
        "category": category,
        "latitude": float(lat),
        "longitude": float(lon),
        "distance_miles": round(distance_miles, 3),
        "rating": place.get("rating"),
        "user_rating_count": place.get("userRatingCount"),
        "address": place.get("formattedAddress"),
        "types": place.get("types") or [],
        "primary_type": place.get("primaryType"),
        "google_maps_uri": place.get("googleMapsUri"),
    }


def select_nearest_by_name(
    places: list[dict[str, Any]],
    keyword: str,
    category: str,
    origin_lat: float,
    origin_lon: float,
) -> dict[str, Any] | None:
    keyword_l = keyword.lower()
    normalized: list[dict[str, Any]] = []
    for place in places:
        normalized_place = normalize_place(place, category, origin_lat, origin_lon)
        if not normalized_place:
            continue
        name = (normalized_place.get("name") or "").lower()
        if keyword_l in name:
            normalized.append(normalized_place)
    if not normalized:
        return None
    normalized.sort(key=lambda p: p["distance_miles"])
    return normalized[0]


def zoom_for_radius(latitude: float, radius_miles: float, image_size_px: int = 640) -> int:
    """
    Compute a Google Static Maps zoom where diameter (~2 * radius) fits in image.
    """
    diameter_meters = max(radius_miles * 2 * METERS_PER_MILE, 1.0)
    meters_per_pixel_target = diameter_meters / image_size_px
    base = math.cos(math.radians(latitude)) * 156543.03392
    zoom = math.log2(base / meters_per_pixel_target) if meters_per_pixel_target > 0 else 12
    # Clamp into Google Static Maps supported range.
    return max(1, min(21, int(math.floor(zoom))))


def build_static_map_markers(
    origin: dict[str, Any],
    car_washes: list[dict[str, Any]],
    costco: dict[str, Any] | None,
    walmart: dict[str, Any] | None,
) -> list[str]:
    markers = [
        f"color:blue|label:S|{origin['latitude']},{origin['longitude']}",
    ]
    for place in car_washes[:15]:
        markers.append(f"color:red|size:mid|{place['latitude']},{place['longitude']}")
    if costco:
        markers.append(f"color:green|label:C|{costco['latitude']},{costco['longitude']}")
    if walmart:
        markers.append(f"color:orange|label:W|{walmart['latitude']},{walmart['longitude']}")
    return markers


def download_static_map(
    api_key: str,
    output_path: Path,
    origin: dict[str, Any],
    radius_miles: float,
    car_washes: list[dict[str, Any]],
    costco: dict[str, Any] | None,
    walmart: dict[str, Any] | None,
) -> str:
    zoom = zoom_for_radius(float(origin["latitude"]), radius_miles, image_size_px=640)
    marker_params = build_static_map_markers(origin, car_washes, costco, walmart)
    params: list[tuple[str, str]] = [
        ("center", f"{origin['latitude']},{origin['longitude']}"),
        ("zoom", str(zoom)),
        ("size", "640x640"),
        ("scale", "2"),
        ("maptype", "roadmap"),
        ("key", api_key),
    ]
    for marker in marker_params:
        params.append(("markers", marker))
    response = requests.get(GOOGLE_STATIC_MAPS_URL, params=params, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return response.url


def map_marker_color(category: str) -> str:
    colors = {
        "origin": "#ef4444",
        "car_wash": "#dc2626",
        "costco": "#16a34a",
        "walmart": "#f59e0b",
    }
    return colors.get(category, "#6b7280")


def build_map_html(
    origin: dict[str, Any],
    radius_miles: float,
    car_washes: list[dict[str, Any]],
    costco: dict[str, Any] | None,
    walmart: dict[str, Any] | None,
) -> str:
    markers = []
    for place in car_washes:
        markers.append(place)
    if costco:
        markers.append(costco)
    if walmart:
        markers.append(walmart)

    center_lat = origin["latitude"]
    center_lon = origin["longitude"]
    radius_meters = radius_miles * METERS_PER_MILE
    data_json = json.dumps(
        {
            "origin": origin,
            "radius_meters": radius_meters,
            "markers": markers,
        }
    )
    one_mile_meters = 1.0 * METERS_PER_MILE
    circle_blue = "#2563eb"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Site Map Viewer</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f4f6;
      color: #111827;
    }}
    #map {{
      width: 100%;
      height: 78vh;
    }}
    .header {{
      padding: 14px 18px;
      border-bottom: 1px solid #e5e7eb;
      background: #ffffff;
    }}
    .meta {{
      font-size: 13px;
      color: #374151;
      margin-top: 5px;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 8px;
      font-size: 12px;
      color: #374151;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 9999px;
      display: inline-block;
      margin-right: 5px;
    }}
    .marker-pin-img {{
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.15s ease;
    }}
    .marker-pin-img:hover {{
      transform: scale(1.12);
      z-index: 1000;
    }}
    .marker-pin-img img {{
      display: block;
      object-fit: contain;
      background: #ffffff;
      border-radius: 4px;
      box-shadow: 0 1px 5px rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.9);
    }}
    .site-bounce {{
      width: 10px;
      height: 10px;
      border-radius: 9999px;
      background: #22c55e;
      border: 1.5px solid #ffffff;
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.22);
      animation: bouncePin 1s ease-in-out infinite;
    }}
    @keyframes bouncePin {{
      0%, 100% {{ transform: translateY(0); }}
      50% {{ transform: translateY(-4px); }}
    }}
    .pulse-ring {{
      stroke: {circle_blue};
      stroke-width: 2;
      stroke-dasharray: 8 6;
      fill: rgba(37, 99, 235, 0.035);
      animation: pulseStroke 1.8s ease-in-out infinite;
    }}
    @keyframes pulseStroke {{
      0% {{ stroke-opacity: 0.25; }}
      50% {{ stroke-opacity: 0.85; }}
      100% {{ stroke-opacity: 0.25; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <strong>Input Site Map (2D tile)</strong>
    <div class="meta">{origin['formatted_address']} | Radius: {radius_miles:.1f} miles</div>
    <div class="legend">
      <span><span class="dot" style="background:#22c55e"></span>Site</span>
      <span><span class="dot" style="background:{map_marker_color('car_wash')}"></span>Car wash</span>
      <span><span class="dot" style="background:{circle_blue}"></span>1 mi / 2 mi radius</span>
      <span><span class="dot" style="background:#111827"></span>Custom marker art</span>
    </div>
  </div>
  <div id="map"></div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const data = {data_json};
    const map = L.map("map", {{
      zoomControl: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      touchZoom: true,
      boxZoom: true,
      keyboard: true
    }}).setView([{center_lat}, {center_lon}], 15);

    const streets = L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    const satellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
      {{
        maxZoom: 19,
        attribution:
          "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics"
      }}
    );

    const siteMarker = L.marker([data.origin.latitude, data.origin.longitude], {{
      icon: L.divIcon({{
        className: "",
        html: '<div class="site-bounce"></div>',
        iconSize: [10, 10],
        iconAnchor: [5, 5]
      }})
    }}).addTo(map);
    siteMarker.bindPopup(`<b>Input Site</b><br>${{data.origin.formatted_address}}`);

    const radiusCircle2Mi = L.circle([data.origin.latitude, data.origin.longitude], {{
      radius: data.radius_meters,
      color: "{circle_blue}",
      weight: 2,
      fillOpacity: 0.05
    }});

    const oneMilePulse = L.circle([data.origin.latitude, data.origin.longitude], {{
      radius: {one_mile_meters},
      color: "{circle_blue}",
      weight: 2,
      fillOpacity: 0.03,
      className: "pulse-ring"
    }}).addTo(map);

    const markerAssets = {{
      car_wash: "assets/marker_car_wash.png",
      costco: "assets/marker_costco.png",
      walmart: "assets/marker_walmart.png",
    }};
    const markerLayout = {{
      car_wash: {{ w: 26, h: 26 }},
      costco: {{ w: 22, h: 22 }},
      walmart: {{ w: 40, h: 18 }},
    }};

    const markerLayer = L.layerGroup().addTo(map);

    for (const p of data.markers) {{
      const cat = p.category || "car_wash";
      const src = markerAssets[cat] || markerAssets.car_wash;
      const lay = markerLayout[cat] || markerLayout.car_wash;
      const icon = L.divIcon({{
        className: "",
        html: `<div class="marker-pin-img"><img src="${{src}}" alt="" width="${{lay.w}}" height="${{lay.h}}" /></div>`,
        iconSize: [lay.w, lay.h],
        iconAnchor: [Math.round(lay.w / 2), Math.round(lay.h / 2)]
      }});
      const marker = L.marker([p.latitude, p.longitude], {{ icon }}).addTo(markerLayer);

      marker.bindPopup(
        `<b>${{p.name || "Unknown"}}</b><br>` +
        `Category: ${{p.category}}<br>` +
        `Distance: ${{p.distance_miles}} mi<br>` +
        `${{p.address || ""}}`
      );
    }}

    L.control.layers(
      {{ "Street Map": streets, "Satellite": satellite }},
      {{ "Markers": markerLayer, "Radius 1mi": oneMilePulse, "Radius 2mi": radiusCircle2Mi }},
      {{ collapsed: false }}
    ).addTo(map);

    const defaultGroup = new L.featureGroup([siteMarker, oneMilePulse, markerLayer]);
    map.fitBounds(defaultGroup.getBounds().pad(0.035));
    map.setZoom(Math.min(19, Math.max(16, map.getZoom())));
  </script>
</body>
</html>
"""


def main() -> None:
    load_env()

    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    tomtom_api_key = os.getenv("TOMTOM_API_KEY", "")
    tomtom_geocode_url = os.getenv("TOMTOM_GEOCODE_API_URL", "")

    if not google_maps_api_key:
        raise RuntimeError("Missing GOOGLE_MAPS_API_KEY in environment.")
    if not tomtom_api_key or not tomtom_geocode_url:
        raise RuntimeError("Missing TOMTOM_API_KEY or TOMTOM_GEOCODE_API_URL in environment.")

    input_address = os.getenv("MAP_VIEWER_ADDRESS", DEFAULT_ADDRESS)
    radius_miles = float(os.getenv("MAP_VIEWER_RADIUS_MILES", str(DEFAULT_RADIUS_MILES)))
    if radius_miles <= 0:
        raise RuntimeError("MAP_VIEWER_RADIUS_MILES must be > 0.")

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_marker_assets_to_output(output_dir)

    origin = geocode_with_tomtom(input_address, tomtom_geocode_url, tomtom_api_key)
    origin_lat = float(origin["latitude"])
    origin_lon = float(origin["longitude"])

    raw_car_washes = places_search_nearby(
        api_key=google_maps_api_key,
        latitude=origin_lat,
        longitude=origin_lon,
        radius_miles=radius_miles,
        included_types=["car_wash"],
        max_results=20,
    )
    car_washes: list[dict[str, Any]] = []
    for place in raw_car_washes:
        norm = normalize_place(place, "car_wash", origin_lat, origin_lon)
        if not norm:
            continue
        # Exclude gas stations that include car_wash as a secondary type.
        if "gas_station" in (norm.get("types") or []):
            continue
        if norm["distance_miles"] <= radius_miles:
            car_washes.append(norm)
    car_washes.sort(key=lambda p: p["distance_miles"])

    raw_costco = places_search_text(
        api_key=google_maps_api_key,
        latitude=origin_lat,
        longitude=origin_lon,
        radius_miles=radius_miles,
        text_query="Costco",
        max_results=20,
    )
    nearest_costco = select_nearest_by_name(
        raw_costco, "costco", "costco", origin_lat, origin_lon
    )
    if nearest_costco and nearest_costco["distance_miles"] > radius_miles:
        nearest_costco = None

    raw_walmart = places_search_text(
        api_key=google_maps_api_key,
        latitude=origin_lat,
        longitude=origin_lon,
        radius_miles=radius_miles,
        text_query="Walmart",
        max_results=20,
    )
    nearest_walmart = select_nearest_by_name(
        raw_walmart, "walmart", "walmart", origin_lat, origin_lon
    )
    if nearest_walmart and nearest_walmart["distance_miles"] > radius_miles:
        nearest_walmart = None

    html = build_map_html(
        origin=origin,
        radius_miles=radius_miles,
        car_washes=car_washes,
        costco=nearest_costco,
        walmart=nearest_walmart,
    )
    html_path = output_dir / "site_map.html"
    html_path.write_text(html, encoding="utf-8")

    static_map_path = output_dir / "site_map_static.png"
    download_static_map(
        api_key=google_maps_api_key,
        output_path=static_map_path,
        origin=origin,
        radius_miles=radius_miles,
        car_washes=car_washes,
        costco=nearest_costco,
        walmart=nearest_walmart,
    )

    payload = {
        "origin": origin,
        "radius_miles": radius_miles,
        "counts": {
            "car_washes": len(car_washes),
            "nearest_costco_found": nearest_costco is not None,
            "nearest_walmart_found": nearest_walmart is not None,
        },
        "car_washes": car_washes,
        "nearest_costco": nearest_costco,
        "nearest_walmart": nearest_walmart,
        "artifacts": {
            "html_map": str(html_path),
            "static_map_png": str(static_map_path),
        },
    }
    json_path = output_dir / "site_places.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Origin: {origin['formatted_address']} ({origin_lat}, {origin_lon})")
    print(f"Car washes found within {radius_miles} miles: {len(car_washes)}")
    print(f"Nearest Costco found: {'yes' if nearest_costco else 'no'}")
    print(f"Nearest Walmart found: {'yes' if nearest_walmart else 'no'}")
    print(f"Saved: {html_path}")
    print(f"Saved: {static_map_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
