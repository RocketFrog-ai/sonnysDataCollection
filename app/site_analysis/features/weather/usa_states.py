"""
US state reference for weather: state name, abbreviation, and representative coordinates
(state capital) for fetching Open-Meteo climate data per state.
Used for national and state-wise climate averages (USA).
"""

# (state_abbr, state_name, latitude, longitude) — state capital coordinates
USA_STATES = [
    ("AL", "Alabama", 32.3770, -86.3006),
    ("AK", "Alaska", 58.3016, -134.4197),
    ("AZ", "Arizona", 33.4484, -112.0740),
    ("AR", "Arkansas", 34.7465, -92.2896),
    ("CA", "California", 38.5767, -121.4936),
    ("CO", "Colorado", 39.7392, -104.9903),
    ("CT", "Connecticut", 41.7658, -72.6734),
    ("DE", "Delaware", 38.9108, -75.5277),
    ("FL", "Florida", 30.4383, -84.2807),
    ("GA", "Georgia", 33.7490, -84.3880),
    ("HI", "Hawaii", 21.3070, -157.8584),
    ("ID", "Idaho", 43.6178, -116.1997),
    ("IL", "Illinois", 39.7980, -89.6440),
    ("IN", "Indiana", 39.7684, -86.1581),
    ("IA", "Iowa", 41.5868, -93.6250),
    ("KS", "Kansas", 39.0473, -95.6752),
    ("KY", "Kentucky", 38.1868, -84.8753),
    ("LA", "Louisiana", 30.4583, -91.1403),
    ("ME", "Maine", 44.3070, -69.7817),
    ("MD", "Maryland", 38.9784, -76.4922),
    ("MA", "Massachusetts", 42.3601, -71.0589),
    ("MI", "Michigan", 42.7325, -84.5555),
    ("MN", "Minnesota", 44.9551, -93.1022),
    ("MS", "Mississippi", 32.2988, -90.1848),
    ("MO", "Missouri", 38.5767, -92.1736),
    ("MT", "Montana", 46.5891, -112.0391),
    ("NE", "Nebraska", 40.8081, -96.6997),
    ("NV", "Nevada", 39.1638, -119.7674),
    ("NH", "New Hampshire", 43.2081, -71.5376),
    ("NJ", "New Jersey", 40.2206, -74.7597),
    ("NM", "New Mexico", 35.6870, -105.9378),
    ("NY", "New York", 42.6526, -73.7562),
    ("NC", "North Carolina", 35.7796, -78.6382),
    ("ND", "North Dakota", 46.8208, -100.7837),
    ("OH", "Ohio", 39.9612, -82.9988),
    ("OK", "Oklahoma", 35.4922, -97.5034),
    ("OR", "Oregon", 44.9429, -123.0351),
    ("PA", "Pennsylvania", 40.2732, -76.8867),
    ("RI", "Rhode Island", 41.8301, -71.4148),
    ("SC", "South Carolina", 34.0007, -81.0348),
    ("SD", "South Dakota", 43.9695, -99.9018),
    ("TN", "Tennessee", 36.1659, -86.7844),
    ("TX", "Texas", 30.2672, -97.7431),
    ("UT", "Utah", 40.7608, -111.8910),
    ("VT", "Vermont", 44.2601, -72.5754),
    ("VA", "Virginia", 37.5385, -77.4360),
    ("WA", "Washington", 47.0379, -122.9007),
    ("WV", "West Virginia", 38.3498, -81.6326),
    ("WI", "Wisconsin", 43.0731, -89.4012),
    ("WY", "Wyoming", 41.1400, -104.8202),
    ("DC", "District of Columbia", 38.9072, -77.0369),
]


def get_usa_state_coordinates():
    """Return list of (state_abbr, state_name, lat, lon) for all US states + DC."""
    return list(USA_STATES)


def get_state_abbr_to_name():
    """Return dict mapping state abbreviation to full name."""
    return {abbr: name for abbr, name, _, _ in USA_STATES}


# Bounding boxes (min_lat, max_lat, min_lon, max_lon) for area-aggregated weather.
# Used with bbox_to_grid_points + get_climate_data_for_polygon for "state as region" average.
# Approximate state extents (mainland where applicable).
STATE_BOUNDING_BOXES = {
    "AL": (30.22, 35.00, -88.47, -84.89),
    "AK": (51.21, 71.35, -179.15, 179.77),
    "AZ": (31.33, 37.00, -114.82, -109.05),
    "AR": (33.00, 36.50, -94.62, -89.64),
    "CA": (32.53, 42.01, -124.41, -114.13),
    "CO": (36.99, 41.00, -109.06, -102.04),
    "CT": (40.98, 42.05, -73.73, -71.79),
    "DE": (38.45, 39.84, -75.79, -75.05),
    "FL": (24.52, 31.00, -87.63, -80.03),
    "GA": (30.36, 35.00, -85.61, -80.84),
    "HI": (18.91, 22.24, -160.26, -154.81),
    "ID": (42.00, 49.00, -117.24, -111.04),
    "IL": (36.97, 42.51, -91.51, -87.50),
    "IN": (37.77, 41.76, -88.10, -84.78),
    "IA": (40.38, 43.50, -96.64, -90.14),
    "KS": (37.00, 40.00, -102.05, -94.59),
    "KY": (36.50, 39.15, -89.57, -82.03),
    "LA": (28.93, 33.02, -94.04, -88.82),
    "ME": (43.06, 47.46, -71.08, -66.95),
    "MD": (37.91, 39.72, -79.49, -75.05),
    "MA": (41.24, 42.89, -73.51, -69.93),
    "MI": (41.70, 48.19, -90.42, -82.41),
    "MN": (43.50, 49.38, -97.24, -89.49),
    "MS": (30.17, 35.00, -91.66, -88.10),
    "MO": (35.99, 40.61, -95.77, -89.10),
    "MT": (44.36, 49.00, -116.05, -104.04),
    "NE": (40.00, 43.00, -104.06, -95.31),
    "NV": (35.00, 42.00, -120.01, -114.04),
    "NH": (42.70, 45.31, -72.56, -70.70),
    "NJ": (38.93, 41.36, -75.56, -73.89),
    "NM": (31.33, 37.00, -109.05, -103.00),
    "NY": (40.50, 45.02, -79.76, -71.86),
    "NC": (33.84, 36.59, -84.32, -75.46),
    "ND": (45.94, 49.00, -104.05, -96.55),
    "OH": (38.40, 42.00, -84.82, -80.52),
    "OK": (33.62, 37.00, -103.00, -94.43),
    "OR": (42.00, 46.29, -124.57, -116.46),
    "PA": (39.72, 42.27, -80.52, -74.69),
    "RI": (41.15, 42.02, -71.86, -71.12),
    "SC": (32.03, 35.22, -83.35, -78.54),
    "SD": (42.48, 45.95, -104.06, -96.44),
    "TN": (34.98, 36.68, -90.31, -81.65),
    "TX": (25.84, 36.50, -106.65, -93.51),
    "UT": (37.00, 42.00, -114.05, -109.04),
    "VT": (42.73, 45.02, -73.44, -71.46),
    "VA": (36.54, 39.47, -83.68, -75.24),
    "WA": (45.54, 49.00, -124.85, -116.92),
    "WV": (37.20, 40.64, -82.64, -77.72),
    "WI": (42.49, 47.08, -92.89, -86.25),
    "WY": (41.00, 45.01, -111.06, -104.05),
    "DC": (38.79, 38.99, -77.12, -76.91),
}


def get_state_bbox(state_abbr):
    """Return (min_lat, max_lat, min_lon, max_lon) for state, or None if not defined."""
    return STATE_BOUNDING_BOXES.get(state_abbr.upper())


def get_state_for_point(lat, lon):
    """
    Return the state abbreviation whose bounding box contains (lat, lon), or None.
    Uses STATE_BOUNDING_BOXES; first matching state is returned (no overlap resolution).
    """
    for abbr, (min_lat, max_lat, min_lon, max_lon) in STATE_BOUNDING_BOXES.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return abbr
    return None
