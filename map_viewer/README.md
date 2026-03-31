# Map Viewer

Generates a 2D site map for an input address and overlays nearby places:
- car washes within a 4-mile radius
- nearest Costco
- nearest Walmart

## Run

From repo root:

```bash
python "map_viewer/generate_site_map.py"
```

Optional environment overrides:

```bash
MAP_VIEWER_ADDRESS="1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
MAP_VIEWER_RADIUS_MILES="4"
python "map_viewer/generate_site_map.py"
```

## Outputs

Created in `map_viewer/output/`:
- `site_map.html` (interactive 2D tile map with markers + 4-mile circle)
- `site_map_static.png` (Google Static Maps 2D map image with markers)
- `site_places.json` (structured data with coordinates and metadata)
