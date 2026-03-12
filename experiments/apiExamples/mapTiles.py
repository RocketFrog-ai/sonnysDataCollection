import requests
import math
import json # For parsing JSON and error checking

# IMPORTANT: Replace with your actual API key
# Ensure your API key has "Map Tiles API" enabled in Google Cloud Console.
API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA"

def fetch_map_tile(lat, lon, zoom=15, tile_format='2d',
                   map_type='roadmap', language='en-US', region='US'):
    """
    Fetch a single map tile.
    For tile_format='2d', returns image bytes (PNG or JPEG).
    For tile_format='3d', returns 3D tile data (e.g., tileset.json or binary 3D model data).

    1) Create a session appropriate for the tile format.
    2) Compute the tile X/Y for the given lat/lng at the zoom.
    3) GET the tile using your session token.

    Returns a tuple: (raw_tile_bytes, suggested_file_extension_including_dot)
    """
    session_payload = {
        "language": language,
        "region": region
    }
    api_url_tile_type_segment = ""
    suggested_extension = "" # e.g., ".png", ".jpg", ".json", ".glb"

    if tile_format == '3d':
        # For Photorealistic 3D Tiles:
        session_payload["mapType"] = "satellite"  # 'map_type' arg is effectively ignored for 3D
        session_payload["wantedFeatures"] = ["PHOTOREALISTIC_3D_TILES"] # Correct field name
        api_url_tile_type_segment = "3dtiles"
        # Extension will be determined after fetching content for 3D tiles
    elif tile_format == '2d':
        session_payload["mapType"] = map_type
        if map_type == "roadmap":
            session_payload["imageFormat"] = "PNG" # Explicitly request PNG
            suggested_extension = ".png"
        elif map_type == "satellite" or map_type == "terrain":
            session_payload["imageFormat"] = "JPEG" # Explicitly request JPEG
            suggested_extension = ".jpg"
        else:
            # Fallback if an unknown map_type is provided for 2D
            print(f"Warning: Unknown map_type '{map_type}' for 2D. Defaulting to roadmap behavior (PNG).")
            session_payload["mapType"] = "roadmap" # Fallback to a known type
            session_payload["imageFormat"] = "PNG"
            suggested_extension = ".png"
        api_url_tile_type_segment = "2dtiles"
    else:
        raise ValueError(f"Unsupported tile_format: {tile_format}. Use '2d' or '3d'.")

    # 1) createSession to get a session token
    print(f"Creating session with payload: {session_payload}")
    session_resp = requests.post(
        "https://tile.googleapis.com/v1/createSession",
        params={"key": API_KEY},
        json=session_payload
    )
    try:
        session_resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error during createSession:")
        print(f"  Status Code: {e.response.status_code}")
        print(f"  Response URL: {e.response.url}")
        print(f"  Response Text: {e.response.text}")
        raise
        
    session_data = session_resp.json()
    token = session_data["session"]
    print(f"Session created. Token: {token}, ExpiresIn: {session_data.get('expiry')}")

    # 2) lat/lng → tile X/Y in Web‑Mercator
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2.0 * n)

    # 3) fetch the tile
    tile_url = (
        f"https://tile.googleapis.com/v1/{api_url_tile_type_segment}/{zoom}/{xtile}/{ytile}"
    )
    print(f"Fetching tile from: {tile_url}")

    resp = requests.get(
        tile_url,
        params={
            "key": API_KEY,
            "session": token
        }
    )
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error during tile GET:")
        print(f"  Status Code: {e.response.status_code}")
        print(f"  Response URL: {e.response.url}")
        print(f"  Response Text: {e.response.text}")
        raise
        
    tile_content = resp.content

    # Refine suggested_extension for 3D tiles based on content
    if tile_format == '3d':
        try:
            # Check if it's JSON (likely tileset.json)
            json.loads(tile_content.decode('utf-8'))
            suggested_extension = ".json"
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, try to identify common 3D tile binary formats by magic numbers
            if tile_content[:4] == b'glTF':
                suggested_extension = ".glb"
            elif tile_content[:4] == b'b3dm':
                suggested_extension = ".b3dm"
            elif tile_content[:4] == b'i3dm':
                suggested_extension = ".i3dm"
            elif tile_content[:4] == b'pnts':
                suggested_extension = ".pnts"
            elif tile_content[:4] == b'cmpt':
                suggested_extension = ".cmpt"
            else:
                suggested_extension = ".bin" # Generic binary if not recognized

    return tile_content, suggested_extension


# --- Example Usage ---
# Remember to set your API_KEY above! If you haven't, the script will fail.
if API_KEY == "YOUR_API_KEY_HERE":
    print("ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual Google Maps API key.")
    exit()

# Example for 2D tile (saves as tile_2d.jpg for satellite)
try:
    print("Fetching 2D tile...")
    # Using map_type='satellite' which should now request JPEG
    img_bytes_2d, ext_2d = fetch_map_tile(33.4647676, -112.2397298, tile_format='2d', map_type='satellite', zoom=15)
    filename_2d = f"tile_2d{ext_2d}" # e.g., tile_2d.jpg
    with open(filename_2d, "wb") as f:
        f.write(img_bytes_2d)
    print(f"2D tile saved as {filename_2d}")
except requests.exceptions.HTTPError as e:
    print(f"HTTPError caught in example usage for 2D. Details should be above.")
except Exception as e:
    print(f"An unexpected error occurred with 2D: {e}")


# Example for 3D tile
try:
    print("\nFetching 3D tile...")
    # Using the lat/lon and zoom from the failing example
    data_3d, ext_3d = fetch_map_tile(33.4647676, -112.2397298, zoom=15, tile_format='3d')
    
    filename_3d = f"tile_3d_content{ext_3d}" # e.g., tile_3d_content.json or tile_3d_content.glb
    with open(filename_3d, "wb") as f:
        f.write(data_3d)
    print(f"3D tile content saved as {filename_3d}")
    print(f"Size: {len(data_3d)} bytes")

    if ext_3d == ".json":
        print(f"Content preview (first 200 chars): {data_3d.decode('utf-8')[:200]}...")
    elif ext_3d != ".bin": # Known binary format
        print(f"Content is a {ext_3d} 3D tile format.")
    else: # Generic binary
        print(f"Content is binary, first 16 bytes (hex): {data_3d[:16].hex()}")

except requests.exceptions.HTTPError as e:
    print(f"HTTPError caught in example usage for 3D. Details should be above.")
    # If you still get a 404 here, it might mean that specific 3D tile (X/Y/Z)
    # is not available, even if the session is correct.
    # You might need to fetch a root tileset.json at a lower zoom for the area first.
except Exception as e:
    print(f"An unexpected error occurred with 3D: {e}")