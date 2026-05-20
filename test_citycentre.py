import geopandas as gpd
from shapely.geometry import Point

# Load CBSA polygons
cbsa = gpd.read_file('/Users/dhruvsood/Downloads/cb_2018_us_cbsa_500k (1)/cb_2018_us_cbsa_500k.shp')

# Example coordinates
lat, lon = 32.7767, -96.7970

point = Point(lon, lat)

# Find metro area
match = cbsa[cbsa.contains(point)]

print(match[["NAME", "CBSAFP"]])