"""Generate a rectangular bounding box polygon from a WKT geometry (EPSG:3857 → GeoJSON EPSG:4326)."""

import json

import geopandas as gpd
from shapely import wkt
from shapely.geometry import box

# Margin in meters (EPSG:3857 units) added on each side of the bounding box
MARGIN = 1500

# WKT geometry in EPSG:3857 (Web Mercator)
wkt_str = (
    """POLYGON ((33613.334949522614 5031722.596604069, 33593.068685444116 5031760.739927518, 33590.67069747535 5031764.30780794, 33582.12387646556 5031780.152379262, 33553.59052553799 5031809.822338418, 33552.564242671506 5031824.787427334, 33588.15822190977 5031841.696239057, 33595.89635180184 5031845.36461716, 33603.09031570816 5031831.167560591, 33606.33086520837 5031823.671257508, 33615.80898263514 5031801.745421871, 33625.147046367696 5031784.127834359, 33637.36185590046 5031764.727762703, 33651.41014650662 5031739.2080491325, 33659.04451909826 5031728.130445795, 33660.843383302534 5031725.089440129, 33661.34752061753 5031724.237146061, 33661.98713958741 5031723.155798068, 33621.58328167979 5031706.987374397, 33614.15605046134 5031720.7630574675, 33613.334949522614 5031722.596604069))"""
)

# Parse WKT to shapely geometry and get bounding box
geom = wkt.loads(wkt_str)
minx, miny, maxx, maxy = geom.bounds

# Expand bbox by margin on each side
bbox_geom = box(minx - MARGIN, miny - MARGIN, maxx + MARGIN, maxy + MARGIN)

# Create GeoDataFrame with CRS EPSG:3857
gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:3857")

# Reproject to EPSG:4326 (WGS84)
gdf_4326 = gdf.to_crs("EPSG:4326")
gdf_3857 = gdf.to_crs("EPSG:3857")
# Convert to GeoJSON
geojson = json.loads(gdf_4326.to_json())

print(json.dumps(geojson, indent=2))

# WKT output in EPSG:4326
print("\nWKT (EPSG:3857):")
print(gdf_3857.geometry[0].wkt)
