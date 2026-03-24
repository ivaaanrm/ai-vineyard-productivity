"""Convert WKT (EPSG:3857) to GeoJSON (EPSG:4326)."""

import json

import geopandas as gpd
from shapely import wkt

# WKT geometry in EPSG:3857 (Web Mercator)
wkt_str = (
    "POLYGON (("
    "166832.71684810915 5072047.526498498, "
    "166832.71684810915 5070961.819498498, "
    "168349.77994310915 5070961.819498498, "
    "168349.77994310915 5072047.526498498, "
    "166832.71684810915 5072047.526498498))"
)

# Parse WKT to shapely geometry
geom = wkt.loads(wkt_str)

# Create GeoDataFrame with CRS EPSG:3857
gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:3857")

# Reproject to EPSG:4326 (WGS84)
gdf_4326 = gdf.to_crs("EPSG:4326")

# Convert to GeoJSON
geojson = json.loads(gdf_4326.to_json())

print(json.dumps(geojson, indent=2))
