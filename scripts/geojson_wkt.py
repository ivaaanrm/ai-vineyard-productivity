import geopandas as gpd
from shapely.geometry import shape

geojson = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [1.498780776343068, 41.379381629304504],
            [1.498780776343068, 41.36963213136582],
            [1.5122740758273778, 41.36963213136582],
            [1.5122740758273778, 41.379381629304504],
            [1.498780776343068, 41.379381629304504]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}

# Convert GeoJSON feature to shapely geometry
geom = shape(geojson["features"][0]["geometry"])

# Create GeoDataFrame with CRS WGS84
gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

# Reproject to EPSG:3857
gdf_3857 = gdf.to_crs("EPSG:3857")

# Convert to WKT
wkt_geom = gdf_3857.geometry.iloc[0].wkt

print(wkt_geom)