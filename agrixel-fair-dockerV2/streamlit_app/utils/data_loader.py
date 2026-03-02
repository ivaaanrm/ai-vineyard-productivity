"""Data loading utilities for AOI table, GeoTIFFs, Zarr cubes, and quality JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import transform

# ---------------------------------------------------------------------------
# Paths (relative to the agrixel-fair-dockerV2 root)
# ---------------------------------------------------------------------------

_DOCKER_ROOT = Path(__file__).resolve().parents[2]
AOI_CSV = _DOCKER_ROOT / "data" / "input" / "aoi_table.csv"
OUTPUT_FILES = _DOCKER_ROOT / "data" / "output" / "files"

SENSOR_FOLDERS = {
    "S1": "SENTINEL-1",
    "S2": "SENTINEL-2",
}

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI"]
S1_POLARIZATIONS = ["VV", "VH"]


# ---------------------------------------------------------------------------
# AOI table
# ---------------------------------------------------------------------------


def load_aoi_table() -> pd.DataFrame:
    """Load aoi_table.csv and parse WKT geometries to Shapely objects."""
    df = pd.read_csv(AOI_CSV, encoding="utf-8-sig")
    df["geom"] = df["geometry"].apply(wkt.loads)
    return df


def reproject_geom_to_4326(geom, src_epsg: int = 3857):
    """Reproject a Shapely geometry from *src_epsg* to EPSG:4326."""
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326", always_xy=True)
    return transform(transformer.transform, geom)


def parcel_centroid_4326(df: pd.DataFrame) -> tuple[float, float]:
    """Return (lat, lon) centroid of all parcels (for map centering)."""
    geoms_4326 = [reproject_geom_to_4326(g) for g in df["geom"]]
    lats = [g.centroid.y for g in geoms_4326]
    lons = [g.centroid.x for g in geoms_4326]
    return float(np.mean(lats)), float(np.mean(lons))


def parcel_geojson_4326(geom) -> dict:
    """Convert a single EPSG:3857 geometry to a GeoJSON dict in EPSG:4326."""
    return mapping(reproject_geom_to_4326(geom))


# ---------------------------------------------------------------------------
# Output directory scanning
# ---------------------------------------------------------------------------


def list_parcels_with_data(sensor_key: str) -> list[str]:
    """Return parcel IDs that have downloaded data for the given sensor.

    In non-batch mode the scenes sit directly under the sensor folder
    (no parcel subdirectory), so we return ``["(single)"]`` in that case.
    """
    sensor_dir = OUTPUT_FILES / SENSOR_FOLDERS.get(sensor_key, sensor_key)
    if not sensor_dir.is_dir():
        return []

    # Batch mode: subdirectories whose names look like parcel IDs (Lxxx)
    parcel_dirs = [
        d.name
        for d in sorted(sensor_dir.iterdir())
        if d.is_dir() and d.name.startswith("L")
    ]
    if parcel_dirs:
        return parcel_dirs

    # Single-mode: scenes sit directly under the sensor folder
    scene_dirs = [
        d.name
        for d in sorted(sensor_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]
    if scene_dirs:
        return ["(single)"]
    return []


def list_scenes(sensor_key: str, parcel_id: str) -> list[str]:
    """Return scene directory names for a given sensor + parcel."""
    sensor_dir = OUTPUT_FILES / SENSOR_FOLDERS.get(sensor_key, sensor_key)
    if parcel_id == "(single)":
        base = sensor_dir
    else:
        base = sensor_dir / parcel_id
    if not base.is_dir():
        return []
    return sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "cube.zarr"
    )


def scene_path(sensor_key: str, parcel_id: str, scene_name: str) -> Path:
    """Return the absolute path to a scene directory."""
    sensor_dir = OUTPUT_FILES / SENSOR_FOLDERS.get(sensor_key, sensor_key)
    if parcel_id == "(single)":
        return sensor_dir / scene_name
    return sensor_dir / parcel_id / scene_name


# ---------------------------------------------------------------------------
# GeoTIFF loading
# ---------------------------------------------------------------------------


def list_bands(scene_dir: Path) -> list[str]:
    """Return band/variable names available in a scene directory."""
    bands = []
    for f in sorted(scene_dir.glob("*.tif")):
        # Band name is the prefix before the first underscore that starts
        # a scene ID (e.g.  B04_S2A_MSIL2A... or NDVI_S2A_MSIL2A...)
        name = f.stem.split("_")[0]
        if name not in bands:
            bands.append(name)
    return bands


def load_geotiff(scene_dir: Path, band: str) -> tuple[np.ndarray, dict]:
    """Load the first matching GeoTIFF for *band* and return (array, profile).

    The profile contains CRS, transform, bounds, etc.
    """
    matches = sorted(scene_dir.glob(f"{band}_*.tif"))
    if not matches:
        raise FileNotFoundError(f"No GeoTIFF for band {band!r} in {scene_dir}")
    with rasterio.open(matches[0]) as src:
        data = src.read(1).astype(np.float32)
        profile = dict(src.profile)
        profile["bounds"] = src.bounds
    return data, profile


def geotiff_bounds_4326(profile: dict) -> list[list[float]]:
    """Return [[south, west], [north, east]] in EPSG:4326 for Folium."""
    bounds = profile["bounds"]
    crs = profile.get("crs")
    if crs and str(crs) != "EPSG:4326":
        transformer = Transformer.from_crs(str(crs), "EPSG:4326", always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    return [[south, west], [north, east]]


# ---------------------------------------------------------------------------
# Quality JSON
# ---------------------------------------------------------------------------


def load_quality_json(scene_dir: Path) -> dict | None:
    """Load the QUALITY_*.json scene summary if it exists."""
    matches = sorted(scene_dir.glob("QUALITY_*.json"))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def load_band_quality(scene_dir: Path, band: str) -> dict | None:
    """Load per-band quality JSON (<band>_*.quality.json)."""
    matches = sorted(scene_dir.glob(f"{band}_*.quality.json"))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Zarr time-series
# ---------------------------------------------------------------------------


def find_zarr_store(sensor_key: str, parcel_id: str) -> Path | None:
    """Return the path to cube.zarr for the given sensor/parcel, or None."""
    sensor_dir = OUTPUT_FILES / SENSOR_FOLDERS.get(sensor_key, sensor_key)
    if parcel_id == "(single)":
        zarr_path = sensor_dir / "cube.zarr"
    else:
        zarr_path = sensor_dir / parcel_id / "cube.zarr"
    return zarr_path if zarr_path.is_dir() else None


def load_zarr_cube(zarr_path: Path) -> xr.Dataset:
    """Open a Zarr store as an xarray Dataset."""
    return xr.open_zarr(zarr_path)
