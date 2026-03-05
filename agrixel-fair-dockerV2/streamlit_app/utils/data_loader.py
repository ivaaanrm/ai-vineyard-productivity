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
    "S3": "SENTINEL-3",
    "ERA5": "ERA5",
    "MODIS": "MODIS",
}

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI"]
S1_POLARIZATIONS = ["VV", "VH"]
S3_PRODUCTS = ["lst-in"]
MODIS_PRODUCTS = ["ET_500m", "PET_500m", "LE_500m", "PLE_500m", "ET_QC_500m"]
ERA5_VARIABLES = {
    "tp": "Precipitación total",
    "t2m": "Temperatura a 2 m",
    "d2m": "Punto de rocío a 2 m",
    "u10": "Viento U a 10 m",
    "v10": "Viento V a 10 m",
}
ERA5_AGGREGATIONS = ["sum", "mean", "min", "max"]


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


def _extract_band_name(stem: str) -> str:
    """Extract band/product name from a filename stem.

    Handles different naming conventions:
    - S1/S2: B04_S2A_MSIL2A... → B04, NDVI_S2A_... → NDVI
    - MODIS: ET_500m_granule_... → ET_500m, ET_QC_500m_granule_... → ET_QC_500m
    - S3: lst-in_S3A_... → lst-in
    - ERA5: tp_ERA5L_... → tp

    Strategy: split on known scene-id prefixes (S1A_, S2A_, S2B_, S3A_, S3B_,
    granule_, ERA5L_, __tmp_).
    """
    import re

    m = re.split(r"_(S[123][AB]_|granule_|ERA5L_|__tmp_)", stem, maxsplit=1)
    return m[0]


def list_bands(scene_dir: Path) -> list[str]:
    """Return band/variable names available in a scene directory.

    Supports GeoTIFF (.tif) for S1/S2/MODIS and NetCDF (.nc) for S3/ERA5.
    Adds a virtual "RGB" band when B02, B03, and B04 are all present.
    """
    bands = []
    for f in sorted(scene_dir.glob("*.tif")):
        name = _extract_band_name(f.stem)
        if name not in bands:
            bands.append(name)
    for f in sorted(scene_dir.glob("*.nc")):
        name = _extract_band_name(f.stem)
        if name not in bands:
            bands.append(name)
    # Add virtual RGB composite when all three visible bands are available
    if all(b in bands for b in ("B02", "B03", "B04")):
        bands.insert(0, "RGB")
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


def load_netcdf(scene_dir: Path, band: str) -> tuple[np.ndarray, dict]:
    """Load the first matching NetCDF for *band* and return (array, profile).

    Returns a synthetic profile dict compatible with the GeoTIFF profile
    so map overlay code works uniformly.
    """
    matches = sorted(scene_dir.glob(f"{band}_*.nc"))
    if not matches:
        raise FileNotFoundError(f"No NetCDF for band {band!r} in {scene_dir}")
    ds = xr.open_dataset(matches[0])
    # Find the primary data variable (skip coordinate vars)
    data_vars = [v for v in ds.data_vars if v not in ("x", "y", "lat", "lon", "spatial_ref")]
    if not data_vars:
        raise FileNotFoundError(f"No data variables found in {matches[0]}")
    var_name = data_vars[0]
    arr = ds[var_name].values
    # If 3D (time, y, x), take first time step
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float32)

    # Build a synthetic profile for map overlay
    # Detect coordinate names (x/y or lon/lat)
    if "x" in ds.coords and "y" in ds.coords:
        x_vals = ds["x"].values
        y_vals = ds["y"].values
    elif "lon" in ds.coords and "lat" in ds.coords:
        x_vals = ds["lon"].values
        y_vals = ds["lat"].values
    else:
        x_vals = y_vals = None

    if x_vals is not None and y_vals is not None:
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        res_x = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.0001
        res_y = float(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else -0.0001
        west = float(x_vals.min()) - abs(res_x) / 2
        east = float(x_vals.max()) + abs(res_x) / 2
        south = float(y_vals.min()) - abs(res_y) / 2
        north = float(y_vals.max()) + abs(res_y) / 2
        t = from_bounds(west, south, east, north, arr.shape[1], arr.shape[0])
        profile = {
            "crs": CRS.from_epsg(4326),
            "transform": t,
            "height": arr.shape[0],
            "width": arr.shape[1],
            "bounds": rasterio.coords.BoundingBox(west, south, east, north),
        }
    else:
        profile = {"height": arr.shape[0], "width": arr.shape[1]}
    ds.close()
    return arr, profile


def load_raster(scene_dir: Path, band: str) -> tuple[np.ndarray, dict]:
    """Load a raster (GeoTIFF or NetCDF) for the given band.

    Tries GeoTIFF first, falls back to NetCDF.
    """
    tif_matches = sorted(scene_dir.glob(f"{band}_*.tif"))
    if tif_matches:
        return load_geotiff(scene_dir, band)
    return load_netcdf(scene_dir, band)


def load_rgb_composite(scene_dir: Path) -> tuple[np.ndarray, dict]:
    """Load B04, B03, B02 and stack them into an (H, W, 3) uint8 RGB array.

    Returns (rgb_array, profile) where profile comes from B04.
    """
    r, profile = load_geotiff(scene_dir, "B04")
    g, _ = load_geotiff(scene_dir, "B03")
    b, _ = load_geotiff(scene_dir, "B02")

    rgb = np.stack([r, g, b], axis=-1)
    # Mask nodata (any band NaN → transparent)
    nodata_mask = np.isnan(r) | np.isnan(g) | np.isnan(b)
    # Clip to 2nd–98th percentile for contrast stretch
    valid = rgb[~nodata_mask]
    if valid.size > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        if vmax > vmin:
            rgb = (rgb - vmin) / (vmax - vmin)
        else:
            rgb = np.zeros_like(rgb)
    rgb = np.clip(rgb, 0, 1)
    # Set nodata pixels to NaN for downstream transparency handling
    rgb[nodata_mask] = np.nan
    return rgb, profile


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
    return xr.open_zarr(zarr_path, consolidated=False)
