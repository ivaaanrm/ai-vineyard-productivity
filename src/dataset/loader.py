from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Protocol

import numpy as np
import rioxarray  # noqa: F401 (registers .rio accessor)
import xarray as xr
from shapely import wkt

from .temporal import smooth_dataset

if TYPE_CHECKING:
    from .config import TemporalConfig
    from .sensors import IndexCalculator


def _unpack_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Convert 4D (time, variable, y, x) zarr format to a standard Dataset.

    Each value along the ``variable`` dimension becomes its own data variable
    with dimensions (time, y, x), matching the canonical xarray Dataset layout.
    """
    data = ds["data"]
    variables = {}
    for v in data.coords["variable"].values:
        name = v.decode() if isinstance(v, bytes) else str(v)
        variables[name] = data.sel(variable=v).drop_vars("variable")
    return xr.Dataset(variables, attrs=ds.attrs)


class SampleCube:
    """Spatio-temporal cube for a single parcel/sensor pair.

    Internally backed by a standard ``xr.Dataset`` with dimensions
    ``(time, y, x)`` and one data variable per band/index, so you can
    use the full xarray API directly via ``cube.ds``::

        cube.ds.NDVI.mean(["x", "y"]).plot.line("b-^", figsize=(11, 4))
    """

    def __init__(
        self, ds: xr.Dataset, sensor: str, parcel_key: str, parcel_mask: str = None
    ) -> None:
        self.sensor = sensor
        self.parcel_key = parcel_key
        # Accept raw 4D zarr format (time, variable, y, x) or standard Dataset
        if "data" in ds.data_vars and "variable" in ds["data"].dims:
            self._ds = _unpack_zarr(ds)
        else:
            self._ds = ds
        self.parcel_mask = parcel_mask

    @property
    def ds(self) -> xr.Dataset:
        """Standard xarray Dataset — each band is a (time, y, x) data variable."""
        return self._ds

    @property
    def variables(self) -> List[str]:
        return list(self._ds.data_vars)

    @property
    def times(self) -> np.ndarray:
        return self._ds.coords["time"].values

    def band(self, name: str) -> xr.DataArray:
        """Return (time, y, x) DataArray for a named band."""
        return self._ds[name]

    def has_band(self, name: str) -> bool:
        return name in self._ds.data_vars

    def mask(self, geometry, erosion_pixels: float = 0.0) -> "SampleCube":
        """Return a new SampleCube with pixels outside *geometry* set to NaN.

        Both the cube and geometry are expected to share the same CRS
        (EPSG:3857 as produced by the agrixel pipeline).

        When the geometry falls outside the cube extent (common for
        coarse-resolution sensors like S3 / ERA5 / MODIS), the nearest
        pixel to the geometry centroid is kept instead.

        Args:
            geometry: WKT string or Shapely geometry to use for clipping.
            erosion_pixels: Number of pixels to erode the geometry inward (default 0).
                When > 0, the geometry is shrunk inward by this many pixels before
                clipping, to exclude border pixels that partially cover non-parcel area.
        """
        if isinstance(geometry, str):
            geometry = wkt.loads(geometry)

        ds = self._ds
        ds = ds.rio.write_crs("EPSG:3857")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

        x_res = abs(float(ds.coords["x"][1] - ds.coords["x"][0]))

        # Inward erosion to exclude border pixels (if requested)
        if erosion_pixels > 0:
            eroded = geometry.buffer(-erosion_pixels * x_res)
            clip_geom = eroded if not eroded.is_empty else geometry
        else:
            # No erosion: expand outward by one pixel so small parcels aren't lost
            clip_geom = geometry.buffer(x_res)

        # Try clipping; fall back to nearest pixel if geometry doesn't
        # intersect the cube (e.g. coarse-resolution sensors).
        clipped = None
        try:
            clipped = ds.rio.clip(
                [clip_geom], crs="EPSG:3857", drop=True, all_touched=True
            )
        except Exception:
            pass

        if clipped is None or bool(clipped.to_array().isnull().all()):
            x = ds.coords["x"].values
            y = ds.coords["y"].values
            cx, cy = geometry.centroid.x, geometry.centroid.y
            ix = int((abs(x - cx)).argmin())
            iy = int((abs(y - cy)).argmin())
            mask_arr = xr.zeros_like(ds[list(ds.data_vars)[0]].isel(time=0), dtype=bool)
            mask_arr[iy, ix] = True
            clipped = ds.where(mask_arr)

        return SampleCube(clipped, sensor=self.sensor, parcel_key=self.parcel_key)

    def replace_band(self, name: str, new_da: xr.DataArray) -> None:
        """Replace an existing band's data in place."""
        self._ds[name] = new_da

    def composite_temporal(self, config: TemporalConfig) -> "SampleCube":
        ds = self._ds

        # If both resample and rolling are configured, use smooth (resample + per-year rolling)
        if config.resample is not None and config.rolling is not None:
            agg = config.resample.agg
            if isinstance(agg, list):
                agg = agg[0]
            ds = smooth_dataset(ds, config.resample.freq, config.rolling.window, agg)
        else:
            raise Exception("Rolling not configured")


        return SampleCube(ds, sensor=self.sensor, parcel_key=self.parcel_key)

    def compute_indices(self, calculators: List[IndexCalculator]) -> None:
        """Compute indices and append them as new data variables. Mutates in place.

        Skips calculators whose required bands are missing or whose output
        name already exists as a variable.
        """
        for calc in calculators:
            if (
                calc.name in self._ds.data_vars
                and not self._ds[calc.name].isnull().all()
            ):
                continue
            if not calc.supports(self):
                continue
            self._ds[calc.name] = calc.compute(self)

    def __str__(self) -> str:
        return str(self._ds)

    def __repr__(self) -> str:
        return (
            f"SampleCube(sensor={self.sensor!r}, parcel={self.parcel_key!r}, "
            f"variables={self.variables}, n_times={len(self.times)})"
        )


class SampleLoaderProtocol(Protocol):
    """Protocol for loading band cubes — implement this to plug in a new data source."""

    def load(self, parcel_key: str, sensor: str) -> SampleCube: ...


class ZarrBandLoader:
    """Adapter that reads cube.zarr stores from the agrixel output directory structure.

    Expected layout:
      - {base_path}/{SENSOR}/{parcel_key}/cube.zarr          (single cube)
      - {base_path}/{SENSOR}/{parcel_key}/*/cube.zarr         (split cubes, e.g. ERA5)
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)

    def load(self, parcel_key: str, sensor: str) -> SampleCube:
        parcel_dir = self.base_path / sensor / parcel_key
        zarr_path = parcel_dir / "cube.zarr"
        if zarr_path.exists():
            ds = xr.open_zarr(str(zarr_path), consolidated=False)
            return SampleCube(ds, sensor=sensor, parcel_key=parcel_key)
        # Fallback: look for split cubes in subdirectories (e.g. ERA5 time windows)
        sub_zarrs = sorted(parcel_dir.glob("*/cube.zarr"))
        if not sub_zarrs:
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
        datasets = [xr.open_zarr(str(p), consolidated=False) for p in sub_zarrs]
        ds = xr.concat(datasets, dim="time")
        # Drop duplicate timestamps from overlapping split cubes
        _, unique_idx = np.unique(ds.time.values, return_index=True)
        if len(unique_idx) < ds.sizes["time"]:
            ds = ds.isel(time=np.sort(unique_idx))
        return SampleCube(ds, sensor=sensor, parcel_key=parcel_key)
