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
        self.is_normalized = False

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

    def drop_duplicate_times(self) -> "SampleCube":
        """Drop duplicate timestamps, keeping the first occurrence.

        Duplicate acquisitions can appear when split zarr cubes overlap or when
        the same tile is ingested twice.  Returns ``self`` when no duplicates
        are found (zero-cost path).
        """
        times = self.times
        _, unique_idx = np.unique(times, return_index=True)
        n_dropped = len(times) - len(unique_idx)
        if n_dropped == 0:
            return self
        # print(
        #     f"   🔁  Dedup ({self.parcel_key}): dropped {n_dropped} duplicate "
        #     f"timestep(s) out of {len(times)}"
        # )
        return SampleCube(
            self._ds.isel(time=np.sort(unique_idx)),
            sensor=self.sensor,
            parcel_key=self.parcel_key,
        )

    def drop_corrupt_times(self, max_nan_fraction: float = 0.8) -> "SampleCube":
        """Drop timesteps where the fraction of NaN pixels exceeds *max_nan_fraction*.

        Fully corrupt or entirely clouded acquisitions appear as all-NaN frames.
        A threshold of 0.8 (80%) catches broken images while tolerating small
        patches of missing data at tile edges.

        Returns ``self`` unchanged when no timesteps would survive the filter.
        """
        if not self._ds.data_vars:
            return self

        stacked = self._ds.to_array(dim="_band")  # (band, time, y, x)
        nan_frac = stacked.isnull().mean(dim=["_band", "y", "x"])  # (time,)

        good_times = nan_frac.time.where(nan_frac <= max_nan_fraction, drop=True)
        n_dropped = len(self.times) - len(good_times)
        # if n_dropped > 0:
        #     print(
        #         f"   🗑  Corrupt filter ({self.parcel_key}): dropped {n_dropped} / "
        #         f"{len(self.times)} timesteps (NaN fraction > {max_nan_fraction:.0%})"
        #     )
        if len(good_times) == 0:
            # print(
            #     f"   ⚠  Corrupt filter: all timesteps dropped for {self.parcel_key}"
            #     " — keeping original."
            # )
            return self

        return SampleCube(
            self._ds.sel(time=good_times), sensor=self.sensor, parcel_key=self.parcel_key
        )

    def drop_cloudy_times(self, ndvi_threshold: float = 0.1) -> "SampleCube":
        """Drop timesteps where the spatial-mean NDVI is below *ndvi_threshold*.

        A fully-clouded acquisition has no vegetation signal — NDVI collapses
        toward 0 (or negative).  Bare soil sits around 0.1–0.2, so the default
        threshold of 0.1 catches cloud-only frames while preserving winter/bare
        scenes.

        Requires B08 and B04 to be present and already in reflectance units
        (i.e. ``preprocess_s2`` must have run first).  Returns ``self`` unchanged
        when the bands are absent so the pipeline degrades gracefully.
        """
        if not (self.has_band("B08") and self.has_band("B04")):
            return self

        nir = self._ds["B08"].astype("float32")
        red = self._ds["B04"].astype("float32")
        denom = nir + red
        ndvi = ((nir - red) / denom.where(denom != 0)).mean(dim=["y", "x"])

        good_times = ndvi.time.where(ndvi >= ndvi_threshold, drop=True)
        n_dropped = len(self.times) - len(good_times)
        # if n_dropped > 0:
        #     print(
        #         f"   ☁️  Cloud filter ({self.parcel_key}): dropped {n_dropped} / "
        #         f"{len(self.times)} timesteps (mean NDVI < {ndvi_threshold})"
        #     )
        if len(good_times) == 0:
            # print(f"   ⚠️  Cloud filter: all timesteps dropped for {self.parcel_key} — keeping original.")
            return self

        ds_filtered = self._ds.sel(time=good_times)
        return SampleCube(ds_filtered, sensor=self.sensor, parcel_key=self.parcel_key)

    def composite_temporal(self, config: TemporalConfig) -> "SampleCube":
        ds = self._ds

        if config.resample is not None and config.rolling is not None:
            # Resample to a regular grid then apply per-year rolling smoothing
            agg = config.resample.agg
            if isinstance(agg, list):
                agg = agg[0]
            ds = smooth_dataset(ds, config.resample.freq, config.rolling.window, agg)
        elif config.resample is not None:
            # Resample only (no rolling) — used for cumulative variables like precipitation
            agg = config.resample.agg
            if isinstance(agg, list):
                agg = agg[0]
            agg_by_var = config.resample.agg_by_variable or {}
            if agg_by_var:
                # Apply per-variable aggregation then merge back into a Dataset
                resampled_vars = {}
                for vname in ds.data_vars:
                    var_agg = agg_by_var.get(vname, agg)
                    resampled_vars[vname] = getattr(
                        ds[vname].resample(time=config.resample.freq), var_agg
                    )()
                ds = xr.Dataset(resampled_vars)
            else:
                resampler = ds.resample(time=config.resample.freq)
                if agg == "sum":
                    ds = resampler.sum()
                elif agg == "mean":
                    ds = resampler.mean()
                elif agg == "median":
                    ds = resampler.median()
                else:
                    ds = getattr(resampler, agg)()
        else:
            raise ValueError("TemporalConfig must specify at least resample.")

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
