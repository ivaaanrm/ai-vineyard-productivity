from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Protocol

import numpy as np
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import Affine

if TYPE_CHECKING:
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

    def __init__(self, ds: xr.Dataset, sensor: str, parcel_key: str) -> None:
        self.sensor = sensor
        self.parcel_key = parcel_key
        # Accept raw 4D zarr format (time, variable, y, x) or standard Dataset
        if "data" in ds.data_vars and "variable" in ds["data"].dims:
            self._ds = _unpack_zarr(ds)
        else:
            self._ds = ds

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

    def mask(self, geometry) -> "SampleCube":
        """Return a new SampleCube with pixels outside *geometry* set to NaN.

        Args:
            geometry: A WKT string (``"POLYGON ((...))"``) or a Shapely
                      geometry in the same CRS as the cube.  Anything that
                      implements ``__geo_interface__`` is also accepted
                      (e.g. a GeoDataFrame row's geometry).

        Returns:
            A new SampleCube; the original is not modified.
        """
        if isinstance(geometry, str):
            from shapely import wkt
            geometry = wkt.loads(geometry)
        x = self._ds.coords["x"].values
        y = self._ds.coords["y"].values
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])  # negative when y runs north→south

        # Build affine so pixel (col=0, row=0) centres on (x[0], y[0])
        transform = Affine(dx, 0.0, float(x[0]) - dx / 2,
                           0.0, dy, float(y[0]) - dy / 2)

        valid = geometry_mask(
            [geometry],
            out_shape=(len(y), len(x)),
            transform=transform,
            invert=True,  # True = inside polygon (keep), False = outside (mask)
        )
        valid_da = xr.DataArray(valid, dims=["y", "x"],
                                coords={"y": y, "x": x})
        return SampleCube(self._ds.where(valid_da),
                          sensor=self.sensor, parcel_key=self.parcel_key)

    def replace_band(self, name: str, new_da: xr.DataArray) -> None:
        """Replace an existing band's data in place."""
        self._ds[name] = new_da

    def compute_indices(self, calculators: List[IndexCalculator]) -> None:
        """Compute indices and append them as new data variables. Mutates in place.

        Skips calculators whose required bands are missing or whose output
        name already exists as a variable.
        """
        for calc in calculators:
            if calc.name in self._ds.data_vars:
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

    Expected layout: {base_path}/{SENSOR}/{parcel_key}/cube.zarr
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)

    def load(self, parcel_key: str, sensor: str) -> SampleCube:
        zarr_path = self.base_path / sensor / parcel_key / "cube.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
        ds = xr.open_zarr(str(zarr_path), consolidated=False)
        return SampleCube(ds, sensor=sensor, parcel_key=parcel_key)
