"""Band cube loader — Protocol definition and Zarr adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Protocol

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .sensors import IndexCalculator

class SampleCube:
    """Wraps a (time, variable, y, x) xarray Dataset for a parcel+sensor combination.

    Variable coordinates are stored as bytes in the zarr files; this class
    handles the encoding transparently so callers always use plain strings.
    """

    def __init__(self, ds: xr.Dataset, sensor: str, parcel_key: str) -> None:
        self._ds = ds
        self.sensor = sensor
        self.parcel_key = parcel_key
        # Decode byte-string variable coords to plain strings
        raw = ds.coords["variable"].values
        self._variables: List[str] = [
            v.decode() if isinstance(v, bytes) else str(v) for v in raw
        ]
    
    @property
    def ds(self):
        return self._ds

    @property
    def variables(self) -> List[str]:
        return self._variables

    @property
    def times(self) -> np.ndarray:
        return self._ds.coords["time"].values

    def band(self, name: str) -> xr.DataArray:
        """Return (time, y, x) DataArray for a named band."""
        # zarr stores variable names as bytes
        key = name.encode() if name in self._variables else name
        return self._ds["data"].sel(variable=key)

    def has_band(self, name: str) -> bool:
        return name in self._variables

    def replace_band(self, name: str, new_da: xr.DataArray) -> None:
        """Replace an existing band's data (loads lazy data into memory)."""
        key = name.encode() if name in self._variables else name
        var_list = list(self._ds.coords["variable"].values)
        idx = var_list.index(key)
        data = np.array(self._ds["data"].values)  # (time, variable, y, x)
        data[:, idx, :, :] = new_da.values
        new_arr = xr.DataArray(
            data, dims=self._ds["data"].dims, coords=self._ds["data"].coords
        )
        self._ds = xr.Dataset({"data": new_arr}, attrs=self._ds.attrs)

    def compute_indices(self, calculators: List[IndexCalculator]) -> None:
        """Compute indices and append them as new variables in the dataset.

        Skips calculators whose required bands are missing or whose output
        name already exists as a variable. Mutates the cube in place.
        """
        for calc in calculators:
            if calc.name in self._variables:
                continue
            if not calc.supports(self):
                continue
            da = calc.compute(self)  # (time, y, x)
            new_slice = xr.DataArray(
                da.values[:, np.newaxis, :, :],
                dims=["time", "variable", "y", "x"],
                coords={
                    "time": self._ds.coords["time"],
                    "variable": [calc.name.encode()],
                    "y": self._ds.coords["y"],
                    "x": self._ds.coords["x"],
                },
            )
            new_data = xr.concat([self._ds["data"], new_slice], dim="variable")
            self._variables.append(calc.name)
            # Rebuild dataset and keep attrs in sync with the updated variable List
            self._ds = xr.Dataset(
                {"data": new_data},
                attrs={**self._ds.attrs, "variables": list(self._variables)},
            )

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
