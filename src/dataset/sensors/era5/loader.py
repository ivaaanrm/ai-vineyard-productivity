"""ERA5 reanalysis timeseries loader.

Reads per-parcel ZIP archives downloaded from the Copernicus Climate Data Store (CDS).
Each archive contains two NetCDF4 files:
  - 2-metre air temperature (``t2m``, Kelvin)
  - total precipitation    (``tp``,  metres)

Hourly values are aggregated to daily before the SampleCube is returned:
  - ``t2m``: daily mean
  - ``tp``:  daily sum

Expected directory layout::

    {base_path}/{parcel_key}/{parcel_key}_{year}_era5.nc   ← ZIP archive

The SampleCube dimensions are ``(time, y=1, x=1)`` where *y* and *x* carry the
ERA5 grid-point latitude/longitude as coordinate values.  Because each parcel
is represented by a single point, set ``parcel_mask: false`` in the dataset
config so that geometry clipping is skipped.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from ...loader import SampleCube


class ERA5TimeseriesLoader:
    """Adapter that loads per-parcel ERA5 ZIP archives into SampleCubes.

    Parameters
    ----------
    base_path:
        Root directory that contains one sub-directory per parcel, e.g.
        ``data/output/files/ERA5_TIMESERIES``.
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Public interface (satisfies SampleLoaderProtocol)
    # ------------------------------------------------------------------

    def load(self, parcel_key: str, sensor: str = "ERA5") -> SampleCube:
        """Return a daily SampleCube for *parcel_key*.

        All yearly archives found in ``{base_path}/{parcel_key}/`` are
        loaded and concatenated along the time axis.

        Raises
        ------
        FileNotFoundError
            When the parcel directory or its archives are missing.
        """
        parcel_dir = self.base_path / parcel_key
        if not parcel_dir.exists():
            raise FileNotFoundError(
                f"ERA5 timeseries directory not found: {parcel_dir}"
            )

        zip_files = sorted(parcel_dir.glob("*_era5.nc"))
        if not zip_files:
            raise FileNotFoundError(
                f"No ERA5 timeseries ZIP archives found in: {parcel_dir}"
            )

        yearly_datasets = [self._load_zip(zf) for zf in zip_files]
        ds = xr.concat(yearly_datasets, dim="time")

        # Drop duplicate timestamps that may arise from overlapping year windows
        _, unique_idx = np.unique(ds.time.values, return_index=True)
        if len(unique_idx) < ds.sizes["time"]:
            ds = ds.isel(time=np.sort(unique_idx))

        return SampleCube(ds, sensor=sensor, parcel_key=parcel_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_zip(self, zip_path: Path) -> xr.Dataset:
        """Read one year's ZIP archive and return a daily-aggregated Dataset."""
        t2m_raw: Optional[xr.Dataset] = None
        tp_raw: Optional[xr.Dataset] = None

        with zipfile.ZipFile(zip_path) as z:
            for entry_name in z.namelist():
                raw_nc = self._extract_nc(z, entry_name)
                if "t2m" in raw_nc.data_vars:
                    t2m_raw = raw_nc
                elif "tp" in raw_nc.data_vars:
                    tp_raw = raw_nc

        return self._build_daily_dataset(t2m_raw, tp_raw)

    @staticmethod
    def _extract_nc(z: zipfile.ZipFile, entry_name: str) -> xr.Dataset:
        """Extract one NetCDF4 entry to a temp file, open it, and delete the temp."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(z.read(entry_name))
            tmp_path = tmp.name
        try:
            ds = xr.open_dataset(tmp_path, engine="netcdf4")
            ds.load()  # pull into memory before temp file is removed
        finally:
            os.unlink(tmp_path)
        return ds

    @staticmethod
    def _build_daily_dataset(
        t2m_ds: Optional[xr.Dataset],
        tp_ds: Optional[xr.Dataset],
    ) -> xr.Dataset:
        """Aggregate hourly ERA5 data to daily and reshape to (time, y=1, x=1)."""
        data_vars: dict[str, xr.DataArray] = {}
        lat: float = 0.0
        lon: float = 0.0
        time_values = None

        if t2m_ds is not None:
            hourly = t2m_ds["t2m"]
            daily_mean = hourly.resample(valid_time="1D").mean()
            daily_min = hourly.resample(valid_time="1D").min()
            lat = float(t2m_ds.coords["latitude"])
            lon = float(t2m_ds.coords["longitude"])
            time_values = daily_mean.coords["valid_time"].values
            data_vars["t2m"] = daily_mean
            data_vars["t2m_min"] = daily_min
            # Frost day: 1 if daily minimum temperature < 0°C (273.15 K), else 0
            data_vars["frost_days"] = (daily_min < 273.15).astype("float32")

        if tp_ds is not None:
            daily = tp_ds["tp"].resample(valid_time="1D").sum()
            if time_values is None:
                lat = float(tp_ds.coords["latitude"])
                lon = float(tp_ds.coords["longitude"])
                time_values = daily.coords["valid_time"].values
            data_vars["tp"] = daily

        if not data_vars:
            raise ValueError(
                "No recognised ERA5 variables (t2m, tp) found in the ZIP archive."
            )

        # Reshape each variable to (time, y=1, x=1) for SampleCube compatibility
        result: dict[str, xr.DataArray] = {}
        for vname, da in data_vars.items():
            values = da.values[:, np.newaxis, np.newaxis].astype("float32")
            result[vname] = xr.DataArray(
                values,
                dims=["time", "y", "x"],
                coords={
                    "time": time_values,
                    "y": np.array([lat], dtype="float64"),
                    "x": np.array([lon], dtype="float64"),
                },
            )

        return xr.Dataset(result)
