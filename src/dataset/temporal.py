"""Cube-level temporal compositing (resample and rolling window).

Operates on xr.Dataset with dimensions (time, y, x) — applied to raw bands
*before* index computation so that ratio indices are computed on composited
reflectance/backscatter values rather than aggregated post-hoc.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def smooth_dataarray(
    da: xr.DataArray,
    resample_period: str,
    window: int,
    agg: str = "median",
) -> xr.DataArray:
    """Resample to regular grid, then apply centered rolling aggregation per year.

    Operates per-year to prevent smoothing across campaign boundaries.

    Args:
        da: DataArray with a ``time`` coordinate.
        resample_period: Pandas offset alias (e.g. "ME" for month end).
        window: Rolling window size in time steps.
        agg: Aggregation method (e.g., "median", "mean").

    Returns:
        A new DataArray with resampled and smoothed values.
    """
    # Resample to regular grid
    # xarray's DataArray.resample returns a DataArrayResample which uses
    # reduction methods like .mean(), .median(), .sum(), etc. rather than
    # .agg(). Map common agg names to the corresponding method.
    resampler = da.resample(time=resample_period)
    if agg == "median":
        resampled = resampler.median()
    elif agg == "mean":
        resampled = resampler.mean()
    elif agg == "sum":
        resampled = resampler.sum()
    else:
        # Fallback to using the generic reduce with getattr if available
        try:
            resampled = getattr(resampler, agg)()
        except AttributeError:
            raise ValueError(f"Unsupported aggregation '{agg}' for resample")

    # Split by year and apply rolling window per year
    years = np.unique(resampled.time.dt.year.values)
    chunks = []
    for yr in years:
        yr_slice = resampled.sel(time=resampled.time.dt.year == yr)
        roller = yr_slice.rolling(
            time=window,
            center=True,
            min_periods=1,
        )
        # Use method names directly (rolling doesn't have .agg())
        if agg == "median":
            smoothed = roller.median()
        elif agg == "mean":
            smoothed = roller.mean()
        elif agg == "sum":
            smoothed = roller.sum()
        else:
            try:
                smoothed = getattr(roller, agg)()
            except AttributeError:
                raise ValueError(f"Unsupported aggregation '{agg}' for rolling")
        chunks.append(smoothed)

    return xr.concat(chunks, dim="time")


def smooth_dataset(
    ds: xr.Dataset,
    resample_period: str,
    window: int,
    agg: str = "median",
) -> xr.Dataset:
    """Apply temporal smoothing to all variables in a Dataset.

    Args:
        ds: Dataset with a ``time`` coordinate.
        resample_period: Pandas offset alias (e.g. "ME" for month end).
        window: Rolling window size in time steps.
        agg: Aggregation method (e.g., "median", "mean").

    Returns:
        A new Dataset with smoothed values.
    """
    smoothed_vars = {}
    for var_name in ds.data_vars:
        smoothed_vars[var_name] = smooth_dataarray(
            ds[var_name], resample_period, window, agg
        )
    return xr.Dataset(smoothed_vars, attrs=ds.attrs)


def resample_cube(ds: xr.Dataset, freq: str, agg: str = "median") -> xr.Dataset:
    """Temporally resample every variable along the *time* dimension.

    Args:
        ds: Dataset with a ``time`` coordinate.
        freq: Pandas offset alias (e.g. ``"ME"``, ``"MS"``, ``"4W"``).
        agg: Aggregation method — ``"median"``, ``"mean"``, ``"sum"``, etc.

    Returns:
        A new Dataset with the resampled time axis.  All-NaN time steps
        (periods with no original data) are dropped.
    """
    resampler = ds.resample(time=freq)
    out = getattr(resampler, agg)()
    return out.dropna(dim="time", how="all")


def rolling_cube(
    ds: xr.Dataset,
    window: int,
    min_periods: int,
    agg: str,
) -> xr.Dataset:
    """Apply a rolling window along the *time* dimension.

    Args:
        ds: Dataset with a ``time`` coordinate.
        window: Window size (number of time steps).
        min_periods: Minimum observations required to produce a value.
        agg: Aggregation method (``"mean"``, ``"median"``, etc.).

    Returns:
        A new Dataset with smoothed values along time.
    """
    if ds.sizes["time"] < window:
        return ds
    roller = ds.rolling(time=window, min_periods=min_periods)
    return getattr(roller, agg)()
