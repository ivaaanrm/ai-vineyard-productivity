"""Cube-level temporal compositing (resample and rolling window).

Operates on xr.Dataset with dimensions (time, y, x) — applied to raw bands
*before* index computation so that ratio indices are computed on composited
reflectance/backscatter values rather than aggregated post-hoc.
"""

from __future__ import annotations

from typing import List, Union

import xarray as xr


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
