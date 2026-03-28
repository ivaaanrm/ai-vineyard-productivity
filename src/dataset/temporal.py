"""Cube-level temporal compositing (resample and rolling window).

Operates on xr.Dataset with dimensions (time, y, x) — applied to raw bands
*before* index computation so that ratio indices are computed on composited
reflectance/backscatter values rather than aggregated post-hoc.
"""

from __future__ import annotations

from typing import List, Union

import xarray as xr


def resample_cube(
    ds: xr.Dataset,
    freq: str,
    agg: Union[str, List[str]],
) -> xr.Dataset:
    """Temporally resample every variable along the *time* dimension.

    Args:
        ds: Dataset with a ``time`` coordinate.
        freq: Pandas offset alias (e.g. ``"ME"``, ``"MS"``, ``"4W"``).
        agg: Aggregation method(s) — ``"median"``, ``"mean"``, ``"sum"``, etc.
            When a list is given each variable is expanded into
            ``{var}_{agg}`` for every aggregation.

    Returns:
        A new Dataset with the resampled time axis.  All-NaN time steps
        (periods with no original data) are dropped.
    """
    resampler = ds.resample(time=freq)

    if isinstance(agg, list):
        parts: list[xr.Dataset] = []
        for a in agg:
            part = getattr(resampler, a)()
            part = part.rename({v: f"{v}_{a}" for v in part.data_vars})
            parts.append(part)
        out = xr.merge(parts)
    else:
        out = getattr(resampler, agg)()

    # Drop time steps where ALL variables are NaN everywhere
    valid_mask = xr.concat(
        [out[v].notnull().any(dim=[d for d in out[v].dims if d != "time"])
         for v in out.data_vars],
        dim="__var__",
    ).any("__var__")
    out = out.sel(time=valid_mask)
    return out


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
    roller = ds.rolling(time=window, min_periods=min_periods)
    return getattr(roller, agg)()
