"""Spatial aggregation of band cubes and computed indices into tabular data."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from .sensors import IndexCalculator
from .loader import SampleCube

# Supported spatial statistics
STAT_FNS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean":   np.nanmean,
    "std":    np.nanstd,
    "median": np.nanmedian,
    "p25":    lambda v: float(np.nanpercentile(v, 25)),
    "p75":    lambda v: float(np.nanpercentile(v, 75)),
    "p90":    lambda v: float(np.nanpercentile(v, 90)),
    "min":    np.nanmin,
    "max":    np.nanmax,
}


class ParcelStatsExtractor:
    """Computes per-timestep spatial statistics for native bands and computed indices.

    For each time step, spatially aggregates every layer (native bands + computed
    indices) into scalar statistics, producing one row per timestamp.

    Args:
        calculators: Index calculators to apply (only those whose required bands
            are present in the cube will run).
        stats: Stat names from STAT_FNS to compute. Defaults to ["mean", "std"].
        skip_existing: If True, skip computing an index whose name already exists
            as a native band (e.g. NDVI pre-computed in the S2 cube).
    """

    def __init__(
        self,
        calculators: List[IndexCalculator] | None = None,
        stats: List[str] | None = None,
        output_bands: List[str] | None = None,
        skip_existing: bool = True,
    ) -> None:
        self.calculators = calculators or []
        self.stats = stats or ["mean", "std"]
        self.output_bands = output_bands  # None → include all
        self.skip_existing = skip_existing

        unknown = set(self.stats) - STAT_FNS.keys()
        if unknown:
            raise ValueError(f"Unknown stats: {unknown}. Available: {List(STAT_FNS)}")

    def get_stats(self, cube: SampleCube) -> pd.DataFrame:
        """Return a DataFrame with one row per timestamp.

        Columns: time, parcel_key, sensor, then {layer}_{stat} for each
        layer/stat combination.
        """
        layers: Dict[str, xr.DataArray] = {}

        for var in cube.variables:
            layers[var] = cube.band(var)

        for calc in self.calculators:
            if self.skip_existing and calc.name in layers:
                continue
            if calc.supports(cube):
                layers[calc.name] = calc.compute(cube)

        if self.output_bands is not None:
            layers = {k: v for k, v in layers.items() if k in self.output_bands}

        stat_fns = {s: STAT_FNS[s] for s in self.stats}
        single_stat = len(stat_fns) == 1
        records: List[Dict[str, Any]] = []

        for t_idx, t in enumerate(cube.times):
            ts = pd.Timestamp(t)
            row: Dict[str, Any] = {
                "parcel_id": cube.parcel_key,
                "year": ts.year,
                "month": ts.month,
                "doy": ts.day_of_year,
                "sensor": cube.sensor,
            }
            for layer_name, da in layers.items():
                values = da.isel(time=t_idx).values.ravel()
                valid = values[np.isfinite(values)]
                for stat_name, fn in stat_fns.items():
                    col = layer_name if single_stat else f"{layer_name}_{stat_name}"
                    row[col] = float(fn(valid)) if len(valid) > 0 else np.nan
            records.append(row)

        return pd.DataFrame(records)

