"""Harvest-date-anchored feature extractor."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class HarvestDateExtractor:
    """Features derived from the actual harvest date per (parcel, year).

    Requires a ``harvest_date`` column to be pre-merged into the time-series
    DataFrame before ``extract_features`` is called (handled by the pipeline).

    Parcel-level features (emitted once, not per column)
    -----------------------------------------------------
    - ``harvest_doy``   : day of year of harvest
    - ``harvest_month`` : calendar month of harvest

    Per-column features
    -------------------
    - ``{col}_at_harvest``              : index value at the closest timestamp
    - ``{col}_pre_harvest_{N}d_mean``   : mean over the N days before harvest
    - ``{col}_peak_to_harvest_delta``   : peak value − value at harvest
    - ``{col}_peak_to_harvest_months``  : harvest month − peak month
    """

    name = "harvest_date"

    def __init__(self, pre_harvest_windows: list[int] | None = None) -> None:
        self.pre_harvest_windows = pre_harvest_windows or [30, 60]

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}

        # ── resolve harvest_date ──────────────────────────────────────────
        if "harvest_date" not in group.columns:
            return result

        hd_series = group["harvest_date"].dropna()
        if len(hd_series) == 0:
            return result

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            harvest_date = pd.to_datetime(hd_series.iloc[0])

        if pd.isna(harvest_date):
            return result

        # ── parcel-level timing features ─────────────────────────────────
        result["harvest_doy"] = float(harvest_date.day_of_year)
        result["harvest_month"] = float(harvest_date.month)

        # ── per-column features ───────────────────────────────────────────
        if "time" not in group.columns:
            return result

        group = group.copy()
        group["time"] = pd.to_datetime(group["time"])

        for col in columns:
            valid = group.dropna(subset=[col])
            if len(valid) == 0:
                result[f"{col}_at_harvest"] = np.nan
                for w in self.pre_harvest_windows:
                    result[f"{col}_pre_harvest_{w}d_mean"] = np.nan
                result[f"{col}_peak_to_harvest_delta"] = np.nan
                result[f"{col}_peak_to_harvest_months"] = np.nan
                continue

            # Value at harvest: closest timestamp
            time_diffs = (valid["time"] - harvest_date).abs()
            closest_idx = time_diffs.idxmin()
            at_harvest = float(valid.loc[closest_idx, col])
            result[f"{col}_at_harvest"] = at_harvest

            # Pre-harvest rolling windows
            for w in self.pre_harvest_windows:
                window_start = harvest_date - pd.Timedelta(days=w)
                window = valid[
                    (valid["time"] >= window_start) & (valid["time"] <= harvest_date)
                ]
                result[f"{col}_pre_harvest_{w}d_mean"] = (
                    float(window[col].mean()) if len(window) > 0 else np.nan
                )

            # Peak → harvest delta and timing
            peak_idx = valid[col].idxmax()
            peak_val = float(valid.loc[peak_idx, col])
            peak_month = float(valid.loc[peak_idx, "time"].month)

            result[f"{col}_peak_to_harvest_delta"] = (
                peak_val - at_harvest if np.isfinite(at_harvest) else np.nan
            )
            result[f"{col}_peak_to_harvest_months"] = harvest_date.month - peak_month

        return result
