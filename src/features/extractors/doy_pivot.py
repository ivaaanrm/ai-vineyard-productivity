"""DOY (day-of-year) pivot extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DOYPivotExtractor:
    """Pivot each column into ``{col}_doy{N}`` wide columns, one per timestep.

    Uses the ``doy`` column (day-of-year) as the time key, giving finer
    temporal resolution than the monthly pivot when the dataset is resampled
    sub-monthly (e.g. 1W).  Only timesteps within the growing-season window
    [doy_start, doy_end] are included.

    Default window (60–274) covers March 1 – September 30, matching the
    growing-season slice used by ``MonthlyPivotExtractor`` (months 3–9).

    Output: ``{col}_doy{N}`` for each DOY value N present in the group.
    """

    name = "doy_pivot"

    def __init__(self, doy_start: int = 60, doy_end: int = 274) -> None:
        self.doy_start = doy_start
        self.doy_end = doy_end

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        if "doy" not in group.columns:
            return {}
        result: dict[str, float] = {}
        for _, row in group.iterrows():
            d = int(row["doy"])
            if not (self.doy_start <= d <= self.doy_end):
                continue
            for col in columns:
                result[f"{col}_doy{d}"] = float(row[col]) if pd.notna(row[col]) else np.nan
        return result
