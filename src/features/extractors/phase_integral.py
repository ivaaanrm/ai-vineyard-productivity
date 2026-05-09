"""Phase-scoped integral extractor — area under the curve per phenology window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DEFAULT_PHASES


class PhaseIntegralExtractor:
    """Compute the trapezoidal integral of selected indices within phenology phases.

    For each (column, phase) pair the extractor filters the monthly time-series
    to the phase months, sorts by month, and applies ``np.trapz(values, x=months)``.
    This captures cumulative exposure — e.g. total NDVI accumulation during
    flowering is a proxy for canopy photosynthetic activity over that window.

    Output keys: ``{col}_{phase_name}_integral``

    NaN is returned when fewer than 2 valid observations exist in the phase
    (trapz requires at least two points).
    """

    name = "phase_integral"

    def __init__(
        self,
        phases: dict[str, list[int]] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self.phases = phases or {
            "flowering": DEFAULT_PHASES["flowering"],
            "veraison": DEFAULT_PHASES["veraison"],
        }
        self.columns = columns  # None → resolved from pipeline-level columns at extract time

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        cols = self.columns if self.columns is not None else columns
        result: dict[str, float] = {}

        # Use DOY as the time axis when available — month integers collapse all
        # weekly samples within a month to the same x value, making trapz wrong.
        use_doy = "doy" in group.columns
        sort_col = "doy" if use_doy else "month"
        x_col = "doy" if use_doy else "month"
        sorted_g = group.sort_values(sort_col)

        for phase_name, months in self.phases.items():
            phase_data = sorted_g[sorted_g["month"].isin(months)]
            x = phase_data[x_col].values.astype(float)

            for col in cols:
                if col not in phase_data.columns:
                    continue
                key = f"{col}_{phase_name}_integral"
                values = phase_data[col].values.astype(float)
                valid = np.isfinite(values)

                if valid.sum() < 2:
                    result[key] = np.nan
                else:
                    result[key] = float(np.trapezoid(values[valid], x=x[valid]))

        return result
