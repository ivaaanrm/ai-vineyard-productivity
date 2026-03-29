"""Consecutive phenology phase-to-phase delta extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DEFAULT_PHASES


class PhaseDeltaExtractor:
    """Delta between consecutive phenology phase aggregates.

    Instead of noisy month-over-month deltas, this computes the difference
    between adjacent phase means — e.g. budbreak→flowering captures green-up
    magnitude, veraison→harvest captures senescence rate.

    Output pattern: ``{col}_delta_{phase_a}_{phase_b}``
    e.g. ``NDVI_mean_delta_budbreak_flowering``
    """

    name = "phase_delta"

    def __init__(
        self,
        phases: dict[str, list[int]] | None = None,
        agg: str = "mean",
    ) -> None:
        self.phases = phases or DEFAULT_PHASES
        self.agg = agg

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        phase_names = list(self.phases.keys())

        # Pre-compute phase aggregate for every (col, phase)
        phase_vals: dict[tuple[str, str], float] = {}
        for phase_name, months in self.phases.items():
            phase_data = group.loc[group["month"].isin(months)]
            for col in columns:
                vals = phase_data[col].dropna()
                phase_vals[(col, phase_name)] = (
                    float(getattr(vals, self.agg)()) if len(vals) > 0 else np.nan
                )

        # Consecutive phase deltas: curr - prev
        for i in range(1, len(phase_names)):
            prev_phase = phase_names[i - 1]
            curr_phase = phase_names[i]
            for col in columns:
                prev_val = phase_vals.get((col, prev_phase), np.nan)
                curr_val = phase_vals.get((col, curr_phase), np.nan)
                key = f"{col}_delta_{prev_phase}_{curr_phase}"
                result[key] = (
                    curr_val - prev_val
                    if np.isfinite(prev_val) and np.isfinite(curr_val)
                    else np.nan
                )

        return result
