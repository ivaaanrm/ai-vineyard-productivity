"""Threshold-free binary feature extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DEFAULT_PHASES


class BooleanFeaturesExtractor:
    """Threshold-free binary flags derived from phase comparisons and peak timing.

    All comparisons are *relative* (phase A vs phase B) so no hard numeric
    thresholds are needed.  Each flag is 1/0 (or NaN when data is missing).

    Per-column flags
    ----------------
    - ``{col}_early_peak``        : peak month ≤ ``early_peak_threshold``
    - ``{col}_peak_in_veraison``  : peak month falls inside veraison window
    - ``{col}_stress_flowering``  : flowering mean < budbreak mean
    - ``{col}_harvest_drop``      : harvest mean < veraison mean
    - ``{col}_veraison_dominant`` : veraison has the highest mean of all phases
    """

    name = "boolean_features"

    def __init__(
        self,
        phases: dict[str, list[int]] | None = None,
        early_peak_threshold: int = 8,
    ) -> None:
        self.phases = phases or DEFAULT_PHASES
        self.early_peak_threshold = early_peak_threshold

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}

        def _phase_mean(col: str, phase_name: str) -> float:
            months = self.phases.get(phase_name, [])
            vals = group.loc[group["month"].isin(months), col].dropna()
            return float(vals.mean()) if len(vals) > 0 else np.nan

        for col in columns:
            valid = group.dropna(subset=[col])
            if len(valid) == 0:
                for flag in (
                    "early_peak", "peak_in_veraison", "stress_flowering",
                    "harvest_drop", "veraison_dominant",
                ):
                    result[f"{col}_{flag}"] = np.nan
                continue

            peak_month = int(valid.loc[valid[col].idxmax(), "month"])

            result[f"{col}_early_peak"] = int(
                peak_month <= self.early_peak_threshold
            )
            result[f"{col}_peak_in_veraison"] = int(
                peak_month in self.phases.get("veraison", [7, 8])
            )

            flowering_mean = _phase_mean(col, "flowering")
            budbreak_mean = _phase_mean(col, "budbreak")
            harvest_mean = _phase_mean(col, "harvest")
            veraison_mean = _phase_mean(col, "veraison")

            result[f"{col}_stress_flowering"] = (
                int(flowering_mean < budbreak_mean)
                if np.isfinite(flowering_mean) and np.isfinite(budbreak_mean)
                else np.nan
            )
            result[f"{col}_harvest_drop"] = (
                int(harvest_mean < veraison_mean)
                if np.isfinite(harvest_mean) and np.isfinite(veraison_mean)
                else np.nan
            )

            # Veraison dominant: highest mean across all phases
            phase_means = {p: _phase_mean(col, p) for p in self.phases}
            valid_means = {p: v for p, v in phase_means.items() if np.isfinite(v)}
            if valid_means:
                dominant = max(valid_means, key=lambda p: valid_means[p])
                result[f"{col}_veraison_dominant"] = int(dominant == "veraison")
            else:
                result[f"{col}_veraison_dominant"] = np.nan

        return result
