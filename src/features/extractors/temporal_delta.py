"""Month-over-month temporal delta extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TemporalDeltaExtractor:
    """Month-over-month differences (greenup / senescence speed).

    Output pattern: ``{col}_delta_m{month}``
    """

    name = "temporal_deltas"

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        sorted_g = group.sort_values("month")
        for col in columns:
            values = sorted_g[col].values
            months = sorted_g["month"].values
            for i in range(1, len(values)):
                delta = values[i] - values[i - 1]
                result[f"{col}_delta_m{months[i]}"] = (
                    float(delta) if np.isfinite(delta) else np.nan
                )
        return result
