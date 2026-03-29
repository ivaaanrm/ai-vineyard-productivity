"""Peak value and peak month extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PeakMetricsExtractor:
    """Peak value and peak month for each index.

    Output: ``{col}_peak``, ``{col}_peak_month``
    """

    name = "peak_metrics"

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for col in columns:
            valid = group.dropna(subset=[col])
            if len(valid) > 0:
                idx_max = valid[col].idxmax()
                result[f"{col}_peak"] = float(valid.loc[idx_max, col])
                result[f"{col}_peak_month"] = float(valid.loc[idx_max, "month"])
            else:
                result[f"{col}_peak"] = np.nan
                result[f"{col}_peak_month"] = np.nan
        return result
