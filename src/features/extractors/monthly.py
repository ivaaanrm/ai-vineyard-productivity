"""Monthly pivot extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MonthlyPivotExtractor:
    """Pivot each index-stat column into ``{col}_m{month}`` wide columns."""

    name = "monthly"

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for _, row in group.iterrows():
            m = int(row["month"])
            if m > 9 or m < 3: # TODO: Remove fot testing porpuses
                continue
            for col in columns:
                result[f"{col}_m{m}"] = (
                    float(row[col]) if pd.notna(row[col]) else np.nan
                )
        return result
