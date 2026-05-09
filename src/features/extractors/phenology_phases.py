"""Phenology phase aggregation extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DEFAULT_PHASES


class PhenologyPhaseExtractor:
    name = "phenology_phases"

    def __init__(
        self,
        phases: dict[str, list[int]] | None = None,
        aggs: list[str] | None = None,
        agg_by_column: dict[str, list[str]] | None = None,
    ) -> None:
        self.phases = phases or {}
        self.aggs = aggs or ["mean", "max"]
        self.agg_by_column = agg_by_column or {}

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for phase_name, months in self.phases.items():
            phase_data = group.loc[group["month"].isin(months)]
            for col in columns:
                values = phase_data[col].dropna()
                col_aggs = self.agg_by_column.get(col, self.aggs)
                for agg in col_aggs:
                    key = f"{col}_{phase_name}_{agg}"
                    result[key] = (
                        float(getattr(values, agg)()) if len(values) > 0 else np.nan
                    )
        return result
