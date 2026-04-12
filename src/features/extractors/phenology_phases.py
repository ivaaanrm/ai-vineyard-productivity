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
    ) -> None:
        self.phases = phases or {}
        self.aggs = aggs or ["mean", "max"]

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for phase_name, months in self.phases.items():
            phase_data = group.loc[group["month"].isin(months)]
            for col in columns:
                values = phase_data[col].dropna()
                for agg in self.aggs:
                    key = f"{col}_{phase_name}"
                    result[key] = (
                        float(getattr(values, agg)()) if len(values) > 0 else np.nan
                    )
        return result
