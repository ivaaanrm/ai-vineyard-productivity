"""Feature engineering configuration — load from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel


class PhenologyPhasesConfig(BaseModel):
    """Config for phenology phase aggregation."""

    phases: Dict[str, List[int]] | None = None  # None → use defaults
    aggs: List[str] = ["mean", "max"]


class SeasonMetricsConfig(BaseModel):
    """Config for threshold-based season metrics (SOS/EOS/POS/…)."""

    threshold_pct: float = 0.2
    base_percentile: float = 10.0
    columns: List[str] | None = None  # None → use parent-level columns


class PhaseDeltaConfig(BaseModel):
    """Config for consecutive phase-to-phase delta extractor."""

    phases: Dict[str, List[int]] | None = None  # None → use defaults
    agg: str = "mean"


class BooleanFeaturesConfig(BaseModel):
    """Config for threshold-free binary flag extractor."""

    phases: Dict[str, List[int]] | None = None  # None → use defaults
    early_peak_threshold: int = 8  # months ≤ this → early_peak=1


class HarvestDateConfig(BaseModel):
    """Config for harvest-date-anchored feature extractor."""

    pre_harvest_windows: List[int] = [30, 60]  # days before harvest_date
    columns: List[str] | None = None  # None → use parent-level columns


class ExtractorsConfig(BaseModel):
    """Which extractors to run and their parameters."""

    monthly: bool = True
    phenology_phases: PhenologyPhasesConfig | None = PhenologyPhasesConfig()
    temporal_deltas: bool = True
    peak_metrics: bool = True
    season_metrics: SeasonMetricsConfig | None = SeasonMetricsConfig()
    phase_delta: PhaseDeltaConfig | None = None
    boolean_features: BooleanFeaturesConfig | None = None
    harvest_date: HarvestDateConfig | None = None


class FeaturesConfig(BaseModel):
    """Top-level feature engineering config."""

    dataset_csv: str
    targets_csv: str
    columns: List[str]
    target_columns: List[str] = ["yield_kg_ha", "alcohol_degree"]
    merge_columns: List[str] = ["parcel_id", "year"]
    extractors: ExtractorsConfig = ExtractorsConfig()
    max_null_pct: float = 10.0  # drop feature columns with more than this % nulls

    @classmethod
    def from_yaml(cls, path: Path | str) -> FeaturesConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
