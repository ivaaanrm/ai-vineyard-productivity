"""Dataset pipeline configuration — load from YAML, resolve calculators."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, model_validator

from .sensors import ALL_CALCULATORS, IndexCalculator

# Registry: name → calculator instance
_CALCULATOR_REGISTRY: Dict[str, IndexCalculator] = {c.name: c for c in ALL_CALCULATORS}

_ALLOWED_TEMPORAL_AGGS = {"mean", "median", "min", "max", "sum", "std"}


class ResampleConfig(BaseModel):
    freq: str
    agg: str = "median"

    @model_validator(mode="after")
    def _check_agg(self) -> ResampleConfig:
        if self.agg not in _ALLOWED_TEMPORAL_AGGS:
            raise ValueError(
                f"Unknown agg '{self.agg}'. Allowed: {_ALLOWED_TEMPORAL_AGGS}"
            )
        return self


class RollingConfig(BaseModel):
    window: int
    min_periods: int = 1
    agg: str = "mean"

    @model_validator(mode="after")
    def _check_agg(self) -> RollingConfig:
        if self.agg not in _ALLOWED_TEMPORAL_AGGS:
            raise ValueError(
                f"Unknown agg '{self.agg}'. Allowed: {_ALLOWED_TEMPORAL_AGGS}"
            )
        return self


class TemporalConfig(BaseModel):
    resample: ResampleConfig | None = None
    rolling: RollingConfig | None = None


class SensorConfig(BaseModel):
    compute_indices: List[str]
    output_bands: List[str]

    @model_validator(mode="after")
    def _check_known_indices(self) -> SensorConfig:
        unknown = set(self.compute_indices) - _CALCULATOR_REGISTRY.keys()
        if unknown:
            raise ValueError(
                f"Unknown indices: {unknown}. Available: {List(_CALCULATOR_REGISTRY)}"
            )
        return self


class DatasetConfig(BaseModel):
    paths: Dict[str, str]
    stats: List[str]
    sensors: Dict[str, SensorConfig]
    temporal: TemporalConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> DatasetConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def calculators_for(self, sensor: str) -> List[IndexCalculator]:
        """Return calculator instances for a sensor based on compute_indices."""
        names = self.sensors.get(
            sensor, SensorConfig(compute_indices=[], output_bands=[])
        ).compute_indices
        return [_CALCULATOR_REGISTRY[n] for n in names if n in _CALCULATOR_REGISTRY]

    def output_bands_for(self, sensor: str) -> List[str] | None:
        cfg = self.sensors.get(sensor)
        return cfg.output_bands if cfg else None
