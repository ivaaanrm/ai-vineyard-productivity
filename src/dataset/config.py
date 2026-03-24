"""Dataset pipeline configuration — load from YAML, resolve calculators."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from .indices import ALL_CALCULATORS, _BaseIndex

# Registry: name → calculator instance
_CALCULATOR_REGISTRY: dict[str, _BaseIndex] = {c.name: c for c in ALL_CALCULATORS}


class SensorConfig(BaseModel):
    compute_indices: list[str]
    output_bands: list[str]

    @model_validator(mode="after")
    def _check_known_indices(self) -> SensorConfig:
        unknown = set(self.compute_indices) - _CALCULATOR_REGISTRY.keys()
        if unknown:
            raise ValueError(f"Unknown indices: {unknown}. Available: {list(_CALCULATOR_REGISTRY)}")
        return self


class DatasetConfig(BaseModel):
    stats: list[str]
    sensors: dict[str, SensorConfig]

    @classmethod
    def from_yaml(cls, path: Path | str) -> DatasetConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def calculators_for(self, sensor: str) -> list[_BaseIndex]:
        """Return calculator instances for a sensor based on compute_indices."""
        names = self.sensors.get(sensor, SensorConfig(compute_indices=[], output_bands=[])).compute_indices
        return [_CALCULATOR_REGISTRY[n] for n in names if n in _CALCULATOR_REGISTRY]

    def output_bands_for(self, sensor: str) -> list[str] | None:
        cfg = self.sensors.get(sensor)
        return cfg.output_bands if cfg else None
