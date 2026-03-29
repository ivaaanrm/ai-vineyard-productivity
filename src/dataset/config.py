"""Dataset pipeline configuration — load from YAML, resolve calculators."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import yaml
from pydantic import BaseModel, model_validator

from .sensors import ALL_CALCULATORS, IndexCalculator

# Registry: name → calculator instance
_CALCULATOR_REGISTRY: Dict[str, IndexCalculator] = {c.name: c for c in ALL_CALCULATORS}

_ALLOWED_TEMPORAL_AGGS = {"mean", "median", "min", "max", "sum", "std"}


class ResampleConfig(BaseModel):
    freq: str | None = None
    agg: Union[str, List[str]] = "median"

    @model_validator(mode="after")
    def _check_agg(self) -> ResampleConfig:
        aggs = [self.agg] if isinstance(self.agg, str) else self.agg
        unknown = set(aggs) - _ALLOWED_TEMPORAL_AGGS
        if unknown:
            raise ValueError(
                f"Unknown agg(s): {unknown}. Allowed: {_ALLOWED_TEMPORAL_AGGS}"
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
    stats: List[str] | None = None          # None → inherit global stats
    temporal: TemporalConfig | None = None  # None → inherit global temporal
    parcel_mask: bool | None = None         # None → inherit global parcel_mask

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
    parcel_mask: bool = False
    fuse_sensors: bool = False

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

    def stats_for(self, sensor: str) -> List[str]:
        """Return spatial stats for a sensor (per-sensor override or global)."""
        cfg = self.sensors.get(sensor)
        if cfg and cfg.stats is not None:
            return cfg.stats
        return self.stats

    def mask_for(self, sensor: str) -> bool:
        """Return whether to apply parcel masking for a sensor.

        Per-sensor ``parcel_mask`` takes priority; falls back to the global flag.
        Coarse-resolution sensors (S3, ERA5, MODIS) should set ``parcel_mask: false``
        in the YAML to avoid the geometry clip failing on 1 km+ pixels.
        """
        cfg = self.sensors.get(sensor)
        if cfg and cfg.parcel_mask is not None:
            return cfg.parcel_mask
        return self.parcel_mask

    def temporal_for(self, sensor: str) -> TemporalConfig | None:
        """Return temporal config for a sensor (per-sensor override merged with global)."""
        cfg = self.sensors.get(sensor)
        sensor_temporal = cfg.temporal if cfg else None

        if sensor_temporal is None:
            return self.temporal
        if self.temporal is None:
            return sensor_temporal

        # Merge: inherit global freq when sensor doesn't specify its own
        merged = sensor_temporal.model_copy(deep=True)
        if (
            merged.resample
            and merged.resample.freq is None
            and self.temporal.resample
        ):
            merged.resample.freq = self.temporal.resample.freq
        return merged
