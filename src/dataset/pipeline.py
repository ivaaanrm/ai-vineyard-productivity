from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import DatasetConfig
from .loader import SampleLoaderProtocol, ZarrBandLoader
from .sensors import (
    S1_CALCULATORS,
    S2_CALCULATORS,
    S3_CALCULATORS,
    SENSOR_PREPROCESSORS,
    IndexCalculator,
)
from .stats import ParcelStatsExtractor

# Default calculators per sensor (used when no config is provided)
_SENSOR_CALCULATORS: Dict[str, List[IndexCalculator]] = {
    "SENTINEL-2": S2_CALCULATORS,
    "SENTINEL-1": S1_CALCULATORS,
    "SENTINEL-3": S3_CALCULATORS,
}


class DatasetPipeline:
    """Orchestrates loading, index computation, and spatial reduction.

    Args:
        loader: Any object satisfying SampleLoaderProtocol.
        stats: Spatial statistics to compute (see reducer.STAT_FNS).
        extra_calculators: Additional index calculators beyond the sensor defaults.
    """

    def __init__(
        self,
        loader: SampleLoaderProtocol,
        stats: List[str] | None = None,
        extra_calculators: List[IndexCalculator] | None = None,
        config: DatasetConfig | None = None,
    ) -> None:
        self.loader = loader
        self.config = config
        self.stats = config.stats if config else (stats or ["mean", "std"])
        self.extra_calculators = extra_calculators or []

    def _calculators_for(self, sensor: str) -> List[IndexCalculator]:
        if self.config:
            return self.config.calculators_for(sensor) + self.extra_calculators
        return _SENSOR_CALCULATORS.get(sensor, []) + self.extra_calculators

    def _output_bands_for(self, sensor: str) -> List[str] | None:
        if self.config:
            return self.config.output_bands_for(sensor)
        return None

    def load(self, parcel_key: str, sensor: str):
        """Load a SampleCube with all applicable indices already computed."""
        cube = self.loader.load(parcel_key, sensor)
        preprocessor = SENSOR_PREPROCESSORS.get(sensor)
        if preprocessor:
            preprocessor(cube)
        cube.compute_indices(self._calculators_for(sensor))
        return cube

    def process(self, parcel_key: str, sensor: str) -> pd.DataFrame:
        """Return tabular stats for one parcel/sensor combination."""
        cube = self.load(parcel_key, sensor)
        calculators = self._calculators_for(sensor)
        stats_extractor = ParcelStatsExtractor(
            calculators=calculators,
            stats=self.stats,
            output_bands=self._output_bands_for(sensor),
        )
        return stats_extractor.get_stats(cube)

    def execute(
        self,
        parcel_keys: List[str],
        sensors: List[str],
    ) -> pd.DataFrame:
        """Process multiple parcel/sensor pairs and concatenate results.

        Silently skips combinations where no zarr store exists.
        """
        frames: List[pd.DataFrame] = []
        for parcel_key in parcel_keys:
            for sensor in sensors:
                try:
                    frames.append(self.process(parcel_key, sensor))
                except FileNotFoundError:
                    pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_pipeline(
    base_path: Path | str,
    stats: List[str] | None = None,
    extra_calculators: List[IndexCalculator] | None = None,
) -> DatasetPipeline:
    """Convenience factory: build a pipeline from an agrixel output base path."""
    return DatasetPipeline(
        loader=ZarrBandLoader(base_path),
        stats=stats,
        extra_calculators=extra_calculators,
    )


def make_pipeline_from_config(
    base_path: Path | str,
    config: Path | str | DatasetConfig,
) -> DatasetPipeline:
    """Build a pipeline fully driven by a YAML config file."""
    if not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_yaml(config)
    return DatasetPipeline(loader=ZarrBandLoader(base_path), config=config)
