from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import pandas as pd

from .config import DatasetConfig
from .loader import SampleLoaderProtocol, ZarrBandLoader
from .sensors import (
    S1_CALCULATORS,
    S2_CALCULATORS,
    S3_CALCULATORS,
    ERA5_CALCULATORS,
    MODIS_CALCULATORS,
    SENSOR_PREPROCESSORS,
    SENSOR_POSTPROCESSORS,
    IndexCalculator,
)
from .stats import ParcelStatsExtractor

# Default calculators per sensor (used when no config is provided)
_SENSOR_CALCULATORS: Dict[str, List[IndexCalculator]] = {
    "SENTINEL-2": S2_CALCULATORS,
    "SENTINEL-1": S1_CALCULATORS,
    "SENTINEL-3": S3_CALCULATORS,
    "ERA5": ERA5_CALCULATORS,
    "MODIS": MODIS_CALCULATORS,
}


class DatasetPipeline:
    """Orchestrates loading, temporal compositing, index computation, and
    spatial reduction.

    Pipeline order:
        Raw → Preprocess → Mask → Resample → Rolling → Indices → (Post) → Stats

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
        """Load a SampleCube with sensor preprocessing applied (no indices)."""
        cube = self.loader.load(parcel_key, sensor)
        preprocessor = SENSOR_PREPROCESSORS.get(sensor)
        if preprocessor:
            preprocessor(cube)
        return cube

    def load_with_indices(self, parcel_key: str, sensor: str):
        """Load a SampleCube with preprocessing and indices computed.

        Convenience method for exploratory / notebook use.
        """
        cube = self.load(parcel_key, sensor)
        cube.compute_indices(self._calculators_for(sensor))
        postprocessor = SENSOR_POSTPROCESSORS.get(sensor)
        if postprocessor:
            postprocessor(cube)
        return cube

    def _stats_for(self, sensor: str) -> List[str]:
        if self.config:
            return self.config.stats_for(sensor)
        return self.stats

    def process(
        self,
        parcel_key: str,
        sensor: str,
        geometry: str | None = None,
    ) -> pd.DataFrame:
        """Return tabular stats for one parcel/sensor combination.

        Pipeline: load → preprocess → mask → temporal composite → indices
                  → (postprocess) → spatial stats
        """
        cube = self.load(parcel_key, sensor)

        # Mask before temporal compositing (exclude non-parcel pixels)
        if geometry and self.config and self.config.parcel_mask:
            cube = cube.mask(geometry)

        # Temporal compositing on raw bands (cube level)
        temporal_cfg = self.config.temporal_for(sensor) if self.config else None
        if temporal_cfg:
            cube = cube.composite_temporal(temporal_cfg)

        # Compute indices on composited bands
        cube.compute_indices(self._calculators_for(sensor))

        # Sensor-specific postprocessing (e.g. S1 dB conversion)
        postprocessor = SENSOR_POSTPROCESSORS.get(sensor)
        if postprocessor:
            postprocessor(cube)

        # Spatial statistics
        sensor_stats = self._stats_for(sensor)
        stats_extractor = ParcelStatsExtractor(
            stats=sensor_stats,
            output_bands=self._output_bands_for(sensor),
        )
        return stats_extractor.get_stats(cube)

    def execute(
        self,
        parcel_keys: List[str],
        sensors: List[str],
        geometries: Dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Process multiple parcel/sensor pairs and concatenate results.

        Args:
            geometries: Optional mapping of parcel_key → WKT geometry string.
                Used for spatial masking when ``parcel_mask`` is enabled in config.

        Silently skips combinations where no zarr store exists.
        """
        frames: List[pd.DataFrame] = []
        for parcel_key in tqdm(parcel_keys, desc="Computing parcels"):
            geometry = geometries.get(parcel_key) if geometries else None
            for sensor in sensors:
                try:
                    frames.append(self.process(parcel_key, sensor, geometry=geometry))
                except FileNotFoundError:
                    pass
                
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty and self.config and self.config.fuse_sensors:
            df = self._fuse_sensors(df)
        return df

    @staticmethod
    def _fuse_sensors(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse multiple sensor rows into one row per (parcel_key, time)."""
        group_cols = ["parcel_id", "time"]
        agg = {}
        for col in df.columns:
            if col in group_cols or col == "sensor":
                continue
            agg[col] = "sum" if col == "n_samples" else "first"
        return df.groupby(group_cols, sort=False).agg(agg).reset_index()


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
