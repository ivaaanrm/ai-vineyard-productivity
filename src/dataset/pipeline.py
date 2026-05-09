from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import pandas as pd

from .config import DatasetConfig
from .loader import SampleCube, SampleLoaderProtocol, ZarrBandLoader
from .sensors.era5 import ERA5TimeseriesLoader
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

    def load(self, parcel_key: str, sensor: str, single: bool = False):
        """Load a SampleCube with sensor preprocessing applied (no indices)."""
        cube = self.loader.load(parcel_key, sensor)
        preprocessor = SENSOR_PREPROCESSORS.get(sensor)
        if preprocessor:
            preprocessor(cube)

        if single:
            cube.compute_indices(self._calculators_for(sensor))
            postprocessor = SENSOR_POSTPROCESSORS.get(sensor)
            if postprocessor:
                postprocessor(cube)

        return cube

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

        # Remove duplicate timestamps (e.g. overlapping split zarrs)
        cube = cube.drop_duplicate_times()
        
        # Drop corrupt/missing timesteps (NaN-dominated frames)
        nan_threshold = self.config.corrupt_filter_nan_fraction_for(sensor) if self.config else None
        if nan_threshold is not None:
            cube = cube.drop_corrupt_times(nan_threshold)

        # Drop cloud-dominated timesteps before temporal compositing
        ndvi_threshold = self.config.cloud_filter_ndvi_for(sensor) if self.config else None
        if ndvi_threshold is not None:
            cube = cube.drop_cloudy_times(ndvi_threshold)

        should_mask = (
            geometry is not None
            and self.config is not None
            and self.config.mask_for(sensor)
        )
        if should_mask:
            erosion = self.config.erosion_for(sensor) if self.config else 0.0
            cube = cube.mask(geometry, erosion_pixels=erosion)



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

        # Deduplicate parcel keys before processing to avoid processing same parcel twice
        seen: set = set()
        unique_keys = [k for k in parcel_keys if not (k in seen or seen.add(k))]
        # if len(unique_keys) < len(parcel_keys):
        #     print(f"   Dedup: removed {len(parcel_keys) - len(unique_keys)} duplicate parcel key(s)")
        parcel_keys = unique_keys

        frames: List[pd.DataFrame] = []
        error_frames = []
        for parcel_key in tqdm(parcel_keys, desc="Computing parcels"):
            try:
                geometry = geometries.get(parcel_key) if geometries else None
                for sensor in sensors:
                    try:
                        frames.append(self.process(parcel_key, sensor, geometry=geometry))
                    except FileNotFoundError:
                        pass
            except Exception as e:
                print(e)
                error_frames.append(parcel_key)
        
        print("Parcel with errors: ", error_frames)

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Drop exact duplicate rows (same parcel, time, sensor, and all feature values)
        if not df.empty:
            n_before = len(df)
            df = df.drop_duplicates()
            n_dropped = n_before - len(df)
            # if n_dropped > 0:
            #     print(f"   Dedup: dropped {n_dropped} duplicate row(s) from output DataFrame")

        if not df.empty and self.config and self.config.fuse_sensors:
            df = self._fuse_sensors(df)

        # Drop rows where every sensor feature column is NaN
        if not df.empty:
            key_cols = [c for c in ("parcel_id", "year", "month", "doy", "sensor") if c in df.columns]
            feature_cols = [c for c in df.columns if c not in key_cols]
            if feature_cols:
                df = df.dropna(subset=feature_cols, how="all")

        return df

    @staticmethod
    def _fuse_sensors(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse multiple sensor rows into one row per (parcel_key, time)."""
        group_cols = ["parcel_id", "year", "month", "doy"]
        agg = {}
        for col in df.columns:
            if col in group_cols or col == "sensor":
                continue
            agg[col] = "sum" if col == "n_samples" else "first"
        return df.groupby(group_cols, sort=False).agg(agg).reset_index()

    def _calculators_for(self, sensor: str) -> List[IndexCalculator]:
        if self.config:
            return self.config.calculators_for(sensor) + self.extra_calculators
        return _SENSOR_CALCULATORS.get(sensor, []) + self.extra_calculators

    def _output_bands_for(self, sensor: str) -> List[str] | None:
        if self.config:
            return self.config.output_bands_for(sensor)
        return None

    def _stats_for(self, sensor: str) -> List[str]:
        if self.config:
            return self.config.stats_for(sensor)
        return self.stats


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


class CompositeLoader:
    """Routes ``load()`` calls to sensor-specific loaders.

    Useful for mixing loaders with different storage backends, e.g. a
    ``ZarrBandLoader`` for optical/SAR data and an ``ERA5TimeseriesLoader``
    for climate reanalysis files.

    Parameters
    ----------
    loaders:
        Mapping of sensor name → loader instance.
    default:
        Fallback loader for sensors not present in *loaders*.  If ``None``
        and the sensor is not found, a ``ValueError`` is raised.
    """

    def __init__(
        self,
        loaders: Dict[str, SampleLoaderProtocol],
        default: SampleLoaderProtocol | None = None,
    ) -> None:
        self.loaders = loaders
        self.default = default

    def load(self, parcel_key: str, sensor: str) -> SampleCube:
        loader = self.loaders.get(sensor, self.default)
        if loader is None:
            raise ValueError(
                f"No loader configured for sensor '{sensor}'. "
                f"Available: {list(self.loaders)}"
            )
        return loader.load(parcel_key, sensor)


def make_era5_timeseries_pipeline(
    era5_path: Path | str,
    config: Path | str | DatasetConfig | None = None,
    stats: List[str] | None = None,
) -> DatasetPipeline:
    """Build a pipeline backed by per-parcel ERA5 timeseries ZIP archives.

    Parameters
    ----------
    era5_path:
        Root directory that contains one sub-directory per parcel, e.g.
        ``data/output/files/ERA5_TIMESERIES``.
    config:
        Optional dataset config (path to YAML or ``DatasetConfig`` instance).
        When provided the ERA5 sensor section is used for indices, stats, and
        temporal compositing.
    stats:
        Fallback spatial statistics when *config* is not provided.
    """
    if config is not None and not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_yaml(config)
    return DatasetPipeline(
        loader=ERA5TimeseriesLoader(era5_path),
        stats=stats,
        config=config,
    )


def make_composite_pipeline(
    base_path: Path | str,
    era5_path: Path | str,
    config: Path | str | DatasetConfig | None = None,
    stats: List[str] | None = None,
) -> DatasetPipeline:
    """Build a pipeline that serves ERA5 from timeseries ZIPs and everything
    else from the zarr store.

    Parameters
    ----------
    base_path:
        Agrixel zarr output root (used for SENTINEL-2, SENTINEL-1, etc.).
    era5_path:
        Root directory for per-parcel ERA5 ZIP archives.
    config:
        Optional dataset config (path to YAML or ``DatasetConfig`` instance).
    stats:
        Fallback spatial statistics when *config* is not provided.
    """
    if config is not None and not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_yaml(config)
    loader = CompositeLoader(
        loaders={"ERA5": ERA5TimeseriesLoader(era5_path)},
        default=ZarrBandLoader(base_path),
    )
    return DatasetPipeline(loader=loader, stats=stats, config=config)
