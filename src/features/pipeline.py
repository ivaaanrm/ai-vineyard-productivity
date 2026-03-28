"""Feature pipeline — orchestrates extraction from dataset pipeline output."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import FeaturesConfig
from .extractors import (
    FeatureExtractor,
    MonthlyPivotExtractor,
    PeakMetricsExtractor,
    PhenologyPhaseExtractor,
    SeasonMetricsExtractor,
    TemporalDeltaExtractor,
)


class FeaturePipeline:
    """Collapse a time-series DataFrame into one row per (parcel_id, year).

    Data flow:
        1. Load dataset CSV (parcel_id, time, {index}_{stat} columns)
        2. Add year / month columns from ``time``
        3. Group by (parcel_id, year)
        4. Run all enabled extractors per group
        5. Merge with targets CSV on (parcel_id, year)
    """

    def __init__(self, config: FeaturesConfig) -> None:
        self.config = config
        self._extractors: List[tuple[FeatureExtractor, List[str]]] = (
            self._build_extractors()
        )

    # ── public API ───────────────────────────────────────────────────────

    def execute(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Full pipeline: extract features then merge targets.

        Args:
            df: Pre-loaded time-series DataFrame.  When *None*, reads from
                ``config.dataset_csv``.
        """
        if df is None:
            df = pd.read_csv(self.config.dataset_csv, parse_dates=["time"])

        features_df = self.extract_features(df)
        targets_df = pd.read_csv(self.config.targets_csv)
        return self._merge_targets(features_df, targets_df)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features only (no target merge).  Useful for inference."""
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        records: list[Dict[str, object]] = []
        for (parcel_id, year), group in df.groupby(["parcel_id", "year"]):
            row: Dict[str, object] = {"parcel_id": parcel_id, "year": year}
            sorted_group = group.sort_values("month")

            for extractor, columns in self._extractors:
                valid_cols = [c for c in columns if c in sorted_group.columns]
                if valid_cols:
                    row.update(extractor.extract(sorted_group, valid_cols))

            records.append(row)

        return pd.DataFrame(records)

    # ── internals ────────────────────────────────────────────────────────

    def _build_extractors(self) -> List[tuple[FeatureExtractor, List[str]]]:
        cfg = self.config.extractors
        extractors: List[tuple[FeatureExtractor, List[str]]] = []

        if cfg.monthly:
            extractors.append((MonthlyPivotExtractor(), self.config.columns))

        if cfg.phenology_phases is not None:
            ext = PhenologyPhaseExtractor(
                phases=cfg.phenology_phases.phases,
                aggs=cfg.phenology_phases.aggs,
            )
            extractors.append((ext, self.config.columns))

        if cfg.temporal_deltas:
            extractors.append((TemporalDeltaExtractor(), self.config.columns))

        if cfg.peak_metrics:
            extractors.append((PeakMetricsExtractor(), self.config.columns))

        if cfg.season_metrics is not None:
            cols = cfg.season_metrics.columns or self.config.columns
            ext = SeasonMetricsExtractor(
                threshold_pct=cfg.season_metrics.threshold_pct,
                base_percentile=cfg.season_metrics.base_percentile,
            )
            extractors.append((ext, cols))
        
        # Añadir nuevos extractores

        return extractors

    def _merge_targets(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> pd.DataFrame:
        merge_cols = self.config.merge_columns
        keep_cols = list(merge_cols)
        for c in self.config.target_columns:
            if c in targets_df.columns:
                keep_cols.append(c)
        if "split" in targets_df.columns:
            keep_cols.append("split")

        return features_df.merge(targets_df[keep_cols], on=merge_cols, how="inner")
