"""Feature pipeline — orchestrates extraction from dataset pipeline output."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import FeaturesConfig
from .extractors import (
    BooleanFeaturesExtractor,
    FeatureExtractor,
    HarvestDateExtractor,
    MonthlyPivotExtractor,
    PeakMetricsExtractor,
    PhaseDeltaExtractor,
    PhenologyPhaseExtractor,
    SeasonMetricsExtractor,
    TemporalDeltaExtractor,
)


class FeaturePipeline:
    """Collapse a time-series DataFrame into one row per (parcel_id, year).

    Data flow:
        1. Load dataset CSV (parcel_id, time, {index}_{stat} columns)
        2. Load targets CSV — pre-merge ``harvest_date`` if the extractor
           is enabled and the column is present.
        3. Add year / month columns from ``time``
        4. Group by (parcel_id, year)
        5. Run all enabled extractors per group
        6. Merge remaining target columns on (parcel_id, year)
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

        targets_df = pd.read_csv(self.config.targets_csv)

        # Pre-merge harvest_date into the time-series df so the
        # HarvestDateExtractor can access it per (parcel_id, year) group.
        if (
            self.config.extractors.harvest_date is not None
            and "harvest_date" in targets_df.columns
        ):
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"])
            df["year"] = df["time"].dt.year
            hd = (
                targets_df[self.config.merge_columns + ["harvest_date"]]
                .drop_duplicates()
            )
            df = df.merge(hd, on=self.config.merge_columns, how="left")

        features_df = self.extract_features(df)
        result = self._merge_targets(features_df, targets_df)
        return self._drop_high_null_columns(result)

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

        if cfg.phase_delta is not None:
            ext = PhaseDeltaExtractor(
                phases=cfg.phase_delta.phases,
                agg=cfg.phase_delta.agg,
            )
            extractors.append((ext, self.config.columns))

        if cfg.boolean_features is not None:
            ext = BooleanFeaturesExtractor(
                phases=cfg.boolean_features.phases,
                early_peak_threshold=cfg.boolean_features.early_peak_threshold,
            )
            extractors.append((ext, self.config.columns))

        if cfg.harvest_date is not None:
            cols = cfg.harvest_date.columns or self.config.columns
            ext = HarvestDateExtractor(
                pre_harvest_windows=cfg.harvest_date.pre_harvest_windows,
            )
            extractors.append((ext, cols))

        return extractors

    def _drop_high_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop feature columns whose null percentage exceeds the threshold."""
        threshold = self.config.max_null_pct
        if threshold >= 100.0:
            return df

        protected = set(self.config.merge_columns + self.config.target_columns + ["split"])
        null_pct = df.isnull().mean() * 100
        to_drop = [
            col for col in df.columns
            if col not in protected and null_pct[col] > threshold
        ]

        if to_drop:
            print(f"Dropping {len(to_drop)} columns with >{threshold}% nulls: {to_drop}")

        return df.drop(columns=to_drop)

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
