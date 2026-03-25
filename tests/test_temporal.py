"""Tests for temporal aggregation (resample + rolling)."""

import datetime

import numpy as np
import pandas as pd
import pytest

from src.dataset.config import (
    DatasetConfig,
    ResampleConfig,
    RollingConfig,
    TemporalConfig,
)
from src.dataset.stats import aggregate_temporal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    dates: list[str],
    parcel_key: str = "P1",
    sensor: str = "SENTINEL-2",
) -> pd.DataFrame:
    """Build a synthetic stats DataFrame with known values."""
    n = len(dates)
    return pd.DataFrame(
        {
            "time": [datetime.date.fromisoformat(d) for d in dates],
            "parcel_key": parcel_key,
            "sensor": sensor,
            "NDVI_mean": np.arange(1, n + 1, dtype=float),
            "EVI_mean": np.arange(10, 10 + n, dtype=float),
        }
    )


DAILY_DATES = [
    "2021-07-01",
    "2021-07-10",
    "2021-07-20",
    "2021-08-05",
    "2021-08-15",
    "2021-09-10",
]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_resample_agg_raises(self):
        with pytest.raises(ValueError, match="Unknown agg"):
            ResampleConfig(freq="MS", agg="bogus")

    def test_invalid_rolling_agg_raises(self):
        with pytest.raises(ValueError, match="Unknown agg"):
            RollingConfig(window=3, agg="bogus")

    def test_valid_resample_agg(self):
        cfg = ResampleConfig(freq="MS", agg="median")
        assert cfg.agg == "median"

    def test_temporal_from_yaml(self):
        raw = {
            "paths": {"aoi_table": "/tmp/test.csv"},
            "stats": ["mean"],
            "sensors": {},
            "temporal": {
                "resample": {"freq": "MS", "agg": "median"},
                "rolling": {"window": 4},
            },
        }
        cfg = DatasetConfig.model_validate(raw)
        assert cfg.temporal is not None
        assert cfg.temporal.resample.freq == "MS"
        assert cfg.temporal.rolling.window == 4

    def test_temporal_absent_is_none(self):
        raw = {
            "paths": {"aoi_table": "/tmp/test.csv"},
            "stats": ["mean"],
            "sensors": {},
        }
        cfg = DatasetConfig.model_validate(raw)
        assert cfg.temporal is None


# ---------------------------------------------------------------------------
# Resample only
# ---------------------------------------------------------------------------


class TestResampleOnly:
    def test_monthly_median(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(resample=ResampleConfig(freq="MS", agg="median"))
        result = aggregate_temporal(df, config)

        assert "n_samples" in result.columns
        # July: 3 obs, Aug: 2 obs, Sep: 1 obs
        assert list(result["n_samples"]) == [3, 2, 1]
        # July median of [1,2,3] = 2.0
        assert result.iloc[0]["NDVI_mean"] == 2.0
        # Aug median of [4,5] = 4.5
        assert result.iloc[1]["NDVI_mean"] == 4.5
        # Sep: single value = 6.0
        assert result.iloc[2]["NDVI_mean"] == 6.0

    def test_monthly_mean(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(resample=ResampleConfig(freq="MS", agg="mean"))
        result = aggregate_temporal(df, config)

        assert result.iloc[0]["NDVI_mean"] == pytest.approx(2.0)  # mean(1,2,3)
        assert result.iloc[1]["NDVI_mean"] == pytest.approx(4.5)  # mean(4,5)

    def test_time_column_is_date(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(resample=ResampleConfig(freq="MS"))
        result = aggregate_temporal(df, config)
        assert all(isinstance(t, datetime.date) for t in result["time"])

    def test_empty_periods_dropped(self):
        # Data in Jan and Mar, gap in Feb
        dates = ["2021-01-15", "2021-03-10"]
        df = _make_df(dates)
        config = TemporalConfig(resample=ResampleConfig(freq="MS"))
        result = aggregate_temporal(df, config)
        # Feb should be dropped (no data)
        assert len(result) == 2

    def test_meta_columns_preserved(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(resample=ResampleConfig(freq="MS"))
        result = aggregate_temporal(df, config)
        assert all(result["parcel_key"] == "P1")
        assert all(result["sensor"] == "SENTINEL-2")


# ---------------------------------------------------------------------------
# Rolling only
# ---------------------------------------------------------------------------


class TestRollingOnly:
    def test_row_count_unchanged(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(rolling=RollingConfig(window=3, min_periods=1))
        result = aggregate_temporal(df, config)
        assert len(result) == len(df)

    def test_no_n_samples_column(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(rolling=RollingConfig(window=3))
        result = aggregate_temporal(df, config)
        assert "n_samples" not in result.columns

    def test_rolling_mean_values(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(
            rolling=RollingConfig(window=3, min_periods=1, agg="mean")
        )
        result = aggregate_temporal(df, config)
        # NDVI_mean values: [1, 2, 3, 4, 5, 6]
        # Rolling mean(3): [1, 1.5, 2, 3, 4, 5]
        expected = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_allclose(result["NDVI_mean"], expected)


# ---------------------------------------------------------------------------
# Resample + rolling combined
# ---------------------------------------------------------------------------


class TestResampleThenRolling:
    def test_combined(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig(
            resample=ResampleConfig(freq="MS", agg="median"),
            rolling=RollingConfig(window=2, min_periods=1, agg="mean"),
        )
        result = aggregate_temporal(df, config)
        # After resample: NDVI_mean = [2.0, 4.5, 6.0], n_samples = [3, 2, 1]
        # After rolling(2) mean: [2.0, 3.25, 5.25]
        assert len(result) == 3
        assert result.iloc[0]["NDVI_mean"] == pytest.approx(2.0)
        assert result.iloc[1]["NDVI_mean"] == pytest.approx(3.25)
        assert result.iloc[2]["NDVI_mean"] == pytest.approx(5.25)
        # n_samples should NOT be rolled
        assert list(result["n_samples"]) == [3, 2, 1]


# ---------------------------------------------------------------------------
# No-op when config is empty
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_none_config(self):
        df = _make_df(DAILY_DATES)
        config = TemporalConfig()
        result = aggregate_temporal(df, config)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Multi-group
# ---------------------------------------------------------------------------


class TestMultiGroup:
    def test_independent_groups(self):
        df1 = _make_df(DAILY_DATES, parcel_key="P1")
        df2 = _make_df(DAILY_DATES[:3], parcel_key="P2")
        df = pd.concat([df1, df2], ignore_index=True)

        config = TemporalConfig(resample=ResampleConfig(freq="MS", agg="mean"))
        result = aggregate_temporal(df, config)

        p1 = result[result["parcel_key"] == "P1"]
        p2 = result[result["parcel_key"] == "P2"]
        assert len(p1) == 3  # Jul, Aug, Sep
        assert len(p2) == 1  # Jul only
