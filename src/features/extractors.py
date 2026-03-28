"""Feature extractors — Protocol + concrete implementations."""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ── Default Mediterranean vineyard phenology phases ──────────────────────────

DEFAULT_PHASES: Dict[str, List[int]] = {
    "dormancy": [12, 1, 2],
    "budbreak": [3, 4],
    "flowering": [5, 6],
    "veraison": [7, 8],
    "harvest": [9, 10],
    "postharvest": [11],
}


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract for all feature extractors.

    Each extractor receives a DataFrame filtered to a single (parcel_id, year)
    with a ``month`` column (int 1–12) and value columns.  It returns a flat
    dict of ``{feature_name: value}``.
    """

    name: str

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]: ...


# ── Implementations ──────────────────────────────────────────────────────────


class MonthlyPivotExtractor:
    """Pivot each index-stat column into ``{col}_m{month}`` wide columns."""

    name = "monthly"

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for _, row in group.iterrows():
            m = int(row["month"])
            for col in columns:
                result[f"{col}_m{m}"] = (
                    float(row[col]) if pd.notna(row[col]) else np.nan
                )
        return result


class PhenologyPhaseExtractor:
    """Aggregate values within predefined vineyard phenology phases."""

    name = "phenology_phases"

    def __init__(
        self,
        phases: dict[str, list[int]] | None = None,
        aggs: list[str] | None = None,
    ) -> None:
        self.phases = phases or DEFAULT_PHASES
        self.aggs = aggs or ["mean", "max"]

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for phase_name, months in self.phases.items():
            phase_data = group.loc[group["month"].isin(months)]
            for col in columns:
                values = phase_data[col].dropna()
                for agg in self.aggs:
                    key = f"{col}_{phase_name}_{agg}"
                    if len(values) > 0:
                        result[key] = float(getattr(values, agg)())
                    else:
                        result[key] = np.nan
        return result


class TemporalDeltaExtractor:
    """Month-over-month differences (greenup / senescence speed)."""

    name = "temporal_deltas"

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        sorted_g = group.sort_values("month")
        for col in columns:
            values = sorted_g[col].values
            months = sorted_g["month"].values
            for i in range(1, len(values)):
                delta = values[i] - values[i - 1]
                result[f"{col}_delta_m{months[i]}"] = (
                    float(delta) if np.isfinite(delta) else np.nan
                )
        return result


class PeakMetricsExtractor:
    """Peak value and peak month for each index."""

    name = "peak_metrics"

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for col in columns:
            valid = group.dropna(subset=[col])
            if len(valid) > 0:
                idx_max = valid[col].idxmax()
                result[f"{col}_peak"] = float(valid.loc[idx_max, col])
                result[f"{col}_peak_month"] = float(valid.loc[idx_max, "month"])
            else:
                result[f"{col}_peak"] = np.nan
                result[f"{col}_peak_month"] = np.nan
        return result


class SeasonMetricsExtractor:
    """Phenological season metrics: SOS, EOS, POS, LOS, AOS, ROG, ROS, integral.

    Uses a threshold-based approach where the season boundary is defined where
    the index crosses ``base + threshold_pct * (max - base)`` on ascent (SOS)
    and descent (EOS).
    """

    name = "season_metrics"

    def __init__(
        self,
        threshold_pct: float = 0.2,
        base_percentile: float = 10.0,
    ) -> None:
        self.threshold_pct = threshold_pct
        self.base_percentile = base_percentile

    def extract(self, group: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        sorted_g = group.sort_values("month")
        for col in columns:
            values = sorted_g[col].values.astype(float)
            months = sorted_g["month"].values.astype(float)
            result.update(self._compute(col, values, months))
        return result

    def _compute(
        self, col: str, values: np.ndarray, months: np.ndarray
    ) -> dict[str, float]:
        suffixes = ["SOS", "EOS", "POS", "LOS", "AOS", "ROG", "ROS", "integral"]
        valid_mask = np.isfinite(values)

        if valid_mask.sum() < 3:
            return {f"{col}_{s}": np.nan for s in suffixes}

        v = values[valid_mask]
        m = months[valid_mask]

        vmin = float(np.percentile(v, self.base_percentile))
        vmax = float(np.nanmax(v))
        threshold = vmin + self.threshold_pct * (vmax - vmin)

        # POS: month of maximum
        pos_idx = int(np.argmax(v))
        result: dict[str, float] = {
            f"{col}_POS": float(m[pos_idx]),
            f"{col}_AOS": vmax - vmin,
        }

        # SOS: first month crossing threshold on ascent (before POS)
        sos = np.nan
        for i in range(pos_idx):
            if v[i] >= threshold:
                sos = float(m[i])
                break
        result[f"{col}_SOS"] = sos

        # EOS: first month dropping below threshold after POS
        eos = np.nan
        for i in range(pos_idx, len(v)):
            if v[i] < threshold:
                eos = float(m[i])
                break
        result[f"{col}_EOS"] = eos

        # LOS: length of season
        result[f"{col}_LOS"] = (
            eos - sos if np.isfinite(sos) and np.isfinite(eos) else np.nan
        )

        # ROG: max positive delta during green-up (months up to POS)
        deltas_up = np.diff(v[: pos_idx + 1])
        result[f"{col}_ROG"] = (
            float(np.max(deltas_up)) if len(deltas_up) > 0 else np.nan
        )

        # ROS: max negative delta during senescence (months from POS)
        deltas_down = np.diff(v[pos_idx:])
        result[f"{col}_ROS"] = (
            float(np.min(deltas_down)) if len(deltas_down) > 0 else np.nan
        )

        # Integral: sum of values between SOS and EOS
        if np.isfinite(sos) and np.isfinite(eos):
            season_mask = (m >= sos) & (m <= eos)
            result[f"{col}_integral"] = float(np.sum(v[season_mask]))
        else:
            result[f"{col}_integral"] = np.nan

        return result


# Añadir nuevos extractores
