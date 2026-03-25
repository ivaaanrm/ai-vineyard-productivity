"""Helpers for visualizing processed parcel datasets inside Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from shapely import wkt

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "PRUEBA00"
PROCESSED_DATASET_CSV = EXPERIMENT_DIR / "data" / "parcel_stats.csv"
PARCEL_METADATA_CSV = EXPERIMENT_DIR / "data" / "parcel_metadata.csv"
AGRIXEL_OUTPUT_FILES = REPO_ROOT / "agrixel-fair-dockerV2" / "data" / "output" / "files"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.pipeline import make_pipeline
from src.visual.visualizer import plot_snapshot

SENSOR_LABELS = {
    "SENTINEL-2": "Sentinel-2",
    "SENTINEL-1": "Sentinel-1",
    "SENTINEL-3": "Sentinel-3",
}

_DEFAULT_EXTRA_BANDS = {
    "SENTINEL-2": ["EVI", "B08", "B11"],
    "SENTINEL-1": ["VV", "VH", "RVI", "VH_VV"],
    "SENTINEL-3": ["LST", "LST_C"],
}


def load_processed_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed statistics and parcel metadata."""
    if not PROCESSED_DATASET_CSV.is_file():
        raise FileNotFoundError(f"No se encontró {PROCESSED_DATASET_CSV}")
    if not PARCEL_METADATA_CSV.is_file():
        raise FileNotFoundError(f"No se encontró {PARCEL_METADATA_CSV}")

    stats_df = pd.read_csv(PROCESSED_DATASET_CSV)
    meta_df = pd.read_csv(PARCEL_METADATA_CSV)

    stats_df["time"] = pd.to_datetime(stats_df["time"], errors="coerce")
    stats_df = stats_df.dropna(subset=["parcel_id", "time"]).copy()
    stats_df["year"] = stats_df["time"].dt.year
    stats_df["doy"] = stats_df["time"].dt.dayofyear

    for col in stats_df.columns:
        if col not in {"parcel_id", "time"}:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    meta_df["year"] = pd.to_numeric(meta_df["year"], errors="coerce").astype("Int64")
    meta_df["harvest_date"] = pd.to_datetime(meta_df["harvest_date"], errors="coerce")
    for col in [
        "declared_area_ha",
        "area_ha",
        "production_kg",
        "yield_kg_ha",
        "alcohol_degree",
    ]:
        if col in meta_df.columns:
            meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce")

    geom_source = (
        meta_df["parcel_geometry"]
        if "parcel_geometry" in meta_df.columns
        else meta_df["geometry"]
    )
    meta_df["parcel_geom"] = geom_source.fillna(meta_df["geometry"]).apply(wkt.loads)
    meta_df["bbox_geom"] = meta_df["geometry"].apply(wkt.loads)

    stats_df = stats_df.sort_values(["parcel_id", "time"]).reset_index(drop=True)
    meta_df = meta_df.sort_values(["parcel_id", "year"]).reset_index(drop=True)
    return stats_df, meta_df


def list_metric_columns(stats_df: pd.DataFrame) -> list[str]:
    """Return numeric columns that make sense as plot metrics."""
    excluded = {"year", "doy"}
    metrics = [
        col
        for col in stats_df.columns
        if col not in {"parcel_id", "time"}
        and col not in excluded
        and pd.api.types.is_numeric_dtype(stats_df[col])
    ]
    preferred = [col for col in metrics if col.endswith("_mean")]
    secondary = [col for col in metrics if col not in preferred]
    return preferred + secondary


def format_metric_label(metric: str) -> str:
    """Format a dataset metric for UI labels."""
    return metric.replace("_", " ")


def plot_metric_yoy(parcel_stats: pd.DataFrame, parcel_id: str, metric: str) -> plt.Figure:
    """Plot day-of-year evolution of a metric, grouped by year."""
    df_plot = parcel_stats.dropna(subset=[metric]).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    for year, df_year in df_plot.groupby("year"):
        ax.plot(
            df_year["doy"],
            df_year[metric],
            marker="o",
            linewidth=2,
            label=str(year),
        )

    ax.set_title(f"{metric} evolution - Parcel {parcel_id}")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel(metric)
    ax.legend(title="Year")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_distribution(parcel_stats: pd.DataFrame, metric: str) -> plt.Figure:
    """Render a by-year distribution view for the selected metric."""
    grouped = []
    labels = []
    for year, df_year in parcel_stats.groupby("year"):
        values = df_year[metric].dropna().values
        if len(values) == 0:
            continue
        grouped.append(values)
        labels.append(str(year))

    fig, ax = plt.subplots(figsize=(8, 4))
    if grouped:
        ax.boxplot(grouped, labels=labels, patch_artist=True)
        ax.set_xlabel("Year")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} distribution by year")
        ax.grid(True, axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_yearly_summary(
    parcel_stats: pd.DataFrame,
    parcel_meta: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Build a compact yearly summary table for the selected parcel."""
    summary = (
        parcel_stats.groupby("year", dropna=True)
        .agg(
            metric_mean=(metric, "mean"),
            metric_min=(metric, "min"),
            metric_max=(metric, "max"),
            observations=(metric, "count"),
            total_samples=("n_samples", "sum"),
        )
        .reset_index()
    )

    if not parcel_meta.empty and "yield_kg_ha" in parcel_meta.columns:
        target = parcel_meta[["year", "yield_kg_ha", "harvest_date"]].copy()
        summary = summary.merge(target, on="year", how="left")

    return summary


def available_cube_sensors(parcel_id: str) -> list[str]:
    """Return sensors that have a cube.zarr store for the parcel."""
    sensors = []
    for sensor in SENSOR_LABELS:
        zarr_path = AGRIXEL_OUTPUT_FILES / sensor / parcel_id / "cube.zarr"
        if zarr_path.is_dir():
            sensors.append(sensor)
    return sensors


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Create a reusable datacube pipeline."""
    return make_pipeline(AGRIXEL_OUTPUT_FILES)


@st.cache_resource(show_spinner=False)
def load_cube(parcel_id: str, sensor: str):
    """Load a datacube for a parcel and sensor."""
    return get_pipeline().load(parcel_id, sensor)


def default_extra_bands(cube) -> list[str]:
    """Choose sensible extra bands for the snapshot renderer."""
    preferred = _DEFAULT_EXTRA_BANDS.get(cube.sensor, [])
    return [band for band in preferred if cube.has_band(band)]


def pick_gallery_times(times: pd.DatetimeIndex, max_items: int = 3) -> list[pd.Timestamp]:
    """Pick evenly spaced timestamps for a compact gallery."""
    if times.empty:
        return []

    n = min(max_items, len(times))
    idxs = np.linspace(0, len(times) - 1, num=n, dtype=int)
    selected = [pd.Timestamp(times[i]) for i in idxs]
    unique = []
    for ts in selected:
        if ts not in unique:
            unique.append(ts)
    return unique


def build_snapshot_gallery(cube, times: list[pd.Timestamp]) -> plt.Figure:
    """Render a row of primary-composite snapshots for multiple dates."""
    if not times:
        raise ValueError("No hay fechas para construir la galería.")

    fig, axes = plt.subplots(1, len(times), figsize=(4.2 * len(times), 4.2), squeeze=False)
    for ax, ts in zip(axes[0], times):
        plot_snapshot(cube, time=ts, ax=ax)
    fig.tight_layout()
    return fig
