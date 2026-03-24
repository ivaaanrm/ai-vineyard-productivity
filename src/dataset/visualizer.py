"""Snapshot visualizer — RGB, SAR composite, NDVI, and arbitrary bands."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .indices import EVICalculator, NDVICalculator, RVICalculator
from .loader import BandCube


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _percentile_stretch(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Stretch array to [0, 1] using percentile clipping."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(valid, low), np.nanpercentile(valid, high)
    if hi == lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _make_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Stack and stretch three 2-D arrays into an (H, W, 3) uint8 image."""
    rgb = np.dstack([
        _percentile_stretch(red),
        _percentile_stretch(green),
        _percentile_stretch(blue),
    ])
    return (np.nan_to_num(rgb, nan=0.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _select_time(cube: BandCube, time: int | str | pd.Timestamp) -> int:
    """Return a time-axis index from an integer index or a timestamp-like value."""
    if isinstance(time, int):
        return time
    target = pd.Timestamp(time)
    times = pd.DatetimeIndex(cube.times)
    return int(np.argmin(np.abs(times - target)))


def _get_slice(cube: BandCube, band: str, t_idx: int) -> np.ndarray:
    return cube.band(band).isel(time=t_idx).values


def _build_rgb_panel(cube: BandCube, t_idx: int) -> tuple[np.ndarray, str] | None:
    """True-colour RGB: B04 (Red), B03 (Green), B02 (Blue). S2 only."""
    if not all(cube.has_band(b) for b in ("B04", "B03", "B02")):
        return None
    img = _make_rgb(
        _get_slice(cube, "B04", t_idx),
        _get_slice(cube, "B03", t_idx),
        _get_slice(cube, "B02", t_idx),
    )
    return img, "RGB (B04/B03/B02)"


def _build_sar_rgb_panel(cube: BandCube, t_idx: int) -> tuple[np.ndarray, str] | None:
    """SAR pseudo-colour: R=VV, G=VH, B=VV/VH. S1 only."""
    if not all(cube.has_band(b) for b in ("VV", "VH")):
        return None
    vv = _get_slice(cube, "VV", t_idx).astype("float32")
    vh = _get_slice(cube, "VH", t_idx).astype("float32")
    ratio = np.where(vv != 0, vv / vh, np.nan)
    img = _make_rgb(vv, vh, ratio)
    return img, "SAR RGB (VV/VH/VV·VH⁻¹)"


def _build_ndvi_panel(cube: BandCube, t_idx: int) -> tuple[np.ndarray, str] | None:
    """NDVI: use native band if present, otherwise compute from B08/B04."""
    if cube.has_band("NDVI"):
        arr = _get_slice(cube, "NDVI", t_idx)
        return arr, "NDVI"
    calc = NDVICalculator()
    if calc.supports(cube):
        arr = calc.compute(cube).isel(time=t_idx).values
        return arr, "NDVI (computed)"
    return None


def _build_band_panel(cube: BandCube, band: str, t_idx: int) -> tuple[np.ndarray, str] | None:
    """Single-band panel, raw values."""
    if not cube.has_band(band):
        return None
    return _get_slice(cube, band, t_idx), band


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_snapshot(
    cube: BandCube,
    time: int | str | pd.Timestamp = 0,
    extra_bands: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
) -> plt.Figure:
    """Plot a spatial snapshot for a single timestamp.

    Always shows: RGB (S2) or SAR-RGB (S1), NDVI (when computable).
    Extra band names can be passed via *extra_bands*.

    Args:
        cube: Loaded BandCube.
        time: Time index (int) or timestamp string / pd.Timestamp.
        extra_bands: Additional band names to display alongside defaults.
        save_path: If provided, save the figure to this path.
        figsize_per_panel: (width, height) in inches for each subplot panel.

    Returns:
        The matplotlib Figure.
    """
    t_idx = _select_time(cube, time)
    timestamp = pd.Timestamp(cube.times[t_idx])

    panels: list[tuple] = []  # (data, title, cmap, vmin, vmax)

    # --- Composite panels (return (array, title) or None) ---
    for builder in (_build_rgb_panel, _build_sar_rgb_panel):
        result = builder(cube, t_idx)
        if result is not None:
            arr, title = result
            panels.append((arr, title, None, None, None))

    ndvi_result = _build_ndvi_panel(cube, t_idx)
    if ndvi_result is not None:
        arr, title = ndvi_result
        panels.append((arr, title, "RdYlGn", -1, 1))

    # --- Extra band panels ---
    for band in (extra_bands or []):
        result = _build_band_panel(cube, band, t_idx)
        if result is not None:
            arr, title = result
            panels.append((arr, title, "viridis", None, None))
        else:
            print(f"Warning: band '{band}' not found in cube — skipped.")

    if not panels:
        raise ValueError("No panels to display for this cube.")

    n = len(panels)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for i, (arr, title, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        if arr.ndim == 3:  # RGB uint8
            ax.imshow(arr)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"{cube.sensor} · parcel {cube.parcel_key} · {timestamp.date()}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
