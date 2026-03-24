"""Sentinel-1 SAR index calculators and preprocessing utilities.

Indices: RVI, VH_VV, DpRVI
Utilities: dn_to_sigma0, median_speckle_filter, to_db
"""

from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr
from scipy.ndimage import median_filter

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


# ---------------------------------------------------------------------------
# SAR preprocessing utilities
# ---------------------------------------------------------------------------


def dn_to_sigma0(da: xr.DataArray) -> xr.DataArray:
    """Convert digital numbers to sigma0 linear: sigma0 = DN^2 / 1e9."""
    return (da.astype("float64") ** 2) / 1e9


def median_speckle_filter(da: xr.DataArray, size: int = 3) -> xr.DataArray:
    """Apply a median filter for speckle reduction."""
    data = da.values.copy()
    if data.ndim == 3:  # (time, y, x)
        for t in range(data.shape[0]):
            data[t] = median_filter(data[t], size=size)
    elif data.ndim == 2:  # (y, x)
        data = median_filter(data, size=size)
    return da.copy(data=data)


def to_db(da: xr.DataArray, eps: float = 1e-10) -> xr.DataArray:
    """Convert linear power backscatter to decibels: 10 * log10(clip(da, min=eps))."""
    return 10 * np.log10(da.clip(min=eps))


def preprocess_sar(cube: SampleCube) -> None:
    """SAR preprocessing: DN → σ₀ → speckle filter → indices (linear) → dB."""
    if not (cube.has_band("VV") and cube.has_band("VH")):
        return
    # 1. Calibrate: DN → sigma0 linear (σ₀ = DN² / 1e9)
    for band in ("VV", "VH"):
        cube.replace_band(band, dn_to_sigma0(cube.band(band)))
    # 2. Speckle filter on linear-scale sigma0
    for band in ("VV", "VH"):
        cube.replace_band(band, median_speckle_filter(cube.band(band)))
    # 3. All SAR indices require linear-scale bands — compute before dB conversion
    cube.compute_indices(S1_CALCULATORS)
    # 4. Convert VV / VH to decibels
    for band in ("VV", "VH"):
        cube.replace_band(band, to_db(cube.band(band)))


# ---------------------------------------------------------------------------
# Sentinel-1 indices
# ---------------------------------------------------------------------------


class RVI(IndexCalculator):
    """Radar Vegetation Index = 4*VH / (VV + VH)."""

    def __init__(self) -> None:
        super().__init__(
            name="RVI",
            required_bands=["VV", "VH"],
            plot_style=PlotStyle(cmap="Greens", vmin=0.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        vv = cube.band("VV").astype("float32")
        vh = cube.band("VH").astype("float32")
        denom = vv + vh
        return (4 * vh / denom).where(denom != 0)


class VHVVRatio(IndexCalculator):
    """VH/VV backscatter ratio — sensitive to canopy structure."""

    def __init__(self) -> None:
        super().__init__(
            name="VH_VV",
            required_bands=["VV", "VH"],
            plot_style=PlotStyle(cmap="viridis"),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        vv = cube.band("VV").astype("float32")
        vh = cube.band("VH").astype("float32")
        return (vh / vv).where(vv != 0)


class DpRVI(IndexCalculator):
    """Dual-pol Radar Vegetation Index.

    DpRVI = (1 - dop) * 4*VH / (VV + VH), where dop = VV / (VV + VH).
    Operates on linear-scale sigma0 bands (before dB conversion).
    """

    def __init__(self) -> None:
        super().__init__(
            name="DpRVI",
            required_bands=["VV", "VH"],
            plot_style=PlotStyle(cmap="Greens", vmin=0.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        vv = cube.band("VV").astype("float32")
        vh = cube.band("VH").astype("float32")
        total = vv + vh
        safe = total.where(total != 0)
        dop = vv / safe
        m = 1 - dop
        return (m * (4 * vh / safe)).where(total != 0)


S1_CALCULATORS: List[IndexCalculator] = [
    RVI(),
    VHVVRatio(),
    DpRVI(),
]
