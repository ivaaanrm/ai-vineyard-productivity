"""Sentinel-1 SAR index calculators and preprocessing utilities.

Indices: RVI, VH_VV, DpRVI
Utilities: lee_filter, to_db
"""

from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


# ---------------------------------------------------------------------------
# SAR preprocessing utilities
# ---------------------------------------------------------------------------


def lee_filter(da: xr.DataArray, size: int = 7) -> xr.DataArray:
    """Refined Lee speckle filter approximation using local statistics.

    Estimates noise variance as the mean of local variances across valid
    pixels, then adaptively blends the local mean with the original value
    based on the ratio of signal to noise variance.
    """

    def _filter_2d(img: np.ndarray, size: int) -> np.ndarray:
        mask = np.isnan(img)
        filled = np.where(mask, 0.0, img).astype("float64")
        valid = (~mask).astype("float64")

        # NaN-aware local statistics
        frac = uniform_filter(valid, size=size)
        safe_frac = np.where(frac > 0, frac, 1.0)
        local_mean = uniform_filter(filled, size=size) / safe_frac
        local_sq_mean = uniform_filter(filled**2, size=size) / safe_frac
        local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)

        # Estimate noise variance as mean local variance over valid pixels
        noise_var = float(np.mean(local_var[~mask])) if np.any(~mask) else 0.0

        # Lee weight: preserve detail where local_var >> noise_var
        w = np.where(
            local_var > noise_var,
            (local_var - noise_var) / local_var,
            0.0,
        )

        result = local_mean + w * (filled - local_mean)
        result[mask] = np.nan
        return result.astype("float32")

    data = np.array(da.values, dtype="float32")
    if data.ndim == 3:  # (time, y, x)
        for t in range(data.shape[0]):
            data[t] = _filter_2d(data[t], size)
    elif data.ndim == 2:  # (y, x)
        data = _filter_2d(data, size)

    return da.copy(data=data)


def to_db(da: xr.DataArray, eps: float = 1e-10) -> xr.DataArray:
    """Convert linear power backscatter to decibels: 10 * log10(clip(da, min=eps))."""
    return 10 * np.log10(da.clip(min=eps))


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
    Operates on linear-scale bands (before dB conversion).
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
