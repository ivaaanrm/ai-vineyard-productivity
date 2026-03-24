"""Composite and raw-band visualisation panels.

Composite panels combine multiple bands into a single RGB image.
_RawBandPanel is a fallback for any band that has no dedicated calculator.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.dataset.sensors import PlotStyle, lee_filter, to_db
from src.dataset.loader import SampleCube

_SAR_BANDS = {"VV", "VH"}


# ---------------------------------------------------------------------------
# Shared image helpers
# ---------------------------------------------------------------------------


def _percentile_stretch(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
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
# Base
# ---------------------------------------------------------------------------


class _BaseComposite:
    """Base for multi-band composite panels."""

    is_rgb: bool = True

    def __init__(self, name: str, title: str, required_bands: list[str]) -> None:
        self.name = name
        self.title = title
        self.required_bands = required_bands

    def supports(self, cube: SampleCube) -> bool:
        return all(cube.has_band(b) for b in self.required_bands)

    def render(self, cube: SampleCube, t_idx: int) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Composite panels
# ---------------------------------------------------------------------------


class RGBComposite(_BaseComposite):
    """True-colour composite: R=B04, G=B03, B=B02 (Sentinel-2)."""

    def __init__(self) -> None:
        super().__init__(
            name="RGB",
            title="RGB (B04/B03/B02)",
            required_bands=["B04", "B03", "B02"],
        )

    def render(self, cube: SampleCube, t_idx: int) -> np.ndarray:
        return _make_rgb(
            cube.band("B04").isel(time=t_idx).values,
            cube.band("B03").isel(time=t_idx).values,
            cube.band("B02").isel(time=t_idx).values,
        )


def _sar_to_db(arr: np.ndarray) -> np.ndarray:
    """Ensure a SAR 2-D array is in dB, applying speckle filter if needed."""
    if np.nanmedian(arr) > 0:  # linear scale → filter + convert
        da = xr.DataArray(arr.astype("float32"), dims=["y", "x"])
        return to_db(lee_filter(da)).values
    return arr  # already in dB


class SARRGBComposite(_BaseComposite):
    """SAR false-colour composite: R=VV, G=VH, B=VV−VH (dB)."""

    def __init__(self) -> None:
        super().__init__(
            name="SAR_RGB",
            title="SAR RGB (VV / VH / VV−VH) dB",
            required_bands=["VV", "VH"],
        )

    def render(self, cube: SampleCube, t_idx: int) -> np.ndarray:
        vv = _sar_to_db(cube.band("VV").isel(time=t_idx).values.astype("float32"))
        vh = _sar_to_db(cube.band("VH").isel(time=t_idx).values.astype("float32"))
        diff = vv - vh
        return _make_rgb(vv, vh, diff)


# ---------------------------------------------------------------------------
# Raw band fallback
# ---------------------------------------------------------------------------


class _RawBandPanel:
    """Fallback panel for any band without a dedicated calculator."""

    is_rgb = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.title = name
        self.plot_style = PlotStyle()

    def supports(self, cube: SampleCube) -> bool:
        return cube.has_band(self.name)

    def render(self, cube: SampleCube, t_idx: int) -> np.ndarray:
        return cube.band(self.name).isel(time=t_idx).values


# ---------------------------------------------------------------------------
# SAR band panel
# ---------------------------------------------------------------------------


class SARBandPanel:
    """Single SAR band rendered in dB with speckle filtering."""

    is_rgb = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.title = f"{name} (dB)"
        self.plot_style = PlotStyle(cmap="gray")

    def supports(self, cube: SampleCube) -> bool:
        return cube.has_band(self.name)

    def render(self, cube: SampleCube, t_idx: int) -> np.ndarray:
        arr = cube.band(self.name).isel(time=t_idx).values.astype("float32")
        return _sar_to_db(arr)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

S2_COMPOSITE_PANELS: list[_BaseComposite] = [RGBComposite()]
S1_COMPOSITE_PANELS: list[_BaseComposite] = [SARRGBComposite()]
ALL_COMPOSITE_PANELS: list[_BaseComposite] = S2_COMPOSITE_PANELS + S1_COMPOSITE_PANELS
