"""Composite and raw-band visualisation panels.

Composite panels combine multiple bands into a single RGB image.
_RawBandPanel is a fallback for any band that has no dedicated calculator.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.dataset.sensors import PlotStyle, dn_to_sigma0, median_speckle_filter, to_db
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
    """Stack and stretch three 2-D arrays into an (H, W, 4) RGBA uint8 image.

    NaN pixels become fully transparent (alpha = 0).
    """
    valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    rgb = np.dstack([
        _percentile_stretch(red),
        _percentile_stretch(green),
        _percentile_stretch(blue),
    ])
    rgba = np.dstack([np.nan_to_num(rgb, nan=0.0), valid.astype("float32")])
    return (rgba * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sentinel-2 true-colour optimisation (mirrors Sentinel Hub evalscript)
# Contrast enhance / highlight compress + saturation boost + sRGB encoding
# ---------------------------------------------------------------------------

_S2_MAX_R = 3.0   # max reflectance headroom
_S2_MID_R = 0.13  # mid-tone anchor
_S2_SAT = 1.2     # saturation multiplier
_S2_GAMMA = 1.8
_S2_G_OFF = 0.01
_S2_G_OFF_POW = _S2_G_OFF ** _S2_GAMMA
_S2_G_OFF_RANGE = (1.0 + _S2_G_OFF) ** _S2_GAMMA - _S2_G_OFF_POW


def _adj(a: np.ndarray, tx: float, ty: float, max_c: float) -> np.ndarray:
    ar = np.clip(a / max_c, 0.0, 1.0)
    denom = ar * (2.0 * tx / max_c - 1.0) - tx / max_c
    numer = ar * (ar * (tx / max_c + ty - 1.0) - ty)
    return np.where(denom != 0.0, numer / denom, 0.0)


def _adj_gamma(b: np.ndarray) -> np.ndarray:
    return ((b + _S2_G_OFF) ** _S2_GAMMA - _S2_G_OFF_POW) / _S2_G_OFF_RANGE


def _s_adj(a: np.ndarray) -> np.ndarray:
    return _adj_gamma(_adj(a, _S2_MID_R, 1.0, _S2_MAX_R))


def _sat_enh(
    r: np.ndarray, g: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg_s = (r + g + b) / 3.0 * (1.0 - _S2_SAT)
    return (
        np.clip(avg_s + r * _S2_SAT, 0.0, 1.0),
        np.clip(avg_s + g * _S2_SAT, 0.0, 1.0),
        np.clip(avg_s + b * _S2_SAT, 0.0, 1.0),
    )


def _s_rgb(c: np.ndarray) -> np.ndarray:
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(np.maximum(c, 0.0), 0.41666666666) - 0.055,
    )


def _make_s2_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Sentinel Hub-style S2 true-colour: contrast enhance + highlight compress + sRGB.

    Input bands are expected in Sentinel-2 L2A DN scale (0–10000); divided by
    10000 to match the 0–1 reflectance range assumed by the evalscript constants.
    NaN pixels become fully transparent (alpha = 0).
    """
    valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    r = _s_adj(np.nan_to_num(red.astype("float32"), nan=0.0) / 10000.0)
    g = _s_adj(np.nan_to_num(green.astype("float32"), nan=0.0) / 10000.0)
    b = _s_adj(np.nan_to_num(blue.astype("float32"), nan=0.0) / 10000.0)
    r, g, b = _sat_enh(r, g, b)
    rgb = np.dstack([_s_rgb(r), _s_rgb(g), _s_rgb(b)])
    rgba = np.dstack([np.clip(rgb, 0.0, 1.0), valid.astype("float32")])
    return (rgba * 255).astype(np.uint8)


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
        return _make_s2_rgb(
            cube.band("B04").isel(time=t_idx).values,
            cube.band("B03").isel(time=t_idx).values,
            cube.band("B02").isel(time=t_idx).values,
        )


def _sar_to_db(arr: np.ndarray) -> np.ndarray:
    """Ensure a SAR 2-D array is in dB: DN → σ₀ → median filter → dB."""
    if np.nanmedian(arr) > 0:  # raw DN scale → calibrate + filter + convert
        da = xr.DataArray(arr.astype("float32"), dims=["y", "x"])
        return to_db(median_speckle_filter(dn_to_sigma0(da))).values
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
