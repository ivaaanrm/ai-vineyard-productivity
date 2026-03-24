"""Spectral index calculators following the IndexCalculator protocol.

Sentinel-2 indices: NDVI, EVI, SAVI, NBR2, NDWI
Sentinel-1 indices: RVI, VH_VV
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import xarray as xr

from .loader import BandCube


@runtime_checkable
class IndexCalculator(Protocol):
    """Protocol for spectral index computation."""

    name: str
    required_bands: list[str]

    def supports(self, cube: BandCube) -> bool: ...
    def compute(self, cube: BandCube) -> xr.DataArray: ...


class _BaseIndex:
    """Shared helpers for index calculators."""

    name: str
    required_bands: list[str]

    def supports(self, cube: BandCube) -> bool:
        return all(cube.has_band(b) for b in self.required_bands)


# ---------------------------------------------------------------------------
# Sentinel-2 indices
# ---------------------------------------------------------------------------


class NDVICalculator(_BaseIndex):
    """Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)."""

    name = "NDVI"
    required_bands = ["B08", "B04"]

    def compute(self, cube: BandCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        denom = nir + red
        return ((nir - red) / denom).where(denom != 0)


class EVICalculator(_BaseIndex):
    """Enhanced Vegetation Index = 2.5 * (NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)."""

    name = "EVI"
    required_bands = ["B08", "B04", "B02"]

    def compute(self, cube: BandCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        blue = cube.band("B02").astype("float32")
        denom = nir + 6 * red - 7.5 * blue + 1
        return (2.5 * (nir - red) / denom).where(denom != 0)


class SAVICalculator(_BaseIndex):
    """Soil Adjusted Vegetation Index = (1+L)*(NIR-Red)/(NIR+Red+L), L=0.5."""

    name = "SAVI"
    required_bands = ["B08", "B04"]
    L: float = 0.5

    def compute(self, cube: BandCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        denom = nir + red + self.L
        return ((1 + self.L) * (nir - red) / denom).where(denom != 0)


class NBR2Calculator(_BaseIndex):
    """Normalized Burn Ratio 2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2).
    Useful for crop water stress detection.
    """

    name = "NBR2"
    required_bands = ["B11", "B12"]

    def compute(self, cube: BandCube) -> xr.DataArray:
        swir1 = cube.band("B11").astype("float32")
        swir2 = cube.band("B12").astype("float32")
        denom = swir1 + swir2
        return ((swir1 - swir2) / denom).where(denom != 0)


class NDWICalculator(_BaseIndex):
    """Normalized Difference Water Index = (Green - NIR) / (Green + NIR)."""

    name = "NDWI"
    required_bands = ["B03", "B08"]

    def compute(self, cube: BandCube) -> xr.DataArray:
        green = cube.band("B03").astype("float32")
        nir = cube.band("B08").astype("float32")
        denom = green + nir
        return ((green - nir) / denom).where(denom != 0)


# ---------------------------------------------------------------------------
# Sentinel-1 indices
# ---------------------------------------------------------------------------


class RVICalculator(_BaseIndex):
    """Radar Vegetation Index = 4*VH / (VV + VH)."""

    name = "RVI"
    required_bands = ["VV", "VH"]

    def compute(self, cube: BandCube) -> xr.DataArray:
        vv = cube.band("VV").astype("float32")
        vh = cube.band("VH").astype("float32")
        denom = vv + vh
        return (4 * vh / denom).where(denom != 0)


class VHVVRatioCalculator(_BaseIndex):
    """VH/VV backscatter ratio — sensitive to canopy structure."""

    name = "VH_VV"
    required_bands = ["VV", "VH"]

    def compute(self, cube: BandCube) -> xr.DataArray:
        vv = cube.band("VV").astype("float32")
        vh = cube.band("VH").astype("float32")
        return (vh / vv).where(vv != 0)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

S2_CALCULATORS: list[_BaseIndex] = [
    NDVICalculator(),
    EVICalculator(),
    SAVICalculator(),
    NBR2Calculator(),
    NDWICalculator(),
]

S1_CALCULATORS: list[_BaseIndex] = [
    RVICalculator(),
    VHVVRatioCalculator(),
]

ALL_CALCULATORS: list[_BaseIndex] = S2_CALCULATORS + S1_CALCULATORS
