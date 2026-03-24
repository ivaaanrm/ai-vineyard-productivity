"""Spectral index calculators.

Sentinel-2 indices: NDVI, EVI, SAVI, NBR2, NDWI
Sentinel-1 indices: RVI, VH_VV
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import xarray as xr

from .loader import SampleCube


@dataclass(frozen=True)
class PlotStyle:
    """Visual configuration for a single-band panel."""

    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None


class IndexCalculator(ABC):
    """Abstract base for all spectral index calculators.

    Subclasses must implement `compute()`. Shared logic (supports, render)
    is provided here. Visual configuration is declared via `plot_style`.
    """

    is_rgb: bool = False

    def __init__(
        self,
        name: str,
        required_bands: List[str],
        plot_style: PlotStyle = PlotStyle(),
    ) -> None:
        self.name = name
        self.title = name
        self.required_bands = required_bands
        self.plot_style = plot_style

    def supports(self, cube: SampleCube) -> bool:
        return all(cube.has_band(b) for b in self.required_bands)

    @abstractmethod
    def compute(self, cube: SampleCube) -> xr.DataArray: ...

    def render(self, cube: SampleCube, t_idx: int) -> np.ndarray:
        """Return a (y, x) array at the given time index.

        Uses the pre-computed native band if already present in the cube,
        otherwise computes on the fly.
        """
        if cube.has_band(self.name):
            return cube.band(self.name).isel(time=t_idx).values
        return self.compute(cube).isel(time=t_idx).values


# ---------------------------------------------------------------------------
# Sentinel-2 indices
# ---------------------------------------------------------------------------


class NDVI(IndexCalculator):
    """Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)."""

    def __init__(self) -> None:
        super().__init__(
            name="NDVI",
            required_bands=["B08", "B04"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        denom = nir + red
        return ((nir - red) / denom).where(denom != 0)


class EVI(IndexCalculator):
    """Enhanced Vegetation Index = 2.5 * (NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)."""

    def __init__(self) -> None:
        super().__init__(
            name="EVI",
            required_bands=["B08", "B04", "B02"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        blue = cube.band("B02").astype("float32")
        denom = nir + 6 * red - 7.5 * blue + 1
        return (2.5 * (nir - red) / denom).where(denom != 0)


class SAVI(IndexCalculator):
    """Soil Adjusted Vegetation Index = (1+L)*(NIR-Red)/(NIR+Red+L), L=0.5."""

    def __init__(self, L: float = 0.5) -> None:
        super().__init__(
            name="SAVI",
            required_bands=["B08", "B04"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
        )
        self.L = L

    def compute(self, cube: SampleCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        denom = nir + red + self.L
        return ((1 + self.L) * (nir - red) / denom).where(denom != 0)


class NBR2(IndexCalculator):
    """Normalized Burn Ratio 2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)."""

    def __init__(self) -> None:
        super().__init__(
            name="NBR2",
            required_bands=["B11", "B12"],
            plot_style=PlotStyle(cmap="RdBu", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        swir1 = cube.band("B11").astype("float32")
        swir2 = cube.band("B12").astype("float32")
        denom = swir1 + swir2
        return ((swir1 - swir2) / denom).where(denom != 0)


class NDWI(IndexCalculator):
    """Normalized Difference Water Index = (Green - NIR) / (Green + NIR)."""

    def __init__(self) -> None:
        super().__init__(
            name="NDWI",
            required_bands=["B03", "B08"],
            plot_style=PlotStyle(cmap="Blues", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        green = cube.band("B03").astype("float32")
        nir = cube.band("B08").astype("float32")
        denom = green + nir
        return ((green - nir) / denom).where(denom != 0)


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


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

S2_CALCULATORS: List[IndexCalculator] = [
    NDVI(),
    EVI(),
    SAVI(),
    NBR2(),
    NDWI(),
]

S1_CALCULATORS: List[IndexCalculator] = [
    RVI(),
    VHVVRatio(),
]

ALL_CALCULATORS: List[IndexCalculator] = S2_CALCULATORS + S1_CALCULATORS
