"""Sentinel-2 spectral index calculators: NDVI, EVI, SAVI, NBR2, NDWI."""

from __future__ import annotations

from typing import List

import xarray as xr

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


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


S2_CALCULATORS: List[IndexCalculator] = [
    NDVI(),
    EVI(),
    SAVI(),
    NBR2(),
    NDWI(),
]
