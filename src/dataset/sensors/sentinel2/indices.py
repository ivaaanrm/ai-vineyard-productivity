"""Sentinel-2 spectral index calculators: NDVI, EVI, EVI2, SAVI, NDWI, GVMI, GNDVI, NDRE, IRECI, S2REP.

Preprocessing
-------------
Sentinel-2 L2A bands from Planetary Computer are stored as uint16 with a
scale factor of 10 000 (i.e. reflectance = DN / 10 000).  Call
``preprocess_s2`` on every cube before computing any index that uses additive
constants (EVI, EVI2, SAVI, GVMI).  NDVI and GNDVI are ratio-based and
scale-invariant, but normalising consistently avoids silent errors.
"""

from __future__ import annotations

from typing import List

import xarray as xr

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


# ---------------------------------------------------------------------------
# S2 preprocessing
# ---------------------------------------------------------------------------

_S2_SCALE = 10_000.0  # L2A DN → reflectance


def preprocess_s2(cube: SampleCube) -> None:
    """Normalise Sentinel-2 DN values (0–10 000) to reflectance (0–1).

    Must be called before any index that contains additive constants
    (EVI, EVI2, SAVI, GVMI).  Safe to call multiple times — skipped when
    values are already ≤ 1.
    """
    for band in list(cube.variables):
        da = cube.band(band)
        # Guard: skip if already normalised (median well below 1)
        if float(da.median()) <= 1.0:
            continue
        cube.replace_band(band, (da.astype("float32") / _S2_SCALE).clip(0.0, 1.0))


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


class EVI2(IndexCalculator):
    """2-band Enhanced Vegetation Index = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1).

    Blue-band-free variant of EVI. More temporally stable than NDVI across seasons
    and better suited for Sentinel-2 due to the known blue-band noise in EVI.
    """

    def __init__(self) -> None:
        super().__init__(
            name="EVI2",
            required_bands=["B08", "B04"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        red = cube.band("B04").astype("float32")
        denom = nir + 2.4 * red + 1
        return (2.5 * (nir - red) / denom).where(denom != 0)


class GVMI(IndexCalculator):
    """Global Vegetation Moisture Index = ((NIR+0.1)-(SWIR1+0.02)) / ((NIR+0.1)+(SWIR1+0.02)).

    Sensitive to canopy water content and soil substrate. Identified as the most
    temporally stable predictor for vineyard substrate differentiation.
    """

    def __init__(self) -> None:
        super().__init__(
            name="GVMI",
            required_bands=["B08", "B11"],
            plot_style=PlotStyle(cmap="Blues", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        swir1 = cube.band("B11").astype("float32")
        num = (nir + 0.1) - (swir1 + 0.02)
        denom = (nir + 0.1) + (swir1 + 0.02)
        return (num / denom).where(denom != 0)


class GNDVI(IndexCalculator):
    """Green Normalized Difference Vegetation Index = (NIR - Green) / (NIR + Green).

    More sensitive to chlorophyll concentration than NDVI, particularly
    useful at grape ripening stages when canopy is fully developed.
    """

    def __init__(self) -> None:
        super().__init__(
            name="GNDVI",
            required_bands=["B08", "B03"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        nir = cube.band("B08").astype("float32")
        green = cube.band("B03").astype("float32")
        denom = nir + green
        return ((nir - green) / denom).where(denom != 0)


class NDRE(IndexCalculator):
    """Normalized Difference Red Edge = (B8A - B05) / (B8A + B05).

    Most sensitive to LAI in vineyards. Uses vegetation red edge (B8A)
    and red edge 1 (B05) for superior sensitivity to leaf area and
    canopy development compared to NDVI.
    """

    def __init__(self) -> None:
        super().__init__(
            name="NDRE",
            required_bands=["B8A", "B05"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        vre = cube.band("B8A").astype("float32")  # Vegetation Red Edge (865 nm)
        re1 = cube.band("B05").astype("float32")  # Red Edge 1 (705 nm)
        denom = vre + re1
        return ((vre - re1) / denom).where(denom != 0)


class IRECI(IndexCalculator):
    """Inverted Red-Edge Chlorophyll Index = (B07 - B04) / (B05 / B06).

    Better predictor of yield than NDVI according to literature. Integrates
    multiple red-edge bands for enhanced sensitivity to chlorophyll and
    canopy stress.
    """

    def __init__(self) -> None:
        super().__init__(
            name="IRECI",
            required_bands=["B07", "B04", "B05", "B06"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=0.0, vmax=3.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        re3 = cube.band("B07").astype("float32")  # Red Edge 3 (783 nm)
        red = cube.band("B04").astype("float32")  # Red (665 nm)
        re1 = cube.band("B05").astype("float32")  # Red Edge 1 (705 nm)
        re2 = cube.band("B06").astype("float32")  # Red Edge 2 (740 nm)
        denom = re1 / re2
        return ((re3 - red) / denom).where(denom != 0)


class S2REP(IndexCalculator):
    """Sentinel-2 Red Edge Position = 705 + 35 * ((B04+B08)/2 - B05) / (B06 - B05).

    Detects water stress and nitrogen status. Red edge position shifts toward
    longer wavelengths under water stress and nitrogen deficiency, making this
    a physically-based indicator of crop condition.
    """

    def __init__(self) -> None:
        super().__init__(
            name="S2REP",
            required_bands=["B04", "B05", "B06", "B08"],
            plot_style=PlotStyle(cmap="viridis", vmin=700.0, vmax=750.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        red = cube.band("B04").astype("float32")  # Red (665 nm)
        re1 = cube.band("B05").astype("float32")  # Red Edge 1 (705 nm)
        re2 = cube.band("B06").astype("float32")  # Red Edge 2 (740 nm)
        nir = cube.band("B08").astype("float32")  # NIR (842 nm)
        denom = re2 - re1
        return (705 + 35 * (((red + nir) / 2) - re1) / denom).where(denom != 0)


S2_CALCULATORS: List[IndexCalculator] = [
    NDVI(),
    EVI(),
    EVI2(),
    SAVI(),
    NDWI(),
    GVMI(),
    GNDVI(),
    NDRE(),
    IRECI(),
    S2REP(),
]
