"""MODIS MOD16A2 evapotranspiration index calculators.

Native bands: ET_500m (actual ET, mm/day), PET_500m (potential ET, mm/day)
Derived: ET_RATIO = ET_500m / PET_500m (crop water stress indicator)
"""

from __future__ import annotations

from typing import List

import xarray as xr

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


class ETRatio(IndexCalculator):
    """Evapotranspiration ratio (crop water stress indicator).

    ET_RATIO = ET_500m / PET_500m
    Range: 0 (severe stress) to 1 (no stress).
    """

    def __init__(self) -> None:
        super().__init__(
            name="ET_RATIO",
            required_bands=["ET_500m", "PET_500m"],
            plot_style=PlotStyle(cmap="RdYlGn", vmin=0.0, vmax=1.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        et = cube.band("ET_500m").astype("float32")
        pet = cube.band("PET_500m").astype("float32")
        return (et / pet).where(pet != 0)


MODIS_CALCULATORS: List[IndexCalculator] = [
    ETRatio(),
]
