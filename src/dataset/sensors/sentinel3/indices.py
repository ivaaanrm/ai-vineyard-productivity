"""Sentinel-3 SLSTR index calculators.

Native band: LST (Land Surface Temperature in Kelvin)
Derived: LST_C (Celsius)
"""

from __future__ import annotations

from typing import List

import xarray as xr

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


class LSTCelsius(IndexCalculator):
    """Land Surface Temperature in Celsius = LST - 273.15."""

    def __init__(self) -> None:
        super().__init__(
            name="LST_C",
            required_bands=["LST"],
            plot_style=PlotStyle(cmap="inferno", vmin=-10.0, vmax=50.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        return cube.band("LST").astype("float32") - 273.15


S3_CALCULATORS: List[IndexCalculator] = [
    LSTCelsius(),
]
