"""ERA5 reanalysis index calculators.

Native band: tp (total precipitation in meters)
Derived: TP_MM (total precipitation in millimeters)
"""

from __future__ import annotations

from typing import List

import xarray as xr

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle


class TotalPrecipMM(IndexCalculator):
    """Total precipitation in millimeters = tp * 1000."""

    def __init__(self) -> None:
        super().__init__(
            name="TP_MM",
            required_bands=["tp"],
            plot_style=PlotStyle(cmap="Blues", vmin=0.0, vmax=50.0),
        )

    def compute(self, cube: SampleCube) -> xr.DataArray:
        return cube.band("tp").astype("float32") * 1000.0


ERA5_CALCULATORS: List[IndexCalculator] = [
    TotalPrecipMM(),
]
