"""ERA5 reanalysis index calculators and preprocessing.

Native band: tp (total precipitation in meters)
Derived: TP_MM (total precipitation in millimeters)

Note: ERA5-Land ``tp`` is an accumulated variable.  The agrixel pipeline
aggregates hourly values with ``resample("1D").sum()``, which sums the
*running totals* instead of the hourly increments.  For roughly uniform
within-day rainfall the overestimate factor is (1+2+…+24)/24 = 12.5.
``preprocess_era5`` divides ``tp`` by this factor to approximate the
true daily total.
"""

from __future__ import annotations

from typing import List

import xarray as xr

from ...loader import SampleCube
from ..base import IndexCalculator, PlotStyle

# Correction factor for summing 24 accumulated hourly values instead of
# deaccumulating first: sum(1..24)/24 = 300/24 = 12.5
_ACCUM_CORRECTION = 12.5


def preprocess_era5(cube: SampleCube) -> None:
    """Correct ERA5-Land accumulated-variable bias in ``tp`` (in-place)."""
    if cube.has_band("tp"):
        cube.replace_band("tp", cube.band("tp") / _ACCUM_CORRECTION)


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
