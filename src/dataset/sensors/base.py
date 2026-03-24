"""Base classes for spectral index calculators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import xarray as xr

from ..loader import SampleCube


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
