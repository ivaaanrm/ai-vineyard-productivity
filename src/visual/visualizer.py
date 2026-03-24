"""Snapshot visualiser — assembles panels and renders a matplotlib figure.

Each panel knows how to render itself; this module only handles figure layout.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset.sensors import ALL_CALCULATORS, IndexCalculator, NDVI
from src.dataset.loader import SampleCube
from src.visual.panels import ALL_COMPOSITE_PANELS, SARBandPanel, _RawBandPanel, _SAR_BANDS, _percentile_stretch

_CALCULATOR_REGISTRY: Dict[str, IndexCalculator] = {c.name: c for c in ALL_CALCULATORS}


def _select_time(cube: SampleCube, time: int | str | pd.Timestamp) -> int:
    """Return a time-axis index from an int index or a timestamp-like value."""
    if isinstance(time, int):
        return time
    target = pd.Timestamp(time)
    times = pd.DatetimeIndex(cube.times)
    return int(np.argmin(np.abs(times - target)))


def plot_snapshot(
    cube: SampleCube,
    time: int | str | pd.Timestamp = 0,
    extra_bands: List[str] | None = None,
    save_path: Path | str | None = None,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
) -> plt.Figure:
    """Plot a spatial snapshot for a single timestamp.

    Default panels (when available):
      - RGB composite (S2) or SAR-RGB composite (S1)
      - NDVI

    Additional bands or index names can be passed via *extra_bands*.

    Args:
        cube: Loaded SampleCube.
        time: Time index (int) or timestamp string / pd.Timestamp.
        extra_bands: Names of bands or indices to add (e.g. ["EVI", "B08"]).
        save_path: If provided, save the figure to this path.
        figsize_per_panel: (width, height) in inches per subplot.

    Returns:
        The matplotlib Figure.
    """
    t_idx = _select_time(cube, time)
    timestamp = pd.Timestamp(cube.times[t_idx])

    panels = []

    # 1. Composite panels (RGB, SAR-RGB) — each checks its own required bands
    for composite in ALL_COMPOSITE_PANELS:
        if composite.supports(cube):
            panels.append(composite)

    # 2. NDVI — default index panel
    ndvi = NDVI()
    if ndvi.supports(cube):
        panels.append(ndvi)

    # 3. Extra panels resolved by name
    for name in (extra_bands or []):
        if name in _CALCULATOR_REGISTRY and _CALCULATOR_REGISTRY[name].supports(cube):
            panels.append(_CALCULATOR_REGISTRY[name])
        elif cube.has_band(name):
            panels.append(SARBandPanel(name) if name in _SAR_BANDS else _RawBandPanel(name))
        else:
            print(f"Warning: '{name}' not found — skipped.")

    if not panels:
        raise ValueError("No panels to display for this cube.")

    n = len(panels)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for i, panel in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        arr = panel.render(cube, t_idx)
        if panel.is_rgb:
            ax.imshow(arr)
        else:
            s = panel.plot_style
            if s.vmin is None and s.vmax is None:
                arr = _percentile_stretch(arr)
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = s.vmin, s.vmax
            im = ax.imshow(arr, cmap=s.cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(panel.title, fontsize=10)
        ax.axis("off")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"{cube.sensor} · parcel {cube.parcel_key} · {timestamp.date()}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
