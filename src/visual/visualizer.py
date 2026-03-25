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


def _resolve_panels(cube: SampleCube, extra_bands: List[str] | None = None) -> list:
    """Return the ordered list of panels for a cube + extra_bands spec."""
    panels = []
    for composite in ALL_COMPOSITE_PANELS:
        if composite.supports(cube):
            panels.append(composite)
    ndvi = NDVI()
    if ndvi.supports(cube):
        panels.append(ndvi)
    for name in (extra_bands or []):
        if name in _CALCULATOR_REGISTRY and _CALCULATOR_REGISTRY[name].supports(cube):
            panels.append(_CALCULATOR_REGISTRY[name])
        elif cube.has_band(name):
            panels.append(SARBandPanel(name) if name in _SAR_BANDS else _RawBandPanel(name))
        else:
            print(f"Warning: '{name}' not found — skipped.")
    return panels


def _render_panel(
    fig: plt.Figure, ax: plt.Axes, panel, cube: SampleCube, t_idx: int
) -> None:
    """Render a single panel into *ax*."""
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


def plot_snapshot(
    cube: SampleCube,
    time: int | str | pd.Timestamp = 0,
    extra_bands: List[str] | None = None,
    save_path: Path | str | None = None,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a spatial snapshot for a single timestamp.

    Default panels (when available):
      - RGB composite (S2) or SAR-RGB composite (S1)
      - NDVI

    Additional bands or index names can be passed via *extra_bands*.

    When *ax* is provided the function renders only the primary composite panel
    into that axes and returns its parent figure.  No new figure is created and
    *extra_bands*, *figsize_per_panel*, and *save_path* are ignored.

    Args:
        cube: Loaded SampleCube.
        time: Time index (int) or timestamp string / pd.Timestamp.
        extra_bands: Names of bands or indices to add (e.g. ["EVI", "B08"]).
        save_path: If provided, save the figure to this path.
        figsize_per_panel: (width, height) in inches per subplot.
        ax: Optional axes to render into.  When supplied only the primary
            composite is drawn; the caller owns the figure.

    Returns:
        The matplotlib Figure.
    """
    t_idx = _select_time(cube, time)
    timestamp = pd.Timestamp(cube.times[t_idx])

    # ------------------------------------------------------------------
    # Single-axes mode: render the primary composite into the caller's ax
    # ------------------------------------------------------------------
    if ax is not None:
        for composite in ALL_COMPOSITE_PANELS:
            if composite.supports(cube):
                ax.imshow(composite.render(cube, t_idx))
                ax.set_title(f"{cube.sensor} · {timestamp.date()}", fontsize=10)
                ax.axis("off")
                return ax.get_figure()
        raise ValueError(f"No composite panel available for sensor {cube.sensor!r}.")

    panels = _resolve_panels(cube, extra_bands)
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
        _render_panel(fig, axes[i // ncols][i % ncols], panel, cube, t_idx)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"{cube.sensor} · parcel {cube.parcel_key} · {timestamp.date()}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_mosaic(
    rows: List[tuple],
    save_path: Path | str | None = None,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
) -> plt.Figure:
    """Render multiple snapshots as rows in a single mosaic figure.

    Each entry in *rows* is a ``(cube, time, extra_bands)`` tuple mirroring the
    first three arguments of :func:`plot_snapshot`.  All rows share the same
    column count (determined by the row with the most panels); unused cells are
    hidden.

    Args:
        rows: List of ``(cube, time, extra_bands)`` tuples, one per row.
        save_path: If provided, save the figure to this path.
        figsize_per_panel: ``(width, height)`` in inches per subplot cell.

    Returns:
        The matplotlib Figure.
    """
    resolved = []
    for cube, time, extra_bands in rows:
        t_idx = _select_time(cube, time)
        timestamp = pd.Timestamp(cube.times[t_idx])
        panels = _resolve_panels(cube, extra_bands)
        if not panels:
            raise ValueError(f"No panels for sensor {cube.sensor!r}.")
        resolved.append((cube, t_idx, timestamp, panels))

    nrows = len(resolved)
    ncols = max(len(panels) for _, _, _, panels in resolved)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for row, (cube, t_idx, timestamp, panels) in enumerate(resolved):
        for col, panel in enumerate(panels):
            _render_panel(fig, axes[row][col], panel, cube, t_idx)
            if col == 0:
                axes[row][0].set_title(
                    f"{cube.sensor} · {timestamp.date()}\n{panel.title}",
                    fontsize=9,
                )
        for col in range(len(panels), ncols):
            axes[row][col].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
