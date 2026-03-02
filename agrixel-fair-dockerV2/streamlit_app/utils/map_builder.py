"""Folium map construction helpers."""

from __future__ import annotations

import io

import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .data_loader import geotiff_bounds_4326, parcel_geojson_4326


def build_parcel_map(
    parcels_geom: list,
    parcel_ids: list[str],
    *,
    selected_ids: list[str] | None = None,
    center: tuple[float, float] | None = None,
    zoom: int = 13,
) -> folium.Map:
    """Build a Folium map with parcel boundary polygons.

    Parameters
    ----------
    parcels_geom : list
        List of Shapely geometries in EPSG:3857.
    parcel_ids : list[str]
        Matching parcel IDs.
    selected_ids : list[str] | None
        IDs to highlight (blue fill); others are grey outlines.
    center : tuple[float, float] | None
        (lat, lon) center. Computed from geometries if None.
    zoom : int
        Initial zoom level.
    """
    selected_ids = selected_ids or []

    geojsons = [parcel_geojson_4326(g) for g in parcels_geom]

    if center is None:
        lats, lons = [], []
        for gj in geojsons:
            coords = _flatten_coords(gj["coordinates"])
            lons.extend(c[0] for c in coords)
            lats.extend(c[1] for c in coords)
        center = (float(np.mean(lats)), float(np.mean(lons)))

    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    for pid, gj in zip(parcel_ids, geojsons):
        is_selected = pid in selected_ids
        folium.GeoJson(
            {"type": "Feature", "geometry": gj, "properties": {"id": pid}},
            style_function=lambda _feat, sel=is_selected: {
                "color": "#2563eb" if sel else "#6b7280",
                "weight": 2 if sel else 1,
                "fillColor": "#3b82f6" if sel else "#d1d5db",
                "fillOpacity": 0.35 if sel else 0.10,
            },
            tooltip=pid,
        ).add_to(m)

    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


def add_raster_overlay(
    m: folium.Map,
    data: np.ndarray,
    profile: dict,
    *,
    cmap: str = "viridis",
    name: str = "Band",
    opacity: float = 0.7,
    vmin: float | None = None,
    vmax: float | None = None,
) -> folium.Map:
    """Add a raster array as an ImageOverlay to the Folium map."""
    bounds = geotiff_bounds_4326(profile)

    # Normalize to 0-255 RGBA
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return m

    if vmin is None:
        vmin = float(np.percentile(valid, 2))
    if vmax is None:
        vmax = float(np.percentile(valid, 98))

    colormap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = colormap(norm(np.clip(data, vmin, vmax)))
    # Set nodata pixels transparent
    rgba[np.isnan(data), 3] = 0.0

    # Convert to uint8 PNG in memory
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    buf = io.BytesIO()
    plt.imsave(buf, rgba_uint8, format="png")
    buf.seek(0)

    import base64

    img_b64 = base64.b64encode(buf.read()).decode()
    img_url = f"data:image/png;base64,{img_b64}"

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        name=name,
        opacity=opacity,
        interactive=False,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def make_colorbar(
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    label: str = "",
) -> plt.Figure:
    """Create a standalone matplotlib colorbar figure."""
    fig, ax = plt.subplots(figsize=(6, 0.4))
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label(label, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_coords(coords) -> list[tuple[float, float]]:
    """Recursively flatten GeoJSON coordinate arrays to (x, y) tuples."""
    if isinstance(coords[0], (int, float)):
        return [tuple(coords)]
    result = []
    for c in coords:
        result.extend(_flatten_coords(c))
    return result
