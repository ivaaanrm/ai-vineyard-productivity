"""Page 2: Data visualization & exploration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.data_loader import (
    find_zarr_store,
    list_bands,
    list_parcels_with_data,
    list_scenes,
    load_aoi_table,
    load_quality_json,
    load_raster,
    load_rgb_composite,
    load_zarr_cube,
    scene_path,
)
from utils.map_builder import add_raster_overlay, add_rgb_overlay, build_parcel_map, make_colorbar

st.set_page_config(page_title="Explorer", page_icon="🗺️", layout="wide")
st.title("Explorador de datos")

# ------------------------------------------------------------------
# Load AOI table (for geometry overlays)
# ------------------------------------------------------------------
try:
    aoi_df = load_aoi_table()
    aoi_lookup = dict(zip(aoi_df["parcel_id"], aoi_df["geom"]))
except FileNotFoundError:
    aoi_df = None
    aoi_lookup = {}

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Controles")

    # 1. Sensor filter
    sensor_key = st.radio("Sensor", ["S2", "S1", "S3", "MODIS", "ERA5"], horizontal=True)

    # 2. Parcel selector
    parcels = list_parcels_with_data(sensor_key)
    if not parcels:
        st.warning("No hay datos descargados para este sensor.")
        st.stop()

    parcel_id = st.selectbox("Parcela", parcels)

    # 3. Scene selector
    scenes = list_scenes(sensor_key, parcel_id)
    if not scenes:
        st.warning("No hay escenas para esta parcela.")
        st.stop()

    scene_name = st.selectbox("Escena", scenes)
    s_path = scene_path(sensor_key, parcel_id, scene_name)

    # 4. Band selector
    available_bands = list_bands(s_path)
    if not available_bands:
        st.warning("No se encontraron bandas en esta escena.")
        st.stop()

    band = st.selectbox("Banda / variable", available_bands)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_map, tab_ts, tab_analysis = st.tabs(
    ["Mapa", "Serie temporal", "Análisis de bandas"]
)

# ==================================================================
# Tab 1: Map View
# ==================================================================
with tab_map:
    st.subheader(f"{band} — {scene_name[:40]}…")

    is_rgb = band == "RGB"

    try:
        if is_rgb:
            rgb_data, profile = load_rgb_composite(s_path)
        else:
            data, profile = load_raster(s_path, band)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # Build map with parcel outline + raster overlay
    if parcel_id != "(single)" and parcel_id in aoi_lookup:
        geom = aoi_lookup[parcel_id]
        m = build_parcel_map([geom], [parcel_id], selected_ids=[parcel_id])
    else:
        import folium

        from utils.data_loader import geotiff_bounds_4326

        bounds = geotiff_bounds_4326(profile)
        center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
        ]
        m = folium.Map(location=center, zoom_start=15)

    if is_rgb:
        m = add_rgb_overlay(m, rgb_data, profile, name="RGB")
    else:
        # Choose colormap based on band/sensor
        if band == "NDVI":
            cmap = "RdYlGn"
        elif band in ("lst-in",) or sensor_key == "ERA5":
            cmap = "inferno"
        elif sensor_key == "MODIS":
            cmap = "YlGnBu"
        else:
            cmap = "viridis"

        m = add_raster_overlay(m, data, profile, cmap=cmap, name=band)

    st_folium(m, height=500, width="stretch", returned_objects=[])

    # Colorbar (not applicable for RGB)
    if not is_rgb:
        valid = data[~np.isnan(data)]
        if valid.size > 0:
            fig = make_colorbar(
                cmap=cmap,
                vmin=float(np.percentile(valid, 2)),
                vmax=float(np.percentile(valid, 98)),
                label=band,
            )
            st.pyplot(fig, width="content")

# ==================================================================
# Tab 2: Time Series
# ==================================================================
with tab_ts:
    zarr_path = find_zarr_store(sensor_key, parcel_id)

    if zarr_path is None:
        st.info("No hay cubo Zarr disponible para esta parcela.")
    else:
        try:
            ds = load_zarr_cube(zarr_path)
        except Exception as exc:
            st.error(f"Error al abrir el cubo Zarr: {exc}")
            ds = None

        if ds is not None:
            st.subheader("Cubo espacio-temporal")

            # Detect time and variable dims
            if "time" in ds.dims:
                times = pd.to_datetime(ds["time"].values)
                st.write(f"**{len(times)} fechas** disponibles.")

                if len(times) > 1:
                    t_idx = st.slider(
                        "Fecha",
                        min_value=0,
                        max_value=len(times) - 1,
                        value=0,
                        format="t=%d",
                    )
                    st.caption(f"Fecha seleccionada: {times[t_idx].date()}")
                else:
                    t_idx = 0

                # Try to find a plottable data variable
                data_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
                if data_vars:
                    var_name = data_vars[0]
                    ts_data = ds[var_name]

                    # Line chart: mean over spatial dims per time step
                    spatial_dims = [d for d in ts_data.dims if d != "time"]
                    mean_ts = ts_data.mean(dim=spatial_dims).values

                    chart_df = pd.DataFrame(
                        {"fecha": times, var_name: mean_ts}
                    ).set_index("fecha")
                    st.line_chart(chart_df, y=var_name)

                    # Stats table
                    stats_rows = []
                    for i, t in enumerate(times):
                        arr = ts_data.isel(time=i).values.astype(float)
                        valid_arr = arr[~np.isnan(arr)]
                        if valid_arr.size > 0:
                            stats_rows.append(
                                {
                                    "fecha": t.date(),
                                    "min": float(np.min(valid_arr)),
                                    "max": float(np.max(valid_arr)),
                                    "mean": float(np.mean(valid_arr)),
                                    "std": float(np.std(valid_arr)),
                                }
                            )
                    if stats_rows:
                        st.dataframe(pd.DataFrame(stats_rows), width="stretch")
                else:
                    st.info("No se encontraron variables con dimensión temporal.")
            else:
                st.info("El cubo Zarr no tiene dimensión temporal.")

# ==================================================================
# Tab 3: Band Analysis
# ==================================================================
with tab_analysis:
    import matplotlib.pyplot as plt

    st.subheader(f"Análisis — {band}")

    if is_rgb:
        try:
            rgb_data_analysis, _ = load_rgb_composite(s_path)
        except FileNotFoundError:
            st.error("No se encontraron las bandas B02/B03/B04 para el composito RGB.")
            st.stop()

        fig, ax = plt.subplots(figsize=(6, 6))
        # Replace NaN with 0 for display
        display_rgb = np.nan_to_num(rgb_data_analysis, nan=0.0)
        ax.imshow(np.clip(display_rgb, 0, 1), aspect="equal")
        ax.set_title("Composito RGB (B04/B03/B02)")
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)

        # Per-channel histograms
        fig_h, axes = plt.subplots(1, 3, figsize=(10, 3))
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        labels = ["B04 (Red)", "B03 (Green)", "B02 (Blue)"]
        for i, (ax_h, color, label) in enumerate(zip(axes, colors, labels)):
            channel = rgb_data_analysis[:, :, i]
            valid_ch = channel[~np.isnan(channel)]
            if valid_ch.size > 0:
                ax_h.hist(valid_ch.ravel(), bins=80, color=color, edgecolor="none", alpha=0.8)
            ax_h.set_title(label, fontsize=9)
            ax_h.set_xlabel("Valor")
        fig_h.tight_layout()
        st.pyplot(fig_h)
    else:
        try:
            data, profile = load_raster(s_path, band)
        except FileNotFoundError:
            st.error(f"No se encontraron datos para {band}.")
            st.stop()

        valid = data[~np.isnan(data)]

        # --- Histogram ---
        if valid.size > 0:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(valid.ravel(), bins=100, color="#3b82f6", edgecolor="none", alpha=0.8)
            ax.set_xlabel("Valor de píxel")
            ax.set_ylabel("Frecuencia")
            ax.set_title(f"Histograma — {band}")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Todos los píxeles son NoData.")

        # --- Side-by-side band comparison ---
        other_bands = [b for b in available_bands if b != band and b != "RGB"]
        if other_bands:
            compare_band = st.selectbox("Comparar con", other_bands, key="compare_band")
            try:
                data2, profile2 = load_raster(s_path, compare_band)
                valid2 = data2[~np.isnan(data2)]

                col1, col2 = st.columns(2)
                with col1:
                    st.caption(band)
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    cmap1 = "RdYlGn" if band == "NDVI" else "viridis"
                    ax1.imshow(data, cmap=cmap1, aspect="equal")
                    ax1.set_title(band)
                    ax1.axis("off")
                    fig1.tight_layout()
                    st.pyplot(fig1)

                with col2:
                    st.caption(compare_band)
                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    cmap2 = "RdYlGn" if compare_band == "NDVI" else "viridis"
                    ax2.imshow(data2, cmap=cmap2, aspect="equal")
                    ax2.set_title(compare_band)
                    ax2.axis("off")
                    fig2.tight_layout()
                    st.pyplot(fig2)
            except FileNotFoundError:
                st.warning(f"No se encontraron datos para {compare_band}.")

    # --- Quality info ---
    st.subheader("Calidad de la escena")
    quality = load_quality_json(s_path)
    if quality:
        stac = quality.get("stac_subset", {})
        if sensor_key == "S2":
            metrics = {
                "Cobertura nubosa (%)": stac.get("eo:cloud_cover"),
                "Vegetación (%)": stac.get("s2:vegetation_percentage"),
                "Agua (%)": stac.get("s2:water_percentage"),
                "NoData (%)": stac.get("s2:nodata_pixel_percentage"),
                "Sombra de nubes (%)": stac.get("s2:cloud_shadow_percentage"),
            }
        elif sensor_key == "ERA5":
            metrics = {
                "Variable": quality.get("variable_key"),
                "Agregación": quality.get("aggregation"),
                "Resolución (m)": quality.get("target_res_m"),
            }
        else:
            metrics = {
                "Plataforma": stac.get("platform") or quality.get("platform"),
                "Instrumentos": ", ".join(stac.get("instruments", []) or quality.get("instruments", [])),
            }
        metrics_clean = {k: v for k, v in metrics.items() if v is not None}
        if metrics_clean:
            st.dataframe(
                pd.DataFrame(metrics_clean, index=["Valor"]).T,
                width="stretch",
            )

        # Scene metadata
        st.subheader("Metadatos de la escena")
        meta_items = {
            "Scene ID": quality.get("scene_id"),
            "Collection": quality.get("collection"),
            "Acquisition start": quality.get("acquisition_start"),
            "Acquisition end": quality.get("acquisition_end"),
            "Native EPSG": quality.get("native_epsg"),
        }
        if sensor_key == "S2":
            meta_items["MGRS Tile"] = quality.get("mgrs_tile")
        meta_clean = {k: v for k, v in meta_items.items() if v is not None}
        if meta_clean:
            st.json(meta_clean)
    else:
        st.info("No se encontró archivo de calidad para esta escena.")
