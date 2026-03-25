"""Page 3: Visualization for processed parcel datasets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.map_builder import build_parcel_map
from utils.processed_dataset import (
    PARCEL_METADATA_CSV,
    PROCESSED_DATASET_CSV,
    SENSOR_LABELS,
    available_cube_sensors,
    build_snapshot_gallery,
    build_yearly_summary,
    default_extra_bands,
    format_metric_label,
    list_metric_columns,
    load_cube,
    load_processed_datasets,
    pick_gallery_times,
    plot_metric_distribution,
    plot_metric_yoy,
)

st.set_page_config(page_title="Processed Dataset", page_icon="📈", layout="wide")
st.title("Processed Dataset Visualizer")

st.caption(
    "Visualiza las series temporales procesadas, la metadata de parcela y "
    "snapshots del datacube para una parcela concreta."
)

try:
    stats_df, meta_df = load_processed_datasets()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

parcel_options = sorted(stats_df["parcel_id"].dropna().unique().tolist())
if not parcel_options:
    st.warning("No hay parcelas disponibles en el dataset procesado.")
    st.stop()

metric_options = list_metric_columns(stats_df)
default_metric = "NDVI_mean" if "NDVI_mean" in metric_options else metric_options[0]

with st.sidebar:
    st.header("Controles")
    selected_parcel = st.selectbox("Parcela", parcel_options)
    selected_metric = st.selectbox(
        "Métrica",
        metric_options,
        index=metric_options.index(default_metric),
        format_func=format_metric_label,
    )

    with st.expander("Fuentes de datos"):
        st.code(str(PROCESSED_DATASET_CSV), language="text")
        st.code(str(PARCEL_METADATA_CSV), language="text")

parcel_stats = stats_df[stats_df["parcel_id"] == selected_parcel].copy()
parcel_meta = meta_df[meta_df["parcel_id"] == selected_parcel].copy()
latest_meta = parcel_meta.iloc[-1] if not parcel_meta.empty else None

years = sorted(parcel_stats["year"].dropna().astype(int).unique().tolist())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Registros", f"{len(parcel_stats)}")
with col2:
    st.metric("Años en serie", f"{len(years)}")
with col3:
    target_value = latest_meta["yield_kg_ha"] if latest_meta is not None else None
    st.metric(
        "Target yield (kg/ha)",
        f"{target_value:,.2f}" if pd.notna(target_value) else "N/A",
    )
with col4:
    harvest_date = latest_meta["harvest_date"] if latest_meta is not None else None
    st.metric(
        "Harvest date",
        harvest_date.strftime("%Y-%m-%d") if pd.notna(harvest_date) else "N/A",
    )

overview_col, meta_col = st.columns([1.3, 1.0])
with overview_col:
    st.subheader("Parcela")
    if latest_meta is not None:
        parcel_map = build_parcel_map(
            [latest_meta["parcel_geom"]],
            [selected_parcel],
            selected_ids=[selected_parcel],
            zoom=16,
        )
        st_folium(parcel_map, height=360, width="stretch", returned_objects=[])
    else:
        st.info("No hay geometría disponible para esta parcela.")

with meta_col:
    st.subheader("Metadata")
    if latest_meta is None:
        st.info("No hay metadata disponible para esta parcela.")
    else:
        metadata_table = pd.DataFrame(
            [
                ("parcel_id", latest_meta.get("parcel_id")),
                ("metadata_year", latest_meta.get("year")),
                ("municipality", latest_meta.get("municipality")),
                ("province", latest_meta.get("province")),
                ("declared_area_ha", latest_meta.get("declared_area_ha")),
                ("area_ha", latest_meta.get("area_ha")),
                ("production_kg", latest_meta.get("production_kg")),
                ("yield_kg_ha", latest_meta.get("yield_kg_ha")),
                ("alcohol_degree", latest_meta.get("alcohol_degree")),
            ],
            columns=["field", "value"],
        )
        st.dataframe(metadata_table, hide_index=True, width="stretch")

        if pd.notna(latest_meta.get("year")) and int(latest_meta["year"]) not in years:
            st.info(
                "El año objetivo de la metadata no coincide con los años "
                "disponibles en la serie temporal procesada."
            )

tab_trends, tab_images, tab_data = st.tabs(
    ["Metric trends", "Datacube images", "Raw data"]
)

with tab_trends:
    left_col, right_col = st.columns([1.6, 1.0])

    with left_col:
        st.subheader(f"Evolución YoY de {selected_metric}")
        yoy_fig = plot_metric_yoy(parcel_stats, selected_parcel, selected_metric)
        st.pyplot(yoy_fig, width="stretch")
        plt.close(yoy_fig)

    with right_col:
        st.subheader("Distribución anual")
        dist_fig = plot_metric_distribution(parcel_stats, selected_metric)
        st.pyplot(dist_fig, width="stretch")
        plt.close(dist_fig)

    st.subheader("Resumen anual")
    yearly_summary = build_yearly_summary(parcel_stats, parcel_meta, selected_metric)
    st.dataframe(yearly_summary, width="stretch", hide_index=True)

with tab_images:
    st.subheader("Snapshots del datacube")
    sensors = available_cube_sensors(selected_parcel)

    if not sensors:
        st.info("No se encontró ningún `cube.zarr` para esta parcela.")
    else:
        selected_sensor = st.selectbox(
            "Sensor",
            sensors,
            format_func=lambda sensor: SENSOR_LABELS.get(sensor, sensor),
            key="processed_dataset_sensor",
        )

        try:
            with st.spinner("Cargando datacube..."):
                cube = load_cube(selected_parcel, selected_sensor)
        except Exception as exc:
            st.error(f"No se pudo cargar el datacube: {exc}")
        else:
            time_index = pd.DatetimeIndex(cube.times).sort_values()
            time_labels = [ts.strftime("%Y-%m-%d") for ts in time_index]
            default_idx = len(time_labels) // 2

            selected_date = st.select_slider(
                "Fecha",
                options=time_labels,
                value=time_labels[default_idx],
            )

            extra_band_options = sorted(cube.variables)
            extra_band_defaults = default_extra_bands(cube)
            selected_extra_bands = st.multiselect(
                "Bandas / índices extra",
                options=extra_band_options,
                default=extra_band_defaults,
            )

            snapshot_fig = None
            try:
                from src.visual.visualizer import plot_snapshot

                snapshot_fig = plot_snapshot(
                    cube,
                    time=selected_date,
                    extra_bands=selected_extra_bands,
                )
                st.pyplot(snapshot_fig, width="stretch")
            except Exception as exc:
                st.error(f"No se pudo renderizar el snapshot: {exc}")
            finally:
                if snapshot_fig is not None:
                    plt.close(snapshot_fig)

            st.caption(
                f"{SENSOR_LABELS.get(selected_sensor, selected_sensor)} · "
                f"{len(time_index)} timestamps · variables: {', '.join(cube.variables)}"
            )

            gallery_times = pick_gallery_times(time_index, max_items=3)
            if gallery_times:
                st.subheader("Galería rápida")
                try:
                    gallery_fig = build_snapshot_gallery(cube, gallery_times)
                    st.pyplot(gallery_fig, width="stretch")
                except Exception as exc:
                    st.warning(f"No se pudo construir la galería: {exc}")
                else:
                    plt.close(gallery_fig)

with tab_data:
    st.subheader("Series temporales")
    st.dataframe(parcel_stats, width="stretch", hide_index=True)

    st.subheader("Metadata de parcela")
    if parcel_meta.empty:
        st.info("No hay filas de metadata para esta parcela.")
    else:
        st.dataframe(parcel_meta.drop(columns=["parcel_geom", "bbox_geom"]), width="stretch", hide_index=True)
