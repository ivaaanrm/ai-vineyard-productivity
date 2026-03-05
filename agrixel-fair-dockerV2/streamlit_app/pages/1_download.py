"""Page 1: Download configuration & Docker trigger."""

from __future__ import annotations

from datetime import date

import streamlit as st
from streamlit_folium import st_folium

from utils.data_loader import (
    ERA5_AGGREGATIONS,
    ERA5_VARIABLES,
    MODIS_PRODUCTS,
    S1_POLARIZATIONS,
    S2_BANDS,
    S3_PRODUCTS,
    load_aoi_table,
)
from utils.docker_runner import build_params, run_docker, write_params_file
from utils.map_builder import build_parcel_map

st.set_page_config(page_title="Download", page_icon="⬇️", layout="wide")
st.title("Descarga de imágenes satelitales")

# ------------------------------------------------------------------
# Load AOI table
# ------------------------------------------------------------------
try:
    aoi_df = load_aoi_table()
except FileNotFoundError:
    st.error("No se encontró `data/input/aoi_table.csv`.")
    st.stop()

parcel_options = aoi_df["parcel_id"].tolist()
municipio_map = dict(zip(aoi_df["parcel_id"], aoi_df["municipio"]))

# ------------------------------------------------------------------
# 1. Sensor selector
# ------------------------------------------------------------------
_SENSOR_OPTIONS = {
    "Sentinel-2": "S2",
    "Sentinel-1": "S1",
    "Sentinel-3 (LST)": "S3",
    "MODIS (ET)": "MODIS",
    "ERA5": "ERA5",
}
sensor = st.radio("Sensor", list(_SENSOR_OPTIONS.keys()), horizontal=True)
sensor_key = _SENSOR_OPTIONS[sensor]

# ------------------------------------------------------------------
# 2. Parcel picker
# ------------------------------------------------------------------
st.subheader("Parcelas")

select_all = st.checkbox("Seleccionar todas", value=False)

if select_all:
    selected_parcels = parcel_options
    st.info(f"{len(selected_parcels)} parcelas seleccionadas.")
else:
    selected_parcels = st.multiselect(
        "Parcelas",
        options=parcel_options,
        format_func=lambda pid: f"{pid} — mun. {municipio_map.get(pid, '?')}",
        label_visibility="collapsed",
    )

# ------------------------------------------------------------------
# 3. Map preview
# ------------------------------------------------------------------
if selected_parcels:
    preview_map = build_parcel_map(
        aoi_df["geom"].tolist(),
        parcel_options,
        selected_ids=selected_parcels,
    )
    st_folium(preview_map, height=350, width="stretch", returned_objects=[])

# ------------------------------------------------------------------
# 4. Date range
# ------------------------------------------------------------------
st.subheader("Rango de fechas")
col1, col2 = st.columns(2)
with col1:
    date_from = st.date_input("Desde", value=date(2024, 5, 1))
with col2:
    date_to = st.date_input("Hasta", value=date(2024, 5, 31))

if date_from > date_to:
    st.warning("La fecha de inicio debe ser anterior a la fecha de fin.")

# ------------------------------------------------------------------
# 5. Sensor-specific options
# ------------------------------------------------------------------
st.subheader("Opciones del sensor")

bands = None
cloud_cover = 20
pols = None
products_s3 = None
products_modis = None
era5_variable = None
era5_daily_agg = None

if sensor_key == "S2":
    bands = st.multiselect("Bandas", S2_BANDS, default=["B04", "B08", "NDVI"])
    cloud_cover = st.slider(
        "Cobertura nubosa máxima (%)", min_value=0, max_value=100, value=20
    )
elif sensor_key == "S1":
    pols = st.multiselect("Polarización", S1_POLARIZATIONS, default=["VV", "VH"])
elif sensor_key == "S3":
    products_s3 = st.multiselect("Productos", S3_PRODUCTS, default=["lst-in"])
elif sensor_key == "MODIS":
    products_modis = st.multiselect(
        "Productos", MODIS_PRODUCTS, default=["ET_500m", "PET_500m"]
    )
elif sensor_key == "ERA5":
    era5_variable = st.selectbox(
        "Variable",
        list(ERA5_VARIABLES.keys()),
        format_func=lambda k: f"{k} — {ERA5_VARIABLES[k]}",
    )
    era5_daily_agg = st.selectbox("Agregación diaria", ERA5_AGGREGATIONS)

# ------------------------------------------------------------------
# 6. Advanced options
# ------------------------------------------------------------------
with st.expander("Opciones avanzadas"):
    buffer_m = st.number_input("Buffer (m)", min_value=0, value=100, step=10)
    target_res = st.number_input(
        "Resolución objetivo (m)",
        min_value=0,
        value=0,
        step=1,
        help="0 = resolución nativa",
    )
    max_items = st.number_input(
        "Máximo de escenas",
        min_value=0,
        value=0,
        step=1,
        help="0 = sin límite",
    )

# ------------------------------------------------------------------
# 7. Download button
# ------------------------------------------------------------------
st.divider()

can_run = bool(selected_parcels) and date_from <= date_to
if sensor_key == "S2" and not bands:
    can_run = False
if sensor_key == "S1" and not pols:
    can_run = False
if sensor_key == "S3" and not products_s3:
    can_run = False
if sensor_key == "MODIS" and not products_modis:
    can_run = False

if st.button("Descargar", type="primary", disabled=not can_run):
    params = build_params(
        sensor=sensor_key,
        date_from=str(date_from),
        date_to=str(date_to),
        parcel_ids=selected_parcels if not select_all else None,
        bands_s2=bands,
        cloud_cover=cloud_cover,
        pols_s1=pols,
        products_s3=products_s3,
        products_modis=products_modis,
        era5_variable=era5_variable,
        era5_daily_agg=era5_daily_agg,
        buffer_m=int(buffer_m),
        target_res_m=int(target_res) if target_res else None,
        max_items=int(max_items) if max_items else None,
    )

    params_file = write_params_file(params)
    st.code(f"Params: {params_file.name}", language=None)

    # ------------------------------------------------------------------
    # 8. Docker execution with live log
    # ------------------------------------------------------------------
    log_expander = st.expander("Log de ejecución", expanded=True)
    log_area = log_expander.empty()
    log_lines: list[str] = []

    exit_code = None
    with st.spinner("Ejecutando descarga…"):
        for stream, line in run_docker(params_file):
            if stream == "exit":
                exit_code = int(line)
            else:
                prefix = "ERR| " if stream == "stderr" else ""
                log_lines.append(f"{prefix}{line}")
                log_area.code("\n".join(log_lines[-200:]), language="text")

    # ------------------------------------------------------------------
    # Result summary
    # ------------------------------------------------------------------
    if exit_code == 0:
        st.success("Descarga completada correctamente.")
        st.info("Navega a la página **Explorer** para visualizar los datos.")
    else:
        st.error(f"La descarga falló (exit code {exit_code}).")
